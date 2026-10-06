"""Paper-faithful linear Edge Constraint Adaptive (ECA) optimization.

This module implements Equation (18) of Chen et al. (2025) for a linear SEM,

    1 / (2 n) ||X - X W||_F^2 - sigma^2 b(W; Pi) + lambda ||W||_1,

subject to the exact NOTEARS acyclicity constraint
``trace(expm(W * W)) - d == 0``.  It deliberately has no dependency on the
official ECA implementation: that implementation changes the acyclicity
function and hard-codes the prior weight.  The project-facing API translates
its public configuration and prior types into the matrix primitives accepted
here.

The edge convention is ``W[source, target]``.  The weighted solution remains
unthresholded so its reported objective is well defined; ``binary_adjacency``
and ``graph`` use the paper's strict ``abs(W) > threshold`` rule.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from cd_llm.core.finite_support import Graph, graph_from_edges, standardize


_EXISTENCE_WARM_START = 1e-3


class PaperECAOptimizationError(RuntimeError):
    """The continuous paper-faithful optimization did not produce a result."""


class CyclicThresholdedGraphError(PaperECAOptimizationError):
    """The thresholded weighted solution is not a DAG."""


@dataclass(frozen=True)
class PaperObjectiveTerms:
    """Equation (18) evaluated at one weighted adjacency matrix."""

    objective: float
    data_loss: float
    prior_log_score: float
    l1_penalty: float


@dataclass(frozen=True)
class PaperOptimizationDiagnostics:
    """Termination and work counters from augmented-Lagrangian optimization."""

    outer_iterations: int
    inner_optimizations: int
    objective_evaluations: int
    gradient_evaluations: int
    converged: bool
    acyclicity: float
    rho: float
    alpha: float
    runtime_seconds: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str


@dataclass(frozen=True)
class PaperBackendResult:
    """Internal solver result consumed by the public ECA interface."""

    graph: Graph
    weighted_adjacency: np.ndarray
    binary_adjacency: np.ndarray
    terms: PaperObjectiveTerms
    diagnostics: PaperOptimizationDiagnostics

    def __post_init__(self) -> None:
        weighted = _readonly_copy(self.weighted_adjacency, dtype=float)
        binary = _readonly_copy(self.binary_adjacency, dtype=np.int8)
        if weighted.ndim != 2 or weighted.shape[0] != weighted.shape[1]:
            raise ValueError("weighted_adjacency must be square")
        if binary.shape != weighted.shape:
            raise ValueError("weighted and binary adjacency shapes differ")
        if self.graph.node_count != weighted.shape[0]:
            raise ValueError("graph and adjacency node counts differ")
        object.__setattr__(self, "weighted_adjacency", weighted)
        object.__setattr__(self, "binary_adjacency", binary)


def _readonly_copy(values: np.ndarray, *, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _as_square_weights(weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("weights must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("weights must contain at least one node")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights must be finite")
    return matrix


def _validated_prior_matrices(
    mask: np.ndarray,
    probabilities: np.ndarray,
    shape: tuple[int, int],
    *,
    reject_diagonal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    raw_mask = np.asarray(mask)
    if raw_mask.shape != shape:
        raise ValueError("prior mask shape must match the weighted adjacency")
    if raw_mask.dtype.kind not in "bui":
        try:
            numeric_mask = np.asarray(raw_mask, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("prior mask must contain Boolean or binary values") from error
        if not np.all(np.isfinite(numeric_mask)) or not np.all(
            (numeric_mask == 0.0) | (numeric_mask == 1.0)
        ):
            raise ValueError("prior mask must contain only zero and one")
    elif not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise ValueError("prior mask must contain only zero and one")
    normalized_mask = np.asarray(raw_mask, dtype=bool)
    if reject_diagonal and np.any(np.diag(normalized_mask)):
        raise ValueError("self-edge constraints are not allowed")

    try:
        normalized_probabilities = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("prior probabilities must be real numbers") from error
    if normalized_probabilities.shape != shape:
        raise ValueError(
            "prior probability shape must match the weighted adjacency"
        )
    if not np.all(np.isfinite(normalized_probabilities)):
        raise ValueError("prior probabilities must be finite")
    if np.any(normalized_probabilities < 0.0) or np.any(
        normalized_probabilities > 1.0
    ):
        raise ValueError("prior probabilities must lie in [0, 1]")
    return normalized_mask, normalized_probabilities


def data_loss_and_gradient(
    data: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the normalized linear-SEM squared loss and its gradient."""

    matrix = _as_square_weights(weights)
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a nonempty two-dimensional matrix")
    if values.shape[1] != matrix.shape[0]:
        raise ValueError("data and weights have different node counts")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must be finite")
    residual = values - values @ matrix
    sample_size = values.shape[0]
    loss = 0.5 * float(np.sum(residual * residual)) / sample_size
    gradient = -(values.T @ residual) / sample_size
    return loss, gradient


def prior_log_score_and_gradient(
    weights: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return ``b(W; Pi)`` from Equation (16) and its gradient.

    The paper permits probabilities at zero and one.  Their logarithm can be
    singular at a finite-precision boundary, so the mixture is floored at the
    smallest positive normal float solely for numerical evaluation.  At
    ``W_ij == 0`` the absolute-value proxy is nonsmooth; as in the official
    automatic-differentiation implementation, its zero subgradient is used.
    """

    matrix = _as_square_weights(weights)
    normalized_mask, normalized_probabilities = _validated_prior_matrices(
        mask,
        probabilities,
        matrix.shape,
        reject_diagonal=False,
    )
    gradient = np.zeros_like(matrix)
    if not np.any(normalized_mask):
        return 0.0, gradient

    constrained_weights = matrix[normalized_mask]
    constrained_probabilities = normalized_probabilities[normalized_mask]
    half_tanh = np.tanh(0.5 * constrained_weights)
    soft_presence = np.abs(half_tanh)
    probability_slope = 2.0 * constrained_probabilities - 1.0
    mixture = (
        1.0
        - constrained_probabilities
        + probability_slope * soft_presence
    )
    safe_mixture = np.maximum(mixture, np.finfo(float).tiny)
    score = float(np.sum(np.log(safe_mixture)))

    soft_presence_gradient = (
        0.5 * np.sign(half_tanh) * (1.0 - half_tanh * half_tanh)
    )
    constrained_gradient = (
        probability_slope * soft_presence_gradient / safe_mixture
    )
    # The floor matters only at a singular boundary.  Its derivative is zero.
    constrained_gradient = np.where(
        mixture > np.finfo(float).tiny, constrained_gradient, 0.0
    )
    gradient[normalized_mask] = constrained_gradient
    return score, gradient


def acyclicity_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Return exact NOTEARS ``h(W)`` and its analytical gradient."""

    matrix = _as_square_weights(weights)
    exponential = expm(matrix * matrix)
    if not np.all(np.isfinite(exponential)):
        raise FloatingPointError("matrix exponential overflow in acyclicity term")
    value = float(np.trace(exponential) - matrix.shape[0])
    # Roundoff can produce a tiny negative result although h(W) is nonnegative.
    if value < 0.0 and abs(value) <= 1e-12:
        value = 0.0
    gradient = exponential.T * (2.0 * matrix)
    return value, gradient


def smooth_objective_and_gradient(
    data: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    sigma: float,
) -> tuple[float, np.ndarray]:
    """Return the differentiable data-minus-prior portion of Equation (18)."""

    sigma = _positive_finite(sigma, "sigma")
    data_loss, data_gradient = data_loss_and_gradient(data, weights)
    prior_score, prior_gradient = prior_log_score_and_gradient(
        weights, mask, probabilities
    )
    scale = sigma * sigma
    return (
        data_loss - scale * prior_score,
        data_gradient - scale * prior_gradient,
    )


def objective_and_subgradient(
    data: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    lambda1: float,
    sigma: float,
) -> tuple[float, np.ndarray]:
    """Return Equation (18) and its conventional L1 subgradient.

    The L1 subgradient is chosen as zero where an entry of ``weights`` is
    exactly zero.  Optimization itself uses nonnegative split variables, so it
    does not depend on this diagnostic convention.
    """

    lambda1 = _nonnegative_finite(lambda1, "lambda1")
    matrix = _as_square_weights(weights)
    smooth_value, smooth_gradient = smooth_objective_and_gradient(
        data,
        matrix,
        mask,
        probabilities,
        sigma=sigma,
    )
    value = smooth_value + lambda1 * float(np.sum(np.abs(matrix)))
    gradient = smooth_gradient + lambda1 * np.sign(matrix)
    return float(value), gradient


def objective_terms(
    data: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    lambda1: float,
    sigma: float,
) -> PaperObjectiveTerms:
    """Evaluate and decompose the complete nonsmooth Equation (18)."""

    lambda1 = _nonnegative_finite(lambda1, "lambda1")
    sigma = _positive_finite(sigma, "sigma")
    matrix = _as_square_weights(weights)
    data_loss, _ = data_loss_and_gradient(data, matrix)
    prior_log_score, _ = prior_log_score_and_gradient(
        matrix, mask, probabilities
    )
    l1_penalty = lambda1 * float(np.sum(np.abs(matrix)))
    objective = data_loss - sigma * sigma * prior_log_score + l1_penalty
    return PaperObjectiveTerms(
        objective=float(objective),
        data_loss=float(data_loss),
        prior_log_score=float(prior_log_score),
        l1_penalty=float(l1_penalty),
    )


def _positive_finite(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a real number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _nonnegative_finite(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a real number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _prepare_data(data: np.ndarray, standardize_data: bool) -> np.ndarray:
    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("data must contain real numbers") from error
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must be finite")
    if not isinstance(standardize_data, (bool, np.bool_)):
        raise TypeError("standardize_data must be Boolean")
    prepared = standardize(values) if bool(standardize_data) else values.copy()
    return np.ascontiguousarray(prepared, dtype=float)


def _threshold_graph(
    weights: np.ndarray, threshold: float
) -> tuple[np.ndarray, Graph]:
    binary = (np.abs(weights) > threshold).astype(np.int8)
    np.fill_diagonal(binary, 0)
    directed = tuple(
        (int(source), int(target))
        for source, target in np.argwhere(binary == 1)
    )
    try:
        graph = graph_from_edges(weights.shape[0], directed=directed)
    except ValueError as error:
        raise CyclicThresholdedGraphError(
            "the strict-threshold ECA output contains a directed cycle; "
            "no edges were silently removed"
        ) from error
    return binary, graph


def _initial_split_weights(
    data: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Break the edge-gate cusp deterministically for existence constraints.

    The paper's ``abs(tanh(W / 2))`` proxy is nonsmooth at zero.  Selecting a
    zero subgradient at an all-zero initialization can therefore report a false
    stationary point even when moving away from zero sharply lowers the prior
    loss.  A tiny warm start chooses the coefficient sign favored by the data
    gradient; exact ties use the positive branch.  Absence constraints remain
    at their natural zero initialization.
    """

    node_count = data.shape[1]
    initial = np.zeros((node_count, node_count), dtype=float)
    existence = mask & (probabilities > 0.5)
    if np.any(existence):
        covariance = data.T @ data / data.shape[0]
        signs = np.where(covariance < 0.0, -1.0, 1.0)
        initial[existence] = _EXISTENCE_WARM_START * signs[existence]
    dimension = node_count * node_count
    return np.concatenate(
        (
            np.maximum(initial, 0.0).reshape(dimension),
            np.maximum(-initial, 0.0).reshape(dimension),
        )
    )


def fit_paper_eca(
    data: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    lambda1: float = 0.05,
    sigma: float = 1.0,
    threshold: float = 0.30,
    standardize_data: bool = False,
    max_outer_iterations: int = 100,
    max_inner_iterations: int = 100,
    h_tolerance: float = 1e-8,
    rho_max: float = 1e16,
    rho_multiplier: float = 10.0,
    progress_rate: float = 0.25,
) -> PaperBackendResult:
    """Fit the paper-faithful linear ECA model with deterministic initialization.

    ``mask`` and ``probabilities`` must be square matrices matching the number
    of data columns.  A constrained entry's probability is the prior
    probability that its directed edge exists.  Unmasked probabilities have
    no effect.
    """

    values = _prepare_data(data, standardize_data)
    sample_size, node_count = values.shape
    del sample_size  # The normalized loss helper owns the sample-size factor.
    shape = (node_count, node_count)
    normalized_mask, normalized_probabilities = _validated_prior_matrices(
        mask, probabilities, shape, reject_diagonal=True
    )
    lambda1 = _nonnegative_finite(lambda1, "lambda1")
    sigma = _positive_finite(sigma, "sigma")
    threshold = _positive_finite(threshold, "threshold")
    max_outer_iterations = _positive_integer(
        max_outer_iterations, "max_outer_iterations"
    )
    max_inner_iterations = _positive_integer(
        max_inner_iterations, "max_inner_iterations"
    )
    h_tolerance = _positive_finite(h_tolerance, "h_tolerance")
    rho_max = _positive_finite(rho_max, "rho_max")
    rho_multiplier = _positive_finite(rho_multiplier, "rho_multiplier")
    if rho_max <= 1.0:
        raise ValueError("rho_max must be greater than the initial rho of one")
    if rho_multiplier <= 1.0:
        raise ValueError("rho_multiplier must be greater than one")
    progress_rate = _positive_finite(progress_rate, "progress_rate")
    if progress_rate >= 1.0:
        raise ValueError("progress_rate must be less than one")

    started = time.perf_counter()
    dimension = node_count * node_count
    split_weights = _initial_split_weights(
        values, normalized_mask, normalized_probabilities
    )
    bounds = [
        (0.0, 0.0) if source == target else (0.0, None)
        for _copy in range(2)
        for source in range(node_count)
        for target in range(node_count)
    ]
    rho = 1.0
    alpha = 0.0
    h_value = math.inf
    objective_evaluations = 0
    gradient_evaluations = 0
    inner_optimizations = 0
    outer_iterations = 0
    optimizer_success = True
    optimizer_status = 0
    optimizer_message = "single-node closed-form solution"

    if node_count > 1:
        last_solution = None
        for outer_index in range(max_outer_iterations):
            candidate_split: np.ndarray | None = None
            candidate_h: float | None = None
            while rho < rho_max:

                def augmented_objective(
                    split: np.ndarray,
                ) -> tuple[float, np.ndarray]:
                    matrix = (
                        split[:dimension] - split[dimension:]
                    ).reshape(shape)
                    smooth_value, smooth_gradient = smooth_objective_and_gradient(
                        values,
                        matrix,
                        normalized_mask,
                        normalized_probabilities,
                        sigma=sigma,
                    )
                    constraint_value, constraint_gradient = (
                        acyclicity_and_gradient(matrix)
                    )
                    multiplier = rho * constraint_value + alpha
                    value = (
                        smooth_value
                        + 0.5 * rho * constraint_value * constraint_value
                        + alpha * constraint_value
                        + lambda1 * float(np.sum(split))
                    )
                    gradient = smooth_gradient + multiplier * constraint_gradient
                    split_gradient = np.concatenate(
                        (
                            gradient.ravel() + lambda1,
                            -gradient.ravel() + lambda1,
                        )
                    )
                    return float(value), split_gradient

                solution = minimize(
                    augmented_objective,
                    split_weights,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={
                        "maxiter": max_inner_iterations,
                        "ftol": 1e-12,
                        "gtol": 1e-8,
                    },
                )
                last_solution = solution
                inner_optimizations += 1
                objective_evaluations += int(solution.nfev)
                gradient_evaluations += int(solution.njev)
                if not np.all(np.isfinite(solution.x)) or not math.isfinite(
                    float(solution.fun)
                ):
                    raise PaperECAOptimizationError(
                        "L-BFGS-B returned a nonfinite ECA solution"
                    )
                candidate_split = np.asarray(solution.x, dtype=float)
                candidate_matrix = (
                    candidate_split[:dimension]
                    - candidate_split[dimension:]
                ).reshape(shape)
                candidate_h, _ = acyclicity_and_gradient(candidate_matrix)
                if candidate_h > progress_rate * h_value:
                    rho *= rho_multiplier
                else:
                    break

            if candidate_split is None or candidate_h is None:
                raise PaperECAOptimizationError(
                    "the augmented-Lagrangian rho limit was reached before "
                    "an optimization step"
                )
            split_weights = candidate_split
            h_value = float(candidate_h)
            alpha += rho * h_value
            outer_iterations = outer_index + 1
            if h_value <= h_tolerance or rho >= rho_max:
                break

        assert last_solution is not None
        optimizer_success = bool(last_solution.success)
        optimizer_status = int(last_solution.status)
        optimizer_message = str(last_solution.message)
    else:
        h_value = 0.0

    weights = (
        split_weights[:dimension] - split_weights[dimension:]
    ).reshape(shape)
    np.fill_diagonal(weights, 0.0)
    if not np.all(np.isfinite(weights)):
        raise PaperECAOptimizationError("ECA produced nonfinite weights")
    final_h, _ = acyclicity_and_gradient(weights)
    binary, graph = _threshold_graph(weights, threshold)
    terms = objective_terms(
        values,
        weights,
        normalized_mask,
        normalized_probabilities,
        lambda1=lambda1,
        sigma=sigma,
    )
    diagnostics = PaperOptimizationDiagnostics(
        outer_iterations=outer_iterations,
        inner_optimizations=inner_optimizations,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        converged=bool(final_h <= h_tolerance),
        acyclicity=float(final_h),
        rho=float(rho),
        alpha=float(alpha),
        runtime_seconds=float(time.perf_counter() - started),
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
    )
    return PaperBackendResult(
        graph=graph,
        weighted_adjacency=weights,
        binary_adjacency=binary,
        terms=terms,
        diagnostics=diagnostics,
    )


__all__ = [
    "CyclicThresholdedGraphError",
    "PaperBackendResult",
    "PaperECAOptimizationError",
    "PaperObjectiveTerms",
    "PaperOptimizationDiagnostics",
    "acyclicity_and_gradient",
    "data_loss_and_gradient",
    "fit_paper_eca",
    "objective_and_subgradient",
    "objective_terms",
    "prior_log_score_and_gradient",
    "smooth_objective_and_gradient",
]
