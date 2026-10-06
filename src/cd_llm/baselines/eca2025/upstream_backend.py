"""Lazy wrapper around the archived linear ECA implementation.

This module deliberately preserves the numerical behavior of upstream commit
``c98002720dca8c170ba0f3d25ac3681e5def0bdc``.  The only integration fixes are
to load the supplied prior (the upstream demo omits that call) and to pass the
configured threshold to ``notears_linear`` explicitly.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from cd_llm.core.finite_support import Graph, graph_from_edges, standardize

from .paper_backend import prior_log_score_and_gradient


UPSTREAM_REVISION = "c98002720dca8c170ba0f3d25ac3681e5def0bdc"


@dataclass(frozen=True)
class UpstreamBackendDiagnostics:
    """Diagnostics available from the uninstrumented upstream optimizer."""

    upstream_revision: str
    fit_seconds: float
    acyclicity: float
    assertion_confidence: float | None
    constrained_edge_count: int


@dataclass(frozen=True)
class UpstreamObjectiveTerms:
    """Archived objective evaluated on the returned thresholded weights."""

    objective: float
    data_loss: float
    prior_log_score: float
    prior_penalty: float
    l1_penalty: float


@dataclass(frozen=True)
class UpstreamBackendResult:
    """Validated output of the official-code reproduction backend."""

    weighted_adjacency: np.ndarray
    binary_adjacency: np.ndarray
    graph: Graph
    terms: UpstreamObjectiveTerms
    diagnostics: UpstreamBackendDiagnostics

    def __post_init__(self) -> None:
        for value in (self.weighted_adjacency, self.binary_adjacency):
            value.setflags(write=False)


def _default_upstream_root() -> Path:
    return Path(__file__).resolve().parents[4] / "vendor" / "ECA"


def _verified_upstream_root(upstream_root: str | Path | None) -> Path:
    root = (
        _default_upstream_root()
        if upstream_root is None
        else Path(upstream_root).expanduser().resolve()
    )
    source = root / "src" / "ECA_prior.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing upstream linear implementation: {source}")
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot verify the ECA checkout at {root}") from error
    revision = completed.stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(
            f"ECA checkout is at {revision!r}; expected {UPSTREAM_REVISION}"
        )
    try:
        status = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect the ECA checkout at {root}") from error
    if status.stdout.strip():
        raise RuntimeError("upstream ECA tracked files differ from the pinned commit")
    return root


def _load_upstream_linear(root: Path) -> ModuleType:
    source = root / "src" / "ECA_prior.py"
    module_name = f"_project_eca_upstream_{UPSTREAM_REVISION[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot create an import specification for {source}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "the upstream ECA backend requires gcastle==1.0.3 and torch; "
            "install baselines/eca2025/requirements-upstream.txt"
        ) from error
    if not hasattr(module, "myECA_prior"):
        raise RuntimeError("upstream ECA_prior.py does not define myECA_prior")
    return module


def _validated_inputs(
    data: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(data, dtype=float)
    if observations.ndim != 2 or min(observations.shape) < 1:
        raise ValueError("data must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("data must contain only finite values")
    node_count = observations.shape[1]
    raw_mask = np.asarray(mask)
    if raw_mask.shape != (node_count, node_count):
        raise ValueError("prior mask shape does not match the data")
    if raw_mask.dtype.kind not in "biu" or not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise ValueError("prior mask must be binary")
    normalized_mask = raw_mask.astype(bool, copy=True)
    if np.any(np.diag(normalized_mask)):
        raise ValueError("self-edge priors are not allowed")
    normalized_probabilities = np.asarray(probabilities, dtype=float)
    if normalized_probabilities.shape != (node_count, node_count):
        raise ValueError("prior probability shape does not match the data")
    selected = normalized_probabilities[normalized_mask]
    if selected.size and (
        not np.all(np.isfinite(selected))
        or np.any(selected <= 0.0)
        or np.any(selected >= 1.0)
    ):
        raise ValueError("masked edge probabilities must lie strictly between zero and one")
    return observations, normalized_mask, normalized_probabilities.copy()


def _upstream_prior(
    mask: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray | None, float | None]:
    active = mask & ~np.isclose(probabilities, 0.5, rtol=0.0, atol=1e-15)
    if not np.any(active):
        return None, None
    signs = np.zeros(mask.shape, dtype=float)
    signs[active & (probabilities > 0.5)] = 1.0
    signs[active & (probabilities < 0.5)] = -1.0
    confidences = np.where(
        probabilities[active] > 0.5,
        probabilities[active],
        1.0 - probabilities[active],
    )
    confidence = float(confidences[0])
    if not np.allclose(confidences, confidence, rtol=0.0, atol=1e-12):
        raise ValueError(
            "the upstream backend supports one shared assertion confidence; "
            "use backend='paper' for per-edge probabilities"
        )
    return signs, confidence


def _polynomial_acyclicity(weights: np.ndarray) -> float:
    node_count = weights.shape[0]
    matrix = np.eye(node_count) + weights * weights / node_count
    power = np.linalg.matrix_power(matrix, node_count - 1)
    return float(np.sum(power.T * matrix) - node_count)


def fit_upstream_eca(
    data: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    lambda1: float = 0.05,
    sigma: float = 1.0,
    threshold: float = 0.30,
    standardize_data: bool = True,
    max_outer_iterations: int = 100,
    h_tolerance: float = 1e-8,
    rho_max: float = 1e16,
    upstream_root: str | Path | None = None,
) -> UpstreamBackendResult:
    """Fit the c980 linear implementation with the missing prior wiring fixed."""

    observations, normalized_mask, normalized_probabilities = _validated_inputs(
        data, mask, probabilities
    )
    for name, value, positive in (
        ("lambda1", lambda1, False),
        ("sigma", sigma, True),
        ("threshold", threshold, True),
        ("h_tolerance", h_tolerance, True),
        ("rho_max", rho_max, True),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if (positive and float(value) <= 0.0) or (
            not positive and float(value) < 0.0
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be {qualifier}")
    if isinstance(max_outer_iterations, bool) or not isinstance(max_outer_iterations, int):
        raise TypeError("max_outer_iterations must be an integer")
    if max_outer_iterations < 1:
        raise ValueError("max_outer_iterations must be positive")

    upstream = _verified_upstream_root(upstream_root)
    module = _load_upstream_linear(upstream)
    prior_signs, confidence = _upstream_prior(
        normalized_mask, normalized_probabilities
    )
    fit_data = standardize(observations) if standardize_data else observations.copy()
    model = module.myECA_prior(
        lambda1=float(lambda1),
        sigma=float(sigma),
        loss_type="pdf",
        max_iter=max_outer_iterations,
        h_tol=float(h_tolerance),
        rho_max=float(rho_max),
        w_threshold=float(threshold),
    )
    model.load_prior(prior_signs, 0.0 if confidence is None else confidence)
    started = time.perf_counter()
    weights = np.asarray(
        model.notears_linear(
            fit_data,
            lambda1=float(lambda1),
            sigma=float(sigma),
            loss_type="pdf",
            max_iter=max_outer_iterations,
            h_tol=float(h_tolerance),
            rho_max=float(rho_max),
            w_threshold=float(threshold),
        ),
        dtype=float,
    )
    elapsed = time.perf_counter() - started
    node_count = observations.shape[1]
    if weights.shape != (node_count, node_count) or not np.all(np.isfinite(weights)):
        raise RuntimeError("upstream ECA returned an invalid weighted adjacency matrix")
    if np.any(np.diag(weights) != 0.0):
        raise RuntimeError("upstream ECA returned a nonzero diagonal")
    adjacency = (np.abs(weights) > float(threshold)).astype(np.int8)
    edges = tuple(
        (source, target)
        for source in range(node_count)
        for target in range(node_count)
        if adjacency[source, target]
    )
    try:
        graph = graph_from_edges(node_count, directed=edges)
    except ValueError as error:
        raise RuntimeError("thresholded upstream ECA output is not a DAG") from error
    weights = weights.copy()
    adjacency = adjacency.copy()
    weights.setflags(write=False)
    adjacency.setflags(write=False)
    centered = fit_data - np.mean(fit_data, axis=0, keepdims=True)
    residual = centered - centered @ weights
    data_loss = 0.5 * float(np.sum(residual * residual)) / (
        centered.shape[0] * float(sigma) * float(sigma)
    )
    effective_mask = normalized_mask & ~np.isclose(
        normalized_probabilities, 0.5, rtol=0.0, atol=1e-15
    )
    prior_log_score, _ = prior_log_score_and_gradient(
        weights, effective_mask, normalized_probabilities
    )
    prior_penalty = -0.4 * prior_log_score
    l1_penalty = float(lambda1) * float(np.sum(np.abs(weights)))
    terms = UpstreamObjectiveTerms(
        math.fsum((data_loss, prior_penalty, l1_penalty)),
        data_loss,
        prior_log_score,
        prior_penalty,
        l1_penalty,
    )
    return UpstreamBackendResult(
        weights,
        adjacency,
        graph,
        terms,
        UpstreamBackendDiagnostics(
            UPSTREAM_REVISION,
            elapsed,
            _polynomial_acyclicity(weights),
            confidence,
            int(np.count_nonzero(prior_signs)) if prior_signs is not None else 0,
        ),
    )


__all__ = [
    "UPSTREAM_REVISION",
    "UpstreamBackendDiagnostics",
    "UpstreamBackendResult",
    "UpstreamObjectiveTerms",
    "fit_upstream_eca",
]
