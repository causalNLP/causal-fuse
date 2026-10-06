"""Immutable public values shared by the local ECA interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np

from cd_llm.core.finite_support import Graph, graph_to_adjacency

if TYPE_CHECKING:
    from .adapters import EvidenceAdaptation


Backend = Literal["paper", "upstream"]


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    return normalized


def _real(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    if strictly_positive and normalized <= 0.0:
        raise ValueError(f"{label} must be positive")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return normalized


@dataclass(frozen=True, order=True)
class EdgeConstraint:
    """Paper-native existence probability for one ordered edge.

    Matrix entry ``source, target`` refers to ``source -> target``.  Endpoints
    of zero and one are permitted by the paper, though interior probabilities
    are preferable for numerical optimization.
    """

    source: int
    target: int
    probability: float

    def __post_init__(self) -> None:
        source = _integer(self.source, "source")
        target = _integer(self.target, "target")
        if source == target:
            raise ValueError("an ECA constraint cannot target a self-edge")
        probability = _real(self.probability, "probability", minimum=0.0)
        if probability > 1.0:
            raise ValueError("probability must be at most 1")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True)
class ECAPrior:
    """Validated, deduplicated ECA edge-probability prior."""

    node_count: int
    constraints: tuple[EdgeConstraint, ...] = ()

    def __post_init__(self) -> None:
        node_count = _integer(self.node_count, "node_count", positive=True)
        try:
            constraints = tuple(self.constraints)
        except TypeError as error:
            raise TypeError("constraints must be an iterable of EdgeConstraint values") from error
        if any(not isinstance(item, EdgeConstraint) for item in constraints):
            raise TypeError("constraints must contain EdgeConstraint values")
        constraints = tuple(sorted(constraints))
        seen: set[tuple[int, int]] = set()
        for constraint in constraints:
            if constraint.source >= node_count or constraint.target >= node_count:
                raise ValueError(
                    f"constraint endpoint is outside [0, {node_count})"
                )
            edge = constraint.source, constraint.target
            if edge in seen:
                raise ValueError(
                    f"ordered edge {constraint.source}->{constraint.target} has "
                    "multiple probabilities"
                )
            seen.add(edge)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "constraints", constraints)

    @classmethod
    def empty(cls, node_count: int) -> "ECAPrior":
        return cls(node_count)

    @classmethod
    def from_matrices(
        cls, mask: np.ndarray, probabilities: np.ndarray
    ) -> "ECAPrior":
        """Construct a prior from the paper's ``W_m`` and ``W_p`` matrices."""

        raw_mask = np.asarray(mask)
        raw_probabilities = np.asarray(probabilities)
        if raw_mask.ndim != 2 or raw_mask.shape[0] != raw_mask.shape[1]:
            raise ValueError("mask must be a square matrix")
        if raw_probabilities.shape != raw_mask.shape:
            raise ValueError("probabilities must have the same shape as mask")
        if not np.all(np.isin(raw_mask, (0, 1, False, True))):
            raise ValueError("mask entries must be binary")
        if np.any(np.diag(raw_mask.astype(bool))):
            raise ValueError("the ECA prior cannot mask diagonal entries")
        try:
            numeric_probabilities = np.asarray(raw_probabilities, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("probabilities must contain real numbers") from error
        if not np.all(np.isfinite(numeric_probabilities)):
            raise ValueError("probabilities must be finite")
        if np.any((numeric_probabilities < 0.0) | (numeric_probabilities > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
        constraints = tuple(
            EdgeConstraint(int(source), int(target), numeric_probabilities[source, target])
            for source, target in zip(*np.nonzero(raw_mask))
        )
        return cls(raw_mask.shape[0], constraints)

    @property
    def mask(self) -> np.ndarray:
        result = np.zeros((self.node_count, self.node_count), dtype=np.int8)
        for constraint in self.constraints:
            result[constraint.source, constraint.target] = 1
        result.setflags(write=False)
        return result

    @property
    def probabilities(self) -> np.ndarray:
        result = np.full((self.node_count, self.node_count), 0.5, dtype=float)
        for constraint in self.constraints:
            result[constraint.source, constraint.target] = constraint.probability
        result.setflags(write=False)
        return result

    def as_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mask, self.probabilities


@dataclass(frozen=True)
class ECAConfig:
    """Controls shared by the paper-faithful and upstream ECA backends."""

    backend: Backend = "paper"
    lambda1: float = 0.05
    sigma: float = 1.0
    threshold: float = 0.3
    max_iterations: int = 100
    h_tolerance: float = 1e-8
    rho_max: float = 1e16
    rho_multiplier: float = 10.0
    standardize_data: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("paper", "upstream"):
            raise ValueError("backend must be 'paper' or 'upstream'")
        lambda1 = _real(self.lambda1, "lambda1", minimum=0.0)
        sigma = _real(self.sigma, "sigma", strictly_positive=True)
        threshold = _real(self.threshold, "threshold", strictly_positive=True)
        max_iterations = _integer(
            self.max_iterations, "max_iterations", positive=True
        )
        h_tolerance = _real(
            self.h_tolerance, "h_tolerance", strictly_positive=True
        )
        rho_max = _real(self.rho_max, "rho_max", strictly_positive=True)
        if rho_max <= 1.0:
            raise ValueError("rho_max must exceed the initial rho of 1")
        rho_multiplier = _real(
            self.rho_multiplier, "rho_multiplier", strictly_positive=True
        )
        if rho_multiplier <= 1.0:
            raise ValueError("rho_multiplier must exceed 1")
        if not isinstance(self.standardize_data, bool):
            raise TypeError("standardize_data must be a bool")
        object.__setattr__(self, "lambda1", lambda1)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "h_tolerance", h_tolerance)
        object.__setattr__(self, "rho_max", rho_max)
        object.__setattr__(self, "rho_multiplier", rho_multiplier)


@dataclass(frozen=True)
class ECADiagnostics:
    """Optimization and acyclicity diagnostics for one deterministic fit."""

    iterations: int
    inner_optimizations: int
    objective_evaluations: int
    gradient_evaluations: int
    converged: bool
    acyclicity: float
    rho: float
    alpha: float
    runtime_seconds: float
    optimizer_success: bool
    optimizer_status: int | None = None
    optimizer_message: str = ""

    def __post_init__(self) -> None:
        for label in (
            "iterations",
            "inner_optimizations",
            "objective_evaluations",
            "gradient_evaluations",
        ):
            object.__setattr__(self, label, _integer(getattr(self, label), label))
        for label in ("acyclicity", "rho", "alpha", "runtime_seconds"):
            value = _real(getattr(self, label), label)
            if label in ("acyclicity", "rho", "runtime_seconds") and value < 0.0:
                raise ValueError(f"{label} must be nonnegative")
            object.__setattr__(self, label, value)
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool")
        if not isinstance(self.optimizer_success, bool):
            raise TypeError("optimizer_success must be a bool")
        if self.optimizer_status is not None:
            object.__setattr__(
                self,
                "optimizer_status",
                _integer(self.optimizer_status, "optimizer_status"),
            )
        if not isinstance(self.optimizer_message, str):
            raise TypeError("optimizer_message must be a string")


@dataclass(frozen=True)
class ECAResult:
    """A point-valued ECA DAG with immutable numerical artifacts."""

    graph: Graph
    weighted_adjacency: np.ndarray
    binary_adjacency: np.ndarray
    objective: float
    data_loss: float
    prior_log_likelihood: float
    prior_penalty: float
    l1_penalty: float
    diagnostics: ECADiagnostics
    config: ECAConfig
    prior: ECAPrior
    evidence_adaptation: EvidenceAdaptation | None = None
    support: Mapping[Graph, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise TypeError("graph must be a cd_llm.core.finite_support.Graph")
        if not isinstance(self.diagnostics, ECADiagnostics):
            raise TypeError("diagnostics must be ECADiagnostics")
        if not isinstance(self.config, ECAConfig):
            raise TypeError("config must be ECAConfig")
        if not isinstance(self.prior, ECAPrior):
            raise TypeError("prior must be ECAPrior")
        if self.evidence_adaptation is not None:
            from .adapters import EvidenceAdaptation

            if not isinstance(self.evidence_adaptation, EvidenceAdaptation):
                raise TypeError("evidence_adaptation must be EvidenceAdaptation or None")
            if self.evidence_adaptation.prior != self.prior:
                raise ValueError("evidence adaptation and result use different priors")
        if self.prior.node_count != self.graph.node_count:
            raise ValueError("prior and graph have different node counts")

        try:
            weighted = np.array(self.weighted_adjacency, dtype=float, copy=True)
            raw_binary = np.array(self.binary_adjacency, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError("adjacency matrices must contain real numbers") from error
        expected_shape = (self.graph.node_count, self.graph.node_count)
        if weighted.shape != expected_shape or raw_binary.shape != expected_shape:
            raise ValueError("adjacency matrices do not match the graph size")
        if not np.all(np.isfinite(weighted)):
            raise ValueError("weighted adjacency must be finite")
        if not np.all(np.isfinite(raw_binary)) or not np.all(
            (raw_binary == 0.0) | (raw_binary == 1.0)
        ):
            raise ValueError("binary adjacency entries must be zero or one")
        binary = raw_binary.astype(np.int8)
        if np.any(np.diag(weighted)) or np.any(np.diag(binary)):
            raise ValueError("adjacency matrices cannot contain self-edges")
        if not np.array_equal(binary, graph_to_adjacency(self.graph)):
            raise ValueError("binary adjacency does not encode graph")
        weighted.setflags(write=False)
        binary.setflags(write=False)
        object.__setattr__(self, "weighted_adjacency", weighted)
        object.__setattr__(self, "binary_adjacency", binary)

        scores = {}
        for label in (
            "objective",
            "data_loss",
            "prior_log_likelihood",
            "prior_penalty",
            "l1_penalty",
        ):
            scores[label] = _real(getattr(self, label), label)
            object.__setattr__(self, label, scores[label])
        expected_objective = math.fsum(
            (scores["data_loss"], scores["prior_penalty"], scores["l1_penalty"])
        )
        if not math.isclose(
            scores["objective"], expected_objective, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(
                "objective does not equal data loss, prior penalty, and L1 penalty"
            )
        object.__setattr__(self, "support", MappingProxyType({self.graph: 1.0}))

    @property
    def converged(self) -> bool:
        return self.diagnostics.converged


__all__ = [
    "Backend",
    "ECAConfig",
    "ECADiagnostics",
    "ECAPrior",
    "ECAResult",
    "EdgeConstraint",
]
