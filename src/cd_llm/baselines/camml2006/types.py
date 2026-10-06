"""Immutable public values for the O'Donnell et al. CaMML adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, Mapping

from cd_llm.core.finite_support import Graph, topological_order


DistanceKind = Literal["dag", "tom"]
DataScoreName = Literal["adaptive-mml", "bdeu", "gaussian-bic"]


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _probability(value: object, label: str, *, interior: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if interior:
        if not 0.0 < result < 1.0:
            raise ValueError(f"{label} must lie strictly between zero and one")
    elif not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def _probability_vector(values: object, label: str) -> tuple[float, float, float, float]:
    try:
        result = tuple(_probability(value, label) for value in values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must contain four probabilities") from error
    if len(result) != 4:
        raise ValueError(f"{label} must contain four probabilities")
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{label} must sum to one")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class LocalPairPrior:
    """Probabilities of L->R, R->L, L-before-no-edge, and R-before-no-edge."""

    left: int
    right: int
    probabilities: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        left = _integer(self.left, "left")
        right = _integer(self.right, "right")
        if left >= right:
            raise ValueError("local-pair endpoints must satisfy left < right")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(
            self,
            "probabilities",
            _probability_vector(self.probabilities, "local probabilities"),
        )


@dataclass(frozen=True, order=True)
class GlobalPairPrior:
    """Probabilities of L=>R, R=>L, common cause, and disconnection."""

    left: int
    right: int
    probabilities: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        left = _integer(self.left, "left")
        right = _integer(self.right, "right")
        if left >= right:
            raise ValueError("global-pair endpoints must satisfy left < right")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(
            self,
            "probabilities",
            _probability_vector(self.probabilities, "global probabilities"),
        )


@dataclass(frozen=True)
class ExpertGraphPrior:
    """A full expert DAG with DAG- or TOM-edit confidence."""

    graph: Graph
    confidence: float
    distance: DistanceKind = "dag"

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph) or self.graph.regime != "dag":
            raise TypeError("expert graph prior requires a DAG Graph")
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if self.distance not in ("dag", "tom"):
            raise ValueError("distance must be 'dag' or 'tom'")


@dataclass(frozen=True)
class CaMMLPrior:
    """Validated collection of pairwise and full-network structural priors."""

    node_count: int
    local: tuple[LocalPairPrior, ...] = ()
    global_: tuple[GlobalPairPrior, ...] = ()
    graph_priors: tuple[ExpertGraphPrior, ...] = ()

    def __post_init__(self) -> None:
        node_count = _integer(self.node_count, "node_count", minimum=1)
        local = tuple(self.local)
        global_ = tuple(self.global_)
        graph_priors = tuple(self.graph_priors)
        if any(not isinstance(value, LocalPairPrior) for value in local):
            raise TypeError("local must contain LocalPairPrior values")
        if any(not isinstance(value, GlobalPairPrior) for value in global_):
            raise TypeError("global_ must contain GlobalPairPrior values")
        if any(not isinstance(value, ExpertGraphPrior) for value in graph_priors):
            raise TypeError("graph_priors must contain ExpertGraphPrior values")
        for collection, label in ((local, "local"), (global_, "global")):
            seen: set[tuple[int, int]] = set()
            for value in collection:
                pair = value.left, value.right
                if value.right >= node_count:
                    raise ValueError(f"{label} prior endpoint is outside the graph")
                if pair in seen:
                    raise ValueError(f"duplicate {label} prior for pair {pair}")
                seen.add(pair)
        for value in graph_priors:
            if value.graph.node_count != node_count:
                raise ValueError("expert graph prior has the wrong node count")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "local", tuple(sorted(local)))
        object.__setattr__(self, "global_", tuple(sorted(global_)))
        object.__setattr__(self, "graph_priors", graph_priors)

    @classmethod
    def empty(cls, node_count: int) -> "CaMMLPrior":
        return cls(node_count)


@dataclass(frozen=True, order=True)
class TOM:
    """A DAG together with one compatible total ordering."""

    order: tuple[int, ...]
    graph: Graph

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph) or self.graph.regime != "dag":
            raise TypeError("TOM graph must be a DAG Graph")
        order = tuple(_integer(value, "order entry") for value in self.order)
        if sorted(order) != list(range(self.graph.node_count)):
            raise ValueError("TOM order must be a permutation of graph nodes")
        position = {node: index for index, node in enumerate(order)}
        for source, row in enumerate(self.graph.directed):
            for target in range(self.graph.node_count):
                if row & (1 << target) and position[source] >= position[target]:
                    raise ValueError("TOM graph contains an arc contrary to its order")
        object.__setattr__(self, "order", order)

    @classmethod
    def from_graph(cls, graph: Graph) -> "TOM":
        return cls(topological_order(graph), graph)


@dataclass(frozen=True)
class CaMMLConfig:
    seed: int = 0
    chain_count: int = 4
    iterations: int = 25_000
    burn_in: int = 5_000
    thin: int = 10
    max_parents: int = 7
    temperature: float = 1.0
    arc_probability: float = 0.5
    global_prior_samples: int = 10_000
    initialization_attempts: int = 10_000

    def __post_init__(self) -> None:
        seed = _integer(self.seed, "seed")
        chain_count = _integer(self.chain_count, "chain_count", minimum=1)
        iterations = _integer(self.iterations, "iterations", minimum=2)
        burn_in = _integer(self.burn_in, "burn_in")
        thin = _integer(self.thin, "thin", minimum=1)
        max_parents = _integer(self.max_parents, "max_parents")
        global_samples = _integer(
            self.global_prior_samples, "global_prior_samples", minimum=1
        )
        attempts = _integer(
            self.initialization_attempts, "initialization_attempts", minimum=1
        )
        if burn_in >= iterations:
            raise ValueError("burn_in must be smaller than iterations")
        if (iterations - burn_in) // thin < 2:
            raise ValueError("configuration retains fewer than two samples per chain")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise TypeError("temperature must be a real number")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "chain_count", chain_count)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "burn_in", burn_in)
        object.__setattr__(self, "thin", thin)
        object.__setattr__(self, "max_parents", max_parents)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(
            self, "arc_probability", _probability(self.arc_probability, "arc_probability", interior=True)
        )
        object.__setattr__(self, "global_prior_samples", global_samples)
        object.__setattr__(self, "initialization_attempts", attempts)


@dataclass(frozen=True)
class CaMMLDiagnostics:
    attempted_moves: int
    accepted_moves: int
    acceptance_rate: float
    proposal_attempts: Mapping[str, int]
    proposal_accepts: Mapping[str, int]
    retained_samples: int
    unique_toms: int
    unique_dags: int
    score_ess: float
    edge_count_ess: float
    score_rhat: float
    edge_count_rhat: float
    runtime_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_attempts", MappingProxyType(dict(self.proposal_attempts)))
        object.__setattr__(self, "proposal_accepts", MappingProxyType(dict(self.proposal_accepts)))


@dataclass(frozen=True)
class CaMMLResult:
    graph: Graph
    support: Mapping[Graph, float]
    map_tom: TOM
    data_log_score: float
    structure_log_prior: float
    total_log_score: float
    diagnostics: CaMMLDiagnostics
    config: CaMMLConfig
    prior: CaMMLPrior
    data_score_name: str
    evidence_adaptation: object | None = None

    def __post_init__(self) -> None:
        support = dict(self.support)
        if not support or self.graph not in support:
            raise ValueError("support must contain the returned graph")
        if any(not isinstance(graph, Graph) or graph.regime != "dag" for graph in support):
            raise TypeError("support must contain DAG Graph keys")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in support.values()):
            raise ValueError("support probabilities must be positive and finite")
        total = sum(float(value) for value in support.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("support probabilities must sum to one")
        object.__setattr__(self, "support", MappingProxyType(support))
