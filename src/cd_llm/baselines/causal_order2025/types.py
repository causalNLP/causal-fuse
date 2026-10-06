"""Immutable public values for the Vashishtha et al. causal-order baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping

from cd_llm.baselines.camml2006 import CaMMLPrior, CaMMLResult
from cd_llm.core.finite_support import Graph, PartialGraph, iter_bits


Edge = tuple[int, int]
FallbackOrienter = Callable[[int, int], Edge | None]


class CausalOrderError(RuntimeError):
    """Base error for the causal-order baseline package."""


class CausalOrderInfeasibleError(CausalOrderError):
    """The paper-defined integration cannot produce its required DAG/order."""


class UpstreamCausalOrderError(CausalOrderError):
    """The pinned authors' implementation is unavailable or returned invalid output."""


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _finite_real(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return result


def normalize_node_names(values: object, *, minimum: int = 1) -> tuple[str, ...]:
    try:
        names = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("node_names must be an iterable of strings") from error
    if len(names) < minimum:
        raise ValueError(f"node_names must contain at least {minimum} entries")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("node_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("node_names must be unique")
    return names


def _normalized_edges(values: object, node_count: int, label: str) -> tuple[Edge, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable of directed pairs") from error
    edges: list[Edge] = []
    for value in raw:
        try:
            source, target = value
        except (TypeError, ValueError) as error:
            raise TypeError(f"{label} must contain directed pairs") from error
        source = _integer(source, f"{label} source")
        target = _integer(target, f"{label} target")
        if source >= node_count or target >= node_count:
            raise ValueError(f"{label} endpoint is outside the graph")
        if source == target:
            raise ValueError(f"{label} cannot contain self-edges")
        edges.append((source, target))
    if len(set(edges)) != len(edges):
        raise ValueError(f"{label} cannot contain duplicate edges")
    return tuple(edges)


@dataclass(frozen=True)
class TripletConfig:
    """Arguments forwarded unchanged to the pinned upstream triplet strategy."""

    subgraph_model: str = "gpt-4o-mini"
    expert_model: str = "gpt-4o"
    delay: float = 12.0

    def __post_init__(self) -> None:
        for label, value in (
            ("subgraph_model", self.subgraph_model),
            ("expert_model", self.expert_model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a nonempty string")
        object.__setattr__(self, "delay", _finite_real(self.delay, "delay"))


@dataclass(frozen=True)
class PCOrderConfig:
    """Project defaults for the paper's otherwise-unspecified PC invocation."""

    alpha: float = 0.05
    indep_test: str = "fisherz"
    stable: bool = True
    uc_rule: int = 0
    uc_priority: int = 2
    mvpc: bool = False

    def __post_init__(self) -> None:
        alpha = _finite_real(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if not isinstance(self.indep_test, str) or not self.indep_test.strip():
            raise ValueError("indep_test must be a nonempty causal-learn test name")
        for label in ("stable", "mvpc"):
            if not isinstance(getattr(self, label), bool):
                raise TypeError(f"{label} must be boolean")
        object.__setattr__(self, "uc_rule", _integer(self.uc_rule, "uc_rule"))
        object.__setattr__(
            self, "uc_priority", _integer(self.uc_priority, "uc_priority")
        )
        object.__setattr__(self, "alpha", alpha)


@dataclass(frozen=True)
class TripletExpertResult:
    """Validated project-native view of one upstream triplet elicitation."""

    node_names: tuple[str, ...]
    raw_edges: tuple[Edge, ...]
    graph: Graph
    edgewise_distributions: Mapping[tuple[int, int], tuple[float, float, float]]
    subgroup_list: tuple[tuple[int, int, int], ...]
    subgroup_responses: tuple[str, ...]
    expert_order: tuple[int, ...]
    isolated_nodes: tuple[int, ...]
    cycle_removal_threshold: float | None
    upstream_revision: str
    config: TripletConfig
    runtime_seconds: float

    def __post_init__(self) -> None:
        names = normalize_node_names(self.node_names, minimum=3)
        node_count = len(names)
        raw_edges = _normalized_edges(self.raw_edges, node_count, "raw_edges")
        if not isinstance(self.graph, Graph) or self.graph.node_count != node_count:
            raise ValueError("graph must be a project Graph over node_names")
        if self.graph.regime != "dag":
            raise ValueError("cycle-removed expert graph must be a DAG")

        distributions: dict[tuple[int, int], tuple[float, float, float]] = {}
        for pair, probabilities in self.edgewise_distributions.items():
            try:
                left, right = pair
            except (TypeError, ValueError) as error:
                raise TypeError("edgewise-distribution keys must be node pairs") from error
            left = _integer(left, "edgewise left")
            right = _integer(right, "edgewise right")
            if left >= right or right >= node_count:
                raise ValueError("edgewise pairs must satisfy 0 <= left < right < node_count")
            try:
                values = tuple(float(value) for value in probabilities)
            except (TypeError, ValueError) as error:
                raise TypeError("edgewise values must contain three probabilities") from error
            if (
                len(values) != 3
                or any(not math.isfinite(value) or value < 0.0 for value in values)
                or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-10)
            ):
                raise ValueError("edgewise values must be three probabilities summing to one")
            distributions[(left, right)] = values  # type: ignore[assignment]

        subgroups = tuple(tuple(group) for group in self.subgroup_list)
        normalized_subgroups: list[tuple[int, int, int]] = []
        for group in subgroups:
            if len(group) != 3:
                raise ValueError("subgroup_list must contain triplets")
            normalized = tuple(_integer(node, "triplet node") for node in group)
            if len(set(normalized)) != 3 or any(node >= node_count for node in normalized):
                raise ValueError("each subgroup must contain three distinct graph nodes")
            normalized_subgroups.append(normalized)  # type: ignore[arg-type]
        responses = tuple(self.subgroup_responses)
        if len(responses) != len(normalized_subgroups) or any(
            not isinstance(response, str) for response in responses
        ):
            raise ValueError("subgroup_responses must align one-to-one with subgroup_list")

        connected: set[int] = set()
        for source, row in enumerate(self.graph.directed):
            for target in iter_bits(row):
                connected.update((source, target))
        order = tuple(_integer(node, "expert-order node") for node in self.expert_order)
        if len(set(order)) != len(order) or set(order) != connected:
            raise ValueError("expert_order must contain every non-isolated expert node exactly once")
        position = {node: index for index, node in enumerate(order)}
        for source, row in enumerate(self.graph.directed):
            for target in iter_bits(row):
                if position[source] >= position[target]:
                    raise ValueError("expert_order is inconsistent with the expert DAG")
        isolated = tuple(sorted(_integer(node, "isolated node") for node in self.isolated_nodes))
        if len(set(isolated)) != len(isolated) or set(isolated) != set(range(node_count)) - connected:
            raise ValueError("isolated_nodes must be the complement of expert graph nodes")

        threshold = self.cycle_removal_threshold
        if threshold is not None:
            threshold = _finite_real(threshold, "cycle_removal_threshold")
        if not isinstance(self.upstream_revision, str) or not self.upstream_revision:
            raise ValueError("upstream_revision must be nonempty")
        if not isinstance(self.config, TripletConfig):
            raise TypeError("config must be TripletConfig")

        object.__setattr__(self, "node_names", names)
        object.__setattr__(self, "raw_edges", raw_edges)
        object.__setattr__(self, "edgewise_distributions", MappingProxyType(distributions))
        object.__setattr__(self, "subgroup_list", tuple(normalized_subgroups))
        object.__setattr__(self, "subgroup_responses", responses)
        object.__setattr__(self, "expert_order", order)
        object.__setattr__(self, "isolated_nodes", isolated)
        object.__setattr__(self, "cycle_removal_threshold", threshold)
        object.__setattr__(
            self, "runtime_seconds", _finite_real(self.runtime_seconds, "runtime_seconds")
        )


@dataclass(frozen=True)
class PCOrderResult:
    """Successful Algorithm-1 output with a point-mass project support."""

    graph: Graph
    final_order: tuple[int, ...]
    cpdag: PartialGraph
    expert: TripletExpertResult
    fallback_orientations: tuple[Edge, ...]
    config: PCOrderConfig
    pc_runtime_seconds: float
    integration_runtime_seconds: float
    support: Mapping[Graph, float] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph) or self.graph.regime != "dag":
            raise TypeError("Algorithm 1 must return a DAG Graph")
        if not isinstance(self.cpdag, PartialGraph) or self.cpdag.kind != "cpdag":
            raise TypeError("cpdag must be a CPDAG PartialGraph")
        if self.cpdag.node_count != self.graph.node_count:
            raise ValueError("CPDAG and result graph sizes differ")
        order = tuple(_integer(node, "final-order node") for node in self.final_order)
        if sorted(order) != list(range(self.graph.node_count)):
            raise ValueError("final_order must be a permutation of every graph node")
        position = {node: index for index, node in enumerate(order)}
        for source, row in enumerate(self.graph.directed):
            for target in iter_bits(row):
                if position[source] >= position[target]:
                    raise ValueError("final_order is inconsistent with graph")
        fallback = _normalized_edges(
            self.fallback_orientations, self.graph.node_count, "fallback_orientations"
        )
        if not isinstance(self.expert, TripletExpertResult):
            raise TypeError("expert must be TripletExpertResult")
        if len(self.expert.node_names) != self.graph.node_count:
            raise ValueError("expert node names have the wrong size")
        if not isinstance(self.config, PCOrderConfig):
            raise TypeError("config must be PCOrderConfig")
        object.__setattr__(self, "final_order", order)
        object.__setattr__(self, "fallback_orientations", fallback)
        object.__setattr__(
            self, "pc_runtime_seconds", _finite_real(self.pc_runtime_seconds, "pc_runtime_seconds")
        )
        object.__setattr__(
            self,
            "integration_runtime_seconds",
            _finite_real(self.integration_runtime_seconds, "integration_runtime_seconds"),
        )
        object.__setattr__(self, "support", MappingProxyType({self.graph: 1.0}))


@dataclass(frozen=True)
class CaMMLOrderResult:
    """Algorithm-2 wrapper retaining the complete underlying CaMML result."""

    camml: CaMMLResult
    expert: TripletExpertResult
    level_order: Mapping[int, int]
    prior_probability: float
    prior: CaMMLPrior

    def __post_init__(self) -> None:
        if not isinstance(self.camml, CaMMLResult):
            raise TypeError("camml must be CaMMLResult")
        if not isinstance(self.expert, TripletExpertResult):
            raise TypeError("expert must be TripletExpertResult")
        if self.camml.graph.node_count != len(self.expert.node_names):
            raise ValueError("CaMML and expert graph sizes differ")
        levels: dict[int, int] = {}
        for node, level in self.level_order.items():
            normalized_node = _integer(node, "level-order node")
            normalized_level = _integer(level, "level-order level")
            if normalized_node >= self.camml.graph.node_count:
                raise ValueError("level-order node is outside the graph")
            levels[normalized_node] = normalized_level
        probability = _finite_real(self.prior_probability, "prior_probability")
        if not 0.5 <= probability <= 1.0:
            raise ValueError("prior_probability must lie in [0.5, 1]")
        if not isinstance(self.prior, CaMMLPrior) or self.prior != self.camml.prior:
            raise ValueError("prior must equal the prior used by CaMML")
        object.__setattr__(self, "level_order", MappingProxyType(levels))
        object.__setattr__(self, "prior_probability", probability)

    @property
    def graph(self) -> Graph:
        return self.camml.graph

    @property
    def support(self) -> Mapping[Graph, float]:
        return self.camml.support

    @property
    def final_order(self) -> tuple[int, ...]:
        return self.camml.map_tom.order


__all__ = [
    "CaMMLOrderResult",
    "CausalOrderError",
    "CausalOrderInfeasibleError",
    "Edge",
    "FallbackOrienter",
    "PCOrderConfig",
    "PCOrderResult",
    "TripletConfig",
    "TripletExpertResult",
    "UpstreamCausalOrderError",
    "normalize_node_names",
]
