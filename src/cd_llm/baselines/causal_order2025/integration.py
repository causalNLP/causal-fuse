"""Paper-defined causal-order integration for constraint- and score-based search."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import networkx as nx
import numpy as np

from cd_llm.baselines.camml2006 import CaMMLPrior, local_pair_prior
from cd_llm.core.finite_support import (
    Endpoint,
    Graph,
    PartialGraph,
    graph_from_edges,
    topological_order,
)

from .types import (
    CausalOrderInfeasibleError,
    Edge,
    FallbackOrienter,
    TripletExpertResult,
)


@dataclass(frozen=True)
class ConstraintIntegration:
    graph: Graph
    final_order: tuple[int, ...]
    fallback_orientations: tuple[Edge, ...]
    expert_fallback_pairs: tuple[tuple[int, int], ...] = ()
    cycle_guard_pairs: tuple[tuple[int, int], ...] = ()
    forced_safe_orientations: tuple[Edge, ...] = ()


def causal_learn_cpdag(graph: object) -> PartialGraph:
    """Convert a causal-learn CPDAG while preserving every endpoint mark."""

    if not hasattr(graph, "graph"):
        raise TypeError("causal-learn graph must expose its endpoint matrix as .graph")
    raw = np.asarray(getattr(graph, "graph"), dtype=int)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("causal-learn graph matrix must be square")
    endpoint_map = {
        0: Endpoint.NULL,
        -1: Endpoint.TAIL,
        1: Endpoint.ARROW,
        2: Endpoint.CIRCLE,
    }
    converted = np.zeros_like(raw)
    for index, value in np.ndenumerate(raw):
        try:
            converted[index] = int(endpoint_map[int(value)])
        except KeyError as error:
            raise ValueError(
                f"unsupported causal-learn endpoint code {int(value)}"
            ) from error
    return PartialGraph.from_matrix(converted, "cpdag")


def integrate_order_into_cpdag(
    cpdag: PartialGraph,
    expert: TripletExpertResult,
    *,
    fallback_orienter: FallbackOrienter | None = None,
) -> ConstraintIntegration:
    """Implement Appendix-D Algorithm 1 with an edgewise acyclicity guard."""

    if not isinstance(cpdag, PartialGraph) or cpdag.kind != "cpdag":
        raise TypeError("Algorithm 1 requires a CPDAG PartialGraph")
    if not isinstance(expert, TripletExpertResult):
        raise TypeError("expert must be TripletExpertResult")
    if cpdag.node_count != len(expert.node_names):
        raise ValueError("CPDAG and expert order have different node counts")
    if fallback_orienter is not None and not callable(fallback_orienter):
        raise TypeError("fallback_orienter must be callable or None")

    rank = {node: index for index, node in enumerate(expert.expert_order)}
    directed: list[Edge] = []
    undirected: list[tuple[int, int]] = []
    fallbacks: list[Edge] = []
    fallback_pairs: list[tuple[int, int]] = []
    cycle_guard_pairs: list[tuple[int, int]] = []
    forced: list[Edge] = []
    for first in range(cpdag.node_count):
        for second in range(first + 1, cpdag.node_count):
            first_mark = cpdag.mark(first, second)
            second_mark = cpdag.mark(second, first)
            marks = first_mark, second_mark
            if marks == (Endpoint.NULL, Endpoint.NULL):
                continue
            if marks == (Endpoint.TAIL, Endpoint.ARROW):
                directed.append((first, second))
                continue
            if marks == (Endpoint.ARROW, Endpoint.TAIL):
                directed.append((second, first))
                continue
            if marks == (Endpoint.TAIL, Endpoint.TAIL):
                undirected.append((first, second))
                continue
            else:
                raise CausalOrderInfeasibleError(
                    f"Algorithm 1 cannot orient CPDAG endpoint pair {marks}"
                )

    try:
        graph_from_edges(cpdag.node_count, directed=directed)
    except ValueError as error:
        raise CausalOrderInfeasibleError("directed PC output contains a cycle") from error

    working = nx.DiGraph()
    working.add_nodes_from(range(cpdag.node_count))
    working.add_edges_from(directed)

    def acyclic(edge: Edge) -> bool:
        source, target = edge
        return not nx.has_path(working, target, source)

    for first, second in undirected:
        proposed: Edge | None = None
        reason = "missing_order"
        if first in rank and second in rank:
            proposed = (first, second) if rank[first] < rank[second] else (second, first)
            if acyclic(proposed):
                directed.append(proposed)
                working.add_edge(*proposed)
                continue
            reason = "order_cycle"
            cycle_guard_pairs.append((first, second))

        if fallback_orienter is None:
            if reason == "order_cycle":
                message = "the expert-order direction would form a cycle"
            else:
                message = "an undirected PC edge has an endpoint absent from the expert order"
            raise CausalOrderInfeasibleError(message + ", but no superior-expert fallback was supplied")

        expert_edge = fallback_orienter(first, second)
        choices = ((first, second), (second, first))
        edge = expert_edge if expert_edge in choices and acyclic(expert_edge) else None
        if edge is None:
            safe = tuple(candidate for candidate in choices if acyclic(candidate))
            if len(safe) != 1:
                raise CausalOrderInfeasibleError(
                    "fallback expert did not supply a usable direction and no unique acyclic direction exists"
                )
            edge = safe[0]
            forced.append(edge)
        fallback_pairs.append((first, second))
        fallbacks.append(edge)
        directed.append(edge)
        working.add_edge(*edge)

    graph = graph_from_edges(cpdag.node_count, directed=directed)
    return ConstraintIntegration(
        graph=graph,
        final_order=topological_order(graph),
        fallback_orientations=tuple(fallbacks),
        expert_fallback_pairs=tuple(fallback_pairs),
        cycle_guard_pairs=tuple(cycle_guard_pairs),
        forced_safe_orientations=tuple(forced),
    )


def level_order_from_edges(
    node_count: int, edges: Iterable[Edge]
) -> dict[int, int]:
    """Resolve Algorithm 2's cyclic expert graph via SCC condensation."""

    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise TypeError("node_count must be an integer")
    if node_count < 1:
        raise ValueError("node_count must be positive")
    normalized: list[Edge] = []
    for raw_edge in edges:
        try:
            source, target = raw_edge
        except (TypeError, ValueError) as error:
            raise TypeError("edges must contain directed node pairs") from error
        if (
            isinstance(source, bool)
            or isinstance(target, bool)
            or not isinstance(source, int)
            or not isinstance(target, int)
        ):
            raise TypeError("edge endpoints must be integers")
        if source == target or not 0 <= source < node_count or not 0 <= target < node_count:
            raise ValueError("edge endpoints must be distinct nodes inside the graph")
        normalized.append((source, target))
    connected = {node for edge in normalized for node in edge}
    if not connected:
        return {}
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(connected))
    graph.add_edges_from(normalized)
    components = tuple(nx.strongly_connected_components(graph))
    condensation = nx.condensation(graph, components)
    component_levels: dict[int, int] = {}
    for component in nx.topological_sort(condensation):
        predecessors = tuple(condensation.predecessors(component))
        component_levels[component] = (
            0
            if not predecessors
            else 1 + max(component_levels[parent] for parent in predecessors)
        )
    mapping = condensation.graph["mapping"]
    return {
        node: component_levels[int(mapping[node])]
        for node in sorted(connected)
    }


def level_order_prior(
    node_count: int,
    levels: dict[int, int],
    *,
    prior_probability: float,
    arc_probability: float,
) -> CaMMLPrior:
    """Encode cross-level precedence without asserting edge existence."""

    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    node_count = int(node_count)
    if node_count < 1:
        raise ValueError("node_count must be positive")
    if not isinstance(levels, dict):
        raise TypeError("levels must be a dictionary")
    normalized_levels: dict[int, int] = {}
    for raw_node, raw_level in levels.items():
        if (
            isinstance(raw_node, bool)
            or isinstance(raw_level, bool)
            or not isinstance(raw_node, Integral)
            or not isinstance(raw_level, Integral)
        ):
            raise TypeError("level nodes and values must be integers")
        node, level = int(raw_node), int(raw_level)
        if not 0 <= node < node_count:
            raise ValueError("level node is outside the graph")
        if level < 0:
            raise ValueError("level values must be nonnegative")
        normalized_levels[node] = level
    if isinstance(prior_probability, bool) or not isinstance(prior_probability, Real):
        raise TypeError("prior_probability must be a real number")
    probability = float(prior_probability)
    if not np.isfinite(probability) or not 0.5 <= probability <= 1.0:
        raise ValueError("prior_probability must lie in [0.5, 1]")
    if isinstance(arc_probability, bool) or not isinstance(arc_probability, Real):
        raise TypeError("arc_probability must be a real number")
    normalized_arc_probability = float(arc_probability)
    if (
        not np.isfinite(normalized_arc_probability)
        or not 0.0 < normalized_arc_probability < 1.0
    ):
        raise ValueError("arc_probability must lie strictly between zero and one")
    local = []
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if (
                left not in normalized_levels
                or right not in normalized_levels
                or normalized_levels[left] == normalized_levels[right]
            ):
                continue
            left_before = (
                probability
                if normalized_levels[left] < normalized_levels[right]
                else 1.0 - probability
            )
            local.append(
                local_pair_prior(
                    left,
                    right,
                    left_before_right=left_before,
                    default_arc_probability=normalized_arc_probability,
                )
            )
    return CaMMLPrior(node_count=node_count, local=tuple(local))


__all__ = [
    "ConstraintIntegration",
    "causal_learn_cpdag",
    "integrate_order_into_cpdag",
    "level_order_from_edges",
    "level_order_prior",
]
