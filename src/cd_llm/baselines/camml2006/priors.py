"""Structural-prior construction and scoring in Totally Ordered Model space."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable

import numpy as np

from cd_llm.core.finite_support import Graph, graph_from_edges, iter_bits, transitive_closure

from .types import CaMMLPrior, ExpertGraphPrior, GlobalPairPrior, LocalPairPrior, TOM


_TOLERANCE = 1e-10


def _checked_optional(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def local_pair_prior(
    left: int,
    right: int,
    *,
    left_to_right: float | None = None,
    right_to_left: float | None = None,
    adjacency: float | None = None,
    left_before_right: float | None = None,
    default_arc_probability: float = 0.5,
) -> LocalPairPrior:
    """Complete elicited local marginals using the paper's proportional rule."""

    a = _checked_optional(left_to_right, "left_to_right")
    b = _checked_optional(right_to_left, "right_to_left")
    arc = _checked_optional(adjacency, "adjacency")
    before = _checked_optional(left_before_right, "left_before_right")
    default_arc = float(default_arc_probability)
    if not math.isfinite(default_arc) or not 0.0 < default_arc < 1.0:
        raise ValueError("default_arc_probability must lie strictly between zero and one")
    direction = 0.5 if before is None else before
    arc_value = default_arc if arc is None else arc

    if a is not None and b is not None:
        if arc is not None and not math.isclose(a + b, arc, abs_tol=_TOLERANCE):
            raise ValueError("directed probabilities contradict adjacency probability")
        arc_value = a + b
        if before is None:
            c = d = (1.0 - arc_value) / 2.0
        else:
            c = before - a
            d = 1.0 - before - b
        values = a, b, c, d
    elif a is not None:
        if arc is not None:
            b = arc_value - a
            return local_pair_prior(
                left,
                right,
                left_to_right=a,
                right_to_left=b,
                adjacency=arc_value,
                left_before_right=before,
                default_arc_probability=default_arc,
            )
        if before is not None:
            values = (
                a,
                (1.0 - before) * default_arc,
                before - a,
                (1.0 - before) * (1.0 - default_arc),
            )
        else:
            baseline_a = 0.5 * default_arc
            scale = (1.0 - a) / (1.0 - baseline_a)
            values = (
                a,
                0.5 * default_arc * scale,
                0.5 * (1.0 - default_arc) * scale,
                0.5 * (1.0 - default_arc) * scale,
            )
    elif b is not None:
        mirrored = local_pair_prior(
            left,
            right,
            left_to_right=b,
            adjacency=arc,
            left_before_right=None if before is None else 1.0 - before,
            default_arc_probability=default_arc,
        ).probabilities
        values = mirrored[1], mirrored[0], mirrored[3], mirrored[2]
    else:
        values = (
            arc_value * direction,
            arc_value * (1.0 - direction),
            (1.0 - arc_value) * direction,
            (1.0 - arc_value) * (1.0 - direction),
        )

    normalized = []
    for value in values:
        if value < -_TOLERANCE or value > 1.0 + _TOLERANCE:
            raise ValueError("inconsistent local relationship probabilities")
        normalized.append(min(1.0, max(0.0, value)))
    if not math.isclose(sum(normalized), 1.0, abs_tol=_TOLERANCE):
        raise ValueError("local relationship probabilities do not sum to one")
    result = LocalPairPrior(left, right, tuple(normalized))  # type: ignore[arg-type]
    p = result.probabilities
    checks = (
        (a, p[0]),
        (b, p[1]),
        (arc, p[0] + p[1]),
        (before, p[0] + p[2]),
    )
    if any(expected is not None and not math.isclose(expected, actual, abs_tol=_TOLERANCE) for expected, actual in checks):
        raise ValueError("completed local table does not reproduce elicited marginals")
    return result


def _random_graph(node_count: int, arc_probability: float, rng: np.random.Generator) -> Graph:
    order = tuple(int(value) for value in rng.permutation(node_count))
    edges = []
    for first in range(node_count):
        for second in range(first + 1, node_count):
            if rng.random() < arc_probability:
                edges.append((order[first], order[second]))
    return graph_from_edges(node_count, edges)


def global_relation_index(graph: Graph, left: int, right: int) -> int:
    """Return L=>R, R=>L, common-cause, or disconnected state index."""

    closure = transitive_closure(graph)
    if closure[left] & (1 << right):
        return 0
    if closure[right] & (1 << left):
        return 1
    both = (1 << left) | (1 << right)
    if any((descendants & both) == both for descendants in closure):
        return 2
    return 3


@functools.lru_cache(maxsize=64)
def default_global_probabilities(
    node_count: int,
    arc_probability: float,
    sample_count: int = 10_000,
    seed: int = 123_456,
) -> tuple[float, float, float, float]:
    if node_count < 2 or sample_count < 1:
        raise ValueError("global default sampling requires at least two nodes and one sample")
    rng = np.random.default_rng(seed)
    counts = np.full(4, 0.5, dtype=float)
    for _ in range(sample_count):
        graph = _random_graph(node_count, arc_probability, rng)
        counts[global_relation_index(graph, 0, 1)] += 1.0
    probabilities = counts / float(np.sum(counts))
    return tuple(float(value) for value in probabilities)  # type: ignore[return-value]


def global_pair_prior(
    left: int,
    right: int,
    *,
    left_ancestor_right: float | None = None,
    right_ancestor_left: float | None = None,
    correlated: float | None = None,
    defaults: tuple[float, float, float, float],
) -> GlobalPairPrior:
    """Complete global marginals by scaling unused default state mass."""

    a = _checked_optional(left_ancestor_right, "left_ancestor_right")
    b = _checked_optional(right_ancestor_left, "right_ancestor_left")
    corr = _checked_optional(correlated, "correlated")
    base = tuple(float(value) for value in defaults)
    if len(base) != 4 or any(value <= 0.0 for value in base) or not math.isclose(sum(base), 1.0, abs_tol=_TOLERANCE):
        raise ValueError("global defaults must be four positive probabilities summing to one")
    fixed: dict[int, float] = {}
    if a is not None:
        fixed[0] = a
    if b is not None:
        fixed[1] = b
    if corr is not None:
        fixed[3] = 1.0 - corr
    fixed_total = sum(fixed.values())
    if fixed_total > 1.0 + _TOLERANCE:
        raise ValueError("inconsistent global relationship probabilities")
    unused = [index for index in range(4) if index not in fixed]
    base_total = sum(base[index] for index in unused)
    if base_total <= 0.0 and unused:
        raise ValueError("global defaults leave no mass for unspecified states")
    values = [0.0] * 4
    for index, value in fixed.items():
        values[index] = value
    for index in unused:
        values[index] = max(0.0, 1.0 - fixed_total) * base[index] / base_total
    result = GlobalPairPrior(left, right, tuple(values))  # type: ignore[arg-type]
    if corr is not None and not math.isclose(sum(result.probabilities[:3]), corr, abs_tol=_TOLERANCE):
        raise ValueError("completed global table does not reproduce correlation marginal")
    return result


def dag_edit_distance(first: Graph, second: Graph) -> int:
    if first.node_count != second.node_count or first.regime != "dag" or second.regime != "dag":
        raise ValueError("DAG edit distance requires equal-size DAGs")
    return sum((left ^ right).bit_count() for left, right in zip(first.directed, second.directed))


def skeleton_edit_distance(first: Graph, second: Graph) -> int:
    if first.node_count != second.node_count:
        raise ValueError("skeleton edit distance requires equal-size graphs")
    count = 0
    for left in range(first.node_count):
        for right in range(left + 1, first.node_count):
            first_edge = bool(first.directed[left] & (1 << right) or first.directed[right] & (1 << left))
            second_edge = bool(second.directed[left] & (1 << right) or second.directed[right] & (1 << left))
            count += first_edge != second_edge
    return count


def kendall_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    if sorted(first) != sorted(second):
        raise ValueError("Kendall orders must contain the same unique nodes")
    position = {node: index for index, node in enumerate(second)}
    mapped = [position[node] for node in first]
    return sum(mapped[i] > mapped[j] for i in range(len(mapped)) for j in range(i + 1, len(mapped)))


def closest_topological_order(graph: Graph, reference: tuple[int, ...]) -> tuple[int, ...]:
    """Stable topological order minimizing adjacent swaps from ``reference``."""

    if graph.node_count != len(reference) or sorted(reference) != list(range(graph.node_count)):
        raise ValueError("reference must be a graph-node permutation")
    remaining = list(reference)
    placed: set[int] = set()
    parents = [set() for _ in range(graph.node_count)]
    for source, row in enumerate(graph.directed):
        for target in iter_bits(row):
            parents[target].add(source)
    result = []
    while remaining:
        for index, node in enumerate(remaining):
            if parents[node] <= placed:
                result.append(node)
                placed.add(node)
                remaining.pop(index)
                break
        else:
            raise ValueError("expert graph contains a cycle")
    return tuple(result)


def tom_edit_distance(expert: Graph, tom: TOM) -> int:
    closest = closest_topological_order(expert, tom.order)
    return skeleton_edit_distance(expert, tom.graph) + kendall_distance(tom.order, closest)


def _log_probability(value: float) -> float:
    return -math.inf if value == 0.0 else math.log(value)


class StructurePriorScorer:
    """Higher-is-better unnormalized log prior over TOMs."""

    def __init__(
        self,
        prior: CaMMLPrior,
        *,
        arc_probability: float,
    ) -> None:
        self.prior = prior
        self.node_count = prior.node_count
        default = local_pair_prior(
            0,
            1,
            adjacency=arc_probability,
            default_arc_probability=arc_probability,
        ).probabilities
        supplied = {(value.left, value.right): value.probabilities for value in prior.local}
        self.local_tables = {
            (left, right): supplied.get((left, right), default)
            for left in range(self.node_count)
            for right in range(left + 1, self.node_count)
        }
        self.global_tables = {
            (value.left, value.right): value.probabilities for value in prior.global_
        }

    def local_state(self, tom: TOM, left: int, right: int) -> int:
        position = {node: index for index, node in enumerate(tom.order)}
        if tom.graph.directed[left] & (1 << right):
            return 0
        if tom.graph.directed[right] & (1 << left):
            return 1
        return 2 if position[left] < position[right] else 3

    def local_pair_log_score(self, tom: TOM, left: int, right: int) -> float:
        return _log_probability(self.local_tables[(left, right)][self.local_state(tom, left, right)])

    def local_log_score(self, tom: TOM) -> float:
        return sum(
            self.local_pair_log_score(tom, left, right)
            for left in range(self.node_count)
            for right in range(left + 1, self.node_count)
        )

    def nonlocal_log_score(self, tom: TOM) -> float:
        score = 0.0
        for (left, right), probabilities in self.global_tables.items():
            score += _log_probability(probabilities[global_relation_index(tom.graph, left, right)])
        for value in self.prior.graph_priors:
            distance = (
                dag_edit_distance(value.graph, tom.graph)
                if value.distance == "dag"
                else tom_edit_distance(value.graph, tom)
            )
            if value.confidence == 1.0:
                if distance:
                    return -math.inf
                continue
            if value.confidence == 0.0:
                if distance == 0:
                    return -math.inf
                continue
            score -= distance * math.log(value.confidence / (1.0 - value.confidence))
        return score

    def score(self, tom: TOM) -> float:
        return self.local_log_score(tom) + self.nonlocal_log_score(tom)


__all__ = [
    "StructurePriorScorer",
    "closest_topological_order",
    "dag_edit_distance",
    "default_global_probabilities",
    "global_pair_prior",
    "global_relation_index",
    "kendall_distance",
    "local_pair_prior",
    "skeleton_edit_distance",
    "tom_edit_distance",
]
