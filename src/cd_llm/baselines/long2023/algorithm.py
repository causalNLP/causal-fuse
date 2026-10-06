"""Finite-MEC implementation of Long et al. (2023)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from cd_llm.core.finite_support import (
    Endpoint,
    Graph,
    PartialGraph,
    _causal_learn_partial,
    edge_marks,
    graph_from_edges,
    is_compatible_completion,
)
from cd_llm.methods.adaptive_pruning import (
    CompletionEnumerationLimitError,
    enumerate_cpdag_extensions,
)


Expert = Callable[[int, int, str, str], tuple[tuple[int, int], float, dict]]


@dataclass(frozen=True)
class LongResult:
    initial_mec: tuple[Graph, ...]
    final_mec: tuple[Graph, ...]
    decisions: tuple[tuple[int, int], ...]
    retained_probability: float
    expert_records: tuple[dict, ...]


def fit_pc(data: np.ndarray, names: Sequence[str], alpha: float) -> PartialGraph:
    """Fit the finite-data PC initialization used by the benchmark adapter."""
    from causallearn.search.ConstraintBased.PC import pc

    result = pc(
        np.asarray(data),
        alpha=alpha,
        indep_test="gsq",
        stable=True,
        uc_rule=0,
        uc_priority=2,
        mvpc=False,
        correction_name="MV_Crtn_Fisher_Z",
        background_knowledge=None,
        verbose=False,
        show_progress=False,
        node_names=list(names),
    )
    return _causal_learn_partial(result.G, "cpdag")


def ambiguous_pairs(cpdag: PartialGraph) -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, second)
        for first in range(cpdag.node_count)
        for second in range(first + 1, cpdag.node_count)
        if (cpdag.mark(first, second), cpdag.mark(second, first))
        == (Endpoint.TAIL, Endpoint.TAIL)
    )


def enumerate_long_mec(
    cpdag: PartialGraph, max_extensions: int = 50_000
) -> tuple[Graph, ...]:
    """Exactly enumerate the compatible DAGs considered by Long's rules."""
    pairs = ambiguous_pairs(cpdag)
    if len(pairs) > 20:
        return enumerate_cpdag_extensions(cpdag, max_extensions=max_extensions)
    fixed: list[tuple[int, int]] = []
    for first in range(cpdag.node_count):
        for second in range(first + 1, cpdag.node_count):
            marks = (cpdag.mark(first, second), cpdag.mark(second, first))
            if marks == (Endpoint.TAIL, Endpoint.ARROW):
                fixed.append((first, second))
            elif marks == (Endpoint.ARROW, Endpoint.TAIL):
                fixed.append((second, first))
    completions: list[Graph] = []
    for mask in range(1 << len(pairs)):
        edges = list(fixed)
        edges.extend(
            (second, first) if mask & (1 << index) else (first, second)
            for index, (first, second) in enumerate(pairs)
        )
        try:
            graph = graph_from_edges(cpdag.node_count, directed=edges)
        except ValueError:
            continue
        if is_compatible_completion(cpdag, graph):
            completions.append(graph)
            if len(completions) > max_extensions:
                raise CompletionEnumerationLimitError(
                    f"CPDAG has more than {max_extensions} consistent extensions"
                )
    if not completions:
        raise ValueError("CPDAG has no compatible DAG extension")
    return tuple(sorted(completions))


def _fixed_orientations(
    mec: Sequence[Graph], pairs: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    fixed = []
    for first, second in pairs:
        forward = all(
            edge_marks(graph, first, second) == (Endpoint.TAIL, Endpoint.ARROW)
            for graph in mec
        )
        backward = all(
            edge_marks(graph, first, second) == (Endpoint.ARROW, Endpoint.TAIL)
            for graph in mec
        )
        if forward:
            fixed.append((first, second))
        elif backward:
            fixed.append((second, first))
    return tuple(fixed)


def _contains(graph: Graph, arc: tuple[int, int]) -> bool:
    return bool(graph.directed[arc[0]] & (1 << arc[1]))


def _likelihood(
    graph: Graph,
    observations: Sequence[tuple[int, int]],
    confidence: Mapping[tuple[int, int], float],
) -> float:
    value = 1.0
    for arc in observations:
        probability = confidence[arc]
        value *= probability if _contains(graph, arc) else 1.0 - probability
    return value


def run_long(
    cpdag: PartialGraph,
    names: Sequence[str],
    expert: Expert,
    *,
    strategy: str,
    tolerance: float = 0.1,
    max_extensions: int = 50_000,
    initial_mec: Sequence[Graph] | None = None,
) -> LongResult:
    """Apply the paper's Srisk or Ssize greedy rule to a finite MEC."""
    if strategy not in {"srisk", "ssize"}:
        raise ValueError("strategy must be srisk or ssize")
    initial = (
        tuple(initial_mec)
        if initial_mec is not None
        else enumerate_long_mec(cpdag, max_extensions=max_extensions)
    )
    if not initial:
        raise ValueError("initial MEC must not be empty")
    if len(initial) > max_extensions:
        raise CompletionEnumerationLimitError(
            f"CPDAG has more than {max_extensions} consistent extensions"
        )
    pairs = ambiguous_pairs(cpdag)
    observations: list[tuple[int, int]] = []
    confidences: dict[tuple[int, int], float] = {}
    expert_records: list[dict] = []
    for first, second in pairs:
        arc, probability, record = expert(first, second, names[first], names[second])
        if set(arc) != {first, second} or not 0.5 <= probability <= 1.0:
            raise ValueError("expert returned an invalid orientation or confidence")
        observations.append(arc)
        confidences[arc] = probability
        expert_records.append(
            {
                "pair": [first, second],
                "arc": list(arc),
                "confidence": probability,
                **record,
            }
        )

    weights = np.asarray(
        [_likelihood(graph, observations, confidences) for graph in initial],
        dtype=float,
    )
    normalizer = float(weights.sum())
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("expert posterior has zero or invalid normalizer")

    current = initial
    possible = list(observations)
    retained_probability = 1.0
    decisions: tuple[tuple[int, int], ...] = ()
    while len(current) > 1:
        candidates = []
        for arc in possible:
            subset = tuple(graph for graph in current if _contains(graph, arc))
            if not subset or len(subset) == len(current):
                continue
            implied = _fixed_orientations(subset, pairs)
            compatible = [
                index
                for index, graph in enumerate(initial)
                if all(_contains(graph, decision) for decision in implied)
            ]
            probability = float(weights[compatible].sum() / normalizer)
            if retained_probability * probability > 1.0 - tolerance:
                candidates.append((arc, subset, implied, probability))
        if not candidates:
            break
        if strategy == "srisk":
            chosen = max(candidates, key=lambda item: (item[3], tuple(item[0])))
        else:
            chosen = min(candidates, key=lambda item: (len(item[1]), tuple(item[0])))
        possible.remove(chosen[0])
        current = chosen[1]
        decisions = chosen[2]
        retained_probability *= chosen[3]
    return LongResult(
        initial,
        current,
        decisions,
        retained_probability,
        tuple(expert_records),
    )
