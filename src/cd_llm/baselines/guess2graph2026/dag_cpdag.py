"""Native DAG-guided gPC search with an explicit CPDAG extension.

The vendored Guess2Graph ``run_sgs_guess_dag`` routine returns an undirected
skeleton.  This module preserves its edge and conditioning-set ordering, while
also retaining the separating sets needed for the project's requested PC
collider-orientation and Meek-closure extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, UCSepset
from causallearn.utils.PCUtils.Helper import append_value

from cd_llm.baselines.causal_order2025.integration import causal_learn_cpdag
from cd_llm.core.cpc import certify_cpdag
from cd_llm.core.finite_support import PartialGraph

from .paper_methods import (
    expert_edge_order,
    neighbors,
    validate_guessed_dag,
)


@dataclass(frozen=True)
class DAGGPCCPDAGResult:
    """Complete output and audit information for one extended gPC run."""

    skeleton: np.ndarray
    separating_sets: dict[tuple[int, int], tuple[int, ...]]
    raw_cpdag: PartialGraph
    cpdag: PartialGraph
    topological_order: tuple[int, ...]
    ci_tests: int
    certification_changed: bool


def deterministic_topological_order(dag: np.ndarray) -> tuple[int, ...]:
    """Match the vendored FIFO topological sort for a validated DAG."""

    graph = validate_guessed_dag(dag)
    indegree = graph.sum(axis=0).astype(int)
    queue = [index for index, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    if len(order) != graph.shape[0]:  # validation should make this unreachable
        raise ValueError("Expert graph must be acyclic.")
    return tuple(order)


def order_conditioning_sets(
    conditioning_sets: Sequence[tuple[int, ...]],
    left: int,
    right: int,
    topological_order: Sequence[int],
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Apply the two-block ordering in upstream ``_expert_subset_ordering``."""

    positions = {node: position for position, node in enumerate(topological_order)}
    endpoint_position = max(positions[left], positions[right])
    first: list[tuple[int, ...]] = []
    second: list[tuple[int, ...]] = []
    for conditioning in conditioning_sets:
        target = second if any(
            positions[node] > endpoint_position for node in conditioning
        ) else first
        target.append(tuple(int(node) for node in conditioning))
    rng.shuffle(first)
    rng.shuffle(second)
    return first + second


def orient_skeleton_as_cpdag(
    skeleton: np.ndarray,
    separating_sets: dict[tuple[int, int], tuple[int, ...]],
    *,
    node_names: Sequence[str] | None = None,
) -> tuple[PartialGraph, PartialGraph]:
    """Orient unshielded colliders, apply Meek closure, and certify a CPDAG."""

    matrix = np.asarray(skeleton, dtype=int)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.array_equal(matrix, matrix.T)
        or np.diag(matrix).any()
        or not np.all(np.isin(matrix, (0, 1)))
    ):
        raise ValueError("Skeleton must be a symmetric binary matrix without self-loops.")
    count = matrix.shape[0]
    names = None if node_names is None else [str(name) for name in node_names]
    if names is not None and (len(names) != count or len(set(names)) != count):
        raise ValueError("Node names must be unique and match the skeleton size.")

    graph = CausalGraph(count, names)
    for left in range(count):
        for right in range(left + 1, count):
            if not matrix[left, right]:
                edge = graph.G.get_edge(graph.G.nodes[left], graph.G.nodes[right])
                if edge is not None:
                    graph.G.remove_edge(edge)
            separating = separating_sets.get((left, right))
            if separating is None:
                separating = separating_sets.get((right, left))
            if not matrix[left, right] and separating is None:
                raise ValueError(f"Missing separating set for removed edge {(left, right)}.")
            if separating is not None:
                normalized = tuple(int(node) for node in separating)
                append_value(graph.sepset, left, right, normalized)
                append_value(graph.sepset, right, left, normalized)

    oriented = UCSepset.uc_sepset(graph, priority=2)
    closed = Meek.meek(oriented)
    raw = causal_learn_cpdag(closed.G)
    certified = certify_cpdag(raw)
    return raw, certified


def run_native_gpc_guess_dag_cpdag(
    data: np.ndarray,
    guessed_dag: np.ndarray,
    *,
    alpha: float = 0.05,
    ci_test: str = "chisq",
    rng: np.random.Generator | None = None,
    node_names: Sequence[str] | None = None,
) -> DAGGPCCPDAGResult:
    """Run native DAG-gPC ordering and return its PC-oriented CPDAG extension."""

    observations = np.asarray(data)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError("Data must be a two-dimensional matrix with at least two variables.")
    dag = validate_guessed_dag(guessed_dag, observations.shape[1])
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    generator = np.random.default_rng() if rng is None else rng
    order = deterministic_topological_order(dag)
    guessed_skeleton = ((dag != 0) | (dag.T != 0)).astype(int)
    tester = CIT(observations, ci_test)
    current = np.ones(dag.shape, dtype=int) - np.eye(dag.shape[0], dtype=int)
    edges = list(combinations(range(dag.shape[0]), 2))
    separating_sets: dict[tuple[int, int], tuple[int, ...]] = {}
    tests = 0

    for left, right in expert_edge_order(edges, guessed_skeleton, generator):
        if not current[left, right]:
            continue
        left_pool = neighbors(current, left, exclude=right)
        right_pool = neighbors(current, right, exclude=left)
        pool = left_pool if len(left_pool) <= len(right_pool) else right_pool
        removed = False
        for size in range(len(pool) + 1):
            ordered = order_conditioning_sets(
                list(combinations(pool, size)), left, right, order, generator
            )
            for conditioning in ordered:
                tests += 1
                if tester(left, right, list(conditioning)) > alpha:
                    current[left, right] = current[right, left] = 0
                    separating_sets[(left, right)] = conditioning
                    separating_sets[(right, left)] = conditioning
                    removed = True
                    break
            if removed:
                break

    raw, certified = orient_skeleton_as_cpdag(
        current, separating_sets, node_names=node_names
    )
    return DAGGPCCPDAGResult(
        skeleton=current,
        separating_sets=separating_sets,
        raw_cpdag=raw,
        cpdag=certified,
        topological_order=order,
        ci_tests=tests,
        certification_changed=raw != certified,
    )


__all__ = [
    "DAGGPCCPDAGResult",
    "deterministic_topological_order",
    "order_conditioning_sets",
    "orient_skeleton_as_cpdag",
    "run_native_gpc_guess_dag_cpdag",
]
