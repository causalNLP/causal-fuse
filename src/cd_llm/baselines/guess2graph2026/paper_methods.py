from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from causallearn.utils.cit import CIT


class DAGDSeparationOracle:
    """Exact d-separation for a small integer-indexed DAG.

    The implementation uses the ancestral moral graph criterion and bitsets so
    exhaustive gPC ordering does not accumulate an unbounded query cache.
    """

    def __init__(self, dag: np.ndarray) -> None:
        self.dag = validate_guessed_dag(dag)
        self.node_count = self.dag.shape[0]
        self.parents = [sum(1 << int(parent) for parent in np.flatnonzero(self.dag[:, child]))
                        for child in range(self.node_count)]
        self.ancestors = []
        for node in range(self.node_count):
            reached = 0
            frontier = self.parents[node]
            while frontier:
                reached |= frontier
                expanded = 0
                pending = frontier
                while pending:
                    bit = pending & -pending
                    pending ^= bit
                    expanded |= self.parents[bit.bit_length() - 1]
                frontier = expanded & ~reached
            self.ancestors.append(reached)

    def __call__(self, left: int, right: int, conditioned: Sequence[int]) -> bool:
        blocked = sum(1 << int(node) for node in conditioned)
        if blocked & ((1 << left) | (1 << right)):
            return True
        relevant = (1 << left) | (1 << right) | blocked
        pending = relevant
        while pending:
            bit = pending & -pending
            pending ^= bit
            relevant |= self.ancestors[bit.bit_length() - 1]
        neighbors = [0] * self.node_count
        children = relevant
        while children:
            child_bit = children & -children
            children ^= child_bit
            child = child_bit.bit_length() - 1
            parents = self.parents[child] & relevant
            parent_bits = parents
            while parent_bits:
                parent_bit = parent_bits & -parent_bits
                parent_bits ^= parent_bit
                parent = parent_bit.bit_length() - 1
                neighbors[parent] |= child_bit | (parents ^ parent_bit)
                neighbors[child] |= parent_bit
        reached = 1 << left
        frontier = reached
        allowed = relevant & ~blocked
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            node = bit.bit_length() - 1
            unseen = neighbors[node] & allowed & ~reached
            if unseen & (1 << right):
                return False
            reached |= unseen
            frontier |= unseen
        return not bool(reached & (1 << right))


def run_pc_guess_skeleton(
    data: np.ndarray,
    guessed_skeleton: np.ndarray,
    *,
    alpha: float = 0.05,
    ci_test: str = "chisq",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run PC-Guess with expert edge ordering and random subset ordering."""
    node_count = data.shape[1]
    guess = validate_guessed_skeleton(guessed_skeleton, node_count)
    generator = np.random.default_rng() if rng is None else rng
    cit = CIT(data, ci_test)
    current = np.ones((node_count, node_count), dtype=int) - np.eye(
        node_count, dtype=int
    )

    level = 0
    while True:
        testable = [
            (left, right)
            for left in range(node_count)
            for right in range(left + 1, node_count)
            if current[left, right]
            and (
                neighbor_count(current, left, right) >= level
                or neighbor_count(current, right, left) >= level
            )
        ]
        if not testable:
            break

        for left, right in expert_edge_order(testable, guess, generator):
            if not current[left, right]:
                continue
            pool = smaller_valid_neighbor_pool(current, left, right, level)
            if pool is None:
                continue
            conditioning_sets = list(combinations(pool, level))
            generator.shuffle(conditioning_sets)
            found_independence = any(
                cit(left, right, list(conditioning)) > alpha
                for conditioning in conditioning_sets
            )
            if found_independence:
                current[left, right] = current[right, left] = 0
        level += 1
    return current


def run_gpc_guess_skeleton(
    data: np.ndarray,
    guessed_skeleton: np.ndarray,
    *,
    alpha: float = 0.05,
    ci_test: str = "chisq",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run gPC-Guess with one expert-ordered edge pass and random subset ordering."""
    node_count = data.shape[1]
    guess = validate_guessed_skeleton(guessed_skeleton, node_count)
    generator = np.random.default_rng() if rng is None else rng
    cit = CIT(data, ci_test)
    current = np.ones((node_count, node_count), dtype=int) - np.eye(
        node_count, dtype=int
    )
    all_edges = [
        (left, right)
        for left in range(node_count)
        for right in range(left + 1, node_count)
    ]

    for left, right in expert_edge_order(all_edges, guess, generator):
        if not current[left, right]:
            continue
        left_pool = neighbors(current, left, exclude=right)
        right_pool = neighbors(current, right, exclude=left)
        pool = left_pool if len(left_pool) <= len(right_pool) else right_pool
        removed = False
        for size in range(len(pool) + 1):
            conditioning_sets = list(combinations(pool, size))
            generator.shuffle(conditioning_sets)
            for conditioning in conditioning_sets:
                if cit(left, right, list(conditioning)) > alpha:
                    current[left, right] = current[right, left] = 0
                    removed = True
                    break
            if removed:
                break
    return current


def validate_guessed_skeleton(
    guessed_skeleton: np.ndarray,
    node_count: int,
) -> np.ndarray:
    matrix = np.asarray(guessed_skeleton)
    if matrix.shape != (node_count, node_count):
        raise ValueError(
            f"Expert skeleton must have shape {(node_count, node_count)}; got {matrix.shape}."
        )
    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError("Expert skeleton must contain only 0 and 1.")
    skeleton = ((matrix != 0) | (matrix.T != 0)).astype(int)
    np.fill_diagonal(skeleton, 0)
    return skeleton


def validate_guessed_dag(guessed_dag: np.ndarray, node_count: int | None = None) -> np.ndarray:
    matrix = np.asarray(guessed_dag)
    expected = matrix.shape[0] if node_count is None and matrix.ndim == 2 else node_count
    if matrix.ndim != 2 or expected is None or matrix.shape != (expected, expected):
        raise ValueError(f"Expert DAG must be square with {expected} nodes; got {matrix.shape}.")
    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError("Expert DAG must contain only 0 and 1.")
    dag = matrix.astype(int, copy=True)
    if np.diag(dag).any():
        raise ValueError("Expert DAG must not contain self-loops.")
    indegree = dag.sum(axis=0).astype(int)
    queue = [index for index, value in enumerate(indegree) if value == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(dag[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    if visited != expected:
        raise ValueError("Expert graph must be acyclic.")
    return dag


def run_gpc_guess_dag(
    data: np.ndarray,
    guessed_dag: np.ndarray,
    *,
    alpha: float = 0.05,
    ci_test: str = "chisq",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run gPC-Guess with DAG-derived edge and d-separation ordering."""
    node_count = data.shape[1]
    dag = validate_guessed_dag(guessed_dag, node_count)
    skeleton = ((dag != 0) | (dag.T != 0)).astype(int)
    generator = np.random.default_rng() if rng is None else rng
    oracle = DAGDSeparationOracle(dag)
    cit = CIT(data, ci_test)
    current = np.ones((node_count, node_count), dtype=int) - np.eye(node_count, dtype=int)
    all_edges = list(combinations(range(node_count), 2))
    for left, right in expert_edge_order(all_edges, skeleton, generator):
        if not current[left, right]:
            continue
        left_pool = neighbors(current, left, exclude=right)
        right_pool = neighbors(current, right, exclude=left)
        pool = left_pool if len(left_pool) <= len(right_pool) else right_pool
        removed = False
        for size in range(len(pool) + 1):
            candidates = list(combinations(pool, size))
            first: list[tuple[int, ...]] = []
            other: list[tuple[int, ...]] = []
            for subset in candidates:
                (first if oracle(left, right, subset) else other).append(subset)
            generator.shuffle(first)
            generator.shuffle(other)
            for conditioning in first + other:
                if cit(left, right, list(conditioning)) > alpha:
                    current[left, right] = current[right, left] = 0
                    removed = True
                    break
            if removed:
                break
    return current


def expert_edge_order(
    edges: Sequence[tuple[int, int]],
    guessed_skeleton: np.ndarray,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    generator = np.random.default_rng() if rng is None else rng
    predicted_non_edges = [
        edge for edge in edges if not guessed_skeleton[edge[0], edge[1]]
    ]
    predicted_edges = [
        edge for edge in edges if guessed_skeleton[edge[0], edge[1]]
    ]
    generator.shuffle(predicted_non_edges)
    generator.shuffle(predicted_edges)
    return predicted_non_edges + predicted_edges


def neighbors(graph: np.ndarray, node: int, *, exclude: int) -> list[int]:
    return [
        candidate
        for candidate in range(graph.shape[0])
        if candidate != exclude and graph[node, candidate]
    ]


def neighbor_count(graph: np.ndarray, node: int, exclude: int) -> int:
    return len(neighbors(graph, node, exclude=exclude))


def smaller_valid_neighbor_pool(
    graph: np.ndarray,
    left: int,
    right: int,
    level: int,
) -> list[int] | None:
    candidates = [
        pool
        for pool in (
            neighbors(graph, left, exclude=right),
            neighbors(graph, right, exclude=left),
        )
        if len(pool) >= level
    ]
    return min(candidates, key=len) if candidates else None
