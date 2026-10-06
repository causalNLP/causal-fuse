from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import gammaln

from cd_llm.core.finite_support import Graph, iter_bits


@runtime_checkable
class DecomposableDAGScore(Protocol):
    node_count: int
    max_parents: int

    def local_score(self, node: int, parent_mask: int) -> float: ...
    def parent_masks(self, graph: Graph) -> tuple[int, ...]: ...
    def graph_score(self, graph: Graph) -> float: ...
    def metadata(self) -> dict[str, object]: ...


def parent_masks(graph: Graph) -> tuple[int, ...]:
    if any(graph.bidirected) or any(graph.undirected):
        raise ValueError("decomposable DAG scores accept DAGs only")
    masks = [0] * graph.node_count
    for source, row in enumerate(graph.directed):
        for target in iter_bits(row):
            masks[target] |= 1 << source
    return tuple(masks)


class BDeuScore:
    """Cached BDeu log marginal likelihood with an optional structure prior.

    The score is higher-is-better and decomposes by child node.  Explicit
    cardinalities require zero-based integer category codes; otherwise each
    column is factorized independently.  ``max_parents=None`` imposes no
    explicit parent cap; ``max_parents`` remains the effective integer cap for
    consumers of :class:`DecomposableDAGScore`.
    """

    name = "bdeu"
    _DENSE_COUNT_CELL_LIMIT = 65_536

    def __init__(
        self,
        data: np.ndarray | Sequence[Sequence[object]],
        *,
        equivalent_sample_size: float = 1.0,
        cardinalities: Sequence[int] | None = None,
        max_parents: int | None = None,
        structure_prior: float | None = None,
    ) -> None:
        values = _validate_dataset(data)
        self.data, self.cardinalities = _factorize_discrete(values, cardinalities)
        self.sample_size, self.node_count = self.data.shape
        self.requested_max_parents = (
            None if max_parents is None else _integer("max_parents", max_parents)
        )
        self.max_parents = (
            self.node_count - 1
            if self.requested_max_parents is None
            else self.requested_max_parents
        )
        if not 0 <= self.max_parents < self.node_count:
            raise ValueError("max_parents must lie in [0, node_count)")
        ess = float(equivalent_sample_size)
        if not math.isfinite(ess) or ess <= 0.0:
            raise ValueError("equivalent_sample_size must be positive and finite")
        self.equivalent_sample_size = ess
        if structure_prior is None:
            self.structure_prior = None
            self.edge_prior_probability = None
        else:
            value = float(structure_prior)
            possible_parents = self.node_count - 1
            if (
                possible_parents < 1
                or not math.isfinite(value)
                or not 0.0 < value < possible_parents
            ):
                raise ValueError(
                    "structure_prior must lie strictly between zero and node_count - 1"
                )
            self.structure_prior = value
            self.edge_prior_probability = value / possible_parents
        self._local_cache: dict[tuple[int, int], float] = {}
        self._graph_cache: dict[Graph, float] = {}

    def local_score(self, node: int, parent_mask: int) -> float:
        normalized_node = _integer("node", node)
        if not 0 <= normalized_node < self.node_count:
            raise IndexError("node is outside the score")
        mask = _integer("parent_mask", parent_mask)
        if not 0 <= mask < 1 << self.node_count:
            raise ValueError("parent_mask is outside the score")
        if mask & (1 << normalized_node):
            raise ValueError("a node cannot be its own parent")
        if mask.bit_count() > self.max_parents:
            raise ValueError("parent count exceeds score cap")
        key = (normalized_node, mask)
        if key in self._local_cache:
            return self._local_cache[key]
        parents = tuple(iter_bits(mask))
        q = math.prod(self.cardinalities[parent] for parent in parents)
        r = self.cardinalities[normalized_node]
        alpha_parent = self.equivalent_sample_size / q
        alpha_joint = alpha_parent / r
        if q * r <= self._DENSE_COUNT_CELL_LIMIT:
            parent_counts, joint_counts = self._dense_counts(
                normalized_node, parents, q, r
            )
        else:
            parent_counts, joint_counts = self._sparse_observed_counts(
                normalized_node, parents
            )
        result = float(
            np.sum(gammaln(alpha_parent) - gammaln(alpha_parent + parent_counts))
            + np.sum(gammaln(alpha_joint + joint_counts) - gammaln(alpha_joint))
        )
        if self.edge_prior_probability is not None:
            parent_count = mask.bit_count()
            possible_parents = self.node_count - 1
            result += parent_count * math.log(self.edge_prior_probability)
            result += (possible_parents - parent_count) * math.log1p(
                -self.edge_prior_probability
            )
        if not math.isfinite(result):
            raise ArithmeticError("BDeu produced a non-finite local score")
        self._local_cache[key] = 0.0 if result == 0.0 else result
        return self._local_cache[key]

    def _dense_counts(
        self, node: int, parents: tuple[int, ...], q: int, r: int
    ) -> tuple[np.ndarray, np.ndarray]:
        parent_codes = np.zeros(self.sample_size, dtype=np.int64)
        multiplier = 1
        for parent in parents:
            parent_codes += self.data[:, parent] * multiplier
            multiplier *= self.cardinalities[parent]
        counts = np.bincount(
            parent_codes * r + self.data[:, node], minlength=q * r
        ).reshape(q, r)
        return counts.sum(axis=1), counts.reshape(-1)

    def _sparse_observed_counts(
        self, node: int, parents: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        if parents:
            _, parent_inverse = np.unique(
                self.data[:, parents], axis=0, return_inverse=True
            )
        else:
            parent_inverse = np.zeros(self.sample_size, dtype=np.int64)
        parent_counts = np.bincount(parent_inverse)
        observed_joint_rows = np.column_stack(
            (parent_inverse, self.data[:, node])
        )
        _, joint_counts = np.unique(
            observed_joint_rows, axis=0, return_counts=True
        )
        return parent_counts, joint_counts

    def parent_masks(self, graph: Graph) -> tuple[int, ...]:
        masks = parent_masks(graph)
        if any(mask.bit_count() > self.max_parents for mask in masks):
            raise ValueError("graph exceeds score parent cap")
        return masks

    def graph_score(self, graph: Graph) -> float:
        if graph in self._graph_cache:
            return self._graph_cache[graph]
        masks = self.parent_masks(graph)
        value = math.fsum(self.local_score(node, mask) for node, mask in enumerate(masks))
        self._graph_cache[graph] = value
        return value

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "equivalent_sample_size": self.equivalent_sample_size,
            "cardinalities": self.cardinalities,
            "max_parents": self.max_parents,
            "requested_max_parents": self.requested_max_parents,
            "effective_max_parents": self.max_parents,
            "parent_configuration_counting": "adaptive-dense-or-observed-sparse",
            "dense_count_cell_limit": self._DENSE_COUNT_CELL_LIMIT,
            "graph_prior": (
                "uniform"
                if self.structure_prior is None
                else "causal-learn-independent-parent-indicators"
            ),
            "structure_prior": self.structure_prior,
            "edge_prior_probability": self.edge_prior_probability,
        }


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_dataset(
    data: np.ndarray | Sequence[Sequence[object]],
) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("BDeu data must be a nonempty two-dimensional matrix")
    return values


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = value != value
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _factorize_discrete(
    values: np.ndarray, cardinalities: Sequence[int] | None
) -> tuple[np.ndarray, tuple[int, ...]]:
    sample_size, node_count = values.shape
    if any(_is_missing(value) for value in values.flat):
        raise ValueError("BDeu scoring does not support missing observations")

    encoded = np.empty((sample_size, node_count), dtype=np.int64)
    if cardinalities is not None:
        if len(cardinalities) != node_count:
            raise ValueError("cardinalities must contain one entry per node")
        normalized = tuple(_integer("cardinality", value) for value in cardinalities)
        if any(value <= 0 for value in normalized):
            raise ValueError("cardinalities must be positive")
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "explicit cardinalities require zero-based integer category codes"
            ) from error
        if not np.all(np.isfinite(numeric)) or not np.all(
            numeric == np.floor(numeric)
        ):
            raise ValueError(
                "explicit cardinalities require zero-based integer category codes"
            )
        encoded[:, :] = numeric.astype(np.int64)
        for node, cardinality in enumerate(normalized):
            if np.any(encoded[:, node] < 0) or np.any(
                encoded[:, node] >= cardinality
            ):
                raise ValueError(
                    f"node {node} has a value outside [0, {cardinality})"
                )
        return encoded, normalized

    inferred: list[int] = []
    for node in range(node_count):
        try:
            unique, inverse = np.unique(values[:, node], return_inverse=True)
        except TypeError as error:
            raise ValueError(
                "categories within a column must have mutually comparable values"
            ) from error
        encoded[:, node] = inverse
        inferred.append(len(unique))
    return encoded, tuple(inferred)


__all__ = ["BDeuScore", "DecomposableDAGScore", "parent_masks"]
