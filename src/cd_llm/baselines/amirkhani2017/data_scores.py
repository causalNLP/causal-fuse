"""Cached decomposable data scores for the Amirkhani et al. baseline.

The paper's native observational score is BDeu for complete discrete data.
``GaussianBICLocalScore`` is the continuous-data adaptation used by this
project; it changes only the observational term of the combined score.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.core.finite_support import Graph, iter_bits


def _validate_parent_mask(node_count: int, node: int, parent_mask: int) -> int:
    if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
        raise TypeError("node must be an integer")
    if node < 0 or node >= node_count:
        raise IndexError("node is outside the score")
    if isinstance(parent_mask, bool) or not isinstance(
        parent_mask, (int, np.integer)
    ):
        raise TypeError("parent_mask must be an integer")
    mask = int(parent_mask)
    if mask < 0 or mask >= 1 << node_count:
        raise ValueError("parent_mask is outside the score")
    if mask & (1 << node):
        raise ValueError("a node cannot be its own parent")
    return mask


@runtime_checkable
class DecomposableLocalScore(Protocol):
    """Minimal higher-is-better local-score interface accepted by the search."""

    node_count: int

    def local_score(self, node: int, parent_mask: int) -> float:
        """Return the log score for one node and its parent set."""


class BDeuLocalScore(BDeuScore):
    """Baseline-compatible local-score view of the shared BDeu scorer."""

    def __init__(
        self,
        data: np.ndarray | Sequence[Sequence[object]],
        *,
        equivalent_sample_size: float = 1.0,
        cardinalities: Sequence[int] | None = None,
    ) -> None:
        values = np.asarray(data)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("BDeu data must be a nonempty two-dimensional matrix")
        super().__init__(
            data,
            equivalent_sample_size=equivalent_sample_size,
            cardinalities=cardinalities,
            max_parents=values.shape[1] - 1,
        )

    @property
    def cache_size(self) -> int:
        return len(self._local_cache)


class GaussianBICLocalScore:
    """Cached linear-Gaussian BIC log score (continuous-data adaptation)."""

    name = "gaussian-bic"

    def __init__(
        self,
        data: np.ndarray | Sequence[Sequence[float]],
        *,
        standardize_data: bool = True,
    ) -> None:
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
            raise ValueError(
                "Gaussian BIC requires a two-dimensional dataset with at least two rows"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Gaussian BIC requires finite observations")
        self.sample_size, self.node_count = values.shape
        self.standardize_data = bool(standardize_data)
        self._single_score: float | None = None
        if self.node_count == 1:
            variance = float(np.var(values[:, 0], ddof=0))
            if variance <= 0.0:
                raise ValueError("constant column in data")
            if self.standardize_data:
                variance = 1.0
            variance = max(variance, np.finfo(float).eps)
            self._single_score = -0.5 * self.sample_size * (
                1.0 + math.log(variance)
            ) - 0.5 * math.log(self.sample_size)
            self._score: GaussianBICScore | None = None
        else:
            self._score = GaussianBICScore(
                values, standardize_data=self.standardize_data
            )

    def local_score(self, node: int, parent_mask: int) -> float:
        mask = _validate_parent_mask(self.node_count, node, parent_mask)
        if self._score is None:
            assert self._single_score is not None
            result = self._single_score
        else:
            result = float(self._score.local_score(int(node), mask))
        if not math.isfinite(result):
            raise ArithmeticError("Gaussian BIC produced a non-finite local score")
        return result

    @property
    def cache_size(self) -> int:
        if self._score is None:
            return int(self._single_score is not None)
        return len(self._score._local_cache)


def parent_masks(graph: Graph) -> tuple[int, ...]:
    """Return one parent bitmask per node of a DAG."""

    if any(graph.bidirected) or any(graph.undirected):
        raise ValueError("the Amirkhani baseline accepts DAGs only")
    masks = [0] * graph.node_count
    for source, row in enumerate(graph.directed):
        for target in iter_bits(row):
            masks[target] |= 1 << source
    return tuple(masks)


def graph_data_log_score(graph: Graph, score: DecomposableLocalScore) -> float:
    """Sum a decomposable higher-is-better score over one DAG."""

    if graph.node_count != score.node_count:
        raise ValueError("graph and local score have different node counts")
    result = math.fsum(
        float(score.local_score(node, mask))
        for node, mask in enumerate(parent_masks(graph))
    )
    if not math.isfinite(result):
        raise ValueError("local score must return finite values")
    return result


__all__ = [
    "BDeuLocalScore",
    "DecomposableLocalScore",
    "GaussianBICLocalScore",
    "graph_data_log_score",
    "parent_masks",
]
