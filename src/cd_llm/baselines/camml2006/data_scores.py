"""Decomposable data scores used by the CaMML structural-prior adaptation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln

from cd_llm.baselines.amirkhani2017.data_scores import (
    BDeuLocalScore,
    DecomposableLocalScore,
    GaussianBICLocalScore,
    graph_data_log_score,
    parent_masks,
)
from cd_llm.core.finite_support import iter_bits


def _factorize(
    data: np.ndarray | Sequence[Sequence[object]],
    cardinalities: Sequence[int] | None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("adaptive MML requires a nonempty two-dimensional dataset")
    encoded = np.empty(values.shape, dtype=np.int64)
    inferred: list[int] = []
    for node in range(values.shape[1]):
        _, inverse = np.unique(values[:, node], return_inverse=True)
        encoded[:, node] = inverse
        inferred.append(int(np.max(inverse)) + 1)
    if cardinalities is None:
        cards = tuple(inferred)
    else:
        cards = tuple(int(value) for value in cardinalities)
        if len(cards) != values.shape[1] or any(value < 1 for value in cards):
            raise ValueError("cardinalities must contain one positive value per node")
        if any(actual > supplied for actual, supplied in zip(inferred, cards)):
            raise ValueError("observed category exceeds supplied cardinality")
    return encoded, cards


class AdaptiveMMLLocalScore:
    """CaMML's complete-discrete adaptive CPT message length as a log score."""

    name = "adaptive-mml"

    def __init__(
        self,
        data: np.ndarray | Sequence[Sequence[object]],
        *,
        cardinalities: Sequence[int] | None = None,
    ) -> None:
        self.data, self.cardinalities = _factorize(data, cardinalities)
        self.sample_size, self.node_count = self.data.shape
        self._cache: dict[tuple[int, int], float] = {}

    def local_score(self, node: int, parent_mask: int) -> float:
        if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
            raise TypeError("node must be an integer")
        node = int(node)
        mask = int(parent_mask)
        if not 0 <= node < self.node_count or mask < 0 or mask >= 1 << self.node_count:
            raise ValueError("invalid node or parent mask")
        if mask & (1 << node):
            raise ValueError("a node cannot parent itself")
        key = node, mask
        if key in self._cache:
            return self._cache[key]
        parents = tuple(iter_bits(mask))
        child_cardinality = self.cardinalities[node]
        parent_configurations = math.prod(self.cardinalities[parent] for parent in parents)
        parent_codes = np.zeros(self.sample_size, dtype=np.int64)
        multiplier = 1
        for parent in parents:
            parent_codes += self.data[:, parent] * multiplier
            multiplier *= self.cardinalities[parent]
        joint = parent_codes * child_cardinality + self.data[:, node]
        counts = np.bincount(
            joint, minlength=parent_configurations * child_cardinality
        ).reshape(parent_configurations, child_cardinality)
        totals = np.sum(counts, axis=1)
        costs = (
            gammaln(totals + child_cardinality)
            - gammaln(child_cardinality)
            - np.sum(gammaln(counts + 1), axis=1)
            + (child_cardinality - 1) * 0.17649
        )
        result = -float(np.sum(costs))
        if not math.isfinite(result):
            raise ArithmeticError("adaptive MML produced a non-finite local score")
        self._cache[key] = result
        return result

    @property
    def cache_size(self) -> int:
        return len(self._cache)


__all__ = [
    "AdaptiveMMLLocalScore",
    "BDeuLocalScore",
    "DecomposableLocalScore",
    "GaussianBICLocalScore",
    "graph_data_log_score",
    "parent_masks",
]
