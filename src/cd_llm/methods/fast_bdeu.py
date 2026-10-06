"""Fast, cached BDeu local scores for causal-learn GES.

The reference :func:`causallearn.score.LocalScoreFunction.local_score_BDeu`
constructs and groups a pandas data frame for every previously unseen local
score.  GES requests many such scores.  This module factorizes each discrete
column once and computes the same sufficient statistics with NumPy.

``patched_causal_learn_bdeu`` is deliberately narrow: it replaces only the
module global read by causal-learn's GES implementation, holds a process-wide
re-entrant lock for the complete context, and always restores the previous
callable.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np


_GES_BDEU_PATCH_LOCK = threading.RLock()


class FastBDeuLocalScore:
    """Precomputed causal-learn-compatible BDeu local scorer.

    Category labels are factorized independently by column, exactly as in
    causal-learn's ``r_i_map`` construction.  Only observed parent
    configurations need to be accumulated: every unobserved configuration
    contributes exactly zero to the BDeu log marginal likelihood.
    """

    def __init__(
        self,
        data: np.ndarray | Sequence[Sequence[object]],
        *,
        equivalent_sample_size: float = 1.0,
        structure_prior: float = 1.0,
    ) -> None:
        precomputation_started = time.perf_counter()
        values = np.asarray(data)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
            raise ValueError(
                "BDeu data must have at least one row and two columns"
            )
        if any(_is_missing(value) for value in values.flat):
            raise ValueError("BDeu scoring does not support missing observations")

        ess = float(equivalent_sample_size)
        if not math.isfinite(ess) or ess <= 0.0:
            raise ValueError("equivalent_sample_size must be positive and finite")
        prior = float(structure_prior)
        possible_parents = values.shape[1] - 1
        if (
            not math.isfinite(prior)
            or prior <= 0.0
            or prior >= possible_parents
        ):
            raise ValueError(
                "structure_prior must lie strictly between zero and node_count - 1"
            )

        encoded = np.empty(values.shape, dtype=np.int64)
        cardinalities: list[int] = []
        for node in range(values.shape[1]):
            try:
                unique, inverse = np.unique(values[:, node], return_inverse=True)
            except TypeError as error:
                raise ValueError(
                    "categories within a column must have mutually comparable values"
                ) from error
            encoded[:, node] = inverse
            cardinalities.append(int(unique.size))

        self._source_data = data
        self.data = encoded
        self.sample_size, self.node_count = encoded.shape
        self.cardinalities = tuple(cardinalities)
        self.equivalent_sample_size = ess
        self.structure_prior = prior
        self.edge_prior_probability = prior / possible_parents
        self._cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.evaluation_seconds = 0.0
        self.uncached_evaluation_seconds = 0.0
        self.precomputation_seconds = time.perf_counter() - precomputation_started

    def local_score(self, child: int, parents: Sequence[int]) -> float:
        """Return the BDeu local score for ``child`` and ``parents``."""

        evaluation_started = time.perf_counter()
        try:
            normalized_child = _node_index("child", child, self.node_count)
            normalized_parents = tuple(
                sorted(
                    _node_index("parent", parent, self.node_count)
                    for parent in parents
                )
            )
            if len(set(normalized_parents)) != len(normalized_parents):
                raise ValueError("parents must not contain duplicates")
            if normalized_child in normalized_parents:
                raise ValueError("a node cannot be its own parent")

            key = (normalized_child, normalized_parents)
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached

            self.cache_misses += 1
            uncached_started = time.perf_counter()
            try:
                score = self._uncached_local_score(
                    normalized_child, normalized_parents
                )
            finally:
                self.uncached_evaluation_seconds += (
                    time.perf_counter() - uncached_started
                )
            self._cache[key] = score
            return score
        finally:
            self.evaluation_seconds += time.perf_counter() - evaluation_started

    def causal_learn_adapter(
        self,
        runtime_data: Any,
        child: int,
        parents: Sequence[int],
        parameters: Any = None,
    ) -> float:
        """Adapt ``local_score`` to causal-learn's local-score signature."""

        if runtime_data is not self._source_data:
            raise ValueError(
                "the patched BDeu scorer received a different data object"
            )
        if parameters is not None:
            raise ValueError(
                "the patched BDeu scorer uses its construction-time parameters"
            )
        return self.local_score(child, parents)

    def cache_info(self) -> dict[str, int | float]:
        """Return cache and exact accumulated scorer-time diagnostics."""

        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "precomputation_seconds": self.precomputation_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "uncached_evaluation_seconds": self.uncached_evaluation_seconds,
        }

    def _uncached_local_score(
        self, child: int, parents: tuple[int, ...]
    ) -> float:
        parent_cardinality = math.prod(
            self.cardinalities[parent] for parent in parents
        )
        child_cardinality = self.cardinalities[child]

        if parents:
            # A dense mixed-radix code is substantially faster for the small
            # parent sets GES normally visits.  Fall back to observed rows if a
            # general discrete parent product would allocate a large table.
            joint_cell_count = parent_cardinality * child_cardinality
            if joint_cell_count <= max(65_536, 16 * self.sample_size):
                parent_codes = np.zeros(self.sample_size, dtype=np.int64)
                for parent in parents:
                    parent_codes *= self.cardinalities[parent]
                    parent_codes += self.data[:, parent]
                observed_parent_count = parent_cardinality
            else:
                _, parent_codes = np.unique(
                    self.data[:, parents], axis=0, return_inverse=True
                )
                observed_parent_count = int(parent_codes.max()) + 1
        else:
            parent_codes = np.zeros(self.sample_size, dtype=np.int64)
            observed_parent_count = 1

        counts = np.bincount(
            parent_codes * child_cardinality + self.data[:, child],
            minlength=observed_parent_count * child_cardinality,
        ).reshape(observed_parent_count, child_cardinality)
        alpha_parent = self.equivalent_sample_size / parent_cardinality
        alpha_joint = alpha_parent / child_cardinality
        parent_count = len(parents)
        possible_parents = self.node_count - 1
        score = parent_count * np.log(self.edge_prior_probability)
        score += (possible_parents - parent_count) * np.log(
            1.0 - self.edge_prior_probability
        )

        # Keep causal-learn's accumulation order.  Exact score equivalences
        # arise frequently in GES; changing only floating-point summation order
        # can otherwise choose a different (though equivalent) operator trace.
        baseline_joint = math.lgamma(alpha_joint)
        for row in counts:
            parent_observations = int(row.sum())
            if parent_observations == 0:
                continue
            first_term = math.lgamma(alpha_parent) - math.lgamma(
                parent_observations + alpha_parent
            )
            second_term = 0.0
            nonzero_counts = row[row > 0]
            if parents:
                # pandas.Series.value_counts, used by causal-learn for a
                # nonempty parent set, orders values by decreasing frequency.
                nonzero_counts = np.sort(nonzero_counts)[::-1]
            for count in nonzero_counts:
                second_term += math.lgamma(int(count) + alpha_joint) - baseline_joint
            score += first_term + second_term
        return float(score)


@contextmanager
def patched_causal_learn_bdeu(
    data: np.ndarray,
    *,
    equivalent_sample_size: float = 1.0,
    structure_prior: float = 1.0,
) -> Iterator[FastBDeuLocalScore]:
    """Temporarily install a fast BDeu scorer for causal-learn GES.

    The passed NumPy array must also be passed unchanged to ``ges`` inside the
    context.  The global replacement is serialized because causal-learn does
    not expose local-score injection through its public GES API.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("the causal-learn adapter requires a NumPy array")
    scorer = FastBDeuLocalScore(
        data,
        equivalent_sample_size=equivalent_sample_size,
        structure_prior=structure_prior,
    )
    ges_module = importlib.import_module("causallearn.search.ScoreBased.GES")

    with _GES_BDEU_PATCH_LOCK:
        previous = ges_module.local_score_BDeu
        try:
            ges_module.local_score_BDeu = scorer.causal_learn_adapter
            yield scorer
        finally:
            ges_module.local_score_BDeu = previous


def _node_index(name: str, value: object, node_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if not 0 <= normalized < node_count:
        raise IndexError(f"{name} is outside the score")
    return normalized


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = value != value
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


__all__ = ["FastBDeuLocalScore", "patched_causal_learn_bdeu"]
