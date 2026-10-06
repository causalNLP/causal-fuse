from __future__ import annotations

import math
import unittest

import numpy as np
from causallearn.score.LocalScoreFunction import local_score_BDeu

from cd_llm.baselines.amirkhani2017 import BDeuLocalScore
from cd_llm.core.finite_support import graph_from_edges
from cd_llm.methods.graph_scores import BDeuScore


def _dense_bdeu_reference(
    data: np.ndarray,
    node: int,
    parents: list[int],
    *,
    cardinalities: tuple[int, ...],
    equivalent_sample_size: float,
    structure_prior: float | None,
) -> float:
    q = math.prod(cardinalities[parent] for parent in parents)
    r = cardinalities[node]
    parent_codes = np.zeros(data.shape[0], dtype=np.int64)
    multiplier = 1
    for parent in parents:
        parent_codes += data[:, parent] * multiplier
        multiplier *= cardinalities[parent]
    counts = np.bincount(
        parent_codes * r + data[:, node], minlength=q * r
    ).reshape(q, r)
    alpha_parent = equivalent_sample_size / q
    alpha_joint = alpha_parent / r
    result = 0.0
    for row in counts:
        result += math.lgamma(alpha_parent) - math.lgamma(
            alpha_parent + int(row.sum())
        )
        for count in row:
            result += math.lgamma(alpha_joint + int(count)) - math.lgamma(
                alpha_joint
            )
    if structure_prior is not None:
        possible_parents = data.shape[1] - 1
        edge_probability = structure_prior / possible_parents
        result += len(parents) * math.log(edge_probability)
        result += (possible_parents - len(parents)) * math.log1p(
            -edge_probability
        )
    return result


class BDeuScoreTests(unittest.TestCase):
    def test_empty_parent_score_matches_dirichlet_multinomial(self) -> None:
        data = np.asarray([[0], [0], [1], [1], [1]], dtype=int)
        score = BDeuScore(data, equivalent_sample_size=1.0, max_parents=0)
        expected = (
            math.lgamma(1.0)
            - math.lgamma(6.0)
            + math.lgamma(2.5)
            - math.lgamma(0.5)
            + math.lgamma(3.5)
            - math.lgamma(0.5)
        )
        self.assertAlmostEqual(score.local_score(0, 0), expected)

    def test_shared_score_matches_baseline_local_score(self) -> None:
        data = np.asarray(
            [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0]],
            dtype=int,
        )
        shared = BDeuScore(data, equivalent_sample_size=2.5, max_parents=2)
        baseline = BDeuLocalScore(data, equivalent_sample_size=2.5)
        for node in range(3):
            for parent_mask in range(1 << 3):
                if parent_mask & (1 << node):
                    continue
                self.assertAlmostEqual(
                    shared.local_score(node, parent_mask),
                    baseline.local_score(node, parent_mask),
                    places=13,
                )

    def test_parent_cap_is_enforced_for_local_and_graph_scores(self) -> None:
        data = np.asarray(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int
        )
        score = BDeuScore(data, max_parents=1)
        with self.assertRaisesRegex(ValueError, "parent count"):
            score.local_score(2, (1 << 0) | (1 << 1))
        graph = graph_from_edges(3, [(0, 2), (1, 2)])
        with self.assertRaisesRegex(ValueError, "parent cap"):
            score.graph_score(graph)

    def test_none_parent_cap_is_uncapped_and_reported_as_effective_cap(self) -> None:
        data = np.asarray(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int
        )
        score = BDeuScore(data, max_parents=None)
        self.assertIsNone(score.requested_max_parents)
        self.assertEqual(score.max_parents, 2)
        score.local_score(2, (1 << 0) | (1 << 1))
        metadata = score.metadata()
        self.assertIsNone(metadata["requested_max_parents"])
        self.assertEqual(metadata["effective_max_parents"], 2)
        self.assertEqual(metadata["max_parents"], 2)

    def test_sparse_score_matches_dense_and_causal_learn_references(self) -> None:
        rng = np.random.default_rng(1729)
        data = rng.integers(0, 3, size=(96, 11), dtype=np.int64)
        data[:3, :] = np.arange(3)[:, None]
        cardinalities = (3,) * data.shape[1]
        node = 10
        parents = list(range(10))
        score = BDeuScore(
            data,
            equivalent_sample_size=2.5,
            cardinalities=cardinalities,
            max_parents=None,
            structure_prior=1.0,
        )
        self.assertGreater(
            math.prod(cardinalities[parent] for parent in parents)
            * cardinalities[node],
            score.metadata()["dense_count_cell_limit"],
        )
        actual = score.local_score(node, sum(1 << parent for parent in parents))
        dense = _dense_bdeu_reference(
            data,
            node,
            parents,
            cardinalities=cardinalities,
            equivalent_sample_size=2.5,
            structure_prior=1.0,
        )
        causal_learn = local_score_BDeu(
            data,
            node,
            parents,
            {
                "sample_prior": 2.5,
                "structure_prior": 1.0,
                "r_i_map": dict(enumerate(cardinalities)),
            },
        )
        self.assertAlmostEqual(actual, dense, places=11)
        self.assertAlmostEqual(actual, causal_learn, places=11)

    def test_uncapped_large_parent_set_uses_observed_sparse_counts(self) -> None:
        rng = np.random.default_rng(2026)
        data = rng.integers(0, 2, size=(24, 71), dtype=np.int64)
        node = 70
        parents = list(range(70))
        score = BDeuScore(
            data,
            equivalent_sample_size=1.0,
            max_parents=None,
            structure_prior=1.0,
        )
        actual = score.local_score(node, (1 << node) - 1)
        causal_learn = local_score_BDeu(data, node, parents)
        self.assertTrue(math.isfinite(actual))
        self.assertAlmostEqual(actual, causal_learn, places=11)
        self.assertEqual(score.max_parents, 70)
        self.assertEqual(
            score.metadata()["parent_configuration_counting"],
            "adaptive-dense-or-observed-sparse",
        )

    def test_structure_prior_matches_causal_learn_bdeu(self) -> None:
        data = np.asarray(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 1],
            ],
            dtype=int,
        )
        score = BDeuScore(
            data,
            equivalent_sample_size=1.0,
            max_parents=3,
            structure_prior=1.0,
        )
        for node in range(data.shape[1]):
            for parent_mask in range(1 << data.shape[1]):
                if parent_mask & (1 << node):
                    continue
                parents = [
                    parent
                    for parent in range(data.shape[1])
                    if parent_mask & (1 << parent)
                ]
                self.assertAlmostEqual(
                    score.local_score(node, parent_mask),
                    local_score_BDeu(data, node, parents),
                    places=12,
                )

    def test_explicit_codes_and_metadata_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            BDeuScore([[0, 1], [1, np.nan]])
        with self.assertRaisesRegex(ValueError, "outside"):
            BDeuScore([[0, 2], [1, 0]], cardinalities=(2, 2))
        score = BDeuScore([[0, 1], [1, 0]], max_parents=1)
        self.assertEqual(score.metadata()["graph_prior"], "uniform")
        self.assertEqual(score.metadata()["max_parents"], 1)
        self.assertEqual(score.metadata()["requested_max_parents"], 1)
        self.assertEqual(score.metadata()["effective_max_parents"], 1)
        with self.assertRaisesRegex(TypeError, "max_parents"):
            BDeuScore([[0, 1], [1, 0]], max_parents=True)
        with self.assertRaisesRegex(ValueError, "max_parents"):
            BDeuScore([[0, 1], [1, 0]], max_parents=2)
        with self.assertRaisesRegex(ValueError, "structure_prior"):
            BDeuScore([[0, 1], [1, 0]], structure_prior=1.0)


if __name__ == "__main__":
    unittest.main()
