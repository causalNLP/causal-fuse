from __future__ import annotations

import importlib
import unittest

import numpy as np
from causallearn.score.LocalScoreFunction import local_score_BDeu

from cd_llm.methods.fast_bdeu import (
    FastBDeuLocalScore,
    patched_causal_learn_bdeu,
)


class FastBDeuLocalScoreTests(unittest.TestCase):
    def test_matches_causal_learn_for_binary_parent_sets(self) -> None:
        data = np.asarray(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 1],
            ],
            dtype=np.int64,
        )
        scorer = FastBDeuLocalScore(data)

        for child in range(data.shape[1]):
            other_nodes = [node for node in range(data.shape[1]) if node != child]
            for mask in range(1 << len(other_nodes)):
                parents = [
                    node
                    for offset, node in enumerate(other_nodes)
                    if mask & (1 << offset)
                ]
                self.assertAlmostEqual(
                    scorer.local_score(child, parents),
                    local_score_BDeu(data, child, parents),
                    places=12,
                )

    def test_matches_general_discrete_and_constant_columns(self) -> None:
        data = np.asarray(
            [
                ["a", 0, "fixed", 10],
                ["b", 1, "fixed", 20],
                ["c", 1, "fixed", 30],
                ["a", 2, "fixed", 10],
                ["b", 2, "fixed", 30],
                ["c", 0, "fixed", 20],
                ["a", 1, "fixed", 30],
            ],
            dtype=object,
        )
        scorer = FastBDeuLocalScore(data)

        cases = (
            (0, []),
            (0, [1, 2]),
            (1, [0, 3]),
            (2, []),
            (2, [0, 1, 3]),
            (3, [2]),
        )
        for child, parents in cases:
            self.assertAlmostEqual(
                scorer.local_score(child, parents),
                local_score_BDeu(data, child, parents),
                places=12,
            )

    def test_cache_normalizes_parent_order(self) -> None:
        data = np.asarray(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64
        )
        scorer = FastBDeuLocalScore(data)

        first = scorer.local_score(2, [1, 0])
        second = scorer.local_score(2, [0, 1])

        self.assertEqual(first, second)
        diagnostics = scorer.cache_info()
        self.assertEqual(diagnostics["size"], 1)
        self.assertEqual(diagnostics["hits"], 1)
        self.assertEqual(diagnostics["misses"], 1)
        self.assertGreater(diagnostics["precomputation_seconds"], 0.0)
        self.assertGreater(diagnostics["evaluation_seconds"], 0.0)
        self.assertGreater(diagnostics["uncached_evaluation_seconds"], 0.0)
        self.assertGreaterEqual(
            diagnostics["evaluation_seconds"],
            diagnostics["uncached_evaluation_seconds"],
        )

    def test_sparse_count_path_matches_causal_learn(self) -> None:
        rng = np.random.default_rng(17)
        data = rng.integers(0, 20, size=(80, 5), dtype=np.int64)
        scorer = FastBDeuLocalScore(data)

        expected = local_score_BDeu(data, 4, [0, 1, 2, 3])
        observed = scorer.local_score(4, [0, 1, 2, 3])

        self.assertAlmostEqual(observed, expected, places=11)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            FastBDeuLocalScore([[0, 1, 0], [1, np.nan, 1]])
        scorer = FastBDeuLocalScore(np.zeros((2, 3), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            scorer.local_score(0, [1, 1])
        with self.assertRaisesRegex(ValueError, "own parent"):
            scorer.local_score(0, [0])


class PatchedCausalLearnBDeuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ges_module = importlib.import_module(
            "causallearn.search.ScoreBased.GES"
        )
        self.data = np.asarray(
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.int64,
        )

    def test_context_installs_adapter_and_restores_reference(self) -> None:
        original = self.ges_module.local_score_BDeu

        with patched_causal_learn_bdeu(self.data) as scorer:
            installed = self.ges_module.local_score_BDeu
            self.assertIsNot(installed, original)
            self.assertAlmostEqual(
                installed(self.data, 2, [0, 1]),
                local_score_BDeu(self.data, 2, [0, 1]),
                places=12,
            )
            self.assertEqual(scorer.cache_info()["misses"], 1)

        self.assertIs(self.ges_module.local_score_BDeu, original)

    def test_context_restores_reference_after_exception(self) -> None:
        original = self.ges_module.local_score_BDeu

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with patched_causal_learn_bdeu(self.data):
                raise RuntimeError("sentinel")

        self.assertIs(self.ges_module.local_score_BDeu, original)

    def test_adapter_rejects_a_different_data_object(self) -> None:
        with patched_causal_learn_bdeu(self.data):
            with self.assertRaisesRegex(ValueError, "different data object"):
                self.ges_module.local_score_BDeu(self.data.copy(), 2, [0], None)

    def test_patched_ges_matches_reference_ges(self) -> None:
        rng = np.random.default_rng(1)
        cause = rng.integers(0, 2, size=50, dtype=np.int64)
        mediator = cause ^ (rng.random(50) < 0.1)
        outcome = mediator ^ (rng.random(50) < 0.1)
        independent = rng.integers(0, 2, size=50, dtype=np.int64)
        data = np.column_stack((cause, mediator, outcome, independent))
        names = ["A", "B", "C", "D"]
        reference = self.ges_module.ges(
            data,
            score_func="local_score_BDeu",
            maxP=None,
            node_names=names,
        )
        with patched_causal_learn_bdeu(data) as scorer:
            optimized = self.ges_module.ges(
                data,
                score_func="local_score_BDeu",
                maxP=None,
                node_names=names,
            )

        np.testing.assert_array_equal(reference["G"].graph, optimized["G"].graph)
        self.assertAlmostEqual(reference["score"], optimized["score"], places=12)
        self.assertGreater(len(reference["update1"]), 0)
        self.assertEqual(reference["update1"], optimized["update1"])
        self.assertEqual(reference["update2"], optimized["update2"])
        self.assertGreater(scorer.cache_info()["size"], 0)
        self.assertGreater(scorer.cache_info()["evaluation_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
