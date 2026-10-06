import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cd_llm.experiments.alarm_observational import (
    canonical_matrix, edge_matrix, evaluate, fit, standardize_constants,
)
from cd_llm.methods.fast_bdeu import FastBDeuLocalScore


class AlarmObservationalTests(unittest.TestCase):
    def test_constants_are_retained_and_finite(self):
        values = standardize_constants(np.array([[0, 7, 2], [1, 7, 4], [2, 7, 6]]))
        np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(values.std(axis=0), [1, 0, 1])
        self.assertTrue(np.isfinite(values).all())

    def test_permutation_preserves_endpoints_and_metrics(self):
        # Presented nodes are canonical 2, 0, 1. The first edge is 2 -> 0.
        matrix = edge_matrix(3, [(0, 1)])
        matrix[1, 2] = matrix[2, 1] = 1  # Unresolved 0 -- 1.
        actual = canonical_matrix(matrix, [2, 0, 1])
        self.assertEqual((actual[2, 0], actual[0, 2]), (1, 2))
        reference = dict(names=["A", "B", "C"], edges=[[2, 0], [0, 1]])
        metrics = evaluate(actual, reference)
        self.assertEqual(metrics["adjacency_f1"], 1)
        self.assertAlmostEqual(metrics["directed_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["distance"], 1 / 6)
        self.assertEqual(metrics["unresolved_edges"], 1)
        with self.assertRaises(ValueError):
            canonical_matrix(matrix, [0, 0, 1])

    def test_pc_uses_native_discrete_stable_pc_and_raw_endpoints(self):
        raw = np.array([[0, -1, 0], [1, 0, -1], [0, -1, 0]])
        result = SimpleNamespace(G=SimpleNamespace(graph=raw))
        with patch("causallearn.search.ConstraintBased.PC.pc", return_value=result) as pc:
            matrix, diagnostics = fit("pc", np.zeros((5, 3)), ["V2", "V3", "V1"])
        self.assertEqual(pc.call_args.kwargs["indep_test"], "gsq")
        self.assertTrue(pc.call_args.kwargs["stable"])
        self.assertEqual(pc.call_args.kwargs["alpha"], .05)
        self.assertEqual(pc.call_args.kwargs["uc_rule"], 0)
        self.assertEqual(pc.call_args.kwargs["uc_priority"], 2)
        np.testing.assert_array_equal(matrix, [[0, 1, 0], [2, 0, 1], [0, 1, 0]])
        self.assertEqual(diagnostics["raw_causal_learn_endpoints"], raw.tolist())

    def test_discrete_scores_ignore_category_labels_and_handle_constants(self):
        from causallearn.utils.cit import CIT
        from causallearn.score.LocalScoreFunction import local_score_BDeu
        rng = np.random.default_rng(23)
        data = rng.integers(0, 3, (80, 4))
        data[:, 3] = 0
        recoded = np.array([17, -4, 100])[data]
        original_test, recoded_test = CIT(data, "gsq"), CIT(recoded, "gsq")
        for source, target, conditions in [(0, 1, []), (0, 2, [1]), (0, 3, []), (0, 1, [3])]:
            p = original_test(source, target, conditions)
            self.assertTrue(np.isfinite(p))
            self.assertAlmostEqual(p, recoded_test(source, target, conditions))
        original_score, recoded_score = FastBDeuLocalScore(data), FastBDeuLocalScore(recoded)
        for child, parents in [(0, []), (0, [1, 2]), (0, [3]), (3, [0, 1])]:
            score = original_score.local_score(child, parents)
            self.assertTrue(np.isfinite(score))
            self.assertAlmostEqual(score, recoded_score.local_score(child, parents))
            self.assertAlmostEqual(score, float(local_score_BDeu(data, child, parents)))


if __name__ == "__main__":
    unittest.main()
