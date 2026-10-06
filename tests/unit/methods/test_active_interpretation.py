import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.methods.active_query import ClaimQuery, QueryResponse, active_query_smc
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class ActiveInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.path = graph_from_edges(3, [(0, 1), (1, 2)])
        self.edge = graph_from_edges(3, [(0, 2)])
        self.empty = graph_from_edges(3)
        self.score = GaussianBICScore(np.random.default_rng(1).normal(size=(12, 3)), max_parents=2)

    def run_case(self, mode="literal", answer=1, kind="directed", confidence=0.7):
        query = ClaimQuery(kind, 0, 2, "question")
        captured = []
        def mutation(score, particles, claims, evidence, *args):
            captured.extend(claims)
            return particles
        with patch("cd_llm.methods.active_query._rejuvenate", side_effect=mutation):
            result = active_query_smc(None, self.score,
                lambda q: QueryResponse(answer, confidence if answer is not None else None),
                query_pool=(query,), budgets=(1,), seed=9, ess_threshold_fraction=0.01,
                initial_particles=(self.path, self.edge, self.empty), initial_weights=(1/3,)*3,
                direct_yes_interpretation=mode)
        return result, captured

    def test_path_and_edge_both_supported_only_by_broader_yes(self):
        literal, _ = self.run_case()
        broader, claims = self.run_case("ancestry")
        np.testing.assert_allclose(literal.weights, (3/13, 7/13, 3/13))
        np.testing.assert_allclose(broader.weights, (7/17, 7/17, 3/17))
        step = broader.steps[0]
        self.assertEqual(step.query.kind, "directed")
        self.assertEqual(step.applied_claim.kind, "ancestry")
        self.assertEqual(step.applied_claim.group, "II")
        self.assertEqual(claims, [step.applied_claim])
        self.assertAlmostEqual(step.applied_likelihood_normalizer, 1.7/3)
        self.assertEqual(step.expected_information_gain, literal.steps[0].expected_information_gain)
        self.assertEqual(step.reported_confidence, 0.7)

    def test_negative_does_not_exclude_indirect_path(self):
        literal, _ = self.run_case(answer=0)
        broader, claims = self.run_case("ancestry", answer=0)
        self.assertEqual(literal.weights, broader.weights)
        np.testing.assert_allclose(broader.weights, (7/17, 3/17, 7/17))
        self.assertEqual(claims[0].kind, "directed")
        self.assertEqual(claims[0].truth, 0)

    def test_adjacency_and_abstention_unchanged(self):
        for answer in (0, 1, None):
            literal, _ = self.run_case(answer=answer, kind="adjacency")
            broader, _ = self.run_case("ancestry", answer=answer, kind="adjacency")
            self.assertEqual(literal.weights, broader.weights)
            self.assertEqual(literal.steps[0].applied_claim, broader.steps[0].applied_claim)
        self.assertIsNone(self.run_case("ancestry", answer=None)[0].steps[0].applied_claim)

    def test_hard_recovery_uses_ancestry_constraint(self):
        result = active_query_smc(None, self.score,
            lambda q: QueryResponse(1, 1.0, treatment="hard_constraint"),
            query_pool=(ClaimQuery("directed", 0, 2, "q"),), budgets=(1,),
            initial_particles=(self.empty, self.empty), initial_weights=(0.5,0.5),
            direct_yes_interpretation="ancestry", rejuvenation_steps=0)
        self.assertTrue(result.steps[0].support_recovered)
        self.assertEqual(result.steps[0].applied_claim.kind, "ancestry")


if __name__ == "__main__":
    unittest.main()
