from __future__ import annotations

import unittest

import numpy as np

from cd_llm.core.finite_support import (
    Claim,
    EvidenceGraph,
    EvidenceNode,
    evaluate_claim,
    graph_from_edges,
)
from cd_llm.methods.active_query import (
    ClaimQuery,
    QueryResponse,
    active_query_smc,
)
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.methods.evidence_guided import (
    ClaimFuseSMCConfig,
    EvidenceGuidedDAGProposal,
    evidence_guided_tempered_smc,
)


class ActiveEvidenceTreatmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.score = GaussianBICScore(
            np.random.default_rng(17).normal(size=(30, 3)), max_parents=2
        )
        self.empty = graph_from_edges(3)
        self.edge_01 = graph_from_edges(3, ((0, 1),))
        self.edge_12 = graph_from_edges(3, ((1, 2),))
        self.both = graph_from_edges(3, ((0, 1), (1, 2)))

    def test_exact_one_requires_an_explicit_treatment(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        with self.assertRaisesRegex(ValueError, "explicit treatment"):
            active_query_smc(
                {self.empty: 0.5, self.edge_01: 0.5},
                self.score,
                lambda _: 1,
                query_pool=(query,),
                budgets=(1,),
                confidence=1.0,
                rejuvenation_steps=0,
            )
        with self.assertRaisesRegex(ValueError, "explicit treatment"):
            QueryResponse(1, 1.0)
        with self.assertRaisesRegex(ValueError, "requires reported confidence"):
            QueryResponse(1, 0.9, treatment="hard_constraint")

    def test_soft_one_matches_point_99_and_never_projects(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        common = {
            "query_pool": (query,),
            "budgets": (1,),
            "treatment": "soft_evidence",
            "rejuvenation_steps": 0,
            "seed": 9,
        }
        capped = active_query_smc(
            {self.empty: 0.5, self.edge_01: 0.5},
            self.score,
            lambda _: 1,
            confidence=1.0,
            **common,
        )
        direct = active_query_smc(
            {self.empty: 0.5, self.edge_01: 0.5},
            self.score,
            lambda _: 1,
            confidence=0.99,
            **common,
        )
        self.assertEqual(capped.distributions, direct.distributions)
        self.assertAlmostEqual(capped.distributions[1][self.edge_01], 0.99)
        self.assertAlmostEqual(capped.distributions[1][self.empty], 0.01)
        self.assertEqual(capped.support_recovery_count, 0)
        step = capped.steps[0]
        self.assertEqual(step.reported_confidence, 1.0)
        self.assertEqual(step.effective_confidence, 0.99)
        self.assertEqual(step.treatment, "soft_evidence")

    def test_structured_responses_can_mix_hard_and_soft_evidence(self) -> None:
        queries = (
            ClaimQuery("directed", 0, 1, "hard-edge"),
            ClaimQuery("directed", 1, 2, "soft-edge"),
        )

        def provider(query: ClaimQuery) -> QueryResponse:
            if query.query_id == "hard-edge":
                return QueryResponse(1, 1.0, treatment="hard_constraint")
            return QueryResponse(1, 0.8, treatment="soft_evidence")

        result = active_query_smc(
            {
                self.empty: 0.25,
                self.edge_01: 0.25,
                self.edge_12: 0.25,
                self.both: 0.25,
            },
            self.score,
            provider,
            query_pool=queries,
            budgets=(2,),
            confidence=0.9,
            treatment="soft_evidence",
            ess_threshold_fraction=0.1,
            rejuvenation_steps=0,
            seed=3,
        )
        final = result.distributions[2]
        self.assertEqual(set(final), {self.edge_01, self.both})
        self.assertAlmostEqual(final[self.edge_01], 0.2)
        self.assertAlmostEqual(final[self.both], 0.8)
        self.assertEqual(
            tuple(step.treatment for step in result.steps),
            ("hard_constraint", "soft_evidence"),
        )


class PassiveEvidenceTreatmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.score = GaussianBICScore(
            np.random.default_rng(23).normal(size=(30, 3)), max_parents=2
        )
        self.empty = graph_from_edges(3)
        self.edge_01 = graph_from_edges(3, ((0, 1),))
        self.edge_12 = graph_from_edges(3, ((1, 2),))
        self.both = graph_from_edges(3, ((0, 1), (1, 2)))
        self.claims = (
            Claim("edge-01", "I", "directed", 1, 0, 1),
            Claim("edge-12", "I", "directed", 1, 1, 2),
        )

    def _evidence(self) -> EvidenceGraph:
        return EvidenceGraph(
            "target",
            (
                EvidenceNode(
                    "hard",
                    0,
                    1,
                    1.0,
                    "verified-source",
                    "hard-factor",
                    "target",
                    True,
                    True,
                    treatment="hard_constraint",
                ),
                EvidenceNode(
                    "soft",
                    1,
                    1,
                    0.8,
                    "expert-source",
                    "soft-factor",
                    "target",
                    True,
                    True,
                    treatment="soft_evidence",
                ),
            ),
        )

    def test_mixed_target_masks_hard_support_and_weights_soft_odds(self) -> None:
        result = evidence_guided_tempered_smc(
            self.score,
            (self.empty, self.edge_01, self.edge_12, self.both),
            self.claims,
            self._evidence(),
            seed=5,
            temperatures=(0.0, 1.0),
            ess_threshold_fraction=0.1,
            rejuvenation_steps=0,
        )
        self.assertEqual(set(result.distribution), {self.edge_01, self.both})
        self.assertAlmostEqual(result.distribution[self.edge_01], 0.2)
        self.assertAlmostEqual(result.distribution[self.both], 0.8)
        self.assertEqual(result.hard_constraint_count, 1)
        self.assertEqual(result.soft_evidence_count, 1)
        self.assertFalse(result.support_recovered)
        self.assertEqual(
            {
                (
                    record.reported_confidence,
                    record.effective_confidence,
                    record.treatment,
                )
                for record in result.evidence_confidences
            },
            {(1.0, 1.0, "hard_constraint"), (0.8, 0.8, "soft_evidence")},
        )

    def test_zero_empirical_hard_support_is_projected_jointly(self) -> None:
        hard_only = EvidenceGraph("target", (self._evidence().nodes[0],))
        result = evidence_guided_tempered_smc(
            self.score,
            (self.empty, self.empty),
            self.claims,
            hard_only,
            seed=7,
            temperatures=(0.0, 1.0),
            rejuvenation_steps=0,
        )
        self.assertTrue(result.support_recovered)
        self.assertTrue(
            all(evaluate_claim(graph, self.claims[0]) == 1 for graph in result.particles)
        )

    def test_negative_group_i_predicates_are_supported_by_projection(self) -> None:
        claim = Claim("no-edge", "I", "directed_exclusion", 1, 0, 1)
        evidence = EvidenceGraph(
            "target",
            (
                EvidenceNode(
                    "verified-no-edge",
                    0,
                    1,
                    1.0,
                    "verified",
                    "negative-factor",
                    "target",
                    True,
                    True,
                    treatment="hard_constraint",
                ),
            ),
        )
        result = evidence_guided_tempered_smc(
            self.score,
            (self.edge_01, self.edge_01),
            (claim,),
            evidence,
            seed=11,
            temperatures=(0.0, 1.0),
            rejuvenation_steps=0,
        )
        self.assertTrue(result.support_recovered)
        self.assertTrue(all(evaluate_claim(graph, claim) == 1 for graph in result.particles))

    def test_soft_one_and_point_99_have_identical_proposal_targets(self) -> None:
        def evidence(confidence: float) -> EvidenceGraph:
            return EvidenceGraph(
                "target",
                (
                    EvidenceNode(
                        "soft",
                        0,
                        1,
                        confidence,
                        "expert",
                        "factor",
                        "target",
                        True,
                        True,
                        treatment="soft_evidence",
                    ),
                ),
            )

        capped = EvidenceGuidedDAGProposal(
            self.claims,
            evidence(1.0),
            node_count=3,
        )
        direct = EvidenceGuidedDAGProposal(
            self.claims,
            evidence(0.99),
            node_count=3,
        )
        for graph in (self.empty, self.edge_01, self.edge_12, self.both):
            self.assertEqual(capped.evidence_score(graph), direct.evidence_score(graph))
            self.assertEqual(
                capped.proposal_distribution(graph), direct.proposal_distribution(graph)
            )

    def test_production_config_validates_and_records_the_soft_cap(self) -> None:
        self.assertEqual(ClaimFuseSMCConfig().soft_confidence_cap, 0.99)
        self.assertEqual(
            ClaimFuseSMCConfig(soft_confidence_cap=0.95).soft_confidence_cap,
            0.95,
        )
        with self.assertRaisesRegex(ValueError, "soft_confidence_cap"):
            ClaimFuseSMCConfig(soft_confidence_cap=1.0)


if __name__ == "__main__":
    unittest.main()
