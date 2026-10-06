from __future__ import annotations

import math
import unittest
from dataclasses import replace

from cd_llm.baselines.eca2025 import (
    EdgeConstraint,
    adapt_evidence_to_prior,
    confidence_from_log_odds,
    confidence_log_odds,
)
from cd_llm.core.finite_support import Claim, EvidenceGraph, EvidenceNode


def _evidence(
    evidence_id: str,
    claim_index: int,
    stance: int,
    *,
    confidence: float = 3.7 / 4.7,
    family: str | None = None,
    factor: str | None = None,
    compatible: bool = True,
    correct: bool = False,
) -> EvidenceNode:
    return EvidenceNode(
        evidence_id=evidence_id,
        claim_index=claim_index,
        stance=stance,  # type: ignore[arg-type]
        confidence=confidence,
        provenance_family=evidence_id if family is None else family,
        factor_id=evidence_id if factor is None else factor,
        context_id="target" if compatible else "other",
        context_compatible=compatible,
        correct=correct,
    )


class FixedEvidenceMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.claims = (
            Claim("I-4", "I", "adjacency", 1, 1, 4),
            Claim("I-7", "I", "non_adjacency", 0, 8, 9),
            Claim("I-2", "I", "directed_exclusion", 1, 3, 5),
            Claim("I-3", "I", "directed_exclusion", 0, 3, 6),
            Claim("I-5", "I", "adjacency", 0, 0, 5),
        )
        self.nodes = (
            _evidence("ev-I-4", 0, -1, correct=False),
            _evidence("ev-I-7", 1, 1, correct=False),
            _evidence("ev-I-2", 2, -1, correct=True),
            _evidence("ev-I-3", 3, 1, correct=True),
            _evidence("ev-I-5", 4, 1, correct=False),
        )

    def test_all_five_claims_map_to_the_expected_eight_ordered_entries(self) -> None:
        adaptation = adapt_evidence_to_prior(
            self.claims,
            EvidenceGraph("target", self.nodes),
            node_count=10,
            sigma=1.0,
        )
        confidence = 3.7 / 4.7
        forbidden = 1.0 - confidence
        expected = {
            (1, 4): forbidden,
            (4, 1): forbidden,
            (8, 9): forbidden,
            (9, 8): forbidden,
            (3, 5): confidence,
            (3, 6): forbidden,
            (0, 5): confidence,
            (5, 0): confidence,
        }
        actual = {
            (constraint.source, constraint.target): constraint.probability
            for constraint in adaptation.prior.constraints
        }
        self.assertEqual(set(actual), set(expected))
        for edge, probability in expected.items():
            self.assertAlmostEqual(actual[edge], probability, places=14)
        self.assertEqual(int(adaptation.prior.mask.sum()), 8)
        self.assertEqual(len(adaptation.contributions), 8)
        signs = {
            (item.source, item.target): math.copysign(1.0, item.signed_log_odds)
            for item in adaptation.contributions
        }
        self.assertEqual(signs[(3, 5)], 1.0)
        self.assertEqual(signs[(3, 6)], -1.0)

    def test_symmetric_positive_adjacency_has_the_requested_dag_likelihood_ratio(self) -> None:
        confidence = 3.7 / 4.7
        no_edge = (1.0 - confidence) ** 2
        either_orientation = confidence * (1.0 - confidence)
        self.assertAlmostEqual(either_orientation / no_edge, 3.7, places=14)

    def test_evaluation_only_truth_and_correct_fields_do_not_affect_conversion(self) -> None:
        baseline = adapt_evidence_to_prior(
            self.claims,
            EvidenceGraph("target", self.nodes),
            node_count=10,
        )
        changed_claims = tuple(replace(claim, truth=1 - claim.truth) for claim in self.claims)
        changed_nodes = tuple(replace(node, correct=not node.correct) for node in self.nodes)
        changed = adapt_evidence_to_prior(
            changed_claims,
            EvidenceGraph("target", changed_nodes),
            node_count=10,
        )
        self.assertEqual(changed, baseline)


class ProvenanceContextAndConflictTests(unittest.TestCase):
    def test_repeated_provenance_uses_one_conservative_representative_and_filters_context(self) -> None:
        claims = (Claim("edge", "I", "directed", 0, 0, 1),)
        nodes = (
            _evidence(
                "dup-a",
                0,
                1,
                confidence=3.7 / 4.7,
                family="source",
                factor="shared",
                correct=False,
            ),
            _evidence(
                "dup-b",
                0,
                -1,
                confidence=0.6,
                family="source",
                factor="shared",
                correct=True,
            ),
            _evidence(
                "wrong-context",
                0,
                1,
                confidence=0.8,
                family="other-source",
                factor="other-factor",
                compatible=False,
            ),
        )
        adaptation = adapt_evidence_to_prior(
            claims, EvidenceGraph("target", nodes), node_count=2
        )
        self.assertEqual(adaptation.used_evidence_ids, ("dup-a",))
        self.assertEqual(adaptation.excluded_evidence_ids, ("dup-b", "wrong-context"))
        self.assertEqual(
            tuple((item.evidence_id, item.reason) for item in adaptation.exclusions),
            (
                ("dup-b", "provenance_duplicate"),
                ("wrong-context", "context_incompatible"),
            ),
        )
        self.assertEqual(len(adaptation.contributions), 1)
        self.assertAlmostEqual(
            adaptation.contributions[0].signed_log_odds,
            confidence_log_odds(0.6),
        )
        self.assertEqual(
            adaptation.prior.constraints,
            (
                EdgeConstraint(
                    0,
                    1,
                    confidence_from_log_odds(confidence_log_odds(0.6)),
                ),
            ),
        )

    def test_independent_equal_conflict_cancels_without_hiding_the_audit(self) -> None:
        claims = (Claim("edge", "I", "directed", 0, 0, 1),)
        nodes = (
            _evidence("support", 0, 1, confidence=0.7),
            _evidence("oppose", 0, -1, confidence=0.7),
        )
        adaptation = adapt_evidence_to_prior(
            claims, EvidenceGraph("target", nodes), node_count=2
        )
        self.assertEqual(adaptation.prior.constraints, ())
        self.assertEqual(len(adaptation.contributions), 2)
        self.assertEqual(len(adaptation.edge_log_odds), 1)
        self.assertEqual(adaptation.edge_log_odds[0].log_odds, 0.0)
        self.assertEqual(set(adaptation.used_evidence_ids), {"oppose", "support"})

    def test_neutral_confidence_is_omitted_from_prior_mask(self) -> None:
        claims = (Claim("edge", "I", "directed", 0, 0, 1),)
        adaptation = adapt_evidence_to_prior(
            claims,
            EvidenceGraph("target", (_evidence("neutral", 0, 1, confidence=0.5),)),
            node_count=2,
        )
        self.assertEqual(adaptation.prior.constraints, ())
        self.assertEqual(adaptation.contributions[0].signed_log_odds, 0.0)
        self.assertEqual(adaptation.edge_log_odds[0].probability, 0.5)

    def test_evidence_adapter_rejects_nonfinite_endpoint_log_odds(self) -> None:
        claims = (Claim("edge", "I", "directed", 0, 0, 1),)
        for confidence in (0.0, 1.0):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(
                ValueError, "strictly"
            ):
                adapt_evidence_to_prior(
                    claims,
                    EvidenceGraph(
                        "target",
                        (_evidence("endpoint", 0, 1, confidence=confidence),),
                    ),
                    node_count=2,
                )

    def test_unsupported_claims_and_malformed_evidence_fail_loudly(self) -> None:
        group_two = (Claim("ancestor", "II", "ancestry", 1, 0, 1),)
        with self.assertRaisesRegex(ValueError, "Group-I"):
            adapt_evidence_to_prior(
                group_two,
                EvidenceGraph("target", (_evidence("ev", 0, 1),)),
                node_count=2,
            )
        claim = (Claim("edge", "I", "directed", 1, 0, 1),)
        duplicate_ids = (
            _evidence("same", 0, 1, family="a"),
            _evidence("same", 0, 1, family="b"),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            adapt_evidence_to_prior(
                claim, EvidenceGraph("target", duplicate_ids), node_count=2
            )
        with self.assertRaisesRegex(ValueError, "claim_index"):
            adapt_evidence_to_prior(
                claim,
                EvidenceGraph("target", (_evidence("bad-index", 1, 1),)),
                node_count=2,
            )
        with self.assertRaisesRegex(ValueError, "stances"):
            adapt_evidence_to_prior(
                claim,
                EvidenceGraph("target", (_evidence("bad-stance", 0, 0),)),
                node_count=2,
            )


if __name__ == "__main__":
    unittest.main()
