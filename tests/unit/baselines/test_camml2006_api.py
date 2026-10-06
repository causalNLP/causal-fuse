from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from cd_llm.baselines.camml2006 import (
    AdaptiveMMLLocalScore,
    CaMMLConfig,
    CaMMLPrior,
    adapt_evidence_to_prior,
    fit_camml,
    fit_camml_from_local_score,
    fit_camml_with_evidence,
)
from cd_llm.core.finite_support import Claim, EvidenceGraph, EvidenceNode, graph_from_edges


def evidence_node(identifier: str, index: int, stance: int = 1, **changes: object) -> EvidenceNode:
    values = dict(
        evidence_id=identifier,
        claim_index=index,
        stance=stance,
        confidence=3.7 / 4.7,
        provenance_family=identifier,
        factor_id=identifier,
        context_id="target",
        context_compatible=True,
        correct=True,
    )
    values.update(changes)
    return EvidenceNode(**values)  # type: ignore[arg-type]


class AdaptiveMMLTests(unittest.TestCase):
    def test_hand_computed_binary_no_parent_cost(self) -> None:
        data = np.array([[0], [0], [1]], dtype=int)
        score = AdaptiveMMLLocalScore(data, cardinalities=(2,))
        expected_cost = math.lgamma(5) - math.lgamma(2) - math.lgamma(3) - math.lgamma(2) + 0.17649
        self.assertAlmostEqual(score.local_score(0, 0), -expected_cost)

    def test_cache(self) -> None:
        score = AdaptiveMMLLocalScore(np.array([[0, 0], [1, 1]]))
        score.local_score(1, 1)
        score.local_score(1, 1)
        self.assertEqual(score.cache_size, 1)


class EvidenceAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.claims = (
            Claim("d", "I", "directed", 0, 0, 1),
            Claim("a", "I", "adjacency", 0, 1, 2),
            Claim("x", "I", "directed_exclusion", 0, 2, 1),
            Claim("anc", "II", "ancestry", 0, 0, 2),
            Claim("ord", "II", "ordering", 0, 2, 0),
        )
        self.evidence = EvidenceGraph(
            "target", tuple(evidence_node(f"e{i}", i) for i in range(len(self.claims)))
        )

    def test_supported_claims_map_without_target_truth(self) -> None:
        first = adapt_evidence_to_prior(self.claims, self.evidence, node_count=3)
        changed_claims = tuple(replace(claim, truth=1 - claim.truth) for claim in self.claims)
        changed_nodes = tuple(replace(node, correct=not node.correct) for node in self.evidence.nodes)
        second = adapt_evidence_to_prior(
            changed_claims, EvidenceGraph("target", changed_nodes), node_count=3
        )
        self.assertEqual(first.prior, second.prior)
        self.assertEqual(len(first.contributions), 5)
        self.assertEqual(len(first.prior.global_), 1)

    def test_context_and_provenance_filtering(self) -> None:
        repeated = replace(
            self.evidence.nodes[0], evidence_id="repeat", confidence=0.6
        )
        incompatible = replace(
            self.evidence.nodes[1], evidence_id="bad", context_compatible=False
        )
        graph = EvidenceGraph("target", self.evidence.nodes + (repeated, incompatible))
        result = adapt_evidence_to_prior(self.claims, graph, node_count=3)
        self.assertEqual({value.reason for value in result.exclusions}, {"provenance_duplicate", "context_incompatible"})

    def test_unsupported_claim_rejected(self) -> None:
        claim = Claim("b", "III", "bidirected", 0, 0, 1)
        with self.assertRaises(ValueError):
            adapt_evidence_to_prior((claim,), EvidenceGraph("target", (evidence_node("e", 0),)), node_count=2)

    def test_whole_graph_prior(self) -> None:
        claim = Claim("g", "IV", "whole_graph", 1, guess=graph_from_edges(2, ((0, 1),)))
        result = adapt_evidence_to_prior(
            (claim,), EvidenceGraph("target", (evidence_node("e", 0),)), node_count=2
        )
        self.assertEqual(len(result.prior.graph_priors), 1)


class SamplerTests(unittest.TestCase):
    def test_two_node_constant_score_matches_tom_posterior(self) -> None:
        class ConstantScore:
            node_count = 2
            name = "constant"

            @staticmethod
            def local_score(node: int, parent_mask: int) -> float:
                del node, parent_mask
                return 0.0

        result = fit_camml_from_local_score(
            ConstantScore(),
            CaMMLPrior.empty(2),
            config=CaMMLConfig(
                seed=17,
                chain_count=4,
                iterations=10_000,
                burn_in=2_000,
                thin=4,
                max_parents=1,
            ),
        )
        empty = graph_from_edges(2, ())
        forward = graph_from_edges(2, ((0, 1),))
        reverse = graph_from_edges(2, ((1, 0),))
        self.assertAlmostEqual(result.support[empty], 0.5, delta=0.05)
        self.assertAlmostEqual(result.support[forward], 0.25, delta=0.05)
        self.assertAlmostEqual(result.support[reverse], 0.25, delta=0.05)

    def test_deterministic_point_and_support_api(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(100, 2))
        data[:, 1] = 2.0 * data[:, 0] + rng.normal(scale=0.2, size=100)
        config = CaMMLConfig(
            seed=9,
            chain_count=2,
            iterations=800,
            burn_in=200,
            thin=10,
            max_parents=1,
        )
        first = fit_camml(data, data_score="gaussian-bic", config=config)
        second = fit_camml(data, data_score="gaussian-bic", config=config)
        self.assertEqual(first.graph, second.graph)
        self.assertEqual(dict(first.support), dict(second.support))
        self.assertAlmostEqual(sum(first.support.values()), 1.0)
        self.assertEqual(first.map_tom.graph, first.graph)

    def test_evidence_entrypoint_attaches_audit(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(size=(80, 2))
        claim = Claim("d", "I", "directed", 1, 0, 1)
        evidence = EvidenceGraph("target", (evidence_node("e", 0),))
        result = fit_camml_with_evidence(
            data,
            (claim,),
            evidence,
            data_score="gaussian-bic",
            config=CaMMLConfig(chain_count=1, iterations=300, burn_in=100, thin=10),
        )
        self.assertIsNotNone(result.evidence_adaptation)


if __name__ == "__main__":
    unittest.main()
