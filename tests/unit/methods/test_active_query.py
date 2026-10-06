from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import Claim, graph_from_edges, has_adjacency, has_edge
from cd_llm.methods.active_query import (
    AnswerabilityEstimate,
    ClaimQuery,
    QueryResponse,
    _evidence_graph,
    _nonredundant_query_indices,
    _query_truth_probabilities,
    _recover_hard_constraint_support,
    _rejuvenate,
    _select_uniform_candidate_index,
    active_query_smc,
    ancestry_ordering_query_links,
    binary_entropy,
    dag_query_pool,
    expected_information_gain,
    group_i_query_pool,
    query_values,
)
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class ActiveQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.empty = graph_from_edges(3)
        self.forward = graph_from_edges(3, [(0, 1)])
        self.reverse = graph_from_edges(3, [(1, 0)])
        self.distribution = {
            self.empty: 1.0 / 3.0,
            self.forward: 1.0 / 3.0,
            self.reverse: 1.0 / 3.0,
        }
        self.score = GaussianBICScore(
            np.random.default_rng(4).normal(size=(30, 3)), max_parents=2
        )

    def answer_from_truth(self, query: ClaimQuery) -> int:
        if query.kind == "directed":
            return int(query.source == 0 and query.target == 1)
        return int({query.source, query.target} == {0, 1})

    def test_answered_budget_skips_abstentions_and_mutates_at_terminal_answer(self) -> None:
        responses = iter((None, 0.5, None, 0.5, 0.5))

        def provider(query):
            confidence = next(responses)
            return QueryResponse(None, None) if confidence is None else QueryResponse(
                1, confidence, treatment="soft_evidence"
            )

        with patch("cd_llm.methods.active_query._rejuvenate", side_effect=lambda *args: args[1]) as mutation:
            result = active_query_smc(
                self.distribution, self.score, provider, budgets=(1, 3),
                budget_unit="answered", seed=10, rejuvenation_steps=2,
            )
        self.assertEqual(result.query_count, 5)
        self.assertEqual(result.answered_query_count, 3)
        self.assertEqual(result.abstention_count, 2)
        self.assertEqual(result.unused_query_budget, 0)
        self.assertEqual(len({s.query.query_id for s in result.steps}), 5)
        self.assertEqual(result.distributions[1], self.distribution)
        self.assertEqual(result.distributions[3], self.distribution)
        self.assertEqual(result.particles, tuple(self.distribution))
        self.assertEqual(mutation.call_count, 1)
        self.assertEqual(mutation.call_args.args[4], 15)
        self.assertEqual(result.steps[-1].budget, 5)
        self.assertTrue(all(not s.resampled for s in result.steps))

    def test_answered_budget_exhaustion_does_not_fabricate_snapshots(self) -> None:
        result = active_query_smc(
            self.distribution, self.score, lambda q: QueryResponse(None, None),
            budgets=(2,), budget_unit="answered",
        )
        self.assertEqual(result.query_count, 9)
        self.assertEqual(result.answered_query_count, 0)
        self.assertEqual(result.unused_query_budget, 2)
        self.assertEqual(result.early_stop_reason, "query-pool-exhausted")
        self.assertEqual(result.distributions, {})
        self.assertEqual(result.particles, tuple(self.distribution))
        np.testing.assert_allclose(result.weights, tuple(self.distribution.values()))

    def test_unknown_budget_unit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "budget unit"):
            active_query_smc(self.distribution, self.score, self.answer_from_truth, budget_unit="free")

    def test_group_i_pool_is_unlabeled_unique_and_canonical(self) -> None:
        pool = group_i_query_pool(20)
        self.assertEqual(len(pool), 570)
        self.assertEqual(len(set(pool)), 570)
        self.assertTrue(all(query.source != query.target for query in pool))
        self.assertEqual(pool, tuple(sorted(pool)))

    def test_dag_query_family_pools_have_expected_sizes(self) -> None:
        self.assertEqual(len(dag_query_pool(20, ("local_edge",))), 570)
        self.assertEqual(len(dag_query_pool(20, ("causal_order",))), 380)
        self.assertEqual(len(dag_query_pool(20, ("ancestry",))), 380)
        all_queries = dag_query_pool(
            20, ("local_edge", "causal_order", "ancestry")
        )
        self.assertEqual(len(all_queries), 1_330)
        self.assertEqual(len(set(all_queries)), 1_330)
        links = ancestry_ordering_query_links(all_queries)
        self.assertEqual(len(links), 760)
        self.assertTrue(all(links[linked] == query for query, linked in links.items()))

    def test_uniform_candidates_are_exactly_the_nonredundant_query_partitions(
        self,
    ) -> None:
        queries = dag_query_pool(
            20, ("local_edge", "causal_order", "ancestry")
        )
        links = ancestry_ordering_query_links(queries)
        candidates = _nonredundant_query_indices(queries, links)
        candidate_queries = tuple(queries[int(index)] for index in candidates)

        self.assertEqual(len(candidates), 950)
        self.assertEqual(len(set(int(index) for index in candidates)), 950)
        kinds = ("directed", "adjacency", "ancestry", "ordering")
        self.assertEqual(
            {
                kind: sum(query.kind == kind for query in candidate_queries)
                for kind in kinds
            },
            {
                "directed": 380,
                "adjacency": 190,
                "ancestry": 380,
                "ordering": 0,
            },
        )
        candidate_ids = {query.query_id for query in candidate_queries}
        self.assertTrue(
            all(
                (query_id in candidate_ids) != (linked_id in candidate_ids)
                for query_id, linked_id in links.items()
            )
        )

        class FixedUniformPosition:
            def __init__(self, position: int) -> None:
                self.position = position
                self.high: int | None = None

            def integers(self, high: int) -> int:
                self.high = high
                return self.position

        selected: list[int] = []
        for position in range(len(candidates)):
            rng = FixedUniformPosition(position)
            selected.append(_select_uniform_candidate_index(candidates, rng))
            self.assertEqual(rng.high, len(candidates))
        self.assertEqual(selected, [int(index) for index in candidates])

    def test_vectorized_group_ii_probabilities_match_direct_evaluation(self) -> None:
        path = graph_from_edges(3, [(0, 1), (1, 2)])
        fork = graph_from_edges(3, [(0, 1), (0, 2)])
        particles = (self.empty, path, fork, self.reverse)
        weights = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=float)
        queries = dag_query_pool(3, ("causal_order", "ancestry"))
        actual = _query_truth_probabilities(particles, weights, queries, 3)
        expected = np.asarray(
            [weights @ query_values(particles, query) for query in queries]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_vectorized_group_i_probabilities_match_direct_evaluation(self) -> None:
        particles = (self.empty, self.forward, self.reverse)
        weights = np.asarray((0.2, 0.3, 0.5), dtype=float)
        queries = group_i_query_pool(3)
        actual = _query_truth_probabilities(particles, weights, queries, 3)
        expected = np.asarray(
            [weights @ query_values(particles, query) for query in queries]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_fixed_confidence_eig_ranks_like_binary_entropy(self) -> None:
        probabilities = (0.05, 0.2, 0.5, 0.8, 0.95)
        for first in probabilities:
            for second in probabilities:
                entropy_gap = binary_entropy(first) - binary_entropy(second)
                eig_gap = expected_information_gain(
                    first, 0.9
                ) - expected_information_gain(second, 0.9)
                if abs(entropy_gap) < 1e-14:
                    self.assertAlmostEqual(eig_gap, 0.0, places=14)
                else:
                    self.assertEqual(math.copysign(1.0, entropy_gap), math.copysign(1.0, eig_gap))
        self.assertAlmostEqual(expected_information_gain(0.0, 0.9), 0.0)

    def test_eig_matches_direct_expected_kl(self) -> None:
        theta = 0.3
        confidence = 0.8
        positive = confidence * theta + (1.0 - confidence) * (1.0 - theta)
        direct = 0.0
        for truth, truth_mass in ((0, 1.0 - theta), (1, theta)):
            for answer in (0, 1):
                likelihood = confidence if answer == truth else 1.0 - confidence
                answer_mass = positive if answer == 1 else 1.0 - positive
                direct += truth_mass * likelihood * math.log(likelihood / answer_mass)
        self.assertAlmostEqual(
            expected_information_gain(theta, confidence), direct, places=14
        )

    def test_eig_clips_roundoff_but_rejects_invalid_probability(self) -> None:
        self.assertAlmostEqual(
            expected_information_gain(1.0 + 5e-16, 0.9), 0.0, places=14
        )
        with self.assertRaisesRegex(ValueError, "truth probability"):
            expected_information_gain(1.0 + 1e-6, 0.9)

    def test_answerability_eig_matches_eig_when_all_probabilities_are_one(self) -> None:
        pool = group_i_query_pool(3)
        common = dict(
            query_pool=pool,
            budgets=(1,),
            confidence=0.9,
            seed=12,
            rejuvenation_steps=0,
        )
        ordinary = active_query_smc(
            self.distribution, self.score, self.answer_from_truth, **common
        )
        aware = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            policy="answerability_eig",
            answerability_provider=lambda query: AnswerabilityEstimate(1.0, True),
            answerability_batch_size=1,
            answerability_assessment_budget=len(pool),
            answerability_probability_upper_bound=1.0,
            **common,
        )
        self.assertEqual(aware.steps[0].query, ordinary.steps[0].query)
        self.assertAlmostEqual(
            aware.steps[0].answerability_adjusted_information_gain,
            ordinary.steps[0].expected_information_gain,
        )

    def test_answerability_excludes_zero_probability_and_counts_screening(self) -> None:
        pool = group_i_query_pool(3)
        ordinary = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            query_pool=pool,
            budgets=(1,),
            rejuvenation_steps=0,
        )
        excluded = ordinary.steps[0].query.query_id

        def estimate(query: ClaimQuery) -> AnswerabilityEstimate:
            return AnswerabilityEstimate(
                0.0 if query.query_id == excluded else 0.6,
                query.query_id != excluded,
                prompt_tokens=7,
                completion_tokens=1,
            )

        aware = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            query_pool=pool,
            budgets=(1,),
            policy="answerability_eig",
            answerability_provider=estimate,
            answerability_batch_size=1,
            answerability_assessment_budget=len(pool),
            answerability_probability_upper_bound=0.6,
            rejuvenation_steps=0,
        )
        self.assertNotEqual(aware.steps[0].query.query_id, excluded)
        self.assertGreaterEqual(aware.answerability_assessment_count, 2)
        self.assertEqual(
            aware.answerability_prompt_tokens,
            7 * aware.answerability_assessment_count,
        )
        self.assertLessEqual(
            max(
                len(batch)
                for step in aware.steps
                for batch in step.answerability_screening_batches
            ),
            1,
        )

    def test_answerability_stops_without_spending_causal_query(self) -> None:
        result = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            query_pool=group_i_query_pool(3),
            budgets=(1,),
            policy="answerability_eig",
            answerability_provider=lambda query: AnswerabilityEstimate(0.0, False),
            answerability_batch_size=1,
            answerability_assessment_budget=2,
            answerability_probability_upper_bound=0.6,
            rejuvenation_steps=0,
        )
        self.assertEqual(result.query_count, 0)
        self.assertEqual(result.attempted_causal_queries, 0)
        self.assertEqual(result.unused_query_budget, 1)
        self.assertEqual(result.answerability_assessment_count, 2)
        self.assertEqual(
            result.early_stop_reason,
            "no-assessable-candidate-within-screening-budget",
        )
        self.assertIn(1, result.distributions)

    def test_answerability_shortlist_matches_exhaustive_adjusted_eig(self) -> None:
        pool = group_i_query_pool(3)
        particles = tuple(self.distribution)
        weights = np.asarray(tuple(self.distribution.values()))
        probabilities = _query_truth_probabilities(particles, weights, pool, 3)
        gains = [expected_information_gain(float(value), 0.9) for value in probabilities]
        answerability = {
            query.query_id: 0.55 + 0.05 * (index % 5)
            for index, query in enumerate(pool)
        }
        expected_index = min(
            range(len(pool)),
            key=lambda index: (
                -gains[index] * answerability[pool[index].query_id],
                -answerability[pool[index].query_id],
                -gains[index],
                pool[index],
            ),
        )
        calls: list[str] = []

        def estimate(query: ClaimQuery) -> AnswerabilityEstimate:
            calls.append(query.query_id)
            return AnswerabilityEstimate(answerability[query.query_id], True)

        result = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            query_pool=pool,
            budgets=(1,),
            policy="answerability_eig",
            answerability_provider=estimate,
            answerability_batch_size=5,
            answerability_assessment_budget=len(pool),
            answerability_probability_upper_bound=1.0,
            rejuvenation_steps=0,
        )
        self.assertEqual(result.steps[0].query, pool[expected_index])
        self.assertEqual(len(calls), len(set(calls)))
        self.assertTrue(
            all(
                len(batch) <= 5
                for batch in result.steps[0].answerability_screening_batches
            )
        )

    def test_eig_sequence_is_deterministic_and_queries_without_replacement(self) -> None:
        first = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            budgets=(1, 3),
            confidence=0.9,
            seed=12,
            rejuvenation_steps=0,
        )
        second = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            budgets=(1, 3),
            confidence=0.9,
            seed=12,
            rejuvenation_steps=0,
        )
        self.assertEqual(
            tuple(step.query for step in first.steps),
            tuple(step.query for step in second.steps),
        )
        self.assertEqual(first.distributions, second.distributions)
        self.assertEqual(len({step.query for step in first.steps}), 3)
        for budget, distribution in first.distributions.items():
            self.assertIn(budget, (1, 3))
            self.assertAlmostEqual(math.fsum(distribution.values()), 1.0)
        for step in first.steps:
            if step.query.kind == "directed":
                expected = int(
                    step.query.source == 0 and step.query.target == 1
                )
            else:
                expected = int({step.query.source, step.query.target} == {0, 1})
            self.assertEqual(step.answer, expected)
            self.assertEqual(step.confidence, 0.9)
            expected_observed = (
                step.positive_response_probability
                if expected == 1
                else 1.0 - step.positive_response_probability
            )
            self.assertAlmostEqual(
                step.observed_response_probability, expected_observed
            )

    def test_single_query_matches_exact_bayesian_update(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            {self.empty: 0.5, self.forward: 0.5},
            self.score,
            lambda selected: 1,
            query_pool=(query,),
            budgets=(1,),
            confidence=0.9,
            rejuvenation_steps=0,
        )
        self.assertAlmostEqual(result.distributions[1][self.empty], 0.1)
        self.assertAlmostEqual(result.distributions[1][self.forward], 0.9)
        self.assertEqual(result.steps[0].query, query)
        self.assertEqual(result.steps[0].answer, 1)
        self.assertAlmostEqual(result.steps[0].predicate_probability, 0.5)
        self.assertAlmostEqual(
            result.steps[0].expected_information_gain,
            math.log(2.0) - binary_entropy(0.9),
        )
        self.assertEqual(set(result.cumulative_seconds), {1})
        self.assertGreaterEqual(result.cumulative_seconds[1], 0.0)
        timing = result.timing
        components = (
            timing.setup_seconds,
            timing.selection_seconds,
            timing.oracle_seconds,
            timing.conditioning_seconds,
            timing.support_recovery_seconds,
            timing.resampling_seconds,
            timing.rejuvenation_seconds,
            timing.aggregation_seconds,
            timing.bookkeeping_seconds,
        )
        self.assertTrue(all(value >= 0.0 for value in components))
        self.assertAlmostEqual(math.fsum(components), timing.total_seconds)

    def test_structured_response_uses_its_own_confidence_and_records_usage(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            {self.empty: 0.5, self.forward: 0.5},
            self.score,
            lambda selected: QueryResponse(
                1,
                0.8,
                prompt_tokens=11,
                completion_tokens=3,
                attempts=2,
                audit={"provider": "test"},
            ),
            query_pool=(query,),
            budgets=(1,),
            confidence=0.9,
            rejuvenation_steps=0,
        )
        self.assertAlmostEqual(result.distributions[1][self.forward], 0.8)
        self.assertAlmostEqual(result.distributions[1][self.empty], 0.2)
        step = result.steps[0]
        self.assertEqual(step.confidence, 0.8)
        self.assertEqual(step.reported_confidence, 0.8)
        self.assertEqual(step.effective_confidence, 0.8)
        self.assertEqual(step.treatment, "soft_evidence")
        self.assertEqual(step.selection_confidence, 0.9)
        self.assertEqual(step.selection_reported_confidence, 0.9)
        self.assertEqual(step.selection_effective_confidence, 0.9)
        self.assertEqual(step.selection_treatment, "soft_evidence")
        self.assertEqual((step.prompt_tokens, step.completion_tokens), (11, 3))
        self.assertEqual(step.response_attempts, 2)
        self.assertEqual(step.response_audit, {"provider": "test"})
        self.assertEqual(result.answered_query_count, 1)
        self.assertEqual(result.abstention_count, 0)

    def test_abstention_consumes_budget_without_posterior_update(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            {self.empty: 0.5, self.forward: 0.5},
            self.score,
            lambda selected: QueryResponse(
                None,
                None,
                prompt_tokens=7,
                completion_tokens=2,
                abstention_reason="insufficient-evidence",
            ),
            query_pool=(query,),
            budgets=(1,),
            confidence=0.9,
            rejuvenation_steps=2,
        )
        self.assertEqual(
            result.distributions[1], {self.empty: 0.5, self.forward: 0.5}
        )
        step = result.steps[0]
        self.assertFalse(step.valid_response)
        self.assertIsNone(step.answer)
        self.assertIsNone(step.confidence)
        self.assertEqual(step.abstention_reason, "insufficient-evidence")
        self.assertEqual(result.query_count, 1)
        self.assertEqual(result.answered_query_count, 0)
        self.assertEqual(result.abstention_count, 1)

    def test_confidence_one_requires_an_explicit_treatment(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        with self.assertRaisesRegex(ValueError, "requires explicit treatment"):
            active_query_smc(
                {self.empty: 0.5, self.forward: 0.5},
                self.score,
                lambda selected: 1,
                query_pool=(query,),
                budgets=(1,),
                confidence=1.0,
                rejuvenation_steps=0,
            )

    def test_soft_confidence_one_is_capped_and_preserves_support(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            {self.empty: 0.5, self.forward: 0.5},
            self.score,
            lambda selected: 1,
            query_pool=(query,),
            budgets=(1,),
            confidence=1.0,
            treatment="soft_evidence",
            soft_confidence_cap=0.99,
            rejuvenation_steps=0,
        )
        self.assertAlmostEqual(result.distributions[1][self.empty], 0.01)
        self.assertAlmostEqual(result.distributions[1][self.forward], 0.99)
        step = result.steps[0]
        self.assertEqual(step.reported_confidence, 1.0)
        self.assertEqual(step.effective_confidence, 0.99)
        self.assertEqual(step.treatment, "soft_evidence")
        self.assertEqual(step.support_recovery_strategy, None)

    def test_perfect_confidence_performs_hard_update(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            {self.empty: 0.5, self.forward: 0.5},
            self.score,
            lambda selected: 1,
            query_pool=(query,),
            budgets=(1,),
            confidence=1.0,
            treatment="hard_constraint",
            rejuvenation_steps=0,
        )
        self.assertEqual(result.distributions[1], {self.forward: 1.0})
        self.assertAlmostEqual(
            result.steps[0].expected_information_gain, math.log(2.0)
        )
        self.assertEqual(result.support_recovery_count, 0)
        self.assertFalse(result.steps[0].support_recovered)
        self.assertEqual(result.steps[0].reported_confidence, 1.0)
        self.assertEqual(result.steps[0].effective_confidence, 1.0)
        self.assertEqual(result.steps[0].treatment, "hard_constraint")

    def test_perfect_confidence_recovers_missing_particle_support(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        with patch(
            "cd_llm.methods.active_query._rejuvenate",
            return_value=(self.forward, self.forward),
        ):
            result = active_query_smc(
                {self.empty: 0.5, self.reverse: 0.5},
                self.score,
                lambda selected: 1,
                query_pool=(query,),
                budgets=(1,),
                confidence=1.0,
                treatment="hard_constraint",
                rejuvenation_steps=1,
            )
        self.assertEqual(result.distributions[1], {self.forward: 1.0})
        self.assertEqual(result.support_recovery_count, 1)
        self.assertTrue(result.steps[0].support_recovered)

    def test_perfect_confidence_deterministically_projects_zero_support(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        common = dict(
            initial_particles=(self.empty, self.reverse),
            initial_weights=(0.5, 0.5),
            query_pool=(query,),
            budgets=(1,),
            confidence=1.0,
            treatment="hard_constraint",
            rejuvenation_steps=0,
            seed=31,
        )
        first = active_query_smc(None, self.score, lambda selected: 1, **common)
        second = active_query_smc(None, self.score, lambda selected: 1, **common)
        self.assertEqual(first.particles, second.particles)
        self.assertEqual(first.distributions, second.distributions)
        self.assertTrue(all(has_edge(graph, 0, 1) for graph in first.particles))
        step = first.steps[0]
        self.assertEqual(
            step.support_recovery_strategy,
            "deterministic-hard-constraint-projection",
        )
        self.assertEqual(step.support_recovery_projection_attempts, 2)
        self.assertEqual(step.support_recovery_projected_particles, 2)
        self.assertGreaterEqual(step.support_recovery_unique_graphs, 1)

    def test_zero_support_negative_adjacency_is_recovered_by_deletion(self) -> None:
        query = ClaimQuery("adjacency", 0, 1, "no-adjacency-0-1")
        result = active_query_smc(
            {self.forward: 0.5, self.reverse: 0.5},
            self.score,
            lambda selected: 0,
            query_pool=(query,),
            budgets=(1,),
            confidence=1.0,
            treatment="hard_constraint",
            rejuvenation_steps=2,
            seed=44,
        )
        self.assertEqual(result.support_recovery_count, 1)
        self.assertTrue(result.steps[0].support_recovered)
        self.assertEqual(
            result.steps[0].support_recovery_strategy,
            "deterministic-hard-constraint-projection",
        )
        self.assertTrue(
            all(not has_adjacency(graph, 0, 1) for graph in result.distributions[1])
        )

    def test_all_positive_mass_snapshots_preserve_every_hard_answer(self) -> None:
        truth = graph_from_edges(3, [(0, 1), (1, 2)])
        queries = (
            ClaimQuery("directed", 0, 1, "directed-0-1"),
            ClaimQuery("adjacency", 0, 1, "adjacency-0-1"),
            ClaimQuery("adjacency", 1, 2, "adjacency-1-2"),
        )

        def oracle(query: ClaimQuery) -> int:
            if query.kind == "directed":
                return int(has_edge(truth, query.source, query.target))
            return int(has_adjacency(truth, query.source, query.target))

        result = active_query_smc(
            {
                self.empty: 0.25,
                self.forward: 0.25,
                self.reverse: 0.25,
                truth: 0.25,
            },
            self.score,
            oracle,
            query_pool=queries,
            budgets=(1, 2, 3),
            confidence=1.0,
            treatment="hard_constraint",
            rejuvenation_steps=3,
            rejuvenate_after_every_query=True,
            seed=73,
        )
        for budget, distribution in result.distributions.items():
            acquired = result.steps[:budget]
            for graph, mass in distribution.items():
                self.assertGreater(mass, 0.0)
                self.assertTrue(
                    all(
                        int(query_values((graph,), step.query)[0]) == step.answer
                        for step in acquired
                    )
                )
        for graph, weight in zip(result.particles, result.weights):
            if weight > 0.0:
                self.assertTrue(
                    all(
                        int(query_values((graph,), step.query)[0]) == step.answer
                        for step in result.steps
                    )
                )

    def test_projection_preserves_all_claims_and_drops_blocking_edges(self) -> None:
        path = graph_from_edges(3, [(0, 1), (1, 2)])
        claims = (
            Claim("required", "I", "directed", 1, 0, 1),
            Claim("excluded", "I", "adjacency", 0, 1, 2),
        )
        outcome = _recover_hard_constraint_support(
            (path, path), claims, node_count=3, max_parents=2
        )
        self.assertEqual(outcome.projection_attempts, 2)
        self.assertEqual(outcome.projected_particles, 2)
        for graph in outcome.particles:
            self.assertTrue(has_edge(graph, 0, 1))
            self.assertFalse(has_adjacency(graph, 1, 2))

        cycle_blocked = graph_from_edges(3, [(1, 2), (2, 0)])
        addition = (Claim("addition", "I", "directed", 1, 0, 1),)
        repaired = _recover_hard_constraint_support(
            (cycle_blocked, cycle_blocked),
            addition,
            node_count=3,
            max_parents=2,
        )
        self.assertTrue(all(has_edge(graph, 0, 1) for graph in repaired.particles))
        self.assertTrue(all(not has_edge(graph, 2, 0) for graph in repaired.particles))

    def test_projection_respects_parent_cap_for_required_addition(self) -> None:
        capped = graph_from_edges(3, [(2, 1)])
        claims = (Claim("required", "I", "directed", 1, 0, 1),)
        outcome = _recover_hard_constraint_support(
            (capped, capped), claims, node_count=3, max_parents=1
        )
        for graph in outcome.particles:
            self.assertTrue(has_edge(graph, 0, 1))
            self.assertFalse(has_edge(graph, 2, 1))

    def test_projection_rejects_inconsistent_perfect_claims(self) -> None:
        claims = (
            Claim("forward", "I", "directed", 1, 0, 1),
            Claim("reverse", "I", "directed", 1, 1, 0),
        )
        with self.assertRaisesRegex(ValueError, "both directions"):
            _recover_hard_constraint_support(
                (self.empty, self.empty),
                claims,
                node_count=3,
                max_parents=2,
            )

    def test_mixed_projection_enforces_required_and_forbidden_ancestry(self) -> None:
        path = graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
        claims = (
            Claim("required-path", "II", "ancestry", 1, 0, 3),
            Claim("forbidden-path", "II", "ancestry", 0, 1, 3),
        )
        outcome = _recover_hard_constraint_support(
            (path, path), claims, node_count=4, max_parents=2
        )
        self.assertEqual(outcome.projected_particles, 2)
        for graph in outcome.particles:
            self.assertEqual(
                int(
                    query_values(
                        (graph,), ClaimQuery("ancestry", 0, 3, "required")
                    )[0]
                ),
                1,
            )
            self.assertEqual(
                int(
                    query_values(
                        (graph,), ClaimQuery("ancestry", 1, 3, "forbidden")
                    )[0]
                ),
                0,
            )

    def test_mixed_projection_uses_ordering_as_reverse_ancestry_complement(self) -> None:
        reverse_path = graph_from_edges(3, [(2, 1), (1, 0)])
        claims = (
            Claim("ordered", "II", "ordering", 1, 0, 2),
            Claim("not-ordered", "II", "ordering", 0, 0, 1),
        )
        outcome = _recover_hard_constraint_support(
            (reverse_path, reverse_path), claims, node_count=3, max_parents=2
        )
        for graph in outcome.particles:
            self.assertEqual(
                int(
                    query_values(
                        (graph,), ClaimQuery("ordering", 0, 2, "ordered")
                    )[0]
                ),
                1,
            )
            self.assertEqual(
                int(
                    query_values(
                        (graph,), ClaimQuery("ordering", 0, 1, "not-ordered")
                    )[0]
                ),
                0,
            )

    def test_linked_ancestry_ordering_complements_are_never_both_queried(self) -> None:
        queries = dag_query_pool(
            3, ("local_edge", "causal_order", "ancestry")
        )
        links = ancestry_ordering_query_links(queries)
        result = active_query_smc(
            None,
            self.score,
            lambda query: int(query_values((self.forward,), query)[0]),
            initial_particles=(self.empty, self.forward, self.reverse),
            initial_weights=(0.2, 0.3, 0.5),
            query_pool=queries,
            linked_query_ids=links,
            budgets=(15,),
            confidence=1.0,
            treatment="hard_constraint",
            policy="random",
            seed=18,
            rejuvenation_steps=0,
        )
        selected = {step.query.query_id for step in result.steps}
        self.assertEqual(len(selected), 15)
        self.assertTrue(all(links.get(query_id) not in selected for query_id in selected))

    def test_hard_rejuvenation_rejects_any_claim_violating_candidate(self) -> None:
        claims = (Claim("required", "I", "directed", 1, 0, 1),)
        evidence = _evidence_graph(claims, 1.0, ("hard_constraint",))
        with patch(
            "cd_llm.methods.active_query.EvidenceGuidedDAGProposal._sample_candidate",
            return_value=(self.empty, False, False),
        ):
            outcome = _rejuvenate(
                self.score,
                (self.forward, self.forward),
                claims,
                evidence,
                seed=19,
                steps=3,
                gamma=0.5,
                cache_size=32,
                hard_constraints=True,
            )
        self.assertEqual(outcome.particles, (self.forward, self.forward))
        self.assertEqual(outcome.diagnostics.attempted_moves, 6)
        self.assertEqual(outcome.diagnostics.accepted_moves, 0)
        self.assertEqual(outcome.diagnostics.hard_constraint_rejections, 6)

    def test_invalid_answer_is_rejected(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        with self.assertRaisesRegex(ValueError, "zero or one"):
            active_query_smc(
                self.distribution,
                self.score,
                lambda selected: 2,
                query_pool=(query,),
                budgets=(1,),
                rejuvenation_steps=0,
            )

    def test_resampling_resets_weights_and_retains_valid_dags(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            self.distribution,
            self.score,
            lambda selected: 1,
            query_pool=(query,),
            budgets=(1,),
            confidence=0.99,
            ess_threshold_fraction=1.0,
            rejuvenation_steps=1,
            seed=21,
        )
        self.assertEqual(result.resampling_count, 1)
        self.assertTrue(result.steps[0].resampled)
        np.testing.assert_allclose(result.weights, np.full(3, 1.0 / 3.0))
        self.assertTrue(all(graph.regime == "dag" for graph in result.particles))
        self.assertAlmostEqual(math.fsum(result.distributions[1].values()), 1.0)

    def test_random_policy_is_seeded(self) -> None:
        first = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            budgets=(3,),
            policy="random",
            seed=99,
            rejuvenation_steps=0,
        )
        second = active_query_smc(
            self.distribution,
            self.score,
            self.answer_from_truth,
            budgets=(3,),
            policy="random",
            seed=99,
            rejuvenation_steps=0,
        )
        self.assertEqual(
            tuple(step.query for step in first.steps),
            tuple(step.query for step in second.steps),
        )

    def test_explicit_particle_slots_retain_duplicates_and_graph_mass_diagnostics(
        self,
    ) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            None,
            self.score,
            lambda selected: 1,
            initial_particles=(self.empty, self.empty, self.forward),
            initial_weights=(0.2, 0.3, 0.5),
            query_pool=(query,),
            budgets=(1,),
            confidence=0.9,
            ess_threshold_fraction=0.1,
            rejuvenation_steps=0,
        )
        self.assertEqual(result.particles, (self.empty, self.empty, self.forward))
        np.testing.assert_allclose(result.weights, (0.04, 0.06, 0.9))
        self.assertAlmostEqual(result.distributions[1][self.empty], 0.1)
        self.assertAlmostEqual(result.distributions[1][self.forward], 0.9)
        step = result.steps[0]
        self.assertEqual(step.unique_positive_graphs_before, 2)
        self.assertEqual(step.unique_positive_graphs_after, 2)
        self.assertAlmostEqual(step.graph_mass_ess_before, 2.0)
        self.assertAlmostEqual(step.graph_mass_ess_after, 1.0 / (0.1**2 + 0.9**2))
        self.assertIsNone(step.unique_resampling_ancestors)
        self.assertEqual(result.query_count, 1)

    def test_explicit_population_inputs_are_unambiguous(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        with self.assertRaisesRegex(ValueError, "provided together"):
            active_query_smc(
                None,
                self.score,
                lambda selected: 0,
                initial_particles=(self.empty, self.forward),
                query_pool=(query,),
                budgets=(1,),
            )
        with self.assertRaisesRegex(ValueError, "either"):
            active_query_smc(
                self.distribution,
                self.score,
                lambda selected: 0,
                initial_particles=(self.empty, self.forward),
                initial_weights=(0.5, 0.5),
                query_pool=(query,),
                budgets=(1,),
            )

    def test_optimized_eig_matches_naive_selector_and_evaluates_unique_theta_once(
        self,
    ) -> None:
        queries = (
            ClaimQuery("directed", 0, 1, "directed-a"),
            ClaimQuery("directed", 0, 2, "directed-b"),
            ClaimQuery("adjacency", 0, 1, "adjacency-a"),
            ClaimQuery("adjacency", 0, 2, "adjacency-b"),
        )
        particles = (self.empty, self.forward, self.reverse)
        weights = np.asarray((0.2, 0.3, 0.5), dtype=float)
        probabilities = _query_truth_probabilities(particles, weights, queries, 3)
        unique_count = len({float(value) for value in probabilities})

        def naive(probabilities: np.ndarray, confidence: float) -> np.ndarray:
            return np.asarray(
                [
                    expected_information_gain(float(value), confidence)
                    for value in probabilities
                ],
                dtype=float,
            )

        common = dict(
            initial_particles=particles,
            initial_weights=weights,
            query_pool=queries,
            budgets=(1,),
            confidence=0.9,
            rejuvenation_steps=0,
        )
        with patch(
            "cd_llm.methods.active_query._information_gains", side_effect=naive
        ):
            reference = active_query_smc(
                None, self.score, self.answer_from_truth, **common
            )
        with patch(
            "cd_llm.methods.active_query.expected_information_gain",
            wraps=expected_information_gain,
        ) as scalar_eig:
            optimized = active_query_smc(
                None, self.score, self.answer_from_truth, **common
            )
        self.assertEqual(optimized.steps[0].query, reference.steps[0].query)
        self.assertEqual(
            optimized.steps[0].predicate_probability,
            reference.steps[0].predicate_probability,
        )
        self.assertEqual(
            optimized.steps[0].expected_information_gain,
            reference.steps[0].expected_information_gain,
        )
        self.assertEqual(scalar_eig.call_count, unique_count)

    def test_zero_eig_does_not_stop_and_ties_remain_lexicographic(self) -> None:
        queries = (
            ClaimQuery("directed", 1, 0, "z"),
            ClaimQuery("directed", 0, 1, "y"),
            ClaimQuery("adjacency", 0, 1, "x"),
        )
        result = active_query_smc(
            None,
            self.score,
            lambda selected: 0,
            initial_particles=(self.empty, self.empty),
            initial_weights=(0.5, 0.5),
            query_pool=queries,
            budgets=(3,),
            confidence=0.9,
            rejuvenation_steps=0,
        )
        self.assertEqual(result.query_count, 3)
        self.assertEqual(tuple(step.query for step in result.steps), tuple(sorted(queries)))
        self.assertTrue(
            all(step.expected_information_gain == 0.0 for step in result.steps)
        )

    def test_rejuvenation_can_run_a_fixed_number_of_moves_after_every_query(
        self,
    ) -> None:
        def identity_rejuvenation(score, particles, *args):
            return tuple(particles)

        common = dict(
            query_pool=group_i_query_pool(3),
            budgets=(3,),
            confidence=0.5,
            rejuvenation_steps=2,
            ess_threshold_fraction=0.1,
        )
        with patch(
            "cd_llm.methods.active_query._rejuvenate",
            side_effect=identity_rejuvenation,
        ) as rejuvenate:
            active_query_smc(
                self.distribution,
                self.score,
                self.answer_from_truth,
                rejuvenate_after_every_query=True,
                **common,
            )
            self.assertEqual(rejuvenate.call_count, 3)
            rejuvenate.reset_mock()
            active_query_smc(
                self.distribution,
                self.score,
                self.answer_from_truth,
                rejuvenate_after_every_query=False,
                **common,
            )
            self.assertEqual(rejuvenate.call_count, 1)

    def test_final_budget_rejuvenation_can_be_disabled_without_disabling_resample_moves(
        self,
    ) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        common = dict(
            query_pool=(query,), budgets=(1,), confidence=0.99,
            rejuvenation_steps=2,
        )
        with patch(
            "cd_llm.methods.active_query._rejuvenate",
            side_effect=lambda score, particles, *args: tuple(particles),
        ) as rejuvenate:
            active_query_smc(
                self.distribution, self.score, lambda selected: 1,
                ess_threshold_fraction=0.1, **common,
            )
            self.assertEqual(rejuvenate.call_count, 1)
            rejuvenate.reset_mock()
            no_final = active_query_smc(
                self.distribution, self.score, lambda selected: 1,
                ess_threshold_fraction=0.1,
                rejuvenate_at_final_budget=False, **common,
            )
            self.assertEqual(rejuvenate.call_count, 0)
            self.assertFalse(no_final.steps[0].resampled)
            rejuvenate.reset_mock()
            resampled = active_query_smc(
                self.distribution, self.score, lambda selected: 1,
                ess_threshold_fraction=1.0,
                rejuvenate_at_final_budget=False, **common,
            )
            self.assertEqual(rejuvenate.call_count, 1)
            self.assertTrue(resampled.steps[0].resampled)

    def test_rejuvenation_move_diagnostics_are_recorded_per_query(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            self.distribution,
            self.score,
            lambda selected: 1,
            query_pool=(query,),
            budgets=(1,),
            confidence=0.9,
            rejuvenation_steps=2,
            ess_threshold_fraction=0.1,
            seed=7,
        )
        step = result.steps[0]
        self.assertEqual(step.rejuvenation_attempted_moves, 6)
        self.assertLessEqual(
            step.rejuvenation_accepted_moves, step.rejuvenation_attempted_moves
        )
        self.assertLessEqual(
            step.rejuvenation_self_proposals, step.rejuvenation_attempted_moves
        )
        self.assertGreaterEqual(step.rejuvenation_acceptance_rate, 0.0)
        self.assertLessEqual(step.rejuvenation_acceptance_rate, 1.0)

    def test_balanced_rejuvenation_kernel_controls_are_forwarded(self) -> None:
        query = ClaimQuery("directed", 0, 1, "edge-0-1")
        result = active_query_smc(
            self.distribution,
            self.score,
            lambda selected: 1,
            query_pool=(query,),
            budgets=(1,),
            confidence=0.9,
            rejuvenation_steps=1,
            ess_threshold_fraction=0.1,
            ordinary_kernel="balanced",
            screened_additions=((0, 1), (1, 2), (2, 0)),
            seed=8,
        )
        self.assertEqual(result.steps[0].rejuvenation_attempted_moves, 3)
        self.assertTrue(all(graph.regime == "dag" for graph in result.particles))


if __name__ == "__main__":
    unittest.main()
