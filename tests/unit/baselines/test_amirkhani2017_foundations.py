from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from scipy.integrate import dblquad, quad

from cd_llm.baselines.amirkhani2017.accuracy import AccuracyPrior, ExpertAccuracy
from cd_llm.baselines.amirkhani2017.em import (
    AccuracyEMConfig,
    estimate_accuracies_em,
)
from cd_llm.baselines.amirkhani2017.expert_scores import (
    explicit_expert_log_score,
    marginal_expert_log_score,
    opinion_probability,
)
from cd_llm.baselines.amirkhani2017.opinions import (
    EdgeState,
    ExpertOpinion,
    normalize_opinions,
)
from cd_llm.core.finite_support import graph_from_edges


class OpinionAndAccuracyTests(unittest.TestCase):
    def test_opinions_are_canonical_immutable_and_direction_preserving(self) -> None:
        opinion = ExpertOpinion.directed("expert", 3, 1)
        self.assertEqual((opinion.left, opinion.right), (1, 3))
        self.assertEqual(opinion.state, EdgeState.RIGHT_TO_LEFT)
        self.assertEqual(opinion.directed_edge, (3, 1))
        direct = ExpertOpinion("expert", 4, 2, EdgeState.LEFT_TO_RIGHT)
        self.assertEqual(direct, ExpertOpinion.directed("expert", 4, 2))
        with self.assertRaises(FrozenInstanceError):
            opinion.left = 0  # type: ignore[misc]
        with self.assertRaises(ValueError):
            ExpertOpinion.absent("expert", 1, 1)

    def test_duplicate_expert_pair_responses_are_always_rejected(self) -> None:
        first = ExpertOpinion.directed("expert", 0, 1)
        with self.assertRaisesRegex(ValueError, "at most one"):
            normalize_opinions((first, first))
        with self.assertRaisesRegex(ValueError, "at most one"):
            normalize_opinions(
                (first, ExpertOpinion.absent("expert", 0, 1))
            )
        normalized = normalize_opinions(
            (first, ExpertOpinion.absent("other", 0, 1)), node_count=2
        )
        self.assertEqual(len(normalized), 2)

    def test_accuracy_domain_and_paper_prior_parameterization(self) -> None:
        accuracy = ExpertAccuracy(0.6, 0.1, 0.8)
        self.assertAlmostEqual(accuracy.existing_as_absent, 0.3)
        self.assertAlmostEqual(accuracy.absent_as_each_direction, 0.1)
        self.assertEqual(ExpertAccuracy(1.0, 0.0, 0.0).gamma1, 1.0)
        rounded = ExpertAccuracy(0.8, 0.2000000000000004, 0.5)
        self.assertEqual(rounded.gamma1 + rounded.gamma2, 1.0)
        self.assertEqual(rounded.existing_as_absent, 0.0)
        nearly_boundary = ExpertAccuracy(0.8, 0.1999999999999995, 0.5)
        self.assertGreater(nearly_boundary.existing_as_absent, 0.0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            ExpertAccuracy(0.8, 0.3, 0.5)
        self.assertEqual(AccuracyPrior.neutral(), AccuracyPrior(1, 1, 1, 1, 1))
        prior = AccuracyPrior.from_mean(accuracy)
        for actual, expected in zip(
            (
                prior.alpha1,
                prior.alpha2,
                prior.alpha3,
                prior.beta1,
                prior.beta2,
            ),
            (6.0, 1.0, 3.0, 8.0, 2.0),
        ):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            AccuracyPrior.from_mean(ExpertAccuracy(1.0, 0.0, 1.0))


class ExpertScoreTests(unittest.TestCase):
    def test_published_confusion_matrix_is_exact_and_normalized(self) -> None:
        accuracy = ExpertAccuracy(0.6, 0.1, 0.8)
        expected = (
            (0.6, 0.1, 0.3),
            (0.1, 0.6, 0.3),
            (0.1, 0.1, 0.8),
        )
        states = tuple(EdgeState)
        for truth, row in zip(states, expected):
            observed = tuple(
                opinion_probability(truth, state, accuracy) for state in states
            )
            for actual, target in zip(observed, row):
                self.assertAlmostEqual(actual, target)
            self.assertAlmostEqual(sum(observed), 1.0)

    def test_explicit_score_matches_hand_product_and_missing_is_ignored(self) -> None:
        graph = graph_from_edges(3, [(0, 1)])
        opinions = (
            ExpertOpinion.directed("e", 0, 1),
            ExpertOpinion.absent("e", 0, 2),
            ExpertOpinion.directed("e", 2, 1),
        )
        score = explicit_expert_log_score(
            graph, opinions, {"e": ExpertAccuracy(0.6, 0.1, 0.8)}
        )
        self.assertAlmostEqual(score, math.log(0.6 * 0.8 * 0.1))
        self.assertEqual(explicit_expert_log_score(graph, (), {}), 0.0)
        with self.assertRaisesRegex(ValueError, "missing accuracy"):
            explicit_expert_log_score(graph, opinions, {})
        impossible = explicit_expert_log_score(
            graph,
            (ExpertOpinion.absent("e", 0, 1),),
            {"e": ExpertAccuracy(0.7, 0.3, 0.5)},
        )
        self.assertEqual(impossible, -math.inf)

    def test_marginal_score_matches_numerical_integration(self) -> None:
        graph = graph_from_edges(4, [(0, 1), (2, 0), (1, 3)])
        opinions = (
            ExpertOpinion.directed("e", 0, 1),  # n1
            ExpertOpinion.directed("e", 0, 2),  # n2
            ExpertOpinion.absent("e", 1, 3),  # n3
            ExpertOpinion.absent("e", 0, 3),  # m1
            ExpertOpinion.directed("e", 2, 3),  # m2
        )
        score = marginal_expert_log_score(graph, opinions)
        # Uniform Dirichlet density is 2 and uniform Beta density is 1.
        direction_integral = dblquad(
            lambda gamma2, gamma1: 2.0
            * gamma1
            * gamma2
            * (1.0 - gamma1 - gamma2),
            0.0,
            1.0,
            lambda gamma1: 0.0,
            lambda gamma1: 1.0 - gamma1,
        )[0]
        absence_integral = quad(
            lambda gamma3: gamma3 * (1.0 - gamma3) / 2.0,
            0.0,
            1.0,
        )[0]
        self.assertAlmostEqual(
            score, math.log(direction_integral * absence_integral), places=10
        )
        self.assertEqual(marginal_expert_log_score(graph, ()), 0.0)

    def test_marginal_priors_are_per_expert_or_shared(self) -> None:
        graph = graph_from_edges(2, [(0, 1)])
        opinions = (
            ExpertOpinion.directed("a", 0, 1),
            ExpertOpinion.directed("b", 0, 1),
        )
        shared = AccuracyPrior.from_mean(ExpertAccuracy(0.6, 0.1, 0.8))
        self.assertAlmostEqual(
            marginal_expert_log_score(graph, opinions, shared),
            marginal_expert_log_score(
                graph, opinions, {"a": shared, "b": shared}
            ),
        )
        with self.assertRaisesRegex(ValueError, "missing accuracy prior"):
            marginal_expert_log_score(graph, opinions, {"a": shared})


class AccuracyEMTests(unittest.TestCase):
    def test_em_has_normalized_responsibilities_and_monotone_likelihood(self) -> None:
        opinions = (
            ExpertOpinion.directed("a", 0, 1),
            ExpertOpinion.directed("b", 0, 1),
            ExpertOpinion.absent("a", 0, 2),
            ExpertOpinion.absent("b", 0, 2),
            ExpertOpinion.directed("a", 2, 1),
            ExpertOpinion.absent("b", 1, 2),
        )
        result = estimate_accuracies_em(
            opinions,
            node_count=3,
            config=AccuracyEMConfig(tolerance=1e-7, max_iterations=500),
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(sum(result.edge_prior), 1.0)
        for _, _, posterior in result.pair_posteriors:
            self.assertAlmostEqual(sum(posterior), 1.0)
            self.assertTrue(all(value >= 0.0 for value in posterior))
        self.assertTrue(
            all(
                current + 1e-10 >= previous
                for previous, current in zip(
                    result.log_likelihoods, result.log_likelihoods[1:]
                )
            )
        )
        self.assertLess(result.max_parameter_change, 1e-7)

    def test_silent_expert_parameters_are_retained_and_flagged(self) -> None:
        result = estimate_accuracies_em(
            (), node_count=3, expert_ids=("silent",)
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        for actual, expected in zip(result.edge_prior, (0.1, 0.1, 0.8)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result.accuracies["silent"], ExpertAccuracy.neutral())
        self.assertEqual(
            result.unidentified_parameters,
            (
                ("silent", "gamma1"),
                ("silent", "gamma2"),
                ("silent", "gamma3"),
            ),
        )

    def test_zero_existing_prior_retains_unidentifiable_gamma1_gamma2(self) -> None:
        result = estimate_accuracies_em(
            (ExpertOpinion.directed("e", 0, 1),),
            node_count=2,
            initial_edge_prior=(0.0, 0.0, 1.0),
            config=AccuracyEMConfig(max_iterations=2),
        )
        self.assertIn(("e", "gamma1"), result.unidentified_parameters)
        self.assertIn(("e", "gamma2"), result.unidentified_parameters)
        self.assertEqual(result.accuracies["e"].gamma1, 1.0 / 3.0)
        self.assertEqual(result.accuracies["e"].gamma2, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
