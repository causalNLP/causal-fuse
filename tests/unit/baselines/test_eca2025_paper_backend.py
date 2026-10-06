from __future__ import annotations

import math
import unittest

import numpy as np

from cd_llm.baselines.eca2025 import (
    ECAConfig,
    ECAPrior,
    acyclicity_and_gradient,
    data_loss_and_gradient,
    fit_eca,
    objective_and_subgradient,
    objective_terms,
    prior_log_score_and_gradient,
    smooth_objective_and_gradient,
)
from cd_llm.baselines.eca2025.paper_backend import fit_paper_eca
from cd_llm.core.finite_support import graph_from_edges


def _finite_difference(
    function,
    weights: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    result = np.empty_like(weights, dtype=float)
    for index in np.ndindex(weights.shape):
        positive = weights.copy()
        negative = weights.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        result[index] = (function(positive) - function(negative)) / (2.0 * epsilon)
    return result


class PaperObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.asarray(
            [
                [0.3, -1.2, 0.7],
                [1.1, 0.2, -0.4],
                [-0.8, 1.4, 0.6],
                [0.5, -0.1, 1.3],
            ]
        )
        self.weights = np.asarray(
            [
                [0.0, 0.27, -0.41],
                [0.33, 0.0, 0.19],
                [-0.22, 0.44, 0.0],
            ]
        )
        self.mask = np.asarray(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )
        self.probabilities = np.asarray(
            [
                [0.5, 0.8, 0.5],
                [0.25, 0.5, 0.65],
                [0.5, 0.9, 0.5],
            ]
        )

    def test_data_loss_gradient_matches_central_difference(self) -> None:
        value, gradient = data_loss_and_gradient(self.data, self.weights)
        expected_value = 0.5 * np.sum(
            (self.data - self.data @ self.weights) ** 2
        ) / len(self.data)
        self.assertAlmostEqual(value, expected_value, places=14)
        numerical = _finite_difference(
            lambda weights: data_loss_and_gradient(self.data, weights)[0],
            self.weights,
        )
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)

    def test_prior_score_gradient_matches_equation_and_difference(self) -> None:
        score, gradient = prior_log_score_and_gradient(
            self.weights, self.mask, self.probabilities
        )
        presence = np.abs(np.tanh(0.5 * self.weights[self.mask.astype(bool)]))
        probabilities = self.probabilities[self.mask.astype(bool)]
        expected = np.sum(
            np.log((1.0 - probabilities) + (2.0 * probabilities - 1.0) * presence)
        )
        self.assertAlmostEqual(score, expected, places=14)
        numerical = _finite_difference(
            lambda weights: prior_log_score_and_gradient(
                weights, self.mask, self.probabilities
            )[0],
            self.weights,
        )
        np.testing.assert_allclose(gradient, numerical, rtol=3e-6, atol=2e-7)
        self.assertTrue(np.all(gradient[self.mask == 0] == 0.0))

    def test_exact_notears_acyclicity_gradient_matches_difference(self) -> None:
        value, gradient = acyclicity_and_gradient(self.weights)
        self.assertGreater(value, 0.0)
        numerical = _finite_difference(
            lambda weights: acyclicity_and_gradient(weights)[0], self.weights
        )
        np.testing.assert_allclose(gradient, numerical, rtol=3e-6, atol=3e-7)
        dag = np.asarray([[0.0, 0.8, 0.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0]])
        dag_value, _ = acyclicity_and_gradient(dag)
        self.assertAlmostEqual(dag_value, 0.0, places=14)

    def test_complete_objective_subgradient_matches_off_diagonal_difference(self) -> None:
        value, gradient = objective_and_subgradient(
            self.data,
            self.weights,
            self.mask,
            self.probabilities,
            lambda1=0.07,
            sigma=1.3,
        )
        terms = objective_terms(
            self.data,
            self.weights,
            self.mask,
            self.probabilities,
            lambda1=0.07,
            sigma=1.3,
        )
        self.assertAlmostEqual(value, terms.objective, places=14)
        self.assertAlmostEqual(
            terms.objective,
            terms.data_loss - 1.3**2 * terms.prior_log_score + terms.l1_penalty,
            places=14,
        )
        numerical = _finite_difference(
            lambda weights: objective_and_subgradient(
                self.data,
                weights,
                self.mask,
                self.probabilities,
                lambda1=0.07,
                sigma=1.3,
            )[0],
            self.weights,
        )
        off_diagonal = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(
            gradient[off_diagonal],
            numerical[off_diagonal],
            rtol=4e-6,
            atol=4e-7,
        )

    def test_neutral_constraints_are_constant_and_have_no_optimization_gradient(self) -> None:
        neutral_probabilities = np.full((3, 3), 0.5)
        neutral_score, neutral_gradient = prior_log_score_and_gradient(
            self.weights, self.mask, neutral_probabilities
        )
        self.assertAlmostEqual(
            neutral_score, int(np.count_nonzero(self.mask)) * math.log(0.5)
        )
        np.testing.assert_array_equal(neutral_gradient, np.zeros((3, 3)))
        _, constrained_gradient = smooth_objective_and_gradient(
            self.data,
            self.weights,
            self.mask,
            neutral_probabilities,
            sigma=2.0,
        )
        _, empty_gradient = smooth_objective_and_gradient(
            self.data,
            self.weights,
            np.zeros((3, 3), dtype=int),
            neutral_probabilities,
            sigma=2.0,
        )
        np.testing.assert_allclose(constrained_gradient, empty_gradient)

    def test_numerical_boundary_probabilities_remain_finite(self) -> None:
        weights = np.zeros((2, 2))
        mask = np.asarray([[0, 1], [1, 0]])
        probabilities = np.asarray([[0.5, 1.0], [0.0, 0.5]])
        score, gradient = prior_log_score_and_gradient(weights, mask, probabilities)
        self.assertTrue(math.isfinite(score))
        self.assertTrue(np.all(np.isfinite(gradient)))


class PaperFitTests(unittest.TestCase):
    @staticmethod
    def _tiny_sem() -> np.ndarray:
        rng = np.random.default_rng(280)
        source = rng.normal(size=256)
        target = 1.8 * source + 0.2 * rng.normal(size=256)
        return np.column_stack((source, target))

    def test_tiny_linear_sem_fit_is_deterministic_and_recovers_the_edge(self) -> None:
        data = self._tiny_sem()
        config = ECAConfig(max_iterations=30, threshold=0.2)
        first = fit_eca(data, config=config)
        second = fit_eca(data, config=config)
        expected_graph = graph_from_edges(2, [(0, 1)])
        self.assertEqual(first.graph, expected_graph)
        self.assertEqual(second.graph, expected_graph)
        np.testing.assert_array_equal(
            first.binary_adjacency, np.asarray([[0, 1], [0, 0]])
        )
        np.testing.assert_array_equal(first.weighted_adjacency, second.weighted_adjacency)
        self.assertEqual(first.objective, second.objective)
        self.assertTrue(first.converged)
        self.assertLessEqual(first.diagnostics.acyclicity, config.h_tolerance)
        self.assertEqual(first.support, {expected_graph: 1.0})
        self.assertEqual(first.prior, ECAPrior.empty(2))

    def test_existence_prior_escapes_the_nonsmooth_zero_stationary_point(self) -> None:
        data = np.asarray(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        )
        mask = np.asarray([[0, 1], [0, 0]])
        probabilities = np.full((2, 2), 0.5)
        probabilities[0, 1] = 0.9
        zero_terms = objective_terms(
            data,
            np.zeros((2, 2)),
            mask,
            probabilities,
            lambda1=0.05,
            sigma=1.0,
        )
        result = fit_paper_eca(
            data,
            mask,
            probabilities,
            lambda1=0.05,
            sigma=1.0,
            threshold=0.01,
            max_outer_iterations=30,
        )
        self.assertLess(result.terms.objective, zero_terms.objective - 0.1)
        self.assertGreater(abs(result.weighted_adjacency[0, 1]), 0.01)
        self.assertEqual(result.graph, graph_from_edges(2, [(0, 1)]))

    def test_fit_validates_matrix_and_hyperparameter_domains(self) -> None:
        data = self._tiny_sem()[:8]
        probabilities = np.full((2, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_paper_eca(data, np.zeros((3, 3)), np.full((3, 3), 0.5))
        with self.assertRaisesRegex(ValueError, "self-edge"):
            fit_paper_eca(data, np.eye(2), probabilities)
        with self.assertRaisesRegex(ValueError, "zero and one"):
            fit_paper_eca(
                data,
                np.asarray([[0, 2], [0, 0]]),
                probabilities,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_paper_eca(data, np.zeros((2, 2)), probabilities, sigma=0.0)
        with self.assertRaisesRegex(ValueError, "less than one"):
            fit_paper_eca(
                data,
                np.zeros((2, 2)),
                probabilities,
                progress_rate=1.0,
            )


if __name__ == "__main__":
    unittest.main()
