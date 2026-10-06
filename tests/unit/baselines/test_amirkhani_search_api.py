from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from cd_llm.baselines.amirkhani2017 import (
    AmirkhaniConfig,
    BDeuLocalScore,
    ExpertAccuracy,
    ExpertOpinion,
    GaussianBICLocalScore,
    fit_amirkhani,
    fit_amirkhani_from_local_score,
)
from cd_llm.baselines.amirkhani2017.search import _legal_moves
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.core.finite_support import Graph, has_edge


class TableLocalScore:
    name = "table"

    def __init__(
        self,
        node_count: int,
        values: dict[tuple[int, int], float] | None = None,
        *,
        default: float = 0.0,
    ) -> None:
        self.node_count = node_count
        self.values = {} if values is None else dict(values)
        self.default = default

    def local_score(self, node: int, parent_mask: int) -> float:
        return self.values.get((node, parent_mask), self.default)


class DataScoreTests(unittest.TestCase):
    def test_bdeu_matches_hand_implemented_empty_parent_score(self) -> None:
        data = np.asarray([[0], [0], [1], [1], [1]], dtype=int)
        score = BDeuLocalScore(data, equivalent_sample_size=1.0)
        # Dirichlet-multinomial with alpha=(1/2,1/2), counts=(2,3).
        expected = (
            math.lgamma(1.0)
            - math.lgamma(6.0)
            + math.lgamma(2.5)
            - math.lgamma(0.5)
            + math.lgamma(3.5)
            - math.lgamma(0.5)
        )
        self.assertAlmostEqual(score.local_score(0, 0), expected)
        self.assertEqual(score.cache_size, 1)
        self.assertAlmostEqual(score.local_score(0, 0), expected)
        self.assertEqual(score.cache_size, 1)

    def test_bdeu_rejects_missing_and_invalid_explicit_codes(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            BDeuLocalScore([[0, 1], [1, np.nan]])
        with self.assertRaisesRegex(ValueError, "outside"):
            BDeuLocalScore([[0, 2], [1, 0]], cardinalities=(2, 2))

    def test_gaussian_bic_wrapper_matches_repository_scorer(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(size=(60, 3))
        wrapped = GaussianBICLocalScore(data)
        reference = GaussianBICScore(data)
        for node in range(3):
            for mask in range(1 << 3):
                if mask & (1 << node):
                    continue
                self.assertAlmostEqual(
                    wrapped.local_score(node, mask),
                    reference.local_score(node, mask),
                    places=12,
                )

    def test_gaussian_bic_supports_a_single_nonconstant_variable(self) -> None:
        score = GaussianBICLocalScore([[0.0], [1.0], [2.0]])
        expected = -0.5 * 3.0 - 0.5 * math.log(3.0)
        self.assertAlmostEqual(score.local_score(0, 0), expected)


class GreedySearchTests(unittest.TestCase):
    def test_config_rejects_boolean_tolerances(self) -> None:
        with self.assertRaises(TypeError):
            AmirkhaniConfig(improvement_tolerance=True)
        with self.assertRaises(TypeError):
            AmirkhaniConfig(em_tolerance=False)

    def test_addition_ties_use_source_then_target_order(self) -> None:
        # Every node receives the same gain from acquiring its first parent.
        values = {
            (node, mask): (1.0 if mask else 0.0)
            for node in range(3)
            for mask in range(1 << 3)
            if not mask & (1 << node)
        }
        result = fit_amirkhani_from_local_score(TableLocalScore(3, values))
        self.assertGreaterEqual(len(result.moves), 1)
        self.assertEqual(
            (result.moves[0].kind, result.moves[0].source, result.moves[0].target),
            ("add", 0, 1),
        )

    def test_cycle_creating_move_is_never_considered(self) -> None:
        values: dict[tuple[int, int], float] = {}
        preferred = {(0, 1), (1, 2), (2, 0)}
        for child in range(3):
            for mask in range(1 << 3):
                if mask & (1 << child):
                    continue
                values[(child, mask)] = sum(
                    5.0
                    for parent in range(3)
                    if mask & (1 << parent) and (parent, child) in preferred
                )
        result = fit_amirkhani_from_local_score(TableLocalScore(3, values))
        selected = {
            (source, target)
            for source, row in enumerate(result.graph.directed)
            for target in range(3)
            if row & (1 << target)
        }
        self.assertEqual(len(selected & preferred), 2)
        self.assertIsInstance(result.graph, Graph)  # constructor validates acyclicity

    def test_parent_cap_is_enforced(self) -> None:
        values = {
            (node, mask): float(mask.bit_count())
            for node in range(4)
            for mask in range(1 << 4)
            if not mask & (1 << node)
        }
        result = fit_amirkhani_from_local_score(
            TableLocalScore(4, values),
            config=AmirkhaniConfig(max_parents=1),
        )
        self.assertTrue(all(mask.bit_count() <= 1 for mask in result.parent_masks))

    def test_result_is_a_strict_local_optimum(self) -> None:
        values = {
            (node, mask): 2.0 * mask.bit_count() - 0.4 * (mask.bit_count() ** 2)
            for node in range(3)
            for mask in range(1 << 3)
            if not mask & (1 << node)
        }
        score = TableLocalScore(3, values)
        result = fit_amirkhani_from_local_score(score)

        def graph_score(graph: Graph) -> float:
            parents = [0] * 3
            for source, row in enumerate(graph.directed):
                for target in range(3):
                    if row & (1 << target):
                        parents[target] |= 1 << source
            return sum(score.local_score(node, mask) for node, mask in enumerate(parents))

        for move in _legal_moves(result.graph, 2):
            self.assertLessEqual(graph_score(move.graph), result.total_log_score + 1e-12)

    def test_empty_opinions_reduce_to_data_only_search(self) -> None:
        values = {(1, 1): 3.0, (0, 0): 0.0, (1, 0): 0.0}
        result = fit_amirkhani_from_local_score(TableLocalScore(2, values))
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertEqual(result.expert_log_score, 0.0)
        self.assertAlmostEqual(result.total_log_score, result.data_log_score)


class PublicAPIModeTests(unittest.TestCase):
    def test_marginal_mode_uses_neutral_prior_by_default(self) -> None:
        result = fit_amirkhani_from_local_score(
            TableLocalScore(2),
            [ExpertOpinion.directed("e", 0, 1)],
        )
        self.assertEqual(result.expert_score_name, "marginal")
        self.assertTrue(has_edge(result.graph, 0, 1))

    def test_explicit_mode_can_escape_impossible_empty_dag(self) -> None:
        result = fit_amirkhani_from_local_score(
            TableLocalScore(2),
            [ExpertOpinion.directed("perfect", 0, 1)],
            config=AmirkhaniConfig(expert_score="explicit"),
            accuracies={"perfect": ExpertAccuracy(1.0, 0.0, 1.0)},
        )
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertEqual(result.expert_log_score, 0.0)
        self.assertTrue(math.isinf(result.moves[0].improvement))

    def test_explicit_mode_accepts_one_broadcast_accuracy(self) -> None:
        result = fit_amirkhani_from_local_score(
            TableLocalScore(2),
            [ExpertOpinion.directed("shared", 0, 1)],
            config=AmirkhaniConfig(expert_score="explicit"),
            accuracies=ExpertAccuracy(0.8, 0.1, 0.8),
        )
        self.assertTrue(has_edge(result.graph, 0, 1))

    def test_explicit_mode_reports_incompatible_perfect_opinions(self) -> None:
        opinions = (
            ExpertOpinion.directed("left", 0, 1),
            ExpertOpinion.directed("right", 1, 0),
        )
        accuracies = {
            "left": ExpertAccuracy(1.0, 0.0, 1.0),
            "right": ExpertAccuracy(1.0, 0.0, 1.0),
        }
        with self.assertRaisesRegex(ValueError, "zero-expert-likelihood plateau"):
            fit_amirkhani_from_local_score(
                TableLocalScore(2),
                opinions,
                config=AmirkhaniConfig(expert_score="explicit"),
                accuracies=accuracies,
            )

    def test_strict_search_reports_multi_edge_boundary_plateau(self) -> None:
        opinions = (
            ExpertOpinion.directed("perfect", 0, 1),
            ExpertOpinion.directed("perfect", 1, 2),
        )
        # The two-edge chain has likelihood one, but every zero- or one-edge
        # graph has likelihood zero. Strict hill climbing correctly stays put.
        with self.assertRaisesRegex(ValueError, "multiple simultaneous corrections"):
            fit_amirkhani_from_local_score(
                TableLocalScore(3),
                opinions,
                config=AmirkhaniConfig(expert_score="explicit"),
                accuracies={"perfect": ExpertAccuracy(1.0, 0.0, 1.0)},
            )

    def test_explicit_requires_accuracy_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires accuracies"):
            fit_amirkhani_from_local_score(
                TableLocalScore(2),
                config=AmirkhaniConfig(expert_score="explicit"),
            )

    def test_em_mode_returns_estimates_and_diagnostics(self) -> None:
        values = {(0, 0): 0.0, (1, 0): 0.0, (1, 1): 5.0}
        result = fit_amirkhani_from_local_score(
            TableLocalScore(2, values),
            [ExpertOpinion.directed("e", 0, 1)],
            config=AmirkhaniConfig(expert_score="em"),
        )
        self.assertIsNotNone(result.em_result)
        self.assertIsNotNone(result.estimated_accuracies)
        assert result.em_result is not None
        self.assertIn("e", result.em_result.accuracies)
        self.assertTrue(has_edge(result.graph, 0, 1))

    def test_raw_discrete_and_continuous_entrypoints(self) -> None:
        discrete = np.asarray(tuple(itertools.product((0, 1), repeat=2)))
        discrete_result = fit_amirkhani(discrete)
        self.assertEqual(discrete_result.data_score_name, "bdeu")

        rng = np.random.default_rng(8)
        continuous = rng.normal(size=(30, 2))
        continuous_result = fit_amirkhani(
            continuous,
            data_score="gaussian-bic",
        )
        self.assertEqual(continuous_result.data_score_name, "gaussian-bic")

    def test_invalid_cross_mode_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not apply"):
            fit_amirkhani_from_local_score(
                TableLocalScore(2),
                config=AmirkhaniConfig(expert_score="em"),
                accuracies={"e": ExpertAccuracy.neutral()},
            )
        with self.assertRaisesRegex(ValueError, "applies only to EM"):
            fit_amirkhani_from_local_score(
                TableLocalScore(2),
                initial_edge_prior=(0.2, 0.2, 0.6),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
