from __future__ import annotations

import itertools
import math
import unittest

from cd_llm.baselines.camml2006 import (
    CaMMLPrior,
    StructurePriorScorer,
    TOM,
    closest_topological_order,
    dag_edit_distance,
    global_pair_prior,
    global_relation_index,
    kendall_distance,
    local_pair_prior,
    skeleton_edit_distance,
    tom_edit_distance,
)
from cd_llm.core.finite_support import graph_from_edges


class LocalPriorTests(unittest.TestCase):
    def test_paper_figure_two_table(self) -> None:
        self.assertEqual(
            local_pair_prior(0, 1, left_to_right=0.7, right_to_left=0.2).probabilities,
            (0.7, 0.2, 0.050000000000000044, 0.050000000000000044),
        )
        self.assertEqual(
            local_pair_prior(0, 2, adjacency=0.6).probabilities,
            (0.3, 0.3, 0.2, 0.2),
        )
        self.assertEqual(local_pair_prior(1, 2).probabilities, (0.25,) * 4)

    def test_paper_example_cost(self) -> None:
        prior = CaMMLPrior(
            3,
            (
                local_pair_prior(0, 1, left_to_right=0.7, right_to_left=0.2),
                local_pair_prior(0, 2, adjacency=0.6),
            ),
        )
        tom = TOM((0, 2, 1), graph_from_edges(3, ((0, 1),)))
        score = StructurePriorScorer(prior, arc_probability=0.5).score(tom)
        self.assertAlmostEqual(score, math.log(0.7) + math.log(0.2) + math.log(0.25))

    def test_inconsistent_marginals_rejected(self) -> None:
        with self.assertRaises(ValueError):
            local_pair_prior(0, 1, left_to_right=0.7, adjacency=0.4)
        with self.assertRaises(ValueError):
            local_pair_prior(0, 1, left_to_right=0.7, left_before_right=0.6)

    def test_hard_local_prior(self) -> None:
        prior = CaMMLPrior(2, (local_pair_prior(0, 1, left_to_right=1.0),))
        scorer = StructurePriorScorer(prior, arc_probability=0.5)
        self.assertEqual(scorer.score(TOM((0, 1), graph_from_edges(2, ((0, 1),)))), 0.0)
        self.assertEqual(scorer.score(TOM((1, 0), graph_from_edges(2))), -math.inf)


class GlobalAndDistanceTests(unittest.TestCase):
    def test_global_states(self) -> None:
        self.assertEqual(global_relation_index(graph_from_edges(3, ((0, 1),)), 0, 1), 0)
        self.assertEqual(global_relation_index(graph_from_edges(3, ((1, 0),)), 0, 1), 1)
        self.assertEqual(
            global_relation_index(graph_from_edges(3, ((2, 0), (2, 1))), 0, 1), 2
        )
        self.assertEqual(global_relation_index(graph_from_edges(3), 0, 1), 3)

    def test_global_completion(self) -> None:
        prior = global_pair_prior(
            0, 1, left_ancestor_right=0.7, defaults=(0.2, 0.2, 0.2, 0.4)
        )
        self.assertAlmostEqual(prior.probabilities[0], 0.7)
        self.assertAlmostEqual(sum(prior.probabilities), 1.0)

    def test_edit_distances_count_reversal_as_two(self) -> None:
        first = graph_from_edges(3, ((0, 1), (1, 2)))
        second = graph_from_edges(3, ((1, 0), (1, 2), (0, 2)))
        self.assertEqual(dag_edit_distance(first, second), 3)
        self.assertEqual(skeleton_edit_distance(first, second), 1)

    def test_closest_order_matches_exhaustive_small_graph(self) -> None:
        graph = graph_from_edges(5, ((0, 3), (1, 3), (1, 4), (2, 4)))
        reference = (4, 3, 2, 1, 0)
        actual = closest_topological_order(graph, reference)
        valid = []
        for order in itertools.permutations(range(5)):
            position = {node: index for index, node in enumerate(order)}
            if all(position[s] < position[t] for s, t in ((0, 3), (1, 3), (1, 4), (2, 4))):
                valid.append(order)
        optimum = min(kendall_distance(reference, order) for order in valid)
        self.assertEqual(kendall_distance(reference, actual), optimum)

    def test_tom_distance(self) -> None:
        expert = graph_from_edges(3, ((0, 1), (1, 2)))
        candidate = TOM((1, 0, 2), graph_from_edges(3, ((1, 2), (0, 2))))
        self.assertEqual(tom_edit_distance(expert, candidate), 3)


if __name__ == "__main__":
    unittest.main()
