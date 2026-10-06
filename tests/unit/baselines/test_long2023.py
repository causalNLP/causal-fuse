from __future__ import annotations

import unittest

import numpy as np

from cd_llm.baselines.long2023 import ambiguous_pairs, prompt_for, run_long
from cd_llm.core.finite_support import Endpoint, PartialGraph
from cd_llm.experiments.neuropathic_long2023 import full_variable_name
from cd_llm.methods.adaptive_pruning import CompletionEnumerationLimitError


def undirected_two_node_cpdag() -> PartialGraph:
    matrix = np.zeros((2, 2), dtype=int)
    matrix[0, 1] = int(Endpoint.TAIL)
    matrix[1, 0] = int(Endpoint.TAIL)
    return PartialGraph.from_matrix(matrix, "cpdag")


class Long2023Tests(unittest.TestCase):
    def test_both_strategies_orient_two_node_mec(self) -> None:
        for strategy in ("srisk", "ssize"):
            with self.subTest(strategy=strategy):
                calls = []

                def expert(first, second, first_name, second_name):
                    calls.append((first, second, first_name, second_name))
                    return (first, second), 0.9, {}

                result = run_long(
                    undirected_two_node_cpdag(),
                    ("cause", "effect"),
                    expert,
                    strategy=strategy,
                    tolerance=0.2,
                )
                self.assertEqual(len(result.initial_mec), 2)
                self.assertEqual(len(result.final_mec), 1)
                self.assertEqual(result.decisions, ((0, 1),))
                self.assertEqual(calls, [(0, 1, "cause", "effect")])

    def test_preflight_limit_happens_before_expert_call(self) -> None:
        called = False

        def expert(*_args):
            nonlocal called
            called = True
            raise AssertionError("expert must not be called")

        with self.assertRaises(CompletionEnumerationLimitError):
            run_long(
                undirected_two_node_cpdag(),
                ("x", "y"),
                expert,
                strategy="srisk",
                max_extensions=1,
            )
        self.assertFalse(called)

    def test_ambiguous_pairs_and_prompt_order(self) -> None:
        cpdag = undirected_two_node_cpdag()
        self.assertEqual(ambiguous_pairs(cpdag), ((0, 1),))
        prompt = prompt_for("smoking", "cancer", "causes", swapped=False)
        self.assertIn("(A) smoking causes cancer", prompt)
        self.assertIn("(B) cancer causes smoking", prompt)

    def test_experiment_expands_abbreviated_metadata(self) -> None:
        self.assertEqual(
            full_variable_name("DLS C1-C2"),
            "disc lesion or disorder at cervical level C1-C2",
        )
        self.assertEqual(
            full_variable_name("R T4 Radikulopati"), "right T4 radiculopathy"
        )


if __name__ == "__main__":
    unittest.main()
