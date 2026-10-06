from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from cd_llm.experiments.neuropathic_panel_point_benchmark import (
    PC_ALPHA,
    PC_INDEPENDENCE_TEST,
    _distribution,
    run_pc,
    slice_for_task,
)


class NeuropathicPanelPointBenchmarkTests(unittest.TestCase):
    def test_task_index_covers_twenty_seeds_and_two_sizes(self) -> None:
        self.assertEqual(slice_for_task(0), (0, 200))
        self.assertEqual(slice_for_task(1), (0, 1_000))
        self.assertEqual(slice_for_task(38), (19, 200))
        self.assertEqual(slice_for_task(39), (19, 1_000))
        with self.assertRaises(ValueError):
            slice_for_task(40)

    def test_pc_protocol_is_frozen_at_panel_a_alpha(self) -> None:
        data = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
        captured: dict[str, object] = {}
        expected_graph = object()

        def runner(observed: np.ndarray, **kwargs: object) -> object:
            captured["data"] = observed
            captured.update(kwargs)
            return SimpleNamespace(G=expected_graph)

        graph, settings = run_pc(data, ("A", "B"), runner=runner)

        self.assertIs(graph, expected_graph)
        self.assertIs(captured.pop("data"), data)
        self.assertEqual(captured["alpha"], 0.01)
        self.assertEqual(captured["indep_test"], "gsq")
        self.assertEqual(captured["stable"], True)
        self.assertEqual(captured["uc_rule"], 0)
        self.assertEqual(captured["uc_priority"], 2)
        self.assertEqual(settings["alpha"], PC_ALPHA)
        self.assertEqual(
            settings["conditional_independence_test"], PC_INDEPENDENCE_TEST
        )
        self.assertFalse(settings["standardized"])

    def test_distribution_rejects_duplicates_and_bad_mass(self) -> None:
        graph = {"directed": [[0, 1]], "bidirected": [], "undirected": []}
        with self.assertRaises(ValueError):
            _distribution([graph, graph], [0.5, 0.5])
        with self.assertRaises(ValueError):
            _distribution([graph], [0.5])


if __name__ == "__main__":
    unittest.main()
