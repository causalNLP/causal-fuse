from __future__ import annotations

import unittest

from cd_llm.baselines.scp2025 import (
    apply_fci_endpoint_constraints,
    bootstrap_endpoint_frequencies,
    build_knowledge_prompt,
    build_pc_background_knowledge,
)
from cd_llm.methods.fci.graph import ARROW, CIRCLE, TAIL, PAGGraph


class SCPTests(unittest.TestCase):
    def test_required_ancestry_orients_existing_edge(self) -> None:
        graph = PAGGraph(("V01", "V02"))
        graph.add_edge("V01", "V02", CIRCLE, CIRCLE)
        constrained, records = apply_fci_endpoint_constraints(
            graph,
            {("V01", "V02"): 0.99, ("V02", "V01"): 0.01},
        )
        self.assertEqual(constrained.endpoints("V01", "V02"), (TAIL, ARROW))
        self.assertEqual(records[0].status, "applied")

    def test_conflicting_required_directions_are_skipped(self) -> None:
        graph = PAGGraph(("V01", "V02"))
        graph.add_edge("V01", "V02", CIRCLE, CIRCLE)
        constrained, records = apply_fci_endpoint_constraints(
            graph,
            {("V01", "V02"): 0.99, ("V02", "V01"): 0.99},
        )
        self.assertEqual(constrained.endpoints("V01", "V02"), (CIRCLE, CIRCLE))
        self.assertTrue(any(record.status == "skipped" for record in records))

    def test_pc_constraints_skip_bidirectional_requirements(self) -> None:
        _knowledge, records = build_pc_background_knowledge(
            ("V01", "V02"),
            {("V01", "V02"): 0.99, ("V02", "V01"): 0.99},
        )
        self.assertTrue(records)
        self.assertTrue(all(record.status == "skipped" for record in records))

    def test_bootstrap_preserves_endpoint_states(self) -> None:
        first = PAGGraph(("V01", "V02"))
        first.add_edge("V01", "V02", TAIL, ARROW)
        second = PAGGraph(("V01", "V02"))
        second.add_edge("V01", "V02", CIRCLE, CIRCLE)
        frequencies = bootstrap_endpoint_frequencies(
            [first, second], ("V01", "V02")
        )
        self.assertEqual(frequencies[("V01", "V02")]["tail-arrow"], 0.5)
        self.assertEqual(frequencies[("V01", "V02")]["circle-circle"], 0.5)

    def test_prompt_supports_opaque_identifiers_and_descriptions(self) -> None:
        prompt = build_knowledge_prompt(
            pattern=0,
            source="V01",
            target="V02",
            nodes=("V01", "V02"),
            method="PC",
            dataset_description="binary clinical observations",
            theme="a clinical system",
            variable_descriptions={"V01": "first measurement", "V02": "second measurement"},
        )
        self.assertIn("V01: first measurement", prompt)
        self.assertNotIn("neuropathic pain", prompt)


if __name__ == "__main__":
    unittest.main()
