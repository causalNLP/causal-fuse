from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cd_llm.data.generation.neuropathic import write_csv, write_metadata
from cd_llm.experiments.neuropathic_benchmark import (
    EVIDENCE_CONFIDENCE,
    METHODS,
    _ensemble,
    _method_distribution,
    build_panel_spec,
    config_for_seed_count,
    ensure_nested_datasets,
    select_evidence,
)
from cd_llm.paths import RESOURCE_ROOT


class NeuropathicPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = RESOURCE_ROOT / "neuropathic" / "bnm.pickle"
        self.panel = build_panel_spec(self.model_path)

    def test_panel_is_locked_and_ancestor_closed(self) -> None:
        self.assertEqual(self.panel.name, "cervical_c2_c3")
        self.assertEqual(self.panel.anchor, "DLS C2-C3")
        self.assertEqual(self.panel.node_count, 35)
        self.assertEqual(self.panel.edge_count, 102)
        self.assertEqual(
            (
                self.panel.pathophysiology_count,
                self.panel.pattern_count,
                self.panel.symptom_count,
            ),
            (7, 12, 16),
        )
        self.assertIn("DLS C2-C3", self.panel.node_names)
        self.assertIn("Nackbesvär", self.panel.node_names)

    def test_nested_panel_data_are_exact_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            rng = np.random.default_rng(4)
            full = rng.integers(0, 2, size=(1000, 222), dtype=np.int64)
            full_path = data_root / "neuropathic_n1000_level0_seed0.csv"
            full_meta = data_root / "neuropathic_n1000_level0_seed0.json"
            write_csv(full_path, full, list(range(222)))
            write_metadata(
                full_meta,
                sample_size=1000,
                level=0,
                seed=0,
                observed_indices=list(range(222)),
            )

            datasets = ensure_nested_datasets(
                data_root,
                self.model_path,
                self.panel,
                seed=0,
            )

            data_200 = datasets[200][0]
            data_1000 = datasets[1000][0]
            np.testing.assert_array_equal(data_200, data_1000[:200])
            np.testing.assert_array_equal(
                data_1000,
                full[:, self.panel.original_indices],
            )
            self.assertEqual(data_1000.shape, (1000, 35))
            self.assertEqual(set(np.unique(data_1000)), {0, 1})

    def test_claims_are_fixed_accurate_group_one_evidence(self) -> None:
        first_claims, first_evidence, first_settings = select_evidence(self.panel)
        second_claims, second_evidence, second_settings = select_evidence(self.panel)

        self.assertEqual(first_claims, second_claims)
        self.assertEqual(first_settings, second_settings)
        self.assertEqual(first_settings["selected_pool_indices"], (4, 7, 2, 3, 5))
        self.assertEqual(len(first_claims), 5)
        self.assertTrue(all(claim.group == "I" for claim in first_claims))
        self.assertTrue(all(node.correct for node in first_evidence.nodes))
        self.assertTrue(all(node.context_compatible for node in first_evidence.nodes))
        self.assertTrue(
            all(node.confidence == EVIDENCE_CONFIDENCE for node in first_evidence.nodes)
        )
        self.assertEqual(len({node.provenance_family for node in first_evidence.nodes}), 5)
        self.assertEqual(first_evidence, second_evidence)

    def test_method_distributions_use_expected_family_sets(self) -> None:
        claims, evidence, _settings = select_evidence(self.panel)
        target = self.panel.target
        completions = {
            "cpc_alpha_0.005": (target,),
            "cpc_alpha_0.01": (target,),
            "cpc_alpha_0.05": (target,),
            "ges_bdeu_ess1": (target,),
        }

        for method in METHODS:
            with self.subTest(method=method):
                prepared = _method_distribution(method, completions, claims, evidence)
                self.assertIsNotNone(prepared)
                ensemble, distribution = prepared  # type: ignore[misc]
                self.assertAlmostEqual(float(np.sum(distribution)), 1.0)
                self.assertEqual(ensemble.graphs, (target,))

    def test_equal_family_weighting_aggregates_duplicate_graphs(self) -> None:
        target = self.panel.target
        ensemble = _ensemble(((target,), (target,)))
        self.assertEqual(ensemble.graphs, (target,))
        np.testing.assert_allclose(ensemble.q_data, np.asarray([1.0]))

    def test_smoke_seed_count_is_a_prefix_of_the_full_protocol(self) -> None:
        self.assertEqual(config_for_seed_count(2).data_seeds, (0, 1))
        self.assertEqual(config_for_seed_count(20).data_seeds, tuple(range(20)))


if __name__ == "__main__":
    unittest.main()
