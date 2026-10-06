from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cd_llm.experiments.claim_fuse_smc_sidecar import (
    ARCHIVED_EVIDENCE_PROFILE,
    PASSIVE_CONFIDENCE,
    PASSIVE_GROUP_I_20_PROFILE,
    ClaimFuseSMCConfig,
    _passive_group_i_20,
    _runtime_summary_record,
    _validate_evidence_profile,
    aggregate,
    build_parser,
    compute,
)
from cd_llm.experiments.neuropathic_benchmark import build_panel_spec
from cd_llm.paths import RESOURCE_ROOT


def _graph(edges: list[list[int]]) -> dict[str, list[list[int]]]:
    return {"directed": edges, "bidirected": [], "undirected": []}


class ClaimFuseSMCSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.results_root = Path(self.temporary.name)
        self.reference_run = "20260101T000000Z"
        self.output_run = "20260102T000000Z"
        self.config = ClaimFuseSMCConfig(
            particle_count=12,
            initialization_updates=8,
            temperatures=(0.0, 0.5, 1.0),
            rejuvenation_steps=1,
            proposal_cache_size=8,
            max_parents=2,
        )
        rng = np.random.default_rng(7)
        data = rng.normal(size=(24, 3)).tolist()
        payload = {
            "graph_index": 100,
            "protocol": {
                "node_count": 3,
                "sample_sizes": [200, 1000],
                "r_max": 4.0,
            },
            "target": _graph([[0, 1]]),
            "evidence": {
                "claims": [
                    {
                        "claim_id": "I-0",
                        "group": "I",
                        "kind": "directed",
                        "truth": 1,
                        "source": 0,
                        "target": 1,
                        "stance": 1,
                        "uncertainty": 0.1,
                        "provenance_family": "source-0",
                        "factor_id": "factor-0",
                        "context_id": "target",
                        "context_compatible": True,
                        "correct": True,
                    }
                ]
            },
            "nested_data": {"200": data, "1000": data},
            "rows": [
                {
                    "sample_size": sample_size,
                    "method": method,
                    "feasible": 1,
                    "runtime_seconds": runtime,
                }
                for sample_size in (200, 1000)
                for method, runtime in (("PC", 0.4), ("GES", 0.6))
            ],
            "samples": {
                sample_size: {
                    "methods": {
                        "CEG-Fuse (PC+GES)": {
                            "graphs": [_graph([]), _graph([[0, 1]])]
                        }
                    }
                }
                for sample_size in ("200", "1000")
            },
        }
        reference_path = (
            self.results_root
            / self.reference_run
            / "test"
            / "partials"
            / "task_000.json"
        )
        reference_path.parent.mkdir(parents=True)
        reference_path.write_text(json.dumps(payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_compute_and_aggregate_preserve_data_only_diagnostics(self) -> None:
        partial_path = compute(
            "synthetic",
            self.reference_run,
            0,
            self.output_run,
            self.results_root,
            config=self.config,
            profile="smoke",
        )
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        self.assertEqual(partial["claim_count"], 1)
        self.assertEqual(partial["warm_start_source"], "pc+ges-completion-support")
        self.assertEqual(len(partial["rows"]), 2)
        self.assertIn("data_expected_distance", partial["rows"][0])
        self.assertIn("data_distribution", partial["artifacts"]["200"])
        self.assertIn("fused_distribution", partial["artifacts"]["200"])

        output_root = aggregate(
            "synthetic",
            self.reference_run,
            self.output_run,
            self.results_root,
            1,
            config=self.config,
            profile="smoke",
        )
        self.assertTrue((output_root / "summary.csv").is_file())
        self.assertTrue((output_root / "diagnostics.csv").is_file())
        self.assertTrue((output_root / "runtime_summary.csv").is_file())
        manifest = json.loads(
            (output_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["profile"], "smoke")
        self.assertEqual(manifest["count"], 1)
        self.assertAlmostEqual(manifest["runtime"]["mean_baseline_runtime_seconds"], 2.0)
        self.assertTrue(manifest["runtime"]["runtime_gate_passed"])

    def test_runtime_gate_has_explicit_pass_and_fail_outcomes(self) -> None:
        passing = _runtime_summary_record(
            "synthetic", "candidate-160", [4.0, 5.0], [1.0, 1.0]
        )
        failing = _runtime_summary_record(
            "synthetic", "candidate-160", [6.0, 7.0], [1.0, 1.0]
        )
        self.assertTrue(passing["runtime_gate_passed"])
        self.assertFalse(failing["runtime_gate_passed"])
        self.assertGreater(failing["mean_runtime_ratio"], 5.0)

    def test_synthetic_aggregate_records_per_scm_claim_fingerprints(self) -> None:
        first_path = compute(
            "synthetic",
            self.reference_run,
            0,
            self.output_run,
            self.results_root,
            config=self.config,
            profile="smoke",
        )
        second = json.loads(first_path.read_text(encoding="utf-8"))
        second["task_index"] = 1
        second["replicate"] = 101
        second["claim_fingerprint"] = "different-scm-claims"
        for row in second["rows"]:
            row["replicate"] = 101
        second_path = first_path.with_name("task_001.json")
        second_path.write_text(json.dumps(second), encoding="utf-8")

        output_root = aggregate(
            "synthetic",
            self.reference_run,
            self.output_run,
            self.results_root,
            2,
            task_indices=(0, 1),
            config=self.config,
            profile="smoke",
        )
        manifest = json.loads(
            (output_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(manifest["claim_fingerprint"])
        self.assertEqual(set(manifest["claim_fingerprints"]), {"0", "1"})
        claims = json.loads(
            (output_root / "claims.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(claims), {"0", "1"})

    def test_parser_defaults_to_production(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--stage",
                "compute",
                "--dataset",
                "synthetic",
                "--run-id",
                self.output_run,
                "--task-index",
                "0",
            ]
        )
        self.assertEqual(arguments.profile, "production")
        self.assertEqual(arguments.evidence_profile, ARCHIVED_EVIDENCE_PROFILE)

    def test_passive_neuropathic_profile_is_fixed_balanced_and_accurate(self) -> None:
        panel = build_panel_spec(RESOURCE_ROOT / "neuropathic" / "bnm.pickle")
        first = _passive_group_i_20(panel.target, panel.node_names)
        second = _passive_group_i_20(panel.target, panel.node_names)
        claims, evidence, records, settings = first

        self.assertEqual(first, second)
        self.assertEqual(len(claims), 20)
        self.assertEqual(len(set(claims)), 20)
        self.assertEqual(sum(claim.truth == 1 for claim in claims), 10)
        self.assertTrue(all(claim.group == "I" for claim in claims))
        self.assertTrue(all(node.correct for node in evidence.nodes))
        self.assertTrue(all(node.context_compatible for node in evidence.nodes))
        self.assertTrue(all(node.confidence == PASSIVE_CONFIDENCE for node in evidence.nodes))
        self.assertEqual(len({node.provenance_family for node in evidence.nodes}), 20)
        self.assertEqual(len(records), 20)
        self.assertFalse(settings["active_query"])
        self.assertTrue(all(record["confidence"] == PASSIVE_CONFIDENCE for record in records))
        self.assertNotIn("uncertainty", json.dumps(records))
        self.assertNotIn("r_max", json.dumps(settings))

    def test_passive_neuropathic_profile_is_not_accepted_for_synthetic_data(self) -> None:
        _validate_evidence_profile("neuropathic", PASSIVE_GROUP_I_20_PROFILE)
        with self.assertRaisesRegex(ValueError, "only for neuropathic"):
            _validate_evidence_profile("synthetic", PASSIVE_GROUP_I_20_PROFILE)


if __name__ == "__main__":
    unittest.main()
