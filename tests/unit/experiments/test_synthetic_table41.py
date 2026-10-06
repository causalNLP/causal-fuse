from __future__ import annotations

import csv
import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from cd_llm.experiments.claim_fuse_smc_eig import (
    CONFIDENCE,
    INITIALIZATION_SOURCE,
    PARTICLE_COUNT,
    TABLE41_COMMON_TASK_INDICES,
)
from cd_llm.experiments.common.artifacts import json_value, read_json, sha256, write_json
from cd_llm.experiments.synthetic_table41 import (
    METHOD_SOURCES, PASSIVE_METHOD, STUDY, aggregate, _active_distribution,
)
from cd_llm.experiments.claim_fuse_smc_sidecar import (
    PASSIVE_CONFIDENCE, PRODUCTION_CONFIG, _claim_fingerprint, _confidence_records,
)


def _graph() -> dict[str, list[list[int]]]:
    return {"directed": [[0, 1]], "bidirected": [], "undirected": []}


def _artifact() -> dict[str, object]:
    return {
        "feasible": True,
        "graphs": [_graph()],
        "probabilities": [1.0],
        "diagnostics": {},
    }


def _partial_output() -> dict[str, object]:
    return {
        "kind": "cpdag",
        "endpoint_matrix": [[0, 1, 0], [2, 0, 0], [0, 0, 0]],
    }


class SyntheticTable41Tests(unittest.TestCase):
    def test_aggregate_recomputes_every_method_on_the_locked_common_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            reference_run = "20260816T145618Z"
            camml_run = "20260817T032052Z"
            active_run = "20260826T120000Z"
            passive_run = "20260920T120000Z"
            passive_metadata = {
                "dataset": "synthetic", "profile": "production",
                "run_id": passive_run, "reference_run_id": reference_run,
                "active_query": False, "evidence_profile": "archived-5",
                "claim_count": 5, "config": asdict(PRODUCTION_CONFIG),
                "task_indices": TABLE41_COMMON_TASK_INDICES,
            }
            passive_root = results_root / passive_run / "claim_fuse_smc" / "synthetic"
            write_json(passive_root / "run_manifest.json", passive_metadata)
            claims = json_value(_confidence_records([
                {
                    "claim_id": f"I-{index}", "group": "I", "kind": "directed",
                    "source": 0, "target": 1, "truth": 1, "stance": 1,
                    "confidence": PASSIVE_CONFIDENCE, "correct": True,
                    "context_compatible": True, "provenance_family": f"expert-{index}",
                } for index in range(5)
            ], PASSIVE_CONFIDENCE))
            write_json(
                results_root / reference_run / "run_manifest.json",
                {"common_complete_indices": TABLE41_COMMON_TASK_INDICES},
            )
            write_json(
                results_root / camml_run / "run_manifest.json",
                {
                    "common_complete_indices": TABLE41_COMMON_TASK_INDICES,
                    "reference_run_id": reference_run,
                },
            )
            write_json(
                results_root
                / active_run
                / "claim_fuse_smc_eig"
                / "synthetic"
                / "run_manifest.json",
                {
                    "task_indices": TABLE41_COMMON_TASK_INDICES,
                    "graph_indices": TABLE41_COMMON_TASK_INDICES,
                    "source_synthetic_run": reference_run,
                    "random_seed_count": 0,
                    "confidence": CONFIDENCE,
                    "initialization_source": INITIALIZATION_SOURCE,
                    "particle_count": PARTICLE_COUNT,
                },
            )
            artifact_methods = tuple(
                artifact
                for method, artifact in METHOD_SOURCES
                if method
                not in (
                    "PC",
                    "GES",
                    "CaMML",
                    "Claim-Fuse-SMC-Active (5 queries)",
                    "Claim-Fuse-SMC-Active (20 queries)",
                )
            )
            for task_index in TABLE41_COMMON_TASK_INDICES:
                samples = {}
                for sample_size in (200, 1000):
                    methods = {method: _artifact() for method in artifact_methods}
                    for method in ("PC", "GES"):
                        methods[method] = {
                            **_artifact(),
                            "diagnostics": {"output": _partial_output()},
                        }
                    samples[str(sample_size)] = {"methods": methods}
                reference_path = (
                    results_root
                    / reference_run
                    / "partials"
                    / f"task_{task_index:03d}.json"
                )
                write_json(
                    reference_path,
                    {
                        "run_id": reference_run,
                        "task_index": task_index,
                        "graph_index": task_index,
                        "protocol": {"node_count": 3},
                        "target": _graph(),
                        "samples": samples,
                        "evidence": {"claims": claims},
                    },
                )
                write_json(passive_root / "partials" / f"task_{task_index:03d}.json", {
                    **passive_metadata,
                    "task_index": task_index, "replicate": task_index,
                    "reference_sha256": sha256(reference_path),
                    "warm_start_source": "pc+ges-completion-support",
                    "claims": claims, "claim_fingerprint": _claim_fingerprint(claims),
                    "artifacts": {
                        str(n): {
                            "score": {"name": "gaussian-bic", "graph_prior": "uniform", "max_parents": 12},
                            "fused_distribution": [{"graph": {"directed": []}, "probability": 1.0}],
                        } for n in (200, 1000)
                    },
                })
                write_json(
                    results_root
                    / camml_run
                    / "partials"
                    / f"task_{task_index:03d}.json",
                    {
                        "task_index": task_index,
                        "graph_index": task_index,
                        "reference_run_id": reference_run,
                        "reference_partial_sha256": sha256(reference_path),
                        "target": _graph(),
                        "fits": {
                            str(sample_size): {
                                "result": {
                                    "support": [
                                        {
                                            "directed_rows": [2, 0, 0],
                                            "probability": 1.0,
                                        }
                                    ]
                                }
                            }
                            for sample_size in (200, 1000)
                        },
                    },
                )
                write_json(
                    results_root
                    / active_run
                    / "claim_fuse_smc_eig"
                    / "synthetic"
                    / "partials"
                    / f"task_{task_index:03d}.json",
                    {
                        "task_index": task_index,
                        "graph_index": task_index,
                        "confidence": CONFIDENCE,
                        "initialization_source": INITIALIZATION_SOURCE,
                        "particle_count": PARTICLE_COUNT,
                        "source_synthetic_run_id": reference_run,
                        "source_data_sha256": sha256(reference_path),
                        "artifacts": {
                            str(sample_size): {
                                "eig": {
                                    "budgets": {
                                        "5": {
                                            "distribution": [
                                                {
                                                    "graph": _graph(),
                                                    "probability": 1.0,
                                                }
                                            ]
                                        },
                                        "20": {
                                            "distribution": [
                                                {
                                                    "graph": _graph(),
                                                    "probability": 1.0,
                                                }
                                            ]
                                        }
                                    }
                                }
                            }
                            for sample_size in (200, 1000)
                        },
                    },
                )

            output = aggregate(
                active_run,
                results_root,
                reference_run=reference_run,
                camml_run=camml_run,
                active_run=active_run,
            )
            self.assertEqual(output, results_root / active_run / STUDY)
            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2 * len(METHOD_SOURCES))
            self.assertTrue(all(row["replicate_count"] == "10" for row in rows))
            self.assertTrue(
                all(
                    row["expected_distance_mean"] == row["energy_score_mean"] == "0.0"
                    for row in rows
                )
            )

            output = aggregate(
                passive_run, results_root, reference_run=reference_run,
                camml_run=camml_run, passive_run=passive_run,
            )
            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2 * (len(METHOD_SOURCES) - 2 + 1))
            self.assertFalse(any("Active" in row["method"] for row in rows))
            passive_rows = [row for row in rows if row["method"] == PASSIVE_METHOD]
            self.assertEqual(len(passive_rows), 2)
            for row in passive_rows:
                self.assertEqual(float(row["expected_distance_mean"]), 1.0)
                self.assertEqual(float(row["adjacency_f1_mean"]), 0.0)
                self.assertEqual(float(row["orientation_f1_mean"]), 0.0)
                self.assertEqual(row["replicate_count"], "10")
            self.assertEqual(read_json(output / "summary.csv.metrics.json")["metric_scale"], "unnormalized")
            first_path = passive_root / "partials" / f"task_{TABLE41_COMMON_TASK_INDICES[0]:03d}.json"
            original = read_json(first_path)
            mutations = []
            for key, value in (("reference_sha256", "wrong"), ("active_query", True), ("claim_count", 3)):
                mutated = copy.deepcopy(original)
                mutated[key] = value
                mutations.append((key, mutated))
            mutated = copy.deepcopy(original)
            mutated["claims"][0]["confidence"] = 1.0
            mutations.append(("claims", mutated))
            mutated = copy.deepcopy(original)
            mutated["claims"][0]["stance"] = -1
            mutations.append(("claims", mutated))
            mutated = copy.deepcopy(original)
            mutated["config"]["particle_count"] = 20
            mutations.append(("config", mutated))
            mutated = copy.deepcopy(original)
            del mutated["artifacts"]["1000"]
            mutations.append(("sample sizes", mutated))
            for name, mutated in mutations:
                with self.subTest(mismatch=name):
                    write_json(first_path, mutated)
                    with self.assertRaises(ValueError):
                        aggregate(passive_run, results_root, reference_run=reference_run,
                                  camml_run=camml_run, passive_run=passive_run)
            write_json(first_path, original)
            aggregate(passive_run, results_root, reference_run=reference_run,
                      camml_run=camml_run, passive_run=passive_run, active_run=active_run)
            self.assertIn(PASSIVE_METHOD, read_json(output / "run_manifest.json")["methods"])

    def test_invalid_smc_distributions_are_rejected(self) -> None:
        for records in (
            [{"graph": _graph(), "probability": float("nan")}],
            [{"graph": {"directed": [[0, 1], [1, 0]]}, "probability": 1.0}],
            [{"graph": {"bidirected": [[0, 1]]}, "probability": 1.0}],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                _active_distribution(records, 3)


if __name__ == "__main__":
    unittest.main()
