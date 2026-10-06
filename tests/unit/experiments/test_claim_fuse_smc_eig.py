from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments.claim_fuse_smc_eig import (
    BUDGETS,
    CONFIDENCE,
    INITIALIZATION_METHOD,
    INITIALIZATION_SOURCE,
    PARTICLE_COUNT,
    SAMPLE_SIZES,
    STUDY,
    TABLE41_COMMON_TASK_INDICES,
    _initial_particle_population,
    _parse_task_indices,
    _pc_ges_data_distribution,
    _truth_answer_provider,
    aggregate,
)
from cd_llm.experiments.common.artifacts import read_json, write_json
from cd_llm.methods.active_query import ClaimQuery


class ClaimFuseSMCEIGExperimentTests(unittest.TestCase):
    def test_truth_adapter_answers_only_the_selected_predicate(self) -> None:
        truth = graph_from_edges(3, [(0, 1)])
        answer = _truth_answer_provider(truth)

        self.assertEqual(answer(ClaimQuery("directed", 0, 1, "forward")), 1)
        self.assertEqual(answer(ClaimQuery("directed", 1, 0, "reverse")), 0)
        self.assertEqual(answer(ClaimQuery("adjacency", 0, 1, "adjacent")), 1)
        self.assertEqual(answer(ClaimQuery("adjacency", 0, 2, "separate")), 0)

    def test_table41_protocol_uses_locked_tasks_and_twenty_query_cap(self) -> None:
        encoded = ",".join(str(index) for index in TABLE41_COMMON_TASK_INDICES)
        self.assertEqual(_parse_task_indices(encoded), TABLE41_COMMON_TASK_INDICES)
        self.assertEqual(
            _parse_task_indices(encoded.replace(",", ":")),
            TABLE41_COMMON_TASK_INDICES,
        )
        self.assertEqual(BUDGETS[-1], 20)
        self.assertEqual(CONFIDENCE, 1.0)
        self.assertEqual(PARTICLE_COUNT, 384)
        self.assertEqual(INITIALIZATION_SOURCE, "pc+ges-completion-data-distribution")

    def test_initialization_uses_pc_ges_data_probabilities(self) -> None:
        payload = {
            "samples": {
                "200": {
                    "methods": {
                        INITIALIZATION_METHOD: {
                            "feasible": True,
                            "graphs": [
                                {"directed": [[0, 1]], "bidirected": [], "undirected": []},
                                {"directed": [[1, 0]], "bidirected": [], "undirected": []},
                            ],
                            "probabilities": [0.99, 0.01],
                            "diagnostics": {"data_probabilities": [0.25, 0.75]},
                        }
                    }
                }
            }
        }
        distribution = _pc_ges_data_distribution(payload, 200, 2)
        forward = graph_from_edges(2, [(0, 1)])
        reverse = graph_from_edges(2, [(1, 0)])

        self.assertEqual(distribution, {forward: 0.25, reverse: 0.75})
        particles, weights, empirical = _initial_particle_population(distribution, 7)
        self.assertEqual(len(particles), PARTICLE_COUNT)
        self.assertEqual(len(weights), PARTICLE_COUNT)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(sum(empirical.values()), 1.0)
        self.assertEqual(set(empirical), {forward, reverse})

    def test_table41_aggregate_accepts_eig_only_partials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            run_id = "20260822T120000Z"
            source_synthetic_run = "20260816T145618Z"
            partial_root = results_root / run_id / STUDY / "synthetic" / "partials"
            for task_index in TABLE41_COMMON_TASK_INDICES:
                write_json(
                    results_root
                    / source_synthetic_run
                    / "partials"
                    / f"task_{task_index:03d}.json",
                    {"graph_index": task_index},
                )
                rows = []
                for sample_size in SAMPLE_SIZES:
                    for policy, budgets in (("data-only", (0,)), ("eig", BUDGETS)):
                        for budget in budgets:
                            rows.append(
                                {
                                    "graph_index": task_index,
                                    "sample_size": sample_size,
                                    "policy": policy,
                                    "budget": budget,
                                    "expected_distance": 0.1,
                                    "energy_score": 0.1,
                                    "adjacency_f1": 0.8,
                                    "orientation_f1": 0.7,
                                    "entropy": 1.0,
                                    "runtime_seconds": float(budget),
                                }
                            )
                write_json(
                    partial_root / f"task_{task_index:03d}.json",
                    {
                        "task_index": task_index,
                        "graph_index": task_index,
                        "source_synthetic_run_id": source_synthetic_run,
                        "initialization_source": INITIALIZATION_SOURCE,
                        "confidence": CONFIDENCE,
                        "particle_count": PARTICLE_COUNT,
                        "random_seed_count": 0,
                        "rows": rows,
                    },
                )

            output = aggregate(
                run_id,
                results_root,
                source_synthetic_run,
            )
            manifest = read_json(output / "run_manifest.json")
            self.assertEqual(manifest["count"], 10)
            self.assertEqual(manifest["random_seed_count"], 0)
            self.assertEqual(manifest["confidence"], 1.0)
            self.assertEqual(manifest["initialization_source"], INITIALIZATION_SOURCE)
            self.assertEqual(manifest["particle_count"], PARTICLE_COUNT)
            self.assertFalse((output / "paired_eig_vs_random.csv").exists())
            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue(
                any(
                    row["policy"] == "eig"
                    and row["budget"] == "20"
                    and row["replicate_count"] == "10"
                    for row in summary
                )
            )


if __name__ == "__main__":
    unittest.main()
