from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments.synthetic_dag_claim_ablation import (
    ALGORITHM_SEED_COUNT,
    BUDGETS,
    CONFIDENCE,
    FAMILY_CONFIGS,
    LOCKED_SOURCE_TASK_INDICES,
    SAMPLE_SIZES,
    _audit_result,
    _paired_bootstrap,
    _run_seed,
    _scm_rows,
    _truth_answer_provider,
    aggregate_available,
    task_specification,
    task_specifications,
)
from cd_llm.methods.active_query import (
    ClaimQuery,
    ancestry_ordering_query_links,
    dag_query_pool,
)


class SyntheticDAGClaimAblationTests(unittest.TestCase):
    def test_locked_factorial_task_grid(self) -> None:
        tasks = task_specifications()
        expected = (
            len(LOCKED_SOURCE_TASK_INDICES)
            * len(SAMPLE_SIZES)
            * len(FAMILY_CONFIGS)
            * 2
        )
        self.assertEqual(len(tasks), 736)
        self.assertEqual(len(tasks), expected)
        self.assertEqual(task_specification(0).family, "local_edge")
        self.assertEqual(task_specification(0).policy, "eig")
        self.assertEqual(task_specification(len(tasks) - 1).family, "all_observable")
        self.assertEqual(task_specification(len(tasks) - 1).policy, "random")
        self.assertEqual(BUDGETS, (1, 3, 5, 10, 20))
        self.assertEqual(CONFIDENCE, 1.0)
        self.assertEqual(ALGORITHM_SEED_COUNT, 20)

    def test_query_pool_sizes_and_all_observable_complement_links(self) -> None:
        expected_sizes = {
            "local_edge": (570, 570, 0),
            "causal_order": (380, 380, 0),
            "ancestral_relation": (380, 380, 0),
            "all_observable": (1_330, 950, 760),
        }
        for family, query_families in FAMILY_CONFIGS:
            queries = dag_query_pool(20, query_families)
            links = ancestry_ordering_query_links(queries)
            actual = (len(queries), len(queries) - len(links) // 2, len(links))
            self.assertEqual(actual, expected_sizes[family])

    def test_truth_adapter_implements_group_i_and_group_ii_predicates(self) -> None:
        truth = graph_from_edges(3, [(0, 1), (1, 2)])
        answer = _truth_answer_provider(truth)
        self.assertEqual(answer(ClaimQuery("directed", 0, 1, "edge")), 1)
        self.assertEqual(answer(ClaimQuery("adjacency", 0, 2, "adjacency")), 0)
        self.assertEqual(answer(ClaimQuery("ancestry", 0, 2, "ancestry")), 1)
        self.assertEqual(answer(ClaimQuery("ordering", 0, 2, "ordering")), 1)
        self.assertEqual(answer(ClaimQuery("ordering", 2, 0, "reverse-order")), 0)

    def test_seed_is_paired_across_family_and_policy(self) -> None:
        first = _run_seed(7, 200, 3)
        self.assertEqual(first, _run_seed(7, 200, 3))
        self.assertNotEqual(first, _run_seed(7, 200, 4))

    def test_audit_does_not_flag_a_single_member_of_a_linked_pair(self) -> None:
        graph = graph_from_edges(3, [(0, 1)])
        ancestry = ClaimQuery("ancestry", 0, 1, "ancestry-0-1")
        result = SimpleNamespace(
            steps=(SimpleNamespace(query=ancestry, answer=1),),
            particles=(graph,),
            weights=(1.0,),
        )
        audit = _audit_result(
            result,
            {
                "ancestry-0-1": "ordering-1-0",
                "ordering-1-0": "ancestry-0-1",
            },
        )
        self.assertEqual(audit["linked_pair_violations"], 0)
        self.assertEqual(audit["hard_claim_violations"], 0)

    def test_scm_aggregation_averages_exactly_twenty_seed_rows(self) -> None:
        rows = []
        for seed_index in range(ALGORITHM_SEED_COUNT):
            rows.append(
                {
                    "graph_index": 0,
                    "sample_size": 200,
                    "family": "local_edge",
                    "policy": "eig",
                    "budget": 20,
                    "expected_distance": float(seed_index),
                    "energy_score": 1.0,
                    "adjacency_f1": 0.8,
                    "orientation_f1": 0.7,
                    "entropy": 2.0,
                    "active_runtime_seconds": 3.0,
                }
            )
        aggregated = _scm_rows(rows)
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["seed_count"], 20)
        self.assertEqual(aggregated[0]["expected_distance"], 9.5)

    def test_paired_bootstrap_preserves_a_constant_paired_difference(self) -> None:
        result = _paired_bootstrap((2.0, 3.0, 4.0), (1.0, 2.0, 3.0), seed=8)
        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["ci_lower"], 1.0)
        self.assertEqual(result["ci_upper"], 1.0)

    def test_available_aggregate_selects_only_common_complete_scms(self) -> None:
        with TemporaryDirectory() as directory:
            results_root = Path(directory)
            run = "20260826T120000Z"
            partials = results_root / run / "synthetic_dag_claim_ablation" / "partials"
            partials.mkdir(parents=True)
            tasks = task_specifications()
            complete_graph = LOCKED_SOURCE_TASK_INDICES[0]
            incomplete_graph = LOCKED_SOURCE_TASK_INDICES[1]
            for task_index, specification in enumerate(tasks):
                if specification.source_task_index == complete_graph or (
                    specification.source_task_index == incomplete_graph
                    and task_index % 16 != 0
                ):
                    (partials / f"task_{task_index:04d}.json").touch()
            with patch(
                "cd_llm.experiments.synthetic_dag_claim_ablation._aggregate_selected"
            ) as selected:
                selected.return_value = partials.parent / "available_common_complete"
                aggregate_available(run, results_root)
            self.assertEqual(selected.call_args.args[3], (complete_graph,))
            self.assertEqual(len(selected.call_args.args[2]), 16)


if __name__ == "__main__":
    unittest.main()
