from __future__ import annotations

import unittest
from unittest.mock import patch

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments.synthetic_claim_robustness import (
    CONDITIONS,
    RESPONSE_SEED_COUNT,
    SAMPLE_SIZES,
    SOFT_BUDGETS,
    STUDY_VERSION,
    ResponseCondition,
    SimulatedResponseProvider,
    _condition_payloads,
    _paired_contrasts,
    _response_metrics,
    _response_protocol,
    _result_row,
    _scm_rows,
    _summary_rows,
    task_specification,
    task_specifications,
)
from cd_llm.experiments.claim_fuse_smc_eig import TABLE41_COMMON_TASK_INDICES
from cd_llm.methods.active_query import ClaimQuery


class SyntheticClaimRobustnessTests(unittest.TestCase):
    def test_locked_task_and_condition_grids(self) -> None:
        tasks = task_specifications()
        self.assertEqual(len(tasks), len(TABLE41_COMMON_TASK_INDICES) * len(SAMPLE_SIZES))
        self.assertEqual(len(tasks), 20)
        self.assertEqual(task_specification(0).source_task_index, TABLE41_COMMON_TASK_INDICES[0])
        self.assertEqual(task_specification(0).sample_size, 200)
        self.assertEqual(task_specification(1).sample_size, 1_000)
        self.assertEqual(STUDY_VERSION, 5)
        self.assertEqual(len(CONDITIONS), 3)
        self.assertEqual(RESPONSE_SEED_COUNT, 5)
        self.assertEqual(SOFT_BUDGETS, (5, 10))
        self.assertEqual(len(tasks) * len(CONDITIONS) * RESPONSE_SEED_COUNT, 300)
        self.assertEqual(_condition_payloads()[0]["budgets"], [5, 10])
        self.assertTrue(
            all(condition.treatment == "soft_evidence" for condition in CONDITIONS)
        )
        identifiers = tuple(condition.condition_id for condition in CONDITIONS)
        self.assertEqual(
            identifiers,
            (
                "iid_overconfident_40_as_70",
                "iid_calibrated_70",
                "iid_underconfident_100_as_70",
            ),
        )
        self.assertEqual(
            {condition.actual_accuracy for condition in CONDITIONS}, {0.4, 0.7, 1.0}
        )
        self.assertEqual(
            {condition.reported_confidence for condition in CONDITIONS}, {0.7}
        )
        self.assertEqual(
            {condition.effective_confidence for condition in CONDITIONS}, {0.7}
        )
        self.assertIn(
            "correctness is nested",
            _response_protocol()["paired_answer_field"],
        )
        self.assertIn("explicit soft evidence", _response_protocol()["evidence_treatment"])

    def test_paired_contrasts_cover_regime_and_baseline_differences(self) -> None:
        rows = []
        for sample_size in SAMPLE_SIZES:
            rows.append(
                {
                    "sample_size": sample_size,
                    "condition_id": "data_only_pc_ges",
                    "budget": 0,
                    "graph_index": 1,
                    "expected_distance": 0.1,
                    "energy_score": 0.1,
                    "adjacency_f1": 0.9,
                    "orientation_f1": 0.8,
                }
            )
            for budget in SOFT_BUDGETS:
                for condition in CONDITIONS:
                    rows.append(
                        {
                            "sample_size": sample_size,
                            "condition_id": condition.condition_id,
                            "budget": budget,
                            "graph_index": 1,
                            "expected_distance": 0.1,
                            "energy_score": 0.1,
                            "adjacency_f1": 0.9,
                            "orientation_f1": 0.8,
                        }
                    )
        with patch(
            "cd_llm.experiments.synthetic_claim_robustness._paired_bootstrap",
            return_value={},
        ):
            contrasts = _paired_contrasts(rows, (1,))
        self.assertEqual(len(contrasts), 80)
        self.assertEqual(
            {row["comparison"] for row in contrasts},
            {
                "calibrated70-minus-overconfident40",
                "underconfident100-minus-calibrated70",
                "overconfident40-minus-data-only",
                "calibrated70-minus-data-only",
                "underconfident100-minus-data-only",
            },
        )

    def test_response_field_is_shared_and_correctness_is_nested(self) -> None:
        truth = graph_from_edges(3, [(0, 1), (1, 2)])
        query = ClaimQuery("ancestry", 0, 2, "ancestry-0-2")
        responses = {}
        for condition in CONDITIONS:
            provider = SimulatedResponseProvider(truth, 7, 200, 3, condition)
            responses[condition.actual_accuracy] = provider(query)
        uniforms = {
            float(response.audit["seeded_uniform"]) for response in responses.values()
        }
        self.assertEqual(len(uniforms), 1)
        correctness = {
            accuracy: bool(response.audit["correct"])
            for accuracy, response in responses.items()
        }
        self.assertLessEqual(int(correctness[0.4]), int(correctness[0.7]))
        self.assertLessEqual(int(correctness[0.7]), int(correctness[1.0]))
        self.assertTrue(correctness[1.0])
        self.assertEqual({response.confidence for response in responses.values()}, {0.7})
        self.assertEqual(
            {response.treatment for response in responses.values()}, {"soft_evidence"}
        )

    def test_accuracy_one_never_flips_truth(self) -> None:
        truth = graph_from_edges(3, [(0, 1), (1, 2)])
        condition = ResponseCondition(
            "perfect", 1.0, 0.7, 0.7, "eig", SOFT_BUDGETS
        )
        provider = SimulatedResponseProvider(truth, 1, 1_000, 19, condition)
        queries = (
            ClaimQuery("directed", 0, 1, "directed-0-1"),
            ClaimQuery("adjacency", 0, 2, "adjacency-0-2"),
            ClaimQuery("ordering", 2, 0, "ordering-2-0"),
        )
        self.assertEqual([provider(query).answer for query in queries], [1, 0, 0])
        self.assertTrue(all(record["correct"] for record in provider.records))

    def test_response_metrics_use_correctness_not_answer_value(self) -> None:
        records = (
            {
                "correct": True,
                "reported_confidence": 1.0,
                "effective_confidence": 0.99,
            },
            {
                "correct": False,
                "reported_confidence": 1.0,
                "effective_confidence": 0.99,
            },
        )
        metrics = _response_metrics(records)
        self.assertEqual(metrics["response_count"], 2)
        self.assertEqual(metrics["realized_accuracy"], 0.5)
        self.assertAlmostEqual(float(metrics["reported_calibration_gap"]), 0.5)
        self.assertAlmostEqual(float(metrics["reported_brier_score"]), 0.5)
        self.assertAlmostEqual(float(metrics["effective_calibration_gap"]), 0.49)
        self.assertAlmostEqual(float(metrics["effective_brier_score"]), 0.4901)

    def test_result_row_defines_positive_ed_change_as_negative_transfer(self) -> None:
        condition = CONDITIONS[0]
        baseline = {
            "expected_distance": 0.2,
            "energy_score": 0.15,
            "adjacency_f1": 0.7,
            "orientation_f1": 0.6,
            "entropy": 1.0,
        }
        metrics = {
            "expected_distance": 0.25,
            "energy_score": 0.14,
            "adjacency_f1": 0.75,
            "orientation_f1": 0.55,
            "entropy": 0.9,
        }
        records = [
            {
                "correct": True,
                "reported_confidence": 0.7,
                "effective_confidence": 0.7,
            }
            for _ in range(5)
        ]
        row = _result_row(
            task_specification(0),
            condition,
            0,
            1,
            5,
            baseline,
            records,
            metrics=metrics,
            runtime_seconds=1.0,
        )
        self.assertAlmostEqual(row["negative_transfer_ed"], 0.05)
        self.assertEqual(row["negative_transfer"], 1)
        self.assertAlmostEqual(row["adjacency_f1_loss"], -0.05)

    def test_scm_and_summary_aggregation_report_negative_transfer_rate(self) -> None:
        graph_index = TABLE41_COMMON_TASK_INDICES[0]
        baseline = {
            (graph_index, 200): {
                "expected_distance": 0.2,
                "energy_score": 0.15,
                "adjacency_f1": 0.7,
                "orientation_f1": 0.6,
                "entropy": 1.0,
            }
        }
        rows = []
        for seed in range(RESPONSE_SEED_COUNT):
            harmed = seed == 0
            rows.append(
                {
                    "graph_index": graph_index,
                    "sample_size": 200,
                    "condition_id": "iid_calibrated_70",
                    "actual_accuracy": 0.7,
                    "reported_confidence": 0.7,
                    "effective_confidence": 0.7,
                    "treatment": "soft_evidence",
                    "policy": "eig",
                    "budget": 10,
                    "status": "success",
                    "negative_transfer": int(harmed),
                    **baseline[(graph_index, 200)],
                    "runtime_seconds": 1.0,
                    "realized_accuracy": 0.7,
                    "mean_reported_confidence": 0.7,
                    "reported_calibration_gap": 0.0,
                    "reported_brier_score": 0.21,
                    "mean_effective_confidence": 0.7,
                    "effective_calibration_gap": 0.0,
                    "effective_brier_score": 0.21,
                    "negative_transfer_ed": 0.01 if harmed else 0.0,
                    "negative_transfer_es": 0.0,
                    "adjacency_f1_loss": 0.0,
                    "orientation_f1_loss": 0.0,
                }
            )
        scm = _scm_rows(rows, baseline)
        condition = next(
            row for row in scm if row["condition_id"] == "iid_calibrated_70"
        )
        self.assertEqual(condition["trajectory_count"], 5)
        self.assertEqual(condition["negative_transfer_rate"], 0.2)
        summary = _summary_rows(scm, (graph_index,))
        condition_summary = next(
            row for row in summary if row["condition_id"] == "iid_calibrated_70"
        )
        self.assertEqual(condition_summary["negative_transfer_rate_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
