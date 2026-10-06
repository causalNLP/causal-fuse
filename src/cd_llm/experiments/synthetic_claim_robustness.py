"""Paired robustness study across accuracy--calibration source regimes."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cd_llm.core.finite_support import EvidenceTreatment, Graph, stable_seed
from cd_llm.experiments.claim_fuse_smc_eig import (
    INITIALIZATION_METHOD,
    INITIALIZATION_SOURCE,
    PARTICLE_COUNT,
    TABLE41_COMMON_TASK_INDICES,
    _graph,
    _initial_particle_population,
    _metrics,
    _pc_ges_data_distribution,
    _serialize,
    _source_data_path,
)
from cd_llm.experiments.common.artifacts import (
    read_json,
    run_id,
    sha256,
    write_csv,
    write_json,
)
from cd_llm.experiments.synthetic_dag_claim_ablation import (
    SOURCE_SYNTHETIC_RUN_ID,
    _paired_bootstrap,
    _run_seed,
    _validate_source,
)
from cd_llm.methods.active_query import (
    ClaimQuery,
    QueryResponse,
    active_query_smc,
    ancestry_ordering_query_links,
    dag_query_pool,
    query_values,
)
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "synthetic_claim_robustness"
STUDY_VERSION = 5
SAMPLE_SIZES = (200, 1_000)
SOFT_BUDGETS = (5, 10)
SOFT_CONFIDENCE_CAP = 0.99
RESPONSE_SEED_COUNT = 5
BOOTSTRAP_RESAMPLES = 10_000
QUERY_FAMILIES = ("local_edge", "causal_order", "ancestry")
METRIC_KEYS = (
    "expected_distance",
    "energy_score",
    "adjacency_f1",
    "orientation_f1",
    "entropy",
)


@dataclass(frozen=True)
class ResponseCondition:
    condition_id: str
    actual_accuracy: float
    reported_confidence: float
    effective_confidence: float
    policy: str
    budgets: tuple[int, ...]
    treatment: EvidenceTreatment = "soft_evidence"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.actual_accuracy) <= 1.0:
            raise ValueError("actual accuracy must lie in [0, 1]")
        if not 0.5 <= float(self.reported_confidence) <= 1.0:
            raise ValueError("reported confidence must lie in [0.5, 1]")
        if self.treatment != "soft_evidence":
            raise ValueError("new robustness conditions must use soft evidence")
        expected = min(float(self.reported_confidence), SOFT_CONFIDENCE_CAP)
        if not math.isclose(float(self.effective_confidence), expected):
            raise ValueError("effective confidence must equal the capped report")
        if self.policy not in ("eig", "random"):
            raise ValueError("response policy must be 'eig' or 'random'")
        if tuple(self.budgets) != SOFT_BUDGETS:
            raise ValueError("robustness conditions must use the locked budgets")


CONDITIONS = (
    ResponseCondition(
        "iid_overconfident_40_as_70",
        0.4,
        0.7,
        0.7,
        "eig",
        SOFT_BUDGETS,
    ),
    ResponseCondition("iid_calibrated_70", 0.7, 0.7, 0.7, "eig", SOFT_BUDGETS),
    ResponseCondition(
        "iid_underconfident_100_as_70",
        1.0,
        0.7,
        0.7,
        "eig",
        SOFT_BUDGETS,
    ),
)


def _condition_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        payload = asdict(condition)
        payload["budgets"] = list(condition.budgets)
        payloads.append(payload)
    return payloads


def _response_protocol() -> dict[str, Any]:
    return {
        "error_model": "iid-symmetric-binary",
        "actual_accuracy": (
            "For every queried predicate, independently retain its ground-truth "
            "binary answer with this probability and flip it otherwise."
        ),
        "reported_confidence": (
            "The confidence emitted by the simulated source before inference-time "
            "treatment; it can differ from the answer channel's actual accuracy."
        ),
        "effective_confidence": (
            "The confidence used by query selection and Bayesian conditioning after "
            f"the soft-evidence cap of {SOFT_CONFIDENCE_CAP:.2f} is applied."
        ),
        "paired_answer_field": (
            "Conditions with the same SCM, sample size, response seed, and query ID "
            "share the same uniform answer draw. Accuracy is applied only as a "
            "threshold, so correctness is nested across the 40%, 70%, and 100% "
            "regimes for every query selected by more than one trajectory."
        ),
        "evidence_treatment": (
            "Every newly simulated robustness claim is explicit soft evidence. "
            "All three regimes report confidence 0.70, below the soft cap, and no "
            "response is interpreted as a hard constraint."
        ),
        "regime_interpretation": (
            "At fixed reported and effective confidence 0.70, actual accuracy 0.40 "
            "is overconfident, 0.70 is calibrated, and 1.00 is underconfident. "
            "These are joint source regimes rather than a factorial decomposition "
            "of accuracy and calibration."
        ),
        "query_groups": {
            "Group I": ["directed edge", "adjacency"],
            "Group II": ["causal order", "ancestry"],
        },
    }


@dataclass(frozen=True)
class TaskSpecification:
    source_task_index: int
    sample_size: int


def task_specifications() -> tuple[TaskSpecification, ...]:
    return tuple(
        TaskSpecification(source_task_index, sample_size)
        for source_task_index in TABLE41_COMMON_TASK_INDICES
        for sample_size in SAMPLE_SIZES
    )


def task_specification(task_index: int) -> TaskSpecification:
    tasks = task_specifications()
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task index must lie in [0, {len(tasks) - 1}]")
    return tasks[task_index]


def _query_group(query: ClaimQuery) -> str:
    return "I" if query.kind in ("directed", "adjacency") else "II"


def _response_uniform(
    graph_index: int,
    sample_size: int,
    response_seed: int,
    query_id: str,
) -> float:
    seed = stable_seed(
        STUDY,
        "iid-response",
        graph_index,
        sample_size,
        response_seed,
        query_id,
    )
    return float(np.random.default_rng(seed).random())


class SimulatedResponseProvider:
    """Deterministic IID binary-error channel with an auditable confidence report."""

    def __init__(
        self,
        truth: Graph,
        graph_index: int,
        sample_size: int,
        response_seed: int,
        condition: ResponseCondition,
    ) -> None:
        self.truth = truth
        self.graph_index = graph_index
        self.sample_size = sample_size
        self.response_seed = response_seed
        self.condition = condition
        self.records: list[dict[str, Any]] = []

    def __call__(self, query: ClaimQuery) -> QueryResponse:
        true_answer = int(query_values((self.truth,), query)[0])
        uniform = _response_uniform(
            self.graph_index,
            self.sample_size,
            self.response_seed,
            query.query_id,
        )
        correct = uniform < self.condition.actual_accuracy
        answer = true_answer if correct else 1 - true_answer
        audit = {
            "condition_id": self.condition.condition_id,
            "error_model": "iid-symmetric-binary",
            "nominal_accuracy": self.condition.actual_accuracy,
            "reported_confidence": self.condition.reported_confidence,
            "effective_confidence": self.condition.effective_confidence,
            "treatment": self.condition.treatment,
            "response_seed": self.response_seed,
            "seeded_uniform": uniform,
            "query_group": _query_group(query),
            "true_answer": true_answer,
            "returned_answer": answer,
            "correct": bool(correct),
        }
        self.records.append({"query": asdict(query), **audit})
        return QueryResponse(
            answer=answer,
            reported_confidence=self.condition.reported_confidence,
            treatment=self.condition.treatment,
            audit=audit,
        )


def _response_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "response_count": 0,
            "realized_accuracy": math.nan,
            "mean_reported_confidence": math.nan,
            "reported_calibration_gap": math.nan,
            "reported_brier_score": math.nan,
            "mean_effective_confidence": math.nan,
            "effective_calibration_gap": math.nan,
            "effective_brier_score": math.nan,
        }
    correct = np.asarray([int(bool(record["correct"])) for record in records], dtype=float)
    reported = np.asarray(
        [float(record["reported_confidence"]) for record in records], dtype=float
    )
    effective = np.asarray(
        [float(record["effective_confidence"]) for record in records], dtype=float
    )
    accuracy = float(np.mean(correct))
    mean_reported = float(np.mean(reported))
    mean_effective = float(np.mean(effective))
    return {
        "response_count": len(records),
        "realized_accuracy": accuracy,
        "mean_reported_confidence": mean_reported,
        "reported_calibration_gap": abs(accuracy - mean_reported),
        "reported_brier_score": float(np.mean((reported - correct) ** 2)),
        "mean_effective_confidence": mean_effective,
        "effective_calibration_gap": abs(accuracy - mean_effective),
        "effective_brier_score": float(np.mean((effective - correct) ** 2)),
    }


def _result_row(
    specification: TaskSpecification,
    condition: ResponseCondition,
    seed_index: int,
    algorithm_seed: int,
    budget: int,
    baseline: Mapping[str, float],
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, float],
    runtime_seconds: float,
) -> dict[str, Any]:
    prefix = tuple(records[:budget])
    row: dict[str, Any] = {
        "source_task_index": specification.source_task_index,
        "graph_index": specification.source_task_index,
        "sample_size": specification.sample_size,
        "condition_id": condition.condition_id,
        "actual_accuracy": condition.actual_accuracy,
        "reported_confidence": condition.reported_confidence,
        "effective_confidence": condition.effective_confidence,
        "treatment": condition.treatment,
        "policy": condition.policy,
        "budget": budget,
        "seed_index": seed_index,
        "algorithm_seed": algorithm_seed,
        "status": "success",
        "runtime_seconds": runtime_seconds,
        **_response_metrics(prefix),
    }
    for key in METRIC_KEYS:
        row[key] = float(metrics[key])
    row.update(
        {
            "negative_transfer_ed": float(metrics["expected_distance"])
            - float(baseline["expected_distance"]),
            "negative_transfer_es": float(metrics["energy_score"])
            - float(baseline["energy_score"]),
            "adjacency_f1_loss": float(baseline["adjacency_f1"])
            - float(metrics["adjacency_f1"]),
            "orientation_f1_loss": float(baseline["orientation_f1"])
            - float(metrics["orientation_f1"]),
            "negative_transfer": int(
                float(metrics["expected_distance"])
                > float(baseline["expected_distance"])
            ),
        }
    )
    return row


def compute(
    task_index: int,
    output_run: str,
    results_root: Path,
    *,
    source_synthetic_run: str = SOURCE_SYNTHETIC_RUN_ID,
    response_seed_count: int = RESPONSE_SEED_COUNT,
) -> Path:
    if response_seed_count != RESPONSE_SEED_COUNT:
        raise ValueError(
            f"the locked robustness protocol requires exactly {RESPONSE_SEED_COUNT} seeds"
        )
    specification = task_specification(task_index)
    source_path = _source_data_path(
        results_root, source_synthetic_run, specification.source_task_index
    )
    source = read_json(source_path)
    graph_index, node_count = _validate_source(
        source,
        source_synthetic_run,
        specification.source_task_index,
        specification.sample_size,
    )
    truth = _graph(source["target"], node_count)
    reference_initial = _pc_ges_data_distribution(
        source, specification.sample_size, node_count
    )
    initial_particles, initial_weights, initial_distribution = (
        _initial_particle_population(
            reference_initial,
            stable_seed(
                "pc-ges-active-initialization",
                graph_index,
                specification.sample_size,
            ),
        )
    )
    baseline = _metrics(initial_distribution, truth)
    data = np.asarray(
        source["nested_data"][str(specification.sample_size)], dtype=float
    )
    score = GaussianBICScore(data, standardize_data=True, max_parents=12)
    queries = dag_query_pool(node_count, QUERY_FAMILIES)
    links = ancestry_ordering_query_links(queries)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    task_started = time.perf_counter()

    for seed_index in range(response_seed_count):
        algorithm_seed = _run_seed(graph_index, specification.sample_size, seed_index)
        for condition in CONDITIONS:
            provider = SimulatedResponseProvider(
                truth,
                graph_index,
                specification.sample_size,
                seed_index,
                condition,
            )
            started = time.perf_counter()
            result = active_query_smc(
                None,
                score,
                provider,
                query_pool=queries,
                linked_query_ids=links,
                budgets=condition.budgets,
                confidence=condition.reported_confidence,
                treatment=condition.treatment,
                soft_confidence_cap=SOFT_CONFIDENCE_CAP,
                policy=condition.policy,
                seed=algorithm_seed,
                initial_particles=initial_particles,
                initial_weights=initial_weights,
            )

            runtime = time.perf_counter() - started
            metrics_by_budget: dict[str, dict[str, float]] = {}
            distributions: dict[str, list[dict[str, Any]]] = {}
            for budget in condition.budgets:
                metrics = _metrics(result.distributions[budget], truth)
                rows.append(
                    _result_row(
                        specification,
                        condition,
                        seed_index,
                        algorithm_seed,
                        budget,
                        baseline,
                        provider.records,
                        metrics=metrics,
                        runtime_seconds=float(result.cumulative_seconds[budget]),
                    )
                )
                metrics_by_budget[str(budget)] = {
                    key: float(metrics[key]) for key in METRIC_KEYS
                }
                if seed_index == 0:
                    distributions[str(budget)] = _serialize(
                        result.distributions[budget]
                    )
            runs.append(
                {
                    "condition_id": condition.condition_id,
                    "reported_confidence": condition.reported_confidence,
                    "effective_confidence": condition.effective_confidence,
                    "treatment": condition.treatment,
                    "seed_index": seed_index,
                    "algorithm_seed": algorithm_seed,
                    "status": "success",
                    "runtime_seconds": runtime,
                    "responses": provider.records,
                    "trace": [asdict(step) for step in result.steps],
                    "metrics": metrics_by_budget,
                    "seed_zero_distributions": distributions,
                    "resampling_count": result.resampling_count,
                    "support_recovery_count": result.support_recovery_count,
                    "timing": asdict(result.timing),
                }
            )
            print(
                f"task={task_index:03d} seed={seed_index} "
                f"condition={condition.condition_id} "
                f"trajectory_seconds={runtime:.3f} "
                f"task_elapsed_seconds={time.perf_counter() - task_started:.3f}",
                flush=True,
            )

    output = (
        results_root
        / output_run
        / STUDY
        / "partials"
        / f"task_{task_index:03d}.json"
    )
    write_json(
        output,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "task_index": task_index,
            "task_specification": asdict(specification),
            "graph_index": graph_index,
            "node_count": node_count,
            "conditions": _condition_payloads(),
            "response_protocol": _response_protocol(),
            "response_seed_count": response_seed_count,
            "soft_confidence_cap": SOFT_CONFIDENCE_CAP,
            "particle_count": PARTICLE_COUNT,
            "initialization_method": INITIALIZATION_METHOD,
            "initialization_source": INITIALIZATION_SOURCE,
            "query_families": QUERY_FAMILIES,
            "query_pool_size": len(queries),
            "nonredundant_query_pool_size": len(queries) - len(links) // 2,
            "source_synthetic_run_id": source_synthetic_run,
            "source_data_path": str(source_path),
            "source_data_sha256": sha256(source_path),
            "baseline": {
                "metrics": {key: float(baseline[key]) for key in METRIC_KEYS},
                "distribution": _serialize(initial_distribution),
                "reference_support_size": len(reference_initial),
            },
            "rows": rows,
            "runs": runs,
            "runtime_seconds": time.perf_counter() - task_started,
            "command": list(sys.argv),
        },
        exclusive=True,
    )
    print(output, flush=True)
    return output


def _mean_se(values: Sequence[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    return (
        statistics.fmean(finite),
        statistics.stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else 0.0,
    )


def _validate_partial(
    partial: Mapping[str, Any],
    task_index: int,
    source_synthetic_run: str,
) -> None:
    expected = asdict(task_specification(task_index))
    conditions = _condition_payloads()
    if (
        partial.get("study") != STUDY
        or int(partial.get("study_version", -1)) != STUDY_VERSION
        or int(partial.get("task_index", -1)) != task_index
        or partial.get("task_specification") != expected
        or partial.get("conditions") != conditions
        or partial.get("response_protocol") != _response_protocol()
        or int(partial.get("response_seed_count", -1)) != RESPONSE_SEED_COUNT
        or float(partial.get("soft_confidence_cap", -1.0))
        != SOFT_CONFIDENCE_CAP
        or int(partial.get("particle_count", -1)) != PARTICLE_COUNT
        or partial.get("query_families") != list(QUERY_FAMILIES)
        or partial.get("source_synthetic_run_id") != source_synthetic_run
    ):
        raise ValueError(f"partial {task_index} differs from the locked protocol")


def _scm_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    baseline_by_key: Mapping[tuple[int, int], Mapping[str, float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, int], list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        key = (
            int(row["graph_index"]),
            int(row["sample_size"]),
            str(row["condition_id"]),
            int(row["budget"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    numeric = (
        *METRIC_KEYS,
        "runtime_seconds",
        "realized_accuracy",
        "mean_reported_confidence",
        "reported_calibration_gap",
        "reported_brier_score",
        "mean_effective_confidence",
        "effective_calibration_gap",
        "effective_brier_score",
        "negative_transfer_ed",
        "negative_transfer_es",
        "adjacency_f1_loss",
        "orientation_f1_loss",
    )
    for (graph_index, sample_size, condition_id, budget), rows in sorted(grouped.items()):
        if len(rows) != RESPONSE_SEED_COUNT:
            raise ValueError(
                f"SCM condition lacks exactly {RESPONSE_SEED_COUNT} trajectories"
            )
        if any(row.get("status") != "success" for row in rows):
            raise ValueError("the soft-evidence protocol contains a failed trajectory")
        record: dict[str, Any] = {
            "graph_index": graph_index,
            "sample_size": sample_size,
            "condition_id": condition_id,
            "actual_accuracy": float(rows[0]["actual_accuracy"]),
            "reported_confidence": float(rows[0]["reported_confidence"]),
            "effective_confidence": float(rows[0]["effective_confidence"]),
            "treatment": str(rows[0]["treatment"]),
            "policy": str(rows[0]["policy"]),
            "budget": budget,
            "trajectory_count": len(rows),
            "negative_transfer_rate": statistics.fmean(
                float(row["negative_transfer"]) for row in rows
            ),
        }
        for key in numeric:
            values = [
                float(row[key])
                for row in rows
                if row.get(key) is not None and math.isfinite(float(row[key]))
            ]
            record[key] = statistics.fmean(values) if values else None
        output.append(record)

    for (graph_index, sample_size), baseline in sorted(baseline_by_key.items()):
        output.append(
            {
                "graph_index": graph_index,
                "sample_size": sample_size,
                "condition_id": "data_only_pc_ges",
                "actual_accuracy": None,
                "reported_confidence": None,
                "effective_confidence": None,
                "treatment": "none",
                "policy": "none",
                "budget": 0,
                "trajectory_count": 0,
                "negative_transfer_rate": 0.0,
                **{key: float(baseline[key]) for key in METRIC_KEYS},
                "runtime_seconds": 0.0,
                "realized_accuracy": None,
                "mean_reported_confidence": None,
                "reported_calibration_gap": None,
                "reported_brier_score": None,
                "mean_effective_confidence": None,
                "effective_calibration_gap": None,
                "effective_brier_score": None,
                "negative_transfer_ed": 0.0,
                "negative_transfer_es": 0.0,
                "adjacency_f1_loss": 0.0,
                "orientation_f1_loss": 0.0,
            }
        )
    return output


def _summary_rows(
    scm_rows: Sequence[Mapping[str, Any]], graph_indices: Sequence[int]
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (int(row["sample_size"]), str(row["condition_id"]), int(row["budget"]))
            for row in scm_rows
        }
    )
    metrics = (
        *METRIC_KEYS,
        "runtime_seconds",
        "realized_accuracy",
        "mean_reported_confidence",
        "reported_calibration_gap",
        "reported_brier_score",
        "mean_effective_confidence",
        "effective_calibration_gap",
        "effective_brier_score",
        "negative_transfer_ed",
        "negative_transfer_es",
        "adjacency_f1_loss",
        "orientation_f1_loss",
        "negative_transfer_rate",
    )
    output: list[dict[str, Any]] = []
    for sample_size, condition_id, budget in keys:
        rows = [
            row
            for row in scm_rows
            if int(row["sample_size"]) == sample_size
            and row["condition_id"] == condition_id
            and int(row["budget"]) == budget
        ]
        if len(rows) != len(graph_indices):
            raise ValueError("summary cell does not cover every selected SCM")
        record: dict[str, Any] = {
            "sample_size": sample_size,
            "condition_id": condition_id,
            "actual_accuracy": rows[0]["actual_accuracy"],
            "reported_confidence": rows[0]["reported_confidence"],
            "effective_confidence": rows[0]["effective_confidence"],
            "treatment": rows[0]["treatment"],
            "policy": rows[0]["policy"],
            "budget": budget,
            "scm_count": len(rows),
            "trajectory_count": sum(int(row["trajectory_count"]) for row in rows),
        }
        for key in metrics:
            values = [
                float(row[key])
                for row in rows
                if row.get(key) is not None and math.isfinite(float(row[key]))
            ]
            mean, standard_error = _mean_se(values)
            record[f"{key}_mean"] = mean
            record[f"{key}_se"] = standard_error
        output.append(record)
    return output


def _paired_contrasts(
    scm_rows: Sequence[Mapping[str, Any]], graph_indices: Sequence[int]
) -> list[dict[str, Any]]:
    index = {
        (
            int(row["sample_size"]),
            str(row["condition_id"]),
            int(row["budget"]),
            int(row["graph_index"]),
        ): row
        for row in scm_rows
    }
    comparisons = (
        (
            "calibrated70-minus-overconfident40",
            "iid_calibrated_70",
            "iid_overconfident_40_as_70",
        ),
        (
            "underconfident100-minus-calibrated70",
            "iid_underconfident_100_as_70",
            "iid_calibrated_70",
        ),
        (
            "overconfident40-minus-data-only",
            "iid_overconfident_40_as_70",
            "data_only_pc_ges",
        ),
        (
            "calibrated70-minus-data-only",
            "iid_calibrated_70",
            "data_only_pc_ges",
        ),
        (
            "underconfident100-minus-data-only",
            "iid_underconfident_100_as_70",
            "data_only_pc_ges",
        ),
    )
    metrics = (
        "expected_distance",
        "energy_score",
        "adjacency_f1",
        "orientation_f1",
    )
    output: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        for budget in SOFT_BUDGETS:
            for name, first_condition, second_condition in comparisons:
                for metric in metrics:
                    first = [
                        float(index[(sample_size, first_condition, budget, graph)][metric])
                        for graph in graph_indices
                    ]
                    second = [
                        float(
                            index[
                                (
                                    sample_size,
                                    second_condition,
                                    0 if second_condition == "data_only_pc_ges" else budget,
                                    graph,
                                )
                            ][metric]
                        )
                        for graph in graph_indices
                    ]
                    output.append(
                        {
                            "sample_size": sample_size,
                            "budget": budget,
                            "comparison": name,
                            "first_condition": first_condition,
                            "second_condition": second_condition,
                            "metric": metric,
                            **_paired_bootstrap(
                                first,
                                second,
                                stable_seed(
                                    STUDY,
                                    "paired-bootstrap",
                                    sample_size,
                                    budget,
                                    name,
                                    metric,
                                ),
                            ),
                        }
                    )
    return output


def _aggregate_selected(
    output_run: str,
    results_root: Path,
    task_indices: Sequence[int],
    graph_indices: Sequence[int],
    output_root: Path,
    *,
    source_synthetic_run: str,
    aggregation_status: str,
) -> Path:
    root = results_root / output_run / STUDY
    raw_rows: list[dict[str, Any]] = []
    baseline_by_key: dict[tuple[int, int], dict[str, float]] = {}
    references: list[dict[str, Any]] = []
    query_counts: dict[tuple[int, str, str], int] = {}
    trajectory_counts: dict[tuple[int, str], int] = {}

    for task_index in task_indices:
        path = root / "partials" / f"task_{task_index:03d}.json"
        partial = read_json(path)
        _validate_partial(partial, task_index, source_synthetic_run)
        specification = task_specification(task_index)
        key = (specification.source_task_index, specification.sample_size)
        baseline = {
            metric: float(partial["baseline"]["metrics"][metric])
            for metric in METRIC_KEYS
        }
        baseline_by_key[key] = baseline
        raw_rows.extend(dict(row) for row in partial["rows"])
        references.append(
            {
                "task_index": task_index,
                "partial_path": str(path),
                "partial_sha256": sha256(path),
                "source_path": partial["source_data_path"],
                "source_sha256": partial["source_data_sha256"],
            }
        )
        for run in partial["runs"]:
            condition_id = str(run["condition_id"])
            trajectory_key = (specification.sample_size, condition_id)
            trajectory_counts[trajectory_key] = trajectory_counts.get(trajectory_key, 0) + 1
            for response in run["responses"]:
                kind = str(response["query"]["kind"])
                count_key = (*trajectory_key, kind)
                query_counts[count_key] = query_counts.get(count_key, 0) + 1

    scm_rows = _scm_rows(raw_rows, baseline_by_key)
    summary = _summary_rows(scm_rows, graph_indices)
    contrasts = _paired_contrasts(scm_rows, graph_indices)
    query_mix = [
        {
            "sample_size": sample_size,
            "condition_id": condition_id,
            "query_kind": kind,
            "query_count": count,
            "trajectory_count": trajectory_counts[(sample_size, condition_id)],
            "mean_queries_per_trajectory": count
            / trajectory_counts[(sample_size, condition_id)],
        }
        for (sample_size, condition_id, kind), count in sorted(query_counts.items())
    ]
    write_csv(output_root / "raw_results.csv", raw_rows)
    write_csv(output_root / "scm_summary.csv", scm_rows)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "paired_contrasts.csv", contrasts)
    from cd_llm.metrics.units import metric_metadata
    for name in ("raw_results", "scm_summary", "summary", "paired_contrasts"):
        write_json(output_root / f"{name}.csv.metrics.json", metric_metadata(20))
    write_csv(output_root / "query_mix.csv", query_mix)
    write_json(output_root / "references.json", references)
    write_json(
        output_root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "aggregation_status": aggregation_status,
            "run_id": output_run,
            "source_synthetic_run_id": source_synthetic_run,
            "task_indices": tuple(task_indices),
            "graph_indices": tuple(graph_indices),
            "sample_sizes": SAMPLE_SIZES,
            "conditions": _condition_payloads(),
            "response_protocol": _response_protocol(),
            "query_families": QUERY_FAMILIES,
            "soft_budgets": SOFT_BUDGETS,
            "soft_confidence_cap": SOFT_CONFIDENCE_CAP,
            "new_claim_treatment": "soft_evidence",
            "response_seed_count": RESPONSE_SEED_COUNT,
            "particle_count": PARTICLE_COUNT,
            "initialization_method": INITIALIZATION_METHOD,
            "initialization_source": INITIALIZATION_SOURCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "task_count": len(task_indices),
            "new_trajectory_count": len(task_indices)
            * len(CONDITIONS)
            * RESPONSE_SEED_COUNT,
            "analyzed_active_trajectory_count": len(task_indices)
            * len(CONDITIONS)
            * RESPONSE_SEED_COUNT,
            "data_baseline_count": len(task_indices),
        },
    )
    print(output_root, flush=True)
    return output_root


def aggregate(
    output_run: str,
    results_root: Path,
    *,
    source_synthetic_run: str = SOURCE_SYNTHETIC_RUN_ID,
) -> Path:
    tasks = task_specifications()
    return _aggregate_selected(
        output_run,
        results_root,
        tuple(range(len(tasks))),
        TABLE41_COMMON_TASK_INDICES,
        results_root / output_run / STUDY,
        source_synthetic_run=source_synthetic_run,
        aggregation_status="complete",
    )


def aggregate_available(
    output_run: str,
    results_root: Path,
    *,
    source_synthetic_run: str = SOURCE_SYNTHETIC_RUN_ID,
) -> Path:
    root = results_root / output_run / STUDY
    tasks = task_specifications()
    available = {
        index
        for index in range(len(tasks))
        if (root / "partials" / f"task_{index:03d}.json").is_file()
    }
    graph_indices = tuple(
        graph
        for graph in TABLE41_COMMON_TASK_INDICES
        if all(
            index in available
            for index, specification in enumerate(tasks)
            if specification.source_task_index == graph
        )
    )
    if not graph_indices:
        raise ValueError("no SCM is complete at both sample sizes")
    selected = tuple(
        index
        for index, specification in enumerate(tasks)
        if specification.source_task_index in set(graph_indices)
    )
    return _aggregate_selected(
        output_run,
        results_root,
        selected,
        graph_indices,
        root / "available_common_complete",
        source_synthetic_run=source_synthetic_run,
        aggregation_status="provisional_available_common_complete",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("compute", "aggregate", "aggregate-available"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--source-synthetic-run-id", default=SOURCE_SYNTHETIC_RUN_ID)
    parser.add_argument("--response-seed-count", type=int, default=RESPONSE_SEED_COUNT)
    parser.add_argument("--results-root", type=Path, default=WORKSPACE_ROOT / "results")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    source_run = run_id(arguments.source_synthetic_run_id)
    if arguments.stage == "compute":
        if arguments.task_index is None:
            raise ValueError("--task-index is required for compute")
        compute(
            arguments.task_index,
            output_run,
            arguments.results_root,
            source_synthetic_run=source_run,
            response_seed_count=arguments.response_seed_count,
        )
    elif arguments.stage == "aggregate":
        aggregate(
            output_run,
            arguments.results_root,
            source_synthetic_run=source_run,
        )
    else:
        aggregate_available(
            output_run,
            arguments.results_root,
            source_synthetic_run=source_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
