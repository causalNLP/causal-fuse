"""Paired active-query ablation for the DAG panel of Table 4.2."""

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

from cd_llm.core.finite_support import Graph, stable_seed
from cd_llm.experiments.claim_fuse_smc_eig import (
    INITIALIZATION_METHOD,
    INITIALIZATION_SOURCE,
    PARTICLE_COUNT,
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
from cd_llm.methods.active_query import (
    ActiveQueryResult,
    ClaimQuery,
    QueryFamily,
    active_query_smc,
    ancestry_ordering_query_links,
    dag_query_pool,
    query_values,
)
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "synthetic_dag_claim_ablation"
STUDY_VERSION = 1
SOURCE_SYNTHETIC_RUN_ID = "20260816T145618Z"
SAMPLE_SIZES = (200, 1_000)
BUDGETS = (1, 3, 5, 10, 20)
CONFIDENCE = 1.0
EVIDENCE_TREATMENT = "hard_constraint"
ALGORITHM_SEED_COUNT = 20
BOOTSTRAP_RESAMPLES = 10_000
LOCKED_SOURCE_TASK_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    13,
    14,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
)
FAMILY_CONFIGS: tuple[tuple[str, tuple[QueryFamily, ...]], ...] = (
    ("local_edge", ("local_edge",)),
    ("causal_order", ("causal_order",)),
    ("ancestral_relation", ("ancestry",)),
    ("all_observable", ("local_edge", "causal_order", "ancestry")),
)
POLICIES = ("eig", "random")
METRIC_KEYS = (
    "expected_distance",
    "energy_score",
    "adjacency_f1",
    "orientation_f1",
    "entropy",
)


@dataclass(frozen=True)
class TaskSpecification:
    source_task_index: int
    sample_size: int
    family: str
    query_families: tuple[QueryFamily, ...]
    policy: str


def task_specifications() -> tuple[TaskSpecification, ...]:
    return tuple(
        TaskSpecification(
            source_task_index, sample_size, family, query_families, policy
        )
        for source_task_index in LOCKED_SOURCE_TASK_INDICES
        for sample_size in SAMPLE_SIZES
        for family, query_families in FAMILY_CONFIGS
        for policy in POLICIES
    )


def task_specification(task_index: int) -> TaskSpecification:
    tasks = task_specifications()
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task index must lie in [0, {len(tasks) - 1}]")
    return tasks[task_index]


def _truth_answer_provider(truth: Graph):
    def answer(query: ClaimQuery) -> int:
        return int(query_values((truth,), query)[0])

    return answer


def _validate_source(
    payload: Mapping[str, Any],
    source_run: str,
    source_task_index: int,
    sample_size: int,
) -> tuple[int, int]:
    if payload.get("run_id") != source_run:
        raise ValueError("synthetic source partial run ID differs")
    if int(payload.get("task_index", -1)) != source_task_index:
        raise ValueError("synthetic source partial task index differs")
    graph_index = int(payload["graph_index"])
    if graph_index != source_task_index:
        raise ValueError("locked source task and graph indices differ")
    protocol = payload["protocol"]
    if (
        int(protocol.get("node_count", -1)) != 20
        or float(protocol.get("expected_degree", -1.0)) != 2.0
        or int(protocol.get("master_seed", -1)) != 20260810
    ):
        raise ValueError("synthetic source protocol differs from the locked linear study")
    node_count = int(protocol["node_count"])
    _pc_ges_data_distribution(payload, sample_size, node_count)
    if str(sample_size) not in payload.get("nested_data", {}):
        raise ValueError("synthetic source lacks the locked nested dataset")
    return graph_index, node_count


def _run_seed(graph_index: int, sample_size: int, seed_index: int) -> int:
    return stable_seed("table42-dag-active", graph_index, sample_size, seed_index)


def _trace(result: ActiveQueryResult) -> list[dict[str, Any]]:
    return [
        {
            "budget": step.budget,
            "query": asdict(step.query),
            "predicate_probability": step.predicate_probability,
            "expected_information_gain": step.expected_information_gain,
            "answer": step.answer,
            "confidence": step.confidence,
            "reported_confidence": step.reported_confidence,
            "effective_confidence": step.effective_confidence,
            "treatment": step.treatment,
            "selection_confidence": step.selection_confidence,
            "selection_reported_confidence": step.selection_reported_confidence,
            "selection_effective_confidence": step.selection_effective_confidence,
            "selection_treatment": step.selection_treatment,
            "ess_before": step.ess_before,
            "ess_after": step.ess_after,
            "resampled": step.resampled,
            "support_recovered": step.support_recovered,
            "cumulative_seconds": step.cumulative_seconds,
            "unique_positive_graphs_after": step.unique_positive_graphs_after,
            "graph_mass_ess_after": step.graph_mass_ess_after,
            "rejuvenation_attempted_moves": step.rejuvenation_attempted_moves,
            "rejuvenation_accepted_moves": step.rejuvenation_accepted_moves,
            "rejuvenation_hard_constraint_rejections": (
                step.rejuvenation_hard_constraint_rejections
            ),
            "support_recovery_projection_attempts": (
                step.support_recovery_projection_attempts
            ),
            "support_recovery_unique_graphs": step.support_recovery_unique_graphs,
        }
        for step in result.steps
    ]


def _audit_result(
    result: ActiveQueryResult, links: Mapping[str, str]
) -> dict[str, int]:
    selected = {step.query.query_id for step in result.steps}
    selected_linked_pairs = tuple(
        (query_id, linked)
        for query_id, linked in links.items()
        if query_id < linked and query_id in selected and linked in selected
    )
    linked_pair_violations = len(selected_linked_pairs)
    claim_violations = 0
    positive_slots = 0
    for graph, weight in zip(result.particles, result.weights):
        if weight <= 0.0:
            continue
        positive_slots += 1
        claim_violations += sum(
            int(query_values((graph,), step.query)[0]) != int(step.answer)
            for step in result.steps
        )
    if linked_pair_violations or claim_violations:
        raise RuntimeError(
            "active-query audit found "
            f"{linked_pair_violations} linked-pair and "
            f"{claim_violations} hard-support violations; "
            f"selected linked pairs={selected_linked_pairs}"
        )
    return {
        "positive_particle_slots": positive_slots,
        "hard_claim_violations": claim_violations,
        "linked_pair_violations": linked_pair_violations,
    }


def _row(
    specification: TaskSpecification,
    graph_index: int,
    seed_index: int,
    algorithm_seed: int,
    budget: int,
    metrics: Mapping[str, float],
    result: ActiveQueryResult,
) -> dict[str, Any]:
    return {
        "source_task_index": specification.source_task_index,
        "graph_index": graph_index,
        "sample_size": specification.sample_size,
        "family": specification.family,
        "policy": specification.policy,
        "seed_index": seed_index,
        "algorithm_seed": algorithm_seed,
        "budget": budget,
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        "active_runtime_seconds": float(result.cumulative_seconds[budget]),
        "resampling_count": result.resampling_count,
        "support_recovery_count": result.support_recovery_count,
    }


def compute(
    task_index: int,
    output_run: str,
    results_root: Path,
    *,
    source_synthetic_run: str = SOURCE_SYNTHETIC_RUN_ID,
    algorithm_seed_count: int = ALGORITHM_SEED_COUNT,
) -> Path:
    if algorithm_seed_count != ALGORITHM_SEED_COUNT:
        raise ValueError("the locked Table 4.2 protocol requires exactly 20 seeds")
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
    data = np.asarray(
        source["nested_data"][str(specification.sample_size)], dtype=float
    )
    score = GaussianBICScore(data, standardize_data=True, max_parents=12)
    queries = dag_query_pool(node_count, specification.query_families)
    links = ancestry_ordering_query_links(queries)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed_index in range(algorithm_seed_count):
        algorithm_seed = _run_seed(
            graph_index, specification.sample_size, seed_index
        )
        result = active_query_smc(
            None,
            score,
            _truth_answer_provider(truth),
            query_pool=queries,
            linked_query_ids=links,
            budgets=BUDGETS,
            confidence=CONFIDENCE,
            treatment=EVIDENCE_TREATMENT,
            policy=specification.policy,
            seed=algorithm_seed,
            initial_particles=initial_particles,
            initial_weights=initial_weights,
        )
        audit = _audit_result(result, links)
        metrics_by_budget: dict[str, dict[str, float]] = {}
        distributions: dict[str, list[dict[str, Any]]] = {}
        for budget in BUDGETS:
            metrics = _metrics(result.distributions[budget], truth)
            rows.append(
                _row(
                    specification,
                    graph_index,
                    seed_index,
                    algorithm_seed,
                    budget,
                    metrics,
                    result,
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
                "seed_index": seed_index,
                "algorithm_seed": algorithm_seed,
                "trace": _trace(result),
                "metrics": metrics_by_budget,
                "seed_zero_distributions": distributions,
                "resampling_count": result.resampling_count,
                "support_recovery_count": result.support_recovery_count,
                "timing": asdict(result.timing),
                "audit": audit,
            }
        )

    output = (
        results_root
        / output_run
        / STUDY
        / "partials"
        / f"task_{task_index:04d}.json"
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
            "query_pool_size": len(queries),
            "nonredundant_query_pool_size": len(queries) - len(links) // 2,
            "linked_query_count": len(links),
            "budgets": BUDGETS,
            "confidence": CONFIDENCE,
            "evidence_treatment": EVIDENCE_TREATMENT,
            "algorithm_seed_count": algorithm_seed_count,
            "particle_count": PARTICLE_COUNT,
            "initialization_method": INITIALIZATION_METHOD,
            "initialization_source": INITIALIZATION_SOURCE,
            "source_synthetic_run_id": source_synthetic_run,
            "source_data_path": str(source_path),
            "source_data_sha256": sha256(source_path),
            "baseline": {
                "metrics": {
                    key: float(value)
                    for key, value in _metrics(initial_distribution, truth).items()
                },
                "distribution": _serialize(initial_distribution),
                "reference_support_size": len(reference_initial),
            },
            "rows": rows,
            "runs": runs,
            "runtime_seconds": time.perf_counter() - started,
            "command": list(sys.argv),
        },
        exclusive=True,
    )
    print(output, flush=True)
    return output


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
    )


def _paired_bootstrap(
    first: Sequence[float], second: Sequence[float], seed: int
) -> dict[str, float | int]:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if (
        first_array.shape != second_array.shape
        or first_array.ndim != 1
        or not len(first_array)
    ):
        raise ValueError("paired bootstrap inputs must have the same positive length")
    differences = first_array - second_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences))
    )
    means = np.mean(differences[indices], axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5), method="linear")
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "pair_count": len(differences),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def _validate_partial(
    partial: Mapping[str, Any],
    task_index: int,
    source_synthetic_run: str,
) -> None:
    expected = task_specification(task_index)
    expected_payload = asdict(expected)
    expected_payload["query_families"] = list(expected.query_families)
    if (
        partial.get("study") != STUDY
        or int(partial.get("study_version", -1)) != STUDY_VERSION
        or int(partial.get("task_index", -1)) != task_index
        or partial.get("task_specification") != expected_payload
        or partial.get("source_synthetic_run_id") != source_synthetic_run
        or int(partial.get("algorithm_seed_count", -1)) != ALGORITHM_SEED_COUNT
        or float(partial.get("confidence", -1.0)) != CONFIDENCE
        or int(partial.get("particle_count", -1)) != PARTICLE_COUNT
        or tuple(partial.get("budgets", ())) != BUDGETS
    ):
        raise ValueError(f"partial {task_index} differs from the locked protocol")


def _scm_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        key = (
            int(row["graph_index"]),
            int(row["sample_size"]),
            str(row["family"]),
            str(row["policy"]),
            int(row["budget"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (graph_index, sample_size, family, policy, budget), group in sorted(
        grouped.items()
    ):
        if len(group) != ALGORITHM_SEED_COUNT:
            raise ValueError("SCM-policy cell lacks exactly 20 algorithm seeds")
        output.append(
            {
                "graph_index": graph_index,
                "sample_size": sample_size,
                "family": family,
                "policy": policy,
                "budget": budget,
                "seed_count": len(group),
                **{
                    key: statistics.fmean(float(row[key]) for row in group)
                    for key in (*METRIC_KEYS, "active_runtime_seconds")
                },
            }
        )
    return output


def _summary_rows(
    scm_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    graph_indices: Sequence[int] = LOCKED_SOURCE_TASK_INDICES,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        baseline = [row for row in baseline_rows if int(row["sample_size"]) == sample_size]
        record: dict[str, Any] = {
            "sample_size": sample_size,
            "family": "none",
            "policy": "data_only",
            "budget": 0,
            "scm_count": len(baseline),
            "algorithm_trajectory_count": 0,
        }
        for key in METRIC_KEYS:
            mean, se = _mean_se([float(row[key]) for row in baseline])
            record[f"{key}_mean"] = mean
            record[f"{key}_se"] = se
        record["active_runtime_seconds_mean"] = 0.0
        record["active_runtime_seconds_se"] = 0.0
        output.append(record)
        for family, _ in FAMILY_CONFIGS:
            for policy in POLICIES:
                for budget in BUDGETS:
                    group = [
                        row
                        for row in scm_rows
                        if int(row["sample_size"]) == sample_size
                        and row["family"] == family
                        and row["policy"] == policy
                        and int(row["budget"]) == budget
                    ]
                    if len(group) != len(graph_indices):
                        raise ValueError("summary cell lacks every selected SCM")
                    record = {
                        "sample_size": sample_size,
                        "family": family,
                        "policy": policy,
                        "budget": budget,
                        "scm_count": len(group),
                        "algorithm_trajectory_count": len(group) * ALGORITHM_SEED_COUNT,
                    }
                    for key in (*METRIC_KEYS, "active_runtime_seconds"):
                        mean, se = _mean_se([float(row[key]) for row in group])
                        record[f"{key}_mean"] = mean
                        record[f"{key}_se"] = se
                    output.append(record)
    return output


def _paired_contrasts(
    scm_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    graph_indices: Sequence[int] = LOCKED_SOURCE_TASK_INDICES,
) -> list[dict[str, Any]]:
    index = {
        (
            int(row["sample_size"]),
            str(row["family"]),
            str(row["policy"]),
            int(row["budget"]),
            int(row["graph_index"]),
        ): row
        for row in scm_rows
    }
    baseline_index = {
        (int(row["sample_size"]), int(row["graph_index"])): row
        for row in baseline_rows
    }
    comparisons: list[tuple[int, int, str, str, str, str]] = []
    for sample_size in SAMPLE_SIZES:
        for budget in BUDGETS:
            for family, _ in FAMILY_CONFIGS:
                comparisons.append(
                    (sample_size, budget, family, "eig", family, "random")
                )
                comparisons.append(
                    (sample_size, budget, family, "eig", "none", "data_only")
                )
            for family in ("local_edge", "causal_order", "ancestral_relation"):
                comparisons.append(
                    (sample_size, budget, "all_observable", "eig", family, "eig")
                )
    output: list[dict[str, Any]] = []
    graph_indices = tuple(graph_indices)
    for (
        sample_size,
        budget,
        first_family,
        first_policy,
        second_family,
        second_policy,
    ) in comparisons:
        for metric in METRIC_KEYS:
            first = [
                float(
                    index[
                        (sample_size, first_family, first_policy, budget, graph)
                    ][metric]
                )
                for graph in graph_indices
            ]
            if second_family == "none":
                second = [
                    float(baseline_index[(sample_size, graph)][metric])
                    for graph in graph_indices
                ]
            else:
                second = [
                    float(
                        index[
                            (
                                sample_size,
                                second_family,
                                second_policy,
                                budget,
                                graph,
                            )
                        ][metric]
                    )
                    for graph in graph_indices
                ]
            comparison = f"{first_family}:{first_policy}-minus-{second_family}:{second_policy}"
            output.append(
                {
                    "sample_size": sample_size,
                    "budget": budget,
                    "comparison": comparison,
                    "metric": metric,
                    **_paired_bootstrap(
                        first,
                        second,
                        stable_seed(
                            "table42-dag-paired-bootstrap",
                            sample_size,
                            budget,
                            comparison,
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
    source_synthetic_run: str = SOURCE_SYNTHETIC_RUN_ID,
    aggregation_status: str,
    unavailable_task_indices: Sequence[int] = (),
) -> Path:
    root = results_root / output_run / STUDY
    raw_rows: list[dict[str, Any]] = []
    baseline_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    query_counts: dict[tuple[int, str, str, str], int] = {}
    trajectory_counts: dict[tuple[int, str, str], int] = {}
    for task_index in task_indices:
        partial = read_json(root / "partials" / f"task_{task_index:04d}.json")
        _validate_partial(partial, task_index, source_synthetic_run)
        raw_rows.extend(dict(row) for row in partial["rows"])
        key = (
            int(partial["graph_index"]),
            int(partial["task_specification"]["sample_size"]),
        )
        metrics = partial["baseline"]["metrics"]
        candidate = {
            "graph_index": key[0],
            "sample_size": key[1],
            **{metric: float(metrics[metric]) for metric in METRIC_KEYS},
        }
        previous = baseline_by_key.setdefault(key, candidate)
        if previous != candidate:
            raise ValueError("baseline differs across family or policy slices")
        specification = partial["task_specification"]
        mix_key = (
            int(specification["sample_size"]),
            str(specification["family"]),
            str(specification["policy"]),
        )
        for run in partial["runs"]:
            trajectory_counts[mix_key] = trajectory_counts.get(mix_key, 0) + 1
            for step in run["trace"]:
                query_key = (*mix_key, str(step["query"]["kind"]))
                query_counts[query_key] = query_counts.get(query_key, 0) + 1
            diagnostics.append(
                {
                    "graph_index": int(partial["graph_index"]),
                    "sample_size": mix_key[0],
                    "family": mix_key[1],
                    "policy": mix_key[2],
                    "seed_index": int(run["seed_index"]),
                    "algorithm_seed": int(run["algorithm_seed"]),
                    "resampling_count": int(run["resampling_count"]),
                    "support_recovery_count": int(run["support_recovery_count"]),
                    **{f"audit_{key}": value for key, value in run["audit"].items()},
                    **{
                        f"timing_{key}": value
                        for key, value in run["timing"].items()
                    },
                }
            )

    scm_rows = _scm_rows(raw_rows)
    baseline_rows = [baseline_by_key[key] for key in sorted(baseline_by_key)]
    if len(baseline_rows) != len(graph_indices) * len(SAMPLE_SIZES):
        raise ValueError("baseline does not cover every selected SCM/sample-size pair")

    summary = _summary_rows(scm_rows, baseline_rows, graph_indices)
    table_rows = [row for row in summary if int(row["budget"]) in (0, 20)]
    contrasts = _paired_contrasts(scm_rows, baseline_rows, graph_indices)
    query_mix = []
    for (sample_size, family, policy, kind), count in sorted(query_counts.items()):
        trajectories = trajectory_counts[(sample_size, family, policy)]
        query_mix.append(
            {
                "sample_size": sample_size,
                "family": family,
                "policy": policy,
                "query_kind": kind,
                "query_count": count,
                "trajectory_count": trajectories,
                "mean_queries_per_trajectory": count / trajectories,
                "query_fraction": count / (trajectories * BUDGETS[-1]),
            }
        )

    write_csv(output_root / "raw_results.csv", raw_rows)
    write_csv(output_root / "scm_summary.csv", scm_rows)
    write_csv(output_root / "baseline.csv", baseline_rows)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "table42_dag.csv", table_rows)
    write_csv(output_root / "paired_contrasts.csv", contrasts)
    from cd_llm.metrics.units import metric_metadata
    for name in ("raw_results", "scm_summary", "baseline", "summary", "table42_dag", "paired_contrasts"):
        write_json(output_root / f"{name}.csv.metrics.json", metric_metadata(20))
    write_csv(output_root / "query_mix.csv", query_mix)
    write_csv(output_root / "diagnostics.csv", diagnostics)
    write_json(
        output_root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "aggregation_status": aggregation_status,
            "run_id": output_run,
            "source_synthetic_run_id": source_synthetic_run,
            "source_task_indices": tuple(graph_indices),
            "graph_count": len(graph_indices),
            "excluded_source_task_indices": tuple(
                index
                for index in LOCKED_SOURCE_TASK_INDICES
                if index not in set(graph_indices)
            ),
            "included_task_indices": tuple(task_indices),
            "unavailable_task_indices": tuple(unavailable_task_indices),
            "sample_sizes": SAMPLE_SIZES,
            "family_configs": dict(FAMILY_CONFIGS),
            "policies": POLICIES,
            "budgets": BUDGETS,
            "confidence": CONFIDENCE,
            "evidence_treatment": EVIDENCE_TREATMENT,
            "particle_count": PARTICLE_COUNT,
            "algorithm_seed_count": ALGORITHM_SEED_COUNT,
            "task_count": len(task_indices),
            "trajectory_count": len(task_indices) * ALGORITHM_SEED_COUNT,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "initialization_method": INITIALIZATION_METHOD,
            "initialization_source": INITIALIZATION_SOURCE,
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
    root = results_root / output_run / STUDY
    tasks = task_specifications()
    return _aggregate_selected(
        output_run,
        results_root,
        tuple(range(len(tasks))),
        LOCKED_SOURCE_TASK_INDICES,
        root,
        source_synthetic_run=source_synthetic_run,
        aggregation_status="complete",
    )


def aggregate_available(
    output_run: str,
    results_root: Path,
    *,
    source_synthetic_run: str = SOURCE_SYNTHETIC_RUN_ID,
) -> Path:
    """Aggregate the largest SCM set complete across every experimental cell."""
    root = results_root / output_run / STUDY
    tasks = task_specifications()
    available = {
        task_index
        for task_index in range(len(tasks))
        if (root / "partials" / f"task_{task_index:04d}.json").is_file()
    }
    graph_to_tasks = {
        graph_index: tuple(
            task_index
            for task_index, specification in enumerate(tasks)
            if specification.source_task_index == graph_index
        )
        for graph_index in LOCKED_SOURCE_TASK_INDICES
    }
    graph_indices = tuple(
        graph_index
        for graph_index, indices in graph_to_tasks.items()
        if all(task_index in available for task_index in indices)
    )
    if not graph_indices:
        raise ValueError("no source SCM is complete across every experimental cell")
    selected_graphs = set(graph_indices)
    task_indices = tuple(
        task_index
        for task_index, specification in enumerate(tasks)
        if specification.source_task_index in selected_graphs
    )
    unavailable = tuple(
        task_index for task_index in range(len(tasks)) if task_index not in available
    )
    return _aggregate_selected(
        output_run,
        results_root,
        task_indices,
        graph_indices,
        root / "available_common_complete",
        source_synthetic_run=source_synthetic_run,
        aggregation_status="provisional_available_common_complete",
        unavailable_task_indices=unavailable,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired active claim-family ablation for the synthetic DAG panel."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("compute", "aggregate", "aggregate-available"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--source-synthetic-run-id", default=SOURCE_SYNTHETIC_RUN_ID
    )
    parser.add_argument(
        "--algorithm-seed-count", type=int, default=ALGORITHM_SEED_COUNT
    )
    parser.add_argument(
        "--results-root", type=Path, default=WORKSPACE_ROOT / "results"
    )
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
            algorithm_seed_count=arguments.algorithm_seed_count,
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
