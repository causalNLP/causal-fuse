"""Synthetic active-query evaluation for Claim-Fuse-SMC-EIG."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cd_llm.core.finite_support import (
    Graph,
    distribution_metrics,
    graph_edges,
    graph_from_edges,
    has_adjacency,
    has_edge,
    stable_seed,
)
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json
from cd_llm.methods.active_query import ActiveQueryResult, active_query_smc
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "claim_fuse_smc_eig"
STUDY_VERSION = 4
SAMPLE_SIZES = (200, 1_000)
BUDGETS = (1, 3, 5, 10, 20)
CONFIDENCE = 1.0
EVIDENCE_TREATMENT = "hard_constraint"
PARTICLE_COUNT = 384
INITIALIZATION_METHOD = "CEG-Fuse (PC+GES)"
INITIALIZATION_SOURCE = "pc+ges-completion-data-distribution"
RANDOM_SEED_COUNT = 20
TABLE41_COMMON_TASK_INDICES = (1, 2, 7, 12, 15, 34, 36, 40, 45, 47)


def _graph(record: Mapping[str, Any], node_count: int) -> Graph:
    return graph_from_edges(
        node_count,
        record.get("directed", ()),
        record.get("bidirected", ()),
        record.get("undirected", ()),
    )


def _distribution(records: Sequence[Mapping[str, Any]], node_count: int) -> dict[Graph, float]:
    result = {
        _graph(item["graph"], node_count): float(item["probability"])
        for item in records
    }
    total = math.fsum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("source SMC distribution does not sum to one")
    return {graph: mass / total for graph, mass in result.items()}


def _pc_ges_data_distribution(
    data_payload: Mapping[str, Any], sample_size: int, node_count: int
) -> dict[Graph, float]:
    try:
        artifact = data_payload["samples"][str(sample_size)]["methods"][
            INITIALIZATION_METHOD
        ]
    except KeyError as error:
        raise ValueError(
            f"synthetic n={sample_size} lacks the locked PC+GES initialization"
        ) from error
    if not bool(artifact.get("feasible", False)):
        raise ValueError(f"synthetic n={sample_size} PC+GES initialization is unavailable")
    records = tuple(artifact.get("graphs", ()))
    probabilities = tuple(
        artifact.get("diagnostics", {}).get("data_probabilities", ())
    )
    if len(records) != len(probabilities) or len(records) < 2:
        raise ValueError(
            f"synthetic n={sample_size} PC+GES graphs and data probabilities differ"
        )
    masses: dict[Graph, float] = {}
    for record, probability in zip(records, probabilities):
        mass = float(probability)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("PC+GES data probabilities must be finite and positive")
        graph = _graph(record, node_count)
        if graph.regime != "dag":
            raise ValueError("PC+GES initialization contains a non-DAG graph")
        masses[graph] = masses.get(graph, 0.0) + mass
    total = math.fsum(masses.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("PC+GES data probabilities do not sum to one")
    return {graph: mass / total for graph, mass in masses.items()}


def _initial_particle_population(
    distribution: Mapping[Graph, float], seed: int
) -> tuple[tuple[Graph, ...], np.ndarray, dict[Graph, float]]:
    items = tuple(sorted(distribution.items()))
    probabilities = np.asarray([mass for _, mass in items], dtype=float)
    probabilities /= np.sum(probabilities)
    rng = np.random.default_rng(seed)
    positions = (float(rng.random()) + np.arange(PARTICLE_COUNT)) / PARTICLE_COUNT
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, positions, side="right")
    particles = tuple(items[int(index)][0] for index in indices)
    weights = np.full(PARTICLE_COUNT, 1.0 / PARTICLE_COUNT, dtype=float)
    empirical: dict[Graph, float] = {}
    for graph in particles:
        empirical[graph] = empirical.get(graph, 0.0) + 1.0 / PARTICLE_COUNT
    return particles, weights, empirical


def _serialize(distribution: Mapping[Graph, float]) -> list[dict[str, Any]]:
    return [
        {"graph": graph_edges(graph), "probability": float(distribution[graph])}
        for graph in sorted(distribution)
    ]


def _metrics(distribution: Mapping[Graph, float], truth: Graph) -> dict[str, float]:
    graphs = tuple(sorted(distribution))
    probabilities = np.asarray([distribution[graph] for graph in graphs], dtype=float)
    return distribution_metrics(graphs, probabilities, truth, "dag")


def _row(
    graph_index: int,
    sample_size: int,
    policy: str,
    budget: int,
    metrics: Mapping[str, float],
    runtime_seconds: float,
    random_seed_count: int,
) -> dict[str, Any]:
    return {
        "graph_index": graph_index,
        "sample_size": sample_size,
        "policy": policy,
        "budget": budget,
        "expected_distance": metrics["expected_distance"],
        "energy_score": metrics["energy_score"],
        "adjacency_f1": metrics["adjacency_f1"],
        "orientation_f1": metrics["orientation_f1"],
        "entropy": metrics["entropy"],
        "runtime_seconds": runtime_seconds,
        "random_seed_count": random_seed_count,
    }


def _mean_metrics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = (
        "expected_distance",
        "energy_score",
        "adjacency_f1",
        "orientation_f1",
        "entropy",
    )
    return {key: statistics.fmean(float(value[key]) for value in values) for key in keys}


def _trace(result: ActiveQueryResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in result.steps:
        record = asdict(step)
        record["confidence"] = step.confidence
        record["selection_confidence"] = step.selection_confidence
        records.append(record)
    return records


def _truth_answer_provider(truth: Graph):
    def answer(query) -> int:
        if query.kind == "directed":
            return int(has_edge(truth, query.source, query.target))
        return int(has_adjacency(truth, query.source, query.target))

    return answer


def _source_data_path(
    results_root: Path, source_synthetic_run: str, task_index: int
) -> Path:
    candidates = (
        results_root
        / source_synthetic_run
        / "partials"
        / f"task_{task_index:03d}.json",
        results_root
        / source_synthetic_run
        / "test"
        / "partials"
        / f"task_{task_index:03d}.json",
    )
    matches = tuple(path for path in candidates if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one synthetic source partial for task {task_index}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_source(
    data_payload: Mapping[str, Any],
    source_synthetic_run: str,
    task_index: int,
) -> tuple[int, int]:
    if data_payload.get("run_id") != source_synthetic_run:
        raise ValueError("synthetic partial run ID differs")
    if int(data_payload.get("task_index", -1)) != task_index:
        raise ValueError("synthetic partial task index differs")
    graph_index = int(data_payload["graph_index"])
    node_count = int(data_payload["protocol"]["node_count"])
    for sample_size in SAMPLE_SIZES:
        _pc_ges_data_distribution(data_payload, sample_size, node_count)
    return graph_index, node_count


def compute(
    task_index: int,
    output_run: str,
    results_root: Path,
    *,
    source_synthetic_run: str,
    random_seed_count: int = RANDOM_SEED_COUNT,
) -> Path:
    if not 0 <= task_index < 50:
        raise ValueError("synthetic task index must lie in [0, 49]")
    if random_seed_count < 0:
        raise ValueError("random seed count must be nonnegative")
    source_data_path = _source_data_path(
        results_root, source_synthetic_run, task_index
    )
    data_payload = read_json(source_data_path)
    graph_index, node_count = _validate_source(
        data_payload, source_synthetic_run, task_index
    )
    truth = _graph(data_payload["target"], node_count)
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    task_started = time.perf_counter()

    for sample_size in SAMPLE_SIZES:
        reference_initial = _pc_ges_data_distribution(
            data_payload, sample_size, node_count
        )
        initial_particles, initial_weights, initial = _initial_particle_population(
            reference_initial,
            stable_seed("pc-ges-active-initialization", graph_index, sample_size),
        )
        data = np.asarray(data_payload["nested_data"][str(sample_size)], dtype=float)
        score = GaussianBICScore(data, standardize_data=True, max_parents=12)
        data_metrics = _metrics(initial, truth)
        rows.append(
            _row(graph_index, sample_size, "data-only", 0, data_metrics, 0.0, 0)
        )

        eig_started = time.perf_counter()
        eig = active_query_smc(
            None,
            score,
            _truth_answer_provider(truth),
            budgets=BUDGETS,
            confidence=CONFIDENCE,
            treatment=EVIDENCE_TREATMENT,
            policy="eig",
            seed=stable_seed("smc-eig", graph_index, sample_size),
            initial_particles=initial_particles,
            initial_weights=initial_weights,
        )
        eig_runtime = time.perf_counter() - eig_started
        eig_artifacts: dict[str, Any] = {}
        for budget in BUDGETS:
            metrics = _metrics(eig.distributions[budget], truth)
            rows.append(
                _row(
                    graph_index,
                    sample_size,
                    "eig",
                    budget,
                    metrics,
                    eig.cumulative_seconds[budget],
                    1,
                )
            )
            eig_artifacts[str(budget)] = {
                "metrics": metrics,
                "distribution": _serialize(eig.distributions[budget]),
            }

        random_metrics: dict[int, list[dict[str, float]]] = {
            budget: [] for budget in BUDGETS
        }
        random_runs: list[dict[str, Any]] = []
        random_runtime_by_budget: dict[int, list[float]] = {
            budget: [] for budget in BUDGETS
        }
        for random_seed in range(random_seed_count):
            random_started = time.perf_counter()
            random_result = active_query_smc(
                None,
                score,
                _truth_answer_provider(truth),
                budgets=BUDGETS,
                confidence=CONFIDENCE,
                treatment=EVIDENCE_TREATMENT,
                policy="random",
                seed=stable_seed(
                    "smc-random-query",
                    graph_index,
                    sample_size,
                    random_seed,
                ),
                initial_particles=initial_particles,
                initial_weights=initial_weights,
            )
            runtime = time.perf_counter() - random_started
            run_metrics: dict[str, Any] = {}
            for budget in BUDGETS:
                metrics = _metrics(random_result.distributions[budget], truth)
                random_metrics[budget].append(metrics)
                random_runtime_by_budget[budget].append(
                    random_result.cumulative_seconds[budget]
                )
                run_metrics[str(budget)] = metrics
            random_runs.append(
                {
                    "seed": random_seed,
                    "runtime_seconds": runtime,
                    "trace": _trace(random_result),
                    "metrics": run_metrics,
                    "resampling_count": random_result.resampling_count,
                }
            )
        if random_seed_count:
            for budget in BUDGETS:
                metrics = _mean_metrics(random_metrics[budget])
                rows.append(
                    _row(
                        graph_index,
                        sample_size,
                        "random",
                        budget,
                        metrics,
                        statistics.fmean(random_runtime_by_budget[budget]),
                        random_seed_count,
                    )
                )

        artifacts[str(sample_size)] = {
            "data_only": {
                "metrics": data_metrics,
                "distribution": _serialize(initial),
                "reference_metrics": _metrics(reference_initial, truth),
                "reference_distribution": _serialize(reference_initial),
                "reference_support_size": len(reference_initial),
                "particle_count": PARTICLE_COUNT,
            },
            "eig": {
                "runtime_seconds": eig_runtime,
                "trace": _trace(eig),
                "resampling_count": eig.resampling_count,
                "budgets": eig_artifacts,
            },
            "random": {
                "seed_count": random_seed_count,
                "runs": random_runs,
            },
        }

    output = (
        results_root
        / output_run
        / STUDY
        / "synthetic"
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
            "graph_index": graph_index,
            "confidence": CONFIDENCE,
            "evidence_treatment": EVIDENCE_TREATMENT,
            "initialization_source": INITIALIZATION_SOURCE,
            "initialization_method": INITIALIZATION_METHOD,
            "particle_count": PARTICLE_COUNT,
            "budgets": BUDGETS,
            "random_seed_count": random_seed_count,
            "source_synthetic_run_id": source_synthetic_run,
            "source_data_path": str(source_data_path),
            "source_data_sha256": sha256(source_data_path),
            "rows": rows,
            "artifacts": artifacts,
            "runtime_seconds": time.perf_counter() - task_started,
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


def aggregate(
    output_run: str,
    results_root: Path,
    source_synthetic_run: str,
    task_indices: Sequence[int] = TABLE41_COMMON_TASK_INDICES,
) -> Path:
    selected = tuple(int(index) for index in task_indices)
    if selected != TABLE41_COMMON_TASK_INDICES:
        raise ValueError("Table 4.1 aggregation requires its locked common-complete SCMs")
    root = results_root / output_run / STUDY / "synthetic"
    partials = [
        read_json(root / "partials" / f"task_{index:03d}.json")
        for index in selected
    ]
    if tuple(int(partial["task_index"]) for partial in partials) != selected:
        raise ValueError("active-query partial task indices do not match Table 4.1")
    for partial in partials:
        if partial.get("source_synthetic_run_id") != source_synthetic_run:
            raise ValueError("active-query partial synthetic source run differs")
        if partial.get("initialization_source") != INITIALIZATION_SOURCE:
            raise ValueError("active-query partial initialization source differs")
        if float(partial.get("confidence", -1.0)) != CONFIDENCE:
            raise ValueError("active-query partial confidence differs")
        if int(partial.get("particle_count", -1)) != PARTICLE_COUNT:
            raise ValueError("active-query partial particle count differs")
    graph_indices = tuple(int(partial["graph_index"]) for partial in partials)
    expected_graph_indices = tuple(
        int(read_json(_source_data_path(results_root, source_synthetic_run, index))["graph_index"])
        for index in selected
    )
    if graph_indices != expected_graph_indices:
        raise ValueError("active-query graph indices do not match Table 4.1")
    random_seed_counts = {int(partial["random_seed_count"]) for partial in partials}
    if len(random_seed_counts) != 1:
        raise ValueError("active-query partials use inconsistent random seed counts")
    random_seed_count = random_seed_counts.pop()
    rows = [dict(row) for partial in partials for row in partial["rows"]]
    summary: list[dict[str, Any]] = []
    metric_keys = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1", "entropy", "runtime_seconds")
    for sample_size in SAMPLE_SIZES:
        policy_budgets = [("data-only", (0,)), ("eig", BUDGETS)]
        if any(row["policy"] == "random" for row in rows):
            policy_budgets.append(("random", BUDGETS))
        for policy, budgets in policy_budgets:
            for budget in budgets:
                group = [row for row in rows if row["sample_size"] == sample_size and row["policy"] == policy and row["budget"] == budget]
                record: dict[str, Any] = {"sample_size": sample_size, "policy": policy, "budget": budget, "replicate_count": len(group)}
                for key in metric_keys:
                    mean, se = _mean_se([float(row[key]) for row in group])
                    record[f"{key}_mean"] = mean
                    record[f"{key}_se"] = se
                summary.append(record)
    paired: list[dict[str, Any]] = []
    if any(row["policy"] == "random" for row in rows):
        for sample_size in SAMPLE_SIZES:
            for budget in BUDGETS:
                eig_by_graph = {int(row["graph_index"]): row for row in rows if row["sample_size"] == sample_size and row["policy"] == "eig" and row["budget"] == budget}
                random_by_graph = {int(row["graph_index"]): row for row in rows if row["sample_size"] == sample_size and row["policy"] == "random" and row["budget"] == budget}
                if set(eig_by_graph) != set(random_by_graph):
                    raise ValueError("EIG and random policies have different replicate sets")
                record: dict[str, Any] = {"sample_size": sample_size, "budget": budget, "replicate_count": len(eig_by_graph)}
                for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1", "entropy"):
                    values = [float(eig_by_graph[index][key]) - float(random_by_graph[index][key]) for index in sorted(eig_by_graph)]
                    mean, se = _mean_se(values)
                    record[f"eig_minus_random_{key}_mean"] = mean
                    record[f"eig_minus_random_{key}_se"] = se
                paired.append(record)
    write_csv(root / "raw_results.csv", rows)
    write_csv(root / "summary.csv", summary)
    if paired:
        write_csv(root / "paired_eig_vs_random.csv", paired)
    write_json(
        root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "count": len(selected),
            "task_indices": selected,
            "graph_indices": expected_graph_indices,
            "confidence": CONFIDENCE,
            "evidence_treatment": EVIDENCE_TREATMENT,
            "initialization_source": INITIALIZATION_SOURCE,
            "initialization_method": INITIALIZATION_METHOD,
            "particle_count": PARTICLE_COUNT,
            "budgets": BUDGETS,
            "random_seed_count": random_seed_count,
            "source_synthetic_run": source_synthetic_run,
        },
    )
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PC+GES-initialized active causal-claim querying on synthetic SCMs."
    )
    parser.add_argument("--stage", choices=("compute", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--task-indices",
        default=",".join(str(index) for index in TABLE41_COMMON_TASK_INDICES),
    )
    parser.add_argument("--source-synthetic-run-id", required=True)
    parser.add_argument("--random-seed-count", type=int, default=RANDOM_SEED_COUNT)
    parser.add_argument("--results-root", type=Path, default=WORKSPACE_ROOT / "results")
    return parser


def _parse_task_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.replace(":", ",").split(",") if item)
    except ValueError as error:
        raise ValueError("task indices must be comma-separated integers") from error
    if not indices or len(indices) != len(set(indices)) or any(
        index < 0 or index >= 50 for index in indices
    ):
        raise ValueError("task indices must be unique values in [0, 49]")
    return indices


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    if arguments.stage == "compute":
        if arguments.task_index is None:
            raise ValueError("--task-index is required for compute")
        compute(
            arguments.task_index,
            output_run,
            arguments.results_root,
            source_synthetic_run=run_id(arguments.source_synthetic_run_id),
            random_seed_count=arguments.random_seed_count,
        )
    else:
        aggregate(
            output_run,
            arguments.results_root,
            run_id(arguments.source_synthetic_run_id),
            _parse_task_indices(arguments.task_indices),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
