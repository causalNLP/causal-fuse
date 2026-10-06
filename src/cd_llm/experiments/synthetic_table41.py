"""Recompute the matched synthetic benchmark reported in Table 4.1."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cd_llm.core.finite_support import (
    Graph,
    PartialGraph,
    distribution_metrics,
    graph_from_edges,
    is_acyclic,
    point_prediction_metrics,
)
from cd_llm.experiments.claim_fuse_smc_eig import (
    CONFIDENCE as ACTIVE_CONFIDENCE,
    INITIALIZATION_SOURCE as ACTIVE_INITIALIZATION_SOURCE,
    PARTICLE_COUNT as ACTIVE_PARTICLE_COUNT,
    TABLE41_COMMON_TASK_INDICES,
)
from cd_llm.experiments.common.artifacts import (
    json_value,
    read_json,
    run_id,
    sha256,
    write_csv,
    write_json,
)
from cd_llm.experiments.claim_fuse_smc_sidecar import (
    ARCHIVED_EVIDENCE_PROFILE,
    PASSIVE_CONFIDENCE,
    PRODUCTION_CONFIG,
    _claim_fingerprint,
    _confidence_records,
)
from cd_llm.metrics.units import metric_metadata
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "synthetic_table41"
STUDY_VERSION = 4
SAMPLE_SIZES = (200, 1_000)
REFERENCE_RUN_ID = "20260816T145618Z"
CAMML_RUN_ID = "20260817T032052Z"
ACTIVE_BUDGETS = (5, 20)
METHOD_SOURCES = (
    ("PC", "PC"),
    ("GES", "GES"),
    ("NOTEARS", "NOTEARS"),
    ("CaMML", "CaMML"),
    ("Marg", "Marg"),
    ("ECA", "ECA"),
    ("Claim-Fuse (PC)", "CEG-Fuse (PC)"),
    ("Claim-Fuse (PC+GES)", "CEG-Fuse (PC+GES)"),
    ("Claim-Fuse (PC+GES+NOTEARS)", "CEG-Fuse (PC+GES+NOTEARS)"),
    ("Claim-Fuse (PC grid)", "CEG-Fuse (PC grid)"),
    ("Claim-Fuse-SMC-Active (5 queries)", "active"),
    ("Claim-Fuse-SMC-Active (20 queries)", "active"),
)
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
PASSIVE_METHOD = "Claim-Fuse-SMC-Passive"


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs: {actual!r} != {expected!r}")


def _graph(record: Mapping[str, Any], node_count: int) -> Graph:
    return graph_from_edges(
        node_count,
        record.get("directed", ()),
        record.get("bidirected", ()),
        record.get("undirected", ()),
    )


def _partial(record: Mapping[str, Any]) -> PartialGraph:
    return PartialGraph.from_matrix(
        np.asarray(record["endpoint_matrix"], dtype=int), str(record["kind"])
    )


def _artifact_distribution(
    artifact: Mapping[str, Any], node_count: int
) -> tuple[tuple[Graph, ...], np.ndarray]:
    _require_equal(bool(artifact.get("feasible")), True, "method availability")
    graphs = tuple(_graph(record, node_count) for record in artifact["graphs"])
    probabilities = np.asarray(artifact["probabilities"], dtype=float)
    if not graphs or len(set(graphs)) != len(graphs):
        raise ValueError("method support is empty or contains duplicate DAGs")
    if probabilities.shape != (len(graphs),) or np.any(probabilities < 0.0):
        raise ValueError("method probabilities are invalid")
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("method probabilities do not sum to one")
    return graphs, probabilities / probabilities.sum()


def _camml_distribution(
    records: Sequence[Mapping[str, Any]], node_count: int
) -> tuple[tuple[Graph, ...], np.ndarray]:
    graphs: list[Graph] = []
    probabilities: list[float] = []
    for record in records:
        if set(record) != {"directed_rows", "probability"}:
            raise ValueError("CaMML support record has unexpected fields")
        graph = Graph(tuple(int(value) for value in record["directed_rows"]))
        if graph.node_count != node_count or graph.regime != "dag":
            raise ValueError("CaMML support contains an invalid DAG")
        graphs.append(graph)
        probabilities.append(float(record["probability"]))
    vector = np.asarray(probabilities, dtype=float)
    if (
        not graphs
        or len(set(graphs)) != len(graphs)
        or np.any(~np.isfinite(vector))
        or np.any(vector <= 0.0)
        or not math.isclose(float(vector.sum()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("CaMML support or probabilities are invalid")
    return tuple(graphs), vector / vector.sum()


def _active_distribution(
    records: Sequence[Mapping[str, Any]], node_count: int
) -> tuple[tuple[Graph, ...], np.ndarray]:
    graphs = tuple(_graph(record["graph"], node_count) for record in records)
    probabilities = np.asarray(
        [float(record["probability"]) for record in records], dtype=float
    )
    if (
        not graphs
        or len(set(graphs)) != len(graphs)
        or any(graph.regime != "dag" or not is_acyclic(graph) for graph in graphs)
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("SMC distribution is invalid")
    return graphs, probabilities / probabilities.sum()


def _metric_values(values: Mapping[str, float]) -> dict[str, float]:
    result = {metric: float(values[metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite recovery metric")
    return result


def _validate_passive_metadata(payload: Mapping[str, Any], reference_run: str) -> None:
    for key, expected in {
        "dataset": "synthetic",
        "profile": "production",
        "reference_run_id": reference_run,
        "active_query": False,
        "evidence_profile": ARCHIVED_EVIDENCE_PROFILE,
        "claim_count": 5,
        "config": json_value(asdict(PRODUCTION_CONFIG)),
    }.items():
        _require_equal(payload.get(key), expected, f"passive {key}")


def _validate_passive_partial(
    passive: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_path: Path,
    reference_run: str,
    task_index: int,
) -> None:
    _validate_passive_metadata(passive, reference_run)
    _require_equal(passive.get("task_index"), task_index, "passive task index")
    _require_equal(passive.get("replicate"), task_index, "passive graph index")
    _require_equal(passive.get("reference_sha256"), sha256(reference_path), "passive reference hash")
    _require_equal(passive.get("warm_start_source"), "pc+ges-completion-support", "passive warm starts")
    expected_claims = _confidence_records(reference["evidence"]["claims"], PASSIVE_CONFIDENCE)
    _require_equal(len(expected_claims), 5, "reference claim count")
    _require_equal(passive.get("claims"), json_value(expected_claims), "passive claims")
    _require_equal(passive.get("claim_fingerprint"), _claim_fingerprint(expected_claims), "passive claim fingerprint")
    for claim in expected_claims:
        for key, expected in {
            "reported_confidence": PASSIVE_CONFIDENCE,
            "effective_confidence": PASSIVE_CONFIDENCE,
            "treatment": "soft_evidence",
            "correct": True,
            "context_compatible": True,
        }.items():
            _require_equal(claim.get(key), expected, f"passive claim {key}")
    _require_equal(
        sorted(passive.get("artifacts", {})),
        sorted(str(size) for size in SAMPLE_SIZES),
        "passive sample sizes",
    )
    for sample in passive["artifacts"].values():
        _require_equal(sample.get("score"), {
            "name": "gaussian-bic", "graph_prior": "uniform",
            "max_parents": PRODUCTION_CONFIG.max_parents,
        }, "passive score")


def _distribution_row(
    graph_index: int,
    sample_size: int,
    method: str,
    graphs: Sequence[Graph],
    probabilities: np.ndarray,
    truth: Graph,
) -> dict[str, Any]:
    return {
        "graph_index": graph_index,
        "sample_size": sample_size,
        "method": method,
        **_metric_values(distribution_metrics(graphs, probabilities, truth, "dag")),
    }


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0,
    )


def _reference_path(results_root: Path, reference_run: str, task_index: int) -> Path:
    return results_root / reference_run / "partials" / f"task_{task_index:03d}.json"


def _validate_manifests(
    results_root: Path,
    reference_run: str,
    camml_run: str,
    active_run: str | None,
    passive_run: str | None = None,
) -> None:
    expected = list(TABLE41_COMMON_TASK_INDICES)
    reference = read_json(results_root / reference_run / "run_manifest.json")
    camml = read_json(results_root / camml_run / "run_manifest.json")
    _require_equal(reference.get("common_complete_indices"), expected, "reference common set")
    _require_equal(camml.get("common_complete_indices"), expected, "CaMML common set")
    _require_equal(camml.get("reference_run_id"), reference_run, "CaMML reference run")
    if passive_run is not None:
        passive = read_json(results_root / passive_run / "claim_fuse_smc" / "synthetic" / "run_manifest.json")
        _validate_passive_metadata(passive, reference_run)
        _require_equal(passive.get("task_indices"), expected, "passive task set")
        _require_equal(passive.get("run_id"), passive_run, "passive run")
    if active_run is None:
        return
    active = read_json(
        results_root
        / active_run
        / "claim_fuse_smc_eig"
        / "synthetic"
        / "run_manifest.json"
    )
    _require_equal(active.get("task_indices"), expected, "active task set")
    _require_equal(active.get("graph_indices"), expected, "active graph set")
    _require_equal(active.get("source_synthetic_run"), reference_run, "active reference run")
    _require_equal(active.get("random_seed_count"), 0, "active random-query count")
    _require_equal(active.get("confidence"), ACTIVE_CONFIDENCE, "active confidence")
    _require_equal(
        active.get("initialization_source"),
        ACTIVE_INITIALIZATION_SOURCE,
        "active initialization source",
    )
    _require_equal(
        active.get("particle_count"), ACTIVE_PARTICLE_COUNT, "active particle count"
    )


def aggregate(
    output_run: str,
    results_root: Path,
    *,
    reference_run: str = REFERENCE_RUN_ID,
    camml_run: str = CAMML_RUN_ID,
    active_run: str | None = None,
    passive_run: str | None = None,
) -> Path:
    _validate_manifests(results_root, reference_run, camml_run, active_run, passive_run)
    output_root = results_root / output_run / STUDY
    rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []

    for task_index in TABLE41_COMMON_TASK_INDICES:
        reference_path = _reference_path(results_root, reference_run, task_index)
        camml_path = (
            results_root / camml_run / "partials" / f"task_{task_index:03d}.json"
        )
        active_path = (
            results_root
            / active_run
            / "claim_fuse_smc_eig"
            / "synthetic"
            / "partials"
            / f"task_{task_index:03d}.json"
        ) if active_run is not None else None
        passive_path = (
            results_root / passive_run / "claim_fuse_smc" / "synthetic"
            / "partials" / f"task_{task_index:03d}.json"
        ) if passive_run is not None else None
        reference = read_json(reference_path)
        camml = read_json(camml_path)
        active = read_json(active_path) if active_path is not None else None
        passive = read_json(passive_path) if passive_path is not None else None
        graph_index = int(reference["graph_index"])
        _require_equal(graph_index, task_index, "reference graph index")
        _require_equal(camml.get("task_index"), task_index, "CaMML task index")
        _require_equal(camml.get("graph_index"), graph_index, "CaMML graph index")
        _require_equal(camml.get("reference_run_id"), reference_run, "CaMML reference")
        _require_equal(
            camml.get("reference_partial_sha256"),
            sha256(reference_path),
            "CaMML reference hash",
        )
        if active is not None:
            _require_equal(active.get("task_index"), task_index, "active task index")
            _require_equal(active.get("graph_index"), graph_index, "active graph index")
            _require_equal(active.get("confidence"), ACTIVE_CONFIDENCE, "active confidence")
            _require_equal(
                active.get("initialization_source"),
                ACTIVE_INITIALIZATION_SOURCE,
                "active initialization source",
            )
            _require_equal(
                active.get("particle_count"), ACTIVE_PARTICLE_COUNT, "active particle count"
            )
            _require_equal(
                active.get("source_synthetic_run_id"), reference_run, "active reference"
            )
            _require_equal(
                active.get("source_data_sha256"),
                sha256(reference_path),
                "active reference hash",
            )
        if passive is not None:
            _require_equal(passive.get("run_id"), passive_run, "passive run")
            _validate_passive_partial(passive, reference, reference_path, reference_run, task_index)
        node_count = int(reference["protocol"]["node_count"])
        truth = _graph(reference["target"], node_count)
        _require_equal(_graph(camml["target"], node_count), truth, "CaMML target")

        for sample_size in SAMPLE_SIZES:
            sample = reference["samples"][str(sample_size)]
            for method in ("PC", "GES"):
                output = sample["methods"][method]["diagnostics"]["output"]
                metrics = point_prediction_metrics(_partial(output), truth)
                rows.append(
                    {
                        "graph_index": graph_index,
                        "sample_size": sample_size,
                        "method": method,
                        **_metric_values(metrics),
                    }
                )
            for table_method, artifact_method in METHOD_SOURCES:
                if artifact_method in ("PC", "GES", "CaMML", "active"):
                    continue
                graphs, probabilities = _artifact_distribution(
                    sample["methods"][artifact_method], node_count
                )
                rows.append(
                    _distribution_row(
                        graph_index,
                        sample_size,
                        table_method,
                        graphs,
                        probabilities,
                        truth,
                    )
                )
            camml_result = camml["fits"][str(sample_size)]["result"]
            if camml_result is None:
                raise ValueError("CaMML is unavailable on the locked common set")
            graphs, probabilities = _camml_distribution(
                camml_result["support"], node_count
            )
            rows.append(
                _distribution_row(
                    graph_index,
                    sample_size,
                    "CaMML",
                    graphs,
                    probabilities,
                    truth,
                )
            )
            if passive is not None:
                graphs, probabilities = _active_distribution(
                    passive["artifacts"][str(sample_size)]["fused_distribution"], node_count
                )
                rows.append(_distribution_row(
                    graph_index, sample_size, PASSIVE_METHOD, graphs, probabilities, truth
                ))
            for active_budget in ACTIVE_BUDGETS if active is not None else ():
                budget = active["artifacts"][str(sample_size)]["eig"]["budgets"][
                    str(active_budget)
                ]
                graphs, probabilities = _active_distribution(
                    budget["distribution"], node_count
                )
                rows.append(
                    _distribution_row(
                        graph_index,
                        sample_size,
                        f"Claim-Fuse-SMC-Active ({active_budget} queries)",
                        graphs,
                        probabilities,
                        truth,
                    )
                )
        references.append(
            {
                "task_index": task_index,
                "graph_index": graph_index,
                "reference_path": str(reference_path),
                "reference_sha256": sha256(reference_path),
                "camml_path": str(camml_path),
                "camml_sha256": sha256(camml_path),
                "active_path": str(active_path) if active_path else None,
                "active_sha256": sha256(active_path) if active_path else None,
                "passive_path": str(passive_path) if passive_path else None,
                "passive_sha256": sha256(passive_path) if passive_path else None,
            }
        )

    expected_methods = tuple(
        method for method, source in METHOD_SOURCES if source != "active"
    ) + ((PASSIVE_METHOD,) if passive_run else ()) + tuple(
        method for method, source in METHOD_SOURCES if source == "active" and active_run
    )
    summary: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        for method in expected_methods:
            group = [
                row
                for row in rows
                if int(row["sample_size"]) == sample_size and row["method"] == method
            ]
            if len(group) != len(TABLE41_COMMON_TASK_INDICES):
                raise ValueError(
                    f"Table 4.1 has {len(group)} rows for n={sample_size}/{method}"
                )
            record: dict[str, Any] = {
                "sample_size": sample_size,
                "method": method,
                "replicate_count": len(group),
            }
            for metric in METRICS:
                mean, standard_error = _mean_se(
                    [float(row[metric]) for row in group]
                )
                record[f"{metric}_mean"] = mean
                record[f"{metric}_se"] = standard_error
            summary.append(record)
            availability.append(
                {
                    "sample_size": sample_size,
                    "method": method,
                    "available_count": len(group),
                    "selected_count": len(TABLE41_COMMON_TASK_INDICES),
                }
            )

    write_csv(output_root / "raw_results.csv", rows)
    write_csv(output_root / "summary.csv", summary)
    for filename in ("raw_results.csv", "summary.csv"):
        write_json(output_root / f"{filename}.metrics.json", metric_metadata(node_count))
    write_csv(output_root / "availability.csv", availability)
    write_json(output_root / "references.json", references)
    write_json(
        output_root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "reference_run_id": reference_run,
            "camml_run_id": camml_run,
            "active_run_id": active_run,
            "passive_run_id": passive_run,
            "metric_scale": "unnormalized",
            "task_indices": TABLE41_COMMON_TASK_INDICES,
            "graph_indices": TABLE41_COMMON_TASK_INDICES,
            "sample_sizes": SAMPLE_SIZES,
            "methods": expected_methods,
            "active_budgets": ACTIVE_BUDGETS if active_run else (),
            "metric_policy": {
                "PC": "raw-partial-point",
                "GES": "raw-partial-point",
                "other_methods": "saved-DAG-distribution",
            },
            "artifact_sha256": {
                name: sha256(output_root / name)
                for name in (
                    "raw_results.csv",
                    "summary.csv",
                    "availability.csv",
                    "references.json",
                )
            },
        },
    )
    print(output_root, flush=True)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference-run-id", default=REFERENCE_RUN_ID)
    parser.add_argument("--camml-run-id", default=CAMML_RUN_ID)
    parser.add_argument("--active-run-id")
    parser.add_argument("--passive-run-id")
    parser.add_argument("--results-root", type=Path, default=WORKSPACE_ROOT / "results")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    aggregate(
        output_run,
        arguments.results_root,
        reference_run=run_id(arguments.reference_run_id),
        camml_run=run_id(arguments.camml_run_id),
        active_run=run_id(arguments.active_run_id) if arguments.active_run_id else None,
        passive_run=run_id(arguments.passive_run_id) if arguments.passive_run_id else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
