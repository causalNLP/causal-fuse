"""Refresh Table 4.2 with raw PC/GES point-output metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cd_llm.core.finite_support import (
    Graph,
    PartialGraph,
    _causal_learn_partial,
    distribution_metrics,
    graph_edges,
    graph_from_edges,
    point_prediction_metrics,
)
from cd_llm.experiments.common.artifacts import (
    read_json,
    run_id,
    sha256,
    write_csv,
    write_json,
)
from cd_llm.experiments.neuropathic_benchmark import (
    DATA_SEEDS,
    SAMPLE_SIZES,
    build_panel_spec,
)
from cd_llm.experiments.neuropathic_runtime import run_ges
from cd_llm.paths import CODE_ROOT, WORKSPACE_ROOT


STUDY = "neuropathic_panel_point_benchmark"
STUDY_VERSION = 1
NODE_COUNT = 35
PC_ALPHA = 0.01
PC_INDEPENDENCE_TEST = "gsq"
PC_STABLE = True
PC_UC_RULE = 0
PC_UC_PRIORITY = 2
PC_MVPC = False
METHODS = (
    "PC",
    "GES",
    "Claim-Fuse (PC)",
    "Claim-Fuse (PC+GES)",
    "Claim-Fuse (PC grid)",
    "Claim-Fuse-SMC (20 claims)",
    "Claim-Fuse-SMC-Active (20 queries)",
)
FS_RUN_ID = "20260821T081942Z"
PASSIVE_RUN_ID = "20260822T042553Z"
ACTIVE_RUN_ID = "20260822T145551Z"
FS_STUDY = "neuropathic_benchmark/panel_a"
PASSIVE_STUDY = "claim_fuse_smc/neuropathic/passive-group-i-20"
ACTIVE_STUDY = "neuropathic_smc_active_structure_prior"
FS_SOURCE_FINGERPRINT = "0bc5b00a9f13e75d3a1b60c25f877f22c18e7cd43526bf16238119faf1922fc5"
FS_PROTOCOL_FINGERPRINT = "5081d2041401362525bd4a4196dc358493675ef2468f8ed161b714edb2e6dbbb"
FS_PANEL_HASH = "f273b36387efd46415d7c7cee5ec3514035b6b24b060dfc09308c139db908410"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"
DEFAULT_MODEL_PATH = CODE_ROOT / "resources" / "neuropathic" / "bnm.pickle"


def slice_for_task(task_index: int) -> tuple[int, int]:
    if not 0 <= task_index < len(DATA_SEEDS) * len(SAMPLE_SIZES):
        raise ValueError("task index must lie in [0, 39]")
    seed, size_index = divmod(task_index, len(SAMPLE_SIZES))
    return DATA_SEEDS[seed], SAMPLE_SIZES[size_index]


def _root(results_root: Path, output_run: str) -> Path:
    return results_root / output_run / STUDY


def _partial_path(
    results_root: Path, output_run: str, seed: int, sample_size: int
) -> Path:
    return _root(results_root, output_run) / "partials" / f"seed_{seed:02d}_n{sample_size}.json"


def _fs_path(results_root: Path, seed: int) -> Path:
    return results_root / FS_RUN_ID / FS_STUDY / "partials" / f"seed_{seed:02d}.json"


def _passive_path(results_root: Path, seed: int) -> Path:
    return results_root / PASSIVE_RUN_ID / PASSIVE_STUDY / "partials" / f"task_{seed:03d}.json"


def _active_path(results_root: Path, seed: int, sample_size: int) -> Path:
    return (
        results_root
        / ACTIVE_RUN_ID
        / ACTIVE_STUDY
        / "partials"
        / f"seed_{seed:02d}_n{sample_size}.json"
    )


def _protocol() -> dict[str, Any]:
    return {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "sample_sizes": SAMPLE_SIZES,
        "data_seeds": DATA_SEEDS,
        "node_count": NODE_COUNT,
        "pc": {
            "implementation": "causal-learn",
            "implementation_version": "0.1.4.5",
            "alpha": PC_ALPHA,
            "conditional_independence_test": PC_INDEPENDENCE_TEST,
            "stable": PC_STABLE,
            "uc_rule": PC_UC_RULE,
            "uc_priority": PC_UC_PRIORITY,
            "mvpc": PC_MVPC,
            "background_knowledge": None,
            "standardized": False,
            "constant_columns_retained": True,
        },
        "ges": {
            "implementation": "causal-learn",
            "implementation_version": "0.1.4.5",
            "score": "local_score_BDeu",
            "score_backend": "reference",
            "equivalent_sample_size": 1,
            "structure_prior": 1,
            "max_parents": None,
            "standardized": False,
        },
        "metric": {
            "distance": "normalized_endpoint_hamming",
            "normalization": "2*choose(p,2)",
            "point_energy_score_equals_expected_distance": True,
            "adjacency_f1": "raw_skeleton",
            "orientation_f1": "explicit_tail_arrow_edges",
            "truth_representation": "original_dag",
        },
        "methods": METHODS,
        "immutable_distribution_runs": {
            "finite_support": FS_RUN_ID,
            "passive_smc": PASSIVE_RUN_ID,
            "active_smc": ACTIVE_RUN_ID,
        },
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint() -> str:
    paths = (
        Path(__file__),
        CODE_ROOT / "src" / "cd_llm" / "core" / "finite_support.py",
        CODE_ROOT / "src" / "cd_llm" / "experiments" / "neuropathic_runtime.py",
        CODE_ROOT / "src" / "cd_llm" / "experiments" / "neuropathic_benchmark.py",
        CODE_ROOT / "src" / "cd_llm" / "experiments" / "common" / "artifacts.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().relative_to(CODE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("causal-learn", "networkx", "numpy", "pandas", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _partial_record(graph: PartialGraph) -> dict[str, Any]:
    return {"kind": graph.kind, "node_count": graph.node_count, "marks": graph.marks}


def _partial(record: Mapping[str, Any]) -> PartialGraph:
    return PartialGraph(
        int(record["node_count"]),
        tuple(int(mark) for mark in record["marks"]),
        str(record["kind"]),  # type: ignore[arg-type]
    )


def _graph(record: Mapping[str, Any]) -> Graph:
    return graph_from_edges(
        NODE_COUNT,
        record.get("directed", ()),
        record.get("bidirected", ()),
        record.get("undirected", ()),
    )


def run_pc(
    data: np.ndarray,
    node_names: Sequence[str],
    *,
    runner: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    causal_learn_version = version("causal-learn")
    if causal_learn_version != "0.1.4.5":
        raise RuntimeError("the Table 4.2 protocol requires causal-learn 0.1.4.5")
    if runner is None:
        from causallearn.search.ConstraintBased.PC import pc as runner
    result = runner(
        data,
        alpha=PC_ALPHA,
        indep_test=PC_INDEPENDENCE_TEST,
        stable=PC_STABLE,
        uc_rule=PC_UC_RULE,
        uc_priority=PC_UC_PRIORITY,
        mvpc=PC_MVPC,
        correction_name="MV_Crtn_Fisher_Z",
        background_knowledge=None,
        verbose=False,
        show_progress=False,
        node_names=list(node_names),
    )
    return (result.G if hasattr(result, "G") else result), {
        "algorithm": "pc",
        "implementation": "causal-learn",
        "implementation_version": causal_learn_version,
        "conditional_independence_test": PC_INDEPENDENCE_TEST,
        "alpha": PC_ALPHA,
        "stable": PC_STABLE,
        "uc_rule": PC_UC_RULE,
        "uc_priority": PC_UC_PRIORITY,
        "mvpc": PC_MVPC,
        "background_knowledge": None,
        "standardized": False,
        "constant_columns_retained": True,
    }


def compute_slice(
    seed: int,
    sample_size: int,
    output_run: str,
    results_root: Path,
    model_path: Path,
    *,
    pc_runner: Callable[..., Any] | None = None,
    ges_runner: Callable[..., Any] | None = None,
) -> Path:
    started = time.perf_counter()
    reference_path = _fs_path(results_root, seed)
    reference = read_json(reference_path)
    if reference.get("run_id") != FS_RUN_ID or reference.get("data_seed") != seed:
        raise ValueError("finite-support reference identity differs")
    sample = reference["sample_artifacts"][str(sample_size)]
    data_path = Path(sample["data_path"])
    if sha256(data_path) != sample["data_sha256"]:
        raise ValueError("Panel-A data hash differs")
    frame = pd.read_csv(data_path, index_col=0)
    data = frame.to_numpy(dtype=np.int64, copy=True)
    if data.shape != (sample_size, NODE_COUNT) or np.any((data != 0) & (data != 1)):
        raise ValueError("Panel-A point benchmark requires raw binary integer data")
    panel = build_panel_spec(model_path)
    expected_columns = tuple(str(index) for index in panel.original_indices)
    if tuple(frame.columns) != expected_columns:
        raise ValueError("Panel-A original-index column order differs")
    if tuple(reference["panel"]["node_names"]) != panel.node_names:
        raise ValueError("Panel-A semantic node order differs")
    truth = panel.target

    fit_started = time.perf_counter()
    raw_pc, pc_settings = run_pc(data, panel.node_names, runner=pc_runner)
    pc_fit = {"runtime_seconds": time.perf_counter() - fit_started, **pc_settings}
    pc_graph = _causal_learn_partial(raw_pc, "cpdag")

    fit_started = time.perf_counter()
    raw_ges, ges_settings = run_ges(
        data, panel.node_names, runner=ges_runner, score_backend="reference"
    )
    ges_fit = {"runtime_seconds": time.perf_counter() - fit_started, **ges_settings}
    ges_graph = _causal_learn_partial(raw_ges, "cpdag")

    methods: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method, graph, fit in (
        ("PC", pc_graph, pc_fit),
        ("GES", ges_graph, ges_fit),
    ):
        metrics = point_prediction_metrics(graph, truth)
        methods[method] = {
            "partial_graph": _partial_record(graph),
            "fit": fit,
            "metrics": metrics,
        }
        rows.append(
            {
                "seed": seed,
                "sample_size": sample_size,
                "method": method,
                "feasible": 1,
                **metrics,
                "method_runtime_seconds": float(fit["runtime_seconds"]),
            }
        )

    output = _partial_path(results_root, output_run, seed, sample_size)
    write_json(
        output,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "compute",
            "seed": seed,
            "sample_size": sample_size,
            "protocol": _protocol(),
            "protocol_fingerprint": _fingerprint(_protocol()),
            "source_fingerprint": _source_fingerprint(),
            "reference": {
                "path": str(reference_path.resolve()),
                "sha256": sha256(reference_path),
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256(data_path),
            },
            "target": graph_edges(truth),
            "methods": methods,
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "package_versions": _package_versions(),
            "command": list(sys.argv),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"completed raw PC/GES seed={seed} n={sample_size}: {output}", flush=True)
    return output


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs")


def _validate_metrics(
    recorded: Mapping[str, Any], recomputed: Mapping[str, float], label: str
) -> None:
    for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
        if not math.isclose(
            float(recorded[key]), float(recomputed[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} metric {key!r} differs")


def _distribution(
    graphs: Sequence[Mapping[str, Any]], probabilities: Sequence[float]
) -> tuple[tuple[Graph, ...], np.ndarray]:
    support = tuple(_graph(graph) for graph in graphs)
    weights = np.asarray(probabilities, dtype=float)
    if not support or len(support) != len(weights) or len(support) != len(set(support)):
        raise ValueError("invalid graph support")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(
        weights.sum(), 1.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("invalid graph probabilities")
    return support, weights / weights.sum()


def _record_distribution(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Graph, ...], np.ndarray]:
    return _distribution(
        [record["graph"] for record in records],
        [float(record["probability"]) for record in records],
    )


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def aggregate(output_run: str, results_root: Path, model_path: Path) -> Path:
    output_run = run_id(output_run)
    root = _root(results_root, output_run)
    panel = build_panel_spec(model_path)
    raw_rows: list[dict[str, Any]] = []
    references: dict[str, list[dict[str, Any]]] = {
        "point": [], "finite_support": [], "passive_smc": [], "active_smc": []
    }
    current_source = _source_fingerprint()

    for seed in DATA_SEEDS:
        fs_path = _fs_path(results_root, seed)
        passive_path = _passive_path(results_root, seed)
        fs = read_json(fs_path)
        passive = read_json(passive_path)
        for key, value in {
            "study": "neuropathic_cervical_panel_a",
            "study_version": "1.0",
            "run_id": FS_RUN_ID,
            "data_seed": seed,
            "source_fingerprint": FS_SOURCE_FINGERPRINT,
            "protocol_fingerprint": FS_PROTOCOL_FINGERPRINT,
            "panel_hash": FS_PANEL_HASH,
        }.items():
            _require_equal(fs.get(key), value, f"finite-support {key}")
        _require_equal(fs.get("target"), graph_edges(panel.target), "finite-support target")
        for key, value in {
            "study": "claim_fuse_smc_sidecar",
            "study_version": 1,
            "run_id": PASSIVE_RUN_ID,
            "replicate": seed,
            "reference_path": str(fs_path.resolve()),
            "reference_sha256": sha256(fs_path),
        }.items():
            _require_equal(passive.get(key), value, f"passive-SMC {key}")
        references["finite_support"].append(
            {"seed": seed, "path": str(fs_path.resolve()), "sha256": sha256(fs_path)}
        )
        references["passive_smc"].append(
            {"seed": seed, "path": str(passive_path.resolve()), "sha256": sha256(passive_path)}
        )

        for sample_size in SAMPLE_SIZES:
            point_path = _partial_path(results_root, output_run, seed, sample_size)
            point = read_json(point_path)
            for key, value in {
                "study": STUDY,
                "study_version": STUDY_VERSION,
                "run_id": output_run,
                "stage": "compute",
                "seed": seed,
                "sample_size": sample_size,
                "protocol_fingerprint": _fingerprint(_protocol()),
                "source_fingerprint": current_source,
            }.items():
                _require_equal(point.get(key), value, f"point {key}")
            _require_equal(point.get("target"), fs.get("target"), "point target")
            sample = fs["sample_artifacts"][str(sample_size)]
            _require_equal(
                point["reference"]["data_sha256"], sample["data_sha256"], "point data"
            )
            truth = _graph(point["target"])
            for method in ("PC", "GES"):
                artifact = point["methods"][method]
                metrics = point_prediction_metrics(_partial(artifact["partial_graph"]), truth)
                _validate_metrics(artifact["metrics"], metrics, method)
                raw_rows.append(
                    {
                        "seed": seed,
                        "sample_size": sample_size,
                        "method": method,
                        "feasible": 1,
                        **metrics,
                        "method_runtime_seconds": float(artifact["fit"]["runtime_seconds"]),
                    }
                )

            fs_labels = {
                "Claim-Fuse (PC)": "Claim-Fuse (PC)",
                "Claim-Fuse (PC+GES)": "Claim-Fuse (PC+GES)",
                "Claim-Fuse (PC grid)": "Claim-Fuse (PC grid)",
            }
            recorded_rows = {
                (int(row["sample_size"]), str(row["method"])): row for row in fs["rows"]
            }
            for method, artifact_label in fs_labels.items():
                artifact = sample["methods"][artifact_label]
                _require_equal(artifact.get("available"), True, f"{method} availability")
                graphs, probabilities = _distribution(
                    artifact["graphs"], artifact["probabilities"]
                )
                metrics = distribution_metrics(graphs, probabilities, truth, "dag")
                _validate_metrics(recorded_rows[(sample_size, artifact_label)], metrics, method)
                raw_rows.append(
                    {"seed": seed, "sample_size": sample_size, "method": method, "feasible": 1, **metrics, "method_runtime_seconds": math.nan}
                )

            passive_artifact = passive["artifacts"][str(sample_size)]
            passive_graphs, passive_probabilities = _record_distribution(
                passive_artifact["fused_distribution"]
            )
            passive_metrics = distribution_metrics(
                passive_graphs, passive_probabilities, truth, "dag"
            )
            passive_row = next(
                row for row in passive["rows"] if int(row["sample_size"]) == sample_size
            )
            _validate_metrics(passive_row, passive_metrics, "passive SMC")
            raw_rows.append(
                {"seed": seed, "sample_size": sample_size, "method": "Claim-Fuse-SMC (20 claims)", "feasible": 1, **passive_metrics, "method_runtime_seconds": float(passive_row["method_runtime_seconds"])}
            )

            active_path = _active_path(results_root, seed, sample_size)
            active = read_json(active_path)
            for key, value in {
                "study": ACTIVE_STUDY,
                "study_version": 1,
                "run_id": ACTIVE_RUN_ID,
                "replicate": seed,
                "sample_size": sample_size,
                "maximum_budget": 20,
                "confidence": 1.0,
                "structure_prior": 1.0,
                "reference_path": str(fs_path.resolve()),
                "reference_sha256": sha256(fs_path),
            }.items():
                _require_equal(active.get(key), value, f"active-SMC {key}")
            _require_equal(
                active["data_file"]["sha256"], sample["data_sha256"], "active-SMC data"
            )
            active_graphs, active_probabilities = _record_distribution(
                active["artifact"]["active"]["distribution"]
            )
            active_metrics = distribution_metrics(
                active_graphs, active_probabilities, truth, "dag"
            )
            _validate_metrics(active["row"], active_metrics, "active SMC")
            raw_rows.append(
                {"seed": seed, "sample_size": sample_size, "method": "Claim-Fuse-SMC-Active (20 queries)", "feasible": 1, **active_metrics, "method_runtime_seconds": float(active["row"]["method_runtime_seconds"])}
            )
            references["point"].append(
                {"seed": seed, "sample_size": sample_size, "path": str(point_path.resolve()), "sha256": sha256(point_path)}
            )
            references["active_smc"].append(
                {"seed": seed, "sample_size": sample_size, "path": str(active_path.resolve()), "sha256": sha256(active_path)}
            )

    expected_count = len(DATA_SEEDS) * len(SAMPLE_SIZES) * len(METHODS)
    if len(raw_rows) != expected_count:
        raise RuntimeError("aggregate does not cover the exact method/seed/size grid")
    summary: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        for method in METHODS:
            group = [
                row for row in raw_rows
                if row["sample_size"] == sample_size and row["method"] == method
            ]
            if len(group) != len(DATA_SEEDS):
                raise RuntimeError(f"{method}, n={sample_size} lacks all 20 seeds")
            availability.append(
                {"sample_size": sample_size, "method": method, "available_count": len(group), "total_count": len(DATA_SEEDS)}
            )
            record: dict[str, Any] = {
                "sample_size": sample_size, "method": method, "replicate_count": len(group)
            }
            for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
                mean, standard_error = _mean_se([float(row[key]) for row in group])
                record[f"{key}_mean"] = mean
                record[f"{key}_se"] = standard_error
            runtimes = [float(row["method_runtime_seconds"]) for row in group]
            finite_runtimes = [value for value in runtimes if math.isfinite(value)]
            if finite_runtimes:
                mean, standard_error = _mean_se(finite_runtimes)
                record["method_runtime_seconds_mean"] = mean
                record["method_runtime_seconds_se"] = standard_error
            else:
                record["method_runtime_seconds_mean"] = math.nan
                record["method_runtime_seconds_se"] = math.nan
            summary.append(record)

    write_csv(root / "raw_results.csv", raw_rows)
    write_csv(root / "availability.csv", availability)
    write_csv(root / "summary.csv", summary)
    write_json(
        root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "protocol": _protocol(),
            "protocol_fingerprint": _fingerprint(_protocol()),
            "source_fingerprint": current_source,
            "methods": METHODS,
            "common_complete_seeds": {str(size): DATA_SEEDS for size in SAMPLE_SIZES},
            "artifact_references": references,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"aggregated Table 4.2 metric refresh: {root}", flush=True)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compute", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def _selected_slice(arguments: argparse.Namespace) -> tuple[int, int]:
    direct = arguments.seed is not None or arguments.sample_size is not None
    if arguments.task_index is not None and direct:
        raise ValueError("use either --task-index or --seed/--sample-size")
    if arguments.task_index is not None:
        return slice_for_task(arguments.task_index)
    if arguments.seed not in DATA_SEEDS or arguments.sample_size not in SAMPLE_SIZES:
        raise ValueError("a valid task index or seed/sample-size pair is required")
    return int(arguments.seed), int(arguments.sample_size)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    if arguments.stage == "aggregate":
        if any(value is not None for value in (arguments.task_index, arguments.seed, arguments.sample_size)):
            raise ValueError("slice arguments are invalid for aggregation")
        aggregate(output_run, arguments.results_root, arguments.model_path)
        return 0
    seed, sample_size = _selected_slice(arguments)
    compute_slice(
        seed, sample_size, output_run, arguments.results_root, arguments.model_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
