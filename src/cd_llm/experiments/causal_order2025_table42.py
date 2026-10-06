"""Evaluate Causal Order+PC (Algorithm 1) on the frozen Table 4.2 PC outputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cd_llm.baselines.causal_order2025 import (
    DEFAULT_MODEL,
    CausalOrderInfeasibleError,
    OpenRouterConfig,
    TripletConfig,
    TripletExpertResult,
    elicit_triplet_order_openrouter,
    integrate_order_into_cpdag,
    query_pair_fallbacks_openrouter,
)
from cd_llm.baselines.causal_order2025.upstream_backend import UPSTREAM_REVISION
from cd_llm.core.finite_support import (
    Endpoint,
    Graph,
    PartialGraph,
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
from cd_llm.paths import CODE_ROOT, WORKSPACE_ROOT


STUDY = "causal_order2025_table42"
STUDY_VERSION = 1
REFERENCE_RUN_ID = "20260824T120300Z"
REFERENCE_STUDY = "neuropathic_panel_point_benchmark"
REFERENCE_PROTOCOL_FINGERPRINT = "fc9988e6aa59ab98dec8589a833ce96eacc0c60b961c93fd472e9f3f58754ecf"
CONTEXT = "for neuropathic pain diagnosis"
METHOD = "Causal Order+PC"
NODE_COUNT = 35
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"
DEFAULT_MODEL_PATH = CODE_ROOT / "resources" / "neuropathic" / "bnm.pickle"


def slice_for_task(task_index: int) -> tuple[int, int]:
    if not 0 <= task_index < len(DATA_SEEDS) * len(SAMPLE_SIZES):
        raise ValueError("task index must lie in [0, 39]")
    seed, size_index = divmod(task_index, len(SAMPLE_SIZES))
    return DATA_SEEDS[seed], SAMPLE_SIZES[size_index]


def _root(results_root: Path, output_run: str) -> Path:
    return results_root / output_run / STUDY


def _reference_path(results_root: Path, seed: int, sample_size: int) -> Path:
    return (
        results_root
        / REFERENCE_RUN_ID
        / REFERENCE_STUDY
        / "partials"
        / f"seed_{seed:02d}_n{sample_size}.json"
    )


def _partial_path(results_root: Path, output_run: str, seed: int, sample_size: int) -> Path:
    return _root(results_root, output_run) / "partials" / f"seed_{seed:02d}_n{sample_size}.json"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_fingerprint() -> str:
    paths = (
        Path(__file__),
        CODE_ROOT / "src" / "cd_llm" / "baselines" / "causal_order2025" / "openrouter_backend.py",
        CODE_ROOT / "src" / "cd_llm" / "baselines" / "causal_order2025" / "upstream_backend.py",
        CODE_ROOT / "src" / "cd_llm" / "baselines" / "causal_order2025" / "integration.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().relative_to(CODE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _protocol(config: OpenRouterConfig) -> dict[str, Any]:
    return {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "algorithm": "Appendix D Algorithm 1 (Causal Order+PC)",
        "node_count": NODE_COUNT,
        "sample_sizes": list(SAMPLE_SIZES),
        "data_seeds": list(DATA_SEEDS),
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_protocol_fingerprint": REFERENCE_PROTOCOL_FINGERPRINT,
        "pc_output": "frozen raw causal-learn CPDAG from Table 4.2",
        "expert": {
            "strategy": "all unordered node triplets",
            "triplet_count": math.comb(NODE_COUNT, 3),
            "shared_across_data_seeds_and_sample_sizes": True,
            "context": CONTEXT,
            "node_descriptions": None,
            "provider": "OpenRouter",
            "model": config.model,
            "seed": config.seed,
            "temperature_supplied": False,
            "triplet_max_completion_tokens": 300,
            "tie_and_fallback_max_completion_tokens": 400,
            "upstream_revision": UPSTREAM_REVISION,
        },
        "transport": {
            "base_url": config.base_url,
            "concurrency": config.concurrency,
            "max_attempts": config.max_attempts,
            "cost_cap_usd": config.cost_cap_usd,
            "successful_response_cache": "SQLite, atomic per response",
            "model_substitution": False,
        },
        "fallback": "one shared superior-expert query for each undirected PC pair absent from the expert order",
        "infeasibility": "no graph repair, no semantic retry, and no complete-case subset",
        "metric": {
            "distance": "normalized_endpoint_hamming",
            "energy_score": "point prediction; equals endpoint distance",
            "adjacency_f1": "raw skeleton",
            "orientation_f1": "explicit tail-arrow edges",
            "truth_representation": "original DAG",
        },
    }


def _partial(record: Mapping[str, Any]) -> PartialGraph:
    return PartialGraph(
        int(record["node_count"]),
        tuple(int(value) for value in record["marks"]),
        str(record["kind"]),  # type: ignore[arg-type]
    )


def _graph(record: Mapping[str, Any], node_count: int = NODE_COUNT) -> Graph:
    return graph_from_edges(
        node_count,
        directed=record.get("directed", ()),
        bidirected=record.get("bidirected", ()),
        undirected=record.get("undirected", ()),
    )


def _expert_record(expert: TripletExpertResult) -> dict[str, Any]:
    return {
        "node_names": list(expert.node_names),
        "raw_edges": [list(edge) for edge in expert.raw_edges],
        "graph": graph_edges(expert.graph),
        "edgewise_distributions": [
            {"pair": list(pair), "probabilities": list(probabilities)}
            for pair, probabilities in sorted(expert.edgewise_distributions.items())
        ],
        "subgroup_list": [list(group) for group in expert.subgroup_list],
        "subgroup_responses": list(expert.subgroup_responses),
        "expert_order": list(expert.expert_order),
        "isolated_nodes": list(expert.isolated_nodes),
        "cycle_removal_threshold": expert.cycle_removal_threshold,
        "upstream_revision": expert.upstream_revision,
        "config": {
            "subgraph_model": expert.config.subgraph_model,
            "expert_model": expert.config.expert_model,
            "delay": expert.config.delay,
        },
        "runtime_seconds": expert.runtime_seconds,
    }


def _expert(record: Mapping[str, Any]) -> TripletExpertResult:
    config = record["config"]
    return TripletExpertResult(
        node_names=tuple(record["node_names"]),
        raw_edges=tuple(tuple(edge) for edge in record["raw_edges"]),
        graph=_graph(record["graph"], len(record["node_names"])),
        edgewise_distributions={
            tuple(item["pair"]): tuple(item["probabilities"])
            for item in record["edgewise_distributions"]
        },
        subgroup_list=tuple(tuple(group) for group in record["subgroup_list"]),
        subgroup_responses=tuple(record["subgroup_responses"]),
        expert_order=tuple(record["expert_order"]),
        isolated_nodes=tuple(record["isolated_nodes"]),
        cycle_removal_threshold=record["cycle_removal_threshold"],
        upstream_revision=str(record["upstream_revision"]),
        config=TripletConfig(
            subgraph_model=str(config["subgraph_model"]),
            expert_model=str(config["expert_model"]),
            delay=float(config["delay"]),
        ),
        runtime_seconds=float(record["runtime_seconds"]),
    )


def _validated_reference(results_root: Path, seed: int, sample_size: int) -> tuple[Path, dict[str, Any]]:
    path = _reference_path(results_root, seed, sample_size)
    record = read_json(path)
    expected = {
        "study": REFERENCE_STUDY,
        "study_version": 1,
        "run_id": REFERENCE_RUN_ID,
        "stage": "compute",
        "seed": seed,
        "sample_size": sample_size,
        "protocol_fingerprint": REFERENCE_PROTOCOL_FINGERPRINT,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"reference {key} differs for seed={seed}, n={sample_size}")
    pc = _partial(record["methods"]["PC"]["partial_graph"])
    if pc.node_count != NODE_COUNT or pc.kind != "cpdag":
        raise ValueError("reference PC output is not the required 35-node CPDAG")
    return path, record


def required_fallback_pairs(
    expert: TripletExpertResult, references: Sequence[PartialGraph]
) -> tuple[tuple[int, int], ...]:
    rank = set(expert.expert_order)
    pairs: set[tuple[int, int]] = set()
    for cpdag in references:
        for first in range(cpdag.node_count):
            for second in range(first + 1, cpdag.node_count):
                if (
                    cpdag.mark(first, second) == Endpoint.TAIL
                    and cpdag.mark(second, first) == Endpoint.TAIL
                    and (first not in rank or second not in rank)
                ):
                    pairs.add((first, second))
    return tuple(sorted(pairs))


async def elicit(
    output_run: str,
    results_root: Path,
    model_path: Path,
    config: OpenRouterConfig,
) -> Path:
    output_run = run_id(output_run)
    root = _root(results_root, output_run)
    expert_path = root / "expert.json"
    if expert_path.exists():
        raise FileExistsError(f"elicitation artifact already exists: {expert_path}")
    panel = build_panel_spec(model_path)
    if len(panel.node_names) != NODE_COUNT:
        raise ValueError("Table 4.2 Panel A must contain exactly 35 nodes")
    protocol = _protocol(config)
    elicitation = await elicit_triplet_order_openrouter(
        panel.node_names,
        CONTEXT,
        root / "request_cache.sqlite3",
        config=config,
    )
    references: list[PartialGraph] = []
    reference_records: list[dict[str, Any]] = []
    for seed in DATA_SEEDS:
        for sample_size in SAMPLE_SIZES:
            path, record = _validated_reference(results_root, seed, sample_size)
            references.append(_partial(record["methods"]["PC"]["partial_graph"]))
            reference_records.append(
                {"seed": seed, "sample_size": sample_size, "path": str(path.resolve()), "sha256": sha256(path)}
            )
    fallback_pairs = required_fallback_pairs(elicitation.expert, references)
    fallbacks, fallback_audit = await query_pair_fallbacks_openrouter(
        panel.node_names,
        fallback_pairs,
        CONTEXT,
        root / "request_cache.sqlite3",
        config=config,
    )
    fallback_record = [
        {
            "pair": list(pair),
            "orientation": None if orientation is None else list(orientation),
        }
        for pair, orientation in sorted(fallbacks.items())
    ]
    # expert.json is the completion marker.  A retry may safely replace this
    # derived file after resuming the transactional response cache.
    write_json(root / "fallbacks.json", {"fallbacks": fallback_record})
    cache_path = root / "request_cache.sqlite3"
    write_json(
        expert_path,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "elicit",
            "protocol": protocol,
            "protocol_fingerprint": _fingerprint(protocol),
            "source_fingerprint": _source_fingerprint(),
            "expert": _expert_record(elicitation.expert),
            "elicitation_audit": dict(elicitation.audit),
            "fallback_audit": fallback_audit,
            "request_cache": {"path": str(cache_path.resolve()), "sha256": sha256(cache_path)},
            "reference_artifacts": reference_records,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"completed causal-order elicitation: {expert_path}", flush=True)
    return expert_path


def integrate_slice(
    seed: int,
    sample_size: int,
    output_run: str,
    results_root: Path,
) -> Path:
    started = time.perf_counter()
    output_run = run_id(output_run)
    root = _root(results_root, output_run)
    expert_path = root / "expert.json"
    expert_artifact = read_json(expert_path)
    if expert_artifact.get("run_id") != output_run or expert_artifact.get("stage") != "elicit":
        raise ValueError("expert artifact identity differs")
    expert = _expert(expert_artifact["expert"])
    fallbacks_artifact = read_json(root / "fallbacks.json")
    fallbacks = {
        tuple(item["pair"]): None if item["orientation"] is None else tuple(item["orientation"])
        for item in fallbacks_artifact["fallbacks"]
    }
    reference_path, reference = _validated_reference(results_root, seed, sample_size)
    cpdag = _partial(reference["methods"]["PC"]["partial_graph"])
    truth = _graph(reference["target"])

    used_pairs: list[tuple[int, int]] = []

    def fallback(first: int, second: int) -> tuple[int, int]:
        pair = min(first, second), max(first, second)
        used_pairs.append(pair)
        if pair not in fallbacks:
            raise CausalOrderInfeasibleError("required fallback pair was not pre-elicited")
        orientation = fallbacks[pair]
        if orientation is None:
            raise CausalOrderInfeasibleError(
                "fallback expert returned no relation for an existing PC adjacency"
            )
        return orientation

    integration_started = time.perf_counter()
    try:
        result = integrate_order_into_cpdag(cpdag, expert, fallback_orienter=fallback)
        metrics = point_prediction_metrics(result.graph, truth)
        feasible = True
        error = None
        graph = graph_edges(result.graph)
        final_order = list(result.final_order)
        fallback_orientations = [list(edge) for edge in result.fallback_orientations]
    except CausalOrderInfeasibleError as exc:
        feasible = False
        error = f"{type(exc).__name__}: {exc}"
        graph = None
        final_order = None
        fallback_orientations = None
        metrics = None
    integration_seconds = time.perf_counter() - integration_started
    output = _partial_path(results_root, output_run, seed, sample_size)
    write_json(
        output,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "integrate",
            "seed": seed,
            "sample_size": sample_size,
            "method": METHOD,
            "protocol_fingerprint": expert_artifact["protocol_fingerprint"],
            "source_fingerprint": expert_artifact["source_fingerprint"],
            "expert_path": str(expert_path.resolve()),
            "expert_sha256": sha256(expert_path),
            "reference_path": str(reference_path.resolve()),
            "reference_sha256": sha256(reference_path),
            "feasible": feasible,
            "error": error,
            "graph": graph,
            "final_order": final_order,
            "fallback_orientations": fallback_orientations,
            "fallback_pairs_used": [list(pair) for pair in sorted(set(used_pairs))],
            "metrics": metrics,
            "integration_runtime_seconds": integration_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "command": list(sys.argv),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"completed Algorithm 1 seed={seed} n={sample_size} feasible={feasible}: {output}", flush=True)
    return output


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def aggregate(output_run: str, results_root: Path) -> Path:
    output_run = run_id(output_run)
    root = _root(results_root, output_run)
    expert_path = root / "expert.json"
    expert = read_json(expert_path)
    raw_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for seed in DATA_SEEDS:
        for sample_size in SAMPLE_SIZES:
            path = _partial_path(results_root, output_run, seed, sample_size)
            artifact = read_json(path)
            expected = {
                "study": STUDY,
                "study_version": STUDY_VERSION,
                "run_id": output_run,
                "stage": "integrate",
                "seed": seed,
                "sample_size": sample_size,
                "protocol_fingerprint": expert["protocol_fingerprint"],
                "source_fingerprint": expert["source_fingerprint"],
                "expert_sha256": sha256(expert_path),
            }
            for key, value in expected.items():
                if artifact.get(key) != value:
                    raise ValueError(f"partial {key} differs for seed={seed}, n={sample_size}")
            row: dict[str, Any] = {
                "seed": seed,
                "sample_size": sample_size,
                "method": METHOD,
                "feasible": int(bool(artifact["feasible"])),
                "error": artifact["error"],
                "integration_runtime_seconds": artifact["integration_runtime_seconds"],
            }
            if artifact["metrics"] is not None:
                graph = _graph(artifact["graph"])
                _, reference = _validated_reference(results_root, seed, sample_size)
                recomputed = point_prediction_metrics(graph, _graph(reference["target"]))
                for key, value in recomputed.items():
                    if not math.isclose(
                        float(artifact["metrics"][key]), float(value), rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(f"recorded {key} differs")
                row.update(recomputed)
            else:
                row.update({key: None for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")})
            raw_rows.append(row)
            references.append({"seed": seed, "sample_size": sample_size, "path": str(path.resolve()), "sha256": sha256(path)})

    summary: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        group = [row for row in raw_rows if row["sample_size"] == sample_size]
        feasible = [row for row in group if row["feasible"]]
        availability.append(
            {
                "sample_size": sample_size,
                "method": METHOD,
                "available_count": len(feasible),
                "total_count": len(group),
                "all_feasible": int(len(feasible) == len(group)),
            }
        )
        record: dict[str, Any] = {
            "sample_size": sample_size,
            "method": METHOD,
            "replicate_count": len(group),
            "available_count": len(feasible),
            "all_feasible": int(len(feasible) == len(group)),
        }
        for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
            if len(feasible) == len(group):
                mean, se = _mean_se([float(row[key]) for row in group])
                record[f"{key}_mean"] = mean
                record[f"{key}_se"] = se
            else:
                record[f"{key}_mean"] = None
                record[f"{key}_se"] = None
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
            "protocol": expert["protocol"],
            "protocol_fingerprint": expert["protocol_fingerprint"],
            "source_fingerprint": expert["source_fingerprint"],
            "expert_path": str(expert_path.resolve()),
            "expert_sha256": sha256(expert_path),
            "request_cache": expert["request_cache"],
            "all_40_feasible": all(row["feasible"] for row in raw_rows),
            "artifact_references": references,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"aggregated Algorithm 1 Table 4.2 benchmark: {root}", flush=True)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("elicit", "integrate", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--cost-cap-usd", type=float, default=50.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    if arguments.stage == "elicit":
        if arguments.task_index is not None:
            raise ValueError("--task-index is invalid for elicitation")
        config = OpenRouterConfig(
            model=arguments.model,
            concurrency=arguments.concurrency,
            max_attempts=arguments.max_attempts,
            cost_cap_usd=arguments.cost_cap_usd,
        )
        asyncio.run(elicit(output_run, arguments.results_root, arguments.model_path, config))
    elif arguments.stage == "integrate":
        if arguments.task_index is None:
            raise ValueError("--task-index is required for integration")
        seed, sample_size = slice_for_task(arguments.task_index)
        integrate_slice(seed, sample_size, output_run, arguments.results_root)
    else:
        if arguments.task_index is not None:
            raise ValueError("--task-index is invalid for aggregation")
        aggregate(output_run, arguments.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
