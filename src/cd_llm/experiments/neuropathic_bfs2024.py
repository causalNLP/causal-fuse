"""Evaluate the Jiralerspong et al. (2024) BFS baseline on Table 4.5.

The locked benchmark is consumed directly from its zip archive.  The target
graph is loaded only after inference, when metrics are computed.  This keeps
the evaluation truth out of prompts and out of all graph construction logic.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import time
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from cd_llm.baselines.jiralerspong2024 import Complete, openai_completer, run_bfs
from cd_llm.core.finite_support import graph_from_edges, point_prediction_metrics
from cd_llm.data.generation.neuropathic import CANONICAL_VARIABLE_NAMES
from cd_llm.experiments.common.artifacts import run_id, write_csv, write_json
from cd_llm.paths import WORKSPACE_ROOT


BENCHMARK_ID = "table45-neuropathic-ancestor20-v1"
KIT_PREFIX = "20260826T120230Z/table45_benchmark_kit"
DEFAULT_ARCHIVE = Path("20260826T120230Z.zip")
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"
SAMPLE_SIZES = (200, 1000)
SEEDS = tuple(range(20))


DESCRIPTIONS = {
    "DLS C1-C2": "disc lesion or disorder at cervical level C1-C2",
    "DLS C2-C3": "disc lesion or disorder at cervical level C2-C3",
    "DLS C3-C4": "disc lesion or disorder at cervical level C3-C4",
    "DLS C4-C5": "disc lesion or disorder at cervical level C4-C5",
    "DLS C5-C6": "disc lesion or disorder at cervical level C5-C6",
    "DLS C6-C7": "disc lesion or disorder at cervical level C6-C7",
    "Kraniocervikal ledskada": "craniocervical joint injury",
    "L C2 Radikulopati": "left C2 radiculopathy",
    "R C2 Radikulopati": "right C2 radiculopathy",
    "L C3 Radikulopati": "left C3 radiculopathy",
    "R C3 Radikulopati": "right C3 radiculopathy",
    "L C4 Radikulopati": "left C4 radiculopathy",
    "R C4 Radikulopati": "right C4 radiculopathy",
    "L C5 Radikulopati": "left C5 radiculopathy",
    "R C5 Radikulopati": "right C5 radiculopathy",
    "L C6 Radikulopati": "left C6 radiculopathy",
    "R C6 Radikulopati": "right C6 radiculopathy",
    "L C7 Radikulopati": "left C7 radiculopathy",
    "R C7 Radikulopati": "right C7 radiculopathy",
    "L Nackbesvär": "left-sided neck symptoms or discomfort",
}


def _member(relative: str) -> str:
    return f"{KIT_PREFIX}/{relative}"


def _read_json(archive: zipfile.ZipFile, relative: str) -> dict:
    return json.loads(archive.read(_member(relative)).decode("utf-8"))


def load_manifest(archive: zipfile.ZipFile) -> dict:
    manifest = _read_json(archive, "manifest.json")
    if manifest["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("unexpected benchmark identifier")
    return manifest


def load_dataset(
    archive: zipfile.ZipFile, manifest: dict, seed: int, sample_size: int
) -> tuple[pd.DataFrame, str]:
    record = next(
        item
        for item in manifest["datasets"]
        if item["seed"] == seed and item["sample_size"] == sample_size
    )
    raw = archive.read(_member(record["path"]))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"dataset checksum mismatch for seed={seed}, n={sample_size}")
    frame = pd.read_csv(io.BytesIO(raw))
    original_indices = [int(item["original_index"]) for item in manifest["nodes"]]
    expected_columns = [str(index) for index in original_indices]
    if [str(column) for column in frame.columns] != expected_columns or frame.shape != (sample_size, 20):
        raise ValueError(f"dataset layout mismatch for seed={seed}, n={sample_size}")
    canonical_names = [CANONICAL_VARIABLE_NAMES[index] for index in original_indices]
    manifest_names = [str(item["name"]).replace("\ufffd", "ä") for item in manifest["nodes"]]
    if manifest_names != canonical_names:
        raise ValueError("manifest node mapping differs from the canonical simulator order")
    frame.columns = canonical_names
    if not set(np.unique(frame.to_numpy())).issubset({0, 1}):
        raise ValueError("BFS benchmark expects raw binary observations")
    return frame, digest


def load_truth(archive: zipfile.ZipFile):
    payload = _read_json(archive, "truth_graph.json")
    edges = []
    for edge in payload["graph"]["edges"]:
        if (edge["mark_u"], edge["mark_v"]) == ("tail", "arrow"):
            edges.append((edge["node_u"], edge["node_v"]))
        elif (edge["mark_u"], edge["mark_v"]) == ("arrow", "tail"):
            edges.append((edge["node_v"], edge["node_u"]))
        else:
            raise ValueError("truth graph is not a DAG")
    return graph_from_edges(payload["graph"]["node_count"], directed=edges)


def graph_payload(edges: Iterable[tuple[int, int]], node_count: int = 20) -> dict:
    records = []
    for source, target in sorted(set(edges), key=lambda edge: (min(edge), max(edge))):
        first, second = sorted((source, target))
        records.append(
            {
                "node_u": first,
                "node_v": second,
                "mark_u": "tail" if source == first else "arrow",
                "mark_v": "arrow" if source == first else "tail",
            }
        )
    return {"kind": "dag", "node_count": node_count, "edges": records}


def result_path(root: Path, seed: int, sample_size: int) -> Path:
    return root / f"bfs2024_seed{seed:02d}_n{sample_size}.result.json"


def metric_path(root: Path, seed: int, sample_size: int) -> Path:
    return root / f"bfs2024_seed{seed:02d}_n{sample_size}.metrics.json"


def run_one(
    archive: zipfile.ZipFile,
    manifest: dict,
    truth,
    seed: int,
    sample_size: int,
    complete: Complete,
    output_dir: Path,
    model: str,
    temperature: float,
    include_statistics: bool,
) -> dict:
    frame, digest = load_dataset(archive, manifest, seed, sample_size)
    started = time.perf_counter()
    edges, usage = run_bfs(
        frame,
        complete,
        include_statistics=include_statistics,
        descriptions=DESCRIPTIONS,
        domain_context="neuropathic pain diagnosis",
    )
    runtime = time.perf_counter() - started
    estimate = graph_from_edges(20, directed=edges)
    metrics = point_prediction_metrics(estimate, truth)
    payload = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "method": "BFS2024+statistics" if include_statistics else "BFS2024",
        "seed": seed,
        "sample_size": sample_size,
        "dataset_sha256": digest,
        "status": "success",
        "output_type": "point",
        "graph": graph_payload(edges),
        "runtime_seconds": runtime,
        "random_seed": None,
        "cost": {
            "scope": "per_trajectory" if include_statistics else "shared_once",
            **usage,
        },
        "hyperparameters": {
            "model": model,
            "temperature": temperature,
            "include_pearson_correlations": include_statistics,
            "query_policy": "Jiralerspong2024_BFS",
            "cycle_policy": "Algorithm_1_DFS_reject_cycle_creating_edge",
        },
        "software": {"implementation": "cd_llm.baselines.jiralerspong2024"},
        "notes": "Truth loaded only after inference for evaluation.",
    }
    write_json(result_path(output_dir, seed, sample_size), payload)
    write_json(metric_path(output_dir, seed, sample_size), metrics)
    return payload


def aggregate(output_dir: Path) -> list[dict]:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in output_dir.glob("*.result.json")]
    rows = []
    metric_names = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
    for sample_size in SAMPLE_SIZES:
        group = [run for run in runs if run["sample_size"] == sample_size and run["status"] == "success"]
        row: dict[str, int | float] = {"sample_size": sample_size, "run_count": len(group)}
        for metric in metric_names:
            values = [
                float(
                    json.loads(
                        metric_path(output_dir, run["seed"], sample_size).read_text(encoding="utf-8")
                    )[metric]
                )
                for run in group
            ]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            row[f"{metric}_se"] = (
                statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        row["query_count"] = sum(run["cost"]["query_count"] for run in group)
        row["input_tokens"] = sum(run["cost"]["input_tokens"] for run in group)
        row["output_tokens"] = sum(run["cost"]["output_tokens"] for run in group)
        rows.append(row)
    write_csv(output_dir / "summary.csv", rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(SAMPLE_SIZES))
    parser.add_argument("--without-statistics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_RESULTS_ROOT / run_id(args.run_id) / "neuropathic_bfs2024"
    if args.aggregate_only:
        print(json.dumps(aggregate(args.output_dir), indent=2))
        return
    complete = openai_completer(args.model, args.temperature, args.base_url)
    with zipfile.ZipFile(args.archive) as archive:
        manifest = load_manifest(archive)
        truth = load_truth(archive)
        for sample_size in args.sample_sizes:
            if sample_size not in SAMPLE_SIZES:
                raise ValueError(f"unsupported sample size: {sample_size}")
            for seed in args.seeds:
                if seed not in SEEDS:
                    raise ValueError(f"unsupported seed: {seed}")
                output = result_path(args.output_dir, seed, sample_size)
                if output.exists() and not args.overwrite:
                    print(f"skip existing {output}", flush=True)
                    continue
                print(f"run BFS2024 seed={seed}, n={sample_size}", flush=True)
                run_one(
                    archive, manifest, truth, seed, sample_size, complete,
                    args.output_dir, args.model, args.temperature,
                    not args.without_statistics,
                )
    print(json.dumps(aggregate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
