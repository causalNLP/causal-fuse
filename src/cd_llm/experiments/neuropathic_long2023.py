"""Evaluate Long et al. (2023) on the neuropathic benchmark panels."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from cd_llm.baselines.long2023 import (
    ambiguous_pairs,
    enumerate_long_mec,
    fit_pc,
    openrouter_expert,
    run_long,
)
from cd_llm.core.finite_support import distribution_metrics
from cd_llm.experiments.common.artifacts import run_id, write_csv, write_json
from cd_llm.experiments.neuropathic_bfs2024 import (
    DEFAULT_ARCHIVE,
    DESCRIPTIONS as PANEL_C_DESCRIPTIONS,
    load_dataset as load_panel_c_dataset,
    load_manifest,
    load_truth as load_panel_c_truth,
)
from cd_llm.methods.adaptive_pruning import CompletionEnumerationLimitError
from cd_llm.metrics.units import metric_metadata
from cd_llm.paths import WORKSPACE_ROOT


SEEDS = tuple(range(20))
SAMPLE_SIZES = (200, 1000)


def full_variable_name(name: str) -> str:
    """Return prompt metadata without exposing abbreviated identifiers."""
    if name in PANEL_C_DESCRIPTIONS:
        return PANEL_C_DESCRIPTIONS[name]
    if name.startswith("DLS "):
        return f"disc lesion or disorder at spinal level {name[4:]}"
    parts = name.split()
    if len(parts) == 3 and parts[0] in {"L", "R"} and parts[2] == "Radikulopati":
        side = "left" if parts[0] == "L" else "right"
        return f"{side} {parts[1]} radiculopathy"
    return name


def _load_condition(args, seed: int, sample_size: int):
    import zipfile

    archive = zipfile.ZipFile(args.archive)
    manifest = load_manifest(archive)
    frame, digest = load_panel_c_dataset(archive, manifest, seed, sample_size)
    truth = load_panel_c_truth(archive)
    return frame.to_numpy(), tuple(frame.columns), truth, digest


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        choices=("panel_c_20",),
        default="panel_c_20",
    )
    parser.add_argument("--strategy", choices=("srisk", "ssize"), default="srisk")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=list(SAMPLE_SIZES)
    )
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-extensions", type=int, default=50_000)
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--llm-preflight-only", action="store_true")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expert-cache", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            WORKSPACE_ROOT / "results" / run_id(args.run_id) / "long2023" / args.panel
        )
    if args.expert_cache is None:
        args.expert_cache = args.output_dir / "expert_cache.json"
    if args.preflight_only and args.llm_preflight_only:
        raise SystemExit("choose only one preflight mode")
    if not args.preflight_only and not args.model:
        raise SystemExit("--model is required unless --preflight-only is used")
    if not args.preflight_only and not (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    ):
        raise SystemExit(
            "OPENAI_API_KEY or OPENROUTER_API_KEY must be set before paid runs"
        )
    if args.llm_preflight_only:
        _, names, _, _ = _load_condition(args, args.seeds[0], args.sample_sizes[0])
        metadata = tuple(full_variable_name(name) for name in names)
        expert = openrouter_expert(
            args.model, args.base_url, args.prompt_seed, args.expert_cache
        )
        arc, confidence, record = expert(0, 1, metadata[0], metadata[1])
        print(
            json.dumps(
                {
                    "status": "llm_preflight_passed",
                    "model": args.model,
                    "arc": arc,
                    "confidence": confidence,
                    "record": record,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    records = []
    for sample_size in args.sample_sizes:
        for seed in args.seeds:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / (
                f"long2023_{args.panel}_{args.strategy}_seed{seed:02d}_n{sample_size}.json"
            )
            if path.exists() and not args.overwrite:
                print(f"skip existing {path}", flush=True)
                records.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            data, names, truth, digest = _load_condition(args, seed, sample_size)
            metadata = tuple(full_variable_name(name) for name in names)
            alpha = 0.01
            print(f"fit PC {args.panel} seed={seed}, n={sample_size}", flush=True)
            pc_started = time.perf_counter()
            cpdag = fit_pc(data, names, alpha)
            pc_seconds = time.perf_counter() - pc_started
            print(
                f"enumerate MEC {args.panel} seed={seed}, n={sample_size}, "
                f"ambiguous={len(ambiguous_pairs(cpdag))}",
                flush=True,
            )
            mec_started = time.perf_counter()
            try:
                mec = enumerate_long_mec(cpdag, args.max_extensions)
                status, error = "preflight_passed", None
            except CompletionEnumerationLimitError as exc:
                mec, status, error = (), "not_scalable", str(exc)
            row = {
                "panel": args.panel,
                "seed": seed,
                "sample_size": sample_size,
                "data_sha256": digest,
                "pc_alpha": alpha,
                "ambiguous_edge_count": len(ambiguous_pairs(cpdag)),
                "initial_mec_size": len(mec) if mec else None,
                "pc_seconds": pc_seconds,
                "mec_enumeration_seconds": time.perf_counter() - mec_started,
                "status": status,
                "error": error,
            }
            if not args.preflight_only and status == "preflight_passed":
                expert = openrouter_expert(
                    args.model, args.base_url, args.prompt_seed, args.expert_cache
                )
                method_started = time.perf_counter()
                result = run_long(
                    cpdag,
                    metadata,
                    expert,
                    strategy=args.strategy,
                    tolerance=args.tolerance,
                    max_extensions=args.max_extensions,
                    initial_mec=mec,
                )
                probabilities = np.full(len(result.final_mec), 1 / len(result.final_mec))
                row.update(
                    {
                        "status": "success",
                        "strategy": args.strategy,
                        "model": args.model,
                        "runtime_seconds": time.perf_counter() - method_started,
                        "final_mec_size": len(result.final_mec),
                        "retained_probability": result.retained_probability,
                        "query_count": len(result.expert_records),
                        "input_tokens": sum(
                            int(item["input_tokens"]) for item in result.expert_records
                        ),
                        "output_tokens": sum(
                            int(item["output_tokens"]) for item in result.expert_records
                        ),
                        "cache_hits": sum(
                            bool(item["cache_hit"]) for item in result.expert_records
                        ),
                        "expert_records": result.expert_records,
                        "metadata_policy": "expanded_names_only",
                        "metrics": distribution_metrics(
                            result.final_mec, probabilities, truth, "dag"
                        ),
                        "initial_truth_coverage": int(truth in result.initial_mec),
                        "final_truth_coverage": int(truth in result.final_mec),
                    }
                )
            write_json(path, row)
            records.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    if not args.preflight_only:
        write_summary(args.output_dir, records)


def write_summary(output_dir: Path, records: Sequence[dict]) -> None:
    successful = [record for record in records if record.get("status") == "success"]
    per_seed = []
    for record in successful:
        metrics = record["metrics"]
        per_seed.append({
            "method": "PC+Long-Srisk", "sample_size": int(record["sample_size"]),
            "seed": int(record["seed"]),
            **{name: float(metrics[name]) for name in (
                "expected_distance", "energy_score", "adjacency_f1", "orientation_f1")},
            "logical_requests": int(record["query_count"]),
            "input_tokens": int(record["input_tokens"]),
            "output_tokens": int(record["output_tokens"]),
            "total_tokens": int(record["input_tokens"]) + int(record["output_tokens"]),
            "runtime_seconds": float(record["runtime_seconds"]),
        })
    rows = []
    for sample_size in sorted({row["sample_size"] for row in per_seed}):
        selected = [row for row in per_seed if row["sample_size"] == sample_size]
        row = {"method": "PC+Long-Srisk", "sample_size": sample_size,
               "replicate_count": len(selected)}
        for name in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1",
                     "logical_requests", "input_tokens", "output_tokens", "total_tokens", "runtime_seconds"):
            values = [float(item[name]) for item in selected]
            row[name + "_mean"] = statistics.fmean(values)
            row[name + "_se"] = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(row)
    if not rows:
        raise RuntimeError("PC+Long produced no successful records")
    write_csv(output_dir / "per_seed.csv", per_seed)
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "per_seed.csv.metrics.json", metric_metadata(20))
    write_json(output_dir / "summary.csv.metrics.json", metric_metadata(20))


if __name__ == "__main__":
    main()
