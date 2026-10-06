"""Build the Section 4 RQ3 cost tables from fresh experiment directories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls


FIELDS = (
    "dataset", "method", "source_method", "sample_size", "strategy", "group",
    "budget", "seeds", "expected_distance_mean", "expected_distance_se",
    "approx_runtime_seconds_mean", "approx_runtime_seconds_se", "total_tokens_mean",
    "total_tokens_se", "input_tokens_mean", "input_tokens_se", "output_tokens_mean",
    "output_tokens_se", "response_events_mean", "response_events_se", "source_file",
)
CHECKPOINTS = {10, 20, 50, 100}
BASELINE_LABELS = {
    "gpc": "gPC-Guess", "bfs": "BFS+Pearson", "long": "PC+Long-Srisk",
    "order": "Causal Order+PC", "scp": "SCP Pattern 2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latency_fit(root: Path) -> tuple[float, float, float]:
    records = {}
    for path in root.glob("**/cache/raw_responses.json"):
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            record = value["record"]
            if key in records and records[key] != record:
                raise ValueError(f"conflicting response record: {key}")
            records[key] = record
    points = [(1.0, float(row["prompt_tokens"]), float(row["completion_tokens"]),
               float(row["latency_seconds"])) for row in records.values()
              if float(row["latency_seconds"]) >= 0]
    if not points:
        raise ValueError(f"no timed model responses under {root}")
    coefficients, _ = nnls(np.asarray([point[:3] for point in points]),
                            np.asarray([point[3] for point in points]))
    return tuple(float(value) for value in coefficients)


def cost_row(*, dataset: str, method: str, source_method: str, sample_size: int,
             strategy: str, group: str, budget: int, seed: int,
             expected_distance: float, input_tokens: float, output_tokens: float,
             response_events: float, numerical_seconds: float, source_file: Path,
             fit: tuple[float, float, float]) -> dict:
    alpha, beta_input, beta_output = fit
    return {
        "dataset": dataset, "method": method, "source_method": source_method,
        "sample_size": sample_size, "strategy": strategy, "group": group,
        "budget": budget, "seed": seed, "expected_distance": expected_distance,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens, "response_events": response_events,
        "approx_runtime_seconds": numerical_seconds + alpha * response_events
            + beta_input * input_tokens + beta_output * output_tokens,
        "source_file": str(source_file),
    }


def claim_fuse_rows(root: Path, dataset: str, fit: tuple[float, float, float]) -> list[dict]:
    path = root / "per_run.csv"
    result = []
    for row in read_csv(path):
        budget = int(row["budget"])
        if row["group"] != "all" or budget not in CHECKPOINTS:
            continue
        result.append(cost_row(
            dataset=dataset, method="Claim-Fuse", source_method="Claim-Fuse",
            sample_size=int(row["sample_size"]), strategy=row["strategy"], group="all",
            budget=budget, seed=int(row["seed"]), expected_distance=float(row["expected_distance"]),
            input_tokens=float(row["input_tokens"]), output_tokens=float(row["output_tokens"]),
            response_events=float(row["logical_requests"]),
            numerical_seconds=float(row.get("elapsed_seconds") or 0), source_file=path, fit=fit))
    return result


def neuropathic_gpc(root: Path, fit: tuple[float, float, float]) -> list[dict]:
    path = root / "per_seed.csv"
    return [cost_row(
        dataset="neuropathic_pain", method="gPC-Guess", source_method=row["method"],
        sample_size=int(row["sample_size"]), strategy="native", group="", budget=0,
        seed=int(row["seed"]), expected_distance=float(row["expected_distance"]),
        input_tokens=float(row["input_tokens"]), output_tokens=float(row["output_tokens"]),
        response_events=float(row["logical_requests"]), numerical_seconds=float(row["runtime_seconds"]),
        source_file=path, fit=fit) for row in read_csv(path) if row["model_key"] == "qwen"]


def neuropathic_bfs(root: Path, fit: tuple[float, float, float]) -> list[dict]:
    result = []
    for path in root.glob("*.result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "success":
            continue
        metric_path = path.with_name(path.name.replace(".result.json", ".metrics.json"))
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        usage = row["cost"]
        result.append(cost_row(dataset="neuropathic_pain", method="BFS+Pearson",
            source_method=row["method"], sample_size=int(row["sample_size"]), strategy="native",
            group="", budget=0, seed=int(row["seed"]),
            expected_distance=float(metrics["expected_distance"]),
            input_tokens=float(usage["input_tokens"]), output_tokens=float(usage["output_tokens"]),
            response_events=float(usage["query_count"]), numerical_seconds=0, source_file=path, fit=fit))
    return result


def summarized_baseline(root: Path, method: str, fit: tuple[float, float, float]) -> list[dict]:
    path = root / "per_seed.csv"
    rows = []
    for row in read_csv(path):
        rows.append(cost_row(dataset="neuropathic_pain", method=method, source_method=row["method"],
            sample_size=int(row["sample_size"]), strategy="native", group="", budget=0,
            seed=int(row["seed"]), expected_distance=float(row["expected_distance"]),
            input_tokens=float(row["input_tokens"]), output_tokens=float(row["output_tokens"]),
            response_events=float(row.get("logical_requests") or row.get("query_count") or 0),
            numerical_seconds=float(row.get("runtime_seconds") or 0), source_file=path, fit=fit))
    return rows


def neuropathic_order(root: Path, fit: tuple[float, float, float]) -> list[dict]:
    path = root / "raw_results.csv"
    usage = json.loads((root / "token_summary.json").read_text(encoding="utf-8"))
    return [cost_row(dataset="neuropathic_pain", method="Causal Order+PC",
        source_method=row["method"], sample_size=int(row["sample_size"]), strategy="native",
        group="", budget=0, seed=int(row["seed"]),
        expected_distance=float(row["expected_distance"]), input_tokens=float(usage["input_tokens"]),
        output_tokens=float(usage["output_tokens"]), response_events=float(usage["request_count"]),
        numerical_seconds=float(row["method_runtime_seconds"]), source_file=path, fit=fit)
        for row in read_csv(path) if row["method"].startswith("Causal Order")]


def alarm_baselines(root: Path, fit: tuple[float, float, float]) -> list[dict]:
    path = root / "per_seed.csv"
    result = []
    for row in read_csv(path):
        method = BASELINE_LABELS.get(row["method"])
        if method is None or row["status"] != "complete":
            continue
        result.append(cost_row(dataset="alarm", method=method, source_method=row["method"],
            sample_size=int(row["sample_size"]), strategy="native", group="", budget=0,
            seed=int(row["seed"]), expected_distance=float(row["expected_distance"]),
            input_tokens=float(row["input_tokens"]), output_tokens=float(row["output_tokens"]),
            response_events=float(row["logical_requests"]),
            numerical_seconds=float(row.get("seconds") or 0), source_file=path, fit=fit))
    return result


def summarize(rows: list[dict], dataset: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["sample_size"], row["strategy"], row["group"], row["budget"])].append(row)
    output = []
    for (method, sample_size, strategy, group, budget), values in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate RQ3 seed for {(dataset, method, sample_size, strategy, group, budget)}")
        item = {"dataset": dataset, "method": method, "source_method": values[0]["source_method"],
                "sample_size": sample_size, "strategy": strategy, "group": group,
                "budget": budget, "seeds": len(values), "source_file": values[0]["source_file"]}
        for metric in ("expected_distance", "approx_runtime_seconds", "total_tokens",
                       "input_tokens", "output_tokens", "response_events"):
            numbers = [float(row[metric]) for row in values]
            item[metric + "_mean"] = statistics.fmean(numbers)
            item[metric + "_se"] = statistics.stdev(numbers) / math.sqrt(len(numbers)) if len(numbers) > 1 else 0.0
        output.append(item)
    claim = [row for row in output if row["method"] == "Claim-Fuse"]
    expected_claim = {(n, strategy, budget) for n in (200, 1000)
                      for strategy in ("stated", "consistency") for budget in CHECKPOINTS}
    actual_claim = {(row["sample_size"], row["strategy"], row["budget"]) for row in claim}
    if actual_claim != expected_claim or any(row["seeds"] != 10 for row in claim):
        raise ValueError(f"{dataset} Claim-Fuse RQ3 cells are incomplete")
    baselines = [row for row in output if row["strategy"] == "native"]
    expected_baselines = {(method, n) for method in BASELINE_LABELS.values() for n in (200, 1000)}
    actual_baselines = {(row["method"], row["sample_size"]) for row in baselines}
    if actual_baselines != expected_baselines:
        raise ValueError(f"{dataset} baseline RQ3 cells are incomplete: missing={sorted(expected_baselines-actual_baselines)}")
    return output


def write_output(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({name: row[name] for name in FIELDS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuropathic-claim-fuse", type=Path, required=True)
    parser.add_argument("--neuropathic-cache", type=Path, required=True)
    parser.add_argument("--neuropathic-gpc", type=Path, required=True)
    parser.add_argument("--neuropathic-bfs", type=Path, required=True)
    parser.add_argument("--neuropathic-long", type=Path, required=True)
    parser.add_argument("--neuropathic-order", type=Path, required=True)
    parser.add_argument("--neuropathic-scp", type=Path, action="append", required=True)
    parser.add_argument("--alarm-stated", type=Path, required=True)
    parser.add_argument("--alarm-consistency", type=Path, required=True)
    parser.add_argument("--alarm-baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    neuropathic_fit = latency_fit(args.neuropathic_cache)
    alarm_fit = latency_fit(args.alarm_stated)
    neuropathic = claim_fuse_rows(args.neuropathic_claim_fuse, "neuropathic_pain", neuropathic_fit)
    neuropathic += neuropathic_gpc(args.neuropathic_gpc, neuropathic_fit)
    neuropathic += neuropathic_bfs(args.neuropathic_bfs, neuropathic_fit)
    neuropathic += summarized_baseline(args.neuropathic_long, "PC+Long-Srisk", neuropathic_fit)
    neuropathic += neuropathic_order(args.neuropathic_order, neuropathic_fit)
    for root in args.neuropathic_scp:
        neuropathic += summarized_baseline(root, "SCP Pattern 2", neuropathic_fit)
    alarm = claim_fuse_rows(args.alarm_stated, "alarm", alarm_fit)
    alarm += claim_fuse_rows(args.alarm_consistency, "alarm", alarm_fit)
    alarm += alarm_baselines(args.alarm_baselines, alarm_fit)
    write_output(args.output_dir / "neuropathic_pain_all_methods.csv",
                 summarize(neuropathic, "neuropathic_pain"))
    write_output(args.output_dir / "alarm_all_methods.csv", summarize(alarm, "alarm"))
    print(args.output_dir)


if __name__ == "__main__":
    main()
