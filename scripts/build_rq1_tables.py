"""Combine fresh experiment summaries into the two Section 4 RQ1 tables."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from cd_llm.metrics.units import read_metric_csv  # noqa: E402


REQUIRED = {
    "neuropathic": {
        "data": {"PC", "GES"},
        "terra": {"gPC-Guess", "BFS+Pearson", "Causal Order+PC", "Claim-Fuse (stated)", "Claim-Fuse (consistency)"},
        "qwen": {"gPC-Guess", "SCP Pattern 2", "BFS+Pearson", "Causal Order+PC", "PC+Long-Srisk", "Claim-Fuse (stated)", "Claim-Fuse (consistency)"},
    },
    "alarm": {
        "data": {"PC", "GES", "NOTEARS"},
        "qwen": {"gPC-Guess", "BFS+Pearson", "PC+Long-Srisk", "Causal Order+PC", "SCP Pattern 2", "Claim-Fuse (stated)", "Claim-Fuse (consistency)"},
    },
}


def source(value: str) -> tuple[str, str | None, Path]:
    try:
        descriptor, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source must be BLOCK[:METHOD]=PATH") from exc
    block, separator, method = descriptor.partition(":")
    if block not in {"data", "qwen", "terra"}:
        raise argparse.ArgumentTypeError("BLOCK must be data, qwen, or terra")
    return block, method if separator else None, Path(path)


def method_name(row: dict, default: str | None) -> str | None:
    if default is not None:
        return default
    method = row.get("method", "").strip()
    if not method and row.get("group") == "all" and row.get("strategy") in {"stated", "consistency"}:
        return f"Claim-Fuse ({row['strategy']})"
    replacements = {
        "DAG-gPC-Guess + PC orientation": "gPC-Guess",
        "SMC-Active": f"Claim-Fuse ({row.get('strategy')})",
        "PC+Long-$S_{risk}$": "PC+Long-Srisk",
        "Causal Order+PC (Qwen3.5-9B)": "Causal Order+PC",
        "pc": "PC", "ges": "GES", "notears": "NOTEARS",
        "gpc": "gPC-Guess", "bfs": "BFS+Pearson",
        "long": "PC+Long-Srisk", "order": "Causal Order+PC",
        "scp": "SCP Pattern 2",
    }
    method = replacements.get(method, method)
    for suffix in (" + Terra", " + Qwen3.5-9B"):
        method = method.removesuffix(suffix)
    return method or None


def collect(dataset: str, entries: list[tuple[str, str | None, Path]]) -> list[dict]:
    node_count = 20 if dataset == "neuropathic" else 37
    target_budget = 100 if dataset == "neuropathic" else 50
    indexed = {}
    for block, default_method, path in entries:
        for row in read_metric_csv(path, node_count=node_count):
            model = row.get("model", "").lower()
            if model and block == "qwen" and "qwen" not in model:
                continue
            if model and block == "terra" and "terra" not in model:
                continue
            method = method_name(row, default_method)
            if method is None or method not in REQUIRED[dataset].get(block, set()):
                continue
            if method.startswith("Claim-Fuse"):
                if row.get("group") != "all" or int(row.get("budget", -1)) != target_budget:
                    continue
            sample_size = int(row["sample_size"])
            if sample_size not in (200, 1000):
                continue
            count = int(row.get("seeds") or row.get("replicate_count") or
                        row.get("run_count") or row.get("complete") or 0)
            expected_count = 3 if (dataset, block, method, sample_size) == (
                "alarm", "qwen", "gPC-Guess", 1000) else 10
            if count != expected_count:
                raise ValueError(
                    f"{path}: {block}/{method}/n={sample_size} has {count} replicates; "
                    f"expected {expected_count}")
            key = block, method, sample_size
            if key in indexed:
                raise ValueError(f"duplicate RQ1 cell: {key}")
            values = dict(
                block=block, method=method, sample_size=sample_size, seeds=count,
                expected_distance_mean=float(row["expected_distance_mean"]),
                expected_distance_se=float(row["expected_distance_se"]),
                orientation_f1_mean=float(row["orientation_f1_mean"]),
                orientation_f1_se=float(row["orientation_f1_se"]),
                source=str(path),
            )
            if not all(math.isfinite(value) for name, value in values.items() if name.endswith(("_mean", "_se"))):
                raise ValueError(f"nonfinite RQ1 cell: {key}")
            indexed[key] = values
    expected = {(block, method, n) for block, methods in REQUIRED[dataset].items()
                for method in methods for n in (200, 1000)}
    if set(indexed) != expected:
        raise ValueError(f"{dataset} RQ1 table incomplete: missing={sorted(expected-set(indexed))}")
    return [indexed[key] for key in sorted(indexed)]


def write(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuropathic-source", action="append", type=source, required=True)
    parser.add_argument("--alarm-source", action="append", type=source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write(args.output_dir / "neuropathic_rq1.csv", collect("neuropathic", args.neuropathic_source))
    write(args.output_dir / "alarm_rq1.csv", collect("alarm", args.alarm_source))
    print(args.output_dir)


if __name__ == "__main__":
    main()
