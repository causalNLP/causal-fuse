"""Report completed faithful gPC conditions without altering the registered study."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np

from cd_llm.core.finite_support import (
    PartialGraph,
    distribution_metrics,
    graph_from_edges,
    point_prediction_metrics,
)
from cd_llm.experiments.alarm_matched_baselines import CONDITIONS
from cd_llm.experiments.alarm_matched_store import Store
from cd_llm.experiments.common.artifacts import read_json, sha256, write_json


METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
COMPARATORS = ("pc", "ges", "notears", "bfs", "long", "order", "scp")


def score(record: dict, truth) -> dict:
    if "distribution" in record:
        graphs = [graph_from_edges(37, directed=row["edges"]) for row in record["distribution"]]
        weights = np.asarray([row["weight"] for row in record["distribution"]])
        return distribution_metrics(graphs, weights, truth, "dag", normalized=False)
    return point_prediction_metrics(
        PartialGraph.from_matrix(np.asarray(record["matrix"]), "cpdag"), truth, normalized=False
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    store = Store(root)
    predictions = store.items("prediction")
    truth_record = read_json(root / "evaluation/truth_graph.json")
    truth = graph_from_edges(37, directed=truth_record["graph"]["directed"])
    completed = [i for i in range(20) if store.terminal("prediction", f"gpc:{i}")]
    pending = [i for i in range(20) if i not in completed]
    rows: list[dict] = []
    comparison_rows: list[dict] = []
    for i in completed:
        seed, sample_size = CONDITIONS[i]
        record = predictions[f"gpc:{i}"]
        metrics = score(record, truth)
        rows.append(dict(condition=i, seed=seed, sample_size=sample_size,
                         seconds=record.get("seconds"), **{key: metrics[key] for key in METRICS}))
    for sample_size in (200, 1000):
        indices = [i for i in completed if CONDITIONS[i][1] == sample_size]
        for method in ("gpc", *COMPARATORS):
            values = []
            for i in indices:
                record = predictions[f"{method}:{i}"]
                metrics = score(record, truth)
                values.append(metrics)
            row = dict(method=method, sample_size=sample_size, seeds=len(indices),
                       seed_ids=";".join(str(CONDITIONS[i][0]) for i in indices))
            for key in METRICS:
                samples = [float(value[key]) for value in values]
                row[key + "_mean"] = statistics.mean(samples)
                row[key + "_se"] = statistics.stdev(samples) / len(samples) ** 0.5 if len(samples) > 1 else 0.0
            comparison_rows.append(row)
    write_csv(output / "gpc_per_seed.csv", rows)
    write_csv(output / "matched_partial_summary.csv", comparison_rows)
    usage = store.usage("gpc:shared")
    manifest = dict(
        study="alarm_gpc_partial_report", source=str(root), created=time.time(),
        completed_conditions=completed, pending_conditions=pending,
        complete_n200=sum(CONDITIONS[i][1] == 200 for i in completed),
        complete_n1000=sum(CONDITIONS[i][1] == 1000 for i in completed),
        source_manifest_sha256=sha256(root / "manifest.json"),
        truth_sha256=sha256(root / "evaluation/truth_graph.json"), gpc_shared_usage=usage,
        caveat="n=1000 is a six-seed provisional complete-case summary; it is not the registered ten-seed result.",
    )
    write_json(output / "manifest.json", manifest)
    gpc = {(row["sample_size"]): row for row in comparison_rows if row["method"] == "gpc"}
    lines = [
        "# Partial faithful gPC evaluation", "",
        f"Completed {len(completed)}/20 conditions: 10/10 at n=200 and 6/10 at n=1000.",
        "The n=1000 result is provisional and uses only seeds shared by every method in this partial table.", "",
        "| n | Seeds | ED (SE) | ES (SE) | Adj. F1 (SE) | Orient. F1 (SE) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for sample_size in (200, 1000):
        row = gpc[sample_size]
        cell = lambda key: f"{row[key + '_mean']:.3f} ({row[key + '_se']:.3f})"
        lines.append(f"| {sample_size} | {row['seeds']} | {cell('expected_distance')} | "
                     f"{cell('energy_score')} | {cell('adjacency_f1')} | {cell('orientation_f1')} |")
    lines += ["", f"Shared expert usage: {json.dumps(usage, sort_keys=True)}"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "completion.json", dict(complete=True, partial=True, rows=len(rows),
                                                   source_conditions=20, generated=time.time()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
