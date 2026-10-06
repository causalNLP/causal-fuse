"""Freeze the verified 10/3 ALARM gPC CPDAG results for interim reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from cd_llm.core.finite_support import PartialGraph, point_prediction_metrics
from cd_llm.experiments.alarm_gpc_cpdag_replay import (
    CONDITIONS, METRICS, SOURCE, source_digest, source_item, truth_graph,
)
from cd_llm.experiments.common.artifacts import read_json, sha256, write_json
from cd_llm.paths import WORKSPACE_ROOT


CONDITION_IDS = (0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 16, 18)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot(replay_root: Path, output: Path) -> None:
    replay_manifest = read_json(replay_root / "manifest.json")
    if replay_manifest["study"] != "alarm_shared_skeleton_gpc_cpdag_replay":
        raise ValueError("Wrong replay study")
    if replay_manifest["conditions"] != [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]:
        raise ValueError("Replay source conditions changed")
    if replay_manifest["source_manifest_sha256"] != sha256(SOURCE / "manifest.json"):
        raise ValueError("Matched source manifest changed")
    truth = truth_graph()
    rows = []
    records = []
    for index in CONDITION_IDS:
        path = replay_root / "predictions" / f"{index}.json"
        record = read_json(path)
        seed, n = CONDITIONS[index]
        if (record["condition"], record["seed"], record["sample_size"]) != (index, seed, n):
            raise ValueError(f"Condition identity mismatch {index}")
        source = source_item("prediction", f"gpc:{index}")
        if record["source_prediction_digest"] != source_digest(source):
            raise ValueError(f"Source prediction digest mismatch {index}")
        if not np.array_equal(record["skeleton"], source["matrix"]):
            raise ValueError(f"Replayed skeleton mismatch {index}")
        cpdag = PartialGraph(37, tuple(record["cpdag_marks"]), "cpdag")
        metrics = point_prediction_metrics(cpdag, truth, normalized=False)
        for name in METRICS:
            if not np.isclose(metrics[name], record["metrics"][name], rtol=1e-10, atol=1e-10):
                raise ValueError(f"CPDAG metric mismatch {index}: {name}")
        rows.append({
            "method": "gpc", "condition": index, "seed": seed, "sample_size": n,
            **{name: float(metrics[name]) for name in METRICS},
            "source_search_seconds": record["source_search_seconds"],
            "orientation_seconds": record["orientation_seconds"],
            "numerical_seconds": record["source_search_seconds"] + record["orientation_seconds"],
            "ci_tests": record["ci_tests"],
            "certification_changed": record["certification_changed"],
        })
        records.append((index, path))
    counts = {n: sum(row["sample_size"] == n for row in rows) for n in (200, 1000)}
    if counts != {200: 10, 1000: 3}:
        raise ValueError(f"Unexpected interim counts: {counts}")
    if {row["seed"] for row in rows if row["sample_size"] == 1000} != {1001, 1003, 1004}:
        raise ValueError("Unexpected interim n=1000 seeds")
    summary = []
    for n in (200, 1000):
        group = [row for row in rows if row["sample_size"] == n]
        result = {"method": "gpc", "sample_size": n, "complete_seeds": len(group),
                  "planned_seeds": 10, "seed_ids": ";".join(str(row["seed"]) for row in group)}
        for name in (*METRICS, "numerical_seconds"):
            values = [float(row[name]) for row in group]
            result[name + "_mean"] = statistics.mean(values)
            result[name + "_se"] = statistics.stdev(values) / len(values) ** 0.5
        summary.append(result)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    for index, path in records:
        shutil.copyfile(path, output / "predictions" / f"{index}.json")
    manifest = {
        "study": "alarm_shared_skeleton_gpc_cpdag_interim",
        "source": str(SOURCE.relative_to(WORKSPACE_ROOT)),
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "replay_source": str(replay_root.relative_to(WORKSPACE_ROOT)),
        "replay_manifest_sha256": sha256(replay_root / "manifest.json"),
        "conditions": list(CONDITION_IDS), "planned_conditions": 20,
        "shared_usage": replay_manifest["shared_usage"],
        "metric_normalized": False,
    }
    write_json(output / "manifest.json", manifest)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "provenance.json", {
        "manifest_sha256": sha256(output / "manifest.json"),
        "per_seed_sha256": sha256(output / "per_seed.csv"),
        "summary_sha256": sha256(output / "summary.csv"),
        "prediction_sha256": {str(i): sha256(output / "predictions" / f"{i}.json")
                              for i in CONDITION_IDS},
        "completion": {"n200": 10, "n1000": 3},
        "partial_n1000": True,
    })
    print(output / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot(args.replay_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
