"""Replay completed shared-skeleton ALARM gPC searches for CPDAG scoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

import numpy as np

from cd_llm.baselines.guess2graph2026.dag_cpdag import orient_skeleton_as_cpdag
from cd_llm.core.finite_support import PartialGraph, graph_from_edges, point_prediction_metrics
from cd_llm.experiments.alarm_matched_disk import DiskCheckpoint
from cd_llm.experiments.alarm_matched_gpc import search
from cd_llm.experiments.alarm_matched_store import digest
from cd_llm.experiments.common.artifacts import read_json, sha256, write_json
from cd_llm.paths import WORKSPACE_ROOT


SOURCE = WORKSPACE_ROOT / "results/20260921T013135Z/alarm_matched_baselines"
PARTIAL = WORKSPACE_ROOT / "results/20260923T141147Z/alarm_gpc_partial"
CONDITIONS = [(seed, n) for seed in range(1000, 1010) for n in (200, 1000)]
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")


def source_item(kind: str, key: str) -> dict:
    uri = f"file:{SOURCE / 'cache.sqlite3'}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        row = db.execute("SELECT value FROM items WHERE kind=? AND key=?", (kind, key)).fetchone()
    if row is None:
        raise ValueError(f"Missing source item {kind}:{key}")
    return json.loads(row[0])


def source_digest(value: object) -> str:
    return digest(value)


def implementation() -> dict[str, str]:
    names = (
        "code/src/cd_llm/experiments/alarm_matched_gpc.py",
        "code/src/cd_llm/experiments/alarm_gpc_cpdag_replay.py",
        "code/src/cd_llm/baselines/guess2graph2026/dag_cpdag.py",
        "code/src/cd_llm/core/finite_support.py",
    )
    return {name: sha256(WORKSPACE_ROOT / name) for name in names}


def prepare(root: Path) -> dict:
    if (root / "manifest.json").exists():
        return verify(root)
    partial_manifest = read_json(PARTIAL / "manifest.json")
    indices = partial_manifest["completed_conditions"]
    if indices != [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]:
        raise ValueError("Completed source conditions changed")
    if partial_manifest["source_manifest_sha256"] != sha256(SOURCE / "manifest.json"):
        raise ValueError("Source manifest changed")
    if partial_manifest["truth_sha256"] != sha256(SOURCE / "evaluation/truth_graph.json"):
        raise ValueError("Truth graph changed")
    guesses = source_item("gpc_guess", "shared")
    matrix = np.asarray(guesses["skeleton"], dtype=int)
    if matrix.shape != (37, 37):
        raise ValueError("Unexpected shared guess shape")
    predictions = {str(index): source_digest(source_item("prediction", f"gpc:{index}"))
                   for index in indices}
    inputs = {}
    for index in indices:
        seed, n = CONDITIONS[index]
        path = SOURCE / f"inputs/alarm_s{seed}_n{n}.npy"
        inputs[str(index)] = sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "predictions").mkdir()
    manifest = {
        "study": "alarm_shared_skeleton_gpc_cpdag_replay",
        "source": str(SOURCE.relative_to(WORKSPACE_ROOT)),
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "partial_source": str(PARTIAL.relative_to(WORKSPACE_ROOT)),
        "partial_manifest_sha256": sha256(PARTIAL / "manifest.json"),
        "partial_per_seed_sha256": sha256(PARTIAL / "gpc_per_seed.csv"),
        "truth_sha256": partial_manifest["truth_sha256"],
        "conditions": indices,
        "planned_conditions": len(CONDITIONS),
        "source_prediction_digests": predictions,
        "source_input_sha256": inputs,
        "shared_guess_digest": source_digest(guesses),
        "shared_usage": partial_manifest["gpc_shared_usage"],
        "settings": {"alpha": 0.05, "ci_test": "categorical G-squared",
                     "search": "original shared-skeleton gPC order",
                     "extension": "unshielded colliders, Meek closure, certified CPDAG",
                     "metric_normalized": False},
        "implementation": implementation(),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def verify(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest["study"] != "alarm_shared_skeleton_gpc_cpdag_replay":
        raise ValueError("Wrong replay study")
    if manifest["implementation"] != implementation():
        raise ValueError("Replay implementation changed")
    if manifest["source_manifest_sha256"] != sha256(SOURCE / "manifest.json"):
        raise ValueError("Source manifest changed")
    if manifest["partial_manifest_sha256"] != sha256(PARTIAL / "manifest.json"):
        raise ValueError("Partial source manifest changed")
    if manifest["partial_per_seed_sha256"] != sha256(PARTIAL / "gpc_per_seed.csv"):
        raise ValueError("Partial source metrics changed")
    if manifest["truth_sha256"] != sha256(SOURCE / "evaluation/truth_graph.json"):
        raise ValueError("Truth graph changed")
    if manifest["shared_guess_digest"] != source_digest(source_item("gpc_guess", "shared")):
        raise ValueError("Shared skeleton guess changed")
    for key in manifest["conditions"]:
        index = int(key)
        seed, n = CONDITIONS[index]
        if manifest["source_prediction_digests"][str(index)] != source_digest(
            source_item("prediction", f"gpc:{index}")
        ):
            raise ValueError(f"Source prediction {index} changed")
        if manifest["source_input_sha256"][str(index)] != sha256(
            SOURCE / f"inputs/alarm_s{seed}_n{n}.npy"
        ):
            raise ValueError(f"Source input {index} changed")
    return manifest


def source_metrics() -> dict[int, dict]:
    with (PARTIAL / "gpc_per_seed.csv").open(newline="") as stream:
        return {int(row["condition"]): row for row in csv.DictReader(stream)}


def truth_graph():
    record = read_json(SOURCE / "evaluation/truth_graph.json")
    return graph_from_edges(37, directed=record["graph"]["directed"])


def replay(root: Path, index: int) -> dict:
    manifest = verify(root)
    if index not in manifest["conditions"]:
        raise ValueError(f"Condition {index} was not complete in the frozen source")
    output = root / "predictions" / f"{index}.json"
    if output.exists():
        return read_json(output)
    seed, n = CONDITIONS[index]
    old = source_item("prediction", f"gpc:{index}")
    guess = np.asarray(source_item("gpc_guess", "shared")["skeleton"], dtype=int)
    data = np.load(SOURCE / f"inputs/alarm_s{seed}_n{n}.npy", allow_pickle=False)
    rng = np.random.default_rng(int(digest(["alarm-gpc", seed])[:8], 16))
    deadline = time.time() + float(os.environ.get("ALARM_GPC_REPLAY_SECONDS", "inf"))
    with DiskCheckpoint(root, index) as checkpoint:
        skeleton, diagnostics = search(data, guess, rng, None, str(index), deadline,
                                       checkpoint=checkpoint, return_diagnostics=True)
    if not np.array_equal(skeleton, np.asarray(old["matrix"], dtype=int)):
        raise ValueError(f"Replayed gPC skeleton differs from saved prediction {index}")
    original_metrics = point_prediction_metrics(
        PartialGraph.from_matrix(np.where(skeleton != 0, 1, 0), "cpdag"),
        truth_graph(), normalized=False)
    archived = source_metrics()[index]
    for metric in METRICS:
        if not np.isclose(original_metrics[metric], float(archived[metric]), rtol=1e-10, atol=1e-10):
            raise ValueError(f"Archived skeleton metric {metric} differs for {index}")
    separating_sets = {
        tuple(int(part) for part in pair.split(":")): tuple(int(node) for node in nodes)
        for pair, nodes in diagnostics["separating_sets"].items()
    }
    started = time.monotonic()
    raw, cpdag = orient_skeleton_as_cpdag(skeleton, separating_sets)
    orientation_seconds = time.monotonic() - started
    metrics = point_prediction_metrics(cpdag, truth_graph(), normalized=False)
    record = {
        "condition": index, "seed": seed, "sample_size": n,
        "source_prediction_digest": manifest["source_prediction_digests"][str(index)],
        "skeleton": skeleton.tolist(),
        "separating_sets": {f"{left}:{right}": list(nodes)
                            for (left, right), nodes in sorted(separating_sets.items())},
        "cpdag_marks": list(cpdag.marks),
        "certification_changed": raw != cpdag,
        "ci_tests": diagnostics["ci_tests"],
        "source_search_seconds": float(old["seconds"]),
        "orientation_seconds": orientation_seconds,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
    }
    write_json(output, record)
    return record


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path) -> None:
    manifest = verify(root)
    rows = []
    for index in manifest["conditions"]:
        path = root / "predictions" / f"{index}.json"
        if not path.exists():
            raise ValueError(f"Missing replayed condition {index}")
        record = read_json(path)
        seed, n = CONDITIONS[index]
        if (record["condition"], record["seed"], record["sample_size"]) != (index, seed, n):
            raise ValueError(f"Condition identity mismatch {index}")
        if record["source_prediction_digest"] != manifest["source_prediction_digests"][str(index)]:
            raise ValueError(f"Source prediction mismatch {index}")
        if not np.array_equal(record["skeleton"], source_item("prediction", f"gpc:{index}")["matrix"]):
            raise ValueError(f"Source skeleton mismatch {index}")
        cpdag = PartialGraph(37, tuple(record["cpdag_marks"]), "cpdag")
        metrics = point_prediction_metrics(cpdag, truth_graph(), normalized=False)
        for metric in METRICS:
            if not np.isclose(metrics[metric], record["metrics"][metric], rtol=1e-10, atol=1e-10):
                raise ValueError(f"Stored CPDAG metric mismatch {index}: {metric}")
        rows.append({
            "method": "gpc", "condition": index, "seed": seed, "sample_size": n,
            **record["metrics"],
            "source_search_seconds": record["source_search_seconds"],
            "orientation_seconds": record["orientation_seconds"],
            "numerical_seconds": record["source_search_seconds"] + record["orientation_seconds"],
            "ci_tests": record["ci_tests"],
            "certification_changed": record["certification_changed"],
        })
    if {n: sum(row["sample_size"] == n for row in rows) for n in (200, 1000)} != {200: 10, 1000: 6}:
        raise ValueError("Unexpected completion counts")
    summary = []
    for n in (200, 1000):
        group = [row for row in rows if row["sample_size"] == n]
        result = {"method": "gpc", "sample_size": n, "complete_seeds": len(group),
                  "planned_seeds": 10, "seed_ids": ";".join(str(row["seed"]) for row in group)}
        for name in (*METRICS, "numerical_seconds"):
            samples = [float(row[name]) for row in group]
            result[name + "_mean"] = statistics.mean(samples)
            result[name + "_se"] = statistics.stdev(samples) / len(samples) ** 0.5
        summary.append(result)
    _csv(root / "per_seed.csv", rows)
    _csv(root / "summary.csv", summary)
    write_json(root / "provenance.json", {
        "manifest_sha256": sha256(root / "manifest.json"),
        "per_seed_sha256": sha256(root / "per_seed.csv"),
        "summary_sha256": sha256(root / "summary.csv"),
        "prediction_sha256": {str(i): sha256(root / "predictions" / f"{i}.json")
                              for i in manifest["conditions"]},
        "completion": {"n200": 10, "n1000": 6},
        "partial_n1000": True,
    })
    print(root / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "replay", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--condition", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
    elif args.stage == "replay":
        if args.condition is None:
            parser.error("replay requires --condition")
        replay(root, args.condition)
    else:
        report(root)


if __name__ == "__main__":
    main()
