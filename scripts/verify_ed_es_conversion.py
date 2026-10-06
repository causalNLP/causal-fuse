"""Audit converted reporting artifacts and saved posterior checkpoints."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cd_llm.core.finite_support import distribution_metrics, graph_from_edges
from cd_llm.metrics.units import convert_metrics, distance_factor


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    out = args.root
    workspace = Path(__file__).resolve().parents[2]
    sources = read(out / "provenance.json")
    for item in sources:
        assert sha(workspace / item["source"]) == item["source_sha256"], item["source"]
        assert sha(out / item["output"]) == item["output_sha256"], item["output"]
        if item["output"].endswith(".csv"):
            expected = convert_metrics(rows(workspace / item["source"]),
                                       distance_factor(item["node_count"], item["regime"]), source_scale="normalized")
            assert rows(out / item["output"]) == expected, item["output"]
    for item in read(out / "observational/provenance.json"):
        assert sha(workspace / item["source"]) == item["source_sha256"]

    source = workspace / "results/20260918T153548Z/neuropathic_confidence_cap"
    target = graph_from_edges(20, **read(source / "evaluation/truth_graph.json")["graph"])
    archived = rows(source / "per_run.csv")
    index = {(float(r["evidence_cap"]), int(r["budget"]), int(r["sample_size"]), r["strategy"], r["group"], int(r["seed"])): r for r in archived}
    checked = 0
    for cap, folder in ((.8, source / "records"), (.99, source / "baseline/records")):
        for path in sorted(folder.glob("*s00_*.json")):
            payload = read(path)
            task = payload["task"]
            for budget, records in payload["distributions"].items():
                graphs = [graph_from_edges(20, **r["graph"]) for r in records]
                weights = np.array([r["weight"] for r in records])
                current = distribution_metrics(graphs, weights, target, "dag")
                legacy = distribution_metrics(graphs, weights, target, "dag", normalized=True)
                reference = index[cap, int(budget), task["sample_size"], task["strategy"], task["group"], 0]
                assert math.isclose(current["expected_distance"], 190 * float(reference["expected_distance"]), abs_tol=1e-10)
                assert math.isclose(current["energy_score"], 190 * legacy["energy_score"], abs_tol=1e-10)
                for metric in ("adjacency_f1", "orientation_f1"):
                    assert current[metric] == float(reference[metric])
                checked += 1
    assert checked == 96
    report = dict(passed=True, archived_source_files_unchanged=len(sources),
                  saved_posterior_checkpoints_checked=checked, inference_reruns=0, api_calls=0)
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
