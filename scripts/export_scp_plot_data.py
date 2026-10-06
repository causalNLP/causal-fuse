"""Export saved SCP ED, token, and estimated-runtime summaries for two datasets."""

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from cd_llm.core.finite_support import (  # noqa: E402
    PartialGraph,
    graph_from_edges,
    point_prediction_metrics,
)


NEUROPATHIC_SUMMARY = (
    ROOT / "results/20260921T220659Z/neuropathic_performance_cost/summary.csv"
)
ALARM_BASELINES = ROOT / "results/20260921T013135Z/alarm_matched_baselines"
ALARM_PROVENANCE = (
    ROOT / "results/20260923T143202Z/alarm_performance_cost/provenance.json"
)
FIELDS = (
    "dataset",
    "method",
    "sample_size",
    "seeds",
    "expected_distance_mean",
    "expected_distance_se",
    "input_tokens_mean",
    "input_tokens_se",
    "output_tokens_mean",
    "output_tokens_se",
    "total_tokens_mean",
    "total_tokens_se",
    "response_events_mean",
    "response_events_se",
    "approx_runtime_seconds_mean",
    "approx_runtime_seconds_se",
)


def mean_se(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    assert len(values) == 10 and np.isfinite(values).all()
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(10))


def neuropathic_rows():
    with NEUROPATHIC_SUMMARY.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] == "SCP Pattern 2"]
    assert len(rows) == 2 and {int(row["sample_size"]) for row in rows} == {200, 1000}
    result = []
    for row in sorted(rows, key=lambda item: int(item["sample_size"])):
        assert row["strategy"] == "native" and row["group"] == "none"
        assert int(row["budget"]) == 0 and int(row["seeds"]) == 10
        result.append({"dataset": "neuropathic_pain", "method": row["method"],
                       "sample_size": row["sample_size"], "seeds": row["seeds"],
                       **{field: row[field] for field in FIELDS[4:]}})
    return result


def alarm_rows():
    provenance = json.loads(ALARM_PROVENANCE.read_text())
    assert provenance["baseline_source"] == str(ALARM_BASELINES.relative_to(ROOT))
    assert provenance["runtime_definition"] == (
        "saved numerical seconds plus ALARM-local fitted serial LLM seconds"
    )
    fit = provenance["latency_fit"]
    truth_record = json.loads((ALARM_BASELINES / "evaluation/truth_graph.json").read_text())
    truth = graph_from_edges(37, directed=truth_record["graph"]["directed"])

    database = ALARM_BASELINES / "cache.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        usage = defaultdict(lambda: [0, 0, 0])
        for unit, input_tokens, output_tokens in connection.execute(
            "SELECT u.unit, r.input, r.output FROM uses u "
            "JOIN responses r ON u.key = r.key WHERE u.unit LIKE 'scp:%'"
        ):
            condition = int(unit.split(":", 2)[1])
            assert 0 <= condition < 20
            assert input_tokens is not None and output_tokens is not None
            usage[condition][0] += 1
            usage[condition][1] += input_tokens
            usage[condition][2] += output_tokens
        assert set(usage) == set(range(20))

        per_seed = []
        for condition in range(20):
            item = connection.execute(
                "SELECT value FROM items WHERE kind='prediction' AND key=?",
                (f"scp:{condition}",),
            ).fetchone()
            assert item is not None
            prediction = json.loads(item[0])
            assert prediction["status"] == "complete"
            graph = PartialGraph.from_matrix(np.asarray(prediction["matrix"]), "cpdag")
            expected_distance = point_prediction_metrics(
                graph, truth, normalized=False
            )["expected_distance"]
            requests, input_tokens, output_tokens = usage[condition]
            assert requests >= 7992
            estimated_runtime = (
                fit["alpha_seconds_per_request"] * requests
                + fit["beta_input_seconds_per_token"] * input_tokens
                + fit["beta_output_seconds_per_token"] * output_tokens
            )
            per_seed.append({
                "sample_size": (200, 1000)[condition % 2],
                "seed": 1000 + condition // 2,
                "expected_distance": expected_distance,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "response_events": requests,
                "approx_runtime_seconds": estimated_runtime,
            })
    finally:
        connection.close()

    result = []
    for sample_size in (200, 1000):
        selected = [row for row in per_seed if row["sample_size"] == sample_size]
        assert len(selected) == 10
        assert {row["seed"] for row in selected} == set(range(1000, 1010))
        summary = {"dataset": "alarm", "method": "SCP Pattern 2",
                   "sample_size": sample_size, "seeds": 10}
        for metric in ("expected_distance", "input_tokens", "output_tokens",
                       "total_tokens", "response_events", "approx_runtime_seconds"):
            summary[metric + "_mean"], summary[metric + "_se"] = mean_se(selected, metric)
        # The independently published ALARM recovery table is a check on the
        # saved prediction identities and the unnormalized ED computation.
        expected_ed = {200: (27.65, 0.93), 1000: (12.20, 0.60)}[sample_size]
        assert round(summary["expected_distance_mean"], 2) == expected_ed[0]
        assert round(summary["expected_distance_se"], 2) == expected_ed[1]
        result.append(summary)
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    neuropathic = neuropathic_rows()
    alarm = alarm_rows()
    write_csv(args.output_dir / "neuropathic_pain_scp.csv", neuropathic)
    write_csv(args.output_dir / "alarm_scp.csv", alarm)
    print(json.dumps({"neuropathic_pain": neuropathic, "alarm": alarm}, indent=2), flush=True)


if __name__ == "__main__":
    main()
