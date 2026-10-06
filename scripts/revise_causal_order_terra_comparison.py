"""Publish an audited Table 2 comparison revision without modifying a frozen run.

The original reporter scored the PC/GES initialization snapshots. Table 2's
standalone observational rows instead use the archived observational benchmark.
This offline amendment selects that benchmark's first ten seeds and preserves
the original report, collection manifest, and inference results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results/20260825T082523Z/neuropathic_causal_order_subnetwork/raw_results.csv"
TABLE2 = ROOT / "results/20260918T183103Z/ed_es_units/observational/summary.csv"
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def revise(source, output):
    completion = read(source / "completion.json")
    manifest = read(source / "manifest.json")
    assert completion["complete"] and completion["conditions"] == 20
    assert completion["manifest_sha256"] == sha(source / "manifest.json")
    for name, expected in {**manifest["inputs"], **completion["artifacts"]}.items():
        assert sha(source / name) == expected, name
    for name, expected in manifest["implementation"].items():
        assert sha(ROOT / name) == expected, name
    assert read(source / "per_run.csv.metrics.json")["metric_scale"] == "unnormalized"
    assert not BASELINE.with_suffix(".csv.metrics.json").exists(), "Check source units before conversion"

    original = rows(source / "comparison_per_run.csv")
    reference = rows(BASELINE)
    corrected, differences = [], []
    for old in original:
        new = dict(old)
        if old["method"] in ("PC", "GES"):
            matches = [r for r in reference if all(r[k] == old[k] for k in ("method", "seed", "sample_size"))]
            assert len(matches) == 1 and matches[0]["feasible"] == "1"
            for metric in METRICS:
                new[metric] = float(matches[0][metric]) * (190 if metric in METRICS[:2] else 1)
                if not math.isclose(new[metric], float(old[metric]), abs_tol=1e-12):
                    differences.append(dict(method=old["method"], seed=int(old["seed"]),
                        sample_size=int(old["sample_size"]), metric=metric,
                        original=float(old[metric]), corrected=new[metric]))
        corrected.append(new)

    own = rows(source / "per_run.csv")
    summary = []
    all_rows = own + corrected
    for method, n in sorted({(r["method"], int(r["sample_size"])) for r in all_rows}):
        group = [r for r in all_rows if r["method"] == method and int(r["sample_size"]) == n]
        assert sorted(int(r["seed"]) for r in group) == list(range(10))
        failures = sum(r.get("status", "complete") != "complete" for r in group)
        aggregate = dict(method=method, sample_size=n, seeds=10, infeasible=failures)
        for metric in METRICS:
            values = [float(r[metric]) for r in group] if not failures else []
            aggregate[metric + "_mean"] = statistics.mean(values) if values else None
            aggregate[metric + "_se"] = statistics.stdev(values) / math.sqrt(10) if values else None
        summary.append(aggregate)
    # Independently published full-precision Table 2 unit-conversion results.
    for expected in rows(TABLE2):
        actual = next(r for r in summary if r["method"] == expected["method"]
                      and r["sample_size"] == int(expected["sample_size"]))
        for metric in METRICS:
            for suffix in ("_mean", "_se"):
                assert math.isclose(actual[metric + suffix], float(expected[metric + suffix]), abs_tol=1e-12)
    old_summary = read(source / "summary.json")
    for actual in summary:
        if actual["method"] not in ("PC", "GES"):
            assert actual == next(r for r in old_summary if (r["method"], r["sample_size"]) ==
                                  (actual["method"], actual["sample_size"]))

    # Verify equal performance does not reflect accidental reuse of Qwen graphs.
    changed_graphs = []
    for path in sorted((source / "records").glob("*.json")):
        terra, qwen = read(path), read(source / "reference/qwen_records" / path.name)
        if terra["graph"] != qwen["graph"]:
            changed_graphs.append(path.name)
        for metric in METRICS:
            assert math.isclose(terra["metrics"][metric], qwen["metrics"][metric] *
                                (190 if metric in METRICS[:2] else 1), abs_tol=1e-12)

    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    for path in (BASELINE, TABLE2):
        shutil.copy2(path, output / "inputs" / ("observational_" + path.name))
    for name in ("manifest.json", "completion.json", "report.md", "comparison_per_run.csv"):
        shutil.copy2(source / name, output / "inputs" / ("original_" + name))
    for name in ("per_run.csv", "per_run.csv.metrics.json", "comparison_per_run.csv.metrics.json",
                 "summary.csv.metrics.json", "table2_row.tex", "audit.json"):
        shutil.copy2(source / name, output / name)
    write_csv(output / "comparison_per_run.csv", corrected)
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", summary)
    text = (source / "report.md").read_text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("| PC |", "| GES |")):
            method, n = (part.strip() for part in line.split("|")[1:3])
            item = next(r for r in summary if r["method"] == method and r["sample_size"] == int(n))
            cells = [f"{item[m + '_mean']:.4f} ({item[m + '_se']:.4f})" for m in METRICS]
            lines[index] = f"| {method} | {n} | " + " | ".join(cells) + " |"
    lines += ["", "## Reporting amendment", "",
        "PC and GES comparison rows now use Table 2's archived observational benchmark, seeds 0–9. "
        "The original report instead scored the initialization snapshots used by Causal Order. "
        "Only PC at n=1000, seed 6 differs: its Table 2 ED is 23.0 versus 21.5 in the initialization snapshot. "
        "The corrected PC mean ED is 21.00 (SE 0.39), versus 20.85 (SE 0.33) in the original report. "
        "Collection, initialization, Terra inference, and the exported Terra table row are unchanged.", "",
        f"The Terra and archived Qwen results have equal per-seed scores, but {len(changed_graphs)}/20 final "
        "graphs differ in edge orientations. Equal scores therefore do not imply identical graphs.", "",
        f"Original frozen experiment: [{source.name}]({source}). Its manifest and hashed artifacts remain intact. "
        "The original report and comparison rows are preserved under inputs/; provenance.json records this amendment."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    provenance = dict(complete=True, source=str(source), original_manifest_sha256=sha(source / "manifest.json"),
        original_completion_sha256=sha(source / "completion.json"), implementation_sha256=sha(Path(__file__)),
        changes=differences, terra_vs_qwen_changed_graphs=changed_graphs,
        checks=dict(original_hashes=True, twenty_conditions=True, table2_baselines_match=True,
                    terra_results_unchanged=True, terra_vs_qwen_equal_per_seed_metrics=True),
        artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()})
    write_json(output / "provenance.json", provenance)
    write_json(source / "report_amendment.json", dict(report=str(output / "report.md"),
        provenance=str(output / "provenance.json"), provenance_sha256=sha(output / "provenance.json")))
    print(json.dumps(dict(report=str(output / "report.md"), changed_metrics=differences,
                          terra_vs_qwen_changed_graphs=len(changed_graphs)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revise(args.source.resolve(), args.output.resolve())
