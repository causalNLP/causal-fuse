"""Extract and audit the Figure 3 Qwen endpoints for Table 2, without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/20260918T153548Z/neuropathic_confidence_cap"
FIGURE = ROOT / "results/20260919T005552Z/neuropathic_performance_cost"
POOLS = (("group_i", "local"), ("group_ii", "ancestry"), ("all", "all"))
STRATEGIES = ("stated", "consistency")
METRICS = ("expected_distance", "orientation_f1")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(stamp):
    if not re.fullmatch(r"\d{8}T\d{6}Z", stamp):
        raise ValueError("Use a UTC timestamp for --run-id")
    out = ROOT / "results" / stamp / "table2_qwen_cap"
    if out.exists():
        raise FileExistsError(out)
    completion = read(SOURCE / "completion.json")
    assert completion["complete"] and completion["audit_passed"]
    assert completion["trajectories"] == 120 and completion["checkpoints"] == 480
    assert sha(SOURCE / "manifest.json") == completion["manifest_sha256"]
    assert sha(SOURCE / "audit.json") == completion["audit_sha256"]
    protocol = read(FIGURE / "protocol.json")
    assert protocol["source_manifest_sha256"] == completion["manifest_sha256"]
    assert protocol["answer_confidence_cap"] == .8
    sources = {}
    for base, names in ((SOURCE, ("completion.json", "manifest.json", "audit.json", "per_run.csv", "summary.csv")),
                        (FIGURE, ("protocol.json", "summary.csv", "per_run.csv"))):
        for name in names:
            path = base / name
            sources[str(path.relative_to(ROOT))] = sha(path)
            if base == SOURCE and name in completion["artifacts"]:
                assert sha(path) == completion["artifacts"][name]
    metadata = SOURCE / "per_run.csv.metrics.json"
    scale = read(metadata)["metric_scale"] if metadata.exists() else "normalized"
    assert scale in ("normalized", "unnormalized")
    factor = 190 if scale == "normalized" else 1
    selected = [r for r in csv_rows(SOURCE / "per_run.csv")
                if int(r["budget"]) == 100 and float(r["evidence_cap"]) == .8]
    assert len(selected) == 120
    original = csv_rows(SOURCE / "summary.csv")
    figure_rows = csv_rows(FIGURE / "summary.csv")
    summary = []
    for strategy in STRATEGIES:
        for pool, label in POOLS:
            for n in (200, 1000):
                key = dict(strategy=strategy, group=pool, sample_size=str(n))
                matches = lambda r: all(r[k] == v for k, v in key.items())
                members = [r for r in selected if matches(r)]
                assert sorted(int(r["seed"]) for r in members) == list(range(10))
                assert all(int(r["attempted"]) == 100 for r in members)
                archived = next(r for r in original if matches(r) and r["budget"] == "100"
                                and float(r["evidence_cap"]) == .8)
                figure = next(r for r in figure_rows if matches(r) and r["budget"] == "100")
                row = dict(strategy=strategy, group=pool, pool_label=label, sample_size=n, seeds=10)
                for metric in METRICS:
                    multiplier = factor if metric == "expected_distance" else 1
                    values = [float(r[metric]) * multiplier for r in members]
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_se"] = statistics.stdev(values) / math.sqrt(10)
                    for suffix in ("_mean", "_se"):
                        assert math.isclose(row[metric + suffix], float(archived[metric + suffix]) * multiplier, abs_tol=1e-12)
                for suffix in ("_mean", "_se"):
                    assert math.isclose(row["expected_distance" + suffix], float(figure["expected_distance" + suffix]), abs_tol=1e-10)
                row["attempted"] = sum(int(r["attempted"]) for r in members)
                row["valid_count"] = sum(int(r["valid_count"]) for r in members)
                row["coverage"] = row["valid_count"] / row["attempted"]
                for metric in ("accuracy", "raw_brier", "effective_brier"):
                    row[metric] = sum(float(r[metric]) * int(r["valid_count"]) for r in members) / row["valid_count"]
                for metric in ("coverage", "accuracy", "raw_brier", "effective_brier"):
                    assert math.isclose(row[metric], float(archived[metric]), abs_tol=1e-12)
                for metric in ("input_tokens", "output_tokens", "total_tokens"):
                    row[metric + "_mean"] = float(figure[metric + "_mean"])
                summary.append(row)
    best = {(n, m): (min if m == "expected_distance" else max)(r[m + "_mean"] for r in summary if r["sample_size"] == n)
            for n in (200, 1000) for m in METRICS}
    panel_a = []
    for strategy in STRATEGIES:
        for pool, label in POOLS:
            cells = []
            for n in (200, 1000):
                row = next(r for r in summary if (r["strategy"], r["group"], r["sample_size"]) == (strategy, pool, n))
                for m in METRICS:
                    precision = 2 if m == "expected_distance" else 3
                    cell = f"{row[m + '_mean']:.{precision}f} ({row[m + '_se']:.{precision}f})"
                    if row[m + "_mean"] == best[n, m]:
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
            panel_a.append(f"SMC-Active ({strategy}, {label}) & " + " & ".join(cells) + r" \\")
    diagnostics = [r for r in summary if r["sample_size"] == 1000]
    winners = {m: (min if m == "raw_brier" else max)(r[m] for r in diagnostics)
               for m in ("coverage", "accuracy", "raw_brier")}
    panel_b = []
    for i, row in enumerate(diagnostics):
        cells = []
        for m in winners:
            cell = f"{row[m]:.4f}"
            if math.isclose(row[m], winners[m], abs_tol=1e-12):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        panel_b.append(("Qwen3.5-9B" if i == 0 else "") + " & " + row["strategy"].capitalize() +
                       f" ({row['pool_label']}) & " + " & ".join(cells) + r" \\")
    out.mkdir(parents=True)
    write_csv(out / "summary.csv", summary)
    endpoints = [{**r, "expected_distance": float(r["expected_distance"]) * factor} for r in selected]
    write_csv(out / "per_run.csv", endpoints)
    (out / "panel_a_rows.tex").write_text("\n".join(panel_a) + "\n")
    (out / "panel_b_rows.tex").write_text("\n".join(panel_b) + "\n")
    for name in ("per_run", "summary"):
        (out / f"{name}.csv.metrics.json").write_text(json.dumps(dict(metric_scale="unnormalized", node_count=20)) + "\n")
    (out / "provenance.json").write_text(json.dumps(dict(complete=True, source_hashes=sources,
        script_sha256=sha(Path(__file__)), trajectories=120, budget=100, confidence_cap=.8,
        raw_confidence_for_brier=True, source_metric_scale=scale, output_metric_scale="unnormalized",
        figure_pdf_sha256=sha(ROOT / "paper/figs/neuropathic_performance_cost.pdf"),
        new_model_calls=0, new_inference_runs=0,
        checks=["source hashes", "paired seeds", "attempted queries", "ED/F1 means and SEs",
                "Figure 3 ED endpoints", "pooled calibration"],
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}), indent=2) + "\n")
    print(out)
    print("\n".join(panel_a + panel_b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args().run_id)
