"""Audit and extract completed Terra 100-query, 0.8-cap endpoints for Table 2."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

from prepare_table2_qwen_cap import POOLS, STRATEGIES, csv_rows, read, sha, write_csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/20260920T031654Z/neuropathic_terra_smc"
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")


def main(stamp):
    if not re.fullmatch(r"\d{8}T\d{6}Z", stamp):
        raise ValueError("Use a UTC timestamp for --run-id")
    out = ROOT / "results" / stamp / "table2_terra_cap"
    if out.exists():
        raise FileExistsError(out)
    completion = read(SOURCE / "completion.json")
    assert completion["complete"] and completion["audit_passed"]
    assert completion["trajectories"] == 120 and completion["checkpoints"] == 480
    assert sha(SOURCE / "manifest.json") == completion["manifest_sha256"]
    assert sha(SOURCE / "audit.json") == completion["audit_sha256"]
    assert read(SOURCE / "manifest.json")["answer_confidence_cap"] == .8
    assert read(SOURCE / "per_run.csv.metrics.json")["metric_scale"] == "unnormalized"
    for name, digest in completion["artifacts"].items():
        assert sha(SOURCE / name) == digest, name
    selected = [r for r in csv_rows(SOURCE / "per_run.csv") if int(r["budget"]) == 100]
    assert len(selected) == 120
    archived = csv_rows(SOURCE / "summary.csv")
    summary = []
    for strategy in STRATEGIES:
        for pool, label in POOLS:
            for n in (200, 1000):
                def matches(r):
                    return (r["strategy"], r["group"], int(r["sample_size"]), int(r["budget"])) == (strategy, pool, n, 100)
                members = [r for r in selected if matches(r)]
                assert sorted(int(r["seed"]) for r in members) == list(range(10))
                assert all(int(r["attempted"]) == 100 for r in members)
                source_rows = [r for r in archived if matches(r)]
                assert len(source_rows) == 1
                source = source_rows[0]
                row = dict(strategy=strategy, group=pool, pool_label=label, sample_size=n, budget=100, seeds=10)
                for metric in (*METRICS, "total_tokens"):
                    values = [float(r[metric]) for r in members]
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_se"] = statistics.stdev(values) / math.sqrt(10)
                    for suffix in ("_mean", "_se"):
                        assert math.isclose(row[metric + suffix], float(source[metric + suffix]), abs_tol=1e-12)
                row["attempted"] = sum(int(r["attempted"]) for r in members)
                row["valid_count"] = sum(int(r["valid_count"]) for r in members)
                row["coverage"] = row["valid_count"] / row["attempted"]
                for metric in ("accuracy", "raw_brier", "effective_brier"):
                    row[metric] = sum(float(r[metric]) * int(r["valid_count"]) for r in members) / row["valid_count"]
                for metric in ("coverage", "accuracy", "raw_brier", "effective_brier"):
                    assert math.isclose(row[metric], float(source[metric]), abs_tol=1e-12)
                summary.append(row)
    displayed = ("expected_distance", "orientation_f1")
    best = {(n, m): (min if m == "expected_distance" else max)(r[m + "_mean"] for r in summary if r["sample_size"] == n)
            for n in (200, 1000) for m in displayed}
    panel_a = []
    for strategy in STRATEGIES:
        for pool, label in POOLS:
            cells = []
            for n in (200, 1000):
                row = next(r for r in summary if (r["strategy"], r["group"], r["sample_size"]) == (strategy, pool, n))
                for metric in displayed:
                    precision = 2 if metric == "expected_distance" else 3
                    cell = f"{row[metric + '_mean']:.{precision}f} ({row[metric + '_se']:.{precision}f})"
                    if row[metric + "_mean"] == best[n, metric]:
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
            panel_a.append(f"SMC-Active ({strategy}, {label}) & " + " & ".join(cells) + r" \\")
    diagnostics = [r for r in summary if r["sample_size"] == 1000]
    winners = {m: (min if m == "raw_brier" else max)(r[m] for r in diagnostics)
               for m in ("coverage", "accuracy", "raw_brier")}
    panel_b = []
    for i, row in enumerate(diagnostics):
        cells = []
        for metric, winner in winners.items():
            cell = f"{row[metric]:.4f}"
            if math.isclose(row[metric], winner, abs_tol=1e-12):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        panel_b.append(("GPT-5.6 Terra" if i == 0 else "") + " & " + row["strategy"].capitalize() +
                       f" ({row['pool_label']}) & " + " & ".join(cells) + r" \\")
    out.mkdir(parents=True)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_run.csv", selected)
    (out / "panel_a_rows.tex").write_text("\n".join(panel_a) + "\n")
    (out / "panel_b_rows.tex").write_text("\n".join(panel_b) + "\n")
    for name in ("per_run", "summary"):
        (out / f"{name}.csv.metrics.json").write_text((SOURCE / "per_run.csv.metrics.json").read_text())
    names = ("completion.json", "manifest.json", "audit.json", "per_run.csv", "summary.csv")
    provenance = dict(complete=True, source_hashes={str((SOURCE / name).relative_to(ROOT)): sha(SOURCE / name) for name in names},
        script_sha256=sha(Path(__file__)), trajectories=120, budget=100, confidence_cap=.8,
        raw_confidence_for_brier=True, output_metric_scale="unnormalized",
        figure_pdf_sha256=sha(ROOT / "paper/figs/neuropathic_performance_cost.pdf"),
        new_model_calls=0, new_inference_runs=0,
        checks=["completion artifact hashes", "paired seeds", "attempted queries", "structural means and SEs", "pooled calibration"],
        artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out)
    print("\n".join(panel_a + panel_b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args().run_id)
