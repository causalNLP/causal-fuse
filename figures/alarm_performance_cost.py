"""Render verified ALARM checkpoints and completed baseline predictions offline."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator
import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
STATED_SOURCE = ROOT / "results/20260920T164354Z/alarm_claim_pools_local"
CONSISTENCY_SOURCE = ROOT / "results/20260922T142041Z/alarm_claim_pools_consistency"
BASELINES = ROOT / "results/20260921T013135Z/alarm_matched_baselines"
GPC_CPDAG = ROOT / "results/20260924T014323Z/alarm_gpc_cpdag"
POOLS = (("group_i", "Local", "#397ca4"),
         ("group_ii", "Ancestral", "#D9822B"),
         ("all", "All", "#117766"))
MARKERS = {10: "o", 20: "s", 50: "^", 100: "D"}
TOKEN_SIZE_EXPONENT = 0.65
CLAIM_FUSE_MARKER_ALPHA = 0.75
TOKEN_MARKER_MIN_AREA = 20.
TOKEN_MARKER_MAX_AREA = 500.
RUNTIME_LIMITS = (.2, 6.*3600.)
RUNTIME_TICKS = ((1., "1s"), (10., "10s"), (60., "1m"),
                 (600., "10m"), (3600., "1h"))
BASELINE_GRAY = "#69737c"
ANNOTATION_GRAY = "#4b535b"
BASELINE_LABELS = {"bfs": "BFS+Pearson", "long": "PC+Long",
                   "order": "Causal\nOrder+PC", "gpc": "gPC"}
BASELINE_OFFSETS = {"bfs": (0, 10, "center", "bottom"),
                    "long": (5, 8, "left", "bottom"),
                    "order": (0, 18, "center", "bottom"),
                    "gpc": (-5, -12, "right", "top")}
BASELINE_TOKEN_OFFSETS = {"order": (0, 40, "center", "bottom")}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_completion(source):
    completion = json.loads((source / "completion.json").read_text())
    assert completion["complete"] and completion["audit_passed"]
    assert completion["trajectories"] == 60 and completion["checkpoints"] == 240
    assert sha256(source / "manifest.json") == completion["manifest_sha256"]
    for name, expected in completion["artifacts"].items():
        assert sha256(source / name) == expected, (source, name)
    return completion


def verified_rows():
    completions = {"stated": verified_completion(STATED_SOURCE),
                   "consistency": verified_completion(CONSISTENCY_SOURCE)}
    units = json.loads((CONSISTENCY_SOURCE / "combined_per_run.csv.metrics.json").read_text())
    assert units["metric_scale"] == "unnormalized" and units["node_count"] == 37
    with (CONSISTENCY_SOURCE / "combined_per_run.csv").open() as handle:
        per_seed = list(csv.DictReader(handle))
    with (CONSISTENCY_SOURCE / "combined_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    assert len(per_seed) == 600 and len(summary) == 60
    selected = []
    for strategy in ("stated", "consistency"):
        for n in (200, 1000):
            for group, _, _ in POOLS:
                for budget in MARKERS:
                    matches = [r for r in summary if int(r["sample_size"]) == n
                               and r["group"] == group and r["strategy"] == strategy
                               and int(r["budget"]) == budget]
                    assert len(matches) == 1
                    row = matches[0]
                    assert int(row["seeds"]) == 10
                    seeds = [r for r in per_seed if int(r["sample_size"]) == n
                             and r["group"] == group and r["strategy"] == strategy
                             and int(r["budget"]) == budget]
                    assert len(seeds) == 10 and {int(r["seed"]) for r in seeds} == set(range(1000, 1010))
                    assert all(int(r["attempted"]) == budget for r in seeds)
                    assert all(float(r["input_tokens"]) + float(r["output_tokens"])
                               == float(r["total_tokens"]) for r in seeds)
                    for metric in ("expected_distance", "total_tokens"):
                        values = np.array([float(r[metric]) for r in seeds])
                        assert np.isfinite(values).all()
                        assert np.isclose(values.mean(), float(row[metric + "_mean"]), rtol=1e-12)
                        assert np.isclose(values.std(ddof=1) / np.sqrt(10),
                                          float(row[metric + "_se"]), rtol=1e-12)
                    selected.append(row)
    for strategy in ("stated", "consistency"):
        for n in (200, 1000):
            at_50 = [row for row in selected if row["strategy"] == strategy
                     and int(row["sample_size"]) == n and int(row["budget"]) == 50]
            assert len(at_50) == 3
            best = min(at_50, key=lambda row: float(row["expected_distance_mean"]))
            assert best["group"] == "all"
    return selected, per_seed, completions


def mean_se(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def fit_local_latency_model(completions):
    """Fit one nonnegative latency model to all used stated and consistency requests."""
    records = {}
    ledger_hashes, raw_hashes = {}, {}
    for strategy, source in (("stated", STATED_SOURCE), ("consistency", CONSISTENCY_SOURCE)):
        completion = completions[strategy]
        cache_audit = json.loads((source / "cache_audit.json").read_text())
        assert cache_audit["passed"]
        assert sha256(source / "cache_audit.json") == completion["artifacts"]["cache_audit.json"]
        for name, expected in cache_audit["artifacts"].items():
            assert sha256(source / name) == expected, (source, name)
        raw_path = source / "cache/raw_responses.json"
        raw = json.loads(raw_path.read_text())
        ledger_path = source / "response_ledger.csv"
        with ledger_path.open() as stream:
            ledger = list(csv.DictReader(stream))
        used_keys = sorted({row["request_key"] for row in ledger})
        assert all(key in raw for key in used_keys)
        for key in used_keys:
            record = raw[key]["record"]
            if key in records:
                assert records[key] == record
            records[key] = record
        ledger_hashes[strategy] = sha256(ledger_path)
        raw_hashes[strategy] = sha256(raw_path)
    used_keys = sorted(records)
    inputs, outputs, latencies = [], [], []
    for key in used_keys:
        record = records[key]
        assert record["request_key"] == key
        assert record["prompt_tokens"] >= 0 and record["completion_tokens"] >= 0
        assert record["latency_seconds"] >= 0
        inputs.append(float(record["prompt_tokens"]))
        outputs.append(float(record["completion_tokens"]))
        latencies.append(float(record["latency_seconds"]))
    design = np.column_stack((np.ones(len(used_keys)), inputs, outputs))
    coefficients, residual_norm = nnls(design, np.array(latencies))
    fitted = design @ coefficients
    residuals = np.array(latencies) - fitted
    total_variation = np.sum((np.array(latencies) - np.mean(latencies)) ** 2)
    return dict(alpha_seconds_per_request=float(coefficients[0]),
                beta_input_seconds_per_token=float(coefficients[1]),
                beta_output_seconds_per_token=float(coefficients[2]),
                training_requests=len(used_keys), residual_norm=float(residual_norm),
                mae_seconds=float(np.mean(np.abs(residuals))),
                rmse_seconds=float(np.sqrt(np.mean(residuals ** 2))),
                r_squared=float(1. - np.sum(residuals ** 2) / total_variation),
                response_ledger_sha256=ledger_hashes,
                raw_responses_sha256=raw_hashes)


def estimated_llm_seconds(requests, input_tokens, output_tokens, timing):
    return (timing["alpha_seconds_per_request"] * float(requests)
            + timing["beta_input_seconds_per_token"] * float(input_tokens)
            + timing["beta_output_seconds_per_token"] * float(output_tokens))


def enrich_claim_fuse_rows(summary_rows, per_seed_rows, timing, completions):
    """Attach saved numerical time and fitted serial LLM time without changing results."""
    sources = {"stated": STATED_SOURCE, "consistency": CONSISTENCY_SOURCE}
    manifests = {strategy: json.loads((source / "manifest.json").read_text())
                 for strategy, source in sources.items()}
    audits = {strategy: json.loads((source / "audit.json").read_text())
              for strategy, source in sources.items()}
    for strategy, source in sources.items():
        assert audits[strategy]["passed"]
        assert audits[strategy]["manifest_sha256"] == sha256(source / "manifest.json")
        assert completions[strategy]["manifest_sha256"] == sha256(source / "manifest.json")
    enriched = []
    for source_row in per_seed_rows:
        row = dict(source_row)
        strategy = row["strategy"]
        source, manifest, audit = sources[strategy], manifests[strategy], audits[strategy]
        seed, n, budget = int(row["seed"]), int(row["sample_size"]), int(row["budget"])
        init_name = f"initializations/alarm_s{seed}_n{n}_grid_union.json"
        init_path = source / init_name
        assert sha256(init_path) == manifest["inputs"][init_name]
        init = json.loads(init_path.read_text())
        assert init["condition"]["id"] == f"alarm_s{seed}_n{n}"
        numerical_seconds = float(init["wall_seconds"])
        if budget:
            task = f"qwen_{strategy}_{row['group']}_s{seed}_n{n}_b100"
            record_path = source / "records" / f"{task}.json"
            assert sha256(record_path) == audit["records"][task]
            record = json.loads(record_path.read_text())
            assert record["task"]["id"] == task
            numerical_seconds += float(record["checkpoint_seconds"][str(budget)])
        llm_seconds = estimated_llm_seconds(row["logical_requests"], row["input_tokens"],
                                            row["output_tokens"], timing)
        row.update(method="Claim-Fuse", response_events=int(row["logical_requests"]),
                   numerical_seconds=numerical_seconds,
                   estimated_llm_seconds=llm_seconds,
                   approx_runtime_seconds=numerical_seconds + llm_seconds)
        enriched.append(row)
    by_key = defaultdict(list)
    for row in enriched:
        by_key[(int(row["sample_size"]), row["group"], row["strategy"],
                int(row["budget"]))].append(row)
    for summary_row in summary_rows:
        key = (int(summary_row["sample_size"]), summary_row["group"], summary_row["strategy"],
               int(summary_row["budget"]))
        seeds = by_key[key]
        assert len(seeds) == 10
        summary_row["method"] = "Claim-Fuse"
        for metric in ("response_events", "numerical_seconds", "estimated_llm_seconds",
                       "approx_runtime_seconds"):
            mean, se = mean_se(row[metric] for row in seeds)
            summary_row[metric + "_mean"] = mean
            summary_row[metric + "_se"] = se
    return enriched


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def baseline_snapshot():
    """Freeze selected records in one read transaction; never initialize a Store."""
    manifest = json.loads((BASELINES / "manifest.json").read_text())
    for name, expected in manifest["inputs"].items():
        assert sha256(BASELINES / name) == expected, name
    for name, expected in manifest["implementation"].items():
        if name == "code/src/cd_llm/experiments/alarm_matched_gpc.py":
            # The cached baseline predictions are frozen; this module now also
            # records separating sets for the separate CPDAG replay.
            assert expected == "ee86823d0df5ba236d4eb8be5a258e0469b77856b34c6afcbd8f049c869868bb"
        else:
            assert sha256(ROOT / name) == expected, name
    db = sqlite3.connect(f"file:{BASELINES / 'cache.sqlite3'}?mode=ro", uri=True, timeout=120)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")

        def item(kind, key):
            record = db.execute("SELECT value FROM items WHERE kind=? AND key=?", (kind, key)).fetchone()
            assert record is not None, (kind, key)
            return json.loads(record[0])

        predictions, audits = {}, {}
        for method in ("bfs", "long", "order"):
            for i in range(20):
                key = f"{method}:{i}"
                predictions[key] = item("prediction", key)
                audits[key] = item("method_audit", key)
                assert audits[key]["passed"], key
                record = predictions[key]
                assert record["status"] == "complete" or (
                    record["status"] == "infeasible" and record.get("scientific_infeasibility")), key
        ties, fallbacks = item("order_ties", "shared"), item("order_fallbacks", "shared")
        required_pairs = {tuple(pair) for pair in ties + fallbacks}
        units = {f"{m}:{i}" for m in ("bfs", "long") for i in range(20)}
        units.update(f"order:pair:{a}:{b}" for a, b in required_pairs)
        ledger = []
        for unit, ordinal, key, inp, out, request, response in db.execute(
            "SELECT u.unit,u.ordinal,u.key,r.input,r.output,r.request,r.response "
            "FROM uses u JOIN responses r ON u.key=r.key "
            "WHERE u.unit LIKE 'bfs:%' OR u.unit LIKE 'long:%' OR u.unit LIKE 'order:%' "
            "ORDER BY u.unit,u.ordinal"
        ):
            if unit not in units and not unit.startswith("order:triplet:"):
                continue
            assert inp is not None and out is not None and inp >= 0 and out >= 0
            ledger.append(dict(unit=unit, ordinal=ordinal, key=key, input_tokens=inp, output_tokens=out,
                               request_sha256=hashlib.sha256(request.encode()).hexdigest(),
                               response_sha256=hashlib.sha256(response.encode()).hexdigest()))
        triplets = {r["unit"] for r in ledger if r["unit"].startswith("order:triplet:")}
        assert len(triplets) == 7770
        assert all(any(r["unit"] == unit for r in ledger) for unit in units)
        correction = item("audit", "order_cycle_removal")
        assert correction["acyclic"]
    finally:
        db.close()
    return dict(source=str(BASELINES.relative_to(ROOT)), captured_utc=datetime.now(timezone.utc).isoformat(),
                manifest=manifest, manifest_sha256=sha256(BASELINES / "manifest.json"),
                predictions=predictions, audits=audits, required_order_pairs=sorted(required_pairs),
                order_cycle_removal=correction, ledger=ledger)


def baseline_rows(snapshot, timing):
    from cd_llm.core.finite_support import (
        PartialGraph, graph_from_edges, distribution_metrics, point_prediction_metrics,
    )
    truth_record = json.loads((BASELINES / "evaluation/truth_graph.json").read_text())
    truth = graph_from_edges(37, directed=truth_record["graph"]["directed"])
    labels = dict(bfs="BFS+Pearson", long="PC+Long-Srisk", order="Causal Order+PC")
    per_seed, summary = [], []
    for method in labels:
        for i in range(20):
            record = snapshot["predictions"][f"{method}:{i}"]
            usage = [r for r in snapshot["ledger"] if (
                r["unit"].startswith("order:") if method == "order" else r["unit"] == f"{method}:{i}")]
            row = dict(method=method, label=labels[method], seed=1000 + i // 2,
                       sample_size=(200, 1000)[i % 2], status=record["status"], reason=record.get("reason", ""),
                       logical_requests=len(usage), input_tokens=sum(r["input_tokens"] for r in usage),
                       output_tokens=sum(r["output_tokens"] for r in usage))
            row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
            row["response_events"] = row["logical_requests"]
            row["numerical_seconds"] = 0.
            row["estimated_llm_seconds"] = estimated_llm_seconds(
                row["logical_requests"], row["input_tokens"], row["output_tokens"], timing)
            row["approx_runtime_seconds"] = row["estimated_llm_seconds"]
            if record["status"] == "complete":
                if "distribution" in record:
                    graphs = [graph_from_edges(37, directed=r["edges"]) for r in record["distribution"]]
                    weights = np.array([r["weight"] for r in record["distribution"]])
                    metrics = distribution_metrics(graphs, weights, truth, "dag", normalized=False)
                else:
                    graph = PartialGraph.from_matrix(np.array(record["matrix"]), "cpdag")
                    metrics = point_prediction_metrics(graph, truth, normalized=False)
                row.update({key: metrics[key] for key in (
                    "expected_distance", "energy_score", "adjacency_f1", "orientation_f1")})
            per_seed.append(row)
        for n in (200, 1000):
            group = [r for r in per_seed if r["method"] == method and r["sample_size"] == n]
            assert len(group) == 10 and {r["seed"] for r in group} == set(range(1000, 1010))
            valid = sum(r["status"] == "complete" for r in group)
            row = dict(method=method, label=labels[method], sample_size=n, strategy="native", group="",
                       budget=0, seeds=10, valid_seeds=valid, accounted=10, plotted=valid == 10)
            for metric in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1",
                           "input_tokens", "output_tokens", "total_tokens", "response_events",
                           "numerical_seconds", "estimated_llm_seconds", "approx_runtime_seconds"):
                # Cost exists for every attempted condition, including infeasible outcomes.
                if valid < 10 and metric in ("expected_distance", "energy_score", "adjacency_f1",
                                             "orientation_f1"):
                    row[metric + "_mean"] = row[metric + "_se"] = None
                    continue
                values = np.array([r[metric] for r in group], dtype=float)
                assert np.isfinite(values).all()
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(10))
            summary.append(row)
    assert len(per_seed) == 60 and sum(r["status"] == "complete" for r in per_seed) == 60
    assert all(r["valid_seeds"] == 10 for r in summary)
    amendment = snapshot["manifest"].get("amendments", {}).get("order_cycle_guard", {})
    assert amendment.get("restored_conditions") == [3, 10, 16]
    assert amendment.get("state") == "complete"
    order_costs = {r["total_tokens"] for r in per_seed if r["method"] == "order"}
    assert len(order_costs) == 1 and next(iter(order_costs)) >= 4794159
    return per_seed, summary


def gpc_rows(timing, gpc_source):
    """Load verified CPDAG scores for the selected shared-skeleton gPC snapshot."""
    manifest_path = gpc_source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["study"] in {"alarm_shared_skeleton_gpc_cpdag_replay",
                                  "alarm_shared_skeleton_gpc_cpdag_interim"}
    assert manifest["source_manifest_sha256"] == sha256(BASELINES / "manifest.json")
    full_conditions = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
    interim_conditions = [0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 16, 18]
    assert manifest["conditions"] in (full_conditions, interim_conditions)
    assert manifest["planned_conditions"] == 20
    provenance = json.loads((gpc_source / "provenance.json").read_text())
    assert provenance["manifest_sha256"] == sha256(manifest_path)
    assert provenance["per_seed_sha256"] == sha256(gpc_source / "per_seed.csv")
    assert provenance["summary_sha256"] == sha256(gpc_source / "summary.csv")
    n1000_count = 3 if manifest["conditions"] == interim_conditions else 6
    assert provenance["completion"] == {"n200": 10, "n1000": n1000_count}
    usage = manifest["shared_usage"]
    assert usage == {"input_tokens": 4285, "logical_requests": 2,
                     "output_tokens": 4701, "total_tokens": 8986}
    with (gpc_source / "per_seed.csv").open() as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == len(manifest["conditions"])
    assert sorted(int(row["condition"]) for row in records) == manifest["conditions"]
    expected_seeds = {200: set(range(1000, 1010)),
                      1000: {1001, 1003, 1004} if n1000_count == 3
                      else {1001, 1003, 1004, 1005, 1006, 1007}}
    assert {n: {int(row["seed"]) for row in records if int(row["sample_size"]) == n}
            for n in expected_seeds} == expected_seeds
    llm_seconds = estimated_llm_seconds(usage["logical_requests"], usage["input_tokens"],
                                        usage["output_tokens"], timing)
    per_seed = []
    for record in records:
        numerical_seconds = float(record["numerical_seconds"])
        per_seed.append(dict(
            method="gpc", label="gPC", condition=int(record["condition"]),
            seed=int(record["seed"]), sample_size=int(record["sample_size"]), status="complete",
            expected_distance=float(record["expected_distance"]),
            energy_score=float(record["energy_score"]),
            adjacency_f1=float(record["adjacency_f1"]),
            orientation_f1=float(record["orientation_f1"]),
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"], response_events=usage["logical_requests"],
            numerical_seconds=numerical_seconds, estimated_llm_seconds=llm_seconds,
            approx_runtime_seconds=numerical_seconds + llm_seconds))
    summary = []
    for n, expected_count in ((200, 10), (1000, n1000_count)):
        group = [row for row in per_seed if row["sample_size"] == n]
        assert len(group) == expected_count
        row = dict(method="gpc", label="gPC", sample_size=n, strategy="native", group="",
                   budget=0, seeds=expected_count, valid_seeds=expected_count,
                   accounted=10, plotted=True)
        for metric in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1",
                       "input_tokens", "output_tokens", "total_tokens", "response_events",
                       "numerical_seconds", "estimated_llm_seconds", "approx_runtime_seconds"):
            mean, se = mean_se(item[metric] for item in group)
            row[metric + "_mean"] = mean
            row[metric + "_se"] = se
        summary.append(row)
    with (gpc_source / "summary.csv").open() as handle:
        expected = {int(row["sample_size"]): row for row in csv.DictReader(handle)}
    for row in summary:
        reference = expected[row["sample_size"]]
        assert int(reference["complete_seeds"]) == row["seeds"]
        for metric in ("expected_distance", "energy_score", "adjacency_f1",
                       "orientation_f1", "numerical_seconds"):
            for suffix in ("mean", "se"):
                key = metric + "_" + suffix
                assert np.isclose(row[key], float(reference[key]))
    return per_seed, summary, manifest


def token_to_area(tokens, tmin, tmax, smin=TOKEN_MARKER_MIN_AREA,
                  smax=TOKEN_MARKER_MAX_AREA, size_exponent=TOKEN_SIZE_EXPONENT):
    values = np.asarray(tokens, dtype=float)
    assert tmin > 0 and tmax > tmin and np.all(values > 0)
    transformed = (values / tmin) ** size_exponent
    transformed_max = (tmax / tmin) ** size_exponent
    result = smin + (transformed - 1.) / (transformed_max - 1.) * (smax - smin)
    return float(result) if result.ndim == 0 else result


def compact_tokens(tokens):
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.0f}k"
    return f"{tokens:.0f}"


def format_runtime_axis(ax):
    ticks = [(value, label) for value, label in RUNTIME_TICKS
             if RUNTIME_LIMITS[0] <= value <= RUNTIME_LIMITS[1]]
    ax.set_xscale("log")
    ax.set_xlim(RUNTIME_LIMITS)
    ax.xaxis.set_major_locator(FixedLocator([value for value, _ in ticks]))
    ax.set_xticklabels([label for _, label in ticks])
    ax.minorticks_off()


def plot_claim_point(ax, row, color, area):
    x, y = float(row["approx_runtime_seconds_mean"]), float(row["expected_distance_mean"])
    xerr, yerr = float(row["approx_runtime_seconds_se"]), float(row["expected_distance_se"])
    assert x - xerr > 0
    ax.scatter(x, y, s=area, marker=MARKERS[int(row["budget"])], color=color,
               edgecolor="white", linewidth=.7, alpha=CLAIM_FUSE_MARKER_ALPHA, zorder=3)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
                elinewidth=.75, capsize=1.8, capthick=.75, zorder=4)


def annotate_baseline(ax, row):
    method = row["method"]
    x, y = row["approx_runtime_seconds_mean"], row["expected_distance_mean"]
    dx, dy, ha, va = BASELINE_OFFSETS[method]
    ax.annotate(BASELINE_LABELS[method], xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va, fontsize=6.6,
                linespacing=.92, color=ANNOTATION_GRAY,
                arrowprops=dict(arrowstyle="-", lw=.55, color=ANNOTATION_GRAY,
                                shrinkA=1.5, shrinkB=4), zorder=5)
    if method in BASELINE_TOKEN_OFFSETS:
        dx, dy, ha, va = BASELINE_TOKEN_OFFSETS[method]
        ax.annotate(f"{compact_tokens(row['total_tokens_mean'])} tokens", xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=6.2, color=ANNOTATION_GRAY, zorder=5)


def plot_baseline(ax, row, area):
    x, y = row["approx_runtime_seconds_mean"], row["expected_distance_mean"]
    xerr, yerr = row["approx_runtime_seconds_se"], row["expected_distance_se"]
    ax.scatter(x, y, s=area, facecolors="none", edgecolors=BASELINE_GRAY,
               linewidths=.8, alpha=.55, zorder=1)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=BASELINE_GRAY,
                elinewidth=.75, capsize=1.8, capthick=.75, zorder=3)
    ax.scatter(x, y, s=18, marker="P", facecolor=BASELINE_GRAY,
               edgecolor="white", linewidth=.45, zorder=4)
    annotate_baseline(ax, row)


def add_legends(legend_ax, tmin, tmax):
    legend_ax.axis("off")
    claim_legend = legend_ax.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label)
                 for _, label, color in POOLS], title="Claim pool", loc="upper left",
        bbox_to_anchor=(0., 1.), frameon=False, fontsize=7.3, title_fontsize=8.2,
        handlelength=1.1, handletextpad=.55, labelspacing=.72, borderaxespad=0.)
    legend_ax.add_artist(claim_legend)
    query_handles = [Line2D([], [], color=ANNOTATION_GRAY, marker=marker, ls="none",
                            markersize=5, label=str(budget))
                     for budget, marker in MARKERS.items()]
    query_legend = legend_ax.legend(
        handles=query_handles, title="Attempted queries", ncol=2, loc="upper left",
        bbox_to_anchor=(0., .60), frameon=False, fontsize=7.3, title_fontsize=8.2,
        columnspacing=.85, handletextpad=.45, labelspacing=.72, borderaxespad=0.)
    legend_ax.add_artist(query_legend)
    token_levels = (2_000, 10_000, 100_000, 1_000_000)
    token_handles = [Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                            markeredgecolor=BASELINE_GRAY, markeredgewidth=.8,
                            markersize=math.sqrt(token_to_area(value, tmin, tmax)),
                            label=compact_tokens(value)) for value in token_levels]
    legend_ax.legend(handles=token_handles, title="Tokens", ncol=2, loc="lower left",
                     bbox_to_anchor=(0., .02), frameon=False, fontsize=7.3,
                     title_fontsize=8.2, handletextpad=.8, columnspacing=.65,
                     labelspacing=1.35, borderaxespad=0.)


def generate(run_id, gpc_source=GPC_CPDAG):
    rows, source_per_seed, completions = verified_rows()
    timing = fit_local_latency_model(completions)
    smc_per_seed = enrich_claim_fuse_rows(rows, source_per_seed, timing, completions)
    root = ROOT / "results" / run_id / "alarm_performance_cost"
    root.mkdir(parents=True, exist_ok=True)
    snapshot = baseline_snapshot()
    (root / "baseline_snapshot.json").write_text(json.dumps(snapshot, sort_keys=True) + "\n")
    per_seed, baselines = baseline_rows(snapshot, timing)
    gpc_per_seed, gpc_summary, gpc_manifest = gpc_rows(timing, gpc_source)
    per_seed.extend(gpc_per_seed)
    baselines.extend(gpc_summary)
    write_csv(root / "claim_fuse_per_seed.csv", smc_per_seed)
    write_csv(root / "claim_fuse_summary.csv", rows)
    write_csv(root / "baseline_per_seed.csv", per_seed)
    write_csv(root / "baseline_summary.csv", baselines)
    plotted = rows + [row for row in baselines if row["plotted"]]
    assert len(rows) == 48 and len(plotted) == 56
    assert all(float(row["approx_runtime_seconds_mean"]) > 0 for row in plotted)
    ylow = min(float(row["expected_distance_mean"]) - float(row["expected_distance_se"])
               for row in plotted)
    yhigh = max(float(row["expected_distance_mean"]) + float(row["expected_distance_se"])
                for row in plotted)
    assert 8.9 < ylow < 9.1
    ylim = (8., max(48., math.ceil(yhigh + 0.2)))
    tmin = min(float(row["total_tokens_mean"]) for row in plotted)
    tmax = max(float(row["total_tokens_mean"]) for row in plotted)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(7.6, 4.75))
    grid = fig.add_gridspec(2, 3, width_ratios=(1., 1., .65),
                            left=.085, right=.985, bottom=.14, top=.92,
                            wspace=.20, hspace=.17)
    axes = {}
    master = None
    for i in range(2):
        for j in range(2):
            axes[i, j] = fig.add_subplot(grid[i, j], sharex=master, sharey=master)
            if master is None:
                master = axes[i, j]
    legend_ax = fig.add_subplot(grid[:, 2])
    color_by_group = {group: color for group, _, color in POOLS}
    for i, n in enumerate((200, 1000)):
        for j, strategy in enumerate(("stated", "consistency")):
            ax = axes[i, j]
            claim_points = [row for row in rows if int(row["sample_size"]) == n
                            and row["strategy"] == strategy]
            assert len(claim_points) == 12
            claim_points.sort(key=lambda row: token_to_area(
                float(row["total_tokens_mean"]), tmin, tmax), reverse=True)
            for row in claim_points:
                area = token_to_area(float(row["total_tokens_mean"]), tmin, tmax)
                plot_claim_point(ax, row, color_by_group[row["group"]], area)
            for row in baselines:
                if row["sample_size"] == n and row["plotted"]:
                    area = token_to_area(float(row["total_tokens_mean"]), tmin, tmax)
                    plot_baseline(ax, row, area)
            format_runtime_axis(ax)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="y", color="#7f8790", alpha=.18, lw=.5, zorder=0)
            ax.tick_params(labelsize=7.5, length=3, labelbottom=i == 1, labelleft=j == 0)
            if j == 0:
                ax.set_ylabel(f"$n={n}$\nED", fontsize=8.5)
            if i == 0:
                title = "Stated Confidence" if strategy == "stated" else "Consistency (8 Responses)"
                ax.set_title(title, fontsize=9.2, pad=6)
    add_legends(legend_ax, tmin, tmax)
    fig.supxlabel("Runtime (log scale)", fontsize=9.2, x=.43, y=.035)
    output = root / "alarm_performance_cost.pdf"
    png_output = root / "alarm_performance_cost.png"
    fig.savefig(output, bbox_inches="tight", pad_inches=.035)
    fig.savefig(png_output, dpi=300, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)
    write_csv(root / "plotted_points.csv", plotted)
    provenance = dict(sources={"stated": str(STATED_SOURCE.relative_to(ROOT)),
                               "consistency": str(CONSISTENCY_SOURCE.relative_to(ROOT))},
                      model="Qwen3.5-9B", strategies=["stated", "consistency"],
                      confidence_cap=.8, baseline_confidence_cap=None, metric_scale="unnormalized",
                      points=len(plotted), claim_fuse_points=len(rows), baseline_points=8,
                      claim_fuse_seeds_per_point=10, trajectories=120, checkpoints=480,
                      includes_abstentions_and_repairs=True,
                      runtime_definition="saved numerical seconds plus ALARM-local fitted serial LLM seconds",
                      runtime_formula="t_num + alpha*N_requests + beta_in*T_in + beta_out*T_out",
                      baseline_runtime_note="LLM-time lower bound; per-dataset numerical timers unavailable",
                      latency_fit=timing,
                      source_manifest_sha256={strategy: completion["manifest_sha256"]
                                              for strategy, completion in completions.items()},
                      source_artifacts={strategy: completion["artifacts"]
                                        for strategy, completion in completions.items()},
                      script_sha256=sha256(Path(__file__)),
                      figure_sha256=sha256(output), png_sha256=sha256(png_output),
                      plotted_points_sha256=sha256(root / "plotted_points.csv"),
                      claim_fuse_per_seed_sha256=sha256(root / "claim_fuse_per_seed.csv"),
                      claim_fuse_summary_sha256=sha256(root / "claim_fuse_summary.csv"),
                      baseline_source=str(BASELINES.relative_to(ROOT)),
                      gpc_source=str(gpc_source.relative_to(ROOT)),
                      gpc_manifest_sha256=sha256(gpc_source / "manifest.json"),
                      gpc_per_seed_sha256=sha256(gpc_source / "per_seed.csv"),
                      gpc_completed_conditions=gpc_manifest["conditions"],
                      gpc_pending_conditions=[i for i in range(20) if i not in gpc_manifest["conditions"]],
                      gpc_complete_case_counts={"n200": 10, "n1000": 3 if len(gpc_manifest["conditions"]) == 13 else 6},
                      gpc_cost_note="shared expert usage charged in full to every dataset; saved search time plus measured CPDAG orientation time",
                      baseline_manifest_sha256=snapshot["manifest_sha256"],
                      baseline_snapshot_sha256=sha256(root / "baseline_snapshot.json"),
                      baseline_per_seed_sha256=sha256(root / "baseline_per_seed.csv"),
                      baseline_summary_sha256=sha256(root / "baseline_summary.csv"),
                      metric_code_sha256=sha256(ROOT / "code/src/cd_llm/core/finite_support.py"),
                      baselines="BFS+Pearson, PC+Long-Srisk, and Causal Order+PC: ten valid seeds per sample size",
                      order_cycle_fallback="edgewise Qwen3.5-9B cycle guard in conditions 3, 10, and 16",
                      shared_order_cost="full required collection charged per dataset; unused diagnostic excluded",
                      token_area_mapping=dict(size_exponent=TOKEN_SIZE_EXPONENT,
                                              min_area=TOKEN_MARKER_MIN_AREA,
                                              max_area=TOKEN_MARKER_MAX_AREA,
                                              global_token_min=tmin, global_token_max=tmax),
                      xlim=RUNTIME_LIMITS, ylim=ylim)
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    paper_pdf = ROOT / "paper/figs/alarm_performance_cost.pdf"
    paper_png = ROOT / "paper/figs/alarm_performance_cost.png"
    shutil.copyfile(output, paper_pdf)
    shutil.copyfile(png_output, paper_png)
    print(f"Verified 48 unchanged Claim-Fuse points and eight baseline points.\n"
          f"Latency fit: {timing}\n{root}\n{paper_pdf}\n{paper_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--gpc-source", type=Path, default=GPC_CPDAG)
    with patch("socket.socket.connect", side_effect=RuntimeError("Figure generation is offline")), \
         patch("socket.create_connection", side_effect=RuntimeError("Figure generation is offline")):
        args = parser.parse_args()
        generate(args.run_id, args.gpc_source.resolve())
