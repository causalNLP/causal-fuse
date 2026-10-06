"""Offline report for the matched ALARM consistency extension."""
from __future__ import annotations

import csv
from collections import defaultdict
import math
import statistics

import numpy as np

from cd_llm.core.finite_support import stable_seed
from cd_llm.experiments import alarm_claim_pools_consistency as study
from cd_llm.experiments.alarm_claim_pools_local_report import score_record
from cd_llm.experiments.common.artifacts import (
    read_json, sha256, write_csv, write_json, write_text,
)
from cd_llm.metrics.units import metric_metadata


METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1", "total_tokens")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_size"], row["group"], row["budget"]].append(row)
    summary = []
    for (sample_size, pool, budget), members in sorted(groups.items()):
        if sorted(row["seed"] for row in members) != list(study.SEEDS):
            raise ValueError("incomplete consistency condition")
        output = dict(
            sample_size=sample_size, group=pool, strategy="consistency",
            budget=budget, seeds=10,
        )
        for metric in (*METRICS, "input_tokens", "output_tokens", "logical_requests"):
            values = [row[metric] for row in members]
            output[metric + "_mean"] = statistics.mean(values)
            output[metric + "_se"] = statistics.stdev(values) / math.sqrt(10)
        valid = sum(row["valid_count"] for row in members)
        attempted = sum(row["attempted"] for row in members)
        output.update(
            attempted=attempted, valid_count=valid,
            coverage=valid / attempted if attempted else None,
        )
        for metric in ("accuracy", "raw_brier", "effective_brier"):
            output[metric] = (
                sum(row[metric] * row["valid_count"] for row in members if row["valid_count"]) / valid
                if valid else None
            )
        summary.append(output)
    if len(summary) != 30:
        raise ValueError("30 consistency summary rows are required")
    return summary


def _read_stated_rows():
    rows = []
    integer = {"sample_size", "seed", "budget", "attempted", "answered", "abstentions",
               "valid_count", "input_tokens", "output_tokens", "logical_requests"}
    text = {"group", "strategy"}
    with (study.SOURCE / "per_run.csv").open(newline="") as stream:
        for source in csv.DictReader(stream):
            row = {}
            for key, value in source.items():
                if key in text:
                    row[key] = value
                elif value == "":
                    row[key] = None
                elif key in integer:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    if len(rows) != 300 or {row["strategy"] for row in rows} != {"stated"}:
        raise ValueError("completed stated-confidence source differs")
    return rows


def _bootstrap(values, identity):
    array = np.asarray(values, dtype=float)
    if array.shape != (10,) or not np.isfinite(array).all():
        raise ValueError("ten finite paired differences are required")
    generator = np.random.default_rng(stable_seed(study.STUDY, "bootstrap", identity))
    means = array[generator.integers(0, 10, size=(10000, 10))].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return dict(
        mean=float(array.mean()), standard_error=float(array.std(ddof=1) / math.sqrt(10)),
        low=float(low), high=float(high), bootstrap_resamples=10000,
    )


def paired_differences(stated_rows, consistency_rows):
    stated = {(r["sample_size"], r["group"], r["budget"], r["seed"]): r for r in stated_rows}
    consistency = {(r["sample_size"], r["group"], r["budget"], r["seed"]): r
                   for r in consistency_rows}
    output = []
    for sample_size in (200, 1000):
        for group in study.GROUPS:
            for budget in study.BUDGETS:
                differences = [
                    consistency[sample_size, group, budget, seed]["expected_distance"]
                    - stated[sample_size, group, budget, seed]["expected_distance"]
                    for seed in study.SEEDS
                ]
                output.append(dict(
                    comparison="consistency-minus-stated", sample_size=sample_size,
                    group=group, budget=budget, paired_seeds=10,
                    **_bootstrap(differences, f"{sample_size}:{group}:{budget}"),
                ))
    if len(output) != 24:
        raise ValueError("24 paired checkpoint comparisons are required")
    return output


def report(root):
    manifest = study.verify(root)
    gate = study.require_cache(root)
    study.require_inference_pilot(root)
    answer = study.provider(root, manifest)
    truth = study.graph(read_json(root / "evaluation/truth_graph.json")["graph"])
    raw = read_json(root / "cache/raw_responses.json")
    rows, events, ledger, hashes = [], [], [], {}
    with study.offline_guard():
        for task in study.tasks():
            path = root / "records" / f'{task["id"]}.json'
            record = read_json(path)
            study.validate_record(root, record, task, answer)
            hashes[task["id"]] = sha256(path)
            for budget in (0, *study.BUDGETS):
                metrics, diagnostics = score_record(record, truth, budget)
                rows.append(dict(
                    sample_size=task["sample_size"], group=task["group"],
                    strategy="consistency", seed=task["seed"], budget=budget, **metrics,
                ))
                if budget == 100:
                    events.extend(dict(
                        task=task["id"], sample_size=task["sample_size"], group=task["group"],
                        seed=task["seed"], **event,
                    ) for event in diagnostics)
            for slot, step in enumerate(record["trace"], 1):
                usage = [raw[key]["record"] for key in step["response_audit"]["request_keys"]]
                if len(usage) != 8:
                    raise ValueError("selected consistency query does not charge eight responses")
                if (sum(item["prompt_tokens"] for item in usage) != step["prompt_tokens"]
                        or sum(item["completion_tokens"] for item in usage) != step["completion_tokens"]):
                    raise ValueError("selected-query token expenditure differs")
                for variant, item in enumerate(usage):
                    ledger.append(dict(
                        task=task["id"], slot=slot, variant=variant,
                        request_key=item["request_key"], prompt_tokens=item["prompt_tokens"],
                        completion_tokens=item["completion_tokens"], repair=False,
                        aggregate_abstention=not step["valid_response"],
                    ))
    if len(rows) != 300 or len(events) != 6000 or len(ledger) != 48000:
        raise ValueError("expected 60 trajectories, 300 rows, 6000 selections, and 48000 charged responses")
    summary = aggregate(rows)
    stated_rows = _read_stated_rows()
    paired = paired_differences(stated_rows, rows)
    combined_rows = stated_rows + rows
    stated_summary = []
    with (study.SOURCE / "summary.csv").open(newline="") as stream:
        stated_summary.extend(csv.DictReader(stream))

    diagnostics = []
    for sample_size in (200, 1000):
        for group in study.GROUPS:
            for low, high in ((1, 10), (11, 20), (21, 50), (51, 100)):
                for kind in ("directed", "adjacency", "ancestry"):
                    selected = [event for event in events
                                if event["sample_size"] == sample_size and event["group"] == group
                                and event["kind"] == kind and low <= event["slot"] <= high]
                    valid = [event for event in selected if event["valid"]]
                    diagnostics.append(dict(
                        sample_size=sample_size, group=group, first_attempt=low,
                        last_attempt=high, kind=kind, attempted=len(selected), valid=len(valid),
                        coverage=len(valid) / len(selected) if selected else None,
                        accuracy=sum(event["correct"] for event in valid) / len(valid) if valid else None,
                    ))

    outputs = dict(
        per_run=rows, summary=summary, paired_strategy_ed=paired,
        selected_claims=events, claim_diagnostics=diagnostics,
        response_ledger=ledger, combined_per_run=combined_rows,
    )
    for name, values in outputs.items():
        write_csv(root / f"{name}.csv", values)
        write_json(root / f"{name}.csv.metrics.json", metric_metadata(37))
    write_csv(root / "combined_summary.csv", stated_summary + summary)
    write_json(root / "combined_summary.csv.metrics.json", metric_metadata(37))
    write_json(root / "summary.json", dict(
        consistency=summary, paired_strategy_ed=paired,
        stated_source=str(study.SOURCE), stated_manifest_sha256=manifest["stated_manifest_sha256"],
    ))

    index_c = {(row["sample_size"], row["group"], row["budget"]): row for row in summary}
    source_summary = list(csv.DictReader((study.SOURCE / "summary.csv").open(newline="")))
    index_s = {(int(row["sample_size"]), row["group"], int(row["budget"])): row
               for row in source_summary}
    lines = [
        "# Matched ALARM stated-versus-consistency study", "",
        "Qwen3.5-9B with answer confidence capped at 0.8. Consistency uses eight variants and "
        "abstains below six valid YES/NO responses. An aggregate abstention consumes one attempted "
        "query and all eight responses' tokens without changing the posterior. ED and ES are unnormalized.", "",
        "| n | Pool | Strategy | Attempts | ED (SE) | Coverage | Tokens (SE) |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for sample_size in (200, 1000):
        for group in study.GROUPS:
            left = index_s[sample_size, group, 100]
            right = index_c[sample_size, group, 100]
            lines.append(
                f"| {sample_size} | {group} | stated | 100 | "
                f"{float(left['expected_distance_mean']):.3f} ({float(left['expected_distance_se']):.3f}) | "
                f"{float(left['coverage']):.3f} | {float(left['total_tokens_mean']):.1f} "
                f"({float(left['total_tokens_se']):.1f}) |")
            lines.append(
                f"| {sample_size} | {group} | consistency | 100 | "
                f"{right['expected_distance_mean']:.3f} ({right['expected_distance_se']:.3f}) | "
                f"{right['coverage']:.3f} | {right['total_tokens_mean']:.1f} "
                f"({right['total_tokens_se']:.1f}) |")
    lines += ["", "## Paired ED differences", "",
              "Differences are consistency minus stated; negative values favor consistency. "
              "Intervals are paired percentile intervals from 10,000 bootstrap resamples of the ten data seeds.", "",
              "| n | Pool | Attempts | Difference [95% interval] |", "|---:|---|---:|---:|"]
    for row in paired:
        lines.append(
            f"| {row['sample_size']} | {row['group']} | {row['budget']} | "
            f"{row['mean']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}] |")
    runtime_rows = [read_json(path) for path in sorted((root / "gpu_runtime").glob("*.json"))]
    write_json(root / "gpu_runtime.json", dict(
        jobs=runtime_rows,
        allocated_gpu_wall_seconds=sum(row["elapsed_seconds"] for row in runtime_rows),
        actual_api_spending_usd=0.0,
    ))
    lines += ["", f"Collection accounting: {gate['accounting']}",
              f"GPU allocation runtime: {sum(row['elapsed_seconds'] for row in runtime_rows):.1f} seconds "
              f"across {len(runtime_rows)} allocations.",
              "The completed stated-confidence artifacts were reused without rerunning their trajectories."]
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "audit.json", dict(
        passed=True, records=hashes, trajectories=60, checkpoints=240,
        selected_questions=6000, charged_variant_responses=48000,
        manifest_sha256=sha256(root / "manifest.json"), network_disabled_during_scoring=True,
    ))
    artifacts = [f"{name}.csv" for name in outputs]
    artifacts += [f"{name}.csv.metrics.json" for name in outputs]
    artifacts += ["combined_summary.csv", "combined_summary.csv.metrics.json", "summary.json",
                  "report.md", "audit.json", "cache_audit.json", "accounting.json", "gpu_runtime.json"]
    write_json(root / "completion.json", dict(
        complete=True, audit_passed=True, trajectories=60, checkpoints=240,
        aggregate_outcomes=study.EXPECTED_OUTCOMES, nominal_responses=study.EXPECTED_RESPONSES,
        manifest_sha256=sha256(root / "manifest.json"),
        artifacts={name: sha256(root / name) for name in artifacts},
    ))
    write_json(root / "status.json", dict(
        stage="complete", trajectories=60, checkpoints=240, accounting=gate["accounting"]))
