"""Matched offline answer-confidence cap ablation with unchanged acquisition."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import fcntl
from itertools import combinations
import math
import shutil

import numpy as np

from cd_llm.core.finite_support import stable_seed
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments import neuropathic_new_prompt_cache as cache
from cd_llm.experiments import neuropathic_new_prompt_smc as baseline
from cd_llm.experiments.common.artifacts import json_value, read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.metrics.units import convert_metrics, metric_metadata
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_active_interpretation_report import interval
from cd_llm.methods.active_query import ClaimQuery, dag_query_pool, query_values
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_confidence_cap"
SOURCE = WORKSPACE_ROOT / "results/20260918T003643Z/neuropathic_new_prompt_smc"
CAP = 0.8
BUDGETS = baseline.BUDGETS
CONTRAST_METRICS = (*original.METRICS, "accuracy", "raw_brier", "effective_brier", "raw_ece", "effective_ece", "coverage")


def cap_response(response, cap=CAP):
    if cap is None:
        return response
    if not 0.5 <= cap < 1:
        raise ValueError("invalid evidence cap")
    raw = response.reported_confidence
    return replace(response, reported_confidence=None if raw is None else min(raw, cap),
        audit=dict(response.audit, raw_reported_confidence=raw, answer_confidence_cap=cap),
        treatment="soft_evidence")


def providers(root, cap=CAP):
    indexed = {(r["strategy"], r["query"]["query_id"]): r for r in read_json(root / "responses.json")}
    def make(strategy):
        def provider(query):
            row = indexed[strategy, query.query_id]
            if row["query"] != asdict(query):
                raise ValueError("cached query identity differs")
            return cap_response(cache.response_for(row), cap)
        return provider
    return {s: make(s) for s in original.STRATEGIES}


def verify(root):
    manifest = read_json(root / "manifest.json")
    if manifest["tasks"] != original.tasks() or manifest["answer_confidence_cap"] != CAP or manifest["budgets"] != list(BUDGETS):
        raise ValueError("registered study configuration differs")
    if manifest["acquisition_confidence"] != .9 or manifest["engine_soft_confidence_cap"] != .99 or manifest["baseline_evidence_cap"] != .99:
        raise ValueError("acquisition or baseline cap differs")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    source = cache.verify(SOURCE)
    cache.require_cache(SOURCE)
    completed, audit = read_json(SOURCE / "completion.json"), read_json(SOURCE / "audit.json")
    if (not completed["complete"] or not completed["audit_passed"]
        or sha256(SOURCE / "manifest.json") != completed["manifest_sha256"]
        or sha256(SOURCE / "audit.json") != completed["audit_sha256"]):
        raise ValueError("baseline completion audit differs")
    for name, digest in completed["artifacts"].items():
        if sha256(SOURCE / name) != digest:
            raise ValueError(f"baseline artifact changed: {name}")
    inputs = {}
    def copy(name, target=None, expected=None):
        target = target or name
        expected = expected or sha256(SOURCE / name)
        if sha256(SOURCE / name) != expected:
            raise ValueError(f"baseline input changed: {name}")
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
        if sha256(destination) != expected:
            raise ValueError("copied input differs")
        inputs[target] = expected
    for name, digest in source["inputs"].items():
        if name.startswith(("inputs/", "initializations/", "evaluation/")) or name == "prompts.json":
            copy(name, expected=digest)
    gate = read_json(SOURCE / "cache_audit.json")
    copy("responses.json", expected=gate["artifacts"]["responses.json"])
    copy("cache_audit.json")
    for name in ("manifest.json", "audit.json", "completion.json", "summary.json"):
        copy(name, "baseline/" + name)
    for task in source["tasks"]:
        name = f"records/{task['id']}.json"
        copy(name, "baseline/" + name, audit["records"][task["id"]])
    names = set(source["implementation"])
    names.update(("code/src/cd_llm/experiments/neuropathic_confidence_cap.py",
                  "code/scripts/submit_neuropathic_confidence_cap.py",
                  "sbatch/neuropathic_confidence_cap.sbatch"))
    manifest = dict(study=STUDY, version=1, source=str(SOURCE),
        source_manifest_sha256=sha256(SOURCE / "manifest.json"), tasks=original.tasks(),
        budgets=list(BUDGETS), answer_confidence_cap=CAP, baseline_evidence_cap=0.99,
        acquisition_confidence=0.9, engine_soft_confidence_cap=0.99,
        configuration=dict(source["configuration"], answer_confidence_cap=CAP,
            selection_confidence=0.9, api_requests=0,
            cap_implementation="clip provider confidence before weighting and MH; preserve raw confidence in audit"),
        inputs=inputs, implementation={name: sha256(WORKSPACE_ROOT / name) for name in sorted(names)})
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", dict(stage="prepared", trajectories=120, checkpoints=480, new_api_requests=0))
    print(f"Prepared {root}: 120 capped trajectories and 480 checkpoints, no API requests.", flush=True)
    return manifest


def validate_record(root, record, task, manifest, experts=None):
    validate_core(record, task, sha256(root / "manifest.json"),
                  manifest["inputs"][f"initializations/{task['condition']}_grid_union.json"])
    baseline.validate_checkpoints(record)
    if record["cache_audit_sha256"] != manifest["inputs"]["cache_audit.json"]:
        raise ValueError("source cache audit differs")
    experts = providers(root) if experts is None else experts
    for step in record["trace"]:
        response = experts[task["strategy"]](ClaimQuery(**step["query"]))
        if (step["answer"] != response.answer or step["reported_confidence"] != response.reported_confidence
            or step["response_audit"] != json_value(response.audit)
            or (step["valid_response"] and step["effective_confidence"] > CAP)):
            raise ValueError("capped response differs from the audited cached answer")


def require_pilot(root):
    gate = read_json(root / "validation/pilot.json")
    if (not gate["passed"] or gate["exact_replays"] != 12 or gate["baseline_reproductions"] != 4
        or gate["manifest_sha256"] != sha256(root / "manifest.json")):
        raise ValueError("pilot validation not satisfied")
    for name, digest in gate["records"].items():
        if sha256(root / f"records/{name}.json") != digest:
            raise ValueError("validated pilot record changed")


def run_block(root, block):
    if block not in range(20):
        raise ValueError("block must be in 0..19")
    manifest = verify(root)
    if block >= 2:
        require_pilot(root)
    lock_path = root / f"locks/block_{block:02d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, original.offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experts = providers(root)
        for task in (t for t in manifest["tasks"] if t["block"] == block):
            path = root / f"records/{task['id']}.json"
            record = read_json(path) if path.exists() else baseline.execute(root, task, experts[task["strategy"]])
            validate_record(root, record, task, manifest, experts)
            if not path.exists():
                write_json(path, record)
            print(f"{task['id']}: cap 0.8, EIG 0.9, {record['elapsed_seconds']:.2f}s", flush=True)
        write_json(root / f"status/block_{block:02d}.json", dict(complete=True, trajectories=6, checkpoints=24,
            manifest_sha256=sha256(root / "manifest.json")))


def validate_pilot(root):
    manifest, hashes, controls = verify(root), {}, {}
    with original.offline_guard():
        experts, uncapped = providers(root), providers(root, cap=None)
        for task in (t for t in manifest["tasks"] if t["seed"] == 0):
            path = root / f"records/{task['id']}.json"
            record = read_json(path)
            validate_record(root, record, task, manifest, experts)
            replay = baseline.execute(root, task, experts[task["strategy"]])
            validate_record(root, replay, task, manifest, experts)
            baseline.assert_replay(replay, record)
            if replay["distributions"] != record["distributions"]:
                raise ValueError("capped checkpoint replay differs")
            hashes[task["id"]] = sha256(path)
            print(f"Capped pilot replay passed: {task['id']}", flush=True)
            if task["group"] == "all":
                reference = read_json(root / f"baseline/records/{task['id']}.json")
                control = baseline.execute(root, task, uncapped[task["strategy"]])
                baseline.assert_replay(control, reference)
                if control["distributions"] != reference["distributions"]:
                    raise ValueError("uncapped baseline checkpoint reproduction differs")
                controls[task["id"]] = sha256(root / f"baseline/records/{task['id']}.json")
                print(f"Baseline reproduction passed: {task['id']}", flush=True)
    gate = dict(passed=True, trajectories=12, checkpoints=48, exact_replays=12,
        baseline_reproductions=4, baseline_records=controls, records=hashes,
        manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "validation/pilot.json", gate)
    return gate


def calibration(raw, effective):
    a, b = confidence_metrics(raw), confidence_metrics(effective)
    return json_value(dict(valid_count=a["valid_count"], accuracy=a["accuracy"],
        raw_brier=a["brier"], raw_ece=a["ece"], effective_brier=b["brier"], effective_ece=b["ece"]))


def metrics_for(record, truth):
    metrics, _ = original.metrics_for(record, truth)
    raw, effective = [], []
    for step in record["trace"]:
        if step["valid_response"]:
            correct = int(step["answer"] == int(query_values((truth,), ClaimQuery(**step["query"]))[0]))
            confidence = step["response_audit"].get("raw_reported_confidence", step["reported_confidence"])
            raw.append((correct, confidence))
            effective.append((correct, step["effective_confidence"]))
    # Explicit names prevent clipped confidence from being mislabeled as model-reported calibration.
    metrics.pop("brier")
    metrics.pop("ece")
    return json_value(dict(metrics, **calibration(raw, effective))), raw, effective


def contrast(left, right, identity):
    result = []
    for metric in CONTRAST_METRICS:
        seeds = [s for s in sorted(set(left) & set(right)) if left[s][metric] is not None and right[s][metric] is not None
                 and math.isfinite(left[s][metric]) and math.isfinite(right[s][metric])]
        values = [left[s][metric] - right[s][metric] for s in seeds]
        estimate = (interval(values, stable_seed(STUDY, *identity.values(), metric)) if len(values) >= 2
                    else dict(mean=values[0] if values else None, low=None, high=None))
        result.append(dict(identity, metric=metric, paired_seeds=len(seeds), missing_paired_seeds=10-len(seeds), **estimate))
    return result


def comparisons(rows):
    index = defaultdict(dict)
    for r in rows:
        index[r["evidence_cap"], r["budget"], r["sample_size"], r["strategy"], r["group"]][r["seed"]] = r
    result = []
    pairs = [(s, b, s, a) for s in original.STRATEGIES for a, b in combinations(original.GROUPS, 2)]
    pairs += [("consistency", g, "stated", g) for g in original.GROUPS]
    for budget in BUDGETS:
        for n in (200, 1000):
            for cap in (CAP, 0.99):
                for ls, lg, rs, rg in pairs:
                    result.extend(contrast(index[cap, budget, n, ls, lg], index[cap, budget, n, rs, rg],
                        dict(comparison="within_cap", budget=budget, sample_size=n,
                            left_cap=cap, left_strategy=ls, left_group=lg,
                            right_cap=cap, right_strategy=rs, right_group=rg)))
            for s in original.STRATEGIES:
                for g in original.GROUPS:
                    result.extend(contrast(index[CAP, budget, n, s, g], index[0.99, budget, n, s, g],
                        dict(comparison="cap_08_minus_099", budget=budget, sample_size=n,
                            left_cap=CAP, left_strategy=s, left_group=g,
                            right_cap=0.99, right_strategy=s, right_group=g)))
    return result


def report(root):
    with original.offline_guard():
        return _report(root)


def _report(root):
    manifest = verify(root)
    require_pilot(root)
    truth, experts = graph(read_json(root / "evaluation/truth_graph.json")["graph"]), providers(root)
    rows, groups, pooled, hashes = [], defaultdict(list), defaultdict(lambda: ([], [])), {}
    for task in manifest["tasks"]:
        path = root / f"records/{task['id']}.json"
        record = read_json(path)
        validate_record(root, record, task, manifest, experts)
        hashes[task["id"]] = sha256(path)
        archived = read_json(root / f"baseline/records/{task['id']}.json")
        for cap, full in ((CAP, record), (0.99, archived)):
            for budget in BUDGETS:
                metrics, raw, effective = metrics_for(baseline.prefix_record(full, budget), truth)
                row = dict(evidence_cap=cap, budget=budget, sample_size=task["sample_size"],
                    strategy=task["strategy"], group=task["group"], seed=task["seed"], **metrics)
                rows.append(row)
                key = cap, budget, task["sample_size"], task["strategy"], task["group"]
                groups[key].append(row)
                pooled[key][0].extend(raw)
                pooled[key][1].extend(effective)
    summaries = []
    for key, members in sorted(groups.items()):
        if sorted(r["seed"] for r in members) != list(range(10)):
            raise ValueError("report requires ten unique seeds per cell")
        cap, budget, n, strategy, group = key
        row = dict(evidence_cap=cap, budget=budget, sample_size=n, strategy=strategy, group=group, seeds=10)
        for metric in original.METRICS:
            values = np.asarray([r[metric] for r in members])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(10))
        row.update({name: sum(r[name] for r in members) for name in original.COUNTS})
        row.update(calibration(*pooled[key]))
        row.update(coverage=row["answered"] / row["attempted"], elapsed_seconds=sum(r["elapsed_seconds"] for r in members))
        summaries.append(row)
    # Guard comparison against accidental changes in archived metric definitions.
    source_payload = read_json(root / "baseline/summary.json")
    source_rows = convert_metrics(source_payload["summaries"], 190,
                                  source_scale=source_payload.get("metric_scale", "normalized"))
    source_summary = {(r["budget"], r["sample_size"], r["strategy"], r["group"]): r for r in source_rows}
    for row in (r for r in summaries if r["evidence_cap"] == .99):
        old = source_summary[row["budget"], row["sample_size"], row["strategy"], row["group"]]
        fields = {**{m+s: m+s for m in original.METRICS for s in ("_mean", "_se")},
                  "accuracy": "accuracy", "coverage": "coverage", "raw_brier": "brier", "raw_ece": "ece"}
        for name, source_name in fields.items():
            if not math.isclose(row[name], old[source_name], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"recomputed baseline summary differs: {name}")
    differences = comparisons(rows)
    write_json(root / "summary.json", dict(metric_scale="unnormalized", metric_units=metric_metadata(20), complete=True, new_trajectories=120, new_checkpoints=480,
        baseline_trajectories=120, summaries=summaries, differences=differences))
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "per_run.csv", rows)
    write_csv(root / "paired_differences.csv", differences)
    for name in ("summary", "per_run", "paired_differences"):
        write_json(root / f"{name}.csv.metrics.json", metric_metadata(20))
    audit = dict(passed=True, new_trajectories=120, new_checkpoints=480, baseline_summary_matches=True,
        pilot_exact_replays=12, baseline_reproductions=4, network_disabled=True, new_api_requests=0,
        records=hashes, manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "audit.json", audit)
    lines = ["# Neuropathic SMC confidence-cap ablation", "",
        "Completed 120 new trajectories and 480 checkpoints with answer confidence capped at 0.8, matched to the completed 0.99-cap baseline. Both arms use the same new-prompt Qwen3.5-9B responses, n=200/1000, ten data seeds, stated/consistency confidence, and Group I/Group II/all. No new model requests were made.", "",
        "Acquisition confidence remains 0.9 in both arms. Only valid answer confidence is clipped before particle weighting and MH; original confidence is preserved in the response audit. Consistency still requires six of eight valid responses. Abstentions consume one slot and leave the posterior unchanged. Each trajectory has 100 distinct attempts; 10/20/50 are checkpoints without additional terminal moves. As in the baseline, terminal mutation is skipped if slot 100 abstains, preserving the existing implementation despite the paper's final-answered-question wording.", "",
        "Structural values are mean (SE) over ten seeds. Accuracy and calibration pool valid selected answers. Raw Brier/ECE use original confidence; effective Brier/ECE use confidence entering inference (including the baseline 0.99 cap). Query selections can differ between arms. Paired comparisons are left minus right with nominal 95% percentile bootstrap intervals from 10,000 resamples over available paired seeds, conditional on cached model responses. Lowest mean ED does not by itself establish superiority.", "",
        "| Cap | Queries | n | Confidence | Pool | ED (SE) | Adjacency F1 (SE) | Orientation F1 (SE) | Accuracy | Raw Brier | Effective Brier | Coverage |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "NA" if v is None else f"{v:.4f}"
    for row in summaries:
        structural = " | ".join(f"{row[m+'_mean']:.6f} ({row[m+'_se']:.6f})" for m in original.METRICS)
        lines.append(f"| {row['evidence_cap']} | {row['budget']} | {row['sample_size']} | {row['strategy']} | {row['group']} | {structural} | {fmt(row['accuracy'])} | {fmt(row['raw_brier'])} | {fmt(row['effective_brier'])} | {row['coverage']:.4f} |")
    lines += ["", "## Paired ED comparisons", "", "Negative differences favor the left condition. All calibration/F1 contrasts and diagnostics are included in the CSV files.", "",
        "| Queries | n | Left | Right | ED difference | 95% interval |", "|---:|---:|---|---|---:|---|"]
    for row in differences:
        if row["metric"] == "expected_distance" and (row["comparison"] != "within_cap" or row["left_cap"] == CAP):
            lines.append(f"| {row['budget']} | {row['sample_size']} | {row['left_cap']} {row['left_strategy']} {row['left_group']} | {row['right_cap']} {row['right_strategy']} {row['right_group']} | {row['mean']:+.6f} | [{row['low']:+.6f}, {row['high']:+.6f}] |")
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "completion.json", dict(complete=True, audit_passed=True, trajectories=120, checkpoints=480,
        manifest_sha256=sha256(root / "manifest.json"), audit_sha256=sha256(root / "audit.json"),
        artifacts={name: sha256(root / name) for name in ("summary.json", "summary.csv", "per_run.csv", "paired_differences.csv", "report.md")}))
    write_json(root / "status.json", dict(stage="complete", trajectories=120, checkpoints=480, new_api_requests=0))
    print(f"Completed report: {root / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run-block", "validate-pilot", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    root = WORKSPACE_ROOT / "results" / run_id(args.run_id) / STUDY
    if args.stage == "run-block":
        if args.block is None:
            parser.error("run-block requires --block")
        run_block(root, args.block)
    else:
        {"prepare": prepare, "validate-pilot": validate_pilot, "report": report}[args.stage](root)


# Retain the archived protocol checks, adapting the pooled confidence to the answer cap.
def validate_core(record, task, manifest_digest, population_digest):
    if record["task"] != task or record["manifest_sha256"] != manifest_digest:
        raise ValueError("trajectory identity differs")
    if record["initialization_sha256"] != population_digest or record["budget_unit"] != "attempted":
        raise ValueError("initialization or budget unit differs")
    trace = record["trace"]
    allowed = {q.query_id: asdict(q) for q in dag_query_pool(20, original.GROUPS[task["group"]])}
    if len(trace) != 100 or len({s["query"]["query_id"] for s in trace}) != 100:
        raise ValueError("expected 100 distinct attempted questions")
    answered = 0
    for slot, step in enumerate(trace, 1):
        if allowed.get(step["query"]["query_id"]) != step["query"]:
            raise ValueError("question outside the selected group")
        valid = step["valid_response"]
        if step["budget"] != slot or step["cumulative_query_cost"] != slot or valid != (step["answer"] is not None):
            raise ValueError("incorrect attempt accounting")
        answered += int(valid)
        if step["selection_effective_confidence"] != 0.9 or step["support_recovered"]:
            raise ValueError("acquisition or soft evidence setting differs")
        if valid:
            if step["treatment"] != "soft_evidence" or not math.isclose(
                step["effective_confidence"], min(step["reported_confidence"], 0.99), abs_tol=1e-12
            ):
                raise ValueError("incorrect soft confidence")
            expected_moves = 768 if step["resampled"] or slot == 100 else 0
            if step["rejuvenation_attempted_moves"] != expected_moves:
                raise ValueError("incorrect mutation schedule")
        elif step["resampled"] or step["rejuvenation_attempted_moves"] or step["ess_before"] != step["ess_after"]:
            raise ValueError("abstention changed the posterior")
        if task["strategy"] == "consistency":
            audit = step["response_audit"]
            votes = [v for v in audit["votes"] if v is not None]
            if (len(audit["votes"]) != 8 or len(audit["request_keys"]) != 8
                or audit["valid_response_count"] != len(votes) or audit["minimum_valid_response_count"] != 6
                or valid != (len(votes) >= 6)):
                raise ValueError("incorrect six-response consistency acceptance")
            if valid:
                positive = sum(votes)
                expected = 0.5 if 2 * positive == len(votes) else max(0.5, (max(positive, len(votes) - positive) + 0.5) / 9)
                if step["answer"] != int(2 * positive > len(votes)) or not math.isclose(step["reported_confidence"], min(expected, CAP), abs_tol=1e-12):
                    raise ValueError("incorrect consistency answer or confidence")
    if (record["attempted_queries"] != 100 or record["answered_queries"] != answered
        or record["unused_query_budget"] != 0 or record["early_stop_reason"] is not None):
        raise ValueError("incorrect final accounting")
    weights = [e["weight"] for e in record["distribution"]]
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-10):
        raise ValueError("invalid posterior weights")



if __name__ == "__main__":
    main()
