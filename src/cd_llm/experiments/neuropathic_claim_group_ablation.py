"""Offline Qwen claim-group comparison with 100 attempted questions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import fcntl
from itertools import combinations
import math
from pathlib import Path
import shutil
import socket
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import distribution_metrics, graph_edges, stable_seed
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_active_interpretation_report import interval
from cd_llm.experiments.neuropathic_claim_cache import load_offline_provider, verify_bundle
from cd_llm.experiments.neuropathic_grid_interpretation import inference_arguments, validate_population
from cd_llm.methods.active_query import ClaimQuery, active_query_smc, dag_query_pool, query_values
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_claim_group_ablation"
SOURCE = WORKSPACE_ROOT / "results/20260917T023943Z/neuropathic_valid_answer_budget"
CACHE = WORKSPACE_ROOT / "results/20260917T033207Z/neuropathic_claim_cache/portable_cache"
STRATEGIES = ("stated", "consistency")
GROUPS = {"group_i": ("local_edge",), "group_ii": ("ancestry",), "all": ("local_edge", "ancestry")}
METRICS = ("expected_distance", "adjacency_f1", "orientation_f1")
COUNTS = ("attempted", "answered", "abstentions", "neutral_answers", "directed_queries",
          "adjacency_queries", "ancestry_queries", "resampling_count", "mh_attempted", "mh_accepted",
          "logical_requests", "prompt_tokens", "completion_tokens")
# Cache/timing metadata can change on replay; these fields define the inference trajectory.
TRACE_FIELDS = ("budget", "query", "answer", "valid_response", "reported_confidence",
                "selection_effective_confidence", "predicate_probability", "expected_information_gain",
                "ess_before", "ess_after", "resampled", "support_recovered",
                "rejuvenation_attempted_moves", "rejuvenation_accepted_moves")


def tasks():
    return [dict(id=f"qwen_{strategy}_{group}_s{seed:02d}_n{n}_b100", group=group,
                 seed=seed, sample_size=n, strategy=strategy, budget=100,
                 block=2 * seed + int(n == 1000), condition=f"neuropathic_s{seed:02d}_n{n}")
            for seed in range(10) for n in (200, 1000) for strategy in STRATEGIES for group in GROUPS]


def baseline_id(task):
    return f"qwen_{task['strategy']}_s{task['seed']:02d}_n{task['sample_size']}_b100"


def verify(root):
    manifest = read_json(root / "manifest.json")
    if manifest["tasks"] != tasks():
        raise ValueError("registered task grid differs")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != digest:
            raise ValueError(f"registered implementation changed: {name}")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    source_manifest = read_json(SOURCE / "manifest.json")
    completion = read_json(SOURCE / "completion.json")
    if not completion["complete"] or not completion["audit_passed"] or completion["manifest_sha256"] != sha256(SOURCE / "manifest.json"):
        raise ValueError("source study is not complete and audited")
    bundle = verify_bundle(CACHE)
    inputs, sources = {}, {}

    def copy(source, relative, expected):
        if sha256(source) != expected:
            raise ValueError(f"source changed: {source}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != expected:
            raise ValueError("copied input differs")
        inputs[relative] = sources[str(source)] = expected

    for name, digest in source_manifest["inputs"].items():
        if name.startswith(("initializations/", "baseline/", "evaluation/")) or (
            name.startswith("inputs/") and (name.endswith(".npy") or name in ("inputs/model_metadata.json", "inputs/runtime.json"))
        ):
            copy(SOURCE / name, name, digest)
    for name, digest in bundle["files"].items():
        copy(CACHE / name, "cache/" + name, digest)
    copy(CACHE / "bundle_manifest.json", "cache/bundle_manifest.json", sha256(CACHE / "bundle_manifest.json"))
    if read_json(root / "inputs/model_metadata.json") != read_json(root / "cache/inputs/model_metadata.json"):
        raise ValueError("cache node descriptions differ from the observations")
    for task in tasks():
        population = f"initializations/{task['condition']}_grid_union.json"
        if task["strategy"] == "stated" and task["group"] == "group_i":
            validate_population(read_json(root / population))
        baseline = read_json(root / f"baseline/{baseline_id(task)}.json")
        if baseline["initialization_sha256"] != inputs[population] or len(baseline["trace"]) != 100:
            raise ValueError("Table 2 baseline provenance differs")
    names = set(source_manifest["implementation"])
    names.update("code/src/cd_llm/" + name for name in (
        "experiments/neuropathic_claim_group_ablation.py", "experiments/neuropathic_claim_cache.py", "cli/main.py"))
    names.update(("code/scripts/submit_neuropathic_claim_group_ablation.py",
                  "sbatch/neuropathic_claim_group_ablation.sbatch"))
    configuration = dict(source_manifest["configuration"], budget_unit="attempted",
        consistency_minimum_valid=6, groups={g: list(f) for g, f in GROUPS.items()},
        pool_sizes={g: len(dag_query_pool(20, f)) for g, f in GROUPS.items()},
        terminal_mutation="after resampling or valid response at slot 100; abstentions never mutate",
        offline_only=True, source_initialization_updates=192, additional_initialization_updates=0)
    configuration.pop("pool_size", None)
    manifest = dict(study=STUDY, version=1, tasks=tasks(), inputs=inputs, sources=sources,
        source_manifest_sha256=sha256(SOURCE / "manifest.json"),
        implementation={name: sha256(WORKSPACE_ROOT / name) for name in sorted(names)},
        configuration=configuration)
    write_json(root / "manifest.json", manifest)
    print(f"Prepared {len(tasks())} offline trajectories at {root}", flush=True)
    return manifest


def execute(root, task, provider):
    data = np.load(root / f"inputs/{task['condition']}.npy", allow_pickle=False)
    population = root / f"initializations/{task['condition']}_grid_union.json"
    particles, weights = validate_population(read_json(population))
    score = BDeuScore(data, equivalent_sample_size=1, structure_prior=1, cardinalities=(2,) * 20, max_parents=12)
    result = active_query_smc(None, score, provider,
        query_pool=dag_query_pool(20, GROUPS[task["group"]]), budget_unit="attempted",
        **inference_arguments(task, particles, weights, "literal"))
    return dict(task=task, manifest_sha256=sha256(root / "manifest.json"),
        initialization_sha256=sha256(population), budget_unit=result.budget_unit,
        trace=[dict(asdict(s), cumulative_query_cost=i) for i, s in enumerate(result.steps, 1)],
        distribution=[dict(graph=graph_edges(g), weight=w) for g, w in sorted(result.distributions[100].items())],
        attempted_queries=result.query_count, answered_queries=result.answered_query_count,
        unused_query_budget=result.unused_query_budget, early_stop_reason=result.early_stop_reason,
        elapsed_seconds=result.timing.total_seconds)


def validate_record(record, task, manifest_digest, population_digest):
    if record["task"] != task or record["manifest_sha256"] != manifest_digest:
        raise ValueError("trajectory identity differs")
    if record["initialization_sha256"] != population_digest or record["budget_unit"] != "attempted":
        raise ValueError("initialization or budget unit differs")
    trace = record["trace"]
    allowed = {q.query_id: asdict(q) for q in dag_query_pool(20, GROUPS[task["group"]])}
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
                if step["answer"] != int(2 * positive > len(votes)) or not math.isclose(step["reported_confidence"], expected, abs_tol=1e-12):
                    raise ValueError("incorrect consistency answer or confidence")
    if (record["attempted_queries"] != 100 or record["answered_queries"] != answered
        or record["unused_query_budget"] != 0 or record["early_stop_reason"] is not None):
        raise ValueError("incorrect final accounting")
    weights = [e["weight"] for e in record["distribution"]]
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-10):
        raise ValueError("invalid posterior weights")


def assert_same_inference(left, right):
    signature = lambda r: [{k: s[k] for k in TRACE_FIELDS} for s in r["trace"]]
    # Normalize tuples produced in memory to the representation read from JSON.
    from cd_llm.experiments.common.artifacts import json_value
    if json_value(signature(left)) != json_value(signature(right)) or json_value(left["distribution"]) != json_value(right["distribution"]):
        raise ValueError("inference replay differs")


def checked_record(root, task, manifest):
    record = read_json(root / f"records/{task['id']}.json")
    validate_record(record, task, sha256(root / "manifest.json"),
                    manifest["inputs"][f"initializations/{task['condition']}_grid_union.json"])
    if task["group"] == "group_i":
        assert_same_inference(record, read_json(root / f"baseline/{baseline_id(task)}.json"))
    return record


def offline_guard():
    return patch.object(socket, "socket", side_effect=AssertionError("offline experiment attempted network access"))


def run_block(root, block):
    if block not in range(20):
        raise ValueError("block must be in 0..19")
    manifest = verify(root)
    if block >= 2:
        gate = read_json(root / "validation/pilot.json")
        if not gate["passed"] or gate["manifest_sha256"] != sha256(root / "manifest.json"):
            raise ValueError("pilot gate not satisfied")
    lock_path = root / f"locks/block_{block:02d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experts = {s: load_offline_provider(root / "cache", s, minimum_valid=6) for s in STRATEGIES}
        for task in (t for t in manifest["tasks"] if t["block"] == block):
            target = root / f"records/{task['id']}.json"
            if not target.exists():
                record = execute(root, task, experts[task["strategy"]])
                validate_record(record, task, sha256(root / "manifest.json"),
                    manifest["inputs"][f"initializations/{task['condition']}_grid_union.json"])
                if task["group"] == "group_i":
                    assert_same_inference(record, read_json(root / f"baseline/{baseline_id(task)}.json"))
                write_json(target, record)
            record = checked_record(root, task, manifest)
            print(f"{task['id']}: {record['answered_queries']}/100 valid; {record['elapsed_seconds']:.2f}s", flush=True)
        write_json(root / f"status/block_{block:02d}.json", dict(complete=True, trajectories=6,
                   manifest_sha256=sha256(root / "manifest.json")))


def validate_pilot(root):
    manifest = verify(root)
    pilot = [t for t in manifest["tasks"] if t["seed"] == 0]
    with offline_guard():
        experts = {s: load_offline_provider(root / "cache", s, minimum_valid=6) for s in STRATEGIES}
        for task in pilot:
            record = checked_record(root, task, manifest)
            assert_same_inference(record, execute(root, task, experts[task["strategy"]]))
    result = dict(passed=True, trajectories=12, table2_reproductions=4, exact_offline_replays=12,
                  manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "validation/pilot.json", result)
    print("Pilot passed: 12 exact offline replays and 4 Table 2 reproductions.", flush=True)
    return result


def metrics_for(record, truth, *, include_energy=False):
    entries = record["distribution"]
    metrics = distribution_metrics(tuple(graph(e["graph"]) for e in entries),
                                   np.asarray([e["weight"] for e in entries]), truth, "dag")
    trace, responses = record["trace"], []
    for step in trace:
        if step["valid_response"]:
            correct = int(query_values((truth,), ClaimQuery(**step["query"]))[0])
            responses.append((int(step["answer"] == correct), step["reported_confidence"]))
    counts = Counter(s["query"]["kind"] for s in trace)
    result = dict(**{k: metrics[k] for k in METRICS}, attempted=len(trace), answered=len(responses),
        abstentions=len(trace) - len(responses), neutral_answers=sum(c == 0.5 for _, c in responses),
        **{f"{kind}_queries": counts[kind] for kind in ("directed", "adjacency", "ancestry")},
        resampling_count=sum(s["resampled"] for s in trace),
        mh_attempted=sum(s["rejuvenation_attempted_moves"] for s in trace),
        mh_accepted=sum(s["rejuvenation_accepted_moves"] for s in trace),
        logical_requests=sum(len(s["response_audit"]["request_keys"]) for s in trace),
        prompt_tokens=sum(s["prompt_tokens"] for s in trace),
        completion_tokens=sum(s["completion_tokens"] for s in trace),
        coverage=len(responses) / len(trace), **confidence_metrics(responses),
        elapsed_seconds=record["elapsed_seconds"])
    if include_energy:
        result["energy_score"] = metrics["energy_score"]
    return result, responses


def paired_differences(rows):
    indexed = {(r["sample_size"], r["strategy"], r["group"], r["seed"]): r for r in rows}
    differences = []
    for n in (200, 1000):
        pairs = [(s, second, s, first) for s in STRATEGIES for first, second in combinations(GROUPS, 2)]
        pairs += [("consistency", g, "stated", g) for g in GROUPS]
        for ls, lg, rs, rg in pairs:
            for metric in (*METRICS, "accuracy", "brier", "coverage"):
                values = [indexed[n, ls, lg, seed][metric] - indexed[n, rs, rg, seed][metric] for seed in range(10)]
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("paired metric undefined for a seed")
                differences.append(dict(sample_size=n, left_strategy=ls, left_group=lg,
                    right_strategy=rs, right_group=rg, metric=metric, paired_seeds=10,
                    **interval(values, stable_seed(STUDY, n, ls, lg, rs, rg, metric))))
    return differences


def report(root):
    manifest = verify(root)
    gate = read_json(root / "validation/pilot.json")
    if not gate["passed"] or gate["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("pilot gate not satisfied")
    truth = graph(read_json(root / "evaluation/truth_graph.json")["graph"])
    rows, pooled, groups, record_hashes = [], defaultdict(list), defaultdict(list), {}
    terminal_abstentions = []
    for task in manifest["tasks"]:
        record = checked_record(root, task, manifest)
        metrics, responses = metrics_for(record, truth)
        row = dict(sample_size=task["sample_size"], strategy=task["strategy"], group=task["group"], seed=task["seed"], **metrics)
        rows.append(row)
        key = (task["sample_size"], task["strategy"], task["group"])
        pooled[key].extend(responses)
        groups[key].append(row)
        record_hashes[task["id"]] = sha256(root / f"records/{task['id']}.json")
        if not record["trace"][-1]["valid_response"]:
            terminal_abstentions.append(task["id"])
    summaries = []
    for (n, strategy, group), members in sorted(groups.items()):
        if sorted(r["seed"] for r in members) != list(range(10)):
            raise ValueError("report requires all ten unique seeds")
        summary = dict(sample_size=n, strategy=strategy, group=group, seeds=10)
        for metric in METRICS:
            values = np.asarray([r[metric] for r in members])
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(10))
        summary.update({k: sum(r[k] for r in members) for k in COUNTS})
        summary.update(confidence_metrics(pooled[n, strategy, group]))
        summary["coverage"] = summary["answered"] / summary["attempted"]
        summary["elapsed_seconds"] = sum(r["elapsed_seconds"] for r in members)
        summaries.append(summary)
    differences = paired_differences(rows)
    payload = dict(complete=True, completed=120, expected=120, summaries=summaries, differences=differences)
    write_json(root / "summary.json", payload)
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "per_run.csv", rows)
    write_csv(root / "paired_differences.csv", differences)
    audit = dict(passed=True, trajectories=120, attempted_questions=12000, table2_reproductions=40,
        pilot_exact_replays=12, network_disabled=True, terminal_abstentions=terminal_abstentions,
        records=record_hashes, manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "audit.json", audit)
    lines = ["# Neuropathic SMC-Active claim-group comparison", "", "Completed and audited all 120 trajectories.", "",
        "Qwen3.5-9B; n=200 and n=1000; ten paired data seeds; stated and semantic-consistency confidence. Each trajectory uses 100 distinct attempted questions, including abstentions. Consistency requires at least six of eight valid responses and uses majority confidence (m+0.5)/9, with ties at 0.5. All accepted responses remain soft, capped at effective confidence 0.99.", "",
        "Group I contains 380 directed-edge and 190 adjacency questions. Group II contains 380 canonical ancestry questions; the union contains 950 questions. Ancestry correctness is evaluated using directed reachability. Cached answers are revealed only after adaptive selection; ground truth is used only for evaluation. No new model requests were made.", "",
        "All conditions reuse the same frozen Table 2 observations, node descriptions, 384-particle PC+GES grid initializations (including their 192 observational initialization updates), and inference seeds. No additional initialization moves are applied. Inference uses BDeu with equivalent sample size and structure prior one, at most 12 parents, EIG working confidence 0.9, ESS threshold 192, and two MH proposals per particle after resampling or a valid response at slot 100. The guided-mixture probability is 0.5, the ordinary kernel is uniform-pair, and the proposal cache holds 256 states.", "",
        "Protocol caveat: the paper says mutation follows the final answered question, but the existing Table 2 implementation skips terminal mutation when slot 100 is an abstention. This study preserves that behavior. All 40 Group I trajectories reproduce the original Table 2 inference traces and final distributions; all 12 pilot trajectories also pass exact offline replay with socket creation disabled.", "",
        "## Structural performance", "", "Means and standard errors across ten data seeds; lower ED and higher F1 are better.", "",
        "| n | Confidence | Group | ED (SE) | Adjacency F1 (SE) | Orientation F1 (SE) |",
        "|---:|---|---|---:|---:|---:|"]
    for row in summaries:
        values = " | ".join(f"{row[m + '_mean']:.6f} ({row[m + '_se']:.6f})" for m in METRICS)
        lines.append(f"| {row['sample_size']} | {row['strategy']} | {row['group']} | {values} |")
    lines += ["", "## Selected-answer diagnostics", "", "Accuracy and Brier pool valid selected answers; coverage uses all 1000 attempted slots per cell. The selected questions differ across groups and confidence methods.", "",
        "| n | Confidence | Group | Valid / attempted | Accuracy | Brier | Directed / adjacency / ancestry | MH attempts |",
        "|---:|---|---|---:|---:|---:|---|---:|"]
    for row in summaries:
        lines.append(f"| {row['sample_size']} | {row['strategy']} | {row['group']} | {row['answered']} / {row['attempted']} | {row['accuracy']:.4f} | {row['brier']:.4f} | {row['directed_queries']} / {row['adjacency_queries']} / {row['ancestry_queries']} | {row['mh_attempted']} |")
    lines += ["", "## Paired ED comparisons", "", "Differences are left minus right. Intervals are nominal 95% paired percentile bootstrap intervals over ten data seeds (10,000 resamples), conditional on the fixed cached model responses. Calibration differences in the CSV average seed-level differences; pooled calibration values above use valid-answer event weights.", "",
        "| n | Left | Right | ED difference | 95% interval |", "|---:|---|---|---:|---|"]
    for row in differences:
        if row["metric"] == "expected_distance":
            lines.append(f"| {row['sample_size']} | {row['left_strategy']} {row['left_group']} | {row['right_strategy']} {row['right_group']} | {row['mean']:+.6f} | [{row['low']:+.6f}, {row['high']:+.6f}] |")
    lines += ["", "Full structural and calibration contrasts, coverage, abstentions, query-type counts, resampling counts, MH acceptance, and logical request/token totals are saved in the CSV/JSON summaries. Logical token totals count reused cached responses and are not new API expenditure. Frozen provenance and per-record checksums are included in the manifest and audit."]
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "completion.json", dict(complete=True, completed=120, expected=120, audit_passed=True,
        manifest_sha256=sha256(root / "manifest.json"), audit_sha256=sha256(root / "audit.json")))
    print(f"Report: {root / 'report.md'}", flush=True)
    return payload


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


if __name__ == "__main__":
    main()
