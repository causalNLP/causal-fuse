"""Offline SMC comparison using causal-criteria prompts for all claim types."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import math

import numpy as np

from cd_llm.core.finite_support import graph_edges, transitive_closure
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments.common.artifacts import json_value, read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.metrics.units import metric_metadata
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_grid_interpretation import inference_arguments, validate_population
from cd_llm.methods.active_query import active_query_smc, dag_query_pool
from cd_llm.methods.graph_scores import BDeuScore, parent_masks
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.paths import WORKSPACE_ROOT

from cd_llm.experiments.neuropathic_new_prompt_cache import (
    STUDY, BUDGETS, prepare, verify, providers, require_cache,
)


def assert_replay(record, reference):
    original.assert_same_inference(record, reference)
    for current, previous in zip(record["trace"], reference["trace"], strict=True):
        for key in previous:
            if key not in ("cumulative_seconds", "response_audit") and json_value(current[key]) != json_value(previous[key]):
                raise ValueError(f"replay trace differs: {key}")
        for key in ("request_keys", "votes", "valid_response_count", "minimum_valid_response_count"):
            if json_value(current["response_audit"].get(key)) != json_value(previous["response_audit"].get(key)):
                raise ValueError(f"replay response differs: {key}")


def validate_checkpoints(record):
    if set(record["distributions"]) != {str(b) for b in BUDGETS}:
        raise ValueError("missing or unexpected checkpoint")
    if record["distributions"]["100"] != record["distribution"]:
        raise ValueError("100-query checkpoint differs from terminal posterior")
    for entries in record["distributions"].values():
        masses = [e["weight"] for e in entries]
        if not masses or any(not math.isfinite(w) or w <= 0 for w in masses) or not math.isclose(sum(masses), 1, abs_tol=1e-10):
            raise ValueError("invalid checkpoint weights")
        graphs = [graph(e["graph"]) for e in entries]
        if len(set(graphs)) != len(graphs):
            raise ValueError("duplicate graph in checkpoint distribution")
        for g in graphs:
            transitive_closure(g)
            if max(p.bit_count() for p in parent_masks(g)) > 12:
                raise ValueError("checkpoint violates parent bound")


def validate_record(root, record, task, manifest):
    original.validate_record(record, task, sha256(root / "manifest.json"),
        manifest["inputs"][f"initializations/{task['condition']}_grid_union.json"])
    validate_checkpoints(record)
    if record["cache_audit_sha256"] != sha256(root / "cache_audit.json"):
        raise ValueError("cache provenance differs")
    experts = providers(root)
    for step in record["trace"]:
        response = experts[task["strategy"]](original.ClaimQuery(**step["query"]))
        if (response.answer != step["answer"] or response.reported_confidence != step["reported_confidence"]
            or json_value(response.audit) != step["response_audit"]):
            raise ValueError("trace differs from the audited cached answer")


def execute(root, task, provider):
    population = root / f"initializations/{task['condition']}_grid_union.json"
    particles, weights = validate_population(read_json(population))
    data = np.load(root / f"inputs/{task['condition']}.npy", allow_pickle=False)
    score = BDeuScore(data, equivalent_sample_size=1, structure_prior=1, cardinalities=(2,) * 20, max_parents=12)
    kwargs = inference_arguments(task, particles, weights, "literal")
    kwargs["budgets"] = BUDGETS
    result = active_query_smc(None, score, provider,
        query_pool=dag_query_pool(20, original.GROUPS[task["group"]]), budget_unit="attempted", **kwargs)
    distributions = {str(b): [dict(graph=graph_edges(g), weight=w) for g, w in sorted(result.distributions[b].items())]
                     for b in BUDGETS}
    return json_value(dict(task=task, manifest_sha256=sha256(root / "manifest.json"),
        initialization_sha256=sha256(population), cache_audit_sha256=sha256(root / "cache_audit.json"),
        budget_unit=result.budget_unit, trace=[dict(asdict(s), cumulative_query_cost=i) for i, s in enumerate(result.steps, 1)],
        distribution=distributions["100"], distributions=distributions,
        checkpoint_seconds=result.cumulative_seconds, elapsed_seconds=result.timing.total_seconds,
        attempted_queries=result.query_count, answered_queries=result.answered_query_count,
        unused_query_budget=result.unused_query_budget, early_stop_reason=result.early_stop_reason))


def require_pilot(root):
    gate = read_json(root / "validation/pilot.json")
    if not gate["passed"] or gate["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("pilot gate not satisfied")
    for name, digest in gate["records"].items():
        if sha256(root / f"records/{name}.json") != digest:
            raise ValueError("validated pilot record changed")


def run_block(root, block):
    if block not in range(20):
        raise ValueError("block must be in 0..19")
    manifest = verify(root)
    require_cache(root)
    if block >= 2:
        require_pilot(root)
    lock_path = root / f"locks/block_{block:02d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, original.offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experts = providers(root)
        for task in (t for t in manifest["tasks"] if t["block"] == block):
            target = root / f"records/{task['id']}.json"
            record = read_json(target) if target.exists() else execute(root, task, experts[task["strategy"]])
            validate_record(root, record, task, manifest)
            if not target.exists():
                write_json(target, record)
            print(f"{task['id']}: four checkpoints; new prompts; {record['elapsed_seconds']:.2f}s", flush=True)
        write_json(root / f"status/block_{block:02d}.json", dict(complete=True, trajectories=6, checkpoints=24,
            manifest_sha256=sha256(root / "manifest.json")))


def validate_pilot(root):
    manifest = verify(root)
    records = {}
    with original.offline_guard():
        experts = providers(root)
        for task in (t for t in manifest["tasks"] if t["seed"] == 0):
            path = root / f"records/{task['id']}.json"
            record = read_json(path)
            validate_record(root, record, task, manifest)
            replay = execute(root, task, experts[task["strategy"]])
            validate_record(root, replay, task, manifest)
            assert_replay(replay, record)
            if replay["distributions"] != record["distributions"]:
                raise ValueError("pilot checkpoint replay differs")
            records[task["id"]] = sha256(path)
            print(f"Pilot exact offline replay: {task['id']}", flush=True)
    gate = dict(passed=True, trajectories=12, checkpoints=48, exact_replays=12,
        manifest_sha256=sha256(root / "manifest.json"), records=records)
    write_json(root / "validation/pilot.json", gate)
    return gate


def prefix_record(record, budget):
    return dict(distribution=record["distributions"][str(budget)], trace=record["trace"][:budget],
                elapsed_seconds=record["checkpoint_seconds"][str(budget)])


def contrast(left, right, identity):
    from cd_llm.core.finite_support import stable_seed
    from cd_llm.experiments.neuropathic_active_interpretation_report import interval
    result = []
    for metric in (*original.METRICS, "accuracy", "brier", "coverage"):
        seeds = [seed for seed in sorted(set(left) & set(right))
                 if left[seed][metric] is not None and right[seed][metric] is not None
                 and math.isfinite(left[seed][metric]) and math.isfinite(right[seed][metric])]
        values = [left[seed][metric] - right[seed][metric] for seed in seeds]
        estimate = (interval(values, stable_seed(STUDY, *identity.values(), metric)) if len(values) >= 2
                    else dict(mean=values[0] if values else None, low=None, high=None))
        result.append(dict(**identity, metric=metric, paired_seeds=len(seeds),
                           missing_paired_seeds=10-len(seeds), **estimate))
    return result


def paired_differences(rows):
    from itertools import combinations
    indexed = {}
    for r in rows:
        indexed.setdefault((r["budget"], r["sample_size"], r["strategy"], r["group"]), {})[r["seed"]] = r
    result = []
    pairs = [(s, second, s, first) for s in original.STRATEGIES
             for first, second in combinations(original.GROUPS, 2)]
    pairs += [("consistency", g, "stated", g) for g in original.GROUPS]
    for budget in BUDGETS:
        for n in (200, 1000):
            for ls, lg, rs, rg in pairs:
                result.extend(contrast(indexed[budget, n, ls, lg], indexed[budget, n, rs, rg],
                    dict(budget=budget, sample_size=n, left_strategy=ls, left_group=lg,
                         right_strategy=rs, right_group=rg)))
    return result


def archived_differences(rows, historical):
    index = lambda items: {(r["budget"], r["sample_size"], r["strategy"], r["group"], r["seed"]): r for r in items}
    current, old = index(rows), index(historical)
    if set(current) != set(old):
        raise ValueError("historical comparison grid differs")
    result = []
    for b, n, s, g in sorted({key[:4] for key in current}):
        result.extend(contrast({seed: current[b, n, s, g, seed] for seed in range(10)},
            {seed: old[b, n, s, g, seed] for seed in range(10)},
            dict(budget=b, sample_size=n, strategy=s, group=g, comparison="new_api_minus_archived_local",
                 interpretation="descriptive_prompt_and_provider_confounded")))
    return result


def report(root):
    with original.offline_guard():
        return _report(root)


def _report(root):
    manifest = verify(root)
    require_cache(root)
    require_pilot(root)
    truth = graph(read_json(root / "evaluation/truth_graph.json")["graph"])
    rows, pooled, groups, hashes = [], defaultdict(list), defaultdict(list), {}
    for task in manifest["tasks"]:
        path = root / f"records/{task['id']}.json"
        record = read_json(path)
        validate_record(root, record, task, manifest)
        hashes[task["id"]] = sha256(path)
        for budget in BUDGETS:
            metrics, responses = original.metrics_for(prefix_record(record, budget), truth)
            row = json_value(dict(budget=budget, sample_size=task["sample_size"], strategy=task["strategy"],
                       group=task["group"], seed=task["seed"], **metrics))
            rows.append(row)
            key = budget, task["sample_size"], task["strategy"], task["group"]
            pooled[key].extend(responses)
            groups[key].append(row)
    summaries, differences = [], []
    for (budget, n, strategy, group), members in sorted(groups.items()):
        if sorted(r["seed"] for r in members) != list(range(10)):
            raise ValueError("report requires all ten seeds")
        row = dict(budget=budget, sample_size=n, strategy=strategy, group=group, seeds=10)
        for metric in original.METRICS:
            values = np.asarray([r[metric] for r in members])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(10))
        row.update({key: sum(r[key] for r in members) for key in original.COUNTS})
        row.update(json_value(confidence_metrics(pooled[budget, n, strategy, group])))
        row["coverage"] = row["answered"] / row["attempted"]
        row["elapsed_seconds"] = sum(r["elapsed_seconds"] for r in members)
        summaries.append(row)
    differences = paired_differences(rows)
    historical = []
    for task in manifest["tasks"]:
        archived = read_json(root / f"reference/records/{task['id']}.json")
        for budget in BUDGETS:
            metrics, _ = original.metrics_for(prefix_record(archived, budget), truth)
            historical.append(json_value(dict(budget=budget, sample_size=task["sample_size"], strategy=task["strategy"],
                                   group=task["group"], seed=task["seed"], **metrics)))
    archive_contrasts = archived_differences(rows, historical)
    write_csv(root / "archived_per_run.csv", historical)
    write_csv(root / "archived_comparisons.csv", archive_contrasts)
    write_json(root / "summary.json", dict(metric_scale="unnormalized", metric_units=metric_metadata(20), complete=True, trajectories=120, checkpoints=480, summaries=summaries, differences=differences))
    write_csv(root / "per_run.csv", rows)
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "paired_differences.csv", differences)
    for name in ("summary", "per_run", "paired_differences", "archived_per_run", "archived_comparisons"):
        write_json(root / f"{name}.csv.metrics.json", metric_metadata(20))
    audit = dict(passed=True, trajectories=120, checkpoints=480,
        network_disabled=True, pilot_exact_replays=12,
        terminal_abstentions=[t["id"] for t in manifest["tasks"]
            if not read_json(root / f"records/{t['id']}.json")["trace"][-1]["valid_response"]],
        records=hashes, manifest_sha256=sha256(root / "manifest.json"), cache=require_cache(root))
    write_json(root / "audit.json", audit)
    lines = ["# Neuropathic SMC-active with revised causal-criteria prompts", "",
        "All 120 trajectories completed, with 480 posterior checkpoints. Qwen3.5-9B through the pinned DeepInfra BF16 API; n=200/1000; ten paired data seeds; stated and consistency confidence; Group I, Group II, and their union. All claim types use revised prompts. The 760 ancestry outcomes were imported from the verified new-prompt API cache.", "",
        "Each trajectory has 100 distinct attempted questions, including abstentions. Consistency requires six valid answers out of eight; ties return NO at confidence 0.5, otherwise confidence is max(0.5, (majority+0.5)/9). Answers are soft evidence capped at 0.99. The same 384 frozen PC+GES particles and inference seeds are used across conditions, without additional initialization moves. BDeu ESS and structure prior are one, maximum parents 12, EIG working confidence 0.9, ESS threshold 0.5. Mutation uses two proposals per particle, guided mixture 0.5, uniform-pair ordinary kernel, cache size 256.", "",
        "Checkpoints at 10/20/50/100 are prefixes of each 100-attempt trajectory; no extra terminal mutation is applied at earlier checkpoints. Mutation follows resampling or a valid response at slot 100. The paper says final answered question; this implementation preserves the archived behavior of skipping terminal mutation when slot 100 abstains.", "",
        "Structural metrics are mean (SE) over ten data seeds. Accuracy and Brier pool valid selected answers; coverage includes all attempted questions. Undefined calibration values remain missing. Paired intervals are nominal 95% percentile bootstrap intervals over available paired seeds, using 10,000 resamples, conditional on the shared cached model responses. Historical comparisons are descriptive: both prompt and provider changed, so differences cannot be attributed solely to prompting.", "",
        "| Queries | n | Confidence | Group | ED (SE) | Adjacency F1 (SE) | Orientation F1 (SE) | Accuracy | Brier | Coverage |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"
    for r in summaries:
        values = " | ".join(f"{r[m + '_mean']:.6f} ({r[m + '_se']:.6f})" for m in original.METRICS)
        lines.append(f"| {r['budget']} | {r['sample_size']} | {r['strategy']} | {r['group']} | {values} | {fmt(r['accuracy'])} | {fmt(r['brier'])} | {r['coverage']:.4f} |")
    cache = audit["cache"]
    lines += ["", f"New API calls: {cache['new_raw_requests']}; reported/estimated new cost: ${cache['new_accounting']['accounted_cost_usd']:.6f}; unresolved reservations: ${cache['new_accounting']['unresolved_reservations_usd']:.6f}. Imported ancestry source cost (not billed again): ${cache['imported_source_cost_usd']:.6f}.", "",
        "The CSV files include all within-study paired structural/calibration contrasts, separate descriptive comparisons to archived results, coverage, abstentions, claim-type counts, request/token counts, inference timing, and MH diagnostics. Cached answers are revealed only after query selection; truth is used only for evaluation."]
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "completion.json", dict(complete=True, audit_passed=True, trajectories=120, checkpoints=480,
        manifest_sha256=sha256(root / "manifest.json"), audit_sha256=sha256(root / "audit.json"),
        artifacts={name: sha256(root / name) for name in ("summary.json", "summary.csv", "per_run.csv", "paired_differences.csv", "archived_comparisons.csv", "report.md")}))
    write_json(root / "status.json", dict(stage="complete", trajectories=120, checkpoints=480))
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


if __name__ == "__main__":
    main()
