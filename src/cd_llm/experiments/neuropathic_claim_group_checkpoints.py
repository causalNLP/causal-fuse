"""Reconstruct exact 10/20/50-query prefixes of the completed 100-query study."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import math
import shutil

import numpy as np

from cd_llm.core.finite_support import graph_edges, transitive_closure
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments.common.artifacts import json_value, read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_claim_cache import load_offline_provider
from cd_llm.experiments.neuropathic_grid_interpretation import inference_arguments, validate_population
from cd_llm.methods.active_query import active_query_smc, dag_query_pool
from cd_llm.methods.graph_scores import BDeuScore, parent_masks
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_claim_group_checkpoints"
SOURCE = WORKSPACE_ROOT / "results/20260917T160019Z/neuropathic_claim_group_ablation"
BUDGETS = (10, 20, 50, 100)


def verify(root):
    manifest = read_json(root / "manifest.json")
    if manifest["tasks"] != original.tasks() or manifest["budgets"] != list(BUDGETS):
        raise ValueError("checkpoint task grid differs")
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
    source = original.verify(SOURCE)
    completion = read_json(SOURCE / "completion.json")
    if (not completion["complete"] or not completion["audit_passed"]
        or completion["manifest_sha256"] != sha256(SOURCE / "manifest.json")
        or completion["audit_sha256"] != sha256(SOURCE / "audit.json")):
        raise ValueError("source experiment is not complete and audited")
    audit = read_json(SOURCE / "audit.json")
    inputs = {}

    def copy(name, target, expected):
        if sha256(SOURCE / name) != expected:
            raise ValueError(f"source changed: {name}")
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
        if sha256(destination) != expected:
            raise ValueError("copied input differs")
        inputs[target] = expected

    for name, digest in source["inputs"].items():
        if not name.startswith("baseline/"):
            copy(name, name, digest)
    for task in source["tasks"]:
        copy(f"records/{task['id']}.json", f"reference/{task['id']}.json", audit["records"][task["id"]])
    for name in ("manifest.json", "audit.json", "completion.json", "summary.json"):
        copy(name, "reference/" + name, sha256(SOURCE / name))
    implementation = dict(source["implementation"])
    for name in ("code/src/cd_llm/experiments/neuropathic_claim_group_checkpoints.py",
                 "code/scripts/submit_neuropathic_claim_group_checkpoints.py",
                 "sbatch/neuropathic_claim_group_checkpoints.sbatch"):
        implementation[name] = sha256(WORKSPACE_ROOT / name)
    manifest = dict(study=STUDY, version=1, tasks=source["tasks"], budgets=list(BUDGETS),
        source=str(SOURCE), source_manifest_sha256=completion["manifest_sha256"],
        inputs=inputs, implementation=implementation,
        configuration=dict(source["configuration"], checkpoints=list(BUDGETS),
            checkpoint_semantics="prefixes of the original 100-attempt trajectory; no extra terminal moves"))
    write_json(root / "manifest.json", manifest)
    print(f"Prepared 120 offline replays and 480 checkpoints at {root}", flush=True)
    return manifest


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
    reference_path = root / f"reference/{task['id']}.json"
    if record["reference_sha256"] != manifest["inputs"][str(reference_path.relative_to(root))]:
        raise ValueError("reference record identity differs")
    validate_checkpoints(record)
    assert_replay(record, read_json(reference_path))


def execute(root, task, provider):
    reference_path = root / f"reference/{task['id']}.json"
    reference = read_json(reference_path)
    population = root / f"initializations/{task['condition']}_grid_union.json"
    particles, weights = validate_population(read_json(population))
    data = np.load(root / f"inputs/{task['condition']}.npy", allow_pickle=False)
    score = BDeuScore(data, equivalent_sample_size=1, structure_prior=1, cardinalities=(2,) * 20, max_parents=12)
    answered = 0

    def checked_provider(query):
        nonlocal answered
        if answered >= 100 or asdict(query) != reference["trace"][answered]["query"]:
            raise ValueError("acquisition diverged from archived trajectory")
        response = provider(query)
        expected = reference["trace"][answered]
        if response.answer != expected["answer"] or response.reported_confidence != expected["reported_confidence"]:
            raise ValueError("cached answer differs from archived trajectory")
        answered += 1
        return response

    kwargs = inference_arguments(task, particles, weights, "literal")
    kwargs["budgets"] = BUDGETS
    result = active_query_smc(None, score, checked_provider,
        query_pool=dag_query_pool(20, original.GROUPS[task["group"]]), budget_unit="attempted", **kwargs)
    distributions = {str(b): [dict(graph=graph_edges(g), weight=w) for g, w in sorted(result.distributions[b].items())]
                     for b in BUDGETS}
    return json_value(dict(task=task, manifest_sha256=sha256(root / "manifest.json"),
        initialization_sha256=sha256(population), reference_sha256=sha256(reference_path),
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
    if block >= 2:
        require_pilot(root)
    lock_path = root / f"locks/block_{block:02d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, original.offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experts = {s: load_offline_provider(root / "cache", s, minimum_valid=6) for s in original.STRATEGIES}
        for task in (t for t in manifest["tasks"] if t["block"] == block):
            target = root / f"records/{task['id']}.json"
            record = read_json(target) if target.exists() else execute(root, task, experts[task["strategy"]])
            validate_record(root, record, task, manifest)
            if not target.exists():
                write_json(target, record)
            print(f"{task['id']}: four checkpoints; exact 100-query match; {record['elapsed_seconds']:.2f}s", flush=True)
        write_json(root / f"status/block_{block:02d}.json", dict(complete=True, trajectories=6, checkpoints=24,
            manifest_sha256=sha256(root / "manifest.json")))


def validate_pilot(root):
    manifest = verify(root)
    records = {}
    for task in (t for t in manifest["tasks"] if t["seed"] == 0):
        path = root / f"records/{task['id']}.json"
        validate_record(root, read_json(path), task, manifest)
        records[task["id"]] = sha256(path)
    gate = dict(passed=True, trajectories=12, checkpoints=48, exact_100_query_matches=12,
        manifest_sha256=sha256(root / "manifest.json"), records=records)
    write_json(root / "validation/pilot.json", gate)
    print("Pilot passed: 48 checkpoints; all 12 trajectories match their archived 100-query results.", flush=True)
    return gate


def prefix_record(record, budget):
    return dict(distribution=record["distributions"][str(budget)], trace=record["trace"][:budget],
                elapsed_seconds=record["checkpoint_seconds"][str(budget)])


def report(root):
    manifest = verify(root)
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
            row = dict(budget=budget, sample_size=task["sample_size"], strategy=task["strategy"],
                       group=task["group"], seed=task["seed"], **metrics)
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
        row.update(confidence_metrics(pooled[budget, n, strategy, group]))
        row["coverage"] = row["answered"] / row["attempted"]
        summaries.append(row)
    for budget in BUDGETS:
        differences.extend(dict(budget=budget, **r) for r in original.paired_differences([r for r in rows if r["budget"] == budget]))
    reference = {(r["sample_size"], r["strategy"], r["group"]): r
                 for r in read_json(root / "reference/summary.json")["summaries"]}
    for row in (r for r in summaries if r["budget"] == 100):
        previous = reference[row["sample_size"], row["strategy"], row["group"]]
        for key in row:
            if key != "budget" and row[key] != previous[key]:
                raise ValueError(f"100-query summary differs: {key}")
    write_json(root / "summary.json", dict(complete=True, trajectories=120, checkpoints=480, summaries=summaries, differences=differences))
    write_csv(root / "per_run.csv", rows)
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "paired_differences.csv", differences)
    audit = dict(passed=True, trajectories=120, checkpoints=480, exact_100_query_matches=120,
        reference_summary_matches=True, new_model_calls=0, network_disabled=True,
        records=hashes, manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "audit.json", audit)
    lines = ["# Neuropathic claim-group performance at 10, 20, 50, and 100 queries", "",
        "All 120 trajectories were reconstructed offline with four saved posterior checkpoints each. Every query, response, inference diagnostic, and final 100-query distribution matches its archived trajectory, excluding timing and cache bookkeeping. No new model calls were made.", "",
        "These checkpoints are prefixes of the original 100-attempt trajectories. Abstentions count toward each budget and leave the posterior unchanged. Checkpoints add no mutation: MH still runs after resampling or a valid response at terminal slot 100. Therefore these are not independently terminated 10-, 20-, or 50-query experiments with additional terminal moves. The original paper's final-answered-question wording differs when terminal slot 100 is an abstention; the archived implementation is preserved.", "",
        "The settings remain Qwen3.5-9B, n=200/1000, ten paired data seeds, stated or six-of-eight-valid consistency confidence, 384 frozen PC+GES particles, and Group I / canonical ancestry Group II / their union. All responses use soft evidence with cap 0.99. Inference settings and provenance are recorded in the manifest.", "",
        "## Structural performance", "", "Means (standard errors) across ten seeds; lower ED and higher F1 are better.", "",
        "| Queries | n | Confidence | Group | ED (SE) | Adjacency F1 (SE) | Orientation F1 (SE) |",
        "|---:|---:|---|---|---:|---:|---:|"]
    for r in summaries:
        values = " | ".join(f"{r[m + '_mean']:.6f} ({r[m + '_se']:.6f})" for m in original.METRICS)
        lines.append(f"| {r['budget']} | {r['sample_size']} | {r['strategy']} | {r['group']} | {values} |")
    lines += ["", "## Answer diagnostics", "", "Accuracy and Brier pool valid answers within each prefix; coverage includes all attempted slots.", "",
        "| Queries | n | Confidence | Group | Accuracy | Brier | Coverage | Abstentions | Directed / adjacency / ancestry |",
        "|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for r in summaries:
        lines.append(f"| {r['budget']} | {r['sample_size']} | {r['strategy']} | {r['group']} | {r['accuracy']:.4f} | {r['brier']:.4f} | {r['coverage']:.4f} | {r['abstentions']} | {r['directed_queries']} / {r['adjacency_queries']} / {r['ancestry_queries']} |")
    lines += ["", "## Paired ED differences", "", "Left minus right; nominal 95% paired percentile bootstrap intervals over ten data seeds with 10,000 resamples, conditional on the cached model responses. All F1 and calibration contrasts are included in the CSV. Calibration contrasts average seed-level differences; displayed calibration summaries pool valid events.", "",
        "| Queries | n | Left | Right | ED difference | 95% interval |", "|---:|---:|---|---|---:|---|"]
    for r in differences:
        if r["metric"] == "expected_distance":
            lines.append(f"| {r['budget']} | {r['sample_size']} | {r['left_strategy']} {r['left_group']} | {r['right_strategy']} {r['right_group']} | {r['mean']:+.6f} | [{r['low']:+.6f}, {r['high']:+.6f}] |")
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "completion.json", dict(complete=True, audit_passed=True, trajectories=120, checkpoints=480,
        manifest_sha256=sha256(root / "manifest.json"), audit_sha256=sha256(root / "audit.json")))
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
