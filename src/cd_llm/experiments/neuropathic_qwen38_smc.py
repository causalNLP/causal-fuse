"""Matched model comparison with fresh API answers and offline SMC inference."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import fcntl
import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request

import numpy as np

from cd_llm.experiments import neuropathic_ancestry_prompt_api as api
from cd_llm.experiments import neuropathic_new_prompt_cache as cache
from cd_llm.experiments import neuropathic_new_prompt_smc as inference
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments.common.artifacts import json_value, read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_active_interpretation_report import interval
from cd_llm.core.finite_support import stable_seed
from cd_llm.metrics.units import metric_metadata
from cd_llm.methods.active_query import ClaimQuery, dag_query_pool
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.paths import WORKSPACE_ROOT as ROOT

STUDY = "neuropathic_qwen38_smc"
MODEL = "qwen/qwen3.8-27b"
PROVIDER = "deepinfra/bf16"
SOURCE = ROOT / "results/20260918T003643Z/neuropathic_new_prompt_smc"
BUDGETS = (10, 20, 50, 100)
METRICS = (*original.METRICS, "energy_score", "total_tokens", "accuracy", "brier", "ece", "coverage", "effective_brier", "effective_ece")


def tasks():
    return [(q, s) for q in dag_query_pool(20, original.GROUPS["all"]) for s in original.STRATEGIES]


def settings(model=MODEL, provider=PROVIDER):
    return dict(model=model, provider=dict(only=[provider], allow_fallbacks=False, require_parameters=True),
                reasoning=dict(enabled=False), temperature=0., max_tokens=256)


def endpoints(model, provider):
    url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    selected = [e for e in payload["data"]["endpoints"] if e["tag"] == provider]
    if len(selected) != 1 or selected[0]["provider_name"] != "DeepInfra" or selected[0]["quantization"] != "bf16":
        raise ValueError("pinned BF16 endpoint unavailable")
    entry = selected[0]
    if not {"seed", "reasoning", "temperature", "max_tokens"} <= set(entry["supported_parameters"]):
        raise ValueError("provider does not support the frozen generation settings")
    for key in ("prompt", "completion"):
        if not math.isfinite(float(entry["pricing"][key])) or float(entry["pricing"][key]) < 0:
            raise ValueError("invalid pricing")
    return payload, entry


def prepare(root, model=MODEL, provider=PROVIDER, cost_limit=10.):
    if (root / "manifest.json").exists():
        manifest = verify(root)
        if manifest["settings"] != settings(model, provider) or manifest["cost_limit_usd"] != cost_limit:
            raise ValueError("resume settings differ")
        return manifest
    if model != MODEL or provider != PROVIDER or cost_limit != 10.:
        raise ValueError("this registered study requires Qwen3.8-27B, DeepInfra BF16, and a $10 limit")
    source, done = read_json(SOURCE / "manifest.json"), read_json(SOURCE / "completion.json")
    if not done["complete"] or not done["audit_passed"] or done["checkpoints"] != 480:
        raise ValueError("9B comparison incomplete")
    if sha256(SOURCE / "manifest.json") != done["manifest_sha256"] or sha256(SOURCE / "audit.json") != done["audit_sha256"]:
        raise ValueError("9B provenance changed")
    audit = read_json(SOURCE / "audit.json")
    if source["configuration"]["soft_confidence_cap"] != .99 or source["tasks"] != original.tasks():
        raise ValueError("9B protocol differs")
    inputs = {}
    def copy(name, target=None, expected=None):
        target = target or name
        digest = sha256(SOURCE / name)
        if expected is not None and digest != expected:
            raise ValueError(f"source changed: {name}")
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
        if sha256(destination) != digest:
            raise ValueError("copied input differs")
        inputs[target] = digest
    for name, digest in source["inputs"].items():
        if name.startswith(("inputs/", "initializations/", "evaluation/")) or name == "prompts.json":
            copy(name, expected=digest)
    for name in ("manifest.json", "audit.json", "completion.json"):
        copy(name, "reference/" + name)
    for task in source["tasks"]:
        name = f"records/{task['id']}.json"
        copy(name, "reference/" + name, audit["records"][task["id"]])
    payload, endpoint = endpoints(model, provider)
    write_json(root / "provider_endpoints.json", payload)
    inputs["provider_endpoints.json"] = sha256(root / "provider_endpoints.json")
    names = set(source["implementation"])
    names.update(("code/src/cd_llm/experiments/neuropathic_qwen38_smc.py",
                  "code/tests/unit/experiments/test_neuropathic_qwen38_smc.py",
                  "code/figures/neuropathic_qwen38_performance_cost.py", "sbatch/neuropathic_qwen38_smc.sbatch"))
    manifest = dict(study=STUDY, version=1, source=str(SOURCE), tasks=original.tasks(), budgets=list(BUDGETS),
        settings=settings(model, provider), response_provider="DeepInfra", require_token_usage=True, seeds=source["seeds"],
        pricing=endpoint["pricing"], cost_limit_usd=cost_limit, concurrency=4, expected_outcomes=1900,
        nominal_requests=8550, imported_outcomes=0, answer_confidence_cap=.99,
        configuration=dict(source["configuration"], model=model, comparison="qwen38_27b_minus_qwen35_9b_matched_protocol"),
        inputs=inputs, implementation={name: sha256(ROOT / name) for name in sorted(names)})
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", dict(stage="prepared", expected_outcomes=1900, expected_trajectories=120))
    return verify(root)


def verify(root):
    manifest = read_json(root / "manifest.json")
    if (manifest["study"] != STUDY or manifest["tasks"] != original.tasks()
        or manifest["budgets"] != list(BUDGETS) or manifest["settings"] != settings()
        or manifest["cost_limit_usd"] != 10 or manifest["answer_confidence_cap"] != .99
        or manifest["configuration"]["selection_confidence"] != .9
        or manifest["configuration"]["soft_confidence_cap"] != .99):
        raise ValueError("registered experiment differs")
    for collection, parent in (("inputs", root), ("implementation", ROOT)):
        for name, digest in manifest[collection].items():
            if sha256(parent / name) != digest:
                raise ValueError(f"frozen {collection} changed: {name}")
    if read_json(root / "prompts.json") != cache.prompt_templates():
        raise ValueError("prompt templates changed")
    return manifest


async def fill(root):
    manifest, inputs = verify(root), cache.load_inputs(root)
    _, endpoint = endpoints(MODEL, PROVIDER)
    if any(float(endpoint["pricing"][k]) > float(manifest["pricing"][k]) for k in ("prompt", "completion")):
        raise RuntimeError("provider pricing increased; collection paused before paid requests")
    with (root / "fill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = ResponseCache(root / "cache/responses.sqlite3")
        chat = cache.APIChat(db, manifest)
        completed, started, stop = {}, time.monotonic(), asyncio.Event()
        def status(stage):
            write_json(root / "status.json", dict(stage=stage, completed_outcomes=len(completed), expected_outcomes=1900,
                elapsed_seconds=time.monotonic()-started, cache=db.summary(), accounting=chat.accounting()))
            write_json(root / "responses.partial.json", list(completed.values()))
        async def run(q, strategy):
            row = await cache.elicit(q, strategy, chat, inputs)
            completed[q.query_id, strategy] = row
            return row
        async def heartbeat():
            while not stop.is_set():
                status("collecting")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=15)
                except asyncio.TimeoutError:
                    pass
        beat = asyncio.create_task(heartbeat())
        try:
            queries = dag_query_pool(20, original.GROUPS["all"])
            pilot = [q for kind in ("directed", "adjacency", "ancestry") for q in [q for q in queries if q.kind == kind][:2]]
            rows = await asyncio.gather(*(run(q, s) for q in pilot for s in original.STRATEGIES), return_exceptions=True)
            errors = [r for r in rows if isinstance(r, BaseException)]
            if errors:
                raise RuntimeError(str(errors[0]))
            if any(not any(d["status"] in ("valid", "abstain") for d in r["decisions"]) for r in rows):
                raise RuntimeError("pilot contains an entirely malformed outcome")
            write_json(root / "api_pilot.json", dict(passed=True, outcomes=rows, accounting=chat.accounting()))
            print("API pilot passed: 12 outcomes across all claim types and confidence strategies", flush=True)
            queue, failed = asyncio.Queue(), asyncio.Event()
            for q, strategy in tasks():
                if (q.query_id, strategy) not in completed:
                    queue.put_nowait((q, strategy))
            async def worker():
                while not failed.is_set():
                    try:
                        task = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await run(*task)
                        if len(completed) % 20 == 0:
                            print(f"Outcomes {len(completed)}/1900; reserved/charged ${chat.accounting()['budget_used_usd']:.4f}", flush=True)
                    except BaseException:
                        failed.set()
                        raise
            outcomes = await asyncio.gather(*(worker() for _ in range(4)), return_exceptions=True)
            errors = [r for r in outcomes if isinstance(r, BaseException)]
            if errors:
                raise RuntimeError(str(errors[0]))
            write_json(root / "responses.json", [completed[q.query_id, s] for q, s in tasks()])
        except BaseException as exc:
            write_json(root / "failure.json", dict(error=str(exc), accounting=chat.accounting()))
            raise
        finally:
            stop.set()
            await beat
            write_json(root / "accounting.json", chat.accounting())
            status("cache-filled" if len(completed) == 1900 else "incomplete")
            await chat.close()
            db.close()


def replay_cached_outcomes(raw, manifest, inputs):
    async def replay():
        # The event loop creates a local socketpair before entering this coroutine.
        # Block socket creation throughout replay, after loop initialization.
        with original.offline_guard():
            expert = api.OfflineChat(raw, manifest)
            return [await cache.elicit(q, s, expert, inputs) for q, s in tasks()]
    return asyncio.run(replay())


def audit_cache(root):
    manifest, inputs = verify(root), cache.load_inputs(root)
    raw = sqlite_snapshot(root / "cache/responses.sqlite3")
    rows = replay_cached_outcomes(raw, manifest, inputs)
    if rows != read_json(root / "responses.json") or len(rows) != 1900:
        raise ValueError("cache replay incomplete or different")
    if {key for row in rows for key in row["request_keys"]} != set(raw):
        raise ValueError("missing or unexpected responses")
    for entry in raw.values():
        r = entry["record"]
        usage = r["response"].get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens"):
            if not isinstance(r[key], int) or r[key] < 0 or r[key] != usage.get(key):
                raise ValueError("token accounting absent or inconsistent")
    with sqlite3.connect(root / "cache/responses.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        attempts = [dict(r) for r in connection.execute("SELECT * FROM api_attempts ORDER BY id")]
    accounting = read_json(root / "accounting.json")
    charged = sum(r["charged_usd"] or 0 for r in attempts)
    uncertain = sum(r["reserved_usd"] for r in attempts if r["charged_usd"] is None)
    if (len(attempts) != accounting["attempts"] or charged + uncertain > manifest["cost_limit_usd"]
        or not math.isclose(charged, accounting["accounted_cost_usd"], abs_tol=1e-10)
        or not math.isclose(uncertain, accounting["unresolved_reservations_usd"], abs_tol=1e-10)):
        raise ValueError("spending ledger differs")
    write_json(root / "cache/raw_responses.json", raw)
    write_json(root / "cache/api_attempts.json", attempts)
    gate = dict(passed=True, outcomes=1900, new_raw_requests=len(raw), imported_outcomes=0,
        accounting=accounting, manifest_sha256=sha256(root / "manifest.json"),
        artifacts={name: sha256(root / name) for name in ("responses.json", "accounting.json", "cache/raw_responses.json", "cache/api_attempts.json")})
    write_json(root / "cache_audit.json", gate)
    print("Full 27B cache audited offline: 1900 outcomes", flush=True)
    return gate


def providers(root, manifest):
    indexed = {(r["strategy"], r["query"]["query_id"]): r for r in read_json(root / "responses.json")}
    def make(strategy):
        def provider(query):
            row = indexed[strategy, query.query_id]
            if row["query"] != asdict(query):
                raise ValueError("cached query differs")
            return cache.response_for(row, manifest)
        return provider
    return {s: make(s) for s in original.STRATEGIES}


def validate_record(root, record, task, manifest, experts):
    original.validate_record(record, task, sha256(root / "manifest.json"),
        manifest["inputs"][f"initializations/{task['condition']}_grid_union.json"])
    inference.validate_checkpoints(record)
    if record["cache_audit_sha256"] != sha256(root / "cache_audit.json"):
        raise ValueError("cache provenance differs")
    for step in record["trace"]:
        response = experts[task["strategy"]](ClaimQuery(**step["query"]))
        if (response.answer != step["answer"] or response.reported_confidence != step["reported_confidence"]
            or json_value(response.audit) != step["response_audit"]
            or response.prompt_tokens != step["prompt_tokens"] or response.completion_tokens != step["completion_tokens"]):
            raise ValueError("trace differs from audited answers")


def run_block(root, block):
    if block not in range(20):
        raise ValueError("invalid block")
    manifest = verify(root)
    cache.require_cache(root)
    if block >= 2:
        inference.require_pilot(root)
    path = root / f"locks/block_{block:02d}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock, original.offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experts = providers(root, manifest)
        for task in (t for t in manifest["tasks"] if t["block"] == block):
            path = root / f"records/{task['id']}.json"
            record = read_json(path) if path.exists() else inference.execute(root, task, experts[task["strategy"]])
            validate_record(root, record, task, manifest, experts)
            if not path.exists():
                write_json(path, record)
            print(f"{task['id']}: {record['elapsed_seconds']:.2f}s, cap .99", flush=True)
        write_json(root / f"status/block_{block:02d}.json", dict(complete=True, trajectories=6, checkpoints=24))


def validate_pilot(root):
    manifest = verify(root)
    cache.require_cache(root)
    records = {}
    with original.offline_guard():
        experts = providers(root, manifest)
        for task in (t for t in manifest["tasks"] if t["seed"] == 0):
            path = root / f"records/{task['id']}.json"
            record = read_json(path)
            validate_record(root, record, task, manifest, experts)
            replay = inference.execute(root, task, experts[task["strategy"]])
            validate_record(root, replay, task, manifest, experts)
            inference.assert_replay(replay, record)
            if replay["distributions"] != record["distributions"]:
                raise ValueError("checkpoint replay differs")
            records[task["id"]] = sha256(path)
    write_json(root / "validation/pilot.json", dict(passed=True, exact_replays=12, records=records,
        manifest_sha256=sha256(root / "manifest.json")))
    print("All 12 inference pilot trajectories replay exactly", flush=True)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["budget"], row["sample_size"], row["strategy"], row["group"]].append(row)
    summary = []
    for (budget, n, strategy, group), members in sorted(groups.items()):
        if sorted(r["seed"] for r in members) != list(range(10)):
            raise ValueError("ten unique seeds required")
        row = dict(budget=budget, sample_size=n, strategy=strategy, group=group, seeds=10)
        for metric in METRICS:
            values = [r[metric] for r in members if r[metric] is not None and math.isfinite(r[metric])]
            row[metric+"_mean"] = float(np.mean(values)) if values else None
            row[metric+"_se"] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        row.update({key: sum(r[key] for r in members) for key in original.COUNTS})
        summary.append(row)
    return summary


def report(root):
    manifest = verify(root)
    gate = cache.require_cache(root)
    inference.require_pilot(root)
    rows, historical, hashes, ledger = [], [], {}, []
    truth = graph(read_json(root / "evaluation/truth_graph.json")["graph"])
    raw = read_json(root / "cache/raw_responses.json")
    with original.offline_guard():
        experts = providers(root, manifest)
        for task in manifest["tasks"]:
            path = root / f"records/{task['id']}.json"
            record = read_json(path)
            validate_record(root, record, task, manifest, experts)
            hashes[task["id"]] = sha256(path)
            for slot, step in enumerate(record["trace"], 1):
                for attempt, key in enumerate(step["response_audit"]["request_keys"]):
                    r = raw[key]["record"]
                    ledger.append(dict(task=task["id"], slot=slot, request_key=key,
                        prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                        abstention=not step["valid_response"], repair=task["strategy"]=="stated" and attempt>0))
            reference = read_json(root / f"reference/records/{task['id']}.json")
            for destination, trajectory in ((rows, record), (historical, reference)):
                for budget in BUDGETS:
                    metrics, responses = original.metrics_for(inference.prefix_record(trajectory, budget), truth, include_energy=True)
                    effective = confidence_metrics([(correct, min(confidence, .99)) for correct, confidence in responses])
                    metrics.update(effective_brier=effective["brier"], effective_ece=effective["ece"],
                                   total_tokens=metrics["prompt_tokens"]+metrics["completion_tokens"])
                    destination.append(json_value(dict(budget=budget, sample_size=task["sample_size"],
                        strategy=task["strategy"], group=task["group"], seed=task["seed"], **metrics)))
    if len(rows) != 480 or len(historical) != 480:
        raise ValueError("incomplete checkpoint grid")
    summary, reference_summary = summarize(rows), summarize(historical)
    identity = lambda r: (r["budget"], r["sample_size"], r["strategy"], r["group"], r["seed"])
    reference_index = {identity(r): r for r in historical}
    comparisons = []
    for row in summary:
        selected = [r for r in rows if all(r[k] == row[k] for k in ("budget", "sample_size", "strategy", "group"))]
        differences = [r["expected_distance"]-reference_index[identity(r)]["expected_distance"] for r in selected]
        comparisons.append(dict(**{k:row[k] for k in ("budget", "sample_size", "strategy", "group")},
            comparison="27b_minus_9b", paired_seeds=10, **interval(differences, stable_seed(STUDY, *identity(selected[0])[:4]))))
    outputs = dict(per_run=rows, summary=summary, reference_per_run=historical, reference_summary=reference_summary,
                   model_comparisons=comparisons, paired_differences=inference.paired_differences(rows), response_ledger=ledger)
    for name, values in outputs.items():
        write_csv(root / f"{name}.csv", values)
        write_json(root / f"{name}.csv.metrics.json", metric_metadata(20))
    write_json(root / "summary.json", dict(metric_scale="unnormalized", summaries=summary, model_comparisons=comparisons))
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "code/figures/neuropathic_qwen38_performance_cost.py"),
                    "--run-id", root.parent.name], check=True)
    lines = ["# Qwen3.8-27B versus Qwen3.5-9B", "", "Matched SMC protocol with confidence cap 0.99, EIG confidence 0.9, ten paired data seeds, and 100 attempted questions including abstentions. Both model version and parameter count change; differences cannot be attributed to size alone.", "",
        "ED and ES are unnormalized. Means and standard errors are across data seeds; calibration averages seed-level valid-answer diagnostics. Bootstrap intervals use 10,000 paired seed resamples, conditional on fixed cached responses. Token counts include all required response variants, invalid answers, abstentions, and repairs without cache amortization. Unresolved transport charges remain reserved separately.", "",
        f"Actual collection accounting: {gate['accounting']}", "",
        "| n | Confidence | Pool | Queries | 27B ED (SE) | 9B ED (SE) | ED difference [95% interval] | 27B tokens |", "|---:|---|---|---:|---:|---:|---|---:|"]
    for r, previous, delta in zip(summary, reference_summary, comparisons, strict=True):
        lines.append(f"| {r['sample_size']} | {r['strategy']} | {r['group']} | {r['budget']} | {r['expected_distance_mean']:.2f} ({r['expected_distance_se']:.2f}) | {previous['expected_distance_mean']:.2f} ({previous['expected_distance_se']:.2f}) | {delta['mean']:+.2f} [{delta['low']:+.2f}, {delta['high']:+.2f}] | {r['total_tokens_mean']:,.0f} |")
    write_text(root / "report.md", "\n".join(lines)+"\n")
    write_json(root / "audit.json", dict(passed=True, records=hashes, checkpoints=480, trajectories=120,
        manifest_sha256=sha256(root / "manifest.json"), network_disabled_during_inference=True))
    names = [f"{n}.csv" for n in outputs] + ["summary.json", "report.md", "neuropathic_qwen38_performance_cost.pdf", "preview.png"]
    write_json(root / "completion.json", dict(complete=True, audit_passed=True, trajectories=120, checkpoints=480,
        manifest_sha256=sha256(root / "manifest.json"), audit_sha256=sha256(root / "audit.json"),
        artifacts={name:sha256(root/name) for name in names}))
    write_json(root / "status.json", dict(stage="complete", trajectories=120, checkpoints=480, accounting=gate["accounting"]))
    print(f"Complete: {root / 'report.md'}", flush=True)


def submit(root, stage):
    stamp = root.parent.name
    logs = ROOT / "out" / stamp / STUDY
    logs.mkdir(parents=True, exist_ok=True)
    if stage != "collect":
        verify(root)
        cache.require_cache(root)
    if stage == "production":
        inference.require_pilot(root)
    def launch(action, options, label):
        environment = dict(os.environ)
        if action != "collect":
            environment.pop("OPENROUTER_API_KEY", None)
        command = ["sbatch", "--parsable", f"--export=ALL,QWEN38_RUN_ID={stamp},QWEN38_STAGE={action}",
            f"--output={logs}/{label}_%A_%a.out", f"--error={logs}/{label}_%A_%a.err", *options,
            str(ROOT / "sbatch/neuropathic_qwen38_smc.sbatch")]
        job = subprocess.check_output(command, text=True, env=environment).strip().split(";")[0]
        return dict(job_id=job, command=command)
    if stage == "collect":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY must be exported; no job submitted")
        jobs = [launch("collect", ["--time=04:00:00"], "collect")]
        write_json(root / "status.json", dict(stage="collection-submitted", jobs=jobs,
            expected_outcomes=1900, expected_trajectories=120, expected_checkpoints=480))
    else:
        blocks = range(2) if stage == "pilot" else range(2,20)
        manifest = verify(root)
        experts = providers(root, manifest)
        missing = []
        for block in blocks:
            incomplete = False
            for task in (t for t in manifest["tasks"] if t["block"] == block):
                path = root / f"records/{task['id']}.json"
                if path.exists():
                    validate_record(root, read_json(path), task, manifest, experts)
                else:
                    incomplete = True
            if incomplete:
                missing.append(block)
        jobs, dependencies = [], []
        if missing:
            jobs.append(launch("run-block", [f"--array={','.join(map(str,missing))}%4"], stage))
            dependencies = [f"--dependency=afterok:{jobs[-1]['job_id']}", "--kill-on-invalid-dep=yes"]
        action = "continue" if stage == "pilot" else "report"
        jobs.append(launch(action, dependencies, action))
        write_json(root / "status.json", dict(stage=stage+"-submitted", jobs=jobs, blocks=missing))
    write_json(root / f"submission_{stage}_{jobs[0]['job_id']}.json", dict(stage=stage, jobs=jobs))
    print(json_value(dict(stage=stage, jobs=jobs)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("submit", "collect", "run-block", "continue", "report", "resume-inference", "audit-resume"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    root = ROOT / "results" / run_id(args.run_id) / STUDY
    if args.stage == "submit":
        submit(root, "collect")
    elif args.stage == "collect":
        print("Freezing inputs and verifying the pinned endpoint before paid requests", flush=True)
        prepare(root)
        asyncio.run(fill(root))
        audit_cache(root)
        submit(root, "pilot")
    elif args.stage == "run-block":
        run_block(root, args.block)
    elif args.stage == "continue":
        validate_pilot(root)
        submit(root, "production")
    elif args.stage == "resume-inference":
        submit(root, "pilot")
    elif args.stage == "audit-resume":
        audit_cache(root)
        submit(root, "pilot")
    else:
        report(root)


if __name__ == "__main__":
    main()
