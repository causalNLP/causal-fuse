"""Fresh Terra answers and matched, capped SMC trajectories for the clinical study."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import fcntl
from itertools import combinations
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import statistics
import subprocess
import time

from cd_llm.experiments import neuropathic_causal_order_terra as terra
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments import neuropathic_confidence_cap as capped
from cd_llm.experiments import neuropathic_new_prompt_cache as prompts
from cd_llm.experiments import neuropathic_new_prompt_smc as inference
from cd_llm.experiments.common.artifacts import json_value, read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_active_interpretation_report import interval
from cd_llm.core.finite_support import stable_seed
from cd_llm.methods.active_query import ClaimQuery, dag_query_pool, query_values
from cd_llm.metrics.units import metric_metadata
from cd_llm.paths import WORKSPACE_ROOT as ROOT

STUDY = "neuropathic_terra_smc"
SOURCE = ROOT / "results/20260918T153548Z/neuropathic_confidence_cap"
MODEL, PROVIDER, CAP, LIMIT = terra.MODEL, terra.PROVIDER, .8, 25.
BUDGETS = (10, 20, 50, 100)
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1", "total_tokens")


def tasks():
    return [dict(t, id=t["id"].replace("qwen_", "terra_", 1)) for t in original.tasks()]


def queries():
    return dag_query_pool(20, original.GROUPS["all"])


def settings():
    body = terra.request_body([dict(role="user", content="placeholder")], 256)
    body.pop("messages")
    return body


def verify(root):
    m = read_json(root / "manifest.json")
    if (m["study"] != STUDY or m["tasks"] != tasks() or m["settings"] != settings()
        or m["cost_limit_usd"] != LIMIT or m["answer_confidence_cap"] != CAP
        or m["budgets"] != list(BUDGETS) or m["configuration"]["selection_confidence"] != .9
        or m["configuration"]["soft_confidence_cap"] != .99):
        raise ValueError("registered Terra protocol differs")
    for collection, base in (("inputs", root), ("implementation", ROOT)):
        for name, digest in m[collection].items():
            if sha256(base / name) != digest:
                raise ValueError(f"frozen {collection} changed: {name}")
    if read_json(root / "prompts.json") != prompts.prompt_templates():
        raise ValueError("frozen Qwen prompts differ")
    return m


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    source, done, audit = (read_json(SOURCE / name) for name in ("manifest.json", "completion.json", "audit.json"))
    if not done["complete"] or not done["audit_passed"] or done["checkpoints"] != 480:
        raise ValueError("Qwen reference incomplete")
    if sha256(SOURCE / "manifest.json") != done["manifest_sha256"] or sha256(SOURCE / "audit.json") != done["audit_sha256"]:
        raise ValueError("Qwen provenance differs")
    for name, digest in done["artifacts"].items():
        if sha256(SOURCE / name) != digest:
            raise ValueError(f"Qwen artifact changed: {name}")
    if source["answer_confidence_cap"] != CAP or source["tasks"] != original.tasks():
        raise ValueError("Qwen settings differ")
    inputs = {}
    def copy(name, target, digest):
        if sha256(SOURCE / name) != digest:
            raise ValueError(f"Qwen input changed: {name}")
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, path)
        if sha256(path) != digest:
            raise ValueError("copy hash differs")
        inputs[target] = digest
    for name, digest in source["inputs"].items():
        if name.startswith(("inputs/", "initializations/", "evaluation/")) or name == "prompts.json":
            copy(name, name, digest)
    for name in ("manifest.json", "completion.json", "audit.json", "summary.csv", "per_run.csv"):
        copy(name, "reference/" + name, sha256(SOURCE / name))
    for task in source["tasks"]:
        name = f"records/{task['id']}.json"
        copy(name, "reference/" + name, audit["records"][task["id"]])
    payload, endpoint = terra.endpoint()
    write_json(root / "provider_endpoints.json", payload)
    inputs["provider_endpoints.json"] = sha256(root / "provider_endpoints.json")
    # Include all dependencies in the historical SMC implementation fingerprint.
    names = set(source["implementation"])
    names.update(("code/src/cd_llm/experiments/neuropathic_terra_smc.py",
                  "code/src/cd_llm/experiments/neuropathic_causal_order_terra.py",
                  "code/tests/unit/experiments/test_neuropathic_terra_smc.py",
                  "sbatch/neuropathic_terra_smc.sbatch"))
    dated = endpoint.get("name", "").split(" | ")[-1]
    if not dated.startswith(MODEL):
        raise ValueError("endpoint does not identify a Terra model revision")
    config = dict(source["configuration"], model=MODEL, comparison="terra_minus_qwen_matched_cap08")
    config.pop("api_requests", None)
    write_json(root / "manifest.json", dict(study=STUDY, version=1, source=str(SOURCE), tasks=tasks(),
        settings=settings(), model=MODEL, provider=PROVIDER, response_provider="OpenAI",
        response_models=sorted({MODEL, dated}), pricing=endpoint["pricing"], cost_limit_usd=LIMIT,
        answer_confidence_cap=CAP, budgets=list(BUDGETS), configuration=config,
        expected_outcomes=1900, nominal_responses=8550, concurrency=4, inputs=inputs,
        implementation={name:sha256(ROOT / name) for name in sorted(names)}))
    write_json(root / "status.json", dict(stage="prepared", expected_outcomes=1900, expected_trajectories=120))
    return verify(root)


class Chat(terra.Transport):
    """Adapt the validated Terra transport to the fixed binary-claim prompts."""
    def __init__(self, *args, **kwargs):
        self.completed = {}
        super().__init__(*args, **kwargs)

    async def request(self, messages, *, strategy, condition, purpose):
        return await super().request(messages, 256, purpose)

    def progress(self, stage="collecting"):
        count = self.cache.summary()["request_count"]
        elapsed = time.monotonic() - self.started
        rate = (count - self.initial_count) / elapsed if elapsed else 0
        write_json(self.root / "status.json", dict(stage=stage, completed_outcomes=len(self.completed),
            expected_outcomes=1900, completed_responses=count, nominal_responses=8550,
            requests_per_minute=60 * rate, eta_seconds=max(0, 8550-count)/rate if rate else None,
            eta_excludes_future_repairs=True, accounting=self.accounting(),
            updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def pilot_queries():
    return [q for kind in ("directed", "adjacency", "ancestry")
            for q in [q for q in queries() if q.kind == kind][:2]]


def validate_api_pilot(rows):
    if len(rows) != 12:
        raise ValueError("pilot requires all twelve outcomes")
    for row in rows:
        decisions = row["decisions"][-1:] if row["strategy"] == "stated" else row["decisions"]
        if any(d["status"] not in ("valid", "abstain") for d in decisions):
            raise ValueError("pilot contains malformed answers; full collection stopped")


async def collect(root):
    m = verify(root)
    _, endpoint = terra.endpoint()
    if endpoint["pricing"] != m["pricing"]:
        raise ValueError("provider pricing changed; collection stopped before requests")
    inputs = prompts.load_inputs(root)
    chat = Chat(root, m, deadline=time.monotonic() + float(os.environ.get("TERRA_SMC_COLLECT_SECONDS", "13800")))
    stop = asyncio.Event()
    async def heartbeat():
        while not stop.is_set():
            chat.progress()
            write_json(root / "responses.partial.json", list(chat.completed.values()))
            try:
                await asyncio.wait_for(stop.wait(), timeout=15)
            except asyncio.TimeoutError:
                pass
    async def one(q, strategy):
        row = await prompts.elicit(q, strategy, chat, inputs)
        chat.completed[q.query_id, strategy] = row
        return row
    beat = asyncio.create_task(heartbeat())
    failure = None
    try:
        pilot = await asyncio.gather(*(one(q, s) for q in pilot_queries() for s in original.STRATEGIES), return_exceptions=True)
        errors = [r for r in pilot if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        validate_api_pilot(pilot)
        write_json(root / "api_pilot.json", dict(passed=True, outcomes=pilot, accounting=chat.accounting()))
        print("API pilot passed: 12 outcomes; replies reusable in the full bank", flush=True)
        queue = iter((q, s) for q in queries() for s in original.STRATEGIES if (q.query_id, s) not in chat.completed)
        async def worker():
            for q, s in queue:
                if chat.halted:
                    return
                try:
                    await one(q, s)
                except Exception as exc:
                    chat.halted = chat.halted or exc
                    return
                if len(chat.completed) % 20 == 0:
                    print(f"Completed {len(chat.completed)}/1900 outcomes; spending {chat.accounting()['budget_used_usd']:.4f}", flush=True)
        await asyncio.gather(*(worker() for _ in range(4)))
        if chat.halted:
            raise chat.halted
        rows = [chat.completed[q.query_id, s] for q in queries() for s in original.STRATEGIES]
        write_json(root / "responses.json", rows)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        stop.set()
        await beat
        write_json(root / "responses.partial.json", list(chat.completed.values()))
        write_json(root / "accounting.json", chat.accounting())
        chat.progress("cache-filled" if failure is None else "checkpointed" if isinstance(failure, terra.CheckpointRequested) else "blocked")
        await chat.close()


class OfflineChat:
    def __init__(self, raw, manifest):
        self.raw, self.manifest = raw, manifest

    async def request(self, messages, *, strategy, condition, purpose):
        body = terra.request_body(messages, 256)
        item = self.raw[terra.fingerprint(body)]
        if item["request"] != body:
            raise ValueError("cached request differs")
        terra.validate_response(item["record"]["response"], self.manifest)
        return item["record"]


def audit_cache(root):
    m = verify(root)
    raw = sqlite_snapshot(root / "cache/responses.sqlite3")
    async def replay():
        with original.offline_guard():
            chat = OfflineChat(raw, m)
            return [await prompts.elicit(q, s, chat, prompts.load_inputs(root)) for q in queries() for s in original.STRATEGIES]
    rows = asyncio.run(replay())
    if len(rows) != 1900 or rows != read_json(root / "responses.json"):
        raise ValueError("cache replay differs or is incomplete")
    if {k for row in rows for k in row["request_keys"]} != set(raw):
        raise ValueError("response bank has missing or unexpected requests")
    for key, item in raw.items():
        r, request = item["record"], item["request"]
        if key != terra.fingerprint(request) or {k:v for k,v in request.items() if k != "messages"} != settings():
            raise ValueError("cache identity or generation settings differ")
        for field in ("prompt_tokens", "completion_tokens"):
            if r[field] != r["response"]["usage"][field]:
                raise ValueError("cached token usage differs")
    with sqlite3.connect(root / "cache/responses.sqlite3") as db:
        db.row_factory = sqlite3.Row
        attempts = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY id")]
    charged = sum(r["charged_usd"] or 0 for r in attempts)
    unresolved = sum(r["reserved_usd"] for r in attempts if r["charged_usd"] is None)
    accounting = read_json(root / "accounting.json")
    if (len(attempts) != accounting["attempts"] or charged + unresolved > LIMIT
        or not math.isclose(charged, accounting["accounted_cost_usd"], abs_tol=1e-10)
        or not math.isclose(unresolved, accounting["unresolved_reservations_usd"], abs_tol=1e-10)):
        raise ValueError("spending ledger differs")
    write_json(root / "cache/raw_responses.json", raw)
    write_json(root / "cache/attempts.json", attempts)
    names = ("responses.json", "accounting.json", "cache/raw_responses.json", "cache/attempts.json")
    write_json(root / "cache_audit.json", dict(passed=True, outcomes=1900, responses=len(raw), accounting=accounting,
        manifest_sha256=sha256(root / "manifest.json"), artifacts={n:sha256(root/n) for n in names}))
    print("Offline cache audit passed: 1900 outcomes", flush=True)


def providers(root, m):
    index = {(r["strategy"], r["query"]["query_id"]):r for r in read_json(root / "responses.json")}
    def make(strategy):
        def response(query):
            row = index[strategy, query.query_id]
            if row["query"] != asdict(query):
                raise ValueError("cached query differs")
            return capped.cap_response(prompts.response_for(row, m), CAP)
        return response
    return {s:make(s) for s in original.STRATEGIES}


def validate_record(root, record, task, m, experts):
    capped.validate_core(record, task, sha256(root / "manifest.json"), m["inputs"][f"initializations/{task['condition']}_grid_union.json"])
    inference.validate_checkpoints(record)
    if record["cache_audit_sha256"] != sha256(root / "cache_audit.json"):
        raise ValueError("cache audit hash differs")
    for step in record["trace"]:
        r = experts[task["strategy"]](ClaimQuery(**step["query"]))
        if (step["answer"] != r.answer or step["reported_confidence"] != r.reported_confidence
            or step["response_audit"] != json_value(r.audit) or step["prompt_tokens"] != r.prompt_tokens
            or step["completion_tokens"] != r.completion_tokens
            or (step["valid_response"] and step["effective_confidence"] > CAP)):
            raise ValueError("trace differs from capped, audited response")


def run_block(root, block):
    if block not in range(20):
        raise ValueError("block must be 0..19")
    m = verify(root)
    prompts.require_cache(root)
    if block >= 2:
        inference.require_pilot(root)
    lock = root / f"locks/block_{block:02d}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle, original.offline_guard():
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experts = providers(root, m)
        for task in (t for t in m["tasks"] if t["block"] == block):
            path = root / f"records/{task['id']}.json"
            record = read_json(path) if path.exists() else inference.execute(root, task, experts[task["strategy"]])
            validate_record(root, record, task, m, experts)
            if not path.exists():
                write_json(path, record)
            print(f"{task['id']}: {record['elapsed_seconds']:.2f}s; four checkpoints", flush=True)
        write_json(root / f"status/block_{block:02d}.json", dict(complete=True, trajectories=6, checkpoints=24))


def validate_pilot(root):
    m = verify(root)
    prompts.require_cache(root)
    hashes = {}
    with original.offline_guard():
        experts = providers(root, m)
        for task in (t for t in m["tasks"] if t["seed"] == 0):
            path = root / f"records/{task['id']}.json"
            record = read_json(path)
            validate_record(root, record, task, m, experts)
            replay = inference.execute(root, task, experts[task["strategy"]])
            validate_record(root, replay, task, m, experts)
            inference.assert_replay(replay, record)
            if replay["distributions"] != record["distributions"]:
                raise ValueError("pilot checkpoint replay differs")
            hashes[task["id"]] = sha256(path)
    write_json(root / "validation/pilot.json", dict(passed=True, exact_replays=12, records=hashes,
        manifest_sha256=sha256(root / "manifest.json")))


def checkpoint_metrics(record, truth, budget):
    prefix = inference.prefix_record(record, budget)
    metrics, _ = original.metrics_for(prefix, truth, include_energy=True)
    raw, effective = [], []
    for step in prefix["trace"]:
        if step["valid_response"]:
            correct = int(step["answer"] == int(query_values((truth,), ClaimQuery(**step["query"]))[0]))
            raw.append((correct, step["response_audit"]["raw_reported_confidence"]))
            effective.append((correct, step["effective_confidence"]))
    metrics.pop("brier")
    metrics.pop("ece")
    metrics.update(capped.calibration(raw, effective), total_tokens=metrics["prompt_tokens"]+metrics["completion_tokens"])
    return json_value(metrics), raw, effective


def summarize(rows, pooled):
    groups = defaultdict(list)
    for r in rows:
        groups[r["budget"], r["sample_size"], r["strategy"], r["group"]].append(r)
    result = []
    for key, members in sorted(groups.items()):
        if sorted(r["seed"] for r in members) != list(range(10)):
            raise ValueError("ten unique seeds required; no incomplete aggregation")
        budget, n, strategy, group = key
        row = dict(budget=budget, sample_size=n, strategy=strategy, group=group, seeds=10)
        for metric in METRICS:
            values = [r[metric] for r in members]
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f"undefined structural/cost metric: {key}, {metric}")
            row[metric+"_mean"] = statistics.mean(values)
            row[metric+"_se"] = statistics.stdev(values)/math.sqrt(10)
        row.update({name:sum(r[name] for r in members) for name in original.COUNTS})
        row.update(capped.calibration(*pooled[key]), coverage=row["answered"]/row["attempted"])
        result.append(row)
    if len(result) != 48:
        raise ValueError("incomplete aggregate grid")
    return result


def comparisons(rows, reference):
    index = {(r["budget"],r["sample_size"],r["strategy"],r["group"],r["seed"]):r for r in rows}
    ref = {(r["budget"],r["sample_size"],r["strategy"],r["group"],r["seed"]):r for r in reference}
    result = []
    for budget in BUDGETS:
        for n in (200,1000):
            for strategy in original.STRATEGIES:
                for group in original.GROUPS:
                    keys = [(budget,n,strategy,group,s) for s in range(10)]
                    values = [index[k]["expected_distance"]-ref[k]["expected_distance"] for k in keys]
                    identity = dict(comparison="terra_minus_qwen",budget=budget,sample_size=n,strategy=strategy,left_group=group,right_group=group)
                    result.append(dict(identity,paired_seeds=10,**interval(values,stable_seed(STUDY,*identity.values()))))
                for right,left in combinations(original.GROUPS,2):
                    values = [index[budget,n,strategy,left,s]["expected_distance"]-index[budget,n,strategy,right,s]["expected_distance"] for s in range(10)]
                    identity = dict(comparison="terra_pool_difference",budget=budget,sample_size=n,strategy=strategy,left_group=left,right_group=right)
                    result.append(dict(identity,paired_seeds=10,**interval(values,stable_seed(STUDY,*identity.values()))))
    return result


def report(root):
    m = verify(root)
    gate = prompts.require_cache(root)
    inference.require_pilot(root)
    rows, reference, hashes, ledger = [], [], {}, []
    pooled, reference_pooled = defaultdict(lambda:[[],[]]), defaultdict(lambda:[[],[]])
    with original.offline_guard():
        experts = providers(root,m)
        truth = graph(read_json(root / "evaluation/truth_graph.json")["graph"])
        raw_bank = read_json(root / "cache/raw_responses.json")
        for task in m["tasks"]:
            path = root / f"records/{task['id']}.json"
            record = read_json(path)
            validate_record(root,record,task,m,experts)
            hashes[task["id"]] = sha256(path)
            previous = read_json(root / "reference/records" / (task["id"].replace("terra_","qwen_",1)+".json"))
            for destination, pool, trajectory in ((rows,pooled,record),(reference,reference_pooled,previous)):
                for budget in BUDGETS:
                    metrics, raw, effective = checkpoint_metrics(trajectory,truth,budget)
                    destination.append(dict(budget=budget,sample_size=task["sample_size"],strategy=task["strategy"],group=task["group"],seed=task["seed"],**metrics))
                    key = budget,task["sample_size"],task["strategy"],task["group"]
                    pool[key][0].extend(raw)
                    pool[key][1].extend(effective)
            for slot,step in enumerate(record["trace"],1):
                usage = [raw_bank[k]["record"] for k in step["response_audit"]["request_keys"]]
                if (sum(r["prompt_tokens"] for r in usage) != step["prompt_tokens"]
                    or sum(r["completion_tokens"] for r in usage) != step["completion_tokens"]):
                    raise ValueError("per-question expenditure differs from raw requests")
                for attempt,r in enumerate(usage):
                    ledger.append(dict(task=task["id"],slot=slot,request_key=r["request_key"],
                        prompt_tokens=r["prompt_tokens"],completion_tokens=r["completion_tokens"],
                        abstention=not step["valid_response"],repair=task["strategy"]=="stated" and attempt>0))
        if len(rows) != 480 or len(reference) != 480:
            raise ValueError("all 480 checkpoints required")
        summary, reference_summary = summarize(rows,pooled), summarize(reference,reference_pooled)
        paired = comparisons(rows,reference)
    outputs = dict(per_run=rows,summary=summary,reference_per_run=reference,reference_summary=reference_summary,
                   paired_ed_differences=paired,response_ledger=ledger)
    for name, values in outputs.items():
        write_csv(root / f"{name}.csv",values)
        write_json(root / f"{name}.csv.metrics.json",metric_metadata(20))
    write_json(root / "summary.json",dict(metric_scale="unnormalized",summaries=summary,
        reference_summaries=reference_summary,paired_ed_differences=paired))
    lines = ["# Terra SMC-Active with confidence capped at 0.8", "",
        "120 trajectories and 480 checkpoints; ten paired data seeds. Intermediate checkpoints are prefixes without additional mutation. ED and ES are unnormalized. Structural/cost entries are means and SEs across ten seeds. Coverage includes all attempts; calibration pools valid answers. Raw and effective calibration are exported separately.", "",
        "Qwen comparisons use its saved matching 0.8-cap experiment. Prompts, observations, starting particles, acquisition, and inference seeds are shared. Terra uses the pinned OpenAI endpoint, reasoning disabled, request seed zero, and no temperature parameter. Bootstrap intervals use 10,000 paired seed resamples, conditional on each model's shared cached expert responses.", "",
        f"API accounting (USD): {gate['accounting']}. Required response tokens include all variants, abstentions, and repairs without amortizing cached reuse across datasets. Unresolved transport attempts remain reserved; their unobserved tokens cannot be included in measured expenditure.", "",
        "| n | Confidence | Pool | Queries | ED (SE) | ES (SE) | Adj. F1 (SE) | Orient. F1 (SE) | Tokens (SE) |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        values = [f"{r[k+'_mean']:.3f} ({r[k+'_se']:.3f})" for k in METRICS]
        lines.append(f"| {r['sample_size']} | {r['strategy']} | {r['group']} | {r['budget']} | " + " | ".join(values) + " |")
    lines += ["", "## Paired ED differences", "",
        "Differences are Terra minus Qwen, or left pool minus right pool for Terra. Negative values favor the left condition. Intervals are descriptive and are not adjusted for multiple comparisons.", "",
        "| Comparison | n | Confidence | Queries | Left | Right | Mean difference [95% interval] |",
        "|---|---:|---|---:|---|---|---:|"]
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['sample_size']} | {r['strategy']} | {r['budget']} | {r['left_group']} | {r['right_group']} | {r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] |")
    lines += ["", "## Confidence at 100 attempts", "",
        "| n | Confidence | Pool | Coverage | Accuracy | Raw Brier | Effective Brier |",
        "|---:|---|---|---:|---:|---:|---:|"]
    for r in (r for r in summary if r["budget"] == 100):
        values = ["undefined" if r[k] is None else f"{r[k]:.4f}" for k in ("coverage","accuracy","raw_brier","effective_brier")]
        lines.append(f"| {r['sample_size']} | {r['strategy']} | {r['group']} | " + " | ".join(values) + " |")
    write_text(root / "report.md","\n".join(lines)+"\n")
    tex = []
    for s in original.STRATEGIES:
        for g in original.GROUPS:
            cells = []
            for n in (200,1000):
                r = next(r for r in summary if (r["budget"],r["sample_size"],r["strategy"],r["group"])==(100,n,s,g))
                for k, digits in (("expected_distance",2),("orientation_f1",3)):
                    cells.append(f"{r[k+'_mean']:.{digits}f} ({r[k+'_se']:.{digits}f})")
            label = dict(group_i="local",group_ii="ancestry",all="all")[g]
            tex.append(f"SMC-Active ({s}, {label}) & " + " & ".join(cells) + r" \\")
    write_text(root / "table2_rows.tex","\n".join(tex)+"\n")
    write_json(root / "audit.json",dict(passed=True,records=hashes,trajectories=120,checkpoints=480,
        manifest_sha256=sha256(root / "manifest.json"),network_disabled_during_inference=True))
    names = [f"{name}.csv" for name in outputs] + [f"{name}.csv.metrics.json" for name in outputs]
    names += ["summary.json","report.md","table2_rows.tex","cache_audit.json","api_pilot.json","accounting.json"]
    write_json(root / "completion.json",dict(complete=True,audit_passed=True,trajectories=120,checkpoints=480,
        manifest_sha256=sha256(root / "manifest.json"),audit_sha256=sha256(root / "audit.json"),artifacts={n:sha256(root/n) for n in names}))
    write_json(root / "status.json",dict(stage="complete",trajectories=120,checkpoints=480,accounting=gate["accounting"]))


def launch(root, stage, *, dependency=None, array=None):
    logs = ROOT / "out" / root.parent.name / STUDY
    logs.mkdir(parents=True,exist_ok=True)
    command = ["sbatch","--parsable",f"--job-name=np-terra-{stage}",
        "--time=04:00:00" if stage=="collect" else "--time=00:20:00",
        f"--output={logs}/{stage}_%A_%a.out",f"--error={logs}/{stage}_%A_%a.err",
        f"--export=ALL,TERRA_SMC_RUN_ID={root.parent.name},TERRA_SMC_STAGE={stage}"]
    if stage=="collect":
        command.append("--signal=USR1@600")
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    if array:
        command.append(f"--array={array}")
    command.append(str(ROOT / "sbatch/neuropathic_terra_smc.sbatch"))
    env = dict(os.environ)
    if stage not in ("prepare","collect"):
        env.pop("OPENROUTER_API_KEY",None)
    job = subprocess.check_output(command,text=True,env=env,timeout=60).strip().split(";")[0]
    write_json(root / f"jobs/{stage}_{job}.json",dict(job_id=job,stage=stage,dependency=dependency,array=array,submitted=time.time()))
    print(f"Submitted {stage}: {job}",flush=True)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("submit","prepare","collect","audit","run-block","validate-pilot","report"))
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--block",type=int)
    parser.add_argument("--chain",action="store_true")
    args = parser.parse_args(argv)
    root = ROOT / "results" / run_id(args.run_id) / STUDY
    root.mkdir(parents=True,exist_ok=True)
    terra._STOP = False
    signal.signal(signal.SIGUSR1,lambda *_: setattr(terra,"_STOP",True))
    current = os.environ.get("SLURM_JOB_ID")
    try:
        if args.stage=="submit":
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY must be exported")
            launch(root,"prepare")
            return
        if args.stage=="run-block":
            run_block(root,args.block)
            return
        with (root / "workflow.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage=="prepare":
                prepare(root)
            elif args.stage=="collect":
                asyncio.run(collect(root))
                audit_cache(root)
            elif args.stage=="audit":
                audit_cache(root)
            elif args.stage=="validate-pilot":
                validate_pilot(root)
            elif args.stage=="report":
                report(root)
        if args.chain:
            if args.stage=="prepare":
                launch(root,"collect",dependency=current)
            elif args.stage=="collect":
                job = launch(root,"run-block",dependency=current,array="0-1%2")
                launch(root,"validate-pilot",dependency=job)
            elif args.stage=="validate-pilot":
                job = launch(root,"run-block",dependency=current,array="2-19%4")
                launch(root,"report",dependency=job)
    except terra.CheckpointRequested:
        if args.chain:
            launch(root,"collect",dependency=current)
        else:
            print("Checkpoint saved; resume with collect",flush=True)
    except Exception as exc:
        failure = dict(stage=args.stage,block=args.block,error_type=type(exc).__name__,reason=str(exc))
        write_json(root / f"failures/{args.stage}_{args.block}.json",failure)
        previous = read_json(root / "status.json") if (root / "status.json").exists() else {}
        write_json(root / "status.json",dict(previous,stage="blocked",failure=failure))
        raise


if __name__ == "__main__":
    main()
