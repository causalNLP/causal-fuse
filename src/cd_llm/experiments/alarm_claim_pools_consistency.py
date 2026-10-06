"""Matched eight-variant consistency extension of the ALARM claim-pool study."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, replace
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

import numpy as np

from cd_llm.experiments import alarm_claim_pools as original
from cd_llm.experiments import alarm_claim_pools_consistency_transport as transport
from cd_llm.experiments import alarm_claim_pools_local as stated
from cd_llm.experiments import neuropathic_new_prompt_cache as prompts
from cd_llm.experiments.alarm_claim_pools import (
    ROOT, GROUPS, BUDGETS, CAP, SEEDS, arguments, graph, validate_population,
    offline_guard, grid, graph_edges, active_query_smc, dag_query_pool, ClaimQuery,
    ResponseCache, sqlite_snapshot, json_value, read_json, write_json, sha256, run_id,
)
from cd_llm.experiments.neuropathic_confidence_cap import cap_response


STUDY = "alarm_claim_pools_consistency"
SOURCE = ROOT / "results/20260920T164354Z/alarm_claim_pools_local"
MODEL = "Qwen/Qwen3.5-9B"
SHARDS = 4
EXPECTED_OUTCOMES = 3330
EXPECTED_RESPONSES = EXPECTED_OUTCOMES * 8
CPU_SCRIPT = ROOT / "sbatch/alarm_claim_pools_consistency.sbatch"
GPU_SCRIPT = ROOT / "sbatch/alarm_claim_pools_consistency_gpu.sbatch"


def settings():
    return dict(
        model=MODEL, temperature=0.0, seed=0, max_tokens=256,
        extra_body=dict(chat_template_kwargs=dict(enable_thinking=False)),
    )


def tasks():
    return [
        dict(
            id=f'qwen_consistency_{group}_s{condition["data_seed"]}_n{condition["sample_size"]}_b100',
            condition=condition["id"], block=condition["block"], seed=condition["data_seed"],
            sample_size=condition["sample_size"], strategy="consistency", group=group, budget=100,
        )
        for condition in original.conditions() for group in GROUPS
    ]


def all_queries():
    return list(dag_query_pool(37, GROUPS["all"]))


def pilot_queries():
    queries = all_queries()
    return [query for kind in ("directed", "adjacency", "ancestry")
            for query in [q for q in queries if q.kind == kind][:2]]


def shard_queries(shard):
    pilots = {query.query_id for query in pilot_queries()}
    return [query for index, query in enumerate(all_queries())
            if query.query_id not in pilots and index % SHARDS == shard]


def _implementation_paths():
    return [
        Path(__file__), Path(transport.__file__),
        Path(__file__).with_name("alarm_claim_pools_consistency_report.py"),
        Path(prompts.__file__), Path(stated.__file__), Path(original.__file__),
        ROOT / "code/src/cd_llm/methods/active_query.py",
        ROOT / "code/tests/unit/experiments/test_alarm_claim_pools_consistency.py",
        CPU_SCRIPT, GPU_SCRIPT,
    ]


def validate_stated_source():
    """Validate immutable source artifacts and the sole compatible code change."""
    source = read_json(SOURCE / "manifest.json")
    if source["study"] != stated.STUDY or source["tasks"] != original.tasks():
        raise ValueError("stated-confidence source protocol differs")
    for name, digest in source["inputs"].items():
        if sha256(SOURCE / name) != digest:
            raise ValueError(f"stated-confidence source input changed: {name}")
    completion = read_json(SOURCE / "completion.json")
    if not completion.get("complete") or not completion.get("audit_passed"):
        raise ValueError("stated-confidence source is not complete and audited")
    for name, digest in completion["artifacts"].items():
        if sha256(SOURCE / name) != digest:
            raise ValueError(f"stated-confidence completed artifact changed: {name}")
    changed = []
    for name, digest in source["implementation"].items():
        path = ROOT / name
        if not path.exists() or sha256(path) != digest:
            changed.append(name)
    expected_change = ["code/src/cd_llm/methods/active_query.py"]
    if changed != expected_change:
        raise ValueError(f"unexpected changes relative to stated-confidence implementation: {changed}")

    hashes = {}
    with offline_guard():
        answer = stated.provider(SOURCE, source)
        for task in (value for value in original.tasks() if value["block"] == 0):
            saved_path = SOURCE / "records" / f'{task["id"]}.json'
            saved = read_json(saved_path)
            replay = stated.execute(SOURCE, task, answer, source)
            if replay["distributions"] != saved["distributions"]:
                raise ValueError("current EIG engine does not reproduce the stated source posterior")
            for left, right in zip(replay["trace"], saved["trace"], strict=True):
                left = {key: value for key, value in left.items() if key != "cumulative_seconds"}
                right = {key: value for key, value in right.items() if key != "cumulative_seconds"}
                if left != right:
                    raise ValueError("current EIG engine does not reproduce the stated source trace")
            hashes[task["id"]] = sha256(saved_path)
    return source, dict(
        passed=True, changed_implementation=changed, exact_replay_trajectories=3,
        replay_records=hashes, source_manifest_sha256=sha256(SOURCE / "manifest.json"),
        source_completion_sha256=sha256(SOURCE / "completion.json"),
    )


def verify(root):
    manifest = read_json(root / "manifest.json")
    expected_rule = dict(
        variants=8, wrappers=4, option_orders=["yes-no", "no-yes"],
        minimum_valid_yes_no=6, abstain_below_valid=6,
        invalid_votes=["explicit-abstain", "malformed", "truncated"],
        repair_variants=False, tie_answer="no", tie_confidence=0.5,
        confidence_rule="max(0.5,(majority_count+0.5)/9)", answer_confidence_cap=0.8,
        abstention_consumes_attempt=True, abstention_updates_posterior=False,
    )
    if (manifest["study"] != STUDY or manifest["tasks"] != tasks()
            or manifest["settings"] != settings() or manifest["consistency_rule"] != expected_rule
            or manifest["query_shards"] != SHARDS):
        raise ValueError("registered ALARM consistency protocol differs")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["initializations"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen initialization changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    if sha256(root / "inputs/runtime.json") != manifest["runtime_fingerprint"]:
        raise ValueError("runtime identity differs")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    source, source_compatibility = validate_stated_source()
    root.mkdir(parents=True, exist_ok=True)
    inputs, initializations = {}, {}

    def copy(source_path, name, digest):
        if sha256(source_path) != digest:
            raise ValueError(f"changed source artifact: {source_path}")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        if sha256(destination) != digest:
            raise ValueError(f"copied artifact differs: {name}")

    for name, digest in source["inputs"].items():
        copy(SOURCE / name, name, digest)
        inputs[name] = digest
    for condition in source["conditions"]:
        name = f'initializations/{condition["id"]}_grid_union.json'
        digest = sha256(SOURCE / name)
        value = read_json(SOURCE / name)
        if value["manifest_sha256"] != source["source_manifest_sha256"]:
            raise ValueError("source initialization provenance differs")
        validate_population(value)
        copy(SOURCE / name, name, digest)
        initializations[name] = digest
    runtime = read_json(root / "inputs/runtime.json")
    for name, value in runtime["assets"].items():
        path = Path(name)
        stat = path.stat()
        if stat.st_size != value["size"] or stat.st_mtime_ns != value["mtime_ns"]:
            raise ValueError(f"runtime asset changed: {name}")
    rule = dict(
        variants=8, wrappers=4, option_orders=["yes-no", "no-yes"],
        minimum_valid_yes_no=6, abstain_below_valid=6,
        invalid_votes=["explicit-abstain", "malformed", "truncated"],
        repair_variants=False, tie_answer="no", tie_confidence=0.5,
        confidence_rule="max(0.5,(majority_count+0.5)/9)", answer_confidence_cap=0.8,
        abstention_consumes_attempt=True, abstention_updates_posterior=False,
    )
    implementation = {str(path.relative_to(ROOT)): sha256(path) for path in _implementation_paths()}
    manifest = dict(
        study=STUDY, version=1, tasks=tasks(), conditions=source["conditions"],
        configuration=source["configuration"], settings=settings(), runtime=runtime,
        runtime_fingerprint=inputs["inputs/runtime.json"], backend="local-vllm",
        source_experiment=str(SOURCE), source_manifest_sha256=source["source_manifest_sha256"],
        stated_manifest_sha256=sha256(SOURCE / "manifest.json"),
        stated_completion_sha256=sha256(SOURCE / "completion.json"),
        expected_aggregate_outcomes=EXPECTED_OUTCOMES,
        expected_nominal_responses=EXPECTED_RESPONSES,
        trajectories=60, checkpoints=240, query_shards=SHARDS, concurrency_per_shard=4,
        consistency_rule=rule, actual_api_spending_usd=0.0, paper_update=False,
        shared_initialization_bytes=True, inputs=inputs, initializations=initializations,
        implementation=implementation, source_compatibility=source_compatibility,
    )
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", dict(stage="prepared", expected_outcomes=EXPECTED_OUTCOMES,
                                           expected_responses=EXPECTED_RESPONSES))
    return verify(root)


def runtime_check(root):
    manifest = verify(root)
    for name, value in manifest["runtime"]["assets"].items():
        stat = Path(name).stat()
        if stat.st_size != value["size"] or stat.st_mtime_ns != value["mtime_ns"]:
            raise ValueError(f"runtime asset changed since preparation: {name}")
    return manifest


def validate_consistency_row(row):
    decisions = row.get("decisions") or []
    if row.get("strategy") != "consistency" or len(decisions) != 8 or len(row.get("request_keys") or []) != 8:
        raise ValueError("consistency outcome must contain eight variants")
    valid = [decision["answer"] for decision in decisions if decision.get("status") == "valid"]
    expected_answer, expected_confidence = prompts.local.pool_votes(decisions)
    if row.get("answer") != expected_answer or row.get("reported_confidence") != expected_confidence:
        raise ValueError("consistency aggregation differs")
    if len(valid) < 6:
        if row.get("answer") is not None or row.get("abstention_reason") != "fewer-than-six-valid-variants":
            raise ValueError("fewer than six valid variants must abstain")
    elif row.get("answer") is None or row.get("abstention_reason") is not None:
        raise ValueError("six valid variants must yield an answer")
    if row.get("valid") != (row.get("answer") is not None):
        raise ValueError("outcome validity differs")
    if row.get("prompt_tokens", 0) < 0 or row.get("completion_tokens", 0) < 0:
        raise ValueError("negative token usage")
    return row


def _cache_path(root, label):
    return root / "cache" / f"{label}.sqlite3"


async def _collect_subset(root, label, queries):
    manifest = runtime_check(root)
    base_url = os.environ["ALARM_CONSISTENCY_BASE_URL"]
    cache_path = _cache_path(root, label)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(cache_path)
    deadline = time.monotonic() + max(
        0.0, float(os.environ["ALARM_CONSISTENCY_DEADLINE"]) - time.time())
    chat = transport.LocalChat(cache, manifest, base_url, deadline=deadline)
    signal.signal(signal.SIGUSR1, lambda *_: setattr(chat, "stopped", True))
    status_path = root / "status" / f"{label}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = prompts.load_inputs(root)
    done, errors = {}, []
    started = time.monotonic()

    def progress(stage):
        elapsed = time.monotonic() - started
        write_json(status_path, dict(
            stage=stage, label=label, completed_outcomes=len(done), expected_outcomes=len(queries),
            cached_responses=cache.summary()["request_count"], expected_responses=len(queries) * 8,
            elapsed_seconds=elapsed, accounting=chat.accounting(), updated=time.time(),
        ))
        write_json(root / "partial" / f"{label}.json", list(done.values()))

    async def one(query):
        row = await prompts.elicit(query, "consistency", chat, inputs)
        validate_consistency_row(row)
        done[query.query_id] = row

    async def worker(iterator):
        for query in iterator:
            if errors:
                return
            try:
                await one(query)
            except transport.CheckpointRequested as error:
                errors.append(error)
                return
            except BaseException as error:
                errors.append(error)
                return
            if len(done) % 25 == 0:
                progress("collecting")

    try:
        models = await chat.client.models.list()
        if MODEL not in [item.id for item in models.data]:
            raise ValueError("local server does not expose the frozen model")
        iterator = iter(queries)
        await asyncio.gather(*(worker(iterator) for _ in range(4)))
        if errors and not all(isinstance(error, transport.CheckpointRequested) for error in errors):
            raise errors[0]
        if not errors:
            ordered = [done[query.query_id] for query in queries]
            write_json(root / "outcomes" / f"{label}.json", ordered)
            progress("complete")
        else:
            progress("checkpointed")
    finally:
        write_json(root / "accounting" / f"{label}.json", chat.accounting())
        await chat.close()
        cache.close()


def collect_pilot(root):
    asyncio.run(_collect_subset(root, "pilot", pilot_queries()))
    status = read_json(root / "status/pilot.json")
    if status["stage"] != "complete" or status["completed_outcomes"] != 6 or status["cached_responses"] != 48:
        raise ValueError("consistency pilot did not complete exactly 48 responses")
    rows = read_json(root / "outcomes/pilot.json")
    kinds = defaultdict(int)
    for row in rows:
        validate_consistency_row(row)
        kinds[row["query"]["kind"]] += 1
        prompts_seen = [decision["request_key"] for decision in row["decisions"]]
        if len(set(prompts_seen)) != 8:
            raise ValueError("pilot variants do not have distinct cache identities")
    if dict(kinds) != {"directed": 2, "adjacency": 2, "ancestry": 2}:
        raise ValueError("pilot does not cover two questions per predicate type")
    write_json(root / "validation/prompt_pilot.json", dict(
        passed=True, aggregate_outcomes=6, nominal_responses=48,
        predicate_counts=dict(kinds), manifest_sha256=sha256(root / "manifest.json"),
    ))


def collect_shard(root, shard):
    if shard not in range(SHARDS):
        raise ValueError("invalid collection shard")
    gate = read_json(root / "validation/prompt_pilot.json")
    if not gate.get("passed") or gate["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("production collection requires the prompt pilot")
    asyncio.run(_collect_subset(root, f"shard_{shard}", shard_queries(shard)))


def merge(root):
    manifest = verify(root)
    labels = ["pilot", *(f"shard_{index}" for index in range(SHARDS))]
    for label in labels:
        status = read_json(root / f"status/{label}.json")
        if status["stage"] != "complete":
            raise ValueError(f"collection unit is incomplete: {label}")
    raw, attempts = {}, []
    accounting = defaultdict(int)
    state_counts = defaultdict(int)
    for label in labels:
        snapshot = sqlite_snapshot(_cache_path(root, label))
        overlap = set(raw) & set(snapshot)
        if overlap:
            raise ValueError(f"duplicate response identities across shards: {len(overlap)}")
        raw.update(snapshot)
        with sqlite3.connect(_cache_path(root, label)) as database:
            database.row_factory = sqlite3.Row
            for row in database.execute("SELECT * FROM local_attempts ORDER BY id"):
                value = dict(row)
                value["collection_unit"] = label
                attempts.append(value)
                state_counts[value["state"]] += 1
                if value["response_json"] is not None:
                    response = json.loads(value["response_json"])
                    usage = response.get("usage") or {}
                    accounting["returned_prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                    accounting["returned_completion_tokens"] += usage.get("completion_tokens", 0) or 0
    if len(raw) != EXPECTED_RESPONSES:
        raise ValueError(f"expected {EXPECTED_RESPONSES} unique responses; found {len(raw)}")

    async def replay():
        with offline_guard():
            chat = transport.OfflineChat(raw, manifest)
            rows = []
            inputs = prompts.load_inputs(root)
            for query in all_queries():
                rows.append(validate_consistency_row(
                    await prompts.elicit(query, "consistency", chat, inputs)))
            return rows

    rows = asyncio.run(replay())
    if len(rows) != EXPECTED_OUTCOMES:
        raise ValueError("aggregate response bank is incomplete")
    used = {key for row in rows for key in row["request_keys"]}
    if used != set(raw) or any(len(row["request_keys"]) != 8 for row in rows):
        raise ValueError("offline replay did not consume exactly the merged response bank")
    accounting.update(dict(
        backend="local-vllm", actual_api_spending_usd=0.0,
        attempts=len(attempts), attempt_states=dict(state_counts),
        unique_responses=len(raw), aggregate_outcomes=len(rows),
        unresolved_attempts=state_counts["pending"] + state_counts["uncertain"],
    ))
    write_json(root / "responses.json", rows)
    write_json(root / "cache/raw_responses.json", raw)
    write_json(root / "cache/attempts.json", attempts)
    write_json(root / "accounting.json", dict(accounting))
    artifacts = {
        name: sha256(root / name) for name in
        ("responses.json", "cache/raw_responses.json", "cache/attempts.json", "accounting.json",
         "validation/prompt_pilot.json")
    }
    write_json(root / "cache_audit.json", dict(
        passed=True, aggregate_outcomes=EXPECTED_OUTCOMES,
        nominal_responses=EXPECTED_RESPONSES, variants_per_outcome=8,
        manifest_sha256=sha256(root / "manifest.json"), network_disabled_during_replay=True,
        accounting=dict(accounting), artifacts=artifacts,
    ))
    write_json(root / "status.json", dict(stage="cache-audited", **dict(accounting)))


def require_cache(root):
    gate = read_json(root / "cache_audit.json")
    if (not gate.get("passed") or gate["aggregate_outcomes"] != EXPECTED_OUTCOMES
            or gate["nominal_responses"] != EXPECTED_RESPONSES
            or gate["manifest_sha256"] != sha256(root / "manifest.json")):
        raise ValueError("consistency cache gate is not satisfied")
    for name, digest in gate["artifacts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"audited cache changed: {name}")
    return gate


def response_for(row, manifest):
    response = prompts.response_for(
        row, dict(settings=dict(model=MODEL, provider=dict(only=["local-vllm"]))))
    response = replace(response, audit=dict(
        response.audit, source="local-vllm-consistency-cache",
        model_revision=manifest["runtime"]["model_revision"],
        runtime_fingerprint=manifest["runtime_fingerprint"],
    ))
    return cap_response(response, CAP)


def provider(root, manifest):
    require_cache(root)
    bank = {row["query"]["query_id"]: row for row in read_json(root / "responses.json")}

    def answer(query):
        row = bank[query.query_id]
        if row["query"] != asdict(query):
            raise ValueError("cached query identity differs")
        return response_for(row, manifest)

    return answer


def execute(root, task, answer, manifest):
    population_path = root / f'initializations/{task["condition"]}_grid_union.json'
    population = read_json(population_path)
    if population["manifest_sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("initial population belongs to another experiment")
    particles, weights = validate_population(population)
    data = np.load(root / f'inputs/{task["condition"]}.npy', allow_pickle=False)
    condition = next(value for value in manifest["conditions"] if value["id"] == task["condition"])
    score = grid.common_score(data, condition)
    result = active_query_smc(
        None, score, answer, query_pool=dag_query_pool(37, GROUPS[task["group"]]),
        **arguments(task["seed"], task["sample_size"], particles, weights),
    )
    initial = defaultdict(float)
    for candidate, weight in zip(particles, weights, strict=True):
        initial[candidate] += float(weight)
    distributions = {
        str(budget): [dict(graph=graph_edges(candidate), weight=weight)
                      for candidate, weight in sorted(distribution.items()) if weight > 0]
        for budget, distribution in {0: initial, **result.distributions}.items()
    }
    return json_value(dict(
        task=task, manifest_sha256=sha256(root / "manifest.json"),
        initialization_sha256=sha256(population_path),
        cache_audit_sha256=sha256(root / "cache_audit.json"),
        budget_unit=result.budget_unit,
        trace=[dict(asdict(step), cumulative_query_cost=index)
               for index, step in enumerate(result.steps, 1)],
        distributions=distributions, distribution=distributions["100"],
        checkpoint_seconds=result.cumulative_seconds, elapsed_seconds=result.timing.total_seconds,
    ))


def validate_record(root, record, task, answer):
    original.validate_record(root, record, task, answer)
    for slot, step in enumerate(record["trace"], 1):
        response = answer(ClaimQuery(**step["query"]))
        audit = response.audit
        if (len(audit["request_keys"]) != 8 or audit["minimum_valid_response_count"] != 6
                or step["budget"] != slot or step["cumulative_query_cost"] != slot):
            raise ValueError("consistency response or attempted-query accounting differs")
        if not step["valid_response"]:
            if (step["resampled"] or step["rejuvenation_attempted_moves"]
                    or step["ess_before"] != step["ess_after"]):
                raise ValueError("consistency abstention changed the posterior")


def run_block(root, block):
    manifest = verify(root)
    require_cache(root)
    if block >= 2:
        require_inference_pilot(root)
    lock_path = root / "locks" / f"block_{block}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        answer = provider(root, manifest)
        completed = 0
        for task in (value for value in tasks() if value["block"] == block):
            path = root / "records" / f'{task["id"]}.json'
            record = read_json(path) if path.exists() else execute(root, task, answer, manifest)
            validate_record(root, record, task, answer)
            if not path.exists():
                write_json(path, record)
            completed += 1
            print(f'{task["id"]}: {record["elapsed_seconds"]:.1f}s', flush=True)
        write_json(root / "status" / f"block_{block}.json", dict(
            complete=True, trajectories=completed, checkpoints=completed * 4))


def validate_inference_pilot(root):
    manifest, hashes = verify(root), {}
    with offline_guard():
        answer = provider(root, manifest)
        for task in (value for value in tasks() if value["block"] < 2):
            path = root / "records" / f'{task["id"]}.json'
            saved = read_json(path)
            replay = execute(root, task, answer, manifest)
            validate_record(root, saved, task, answer)
            validate_record(root, replay, task, answer)
            if replay["distributions"] != saved["distributions"]:
                raise ValueError("inference pilot posterior replay differs")
            for left, right in zip(replay["trace"], saved["trace"], strict=True):
                left = {key: value for key, value in left.items() if key != "cumulative_seconds"}
                right = {key: value for key, value in right.items() if key != "cumulative_seconds"}
                if left != right:
                    raise ValueError("inference pilot trace replay differs")
            hashes[task["id"]] = sha256(path)
    write_json(root / "validation/inference_pilot.json", dict(
        passed=True, exact_replays=6, records=hashes,
        manifest_sha256=sha256(root / "manifest.json"), network_disabled=True,
    ))


def require_inference_pilot(root):
    gate = read_json(root / "validation/inference_pilot.json")
    if (not gate.get("passed") or gate["exact_replays"] != 6
            or gate["manifest_sha256"] != sha256(root / "manifest.json")):
        raise ValueError("inference pilot gate is not satisfied")
    for name, digest in gate["records"].items():
        if sha256(root / "records" / f"{name}.json") != digest:
            raise ValueError("inference pilot record changed")
    return gate


def _job_log_root(root):
    path = ROOT / "out" / root.parent.name / STUDY
    path.mkdir(parents=True, exist_ok=True)
    return path


def launch(root, stage, dependency=None, array=None):
    logs = _job_log_root(root)
    gpu = stage in ("pilot", "collect")
    walls = dict(prepare="00:20:00", pilot="00:30:00", collect="02:00:00",
                 merge="00:20:00", run_block="00:20:00",
                 validate_inference_pilot="00:40:00", report="00:20:00")
    command = [
        "sbatch", "--parsable", f"--job-name=alarm-cons-{stage}", f"--time={walls[stage]}",
        f"--output={logs}/{stage}_%A_%a.out", f"--error={logs}/{stage}_%A_%a.err",
        f"--export=ALL,ALARM_CONSISTENCY_RUN_ID={root.parent.name},ALARM_CONSISTENCY_STAGE={stage}",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    if array:
        command.append(f"--array={array}")
    command.append(str(GPU_SCRIPT if gpu else CPU_SCRIPT))
    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    environment.pop("TMPDIR", None)
    environment.pop("SLURM_TMPDIR", None)
    job = subprocess.check_output(command, text=True, env=environment, timeout=60).strip().split(";")[0]
    write_json(root / "jobs" / f"{stage}_{job}.json", dict(
        job_id=job, stage=stage, dependency=dependency, array=array))
    print(f"Submitted {stage}: {job}", flush=True)
    return job


def chain(root, stage):
    current = os.environ["SLURM_JOB_ID"]
    if stage == "prepare":
        launch(root, "pilot", current)
    elif stage == "pilot":
        collection = launch(root, "collect", current, "0-3%4")
        launch(root, "merge", collection)
    elif stage == "merge":
        pilot = launch(root, "run_block", current, "0-1%2")
        launch(root, "validate_inference_pilot", pilot)
    elif stage == "validate_inference_pilot":
        remaining = launch(root, "run_block", current, "2-19%4")
        launch(root, "report", remaining)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "submit", "prepare", "runtime-check", "pilot", "collect", "merge",
        "run-block", "validate-inference-pilot", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--block", type=int, choices=range(20))
    parser.add_argument("--chain", action="store_true")
    args = parser.parse_args(argv)
    root = ROOT / "results" / run_id(args.run_id) / STUDY
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "submit":
        launch(root, "prepare")
        return
    if args.stage == "prepare":
        prepare(root)
    elif args.stage == "runtime-check":
        runtime_check(root)
    elif args.stage == "pilot":
        collect_pilot(root)
    elif args.stage == "collect":
        if args.shard is None:
            raise ValueError("collect requires --shard")
        collect_shard(root, args.shard)
    elif args.stage == "merge":
        merge(root)
    elif args.stage == "run-block":
        if args.block is None:
            raise ValueError("run-block requires --block")
        run_block(root, args.block)
    elif args.stage == "validate-inference-pilot":
        validate_inference_pilot(root)
    elif args.stage == "report":
        from cd_llm.experiments.alarm_claim_pools_consistency_report import report
        report(root)
    if args.chain and args.stage in ("prepare", "pilot", "merge", "validate-inference-pilot"):
        chain(root, args.stage.replace("-", "_"))


if __name__ == "__main__":
    main()
