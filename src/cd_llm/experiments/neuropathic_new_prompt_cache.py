"""Pinned API cache for the all-claim causal-criteria prompt study."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from email.utils import parsedate_to_datetime
import fcntl
import json
import math
import sqlite3
import time
from pathlib import Path

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.experiments import neuropathic_ancestry_prompt_api as api
from cd_llm.experiments import neuropathic_ancestry_prompt_ablation as local
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments.common.artifacts import read_json, sha256, write_json
from cd_llm.experiments.common.reproduction_inputs import materialize, protocol
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot
from cd_llm.methods.active_query import ClaimQuery, QueryResponse, dag_query_pool
from cd_llm.paths import CODE_ROOT

STUDY = "neuropathic_new_prompt_smc"
BUDGETS = (10, 20, 50, 100)


def prompt_templates():
    base = local.prompt_templates()
    return dict(ancestry=base, group_i=dict(
        system=base["system"], response_instructions=base["response_instructions"],
        malformed_retry=base["malformed_retry"],
        wrappers=["{question}", "Assess whether this causal relation is true: {question}",
                  "Judge this direct causal proposition: {question}",
                  "In {domain_context}, decide whether the following relation holds: {question}"],
        directed=dict(
            question="Assess whether {X} directly causes {Y} in the stated clinical context.",
            yes_rule="Answer YES when your knowledge supports a specific direct causal\nmechanism from {X} to {Y}.",
            no_rule="Answer NO when your knowledge supports the absence of a direct\ncausal edge from {X} to {Y}."),
        adjacency=dict(
            question="Assess whether {X} and {Y} are directly causally related in either\ndirection in the stated clinical context.",
            yes_rule="Answer YES when your knowledge supports a specific direct causal\nmechanism in at least one direction.",
            no_rule="Answer NO when your knowledge supports the absence of a direct\ncausal edge in both directions."),
        user="""Context: {domain_context}.
Glossary:
{X} = {description_X}
{Y} = {description_Y}

{question}

Respect the exact variable definitions, anatomical level, and laterality.
Association, shared symptoms, and a common cause do not by themselves
establish a direct causal edge.
A pathway through an intermediate variable alone does not establish
a direct causal edge.
Do not invent causal mechanisms.

{yes_rule}
{no_rule}
Answer ABSTAIN when you cannot make either judgment defensibly.
Failure to recall a direct mechanism is not sufficient evidence for NO.

{response_instructions}"""))


def messages(query, strategy, inputs, *, variant=0, retry=False):
    if query.kind == "ancestry":
        return local.messages(query, strategy, inputs["metadata"], inputs["templates"]["ancestry"],
                              variant=variant, retry=retry)
    if query.kind not in ("directed", "adjacency") or strategy not in local.STRATEGIES:
        raise ValueError("unsupported query or strategy")
    if variant not in range(8) or (strategy == "stated" and variant != 0):
        raise ValueError("invalid variant")
    metadata, template = inputs["metadata"], inputs["templates"]["group_i"]
    x, y = (metadata["nodes"][i] for i in (query.source, query.target))
    values = dict(X=x["id"], Y=y["id"], description_X=x["description"],
                  description_Y=y["description"], domain_context=metadata["domain_context"])
    rules = {k: v.format(**values) for k, v in template[query.kind].items()}
    instructions = template["response_instructions"][strategy]
    if strategy == "consistency":
        rules["question"] = template["wrappers"][variant // 2].format(question=rules["question"], **values)
        first, second = ("yes", "no") if variant % 2 == 0 else ("no", "yes")
        instructions = instructions.replace("{first_label}", first).replace("{second_label}", second)
    content = template["user"].format(**values, **rules, response_instructions=instructions)
    if retry:
        content += template["malformed_retry"]
    return ({"role": "system", "content": template["system"][strategy]}, {"role": "user", "content": content})


def retry_after(exc, now=None):
    """Seconds, or an HTTP date, without exposing a provider error body."""
    headers = getattr(getattr(exc, "response", None), "headers", {})
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


class APIChat(api.APIChat):
    """Persistent spending reservations with a shared cooldown for all workers."""
    def __init__(self, *args, sleep=asyncio.sleep, clock=time.monotonic, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep, self.clock, self.cooldown_until = sleep, clock, 0.0

    async def wait_cooldown(self):
        while self.cooldown_until > self.clock():
            await self.sleep(self.cooldown_until - self.clock())

    def defer(self, exc, attempt):
        delay = retry_after(exc)
        if delay is None:
            delay = min(300, 30 * 2 ** (attempt - 1))
        self.cooldown_until = max(self.cooldown_until, self.clock() + delay)
        return delay

    async def request(self, messages, *, strategy, condition, purpose):
        request = api.request_record(messages, strategy, condition, self.manifest)
        async with self.semaphore:
            cached = self.cache.get(request)
            if cached is not None:
                api.validate_response(cached["response"], self.manifest)
                return cached
            for attempt in range(1, 21):
                await self.wait_cooldown()
                event = self.reserve(request)
                started, body = self.clock(), request["body"]
                kwargs = {k: v for k, v in body.items() if k not in ("provider", "reasoning")}
                try:
                    value = await self.client.chat.completions.create(**kwargs,
                        extra_body=dict(provider=body["provider"], reasoning=body["reasoning"]))
                except Exception as exc:
                    self.cache.connection.execute("UPDATE api_attempts SET state='uncertain' WHERE id=?", (event,))
                    self.cache.connection.commit()
                    status = getattr(exc, "status_code", None)
                    if attempt == 20 or (status is not None and status not in (408, 409, 429) and status < 500):
                        raise RuntimeError(f"OpenRouter transport failed ({status or type(exc).__name__}); cache incomplete and resumable") from None
                    delay = self.defer(exc, attempt)
                    print(f"Transport retry {attempt}/20 ({status or type(exc).__name__}); shared cooldown {delay:.1f}s", flush=True)
                    continue
                response = value.model_dump()
                usage = response.get("usage") or {}
                prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                cost, estimated = usage.get("cost"), usage.get("cost") is None
                if estimated:
                    if prompt is None or completion is None:
                        raise RuntimeError("response lacks cost and usage; reservation retained")
                    cost = prompt * float(self.manifest["pricing"]["prompt"]) + completion * float(self.manifest["pricing"]["completion"])
                if not math.isfinite(float(cost)) or float(cost) < 0:
                    raise RuntimeError("invalid API cost; reservation retained")
                self.cache.connection.execute("UPDATE api_attempts SET charged_usd=?,state=? WHERE id=?",
                    (float(cost), "estimated" if estimated else "reported", event))
                self.cache.connection.commit()
                choices = response.get("choices") or []
                content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
                self.cache.put(request, purpose, response, content, provider_request_id=response.get("id"),
                    response_model=response.get("model"), prompt_tokens=prompt, completion_tokens=completion,
                    total_tokens=usage.get("total_tokens"), cost_usd=float(cost),
                    latency_seconds=self.clock() - started, attempts=attempt)
                api.validate_response(response, self.manifest)
                if self.accounting()["budget_used_usd"] > self.manifest["cost_limit_usd"]:
                    raise RuntimeError("reported costs exceeded the run limit")
                return self.cache.get(request)
        raise AssertionError("unreachable")


async def elicit(query, strategy, chat, inputs):
    records, decisions = [], []
    for index in range(2 if strategy == "stated" else 8):
        prompt = messages(query, strategy, inputs, variant=index if strategy == "consistency" else 0,
                          retry=strategy == "stated" and index == 1)
        prefix = "ancestry-api" if query.kind == "ancestry" else "direct-api"
        record = await chat.request(prompt, strategy=strategy, condition="new",
                                    purpose=f"{prefix}:new:{strategy}:{query.query_id}:{index}")
        records.append(record)
        decisions.append(local.parse_record(record, strategy))
        if strategy == "stated" and decisions[-1]["status"] in ("valid", "abstain"):
            break
    if strategy == "stated":
        answer, confidence = decisions[-1]["answer"], decisions[-1]["reported_confidence"]
        reason = None if answer is not None else decisions[-1]["status"]
    else:
        answer, confidence = local.pool_votes(decisions)
        reason = None if answer is not None else "fewer-than-six-valid-variants"
    return dict(query=asdict(query), strategy=strategy, condition="new", answer=answer,
        reported_confidence=confidence, valid=answer is not None, abstention_reason=reason,
        decisions=decisions, request_keys=[r["request_key"] for r in records],
        prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in records),
        completion_tokens=sum(r.get("completion_tokens") or 0 for r in records),
        cost_usd=sum(r["cost_usd"] for r in records))


def load_inputs(root):
    return dict(metadata=read_json(root / "inputs/model_metadata.json"), templates=read_json(root / "prompts.json"))


def verify(root):
    manifest = read_json(root / "manifest.json")
    if manifest["tasks"] != original.tasks() or manifest["budgets"] != list(BUDGETS):
        raise ValueError("task grid differs")
    if manifest["settings"] != api.SETTINGS or manifest["cost_limit_usd"] != 2.0:
        raise ValueError("API settings differ")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(CODE_ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    if read_json(root / "prompts.json") != prompt_templates():
        raise ValueError("prompt templates differ")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    inputs = materialize("neuropathic", root)
    if read_json(root / "prompts.json") != prompt_templates():
        raise ValueError("packaged neuropathic prompts differ from the implementation")
    settings = protocol()["hosted_models"]["qwen"]
    if settings["model"] != api.MODEL or settings["provider"] != api.ENDPOINT:
        raise ValueError("registered hosted Qwen route differs")
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("neuropathic_new_prompt_smc.py"),
        Path(original.__file__),
        Path(local.__file__),
        Path(api.__file__),
    )
    implementation = {
        str(path.resolve().relative_to(CODE_ROOT.resolve())): sha256(path)
        for path in implementation_paths
    }
    configuration = dict(
        budget=100, checkpoints=list(BUDGETS), budget_unit="attempted",
        groups=original.GROUPS, confidence_methods=list(local.STRATEGIES),
        consistency_variants=8, consistency_minimum_valid=6,
        particle_count=384, score="BDeu", equivalent_sample_size=1,
        structure_prior=1, max_parents=12, policy="eig",
        selection_confidence=.9, soft_confidence_cap=.99,
        ess_threshold_fraction=.5, rejuvenation_steps=2,
        rejuvenate_after_every_query=False, guided_mixture_probability=.5,
        proposal_cache_size=256, ordinary_kernel="uniform_pair",
        prompt_condition="causal criteria for every claim type",
    )
    manifest = dict(study=STUDY, version=1, tasks=original.tasks(), budgets=list(BUDGETS),
        input_bundle=protocol()["bundles"]["neuropathic"], inputs=inputs,
        implementation=implementation, settings=api.SETTINGS,
        pricing=dict(prompt="0.0000001", completion="0.00000015"),
        endpoint=dict(tag=api.ENDPOINT, model_id=api.MODEL),
        seeds=dict(stated=19094550, consistency=2881584013),
        cost_limit_usd=2.0, nominal_new_requests=8550, expected_outcomes=1900,
        concurrency=4, max_transport_attempts=20, retry_cooldown_seconds=[30, 300],
        configuration=configuration)
    write_json(root / "manifest.json", manifest)
    print(f"Prepared {root}: 120 trajectories, 480 checkpoints; all model answers will be collected fresh.", flush=True)
    return manifest


def tasks():
    return [(q, s) for q in dag_query_pool(20, original.GROUPS["all"]) for s in local.STRATEGIES]


async def fill(root, *, pilot_only=False):
    manifest, inputs = verify(root), load_inputs(root)
    with (root / "fill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = ResponseCache(root / "cache/responses.sqlite3")
        chat = APIChat(cache, manifest)
        completed, started = {}, time.monotonic()
        def status(stage):
            write_json(root / "status.json", dict(stage=stage, completed_outcomes=len(completed),
                expected_outcomes=1900,
                elapsed_seconds=time.monotonic() - started, cache=cache.summary(), accounting=chat.accounting()))
        async def run(task):
            row = await elicit(*task, chat, inputs)
            completed[task[0].query_id, task[1]] = row
            if len(completed) % 20 == 0:
                status("filling")
                print(f"Claim outcomes {len(completed)}/1900; ${chat.accounting()['budget_used_usd']:.4f}", flush=True)
            return row
        heartbeat_stop = asyncio.Event()
        async def heartbeat():
            while not heartbeat_stop.is_set():
                status("filling")
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=15)
                except asyncio.TimeoutError:
                    pass
        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            queries = dag_query_pool(20, original.GROUPS["all"])
            pilot_queries = [next(q for q in queries if q.kind == kind) for kind in ("directed", "adjacency")]
            pilot = await asyncio.gather(*(run((q, s)) for q in pilot_queries for s in local.STRATEGIES), return_exceptions=True)
            errors = [r for r in pilot if isinstance(r, BaseException)]
            if errors:
                raise RuntimeError(str(errors[0]))
            if any(not any(d["status"] in ("valid", "abstain") for d in r["decisions"]) for r in pilot):
                raise RuntimeError("API pilot has an entirely malformed outcome")
            write_json(root / "api_pilot.json", dict(passed=True, outcomes=pilot, accounting=chat.accounting()))
            print("API pilot passed: directed and adjacency, stated and consistency.", flush=True)
            if pilot_only:
                return
            queue, stop = asyncio.Queue(), asyncio.Event()
            for task in tasks():
                if (task[0].query_id, task[1]) not in completed:
                    queue.put_nowait(task)
            async def worker():
                while not stop.is_set():
                    try:
                        task = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await run(task)
                    except BaseException:
                        stop.set()
                        raise
            results = await asyncio.gather(*(worker() for _ in range(4)), return_exceptions=True)
            errors = [r for r in results if isinstance(r, BaseException)]
            if errors:
                raise RuntimeError(str(errors[0]))
            rows = [completed[q.query_id, s] for q, s in tasks()]
            write_json(root / "responses.json", rows)
            write_json(root / "accounting.json", chat.accounting())
        except BaseException as exc:
            write_json(root / "failure.json", dict(error=str(exc), accounting=chat.accounting()))
            raise
        finally:
            heartbeat_stop.set()
            await heartbeat_task
            status("cache-filled" if (root / "responses.json").exists() else "pilot-passed" if pilot_only and (root / "api_pilot.json").exists() else "incomplete")
            write_json(root / "responses.partial.json", list(completed.values()))
            await chat.close()
            cache.close()


def audit_cache(root):
    manifest, inputs = verify(root), load_inputs(root)
    raw = sqlite_snapshot(root / "cache/responses.sqlite3")
    async def replay():
        with original.offline_guard():
            own = api.OfflineChat(raw, manifest)
            return [await elicit(q, s, own, inputs) for q, s in tasks()]
    rows = asyncio.run(replay())
    if rows != read_json(root / "responses.json") or len(rows) != 1900:
        raise ValueError("cache replay differs or incomplete")
    if {k for r in rows if r["query"]["kind"] != "ancestry" for k in r["request_keys"]} != set(raw):
        raise ValueError("unexpected or missing new requests")
    expected = {(s, q.query_id) for q in dag_query_pool(20, original.GROUPS["all"]) for s in local.STRATEGIES}
    if {(r["strategy"], r["query"]["query_id"]) for r in rows} != expected:
        raise ValueError("full claim pool is incomplete")
    connection = ResponseCache(root / "cache/responses.sqlite3")
    try:
        connection.connection.row_factory = sqlite3.Row
        attempts = [dict(r) for r in connection.connection.execute("SELECT * FROM api_attempts ORDER BY id")]
    finally:
        connection.close()
    accounting = read_json(root / "accounting.json")
    if len(attempts) != accounting["attempts"] or not math.isclose(sum(r["charged_usd"] or 0 for r in attempts), accounting["accounted_cost_usd"], abs_tol=1e-10):
        raise ValueError("accounting differs from ledger")
    uncertain = sum(r["reserved_usd"] for r in attempts if r["charged_usd"] is None)
    if (not math.isclose(uncertain, accounting["unresolved_reservations_usd"], abs_tol=1e-10)
            or accounting["budget_used_usd"] > manifest["cost_limit_usd"]):
        raise ValueError("invalid budget accounting")
    write_json(root / "cache/raw_responses.json", raw)
    write_json(root / "cache/api_attempts.json", attempts)
    gate = dict(passed=True, outcomes=1900, new_raw_requests=len(raw), new_accounting=accounting,
        manifest_sha256=sha256(root / "manifest.json"),
        artifacts={name: sha256(root / name) for name in ("responses.json", "cache/raw_responses.json", "cache/api_attempts.json", "accounting.json")})
    write_json(root / "cache_audit.json", gate)
    print("Cache audit passed: 1,900 outcomes reconstructed offline from exact prompts and raw responses.", flush=True)
    return gate


def require_cache(root):
    gate = read_json(root / "cache_audit.json")
    if not gate["passed"] or gate["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("cache gate not satisfied")
    for name, digest in gate["artifacts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"audited cache changed: {name}")
    return gate


def response_for(row, manifest=None):
    decisions = row["decisions"]
    return QueryResponse(answer=row["answer"], reported_confidence=row["reported_confidence"],
        treatment="soft_evidence", prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"],
        attempts=len(decisions), abstention_reason=row["abstention_reason"],
        audit=dict(request_keys=row["request_keys"], votes=[d["answer"] for d in decisions],
            valid_response_count=sum(d["status"] == "valid" for d in decisions),
            minimum_valid_response_count=6 if row["strategy"] == "consistency" else 1,
            source="fresh-model-cache",
            model=api.MODEL if manifest is None else manifest["settings"]["model"],
            provider=api.ENDPOINT if manifest is None else manifest["settings"]["provider"]["only"][0]))


def providers(root):
    require_cache(root)
    indexed = {(r["strategy"], r["query"]["query_id"]): r for r in read_json(root / "responses.json")}
    def make(strategy):
        def provider(query):
            row = indexed[strategy, query.query_id]
            if row["query"] != asdict(query):
                raise ValueError("query identity differs")
            return response_for(row)
        return provider
    return {s: make(s) for s in local.STRATEGIES}
