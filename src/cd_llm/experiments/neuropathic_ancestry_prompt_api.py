"""Matched old/new ancestry prompts through one pinned OpenRouter endpoint."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import time
import urllib.request
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache, _request_key
from cd_llm.experiments import neuropathic_ancestry_prompt_ablation as local
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_ancestry_prompt_api"
SOURCE = WORKSPACE_ROOT / "results/20260917T193033Z/neuropathic_ancestry_prompt_ablation"
MODEL = "qwen/qwen3.5-9b"
ENDPOINT = "deepinfra/bf16"
ENDPOINT_URL = f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints"
CONDITIONS = ("old", "new")
SETTINGS = dict(model=MODEL, provider=dict(only=[ENDPOINT], allow_fallbacks=False, require_parameters=True),
                reasoning=dict(enabled=False), temperature=0.0, max_tokens=256)


def old_prompts(raw):
    result = {}
    for strategy in local.STRATEGIES:
        indexed = {r["purpose"]: r["request"]["messages"] for r in raw[strategy].values()}
        result[strategy] = {}
        for q in local.questions():
            purposes = ([f"stated:{q.query_id}:attempt-1"] if strategy == "stated" else
                        [f"consistency:{q.query_id}:p{i // 2}:o{i % 2}" for i in range(8)])
            result[strategy][q.query_id] = [indexed[p] for p in purposes]
    return result


def render(query, strategy, condition, inputs, *, variant=0, retry=False):
    if condition == "new":
        return local.messages(query, strategy, inputs["metadata"], inputs["templates"], variant=variant, retry=retry)
    if condition != "old":
        raise ValueError("unknown prompt condition")
    messages = [dict(m) for m in inputs["old"][strategy][query.query_id][variant]]
    if retry:
        messages[-1]["content"] += inputs["templates"]["malformed_retry"]
    return messages


def request_record(messages, strategy, condition, manifest):
    return dict(backend="openrouter", condition=condition, strategy=strategy,
        body=dict(manifest["settings"], messages=[dict(m) for m in messages],
                  seed=manifest["seeds"][strategy]))


def validate_response(response, manifest=None):
    expected_model = MODEL if manifest is None else manifest["settings"]["model"]
    expected_provider = "DeepInfra" if manifest is None else manifest.get("response_provider", "DeepInfra")
    if response.get("provider") != expected_provider:
        raise RuntimeError("response did not identify the pinned provider")
    if response.get("model") != expected_model:
        raise RuntimeError("response model differs from the frozen model")
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("API returned no choices")
    message = choices[0].get("message") or {}
    if any(message.get(k) for k in ("reasoning", "reasoning_content", "reasoning_details")):
        raise RuntimeError("provider returned reasoning despite reasoning being disabled")
    details = (response.get("usage") or {}).get("completion_tokens_details") or {}
    if details.get("reasoning_tokens", 0):
        raise RuntimeError("provider billed reasoning tokens despite reasoning being disabled")
    if manifest is not None and manifest.get("require_token_usage", False):
        usage = response.get("usage") or {}
        if any(not isinstance(usage.get(k), int) or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
            raise RuntimeError("provider omitted required token accounting")


class APIChat:
    """Four concurrent requests, persistent accounting, and immutable cache keys."""

    def __init__(self, cache, manifest, *, client=None):
        self.cache, self.manifest = cache, manifest
        self.semaphore = asyncio.Semaphore(4)
        if client is None:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is required; no paid requests were made")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=120.0, max_retries=0)
        self.client = client
        cache.connection.execute("""CREATE TABLE IF NOT EXISTS api_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, request_key TEXT NOT NULL,
            reserved_usd REAL NOT NULL, charged_usd REAL, state TEXT NOT NULL)""")
        cache.connection.commit()

    def accounting(self):
        count, charged, uncertain = self.cache.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(charged_usd),0), "
            "COALESCE(SUM(CASE WHEN charged_usd IS NULL THEN reserved_usd ELSE 0 END),0) FROM api_attempts"
        ).fetchone()
        return dict(attempts=count, accounted_cost_usd=charged, unresolved_reservations_usd=uncertain,
                    budget_used_usd=charged + uncertain, limit_usd=self.manifest["cost_limit_usd"])

    def reserve(self, request):
        # A byte-count bound avoids under-reserving on unusual tokenization.
        size = sum(len(m["content"].encode("utf-8")) + 16 for m in request["body"]["messages"]) + 128
        pricing = self.manifest["pricing"]
        amount = size * float(pricing["prompt"]) + request["body"]["max_tokens"] * float(pricing["completion"])
        if self.accounting()["budget_used_usd"] + amount > self.manifest["cost_limit_usd"] + 1e-12:
            raise RuntimeError(f"the ${self.manifest['cost_limit_usd']:g} run spending limit would be exceeded")
        cursor = self.cache.connection.execute(
            "INSERT INTO api_attempts(request_key,reserved_usd,state) VALUES (?,?,'pending')", (_request_key(request), amount))
        self.cache.connection.commit()
        return cursor.lastrowid

    async def request(self, messages, *, strategy, condition, purpose):
        request = request_record(messages, strategy, condition, self.manifest)
        async with self.semaphore:
            cached = self.cache.get(request)
            if cached is not None:
                validate_response(cached["response"], self.manifest)
                return cached
            for attempt in range(1, 6):
                event = self.reserve(request)
                started = time.monotonic()
                body = request["body"]
                kwargs = {k: v for k, v in body.items() if k not in ("provider", "reasoning")}
                try:
                    value = await self.client.chat.completions.create(**kwargs,
                        extra_body=dict(provider=body["provider"], reasoning=body["reasoning"]))
                except Exception as exc:
                    self.cache.connection.execute("UPDATE api_attempts SET state='uncertain' WHERE id=?", (event,))
                    self.cache.connection.commit()
                    status = getattr(exc, "status_code", None)
                    if attempt == 5 or (status is not None and status not in (408, 409, 429) and status < 500):
                        raise RuntimeError(f"OpenRouter transport failed ({status or type(exc).__name__}); cached work is resumable") from None
                    await asyncio.sleep(min(8, 2 ** (attempt - 1)))
                    continue
                response = value.model_dump()
                usage = response.get("usage") or {}
                prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                cost = usage.get("cost")
                estimated = cost is None
                if estimated and (prompt is None or completion is None):
                    raise RuntimeError("API response lacks cost and token usage; its reservation remains charged")
                if estimated:
                    cost = prompt * float(self.manifest["pricing"]["prompt"]) + completion * float(self.manifest["pricing"]["completion"])
                if not math.isfinite(float(cost)) or float(cost) < 0:
                    raise RuntimeError("invalid API cost; its reservation remains charged")
                self.cache.connection.execute("UPDATE api_attempts SET charged_usd=?,state=? WHERE id=?",
                                              (float(cost), "estimated" if estimated else "reported", event))
                self.cache.connection.commit()
                choices = response.get("choices") or []
                content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
                self.cache.put(request, purpose, response, content, provider_request_id=response.get("id"),
                    response_model=response.get("model"), prompt_tokens=prompt, completion_tokens=completion,
                    total_tokens=usage.get("total_tokens"), cost_usd=float(cost),
                    latency_seconds=time.monotonic() - started, attempts=attempt)
                validate_response(response, self.manifest)
                if self.accounting()["budget_used_usd"] > self.manifest["cost_limit_usd"]:
                    raise RuntimeError("reported API costs exceeded the run limit; stopping")
                return self.cache.get(request)
        raise AssertionError("unreachable")

    async def close(self):
        await self.client.close()


class OfflineChat:
    def __init__(self, records, manifest):
        self.records, self.manifest = records, manifest

    async def request(self, messages, *, strategy, condition, purpose):
        request = request_record(messages, strategy, condition, self.manifest)
        row = self.records[_request_key(request)]
        if row["request"] != request or row["purpose"] != purpose:
            raise ValueError("offline request differs")
        validate_response(row["record"]["response"], self.manifest)
        return row["record"]


async def elicit(query, strategy, condition, chat, inputs):
    records, decisions = [], []
    for index in range(2 if strategy == "stated" else 8):
        messages = render(query, strategy, condition, inputs,
            variant=index if strategy == "consistency" else 0, retry=strategy == "stated" and index == 1)
        record = await chat.request(messages, strategy=strategy, condition=condition,
            purpose=f"ancestry-api:{condition}:{strategy}:{query.query_id}:{index}")
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
    return dict(query=asdict(query), strategy=strategy, condition=condition, answer=answer,
        reported_confidence=confidence, valid=answer is not None, abstention_reason=reason,
        decisions=decisions, request_keys=[r["request_key"] for r in records],
        prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in records),
        completion_tokens=sum(r.get("completion_tokens") or 0 for r in records),
        cost_usd=sum(r["cost_usd"] for r in records))


def load_inputs(root):
    return dict(metadata=read_json(root / "reference/inputs/model_metadata.json"),
                templates=read_json(root / "prompts.json"), old=read_json(root / "old_prompts.json"))


def tasks(queries=None):
    return [(q, s, c) for q in (local.questions() if queries is None else queries)
            for s in local.STRATEGIES for c in CONDITIONS]


def verify(root):
    manifest = read_json(root / "manifest.json")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    if manifest["settings"] != SETTINGS or manifest["cost_limit_usd"] != 1.0:
        raise ValueError("API configuration differs")
    if read_json(root / "prompts.json") != local.prompt_templates():
        raise ValueError("new prompt changed")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    source = local.verify(SOURCE)
    with urllib.request.urlopen(ENDPOINT_URL, timeout=30) as response:
        endpoints = json.load(response)
    selected = [e for e in endpoints["data"]["endpoints"] if e["tag"] == ENDPOINT]
    if len(selected) != 1:
        raise RuntimeError("pinned API endpoint is unavailable")
    endpoint = selected[0]
    if not {"seed", "reasoning", "temperature", "max_tokens"}.issubset(endpoint["supported_parameters"]):
        raise RuntimeError("endpoint does not support required controls")
    if float(endpoint["pricing"]["prompt"]) > 1e-7 or float(endpoint["pricing"]["completion"]) > 1.5e-7:
        raise RuntimeError("endpoint prices exceed the planned rates")
    inputs = {}
    for name, digest in source["inputs"].items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
        inputs[name] = sha256(target)
        if inputs[name] != digest:
            raise ValueError("copied input differs")
    raw = {s: read_json(root / f"reference/inputs/caches/qwen_{s}.json") for s in local.STRATEGIES}
    for name, value in (("old_prompts.json", old_prompts(raw)), ("endpoint_snapshot.json", endpoints)):
        write_json(root / name, value)
        inputs[name] = sha256(root / name)
    implementation = dict(source["implementation"])
    for name in ("code/src/cd_llm/experiments/neuropathic_ancestry_prompt_api.py",
                 "code/scripts/run_neuropathic_ancestry_prompt_api.py"):
        implementation[name] = sha256(WORKSPACE_ROOT / name)
    manifest = dict(study=STUDY, version=1, source=str(SOURCE), source_manifest_sha256=sha256(SOURCE / "manifest.json"),
        inputs=inputs, implementation=implementation, settings=SETTINGS, endpoint=endpoint,
        pricing=endpoint["pricing"], seeds={s: source["request_settings"][s]["seed"] for s in local.STRATEGIES},
        cost_limit_usd=1.0, concurrency=4, question_count=380, expected_outcomes=1520, nominal_requests=6840,
        conditions=list(CONDITIONS), strategies=list(local.STRATEGIES), minimum_valid_consistency=6,
        pilot_questions=2, reasoning_audit="Reject visible reasoning or nonzero reported reasoning tokens; missing token-detail fields are recorded.",
        cost_estimate_usd=[0.20, 0.30], endpoint_snapshot_url=ENDPOINT_URL)
    write_json(root / "manifest.json", manifest)
    print(f"Prepared {root}: 6,840 nominal API calls, estimated $0.20–$0.30; cap $1.", flush=True)
    return manifest


def pilot_check(rows):
    if len(rows) != 8:
        raise ValueError("incomplete pilot")
    for strategy in local.STRATEGIES:
        for condition in CONDITIONS:
            decisions = [d for r in rows if r["strategy"] == strategy and r["condition"] == condition for d in r["decisions"]]
            if not any(d["status"] in ("valid", "abstain") for d in decisions):
                raise RuntimeError(f"pilot produced no parseable answers for {condition}/{strategy}")


async def fill_async(root, *, pilot_only=False):
    manifest, inputs = verify(root), load_inputs(root)
    if not os.environ.get("OPENROUTER_API_KEY"):
        write_json(root / "status.json", dict(stage="awaiting-credential", paid_requests_started_this_invocation=0,
            cached_raw_responses=len(sqlite_snapshot(root / "cache/responses.sqlite3"))))
        raise RuntimeError("OPENROUTER_API_KEY is required; no paid requests were made")
    with (root / "fill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = ResponseCache(root / "cache/responses.sqlite3")
        chat = APIChat(cache, manifest)
        completed, started = {}, time.monotonic()
        def status(stage):
            write_json(root / "status.json", dict(stage=stage, completed_outcomes=len(completed),
                elapsed_seconds=time.monotonic() - started, cache=cache.summary(), accounting=chat.accounting()))
        async def run_task(task):
            row = await elicit(*task, chat, inputs)
            completed[(task[0].query_id, task[1], task[2])] = row
            if len(completed) % 20 == 0:
                status("filling")
                print(f"{len(completed)}/1520 outcomes; accounted ${chat.accounting()['budget_used_usd']:.4f}", flush=True)
            return row
        try:
            # Gather with return_exceptions so every in-flight billed call settles before closing.
            pilot = await asyncio.gather(*(run_task(t) for t in tasks(list(local.questions())[:2])), return_exceptions=True)
            errors = [r for r in pilot if isinstance(r, BaseException)]
            if errors:
                raise RuntimeError(str(errors[0]))
            pilot_check(pilot)
            write_json(root / "pilot.json", dict(passed=True, outcomes=pilot, accounting=chat.accounting()))
            print("API pilot passed: pinned provider, reasoning checks, and parsing for both prompts/methods.", flush=True)
            if pilot_only:
                status("pilot-passed")
                return
            queue = asyncio.Queue()
            for task in tasks():
                if (task[0].query_id, task[1], task[2]) not in completed:
                    queue.put_nowait(task)
            stop = asyncio.Event()
            async def worker():
                while not stop.is_set():
                    try:
                        task = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await run_task(task)
                    except BaseException:
                        stop.set()
                        raise
            workers = await asyncio.gather(*(worker() for _ in range(4)), return_exceptions=True)
            errors = [r for r in workers if isinstance(r, BaseException)]
            if errors:
                raise RuntimeError(str(errors[0]))
            outcomes = [completed[(q.query_id, s, c)] for q, s, c in tasks()]
            write_json(root / "responses.json", outcomes)
            write_json(root / "accounting.json", chat.accounting())
            status("filled")
        except BaseException as exc:
            status("incomplete")
            write_json(root / "failure.json", dict(error=str(exc), accounting=chat.accounting()))
            raise
        finally:
            write_json(root / "responses.partial.json", list(completed.values()))
            await chat.close()
            cache.close()


async def replay(root, manifest, raw):
    chat, inputs = OfflineChat(raw, manifest), load_inputs(root)
    with patch.object(socket, "socket", side_effect=AssertionError("offline audit attempted network")):
        return [await elicit(*task, chat, inputs) for task in tasks()]


def report(root):
    manifest = verify(root)
    raw = sqlite_snapshot(root / "cache/responses.sqlite3")
    rows = asyncio.run(replay(root, manifest, raw))
    if rows != read_json(root / "responses.json") or len(rows) != 1520:
        raise ValueError("offline replay differs or is incomplete")
    if {k for r in rows for k in r["request_keys"]} != set(raw):
        raise ValueError("unexpected or missing raw requests")
    labels = local.truth_labels(read_json(root / "evaluation/truth_graph.json"))
    baseline = read_json(root / "baseline.json")
    local.check_baselines(baseline, labels)
    groups = {f"api_{c}_{s}": [r for r in rows if r["strategy"] == s and r["condition"] == c]
              for s in local.STRATEGIES for c in CONDITIONS}
    groups.update({f"local_{a}": [r for r in baseline if r["arm"] == a]
                   for a in ("historical_stated", "corrected_stated", "historical_consistency")})
    summaries = [dict(arm=a, **local.metrics(r, labels)) for a, r in groups.items()]
    metrics = {r["arm"]: r for r in summaries}
    paired, comparisons = [], []
    for strategy in local.STRATEGIES:
        before = {r["query"]["query_id"]: r for r in groups[f"api_old_{strategy}"]}
        transitions = Counter()
        for row in groups[f"api_new_{strategy}"]:
            qid = row["query"]["query_id"]
            old, truth = before[qid], labels[qid]
            a, b = local.category(truth, old["answer"]), local.category(truth, row["answer"])
            transitions[f"{a} -> {b}"] += 1
            paired.append(dict(strategy=strategy, query_id=qid, truth=truth,
                old_answer=old["answer"], new_answer=row["answer"], old_confidence=old["reported_confidence"],
                new_confidence=row["reported_confidence"], before=a, after=b,
                justifications=[d["justification"] for d in row["decisions"]]))
        a, b = metrics[f"api_old_{strategy}"], metrics[f"api_new_{strategy}"]
        comparisons.append(dict(strategy=strategy, transitions=dict(sorted(transitions.items())),
            fp_decreased=b["FP"] < a["FP"], fp_change=b["FP"] - a["FP"], tp_change=b["TP"] - a["TP"],
            recall_change=b["recall"] - a["recall"], coverage_change=b["coverage"] - a["coverage"],
            eliminated_fp=sum(v for k,v in transitions.items() if k.startswith("FP -> ") and k != "FP -> FP"),
            introduced_fp=sum(v for k,v in transitions.items() if k.endswith(" -> FP") and k != "FP -> FP"),
            lost_tp=sum(v for k,v in transitions.items() if k.startswith("TP -> ") and k != "TP -> TP"),
            recovered_tp=sum(v for k,v in transitions.items() if k.endswith(" -> TP") and k != "TP -> TP")))
    accounting = read_json(root / "accounting.json")
    connection = sqlite3.connect(f"file:{root / 'cache/responses.sqlite3'}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        attempts = [dict(r) for r in connection.execute("SELECT * FROM api_attempts ORDER BY id")]
    finally:
        connection.close()
    if (len(attempts) != accounting["attempts"] or
        not math.isclose(sum(r["charged_usd"] or 0 for r in attempts), accounting["accounted_cost_usd"], abs_tol=1e-10) or
        not math.isclose(sum(r["reserved_usd"] for r in attempts if r["charged_usd"] is None), accounting["unresolved_reservations_usd"], abs_tol=1e-10)):
        raise ValueError("accounting differs from persistent attempt ledger")
    write_json(root / "cache/api_attempts.json", attempts)
    statuses = {a: dict(Counter(d["status"] for r in rr for d in r["decisions"]))
                for a, rr in groups.items() if a.startswith("api_")}
    summary = dict(metrics=summaries, comparisons=comparisons, accounting=accounting,
        prompt_tokens=sum(r["prompt_tokens"] for r in rows), completion_tokens=sum(r["completion_tokens"] for r in rows),
        cached_response_cost_usd=sum(r["cost_usd"] for r in rows), raw_requests=len(raw), raw_statuses=statuses,
        accounting_states=dict(Counter(r["state"] for r in attempts)),
        reasoning_detail_records=sum("reasoning_tokens" in ((r["record"]["response"].get("usage") or {}).get("completion_tokens_details") or {}) for r in raw.values()))
    write_json(root / "cache/raw_responses.json", raw)
    write_json(root / "summary.json", summary)
    write_csv(root / "summary.csv", summaries)
    write_json(root / "per_question.json", paired)
    write_csv(root / "per_question.csv", paired)
    text = ["# Qwen ancestry prompts: matched API comparison", "",
        "Both prompts use the same pinned DeepInfra BF16 endpoint, with reasoning disabled. Historical local results are secondary references; hosted and local runtimes need not produce identical answers.", "",
        "| Arm | TP | FN | FP | TN | Abstain (+/−) | Precision | Recall | Coverage | Accuracy | Brier | FPR/all negatives | FPR/answered negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "NA" if x is None else f"{x:.4f}"
    for r in summaries:
        text.append(f"| {r['arm']} | {r['TP']} | {r['FN']} | {r['FP']} | {r['TN']} | {r['positive_abstain']}/{r['negative_abstain']} | " +
            " | ".join(fmt(r[k]) for k in ("precision", "recall", "coverage", "accuracy", "brier", "fpr_all_negatives", "fpr_answered_negatives")) + " |")
    for c in comparisons:
        text += ["", f"{c['strategy']}: FP {'decreased' if c['fp_decreased'] else 'did not decrease'} (change {c['fp_change']:+d}); eliminated/introduced FP {c['eliminated_fp']}/{c['introduced_fp']}; lost/recovered TP {c['lost_tp']}/{c['recovered_tp']}. Recall change {c['recall_change']:+.4f}; coverage change {c['coverage_change']:+.4f}."]
    text += ["", "Recall includes positive abstentions in its denominator. Accuracy and Brier use answered questions. FPR is shown both over all negatives and over answered negatives. These are descriptive results for the fixed 380-question pool.", "",
        f"Accounted token charges: ${accounting['accounted_cost_usd']:.6f}; unresolved request reservations: ${accounting['unresolved_reservations_usd']:.6f}. Raw requests: {len(raw)}. Prompt/output tokens: {summary['prompt_tokens']}/{summary['completion_tokens']}.", "",
        "Exact old messages and frozen new templates, raw responses, provider identities, costs, justifications, and paired transitions accompany this report. All 1,520 outcomes reproduced offline. No structural inference was rerun."]
    write_text(root / "report.md", "\n".join(text) + "\n")
    write_json(root / "audit.json", dict(passed=True, offline_outcomes=1520, question_count=380,
        raw_requests=len(raw), baseline_matrices_verified=True, provider="DeepInfra", manifest_sha256=sha256(root / "manifest.json")))
    artifacts = ("responses.json", "accounting.json", "cache/raw_responses.json", "cache/api_attempts.json", "summary.json", "summary.csv", "per_question.json", "per_question.csv", "report.md", "audit.json")
    write_json(root / "completion.json", dict(complete=True, audit_passed=True,
        manifest_sha256=sha256(root / "manifest.json"), artifacts={n: sha256(root / n) for n in artifacts}))
    print(f"Completed API report: {root / 'report.md'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "fill", "report", "run"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = WORKSPACE_ROOT / "results" / run_id(args.run_id) / STUDY
    if args.stage in ("prepare", "run"):
        prepare(root)
    if args.stage in ("pilot", "fill", "run"):
        asyncio.run(fill_async(root, pilot_only=args.stage == "pilot"))
    if args.stage in ("report", "run"):
        report(root)


if __name__ == "__main__":
    main()
