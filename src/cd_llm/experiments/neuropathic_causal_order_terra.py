"""Frozen Table 2 Causal Order+PC experiment with audited Terra elicitation."""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import time
from unittest.mock import patch
import urllib.request

from cd_llm.experiments.common.artifacts import (
    json_value, read_json, run_id, sha256, write_csv, write_json, write_text,
)
from cd_llm.paths import WORKSPACE_ROOT as ROOT

STUDY = "neuropathic_causal_order_terra"
SOURCE = ROOT / "results/20260903T022032Z/neuropathic_decontamination_qwen"
TERRA_SOURCE = ROOT / "results/20260903T011254Z/neuropathic_decontamination"
MODEL = "openai/gpt-5.6-terra"
PROVIDER = "openai"
LIMIT = 25.0
NAMES = tuple(f"V{i:02d}" for i in range(1, 21))
TRIPLETS = tuple(itertools.combinations(NAMES, 3))
PAIRS = tuple(itertools.combinations(NAMES, 2))
CONDITIONS = tuple((s, n) for s in range(10) for n in (200, 1000))
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
PC_SETTINGS = dict(alpha=.01, indep_test="gsq", stable=True, uc_rule=0, uc_priority=2)
_STOP = False


class SpendingLimit(RuntimeError):
    pass


class CheckpointRequested(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def chat_messages(messages):
    result = [dict(m) for m in messages]
    if not any(m["role"] == "user" for m in result) and result[-1]["role"] == "assistant":
        result[-1]["role"] = "user"
    return result


def answer_tags(content):
    return re.findall(r"<Answer>(.*?)</Answer>", content, flags=re.DOTALL)


def graph_answer(content):
    """Match the archived graph parser, including isolated-node entries."""
    for tag in reversed(answer_tags(content)):
        try:
            values = ast.literal_eval(tag.strip())
            if not isinstance(values, list):
                continue
            return tag.strip(), [v if isinstance(v, tuple) else (v,) for v in values]
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def pair_answer(content):
    for tag in reversed(answer_tags(content)):
        if tag.strip() in ("A", "B", "C"):
            return tag.strip()
    return None


def passes_gate(parsed, total):
    return total > 0 and parsed * 100 >= 95 * total


def pilot_triplets():
    # A deterministic node-covering prefix followed by evenly spaced triplets.
    selected = []
    for start in range(0, 20, 3):
        group = tuple(sorted(NAMES[(start + k) % 20] for k in range(3)))
        if group not in selected:
            selected.append(group)
    for index in range(20):
        group = TRIPLETS[round(index * (len(TRIPLETS) - 1) / 19)]
        if group not in selected:
            selected.append(group)
        if len(selected) == 20:
            break
    assert len(selected) == 20 and set(itertools.chain.from_iterable(selected)) == set(NAMES)
    return tuple(selected)


def modules():
    from cd_llm.baselines.causal_order2025.upstream_backend import _load_upstream_modules, verified_upstream_root
    return _load_upstream_modules(verified_upstream_root())


@contextmanager
def offline():
    with patch.object(socket, "socket", side_effect=RuntimeError("offline stage attempted network access")):
        yield


@contextmanager
def quiet():
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


def snapshot_rows(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in connection.execute("SELECT * FROM responses ORDER BY request_key")]
    finally:
        connection.close()


def endpoint():
    url = f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    entries = [e for e in payload["data"]["endpoints"] if e["tag"] == PROVIDER]
    if len(entries) != 1 or entries[0]["provider_name"] != "OpenAI":
        raise ValueError("the pinned OpenAI endpoint is unavailable")
    selected = entries[0]
    if not {"reasoning", "seed", "max_tokens"} <= set(selected["supported_parameters"]):
        raise ValueError("endpoint does not support the frozen request parameters")
    if float(selected["pricing"]["prompt"]) > 2e-6 or float(selected["pricing"]["completion"]) > 12e-6:
        raise ValueError("endpoint rates exceed the approved cost estimate")
    return payload, selected


def partial(value):
    from cd_llm.core.finite_support import PartialGraph
    return PartialGraph(value["node_count"], tuple(value["marks"]), value["kind"])


def graph(value):
    from cd_llm.core.finite_support import graph_from_edges
    return graph_from_edges(20, directed=value.get("directed", ()),
                            bidirected=value.get("bidirected", ()), undirected=value.get("undirected", ()))


def verify(root):
    manifest = read_json(root / "manifest.json")
    if (manifest["model"] != MODEL or manifest["provider"] != PROVIDER or
        manifest["cost_limit_usd"] != LIMIT or manifest["pc_settings"] != PC_SETTINGS or
        manifest["conditions"] != [list(c) for c in CONDITIONS]):
        raise ValueError("frozen study settings differ")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        verify(root)
        return
    from cd_llm.experiments.neuropathic_decontamination import build_spec, audit_model_text
    from cd_llm.core.finite_support import graph_edges
    from cd_llm.baselines.causal_order2025.upstream_backend import UPSTREAM_REVISION
    import pandas as pd

    inputs, provenance = {}, {}

    def save(name, value):
        write_json(root / name, value)
        inputs[name] = sha256(root / name)

    def copy(source, name, expected=None):
        digest = sha256(source)
        if expected is not None and digest != expected:
            raise ValueError(f"source hash differs: {source}")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != digest:
            raise ValueError("snapshot hash differs")
        inputs[name] = digest
        provenance[name] = dict(path=str(source), sha256=digest)

    original = read_json(SOURCE / "benchmark/evaluation_manifest.json")
    copy(SOURCE / "benchmark/evaluation_manifest.json", "reference/qwen_manifest.json")
    benchmark = original["source_benchmark_manifest"]
    copy(Path(benchmark["path"]), "reference/benchmark_manifest.json", benchmark["sha256"])
    metadata_path = Path(benchmark["path"]).parent / "model_metadata.json"
    copy(metadata_path, "inputs/model_metadata.json")
    metadata = read_json(metadata_path)
    if tuple(n["id"] for n in metadata["nodes"]) != NAMES:
        raise ValueError("opaque variable order differs")
    spec = build_spec()
    if metadata["nodes"] != [dict(id=n, description=spec.descriptions[n]) for n in NAMES]:
        raise ValueError("clinical descriptions differ")
    save("evaluation/truth.json", graph_edges(spec.target))
    for category, folder in (("datasets", "data"), ("source_statistics", "statistics")):
        if sorted((e["seed"], e["sample_size"]) for e in original[category]) != list(CONDITIONS):
            raise ValueError("Table 2 conditions differ")
        for entry in original[category]:
            extension = "csv" if category == "datasets" else "json"
            name = f"inputs/{folder}/seed_{entry['seed']:02d}_n{entry['sample_size']}.{extension}"
            copy(Path(entry["path"]), name, entry["sha256"])
    for seed in range(10):
        small = pd.read_csv(root / f"inputs/data/seed_{seed:02d}_n200.csv")
        large = pd.read_csv(root / f"inputs/data/seed_{seed:02d}_n1000.csv")
        if tuple(small.columns) != NAMES or not small.equals(large.iloc[:200].reset_index(drop=True)):
            raise ValueError("nested observations differ")

    rows = snapshot_rows(SOURCE / "caches/causal_order.sqlite3")
    save("reference/qwen_responses.json", rows)
    copy(SOURCE / "shared/causal_order_expert.json", "reference/qwen_expert.json")
    context = "to model " + metadata["domain_context"] + ". Variable metadata: " + "; ".join(
        f"{n['id']} means {n['description']}" for n in metadata["nodes"])
    mod = modules()
    triples = {fingerprint(chat_messages(mod.triplet.generate_subgraph_prompt(list(g), context))): g for g in TRIPLETS}
    pairs = {fingerprint(chat_messages(mod.triplet.cot_tiebreaker_prompt(a, b, list(NAMES), context))): (a, b)
             for a, b in PAIRS}
    archived = {}
    for row in rows:
        request = json.loads(row["request_json"])
        messages = request["messages"]
        signature = fingerprint(messages)
        allowed = triples if row["purpose"].startswith("triplet") else pairs
        if signature not in allowed:
            raise ValueError("saved Table 2 prompt does not match its reconstructed template")
        audit_model_text("\n".join(m["content"] for m in messages), spec)
        if row["purpose"] == "triplet":
            group = allowed[signature]
            archived[group] = dict(nodes=list(group), messages=messages)
    if set(archived) != set(TRIPLETS):
        raise ValueError("source does not contain exactly 1,140 triplets")
    save("prompts.json", dict(context=context, triplets=[archived[g] for g in TRIPLETS],
        pairs=[dict(nodes=[a, b], messages=chat_messages(mod.triplet.cot_tiebreaker_prompt(a, b, list(NAMES), context)))
               for a, b in PAIRS]))
    for seed, n in CONDITIONS:
        name = f"seed_{seed:02d}_n{n}.json"
        copy(SOURCE / "records/causal_orderpluspc_plus_qwen3.5-9b" / name, "reference/qwen_records/" + name)
        for folder, label in (
            ("gpc-guess_plus_gpt-5.6_terra", "gpc"),
            ("bfspluspearson_plus_gpt-5.6_terra", "bfs"),
            ("claim-fuse-smc-active_plus_gpt-5.6_terra", "smc_stated"),
            ("claim-fuse-smc-active_plus_gpt-5.6_terra_(semantic_consistency)", "smc_consistency"),
        ):
            copy(TERRA_SOURCE / "records" / folder / name, f"reference/terra/{label}/{name}")
    payload, selected = endpoint()
    save("provider_endpoints.json", payload)
    save("provenance.json", provenance)
    dated = selected.get("name", "").split(" | ")[-1]
    implementation = [Path(__file__).resolve(), ROOT / "sbatch/neuropathic_causal_order_terra.sbatch",
                      ROOT / "code/tests/unit/experiments/test_neuropathic_causal_order_terra.py"]
    implementation += list((ROOT / "code/src/cd_llm/baselines/causal_order2025").glob("*.py"))
    implementation += [ROOT / "code/src/cd_llm/core/finite_support.py", ROOT / "code/src/cd_llm/core/cpc.py"]
    write_json(root / "manifest.json", dict(study=STUDY, version=1, model=MODEL, provider=PROVIDER,
        response_models=sorted({MODEL, dated}), response_provider="OpenAI", upstream_revision=UPSTREAM_REVISION,
        source=str(SOURCE), cost_limit_usd=LIMIT, conditions=CONDITIONS, pc_settings=PC_SETTINGS,
        request_seed=0, reasoning="none", temperature_supplied=False, concurrency=4,
        triplet_limit=4096, triplet_retry_limit=7680, tie_limit=2048, fallback_limit=256,
        parse_gate=.95, triplet_count=1140, pricing=selected["pricing"], pilot_triplets=pilot_triplets(),
        inputs=inputs, implementation={str(p.relative_to(ROOT)):sha256(p) for p in implementation},
        metric_scale="unnormalized", parser="typed_graph_or_ABC_v1", reference_metric_scale="normalized"))
    write_json(root / "status.json", dict(stage="prepared", triplets=0, expected_triplets=1140))
    verify(root)


def request_body(messages, limit):
    return dict(model=MODEL, messages=chat_messages(messages), max_tokens=limit, seed=0,
        provider=dict(only=[PROVIDER], allow_fallbacks=False, require_parameters=True),
        reasoning=dict(effort="none"))


def validate_response(value, manifest):
    if value.get("provider") != "OpenAI" or value.get("model") not in manifest["response_models"]:
        raise ValueError("response does not identify the pinned Terra model and OpenAI provider")
    choices = value.get("choices") or []
    if len(choices) != 1:
        raise ValueError("response must contain one choice")
    message = choices[0].get("message") or {}
    usage = value.get("usage") or {}
    if any(message.get(k) for k in ("reasoning", "reasoning_content", "reasoning_details")) or (
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)):
        raise ValueError("reasoning was returned or billed despite being disabled")
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
        raise ValueError("response omits valid token accounting")


class Transport:
    """One collector process; reservations are durable before network I/O."""
    def __init__(self, root, manifest, *, client=None, deadline=None):
        from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
        self.root, self.manifest, self.deadline = root, manifest, deadline
        self.cache = ResponseCache(root / "cache/responses.sqlite3")
        self.cache.connection.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, request_key TEXT NOT NULL, reserved_usd REAL NOT NULL,
            charged_usd REAL, state TEXT NOT NULL, created REAL NOT NULL)""")
        self.cache.connection.commit()
        if client is None:
            from openai import AsyncOpenAI
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is missing")
            client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=120.)
        self.client = client
        self.semaphore = asyncio.Semaphore(4)
        self.key_locks = defaultdict(asyncio.Lock)
        self.started = time.monotonic()
        self.initial_count = self.cache.summary()["request_count"]
        self.halted = None

    def accounting(self):
        count, charged, unresolved = self.cache.connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(charged_usd),0),"
            "COALESCE(SUM(CASE WHEN charged_usd IS NULL THEN reserved_usd ELSE 0 END),0) FROM attempts").fetchone()
        return dict(attempts=count, accounted_cost_usd=charged, unresolved_reservations_usd=unresolved,
                    budget_used_usd=charged + unresolved, limit_usd=LIMIT)

    def reserve(self, body):
        pricing = self.manifest["pricing"]
        # UTF-8 bytes plus framing bound input tokens; include cache-write rates.
        size = sum(len(m["content"].encode()) + 32 for m in body["messages"]) + 256
        rate = max(float(pricing.get(k, 0)) for k in ("prompt", "input_cache_write", "input_cache_read"))
        amount = size * rate + body["max_tokens"] * float(pricing["completion"])
        connection = self.cache.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            if self.accounting()["budget_used_usd"] + amount > LIMIT + 1e-12:
                raise SpendingLimit("$25 spending limit would be exceeded; cached work is resumable")
            event = connection.execute("INSERT INTO attempts(request_key,reserved_usd,state,created) VALUES (?,?,?,?)",
                (fingerprint(body), amount, "pending", time.time())).lastrowid
            connection.commit()
            return event
        except BaseException:
            connection.rollback()
            raise

    def progress(self, stage="collecting"):
        counts = dict(self.cache.connection.execute("SELECT purpose,COUNT(*) FROM responses GROUP BY purpose"))
        total = self.cache.summary()
        elapsed = time.monotonic() - self.started
        rate = (total["request_count"] - self.initial_count) / elapsed if elapsed > 0 else 0
        write_json(self.root / "status.json", dict(stage=stage, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            triplets=counts.get("triplet", 0), expected_triplets=1140, parse_retries=counts.get("triplet_parse_retry", 0),
            tie_breaks=counts.get("vote_tiebreaker", 0), fallbacks=counts.get("algorithm1_fallback", 0),
            requests_per_minute=60*rate, triplet_eta_seconds=(1140-counts.get("triplet", 0))/rate if rate else None,
            eta_excludes_remaining_repairs_and_ties=True, cache=total, accounting=self.accounting()))

    async def request(self, messages, limit, purpose):
        body = request_body(messages, limit)
        async with self.key_locks[fingerprint(body)], self.semaphore:
            existing = self.cache.get(body)
            if existing is not None:
                validate_response(existing["response"], self.manifest)
                return existing
            if self.halted:
                raise self.halted
            if _STOP or (self.deadline and time.monotonic() >= self.deadline):
                self.halted = CheckpointRequested("collector checkpoint requested")
                raise self.halted
            for attempt in range(1, 6):
                event = self.reserve(body)
                start = time.monotonic()
                kwargs = {k:v for k,v in body.items() if k not in ("provider", "reasoning")}
                try:
                    response = await self.client.chat.completions.create(**kwargs,
                        extra_body={k:body[k] for k in ("provider", "reasoning")})
                except Exception as error:
                    self.cache.connection.execute("UPDATE attempts SET state='uncertain' WHERE id=?", (event,))
                    self.cache.connection.commit()
                    status = getattr(error, "status_code", None)
                    if attempt == 5 or (status is not None and status not in (408,409,429) and status < 500):
                        raise RuntimeError(f"API request failed ({status or type(error).__name__}); reservation retained") from None
                    if _STOP or (self.deadline and time.monotonic() >= self.deadline):
                        raise CheckpointRequested("checkpoint after interrupted request")
                    await asyncio.sleep(min(8, 2 ** (attempt-1)))
                    continue
                value = response.model_dump()
                usage = value.get("usage") or {}
                charged = usage.get("cost")
                state = "reported"
                if charged is None:
                    if any(type(usage.get(k)) is not int for k in ("prompt_tokens", "completion_tokens")):
                        raise ValueError("missing accounting; pending reservation retained")
                    pricing = self.manifest["pricing"]
                    charged = usage["prompt_tokens"] * max(float(pricing[k]) for k in ("prompt", "input_cache_write")) + usage["completion_tokens"] * float(pricing["completion"])
                    state = "estimated"
                if not math.isfinite(float(charged)) or charged < 0:
                    raise ValueError("invalid cost; pending reservation retained")
                self.cache.connection.execute("UPDATE attempts SET charged_usd=?,state=? WHERE id=?", (charged,state,event))
                content = (((value.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                self.cache.put(body, purpose, value, content, provider_request_id=value.get("id"), response_model=value.get("model"),
                    prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"), cost_usd=float(charged), latency_seconds=time.monotonic()-start, attempts=attempt)
                validate_response(value, self.manifest)
                if self.accounting()["budget_used_usd"] > LIMIT:
                    raise SpendingLimit("reported costs exceed the cap; stop all requests")
                self.progress()
                return self.cache.get(body)
        raise AssertionError("unreachable")

    async def many(self, tasks):
        # A bounded worker set avoids reserving money for the entire queued bank.
        iterator = iter(enumerate(tasks))
        result = [None] * len(tasks)
        async def worker():
            for index, task in iterator:
                if self.halted:
                    break
                try:
                    result[index] = await self.request(*task)
                except Exception as error:
                    self.halted = self.halted or error
                    break
        await asyncio.gather(*(worker() for _ in range(4)))
        if self.halted:
            raise self.halted
        return result

    async def close(self):
        self.cache.close()
        if hasattr(self.client, "close"):
            await self.client.close()


def selected_rows(rows, prompts):
    indexed = {}
    for row in rows:
        request = json.loads(row["request_json"])
        indexed[fingerprint(request["messages"]), request.get("max_tokens", request.get("max_completion_tokens"))] = row
    selected = []
    for item in prompts["triplets"]:
        key = fingerprint(item["messages"])
        row = indexed.get((key, 7680), indexed.get((key, 4096)))
        if row is None:
            raise ValueError("triplet cache is incomplete")
        selected.append(row)
    return selected, indexed


def discover_ties(prompts, selected):
    mod = modules()
    parsed = [graph_answer(r["content"]) for r in selected]
    if not passes_gate(sum(p is not None for p in parsed), 1140):
        raise ValueError("95% triplet parsing gate failed")
    ties = []
    def record(a, b, *_args, **_kwargs):
        ties.append((a, b))
        return "No_Conn"
    with patch.object(mod.triplet, "_tiebreaker_query", record), quiet():
        mod.triplet.merge_triplet_votes([tuple(i["nodes"]) for i in prompts["triplets"]],
            [p[1] if p else [] for p in parsed], list(NAMES), prompts["context"], expert_model=MODEL, delay=0)
    return ties


def replay_expert(prompts, rows, *, legacy=False, model=MODEL):
    """Run pinned aggregation with cache-only typed answers, no provider calls."""
    from cd_llm.baselines.causal_order2025.upstream_backend import query_triplet_order
    from cd_llm.baselines.causal_order2025.types import TripletConfig
    selected, indexed = selected_rows(rows, prompts)
    mod = modules()
    def answer(messages, model, max_completion_tokens):
        signature = fingerprint(chat_messages(messages))
        if max_completion_tokens == 300:
            row = indexed.get((signature, 7680), indexed.get((signature, 4096)))
            if row is None:
                raise ValueError("uncached triplet requested by upstream")
            parsed = graph_answer(row["content"])
            return "<Answer>" + (parsed[0] if parsed else "[]") + "</Answer>"
        if max_completion_tokens != 400 or (signature, 2048) not in indexed:
            raise ValueError("uncached tie-break requested by upstream")
        label = pair_answer(indexed[signature, 2048]["content"])
        if label is None and not legacy:
            raise ValueError("malformed tie-break response")
        return "<Answer>" + ("C" if legacy else label) + "</Answer>"
    with quiet(), patch.object(mod.triplet, "query_llm", answer), patch.object(
        mod.triplet, "parse_answer_tag", lambda text: answer_tags(text)[-1].strip()):
        return query_triplet_order(NAMES, prompts["context"], descriptions=None,
            config=TripletConfig(subgraph_model=model, expert_model=model, delay=0))


def expert_record(expert):
    from cd_llm.experiments.neuropathic_causal_order_subnetwork import _expert_record
    return json_value(_expert_record(expert))


def pcs(root):
    return [partial(read_json(root / f"inputs/statistics/seed_{s:02d}_n{n}.json")["pc"]) for s,n in CONDITIONS]


def integrate_expert(root, expert, fallback_answers):
    from cd_llm.baselines.causal_order2025 import integrate_order_into_cpdag, CausalOrderInfeasibleError
    from cd_llm.core.finite_support import graph_edges, point_prediction_metrics
    truth = graph(read_json(root / "evaluation/truth.json"))
    def fallback(a, b):
        label = fallback_answers.get((min(a,b), max(a,b)))
        if label not in ("A", "B"):
            raise CausalOrderInfeasibleError("missing, malformed, or No_Conn PC fallback")
        return (a,b) if label == "A" else (b,a)
    records = []
    for (seed,n), cpdag in zip(CONDITIONS, pcs(root), strict=True):
        item = dict(seed=seed, sample_size=n, status="complete")
        try:
            result = integrate_order_into_cpdag(cpdag, expert, fallback_orienter=fallback)
            item.update(graph=graph_edges(result.graph), metrics=point_prediction_metrics(result.graph,truth),
                        fallback_orientations=result.fallback_orientations)
        except CausalOrderInfeasibleError as error:
            item.update(status="infeasible", reason=str(error))
        records.append(json_value(item))
    return records


def reference_audit(root):
    verify(root)
    from cd_llm.baselines.causal_order2025.api import _run_pc
    from cd_llm.baselines.causal_order2025 import PCOrderConfig
    from cd_llm.baselines.causal_order2025.integration import causal_learn_cpdag
    import pandas as pd
    with offline():
        prompts = read_json(root / "prompts.json")
        rows = read_json(root / "reference/qwen_responses.json")
        old = replay_expert(prompts, rows, legacy=True, model="Qwen/Qwen3.5-9B")
        expected = read_json(root / "reference/qwen_expert.json")
        actual = expert_record(old)
        for name in ("raw_edges", "expert_order", "graph", "edgewise_distributions"):
            if actual[name] != expected[name]:
                raise ValueError(f"archived expert replay differs: {name}")
        legacy_records = integrate_expert(root, old, {})
        for record in legacy_records:
            archived = read_json(root / f"reference/qwen_records/seed_{record['seed']:02d}_n{record['sample_size']}.json")
            if record["status"] != "complete" or record["graph"] != archived["graph"]:
                raise ValueError("archived Qwen graph does not reproduce with frozen PC")
            for metric in METRICS:
                expected_value = archived["metrics"][metric] * (190 if metric in METRICS[:2] else 1)
                if not math.isclose(record["metrics"][metric], expected_value, abs_tol=1e-10):
                    raise ValueError("archived Qwen metrics do not reproduce")
        for n in (200,1000):
            data = pd.read_csv(root / f"inputs/data/seed_00_n{n}.csv").to_numpy()
            estimated = causal_learn_cpdag(_run_pc(data,NAMES,PCOrderConfig(**PC_SETTINGS),None))
            if estimated != partial(read_json(root / f"inputs/statistics/seed_00_n{n}.json")["pc"]):
                raise ValueError("PC settings do not reproduce saved seed-zero CPDAG")
        corrected = replay_expert(prompts, rows, legacy=False, model="Qwen/Qwen3.5-9B")
        corrected_records = integrate_expert(root, corrected, {})
    write_json(root / "reference_replay.json", dict(passed=True, historical_records=legacy_records,
        corrected_records=corrected_records, corrected_expert=expert_record(corrected),
        changed_graphs=sum(a.get("graph") != b.get("graph") for a,b in zip(legacy_records,corrected_records)),
        manifest_sha256=sha256(root / "manifest.json")))


async def collection(root, *, pilot_only=False):
    from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
    manifest = verify(root)
    regression = read_json(root / "reference_replay.json")
    if not regression["passed"] or regression["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("offline regression gate is missing")
    _, selected_endpoint = endpoint()
    if selected_endpoint["pricing"] != manifest["pricing"]:
        raise ValueError("provider pricing changed since preparation")
    prompts = read_json(root / "prompts.json")
    triples = {tuple(p["nodes"]):p["messages"] for p in prompts["triplets"]}
    pairs = {tuple(p["nodes"]):p["messages"] for p in prompts["pairs"]}
    # 10-minute checkpoint margin includes outstanding 120-second requests.
    deadline = time.monotonic() + float(os.environ.get("ORDER_COLLECT_SECONDS", "13800"))
    transport = Transport(root,manifest,deadline=deadline)
    try:
        if not (root / "pilot.json").exists():
            pilot = pilot_triplets()
            responses = await transport.many([(triples[g],4096,"triplet") for g in pilot])
            for index, response in enumerate(responses):
                if graph_answer(response["content"]) is None:
                    responses[index] = await transport.request(triples[pilot[index]],7680,"triplet_parse_retry")
            parsed = sum(graph_answer(r["content"]) is not None for r in responses)
            # The archived tie pair can be reused if Terra also needs this pair.
            pair = await transport.request(pairs[("V06","V07")],2048,"vote_tiebreaker")
            if not passes_gate(parsed,20) or pair_answer(pair["content"]) is None:
                raise ValueError("pilot parsing gate failed")
            write_json(root / "pilot.json", dict(passed=True,triplets=20,parsed=parsed,pair_label=pair_answer(pair["content"]),
                manifest_sha256=sha256(root / "manifest.json"),accounting=transport.accounting(),
                projected_triplet_cost_usd=sum(r["cost_usd"] for r in snapshot_rows(root / "cache/responses.sqlite3")
                    if r["purpose"].startswith("triplet"))/20*1140))
        elif (not read_json(root / "pilot.json")["passed"] or
              read_json(root / "pilot.json")["manifest_sha256"] != sha256(root / "manifest.json")):
            raise ValueError("pilot manifest differs")
        if pilot_only:
            transport.progress("pilot_complete")
            return
        responses = await transport.many([(triples[g],4096,"triplet") for g in TRIPLETS])
        retries = [i for i,r in enumerate(responses) if graph_answer(r["content"]) is None]
        await transport.many([(triples[TRIPLETS[i]],7680,"triplet_parse_retry") for i in retries])
        rows = snapshot_rows(root / "cache/responses.sqlite3")
        chosen, _ = selected_rows(rows,prompts)
        with offline():
            ties = discover_ties(prompts,chosen)
        tie_responses = await transport.many([(pairs[p],2048,"vote_tiebreaker") for p in ties])
        if any(pair_answer(r["content"]) is None for r in tie_responses):
            raise ValueError("a required tie-break response is malformed")
        rows = snapshot_rows(root / "cache/responses.sqlite3")
        with offline():
            expert = replay_expert(prompts,rows)
            fallbacks = required_fallback_pairs(expert,pcs(root))
        await transport.many([(pairs[(NAMES[a],NAMES[b])],256,"algorithm1_fallback") for a,b in fallbacks])
        write_json(root / "collection.json", dict(complete=True, triplets=1140, retries=len(retries),ties=ties,
            fallbacks=fallbacks,expert=expert_record(expert),manifest_sha256=sha256(root / "manifest.json")))
        transport.progress("collection_complete")
    finally:
        write_json(root / "accounting.json", transport.accounting())
        await transport.close()


def cache_audit(root):
    from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
    manifest = verify(root)
    collected = read_json(root / "collection.json")
    if not collected["complete"] or collected["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("collection is incomplete or stale")
    prompts = read_json(root / "prompts.json")
    rows = snapshot_rows(root / "cache/responses.sqlite3")
    with offline():
        selected, indexed = selected_rows(rows,prompts)
        ties = discover_ties(prompts,selected)
        expert = replay_expert(prompts,rows)
        expected_fallbacks = required_fallback_pairs(expert,pcs(root))
        if [list(p) for p in expected_fallbacks] != collected["fallbacks"]:
            raise ValueError("required PC fallback set differs")
        actual = expert_record(expert)
        for name in ("raw_edges","expert_order","graph","edgewise_distributions"):
            if actual[name] != collected["expert"][name]:
                raise ValueError("offline expert replay differs")
    if [list(t) for t in ties] != collected["ties"]:
        raise ValueError("required tie set differs")
    required = set()
    for p in prompts["triplets"]:
        signature = fingerprint(p["messages"])
        original = indexed.get((signature,4096))
        if original is None:
            raise ValueError("missing initial triplet response")
        required.add(original["request_key"])
        if graph_answer(original["content"]) is None:
            retry = indexed.get((signature,7680))
            if retry is None:
                raise ValueError("required parsing retry is missing")
            required.add(retry["request_key"])
    pair_prompts = {tuple(p["nodes"]):p["messages"] for p in prompts["pairs"]}
    for a,b in ties:
        row = indexed.get((fingerprint(pair_prompts[a,b]),2048))
        if row is None or pair_answer(row["content"]) is None:
            raise ValueError("required tie-break missing or malformed")
        required.add(row["request_key"])
    common_required = set(required)
    for a,b in collected["fallbacks"]:
        row = indexed.get((fingerprint(pair_prompts[NAMES[a],NAMES[b]]),256))
        if row is None:
            raise ValueError("required fallback is missing")
        required.add(row["request_key"])
    triplet_signatures = {fingerprint(p["messages"]) for p in prompts["triplets"]}
    pair_signatures = {fingerprint(p["messages"]) for p in prompts["pairs"]}
    for row in rows:
        value = json.loads(row["response_json"])
        validate_response(value,manifest)
        request = json.loads(row["request_json"])
        if request != request_body(request["messages"],request["max_tokens"]):
            raise ValueError("cached generation settings differ")
        signatures = triplet_signatures if request["max_tokens"] in (4096,7680) else pair_signatures
        expected_purpose = {4096:"triplet",7680:"triplet_parse_retry",2048:"vote_tiebreaker",256:"algorithm1_fallback"}.get(request["max_tokens"])
        if fingerprint(request["messages"]) not in signatures or row["purpose"] != expected_purpose:
            raise ValueError("cached response does not belong to the frozen question bank")
        if (row["prompt_tokens"],row["completion_tokens"]) != (value["usage"]["prompt_tokens"],value["usage"]["completion_tokens"]):
            raise ValueError("cached token usage differs")
    connection = sqlite3.connect(f"file:{root / 'cache/responses.sqlite3'}?mode=ro",uri=True)
    connection.row_factory = sqlite3.Row
    attempts = [dict(r) for r in connection.execute("SELECT * FROM attempts ORDER BY id")]
    connection.close()
    accounting = read_json(root / "accounting.json")
    charged = sum(a["charged_usd"] or 0 for a in attempts)
    unresolved = sum(a["reserved_usd"] for a in attempts if a["charged_usd"] is None)
    if len(attempts) != accounting["attempts"] or not math.isclose(charged+unresolved,accounting["budget_used_usd"],abs_tol=1e-9) or charged+unresolved > LIMIT:
        raise ValueError("spending ledger audit failed")
    if not math.isclose(sum(r["cost_usd"] for r in rows),charged,abs_tol=1e-9):
        raise ValueError("successful response charges do not match the attempt ledger")
    used = [r for r in rows if r["request_key"] in required]
    write_json(root / "cache/raw_responses.json",rows)
    write_json(root / "cache/attempts.json",attempts)
    write_json(root / "audit.json",dict(passed=True, manifest_sha256=sha256(root / "manifest.json"),
        triplets=1140,parseable=sum(graph_answer(r["content"]) is not None for r in selected),
        required_request_keys=sorted(required),required_requests=len(used),pilot_only_requests=len(rows)-len(used),
        shared_expert_request_keys=sorted(common_required),
        input_tokens=sum(r["prompt_tokens"] for r in used), output_tokens=sum(r["completion_tokens"] for r in used),
        response_sha256=sha256(root / "cache/raw_responses.json"),accounting=accounting,
        token_expenditure_complete=unresolved==0))


def integrate(root):
    from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
    verify(root)
    audit = read_json(root / "audit.json")
    if not audit["passed"] or sha256(root / "cache/raw_responses.json") != audit["response_sha256"]:
        raise ValueError("cache audit is missing or stale")
    prompts = read_json(root / "prompts.json")
    rows = read_json(root / "cache/raw_responses.json")
    collected = read_json(root / "collection.json")
    with offline():
        expert = replay_expert(prompts,rows)
        _, indexed = selected_rows(rows,prompts)
        pair_prompts = {tuple(p["nodes"]):p["messages"] for p in prompts["pairs"]}
        fallbacks = {(a,b):pair_answer(indexed[fingerprint(pair_prompts[NAMES[a],NAMES[b]]),256]["content"])
                     for a,b in collected["fallbacks"]}
        records = integrate_expert(root,expert,fallbacks)
        if records != integrate_expert(root,expert,fallbacks):
            raise ValueError("integration is not deterministic")
        for record, cpdag in zip(records,pcs(root),strict=True):
            needed = set(audit["shared_expert_request_keys"])
            for a,b in required_fallback_pairs(expert,[cpdag]):
                needed.add(indexed[fingerprint(pair_prompts[NAMES[a],NAMES[b]]),256]["request_key"])
            used = [r for r in rows if r["request_key"] in needed]
            record.update(model=MODEL,method="Causal Order+PC",manifest_sha256=sha256(root / "manifest.json"),
                input_tokens=sum(r["prompt_tokens"] for r in used),output_tokens=sum(r["completion_tokens"] for r in used),
                required_requests=len(used),required_request_keys=sorted(needed))
            write_json(root / f"records/seed_{record['seed']:02d}_n{record['sample_size']}.json",record)


def report(root):
    verify(root)
    from cd_llm.core.finite_support import point_prediction_metrics
    from cd_llm.metrics.units import convert_metrics, metric_metadata
    records, baseline = [], []
    with offline():
        truth = graph(read_json(root / "evaluation/truth.json"))
        for seed,n in CONDITIONS:
            name = f"seed_{seed:02d}_n{n}.json"
            item = read_json(root / "records" / name)
            if item["status"] == "complete" and item["metrics"] != point_prediction_metrics(graph(item["graph"]),truth):
                raise ValueError("independent scoring differs")
            records.append(dict(method="Causal Order+PC",seed=seed,sample_size=n,status=item["status"],
                **{k:item.get("metrics",{}).get(k) for k in METRICS},input_tokens=item["input_tokens"],
                output_tokens=item["output_tokens"],total_tokens=item["input_tokens"]+item["output_tokens"],
                required_requests=item["required_requests"],reason=item.get("reason","")))
            stats = read_json(root / "inputs/statistics" / name)
            for label,key in (("PC","pc"),("GES","ges")):
                metrics = point_prediction_metrics(partial(stats[key]),truth)
                baseline.append(dict(method=label,seed=seed,sample_size=n,**{k:metrics[k] for k in METRICS}))
            for folder,label in (("gpc","gPC-Guess + Terra"),("bfs","BFS+Pearson + Terra"),
                                 ("smc_stated","SMC-Active stated + Terra"),("smc_consistency","SMC-Active consistency + Terra")):
                original = read_json(root / f"reference/terra/{folder}" / name)
                metrics = convert_metrics(original["metrics"],190,source_scale="normalized")
                baseline.append(dict(method=label,seed=seed,sample_size=n,**{k:metrics[k] for k in METRICS}))
    grouped = defaultdict(list)
    for row in records + baseline:
        grouped[row["method"],row["sample_size"]].append(row)
    summary = []
    for (method,n), group in sorted(grouped.items()):
        if sorted(g["seed"] for g in group) != list(range(10)):
            raise ValueError("ten paired seeds are required")
        failures = sum(g.get("status","complete") != "complete" for g in group)
        row = dict(method=method,sample_size=n,seeds=10,infeasible=failures)
        for metric in METRICS:
            values = [g[metric] for g in group]
            row[metric+"_mean"] = statistics.mean(values) if not failures else None
            row[metric+"_se"] = statistics.stdev(values)/math.sqrt(10) if not failures else None
        summary.append(row)
    write_csv(root / "per_run.csv", records)
    write_csv(root / "comparison_per_run.csv",baseline)
    write_csv(root / "summary.csv",summary)
    write_json(root / "summary.json",summary)
    for name in ("per_run.csv","comparison_per_run.csv","summary.csv"):
        write_json(root / (name+".metrics.json"),metric_metadata(20))
    audit = read_json(root / "audit.json")
    replay = read_json(root / "reference_replay.json")
    lines = ["# Terra Causal Order+PC on the Table 2 benchmark", "",
        "One shared exhaustive 1,140-triplet collection; n=200/1000 and ten paired data seeds. ED and ES are unnormalized. Standard errors are conditional on this shared expert collection.", "",
        f"Required input/output tokens across the collected bank: {audit['input_tokens']:,}/{audit['output_tokens']:,}; requests: {audit['required_requests']}. Each dataset is charged the full shared expert plus its own required PC fallbacks; per_run.csv gives its totals. API purchases occur once.",
        f"Accounted API spending: ${audit['accounting']['accounted_cost_usd']:.6f}; unresolved reservations: ${audit['accounting']['unresolved_reservations_usd']:.6f}; limit: $25. Pilot-only responses: {audit['pilot_only_requests']}.",
        "Token counts cover recorded responses; any requests with unresolved reservations have unknown additional token usage." if not audit['token_expenditure_complete'] else "Token accounting covers all recorded attempts with no unresolved reservations.",
        f"Parsed triplets: {audit['parseable']}/1140. Corrected A/B/C tie parsing changes {replay['changed_graphs']}/20 historical Qwen graphs; the historical records and correction diagnostic are in reference_replay.json.", "",
        "| Method | n | ED (SE) | ES (SE) | Adj. F1 (SE) | Orient. F1 (SE) |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        values = ["infeasible" if row["infeasible"] else f"{row[k+'_mean']:.4f} ({row[k+'_se']:.4f})" for k in METRICS]
        lines.append(f"| {row['method']} | {row['sample_size']} | " + " | ".join(values) + " |")
    lines += ["", "Prompt messages match the archived Table 2 Causal Order prompts, including the full glossary. Terra uses the pinned OpenAI endpoint, reasoning disabled, and no temperature parameter. Triplet graph responses and A/B/C tie responses use separate parsers. Native infeasibility is retained without graph repair or complete-case aggregation."]
    write_text(root / "report.md","\n".join(lines)+"\n")
    own = [r for r in summary if r["method"] == "Causal Order+PC"]
    if not any(r["infeasible"] for r in own):
        cells = [f"{r[k+'_mean']:.2f} ({r[k+'_se']:.2f})" if k == "expected_distance" else f"{r[k+'_mean']:.3f} ({r[k+'_se']:.3f})" for r in own for k in ("expected_distance","orientation_f1")]
        write_text(root / "table2_row.tex","Causal Order+PC & " + " & ".join(cells) + " \\\\\n")
    artifacts = ["audit.json","reference_replay.json","collection.json","accounting.json","summary.json","summary.csv","per_run.csv","comparison_per_run.csv","report.md","cache/raw_responses.json","cache/attempts.json"]
    artifacts += [name+".metrics.json" for name in ("per_run.csv","comparison_per_run.csv","summary.csv")]
    if (root / "table2_row.tex").exists():
        artifacts.append("table2_row.tex")
    artifacts += [f"records/seed_{s:02d}_n{n}.json" for s,n in CONDITIONS]
    write_json(root / "completion.json",dict(complete=True,audit_passed=True,conditions=20,
        feasible_conditions=sum(r["status"]=="complete" for r in records),triplets=1140,
        manifest_sha256=sha256(root / "manifest.json"),artifacts={p:sha256(root/p) for p in artifacts}))
    write_json(root / "status.json",dict(stage="complete",conditions=20,triplets=1140,accounting=audit["accounting"],
        feasible_conditions=sum(r["status"]=="complete" for r in records)))


def submit_stage(root, stage, *, dependency=None):
    stamp = root.parent.name
    logs = ROOT / "out" / stamp / STUDY
    logs.mkdir(parents=True,exist_ok=True)
    command = ["sbatch","--parsable",f"--job-name=np-order-{stage}",
        "--time=04:00:00" if stage=="collect" else "--time=00:20:00",
        f"--output={logs}/{stage}-%j.out",f"--error={logs}/{stage}-%j.err",
        f"--export=ALL,ORDER_RUN_ID={stamp},ORDER_STAGE={stage}"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(ROOT / "sbatch/neuropathic_causal_order_terra.sbatch"))
    environment = dict(os.environ)
    if stage not in ("collect","prepare"):
        environment.pop("OPENROUTER_API_KEY",None)
    job = subprocess.check_output(command,text=True,env=environment,timeout=60).strip().split(";")[0]
    write_json(root / f"jobs/{stage}_{job}.json",dict(job_id=job,stage=stage,dependency=dependency,submitted=time.time()))
    print(f"Submitted {stage}: {job}",flush=True)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","pilot","collect","audit","integrate","report","submit","finish"))
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--chain",action="store_true")
    args = parser.parse_args(argv)
    root = ROOT / "results" / run_id(args.run_id) / STUDY
    root.mkdir(parents=True,exist_ok=True)
    global _STOP
    signal.signal(signal.SIGUSR1,lambda *_: globals().__setitem__("_STOP",True))
    try:
        if args.stage == "submit":
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY must be exported before submission")
            submit_stage(root,"prepare")
            return
        with (root / "workflow.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage == "prepare":
                prepare(root)
                reference_audit(root)
            elif args.stage in ("pilot","collect"):
                asyncio.run(collection(root,pilot_only=args.stage=="pilot"))
                if args.stage == "collect":
                    cache_audit(root)
            elif args.stage == "audit":
                cache_audit(root)
            elif args.stage == "integrate":
                integrate(root)
            elif args.stage == "report":
                report(root)
            elif args.stage == "finish":
                cache_audit(root)
                integrate(root)
                report(root)
        if args.chain and args.stage in ("prepare","collect"):
            submit_stage(root,"collect" if args.stage=="prepare" else "finish",dependency=os.environ.get("SLURM_JOB_ID"))
    except CheckpointRequested:
        status = read_json(root / "status.json") if (root / "status.json").exists() else {}
        write_json(root / "status.json",dict(status,stage="checkpointed"))
        if args.chain:
            submit_stage(root,"collect",dependency=os.environ.get("SLURM_JOB_ID"))
    except Exception as error:
        status = read_json(root / "status.json") if (root / "status.json").exists() else {}
        write_json(root / "status.json",dict(status,stage="blocked",error_type=type(error).__name__,reason=str(error)))
        raise


if __name__ == "__main__":
    main()
