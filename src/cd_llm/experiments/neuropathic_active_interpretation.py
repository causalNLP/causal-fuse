"""Matched active interpretation experiment using the main neuropathic table."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd

from cd_llm.baselines.causal_order2025.local_vllm_backend import LocalVLLMConfig, _request
from cd_llm.baselines.causal_order2025.openrouter_backend import (
    OpenRouterConfig, ResponseCache, _canonical_request, _request_key,
)
from cd_llm.core.finite_support import (
    Claim, distribution_metrics, evaluate_claim, graph_edges, graph_from_edges,
    has_edge, map_graph, stable_seed, transitive_closure,
)
from cd_llm.experiments.common.artifacts import (
    read_json, run_id, sha256, write_csv, write_json, write_text,
)
from cd_llm.experiments.neuropathic_active_smc_qwen import SMC_CONFIG
from cd_llm.experiments.neuropathic_table2_query_audit import SOURCES
from cd_llm.methods.active_query import ClaimQuery, QueryResponse, active_query_smc
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.methods.llm_claim_confidence import (
    CachedLocalChatClient, CachedOpenRouterChatClient, QwenClaimConfidenceProvider,
)
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_active_interpretation"
MODELS = {"terra": "GPT-5.6 Terra", "qwen": "Qwen3.5-9B"}
STRATEGIES = ("stated", "consistency")
TERRA_CACHE = {
    "stated": WORKSPACE_ROOT / "results/20260902T234859Z/neuropathic_decontamination/caches/openrouter_request_cache.sqlite3",
    "consistency": SOURCES[MODELS["terra"]] / "caches/openrouter_request_cache_consistency.sqlite3",
}
RUNTIME = WORKSPACE_ROOT / "generated/runtime/20260825T082523Z/qwen35_neuropathic20_manifest.json"


def root_for(timestamp):
    return WORKSPACE_ROOT / "results" / run_id(timestamp) / STUDY


def graph(payload):
    return graph_from_edges(20, directed=payload["directed"],
        bidirected=payload.get("bidirected", ()), undirected=payload.get("undirected", ()))


def snapshot_cache(path, strategy):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM responses WHERE purpose LIKE ? ORDER BY request_key",
                                  (strategy + ":%",)).fetchall()
        if not rows:
            raise ValueError("historical request cache is empty")
        return {r["request_key"]: dict(request=json.loads(r["request_json"]),
            purpose=r["purpose"], record=dict(request_key=r["request_key"], content=r["content"],
            response=json.loads(r["response_json"]), provider_request_id=r["provider_request_id"],
            response_model=r["response_model"], prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"], total_tokens=r["total_tokens"],
            cost_usd=r["cost_usd"], latency_seconds=r["latency_seconds"], attempts=r["attempts"], cached=True))
            for r in rows}
    finally:
        connection.close()


def prepare(timestamp):
    root = root_for(timestamp)
    if (root / "manifest.json").exists():
        verify(root)
        return root
    root.mkdir(parents=True, exist_ok=True)
    inputs, sources, tasks = {}, {}, []
    def save(relative, value):
        path = root / relative
        write_json(path, value)
        inputs[relative] = sha256(path)
    benchmark = SOURCES[MODELS["terra"]] / "benchmark"
    for filename in ("model_metadata.json", "truth_graph.json"):
        source = benchmark / filename
        save("inputs/" + filename, read_json(source))
        sources[str(source)] = sha256(source)
    save("inputs/runtime.json", read_json(RUNTIME))
    for seed in range(10):
        for n in (200, 1000):
            source = benchmark / "data" / f"seed_{seed:02d}_n{n}.csv"
            dest = root / "inputs/data" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            inputs[str(dest.relative_to(root))] = sources[str(source)] = sha256(source)
    seen_stats = set()
    for model, display in MODELS.items():
        source_root = SOURCES[display]
        for directory in sorted((source_root / "records").iterdir()):
            if "claim-fuse-smc-active" not in directory.name or "grounded" in directory.name:
                continue
            for source in sorted(directory.glob("seed_*.json")):
                record = read_json(source)
                strategy = record.get("strategy") or "stated"
                if strategy not in STRATEGIES:
                    continue
                seed, n = record["seed"], record["sample_size"]
                ident = f"{model}_{strategy}_s{seed:02d}_n{n}"
                source_stats = Path(record["source_statistics"]["path"])
                if sha256(source_stats) != record["source_statistics"]["sha256"]:
                    raise ValueError("source initialization hash differs")
                stat_name = f"inputs/particles/seed_{seed:02d}_n{n}.json"
                if stat_name not in seen_stats:
                    stats = read_json(source_stats)
                    save(stat_name, {k: stats[k] for k in ("particles", "weights")})
                    sources[str(source_stats)] = sha256(source_stats)
                    seen_stats.add(stat_name)
                source_name = f"inputs/records/{ident}.json"
                save(source_name, record)
                sources[str(source)] = sha256(source)
                tasks.append(dict(id=ident, model=model, strategy=strategy, seed=seed,
                    sample_size=n, source=source_name, particles=stat_name,
                    data=f"inputs/data/seed_{seed:02d}_n{n}.csv"))
        for strategy in STRATEGIES:
            cache = TERRA_CACHE[strategy] if model == "terra" else source_root / f"caches/claim_fuse/{strategy}.sqlite3"
            save(f"inputs/caches/{model}_{strategy}.json", snapshot_cache(cache, strategy))
    expected = {(m,s,seed,n) for m in MODELS for s in STRATEGIES for seed in range(10) for n in (200,1000)}
    actual = {(t["model"],t["strategy"],t["seed"],t["sample_size"]) for t in tasks}
    if actual != expected or len(tasks) != 80:
        raise ValueError("expected exactly 80 Table 2 conditions")
    paths = [Path(__file__)]
    paths += [WORKSPACE_ROOT / "code/src/cd_llm" / name for name in (
        "methods/active_query.py", "methods/evidence_guided.py", "methods/graph_scores.py",
        "methods/llm_claim_confidence.py", "core/finite_support.py",
        "baselines/causal_order2025/openrouter_backend.py",
        "baselines/causal_order2025/local_vllm_backend.py",
        "experiments/neuropathic_active_interpretation_report.py")]
    write_json(root / "manifest.json", dict(version=1, study=STUDY, tasks=sorted(tasks,key=lambda t:t["id"]),
        inputs=inputs, sources=sources, implementation={str(p.resolve()):sha256(p) for p in paths},
        smc_config=asdict(SMC_CONFIG), budget=20, selection_confidence=0.9, soft_cap=0.99,
        incremental_hosted_cost_cap_usd=10.0, bootstrap_resamples=10000,
        selection="original Group I EIG, recomputed from each arm's particles",
        interpretation="only positive directed responses become positive ancestry; confidence is unchanged"))
    print(json.dumps(dict(prepared=str(root), matched_conditions=len(tasks))), flush=True)
    return root


def verify(root):
    manifest = read_json(root / "manifest.json")
    for name, expected in manifest["inputs"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen input changed {name}")
    for name, expected in manifest["implementation"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"registered implementation changed {name}")
    if manifest["smc_config"] != json.loads(json.dumps(asdict(SMC_CONFIG))):
        raise ValueError("SMC settings changed")
    return manifest


class LayeredCache:
    """Historical responses are immutable; spending counts only new requests."""
    def __init__(self, historical, incremental):
        self.historical, self.incremental = historical, incremental

    def get(self, request):
        key = _request_key(request)
        if key in self.historical:
            return dict(self.historical[key]["record"])
        return self.incremental.get(request) if self.incremental is not None else None

    def put(self, *args, **kwargs):
        return self.incremental.put(*args, **kwargs)

    def summary(self):
        return self.incremental.summary()


class HistoricalChat:
    def __init__(self, model, config, cache, delegate=None):
        self.model, self.config, self.cache, self.delegate = model, config, cache, delegate

    def request(self, messages, *, purpose, max_tokens=256):
        messages = [dict(m) for m in messages]
        for message in messages:
            if message["role"] == "system":
                message["content"] = message["content"].replace(
                    "careful causal domain expert", "careful medical causal expert")
        if self.model == "terra":
            request = _canonical_request(messages, self.config.model, max_tokens,
                                         self.config.seed, self.config.reasoning_effort)
        else:
            request = _request(messages, self.config, max_tokens)
        cached = self.cache.get(request)
        if cached is not None:
            return cached
        if self.delegate is None:
            raise RuntimeError(f"uncached historical prompt {purpose}; no abstention substituted")
        return self.delegate.request(messages, purpose=purpose, max_tokens=max_tokens)


def provider_for(root, model, strategy, incremental=None, base_url=None, online=False,
                 minimum_valid_consistency_responses=6):
    old = read_json(root / f"inputs/caches/{model}_{strategy}.json")
    params = [dict((k,v) for k,v in item["request"].items() if k != "messages") for item in old.values()]
    if any(p != params[0] for p in params):
        raise ValueError("historical request settings are not constant")
    p = params[0]
    cache = LayeredCache(old, incremental)
    delegate = None
    if model == "terra":
        config = OpenRouterConfig(model=p["model"], seed=p["seed"], reasoning_effort=p.get("reasoning_effort"),
            concurrency=1, max_attempts=8, cost_cap_usd=10.0)
        if online:
            delegate = CachedOpenRouterChatClient(cache, config)
    else:
        config = LocalVLLMConfig(model=p["model"], seed=p["seed"], temperature=p["temperature"],
            base_url=base_url or "http://127.0.0.1:18000/v1", concurrency=1,
            triplet_max_tokens=256, pair_max_tokens=256)
        if online:
            delegate = CachedLocalChatClient(cache, config)
    metadata = read_json(root / "inputs/model_metadata.json")
    provider = QwenClaimConfidenceProvider(strategy, [n["id"] for n in metadata["nodes"]],
        HistoricalChat(model, config, cache, delegate), descriptions={n["id"]:n["description"] for n in metadata["nodes"]},
        domain_context=metadata["domain_context"], protocol="legacy_v1",
        minimum_valid_consistency_responses=minimum_valid_consistency_responses)
    return provider, delegate


def response_from_step(step):
    return QueryResponse(step["answer"], step["reported_confidence"],
        prompt_tokens=step.get("prompt_tokens",0), completion_tokens=step.get("completion_tokens",0),
        attempts=step.get("response_attempts",1), abstention_reason=step.get("abstention_reason"),
        audit=step.get("response_audit",{}), treatment=step.get("treatment"))


def validate_cached_providers(root, tasks, model):
    counts = {}
    for strategy in STRATEGIES:
        bank = {}
        for task in tasks:
            if task["strategy"] != strategy:
                continue
            for step in read_json(root / task["source"])["trace"]:
                key = step["query"]["query_id"]
                if key in bank and (step["answer"],step["reported_confidence"]) != (
                    bank[key]["answer"],bank[key]["reported_confidence"]):
                    raise ValueError("historical question has inconsistent answers")
                bank[key] = step
        provider, _ = provider_for(root, model, strategy)
        for key, step in bank.items():
            actual = provider(ClaimQuery(**step["query"]))
            if (actual.answer,actual.reported_confidence) != (step["answer"],step["reported_confidence"]):
                raise ValueError(f"historical response mismatch {model} {strategy} {key}")
        counts[strategy] = len(bank)
    return counts


def execute_case(root, task, provider, mode):
    data = pd.read_csv(root / task["data"])
    names = [n["id"] for n in read_json(root / "inputs/model_metadata.json")["nodes"]]
    if list(data.columns) != names:
        raise ValueError("observation columns differ from the frozen glossary")
    population = read_json(root / task["particles"])
    score = BDeuScore(data.to_numpy(dtype=np.int64), equivalent_sample_size=1.0,
                     cardinalities=(2,)*20, max_parents=SMC_CONFIG.max_parents, structure_prior=1.0)
    active = active_query_smc(None, score, provider, budgets=(20,), confidence=0.9,
        treatment="soft_evidence", soft_confidence_cap=0.99, policy="eig",
        seed=stable_seed("neuropathic_decontamination", "active", task["seed"], task["sample_size"]),
        ess_threshold_fraction=SMC_CONFIG.ess_threshold_fraction,
        rejuvenation_steps=SMC_CONFIG.rejuvenation_steps,
        guided_mixture_probability=SMC_CONFIG.guided_mixture_probability,
        proposal_cache_size=SMC_CONFIG.proposal_cache_size,
        initial_particles=tuple(graph(g) for g in population["particles"]),
        initial_weights=population["weights"], direct_yes_interpretation=mode)
    if len(active.steps) != 20 or len({s.query.query_id for s in active.steps}) != 20:
        raise ValueError("active trajectory must contain twenty distinct questions")
    return dict(task=task, interpretation=mode, trace=[asdict(s) for s in active.steps],
        distribution=[dict(graph=graph_edges(g),weight=w) for g,w in sorted(active.distributions[20].items())],
        elapsed_seconds=active.cumulative_seconds[20])


def check_literal(result, source):
    for actual, expected in zip(result["trace"],source["trace"]):
        for key in ("query", "answer", "reported_confidence", "valid_response", "resampled"):
            if actual[key] != expected[key]:
                raise ValueError(f"literal trace differs at slot {actual['budget']} in {key}")
    actual = {graph(r["graph"]):r["weight"] for r in result["distribution"]}
    expected = {graph(r["graph"]):r["weight"] for r in source["distribution"]}
    error = max(abs(actual.get(g,0)-expected.get(g,0)) for g in actual.keys() | expected.keys())
    if error > 1e-11:
        raise ValueError(f"literal posterior differs from Table 2 by {error}")
    return dict(passed=True, max_mass_difference=error, queries_matched=20)


def validate_model(root, model):
    manifest = verify(root)
    tasks = [t for t in manifest["tasks"] if t["model"] == model]
    counts = validate_cached_providers(root, tasks, model)
    for index,task in enumerate(tasks):
        destination = root / f"records/literal/{task['id']}.json"
        if destination.exists():
            check_literal(read_json(destination), read_json(root / task["source"]))
            continue
        source = read_json(root / task["source"])
        cursor = 0
        def answer(query):
            nonlocal cursor
            step = source["trace"][cursor]
            if asdict(query) != step["query"]:
                raise ValueError(f"literal query sequence diverged in {task['id']} at {cursor+1}")
            cursor += 1
            return response_from_step(step)
        result = execute_case(root, task, answer, "literal")
        result["reproduction"] = check_literal(result, source)
        write_json(destination, result)
        print(json.dumps(dict(model=model, stage="literal", completed=index+1,total=40,task=task["id"])),flush=True)
    write_json(root / f"validation/{model}.json", dict(passed=True, literal_runs=40, cached_questions=counts))


def run_model(root, model, base_url):
    manifest = verify(root)
    if not read_json(root / f"validation/{model}.json")["passed"]:
        raise ValueError("literal validation has not passed")
    tasks = [t for t in manifest["tasks"] if t["model"] == model]
    with (root / f"{model}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        incremental = ResponseCache(root / f"caches/{model}_incremental.sqlite3")
        delegates = []
        try:
            providers = {}
            for strategy in STRATEGIES:
                providers[strategy], delegate = provider_for(root,model,strategy,incremental,base_url,online=True)
                delegates.append(delegate)
            for index,task in enumerate(tasks):
                destination = root / f"records/ancestry/{task['id']}.json"
                if destination.exists():
                    continue
                calls = 0
                def answer(query):
                    nonlocal calls
                    calls += 1
                    write_json(root / f"status/{model}.json",dict(stage="answering",task=task["id"],
                        completed=index,total=40,slot=calls,query=asdict(query),cost=incremental.summary()["cost_usd"]))
                    return providers[task["strategy"]](query)
                result = execute_case(root,task,answer,"ancestry")
                write_json(destination,result)
                print(json.dumps(dict(model=model,stage="ancestry",completed=index+1,total=40,
                    task=task["id"],cost=incremental.summary()["cost_usd"])),flush=True)
            write_json(root / f"status/{model}.json",dict(stage="complete",completed=40,total=40,
                accounting=incremental.summary()))
        finally:
            for delegate in delegates:
                if hasattr(delegate,"close"):
                    delegate.close()
            incremental.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","validate","run","report"))
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--model",choices=tuple(MODELS))
    parser.add_argument("--base-url")
    args=parser.parse_args()
    if args.stage in ("validate", "run") and args.model is None:
        parser.error("--model is required for validate and run")
    root=root_for(args.run_id)
    if args.stage == "prepare":
        prepare(args.run_id)
    elif args.stage == "validate":
        validate_model(root,args.model)
    elif args.stage == "run":
        run_model(root,args.model,args.base_url)
    else:
        from cd_llm.experiments.neuropathic_active_interpretation_report import report
        report(root)


if __name__ == "__main__":
    main()
