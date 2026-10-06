"""Matched ALARM LLM baseline comparison with SCP Pattern 2 deferred."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import time
import traceback
import urllib.request

import numpy as np

from cd_llm.core.finite_support import PartialGraph, graph_edges, graph_from_edges, stable_seed
from cd_llm.experiments import alarm_active as active
from cd_llm.experiments import alarm_bfs_openrouter as bfs
from cd_llm.experiments import alarm_observational as observational
from cd_llm.experiments.alarm_llm_transport import (
    CAP, ENDPOINTS, MODEL, PRICES, PROVIDER, Transport, digest, file_lock, proxy,
)
from cd_llm.experiments.common.artifacts import read_json, write_json, write_text, run_id, sha256
from cd_llm.methods.active_query import active_query_smc, group_i_query_pool
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.paths import WORKSPACE_ROOT

METHODS = ("gpc", "long", "bfs", "active", "order")
LABELS = {"gpc": "gPC-Guess", "long": "PC+Long-Srisk", "bfs": "BFS+Pearson",
          "active": "CLAIM-FUSE-Active, 20 queries", "order": "Causal order + PC"}
TASKS = [(method, seed) for method in METHODS[:-1] for seed in range(5)]
SOURCE_ACTIVE = WORKSPACE_ROOT / "results/20260910T012553Z/alarm_active"
SOURCE_OBS = WORKSPACE_ROOT / "results/20260910T010648Z/alarm_observational"
SOURCE_BFS = WORKSPACE_ROOT / "results/20260909T215919Z/alarm_bfs_openrouter"


def root(identifier):
    return WORKSPACE_ROOT / "results" / run_id(identifier) / "alarm_llm_baselines"


def metadata(directory):
    data = read_json(directory / "inputs/metadata.json")
    return data, tuple(n["id"] for n in data["nodes"]), {n["id"]: n["description"] for n in data["nodes"]}


def verify(directory):
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["input_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"frozen input changed {name}")
    for name, expected in manifest["implementation_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"frozen implementation changed {name}")
    if manifest["methods"] != list(METHODS) or manifest["scp"] != "TBD":
        raise ValueError("method registration changed")
    return manifest


def prepare(identifier):
    directory = root(identifier)
    if directory.exists():
        raise FileExistsError(directory)
    active.verify(SOURCE_ACTIVE)
    observational.verify(SOURCE_OBS)
    bfs.validate(SOURCE_BFS)
    with urllib.request.urlopen(ENDPOINTS, timeout=30) as response:
        endpoints = json.load(response)
    endpoint = next(e for e in endpoints["data"]["endpoints"] if e["tag"] == PROVIDER)
    if endpoint["quantization"] != "bf16" or not {"logprobs", "top_logprobs", "seed", "reasoning"} <= set(endpoint["supported_parameters"]):
        raise ValueError("provider capabilities changed")
    if any(float(endpoint["pricing"][k]) > PRICES[k]+1e-15 for k in PRICES):
        raise ValueError("provider exceeds registered prices")
    files = [f"inputs/observations_{s}.npy" for s in range(5)]
    files += ["inputs/presentation.json", "inputs/encoding.json", "inputs/metadata.json", "evaluation_only/reference.json"]
    for relative in files:
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ACTIVE / relative, target)
    for seed in range(5):
        for method in ("pc", "ges"):
            relative = f"inputs/{method}_{seed}.json"
            shutil.copyfile(SOURCE_OBS / f"predictions/{method}_{seed}.json", directory / relative)
            files.append(relative)
        relative = f"initializations/{seed}.json"
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ACTIVE / relative, target)
        files.append(relative)
        if sha256(directory / f"inputs/observations_{seed}.npy") != sha256(SOURCE_BFS / f"inputs/observations_{seed}.npy"):
            raise ValueError("source datasets disagree")
    write_json(directory / "evaluation_only/observational.json", read_json(SOURCE_OBS / "report.json"))
    write_json(directory / "provider_endpoints.json", endpoints)
    files += ["evaluation_only/observational.json", "provider_endpoints.json"]
    base = WORKSPACE_ROOT / "code/src/cd_llm"
    sources = [Path(__file__), base / "experiments/alarm_llm_transport.py", Path(active.__file__),
               Path(bfs.__file__), Path(observational.__file__), base / "core/finite_support.py",
               base / "methods/active_query.py", base / "methods/evidence_guided.py", base / "methods/graph_scores.py"]
    for name in ("guess2graph2026", "long2023", "causal_order2025", "jiralerspong2024"):
        sources.extend((base / "baselines" / name).glob("*.py"))
    write_json(directory / "manifest.json", dict(run_id=identifier, model=MODEL, provider=PROVIDER,
        methods=METHODS, tasks=TASKS, scp="TBD", seeds=list(range(5)), observations=200, variables=37,
        cost_cap_usd=CAP, prices=PRICES, active_config=asdict(active.CONFIG), active_budgets=active.BUDGETS,
        pc_settings=observational.SETTINGS["pc"], long_tolerance=.1, long_max_extensions=50000,
        order_triplets=7770, order_seed=0, order_parse_gate=.95, order_concurrency=16,
        source_manifests={str(p): sha256(p / "manifest.json") for p in (SOURCE_ACTIVE, SOURCE_OBS, SOURCE_BFS)},
        input_sha256={f: sha256(directory / f) for f in files},
        implementation_sha256={str(p): sha256(p) for p in sources}), exclusive=True)
    Transport(directory, "prepare")
    report(identifier)
    print(json.dumps({"prepared": str(directory), "tasks": len(TASKS), "shared_order_tasks": 1}), flush=True)


def observations(directory, seed):
    order = read_json(directory / "inputs/presentation.json")["order"]
    values = np.load(directory / f"inputs/observations_{seed}.npy", allow_pickle=False)[:, order]
    if values.shape != (200, 37):
        raise ValueError("wrong ALARM dimensions")
    return values


def pc_partial(directory, seed):
    return PartialGraph.from_matrix(np.array(read_json(directory / f"inputs/pc_{seed}.json")["presentation_endpoints"]), "cpdag")


def select_distribution(distribution, order):
    return min(distribution, key=lambda r: (-r["weight"], sorted((order[a], order[b]) for a, b in r["edges"])))


def save_prediction(directory, method, seed, matrix, started, **details):
    matrix = np.asarray(matrix, dtype=int)
    order = read_json(directory / "inputs/presentation.json")["order"]
    # Only mapping, never the reference graph, is accessed before this write.
    prediction = dict(method=method, seed=seed, presentation_endpoints=matrix.tolist(),
        canonical_endpoints=observational.canonical_matrix(matrix, order).tolist(),
        wall_seconds=time.monotonic()-started, manifest_sha256=sha256(directory / "manifest.json"),
        saved_before_reference_evaluation=True, **details)
    write_json(directory / f"predictions/{method}_{seed}.json", prediction)
    write_json(directory / f"status/{method}_{seed}.json", dict(state="complete", seed=seed, method=method))


def gpc_fit(directory, seed, transport):
    from openai import OpenAI
    from cd_llm.baselines.guess2graph2026 import run_gpc_guess_skeleton
    from cd_llm.baselines.guess2graph2026.llm_expert import CachedOpenAISkeletonExpert, LLMExpertSettings
    data, names, descriptions = metadata(directory)
    started = time.monotonic()
    with proxy(transport) as url, OpenAI(base_url=url, api_key="EMPTY", max_retries=0) as client:
        expert = CachedOpenAISkeletonExpert(cache_dir=directory / "caches/gpc",
            settings=LLMExpertSettings(model=MODEL, request_seed=0), client=client,
            descriptions=descriptions, domain_context=data["context"])
        guess = expert.guess(names, trial_seed=seed)
    skeleton = run_gpc_guess_skeleton(observations(directory, seed), guess.skeleton,
        alpha=.05, ci_test="gsq", rng=np.random.default_rng(stable_seed("alarm-gpc", seed)))
    save_prediction(directory, "gpc", seed, np.where(skeleton, 1, 0), started,
        guessed_edges=guess.edges, query_policy="one skeleton guess per dataset", point_estimator="native undirected skeleton")


def long_fit(directory, seed, transport):
    from cd_llm.baselines.long2023 import openrouter_expert, run_long
    data, names, descriptions = metadata(directory)
    started = time.monotonic()
    # A shared seed and identical named pairs allow 22 answers to cover 53 uses.
    class ContextTransport:
        def query(self, request):
            return transport.query(dict(request, messages=[dict(role="system", content=data["context"]), *request["messages"]]))
    with file_lock(directory / "locks/long-cache.lock"), proxy(ContextTransport()) as url:
        expert = openrouter_expert(MODEL, url, 0, directory / "caches/long.json", enable_thinking=False)
        result = run_long(pc_partial(directory, seed), tuple(f"{n} ({descriptions[n]})" for n in names),
                          expert, strategy="srisk", tolerance=.1, max_extensions=50000)
    distribution = [dict(edges=graph_edges(g)["directed"], weight=1/len(result.final_mec)) for g in result.final_mec]
    selected = select_distribution(distribution, read_json(directory / "inputs/presentation.json")["order"])
    save_prediction(directory, "long", seed, observational.edge_matrix(37, selected["edges"]), started,
        distribution=distribution, expert_records=result.expert_records, decisions=result.decisions,
        retained_probability=result.retained_probability, initial_mec_size=len(result.initial_mec),
        point_estimator="uniform surviving MEC mode with canonical lexicographic tie rule")


def bfs_fit(directory, seed, transport):
    from cd_llm.baselines.jiralerspong2024 import Completion, run_bfs
    data, names, descriptions = metadata(directory)
    started, diagnostics = time.monotonic(), []
    original_names = read_json(directory / "inputs/encoding.json")["names"]
    def complete(messages):
        supplied = bfs.bfs_messages(messages)
        bfs.audit_messages(supplied, names, original_names)
        response, path = transport.query(dict(model=MODEL, messages=supplied, temperature=.7, seed=seed, max_tokens=4096))
        choice = response["choices"][0]
        diagnostics.append(dict(request=path, finish_reason=choice["finish_reason"],
            repair=messages[-1]["content"].startswith("Your previous response did not follow")))
        write_json(directory / f"status/bfs_{seed}.json", dict(state="running", queries=len(diagnostics), seed=seed))
        usage = response.get("usage") or {}
        return Completion(bfs.completion_text(choice), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    import pandas as pd
    edges, usage = run_bfs(pd.DataFrame(observations(directory, seed), columns=names), complete,
        include_statistics=True, descriptions=descriptions, domain_context=data["context"],
        opaque_identifiers=False, empty_root_policy="empty_graph")
    save_prediction(directory, "bfs", seed, observational.edge_matrix(37, edges), started,
                    diagnostics=diagnostics, usage=usage, point_estimator="native BFS DAG")


def active_fit(directory, seed, transport):
    data, _, _ = metadata(directory)
    started = time.monotonic()
    score = BDeuScore(observations(directory, seed), equivalent_sample_size=1., structure_prior=1., max_parents=active.CONFIG.max_parents)
    initial = read_json(directory / f"initializations/{seed}.json")
    class ActiveTransport:
        def query(self, messages):
            return transport.query(dict(model=MODEL, messages=messages, temperature=.7, seed=seed, max_tokens=4096))
    provider = active.Provider(directory / "active", seed, data, ActiveTransport())
    config = active.CONFIG
    result = active_query_smc(None, score, provider, query_pool=group_i_query_pool(37), budgets=active.BUDGETS,
        confidence=.9, treatment="soft_evidence", policy="eig", soft_confidence_cap=config.soft_confidence_cap,
        seed=stable_seed("alarm-active-trajectory", seed), ess_threshold_fraction=config.ess_threshold_fraction,
        rejuvenation_steps=config.rejuvenation_steps, guided_mixture_probability=config.guided_mixture_probability,
        proposal_cache_size=config.proposal_cache_size, ordinary_kernel=config.ordinary_kernel,
        initial_particles=tuple(graph_from_edges(37, directed=e) for e in initial["particles"]), initial_weights=initial["weights"])
    if len(result.steps) != 20 or provider.slot != 20:
        raise ValueError("active trajectory did not consume 20 slots")
    distributions = {str(b): active.encode_distribution(d) for b, d in result.distributions.items()}
    selected = select_distribution(distributions["20"], read_json(directory / "inputs/presentation.json")["order"])
    save_prediction(directory, "active", seed, observational.edge_matrix(37, selected["edges"]), started,
        distributions=distributions, steps=[asdict(s) for s in result.steps], timing=asdict(result.timing),
        resampling_count=result.resampling_count, support_recovery_count=result.support_recovery_count,
        point_estimator="aggregated particle mode with canonical lexicographic tie rule")


async def order_fit(directory, transport, smoke=False):
    from cd_llm.baselines.causal_order2025 import LocalVLLMConfig, PCOrderConfig, elicit_triplet_order_local_vllm, query_pair_fallbacks_local_vllm, fit_pc_with_causal_order
    from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
    from cd_llm.experiments.neuropathic_decontamination_qwen import _expert_from_record
    from cd_llm.experiments.neuropathic_causal_order_subnetwork import _expert_record
    data, names, descriptions = metadata(directory)
    if smoke:
        names = names[:3]
        descriptions = {n: descriptions[n] for n in names}
    folder = directory / ("smoke_order" if smoke else "order")
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with proxy(transport) as url:
        config = LocalVLLMConfig(model=MODEL, base_url=url, seed=0, concurrency=16,
            max_attempts=1, triplet_retry_max_tokens=7680)
        path = folder / "expert.json"
        if path.exists():
            expert = _expert_from_record(read_json(path))
        else:
            result = await elicit_triplet_order_local_vllm(names, data["context"], folder / "cache.sqlite3",
                descriptions=descriptions, config=config, minimum_parse_rate=.95)
            expert = result.expert
            write_json(path, dict(_expert_record(expert), audit=dict(result.audit)))
        if smoke:
            return dict(triplets=len(expert.subgroup_list), expert_order=expert.expert_order)
        partials = [pc_partial(directory, seed) for seed in range(5)]
        pairs = required_fallback_pairs(expert, partials)
        fallbacks, audit = await query_pair_fallbacks_local_vllm(names, pairs, data["context"], folder / "cache.sqlite3",
            descriptions=descriptions, config=config)
    def fallback(a, b):
        return fallbacks[tuple(sorted((a, b)))]
    write_json(folder / "fallbacks.json", dict(pairs=pairs, answers=[dict(pair=p, answer=a) for p,a in fallbacks.items()], audit=audit))
    shared_seconds = time.monotonic()-started
    for seed in range(5):
        if (directory / f"predictions/order_{seed}.json").exists():
            continue
        fit_started = time.monotonic()
        result = fit_pc_with_causal_order(observations(directory, seed), names, expert,
            config=PCOrderConfig(alpha=.05, indep_test="gsq", stable=True, uc_rule=0, uc_priority=2), fallback_orienter=fallback)
        save_prediction(directory, "order", seed, observational.edge_matrix(37, graph_edges(result.graph)["directed"]), fit_started,
            final_order=result.final_order, fallback_orientations=result.fallback_orientations,
            shared_collection_seconds=shared_seconds, point_estimator="native causal order plus PC DAG")


def preflight(identifier):
    directory = root(identifier)
    verify(directory)
    from cd_llm.baselines.long2023 import openrouter_expert
    from cd_llm.baselines.long2023.algorithm import ambiguous_pairs
    data, names, descriptions = metadata(directory)
    transport = Transport(directory, "preflight")
    pairs = sorted(set(p for seed in range(5) for p in ambiguous_pairs(pc_partial(directory, seed))))
    class ContextTransport:
        def query(self, request):
            return transport.query(dict(request, messages=[dict(role="system", content=data["context"]), *request["messages"]]))
    with proxy(ContextTransport()) as url:
        expert = openrouter_expert(MODEL, url, 0, directory / "caches/long.json", enable_thinking=False)
        a,b = pairs[0]
        arc, confidence, _ = expert(a,b,f"{names[a]} ({descriptions[names[a]]})",f"{names[b]} ({descriptions[names[b]]})")
    order_check = asyncio.run(order_fit(directory, transport, smoke=True))
    write_json(directory / "preflight.json", dict(state="passed", long_pair_count=len(pairs),
        long_arc=arc, long_confidence=confidence, order=order_check, api=transport.summary()))
    print(json.dumps(read_json(directory / "preflight.json")), flush=True)


def run(identifier, task=None, order=False):
    directory = root(identifier)
    verify(directory)
    if read_json(directory / "preflight.json")["state"] != "passed":
        raise ValueError("preflight has not passed")
    method, seed = ("order", None) if order else TASKS[task]
    key = method if order else f"{method}_{seed}"
    with file_lock(directory / "locks" / f"task-{key}.lock"):
        if not order and (directory / f"predictions/{key}.json").exists():
            report(identifier)
            return
        def interrupted(signum, frame):
            raise TimeoutError("task interrupted; resume using saved responses")
        signal.signal(signal.SIGTERM, interrupted)
        write_json(directory / f"status/{key}.json", dict(state="running", method=method, seed=seed, started=time.time()))
        try:
            transport = Transport(directory, method)
            if order:
                asyncio.run(order_fit(directory, transport))
                write_json(directory / f"status/{key}.json", dict(state="complete", method=method))
            else:
                {"gpc":gpc_fit,"long":long_fit,"bfs":bfs_fit,"active":active_fit}[method](directory, seed, transport)
        except BaseException as error:
            write_json(directory / f"status/{key}.json", dict(state="failed", method=method, seed=seed,
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            raise
        finally:
            report(identifier)


def report(identifier):
    directory = root(identifier)
    with file_lock(directory / "locks/report.lock"):
        verify(directory)
        reference = read_json(directory / "evaluation_only/reference.json")
        order = read_json(directory / "inputs/presentation.json")["order"]
        truth = set(map(tuple,reference["edges"]))
        records, claims, checkpoints = [], [], []
        for path in sorted((directory / "predictions").glob("*.json")):
            p = read_json(path)
            if p["manifest_sha256"] != sha256(directory / "manifest.json"):
                raise ValueError("prediction provenance mismatch")
            records.append(dict(method=p["method"], seed=p["seed"], wall_seconds=p["wall_seconds"],
                **observational.evaluate(np.array(p["canonical_endpoints"]), reference)))
            if p["method"] == "active":
                initial = read_json(directory / f'initializations/{p["seed"]}.json')
                for budget, distribution in {"0":initial["distribution"], **p["distributions"]}.items():
                    checkpoints.append(dict(seed=p["seed"], budget=int(budget), **active.distribution_scores(distribution, order, reference)))
                for step in p["steps"]:
                    q = step["query"]
                    a,b = order[q["source"]],order[q["target"]]
                    value = (a,b) in truth or (q["kind"]=="adjacency" and (b,a) in truth)
                    correct = None if step["answer"] is None else int(step["answer"]==int(value))
                    confidence = step["reported_confidence"]
                    claims.append(dict(seed=p["seed"],slot=step["budget"],kind=q["kind"],answer=step["answer"],
                        confidence=confidence, correct=correct, brier=None if correct is None else (confidence-correct)**2))
        def aggregate(rows):
            result = dict(completed=len(rows))
            for metric in ("distance", "adjacency_f1", "directed_f1", "wall_seconds", "posterior_expected_distance"):
                values = [r[metric] for r in rows if metric in r]
                result[metric] = dict(mean=statistics.mean(values) if values else None,
                    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)
            return result
        summary = {m:aggregate([r for r in records if r["method"]==m]) for m in METHODS}
        answered = [c for c in claims if c["correct"] is not None]
        quality = dict(attempted=len(claims), answered=len(answered),
            accuracy=statistics.mean(c["correct"] for c in answered) if answered else None,
            confidence=statistics.mean(c["confidence"] for c in answered) if answered else None,
            brier=statistics.mean(c["brier"] for c in answered) if answered else None)
        statuses = {p.stem: read_json(p) for p in (directory / "status").glob("*.json")}
        result = dict(completed=len(records), planned=25, scp="TBD", summary=summary,
            records=records, checkpoints={str(b):aggregate([r for r in checkpoints if r["budget"]==b]) for b in (0,5,10,20)},
            checkpoint_records=checkpoints, claims=claims, claim_quality=quality, statuses=statuses,
            api=Transport(directory,"report").summary(),
            observational=read_json(directory / "evaluation_only/observational.json")["summary"])
        result["metric_scale"] = "unnormalized"
        write_json(directory / "report.json", result)
        lines = ["ALARM matched LLM comparison", "", f'{len(records)}/25 graph predictions complete. SCP Pattern 2 is TBD.', "",
            "| Method | Completed | PD | Adjacency F1 | Orientation F1 |", "| --- | --- | --- | --- | --- |"]
        for method,s in summary.items():
            cells = [f'{s[k]["mean"]:.4f} ({s[k]["se"]:.4f})' if s[k]["se"] is not None else str(s[k]["mean"]) for k in ("distance","adjacency_f1","directed_f1")]
            lines.append("| " + " | ".join([LABELS[method],str(s["completed"]),*cells]) + " |")
        lines += ["| SCP Pattern 2 | Deferred | TBD | TBD | TBD |", "", f'Accounted API cost USD {result["api"]["accounted_usd"]:.6f} across {result["api"]["requests"]} dispatches.']
        write_text(directory / "report.md", "\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","preflight","run","order","report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", type=int, choices=range(len(TASKS)))
    args = parser.parse_args()
    if args.command == "run":
        if args.task is None:
            parser.error("run requires --task")
        run(args.run_id,args.task)
    elif args.command == "order":
        run(args.run_id,order=True)
    else:
        {"prepare":prepare,"preflight":preflight,"report":report}[args.command](args.run_id)


if __name__ == "__main__":
    main()
