"""Active local CLAIM-FUSE claims on frozen, masked ALARM observations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import time
import traceback
import urllib.error
import urllib.request

import numpy as np

from cd_llm.core.finite_support import (
    PartialGraph, graph_edges, graph_from_edges, sample_completions_with_diagnostics, stable_seed,
)
from cd_llm.experiments.alarm_observational import (
    canonical_matrix, edge_matrix, evaluate, verify as verify_observational,
)
from cd_llm.experiments.common.artifacts import read_json, write_json, write_text, run_id, sha256
from cd_llm.methods.active_query import QueryResponse, active_query_smc, group_i_query_pool
from cd_llm.methods.evidence_guided import ClaimFuseSMCConfig, initialize_score_particles
from cd_llm.methods.graph_scores import BDeuScore, parent_masks
from cd_llm.paths import WORKSPACE_ROOT

MODEL = "qwen/qwen3.5-9b"
PROVIDER = "deepinfra/bf16"
PRICES = dict(prompt=0.10e-6, completion=0.15e-6)
ENDPOINTS = "https://openrouter.ai/api/v1/models/qwen/qwen3.5-9b/endpoints"
CONFIG = ClaimFuseSMCConfig(particle_count=384, initialization_updates=192, warm_start_score_mass=0.0)
BUDGETS = (5, 10, 20)
SOURCE_RUN = "20260910T010648Z"
BFS_RUN = "20260909T215919Z"


def root(identifier):
    return WORKSPACE_ROOT / "results" / run_id(identifier) / "alarm_active"


@contextmanager
def lock(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def request_for(messages, seed):
    return dict(model=MODEL, messages=messages, temperature=0.7, seed=seed, max_tokens=4096,
                reasoning=dict(enabled=False), provider=dict(only=[PROVIDER], allow_fallbacks=False,
                    require_parameters=True, max_price=dict(prompt=0.10, completion=0.15)))


def reservation(request):
    tokens = sum(len(m["content"].encode("utf-8")) + 64 for m in request["messages"]) + 64
    return tokens * PRICES["prompt"] + request["max_tokens"] * PRICES["completion"]


def accounted(record):
    usage = record.get("response", {}).get("usage") or {}
    if not usage:
        return record["reserved_usd"]
    return max(float(usage.get("cost") or 0),
               usage.get("prompt_tokens", 0) * PRICES["prompt"] + usage.get("completion_tokens", 0) * PRICES["completion"])


def spend(directory):
    records = [read_json(p) for p in sorted((directory / "requests").glob("*.json"))]
    return dict(physical_requests=len(records), accounted_usd=sum(accounted(r) for r in records),
                reported_usd=sum(float((r.get("response", {}).get("usage") or {}).get("cost") or 0) for r in records),
                prompt_tokens=sum((r.get("response", {}).get("usage") or {}).get("prompt_tokens", 0) for r in records),
                completion_tokens=sum((r.get("response", {}).get("usage") or {}).get("completion_tokens", 0) for r in records),
                cost_cap_usd=1.0)


def verify(directory):
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["input_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"frozen artifact changed {name}")
    for name, expected in manifest["implementation_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"frozen implementation changed {name}")
    if manifest["config"] != json.loads(json.dumps(asdict(CONFIG))):
        raise ValueError("configuration changed")
    return manifest


def prepare(identifier):
    directory = root(identifier)
    if directory.exists():
        raise FileExistsError(directory)
    source = WORKSPACE_ROOT / "results" / SOURCE_RUN / "alarm_observational"
    original = verify_observational(source)
    bfs = WORKSPACE_ROOT / "results" / BFS_RUN / "alarm_bfs_openrouter"
    bfs_manifest = read_json(bfs / "manifest.json")
    if sha256(bfs / "manifest.json") != original["source_manifest_sha256"]:
        raise ValueError("BFS source changed")
    metadata = bfs / "inputs/metadata.json"
    expected = {str(Path(k).resolve()): v for k, v in bfs_manifest["frozen_sha256"].items()}
    if sha256(metadata) != expected[str(metadata.resolve())]:
        raise ValueError("masked metadata changed")
    with urllib.request.urlopen(ENDPOINTS, timeout=20) as response:
        endpoints = json.load(response)
    endpoint = next(e for e in endpoints["data"]["endpoints"] if e["tag"] == PROVIDER)
    if any(float(endpoint["pricing"][k]) > PRICES[k] for k in PRICES):
        raise ValueError("provider prices exceed frozen ceilings")
    required = {"reasoning", "temperature", "seed", "max_tokens"}
    if not required <= set(endpoint["supported_parameters"]):
        raise ValueError("provider no longer supports the requested settings")
    files = [f"inputs/observations_{s}.npy" for s in range(5)]
    files += ["inputs/presentation.json", "inputs/encoding.json", "evaluation_only/reference.json"]
    for relative in files:
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    for seed in range(5):
        relative = f"inputs/ges_{seed}.json"
        prediction = read_json(source / f"predictions/ges_{seed}.json")
        if prediction["manifest_sha256"] != sha256(source / "manifest.json") or prediction["seed"] != seed:
            raise ValueError("GES provenance mismatch")
        write_json(directory / relative, prediction)
        files.append(relative)
    write_json(directory / "inputs/metadata.json", read_json(metadata))
    write_json(directory / "evaluation_only/baselines.json", read_json(source / "report.json"))
    write_json(directory / "provider_endpoints.json", endpoints)
    files += ["inputs/metadata.json", "evaluation_only/baselines.json", "provider_endpoints.json"]
    paths = [Path(__file__), *[Path(p) for p in original["implementation_sha256"]],
             WORKSPACE_ROOT / "code/src/cd_llm/methods/active_query.py",
             WORKSPACE_ROOT / "code/src/cd_llm/methods/evidence_guided.py",
             WORKSPACE_ROOT / "code/src/cd_llm/methods/graph_scores.py"]
    write_json(directory / "manifest.json", dict(run_id=identifier, source_run=SOURCE_RUN,
        source_manifest_sha256=sha256(source / "manifest.json"), config=asdict(CONFIG), budgets=BUDGETS,
        selection_confidence=0.9, query_family="local_edge", query_count=1998, completion_limit=200,
        data_score=dict(equivalent_sample_size=1.0, structure_prior=1.0, observed_cardinalities=True),
        model=MODEL, provider=PROVIDER, generation=request_for([], 0), cost_cap_usd=1.0,
        input_sha256={name: sha256(directory / name) for name in files},
        implementation_sha256={str(p): sha256(p) for p in paths}), exclusive=True)
    write_json(directory / "example_prompt.json", messages_for(group_i_query_pool(37)[0], read_json(metadata)))
    report(identifier)


def messages_for(query, metadata):
    names = [node["id"] for node in metadata["nodes"]]
    a, b = names[query.source], names[query.target]
    if query.kind == "directed":
        question = f"Does {a} directly cause {b}?"
    elif query.kind == "adjacency":
        question = f"Are {a} and {b} directly causally related in either direction?"
    else:
        raise ValueError("this experiment accepts local claims only")
    glossary = "\n".join(node["id"] + " = " + node["description"] for node in metadata["nodes"])
    return [dict(role="system", content=(
        "Report causal knowledge you already know, interpreted in the supplied clinical context. "
        "The descriptions identify the variables; they are not the entirety of your available knowledge. "
        "A direct relationship means an influence not mediated by another variable in the supplied set. "
        "Adjacency means such a direct relationship in either direction. "
        "Unknown relationships require abstention. Lack of knowledge is not evidence that a relationship is absent. "
        "Return a single JSON object with fields answer and confidence and no additional text. "
        "The answer is yes, no, or abstain. For yes or no, confidence is your probability that the selected "
        "answer is correct, between 0.5 and 1. For abstain, confidence is null.")),
        dict(role="user", content=metadata["context"] + "\n\n" + glossary + "\n\n" + question)]


def parse_response(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("unfinished response")
    content = choice["message"].get("content") or ""
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", content):
        try:
            item, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "answer" in item:
            candidates.append(item)
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous answer object")
    item = candidates[0]
    if set(item) != {"answer", "confidence"}:
        raise ValueError("unexpected response fields")
    answer = item["answer"]
    if answer == "abstain":
        return QueryResponse(None, None, abstention_reason="model-abstained")
    if answer not in ("yes", "no") or isinstance(item["confidence"], bool) or not isinstance(item["confidence"], (int, float)):
        raise ValueError("invalid answer or confidence")
    return QueryResponse(int(answer == "yes"), float(item["confidence"]), treatment="soft_evidence")


class Transport:
    def __init__(self, directory, seed):
        self.directory, self.seed = directory, seed
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key.startswith("sk-or-") or any(c.isspace() for c in self.key):
            raise ValueError("OPENROUTER_API_KEY missing or malformed")

    def query(self, messages):
        request = request_for(messages, self.seed)
        digest = fingerprint(request)
        for attempt in range(3):
            path = self.directory / "requests" / f"{digest}_{attempt}.json"
            with lock(self.directory, "spending"):
                if path.exists():
                    saved = read_json(path)
                else:
                    reserved = reservation(request)
                    if spend(self.directory)["accounted_usd"] + reserved > 1.0:
                        raise RuntimeError("API spending ceiling reached")
                    saved = dict(request=request, request_sha256=digest, seed=self.seed,
                                 reserved_usd=reserved, state="dispatched", started=time.time())
                    write_json(path, saved, exclusive=True)
                    saved = None
            if saved is None:
                saved = read_json(path)
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(request).encode(), headers={"Authorization": "Bearer " + self.key,
                                                              "Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=120) as response:
                        saved["response"] = json.load(response)
                    saved["state"] = "completed"
                except Exception as error:
                    saved.update(state="failed", error_type=type(error).__name__, status_code=getattr(error, "code", None))
                saved["seconds"] = time.time() - saved["started"]
                with lock(self.directory, "spending"):
                    write_json(path, saved)
            if "response" in saved:
                response = saved["response"]
                if response.get("model") not in (MODEL, MODEL + "-20260310") or response.get("provider", "").lower() != "deepinfra":
                    raise RuntimeError("API returned a different model or provider")
                usage = response.get("usage") or {}
                if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
                    raise RuntimeError("API enabled reasoning despite the frozen request")
                return response, str(path.relative_to(self.directory))
            if saved.get("status_code") not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError("API request failed or completion is uncertain; reservation retained")
            time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")


class Provider:
    def __init__(self, directory, seed, metadata, transport):
        self.directory, self.seed, self.metadata, self.transport = directory, seed, metadata, transport
        self.slot = 0

    def __call__(self, query):
        self.slot += 1
        if self.slot > 20:
            raise ValueError("query budget exceeded")
        messages = messages_for(query, self.metadata)
        records, paths = [], []
        for attempt in range(2):
            response, path = self.transport.query(messages)
            records.append(response)
            paths.append(path)
            try:
                parsed = parse_response(response)
                break
            except (KeyError, IndexError, TypeError, ValueError):
                if attempt == 1:
                    parsed = QueryResponse(None, None, abstention_reason="malformed-after-repair")
                    break
                text = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                messages = messages + [dict(role="assistant", content=text), dict(role="user", content=(
                    "Return only the final JSON object with answer and confidence. Do not include reasoning "
                    "or examples. Abstain if you do not know. Use the same question."))]
        usages = [r.get("usage") or {} for r in records]
        answer = QueryResponse(parsed.answer, parsed.reported_confidence,
            prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usages),
            completion_tokens=sum(u.get("completion_tokens", 0) for u in usages), attempts=len(records),
            abstention_reason=parsed.abstention_reason, audit=dict(request_paths=paths), treatment=parsed.treatment)
        write_json(self.directory / "answers" / f"{self.seed}_{self.slot:02d}.json",
                   dict(seed=self.seed, slot=self.slot, query=asdict(query), response=asdict(answer)))
        write_json(self.directory / "status" / f"{self.seed}.json",
                   dict(state="running", phase="active", seed=self.seed, answered_slots=self.slot, updated=time.time()))
        print(json.dumps(dict(seed=self.seed, query=self.slot, kind=query.kind, answer=answer.answer,
                              confidence=answer.confidence)), flush=True)
        return answer


def encode_graph(graph):
    return graph_edges(graph)["directed"]


def encode_distribution(distribution):
    return [dict(edges=encode_graph(g), weight=float(w)) for g, w in sorted(distribution.items())]


def initialize(directory, seed, score):
    path = directory / "initializations" / f"{seed}.json"
    if path.exists():
        saved = read_json(path)
        if saved["manifest_sha256"] != sha256(directory / "manifest.json"):
            raise ValueError("initialization manifest mismatch")
        return saved
    started = time.perf_counter()
    saved_ges = read_json(directory / "inputs" / f"ges_{seed}.json")
    partial = PartialGraph.from_matrix(np.array(saved_ges["presentation_endpoints"]), "cpdag")
    graphs, diagnostics = sample_completions_with_diagnostics(partial, "dag", 200, stable_seed("alarm-active-completions", seed))
    compatible = tuple(g for g in graphs if max(p.bit_count() for p in parent_masks(g)) <= CONFIG.max_parents)
    if not compatible:
        raise ValueError("no GES completion satisfies the parent bound")
    completion_seconds = time.perf_counter() - started
    result = initialize_score_particles(score, compatible, stable_seed("alarm-active-initialization", seed),
        particle_count=CONFIG.particle_count, updates_per_particle=CONFIG.initialization_updates,
        warm_start_score_mass=CONFIG.warm_start_score_mass, ordinary_kernel=CONFIG.ordinary_kernel,
        proposal_cache_size=CONFIG.proposal_cache_size)
    saved = dict(seed=seed, manifest_sha256=sha256(directory / "manifest.json"),
        particles=[encode_graph(g) for g in result.particles], weights=result.weights,
        distribution=encode_distribution(result.distribution), score=score.metadata(),
        completion=dict(available=len(graphs), compatible=len(compatible), rejected=len(graphs)-len(compatible),
                        diagnostics=asdict(diagnostics), seconds=completion_seconds),
        acceptance_rate=result.acceptance_rate, accepted_moves=result.accepted_moves,
        attempted_moves=result.attempted_moves, wall_seconds=time.perf_counter()-started,
        saved_before_reference_evaluation=True)
    write_json(path, saved)
    return saved


def run(identifier, seed):
    directory = root(identifier)
    verify(directory)
    with lock(directory, f"seed-{seed}"):
        output = directory / "predictions" / f"{seed}.json"
        if output.exists():
            report(identifier)
            return
        def timeout(signum, frame):
            raise TimeoutError("trajectory reached its task time limit; resume from saved initialization and cached answers")
        signal.signal(signal.SIGTERM, timeout)
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(1740)
        try:
            transport = Transport(directory, seed)
            # Check compute-host connectivity without spending on a model call.
            with urllib.request.urlopen(ENDPOINTS, timeout=20) as response:
                if response.status != 200:
                    raise RuntimeError("API connectivity check failed")
            write_json(directory / "status" / f"{seed}.json", dict(state="running", phase="initialization", seed=seed, answered_slots=0))
            presentation = read_json(directory / "inputs/presentation.json")
            metadata = read_json(directory / "inputs/metadata.json")
            names = read_json(directory / "inputs/encoding.json")["names"]
            order = presentation["order"]
            if [n["id"] for n in metadata["nodes"]] != [presentation["mapping"][names[i]] for i in order]:
                raise ValueError("metadata does not match presentation column order")
            data = np.load(directory / f"inputs/observations_{seed}.npy", allow_pickle=False)[:, order]
            score = BDeuScore(data, equivalent_sample_size=1.0, structure_prior=1.0, max_parents=CONFIG.max_parents)
            initial = initialize(directory, seed, score)
            provider = Provider(directory, seed, metadata, transport)
            started = time.perf_counter()
            active = active_query_smc(None, score, provider, query_pool=group_i_query_pool(37), budgets=BUDGETS,
                confidence=0.9, treatment="soft_evidence", policy="eig", soft_confidence_cap=CONFIG.soft_confidence_cap,
                seed=stable_seed("alarm-active-trajectory", seed), ess_threshold_fraction=CONFIG.ess_threshold_fraction,
                rejuvenation_steps=CONFIG.rejuvenation_steps, guided_mixture_probability=CONFIG.guided_mixture_probability,
                proposal_cache_size=CONFIG.proposal_cache_size, ordinary_kernel=CONFIG.ordinary_kernel,
                initial_particles=tuple(graph_from_edges(37, directed=e) for e in initial["particles"]),
                initial_weights=initial["weights"])
            if len(active.steps) != 20 or provider.slot != 20:
                raise ValueError("trajectory did not consume exactly 20 query slots")
            write_json(output, dict(seed=seed, manifest_sha256=sha256(directory / "manifest.json"),
                distributions={str(b): encode_distribution(d) for b, d in active.distributions.items()},
                steps=[asdict(step) for step in active.steps], timing=asdict(active.timing),
                resampling_count=active.resampling_count, support_recovery_count=active.support_recovery_count,
                answered_query_count=active.answered_query_count, abstention_count=active.abstention_count,
                wall_seconds=time.perf_counter()-started, saved_before_reference_evaluation=True))
            write_json(directory / "status" / f"{seed}.json", dict(state="complete", seed=seed, answered_slots=20))
        except BaseException as error:
            state_path = directory / "status" / f"{seed}.json"
            state = read_json(state_path) if state_path.exists() else dict(seed=seed, answered_slots=0)
            write_json(state_path, dict(state, state="timeout" if isinstance(error, TimeoutError) else "failed",
                                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            raise
        finally:
            signal.alarm(0)
            report(identifier)


def distribution_scores(records, order, reference):
    if not records or any(not math.isfinite(r["weight"]) or r["weight"] < 0 for r in records):
        raise ValueError("invalid distribution")
    if not math.isclose(sum(r["weight"] for r in records), 1.0, abs_tol=1e-9):
        raise ValueError("weights do not sum to one")
    # Ties use lexicographically smallest canonical edge list, independent of truth.
    def key(r):
        return (-r["weight"], sorted((order[a], order[b]) for a, b in r["edges"]))
    selected = min(records, key=key)
    scores = evaluate(canonical_matrix(edge_matrix(len(order), selected["edges"]), order), reference)
    truth_matrix = edge_matrix(len(order), reference["edges"])
    expected = sum(r["weight"] * np.count_nonzero(
        canonical_matrix(edge_matrix(len(order), r["edges"]), order) != truth_matrix
    ) / 2 for r in records)
    return dict(scores, posterior_expected_distance=expected, support_size=len(records), map_mass=selected["weight"])


def report(identifier):
    directory = root(identifier)
    with lock(directory, "report"):
        verify(directory)
        reference = read_json(directory / "evaluation_only/reference.json")
        truth = {tuple(e) for e in reference["edges"]}
        order = read_json(directory / "inputs/presentation.json")["order"]
        rows, states, claims, runtimes = [], [], [], []
        for seed in range(5):
            state_path = directory / "status" / f"{seed}.json"
            states.append(read_json(state_path) if state_path.exists() else dict(seed=seed, state="pending", answered_slots=0))
            initial_path = directory / "initializations" / f"{seed}.json"
            if initial_path.exists():
                initial = read_json(initial_path)
                rows.append(dict(seed=seed, budget=0, **distribution_scores(initial["distribution"], order, reference)))
            path = directory / "predictions" / f"{seed}.json"
            if not path.exists():
                continue
            prediction = read_json(path)
            if prediction["manifest_sha256"] != sha256(directory / "manifest.json"):
                raise ValueError("prediction manifest mismatch")
            ges_seconds = read_json(directory / f"inputs/ges_{seed}.json")["wall_seconds"]
            runtimes.append(dict(seed=seed, ges_seconds=ges_seconds,
                initialization_seconds=initial["wall_seconds"], active_seconds=prediction["wall_seconds"],
                timing=prediction["timing"], cold_equivalent_seconds=ges_seconds+initial["wall_seconds"]+prediction["wall_seconds"]))
            for budget, records in prediction["distributions"].items():
                rows.append(dict(seed=seed, budget=int(budget), **distribution_scores(records, order, reference)))
            for step in prediction["steps"]:
                query = step["query"]
                a, b = order[query["source"]], order[query["target"]]
                value = (a, b) in truth or (query["kind"] == "adjacency" and (b, a) in truth)
                correct = None if step["answer"] is None else int(step["answer"] == int(value))
                confidence = step["reported_confidence"]
                claims.append(dict(seed=seed, slot=step["budget"], kind=query["kind"], answer=step["answer"],
                    truth=int(value), correct=correct, confidence=confidence,
                    brier=None if correct is None else (confidence-correct)**2))
        answered = [c for c in claims if c["correct"] is not None]
        quality = dict(attempted=len(claims), answered=len(answered), abstained=len(claims)-len(answered),
                       coverage=len(answered)/len(claims) if claims else None,
                       accuracy=statistics.mean(c["correct"] for c in answered) if answered else None,
                       brier=statistics.mean(c["brier"] for c in answered) if answered else None)
        summary = {}
        for budget in (0, *BUDGETS):
            subset = [r for r in rows if r["budget"] == budget]
            summary[str(budget)] = dict(completed=len(subset))
            for metric in ("adjacency_f1", "directed_f1", "distance", "edges", "posterior_expected_distance"):
                values = [r[metric] for r in subset]
                summary[str(budget)][metric] = dict(mean=statistics.mean(values) if values else None,
                    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)
        with lock(directory, "spending"):
            costs = spend(directory)
        baselines = read_json(directory / "evaluation_only/baselines.json")
        result = dict(completed=sum(s["state"]=="complete" for s in states), states=states,
                      summary=summary, records=rows, claims=claims, claim_quality=quality, api=costs,
                      baselines=baselines["summary"], runtimes=runtimes)
        result["metric_scale"] = "unnormalized"
        write_json(directory / "report.json", result)
        lines = [f'Active CLAIM-FUSE on ALARM\n\n{result["completed"]}/5 trajectories complete, with 20 local query slots per trajectory.',
                 "", "Values are mean (standard error) across completed seeds. Higher F1 and lower distance are better.", "",
                 "| Method | Seeds | Adjacency F1 | Directed F1 | Endpoint distance | Edges |",
                 "| --- | --- | --- | --- | --- | --- |"]
        def cell(x):
            if x["mean"] is None:
                return "pending"
            return f'{x["mean"]:.4f}' + (f' ({x["se"]:.4f})' if x["se"] is not None else "")
        for method, record in baselines["summary"].items():
            lines.append("| " + " | ".join([method, str(record["completed"]), *[cell(record[k]) for k in ("adjacency_f1", "directed_f1", "distance", "edges")]]) + " |")
        for budget, record in summary.items():
            lines.append("| " + " | ".join([f"CLAIM-FUSE budget {budget}", str(record["completed"]), *[cell(record[k]) for k in ("adjacency_f1", "directed_f1", "distance", "edges")]]) + " |")
        lines += ["", "The zero query row uses the GES initialized data score particles. CLAIM-FUSE rows score the highest mass DAG with canonical lexicographic tie breaking. PC and GES retain raw partial endpoints. Posterior expected distance and individual seed results are recorded separately in the JSON report.",
                  "", f'Answered {quality["answered"]} of {quality["attempted"]} completed trajectory slots. Accuracy {quality["accuracy"]}. Brier score {quality["brier"]}. Stated confidence is not assumed calibrated.',
                  "", f'API accounted cost USD {costs["accounted_usd"]:.6f} across {costs["physical_requests"]} requests, including repairs and retained uncertain reservations. Spending ceiling USD 1.',
                  "", "Trajectory progress", ""]
        lines += [f'Seed {s["seed"]} {s["state"]}, {s.get("answered_slots",0)}/20 slots. {s.get("error", "")}' for s in states]
        write_text(directory / "report.md", "\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, choices=range(5))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    elif args.command == "report":
        report(args.run_id)
    else:
        if args.seed is None:
            parser.error("run requires --seed")
        run(args.run_id, args.seed)


if __name__ == "__main__":
    main()
