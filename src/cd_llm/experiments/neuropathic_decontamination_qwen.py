"""Locked opaque-identifier neuropathic benchmark with Qwen3.5-9B."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import math
import sqlite3
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cd_llm.baselines.causal_order2025 import (
    CausalOrderInfeasibleError,
    LocalVLLMConfig,
    PCOrderConfig,
    QWEN35_MODEL,
    QWEN35_REVISION,
    TripletConfig,
    TripletExpertResult,
    elicit_triplet_order_local_vllm,
    fit_pc_with_causal_order,
    query_pair_fallbacks_local_vllm,
)
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.baselines.guess2graph2026 import (
    CachedOpenAISkeletonExpert,
    LLMExpertSettings,
    build_skeleton_prompt,
    run_gpc_guess_skeleton,
)
from cd_llm.baselines.jiralerspong2024 import Completion, run_bfs
from cd_llm.baselines.long2023 import ambiguous_pairs, openrouter_expert, run_long
from cd_llm.baselines.scp2025 import (
    CachedOpenAIJudge,
    LLMSettings,
    PCBootstrapProbabilities,
    build_decision_prompt,
    build_knowledge_prompt,
    build_pc_background_knowledge,
    ceg_fuse_point_metrics,
)
from cd_llm.core.finite_support import (
    Endpoint,
    Graph,
    PartialGraph,
    distribution_metrics,
    graph_edges,
    graph_from_edges,
    point_prediction_metrics,
    stable_seed,
)
from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
from cd_llm.experiments.claim_fuse_smc_eig import _trace
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json
from cd_llm.experiments.neuropathic_active_smc_qwen import SMC_CONFIG, _truth
from cd_llm.experiments.neuropathic_causal_order_subnetwork import _expert_record
from cd_llm.experiments.neuropathic_decontamination import (
    ACTIVE_BUDGET,
    DATA_SEEDS,
    DOMAIN_CONTEXT,
    EVIDENCE_TREATMENT,
    OPAQUE_NAMES,
    SAMPLE_SIZES,
    SELECTION_CONFIDENCE,
    SOFT_CONFIDENCE_CAP,
    SOURCE_CORPUS_PATH,
    SOURCE_STATISTICS_ROOT,
    audit_model_text,
    build_spec,
)
from cd_llm.experiments.neuropathic_scp2025 import run_discovery
from cd_llm.methods.active_query import active_query_smc
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.methods.llm_claim_confidence import (
    CachedLocalChatClient,
    QwenClaimConfidenceProvider,
    confidence_metrics,
    load_document_corpus,
)
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "neuropathic_decontamination_qwen"
STUDY_VERSION = 1
MODEL_DISPLAY = "Qwen3.5-9B"
RESULTS_ROOT = WORKSPACE_ROOT / "results"
SOURCE_BENCHMARK_ROOT = (
    WORKSPACE_ROOT / "results" / "20260903T011254Z" / "neuropathic_decontamination" / "benchmark"
)
DEFAULT_RUNTIME_MANIFEST = (
    WORKSPACE_ROOT
    / "generated"
    / "runtime"
    / "20260825T082523Z"
    / "qwen35_neuropathic20_manifest.json"
)
CONFIDENCE_STRATEGIES = ("stated", "consistency", "grounded")
GPC_METHOD = f"gPC-Guess + {MODEL_DISPLAY}"
SCP_METHOD = f"SCP Pattern 2 + {MODEL_DISPLAY}"
BFS_METHOD = f"BFS+Pearson + {MODEL_DISPLAY}"
ORDER_METHOD = f"Causal Order+PC + {MODEL_DISPLAY}"
LONG_METHOD = f"PC+Long-Srisk + {MODEL_DISPLAY}"
CLAIM_FUSE_METHODS = {
    "stated": f"Claim-Fuse-SMC-Active + {MODEL_DISPLAY} (stated confidence)",
    "consistency": f"Claim-Fuse-SMC-Active + {MODEL_DISPLAY} (semantic consistency)",
    "grounded": f"Claim-Fuse-SMC-Active + {MODEL_DISPLAY} (document grounded)",
}
METHODS = (GPC_METHOD, SCP_METHOD, BFS_METHOD, ORDER_METHOD, LONG_METHOD, *CLAIM_FUSE_METHODS.values())
SCP_CONDITIONS = tuple((seed, size) for seed in DATA_SEEDS for size in SAMPLE_SIZES)
SMOKE_SCP_PAIRS = tuple((OPAQUE_NAMES[i], OPAQUE_NAMES[j]) for i in range(4) for j in range(4) if i != j) + (
    (OPAQUE_NAMES[4], OPAQUE_NAMES[5]),
    (OPAQUE_NAMES[5], OPAQUE_NAMES[4]),
    (OPAQUE_NAMES[6], OPAQUE_NAMES[7]),
    (OPAQUE_NAMES[7], OPAQUE_NAMES[6]),
)


def _root(output_run: str, results_root: Path = RESULTS_ROOT) -> Path:
    return results_root / run_id(output_run) / STUDY


def validate_runtime_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    identity = (payload.get("model"), payload.get("model_revision"))
    if identity != (QWEN35_MODEL, QWEN35_REVISION):
        raise ValueError(f"Qwen runtime identity differs: {identity!r}")
    for key in ("checkpoint_path", "image_path"):
        value = Path(str(payload.get(key, "")))
        if not value.exists():
            raise FileNotFoundError(f"runtime {key} is unavailable: {value}")
    return payload


def _write_or_validate(path: Path, payload: Mapping[str, Any], keys: Sequence[str]) -> None:
    if path.exists():
        old = read_json(path)
        if any(old.get(key) != payload.get(key) for key in keys):
            raise ValueError(f"existing artifact identity differs: {path}")
        return
    write_json(path, dict(payload), exclusive=True)


def prepare(output_run: str, results_root: Path = RESULTS_ROOT) -> Path:
    root = _root(output_run, results_root)
    benchmark = root / "benchmark"
    benchmark.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    source_manifest_path = SOURCE_BENCHMARK_ROOT / "evaluation_manifest.json"
    source_manifest = read_json(source_manifest_path)
    datasets: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    for seed, size in SCP_CONDITIONS:
        source_data = SOURCE_BENCHMARK_ROOT / "data" / f"seed_{seed:02d}_n{size}.csv"
        source_entry = next(
            item for item in source_manifest["datasets"]
            if int(item["seed"]) == seed and int(item["sample_size"]) == size
        )
        if sha256(source_data) != source_entry["sha256"]:
            raise ValueError(f"source data checksum differs: {source_data}")
        frame = pd.read_csv(source_data)
        if tuple(frame.columns) != spec.names or frame.shape != (size, 20):
            raise ValueError(f"source data layout differs: {source_data}")
        stats_path = SOURCE_STATISTICS_ROOT / "statistics" / f"seed_{seed:02d}_n{size}.json"
        stats = read_json(stats_path)
        if (int(stats["seed"]), int(stats["sample_size"])) != (seed, size):
            raise ValueError(f"source statistics identity differs: {stats_path}")
        if len(stats.get("particles", ())) != SMC_CONFIG.particle_count:
            raise ValueError(f"source particle count differs: {stats_path}")
        datasets.append({
            "seed": seed,
            "sample_size": size,
            "path": str(source_data.resolve()),
            "sha256": sha256(source_data),
        })
        statistics.append({
            "seed": seed,
            "sample_size": size,
            "path": str(stats_path.resolve()),
            "sha256": sha256(stats_path),
        })
    for seed in DATA_SEEDS:
        small = pd.read_csv(SOURCE_BENCHMARK_ROOT / "data" / f"seed_{seed:02d}_n200.csv")
        large = pd.read_csv(SOURCE_BENCHMARK_ROOT / "data" / f"seed_{seed:02d}_n1000.csv")
        if not small.equals(large.iloc[:200].reset_index(drop=True)):
            raise ValueError(f"nested sample invariant failed for seed {seed}")
    corpus = load_document_corpus(SOURCE_CORPUS_PATH)
    audit_model_text((SOURCE_BENCHMARK_ROOT / "model_metadata.json").read_text(), spec)
    audit_model_text(SOURCE_CORPUS_PATH.read_text(), spec)
    payload = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "model": QWEN35_MODEL,
        "model_revision": QWEN35_REVISION,
        "methods": list(METHODS),
        "data_seeds": list(DATA_SEEDS),
        "sample_sizes": list(SAMPLE_SIZES),
        "datasets": datasets,
        "source_statistics": statistics,
        "source_benchmark_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256(source_manifest_path),
        },
        "document_corpus": {
            "path": str(SOURCE_CORPUS_PATH.resolve()),
            "sha256": sha256(SOURCE_CORPUS_PATH),
            "source_count": len(corpus),
        },
        "observational_samples_only": True,
        "intervention_labels_present": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_or_validate(
        benchmark / "evaluation_manifest.json",
        payload,
        ("study", "study_version", "model", "model_revision", "methods", "datasets", "source_statistics"),
    )
    return benchmark


def _manifest(root: Path) -> dict[str, Any]:
    return read_json(root / "benchmark" / "evaluation_manifest.json")


def _entry(root: Path, section: str, seed: int, size: int) -> tuple[Path, str]:
    record = next(
        item for item in _manifest(root)[section]
        if int(item["seed"]) == seed and int(item["sample_size"]) == size
    )
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"locked input checksum differs: {path}")
    return path, str(record["sha256"])


def _condition(root: Path, seed: int, size: int) -> tuple[pd.DataFrame, Any]:
    if (seed, size) not in SCP_CONDITIONS:
        raise ValueError("unsupported condition")
    path, _ = _entry(root, "datasets", seed, size)
    frame = pd.read_csv(path)
    spec = build_spec()
    if tuple(frame.columns) != spec.names or frame.shape != (size, 20):
        raise ValueError("opaque dataset layout differs")
    return frame, spec


def _statistics(root: Path, seed: int, size: int) -> tuple[dict[str, Any], Path, str]:
    path, digest = _entry(root, "source_statistics", seed, size)
    return read_json(path), path, digest


def _graph(payload: Mapping[str, Any]) -> Graph:
    return graph_from_edges(
        20,
        directed=payload.get("directed", ()),
        bidirected=payload.get("bidirected", ()),
        undirected=payload.get("undirected", ()),
    )


def _partial(payload: Mapping[str, Any]) -> PartialGraph:
    return PartialGraph(
        int(payload["node_count"]),
        tuple(int(value) for value in payload["marks"]),
        str(payload["kind"]),
    )


def _result_path(root: Path, method: str, seed: int, size: int) -> Path:
    slug = method.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    return root / "records" / slug / f"seed_{seed:02d}_n{size}.json"


def _record(root: Path, method: str, seed: int, size: int, metrics: Mapping[str, Any], **extra: Any) -> None:
    path = _result_path(root, method, seed, size)
    payload = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "method": method,
        "seed": seed,
        "sample_size": size,
        "model": QWEN35_MODEL,
        "model_revision": QWEN35_REVISION,
        "metrics": dict(metrics),
        "discovery_input": {"observational_samples_only": True, "intervention_labels_present": False},
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    _write_or_validate(path, payload, ("study", "study_version", "method", "seed", "sample_size", "model", "model_revision"))


def _runtime_reference(path: Path) -> dict[str, str]:
    validate_runtime_manifest(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _scp_probabilities(payload: Mapping[str, Any]) -> PCBootstrapProbabilities:
    section = payload["scp_bootstrap"]
    return PCBootstrapProbabilities(
        directed={(a, b): float(p) for a, b, p in section["directed"]},
        undirected={(a, b): float(p) for a, b, p in section["undirected"]},
        bidirected={(a, b): float(p) for a, b, p in section["bidirected"]},
    )


class CachedQwenCompleter:
    """Restart-safe synchronous adapter used by the conversational BFS baseline."""

    def __init__(self, cache_path: Path, base_url: str, seed: int, spec: Any) -> None:
        from openai import OpenAI

        self.cache = ResponseCache(cache_path)
        self.client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=0, timeout=300.0)
        self.seed = int(seed)
        self.spec = spec

    def close(self) -> None:
        self.cache.close()

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> Completion:
        _audit_supplied_messages(messages, self.spec)
        request = {
            "model": QWEN35_MODEL,
            "messages": [dict(item) for item in messages],
            "temperature": 0.7,
            "seed": self.seed,
            "max_tokens": 4096,
            "enable_thinking": False,
        }
        cached = self.cache.get(request)
        if cached is not None:
            return Completion(str(cached["content"]), 0, 0)
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=QWEN35_MODEL,
            messages=list(messages),
            temperature=0.7,
            seed=self.seed,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if str(response.model) != QWEN35_MODEL:
            raise ValueError(f"served Qwen model differs: {response.model!r}")
        content = str(response.choices[0].message.content or "")
        usage = response.usage
        prompt_tokens = int(usage.prompt_tokens if usage else 0)
        completion_tokens = int(usage.completion_tokens if usage else 0)
        self.cache.put(
            request,
            "bfs",
            response.model_dump(),
            content,
            provider_request_id=getattr(response, "id", None),
            response_model=str(response.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,
            latency_seconds=time.perf_counter() - started,
            attempts=1,
        )
        return Completion(content, prompt_tokens, completion_tokens)


def _audit_supplied_messages(messages: Sequence[Mapping[str, str]], spec: Any) -> None:
    """Audit externally supplied text while retaining native model conversation history."""

    audit_model_text(
        "\n".join(
            str(item.get("content", ""))
            for item in messages
            if item.get("role") != "assistant"
        ),
        spec,
    )


def _expert_from_record(record: Mapping[str, Any]) -> TripletExpertResult:
    node_names = tuple(record["node_names"])
    graph_payload = record["graph"]
    return TripletExpertResult(
        node_names=node_names,
        raw_edges=tuple(tuple(value) for value in record["raw_edges"]),
        graph=graph_from_edges(
            len(node_names),
            directed=graph_payload.get("directed", ()),
            bidirected=graph_payload.get("bidirected", ()),
            undirected=graph_payload.get("undirected", ()),
        ),
        edgewise_distributions={
            tuple(item["pair"]): tuple(item["probabilities"])
            for item in record["edgewise_distributions"]
        },
        subgroup_list=tuple(tuple(value) for value in record["subgroup_list"]),
        subgroup_responses=tuple(record["subgroup_responses"]),
        expert_order=tuple(record["expert_order"]),
        isolated_nodes=tuple(record["isolated_nodes"]),
        cycle_removal_threshold=record["cycle_removal_threshold"],
        upstream_revision=record["upstream_revision"],
        config=TripletConfig(subgraph_model=QWEN35_MODEL, expert_model=QWEN35_MODEL, delay=0),
        runtime_seconds=float(record["runtime_seconds"]),
    )


async def _run_causal_order(
    root: Path,
    base_url: str,
    runtime_manifest: Path,
    *,
    node_limit: int | None = None,
) -> int:
    spec = build_spec()
    nodes = spec.names if node_limit is None else spec.names[:node_limit]
    descriptions = {name: spec.descriptions[name] for name in nodes}
    suffix = "" if node_limit is None else f"_smoke_{node_limit}"
    expert_path = root / "shared" / f"causal_order_expert{suffix}.json"
    if expert_path.exists():
        expert = _expert_from_record(read_json(expert_path))
    else:
        elicitation = await elicit_triplet_order_local_vllm(
            nodes,
            f"to model {DOMAIN_CONTEXT}",
            root / "caches" / f"causal_order{suffix}.sqlite3",
            descriptions=descriptions,
            config=LocalVLLMConfig(base_url=base_url, triplet_retry_max_tokens=7680),
            minimum_parse_rate=0.95,
        )
        expert = elicitation.expert
        write_json(expert_path, {**_expert_record(expert), "audit": dict(elicitation.audit)}, exclusive=True)
    if node_limit is not None:
        if expert.node_names != tuple(nodes):
            raise ValueError("smoke causal-order node set differs")
        return 1

    references = [
        _partial(_statistics(root, seed, size)[0]["pc"])
        for seed, size in SCP_CONDITIONS
    ]
    fallback_pairs = required_fallback_pairs(expert, references)
    fallbacks, fallback_audit = await query_pair_fallbacks_local_vllm(
        spec.names,
        fallback_pairs,
        f"to model {DOMAIN_CONTEXT}",
        root / "caches" / "causal_order.sqlite3",
        descriptions=spec.descriptions,
        config=LocalVLLMConfig(base_url=base_url, pair_max_tokens=256),
    )

    def fallback(first: int, second: int) -> tuple[int, int]:
        orientation = fallbacks.get((min(first, second), max(first, second)))
        if orientation is None:
            raise CausalOrderInfeasibleError("fallback expert omitted a PC adjacency")
        return orientation

    runtime = _runtime_reference(runtime_manifest)
    for seed, size in SCP_CONDITIONS:
        output = _result_path(root, ORDER_METHOD, seed, size)
        if output.exists():
            continue
        frame, _ = _condition(root, seed, size)
        result = fit_pc_with_causal_order(
            frame.to_numpy(dtype=np.int64),
            spec.names,
            expert,
            config=PCOrderConfig(alpha=0.01, indep_test="gsq", stable=True, uc_rule=0, uc_priority=2),
            fallback_orienter=fallback,
        )
        _record(
            root,
            ORDER_METHOD,
            seed,
            size,
            point_prediction_metrics(result.graph, spec.target),
            graph=graph_edges(result.graph),
            shared_expert=str(expert_path),
            fallback_audit=fallback_audit,
            runtime_manifest=runtime,
            hyperparameters={"pc_alpha": 0.01, "ci_test": "gsq", "stable": True},
        )
    return len(fallback_pairs)


def _run_gpc_and_long(root: Path, base_url: str, runtime_manifest: Path) -> None:
    spec = build_spec()
    descriptions = dict(spec.descriptions)
    audit_model_text(
        build_skeleton_prompt(spec.names, descriptions=descriptions, domain_context=DOMAIN_CONTEXT),
        spec,
    )
    guesser = CachedOpenAISkeletonExpert(
        cache_dir=root / "caches" / "guess2graph",
        settings=LLMExpertSettings(
            model=QWEN35_MODEL,
            temperature=0.7,
            max_tokens=2000,
            enable_thinking=False,
            request_seed=stable_seed(STUDY, "guess2graph"),
        ),
        base_url=base_url,
        api_key="EMPTY",
        domain_context=DOMAIN_CONTEXT,
        descriptions=descriptions,
    )
    guesses = {seed: guesser.guess(spec.names, trial_seed=seed) for seed in DATA_SEEDS}
    metadata = tuple(f"{name} ({descriptions[name]})" for name in spec.names)
    long_expert = openrouter_expert(
        QWEN35_MODEL,
        base_url,
        stable_seed(STUDY, "long"),
        root / "caches" / "long.json",
        enable_thinking=False,
        prompt_auditor=lambda prompt: audit_model_text(prompt, spec),
    )
    pairs = sorted({
        pair
        for seed, size in SCP_CONDITIONS
        for pair in ambiguous_pairs(_partial(_statistics(root, seed, size)[0]["pc"]))
    })
    for first, second in pairs:
        long_expert(first, second, metadata[first], metadata[second])
    runtime = _runtime_reference(runtime_manifest)
    for seed, size in SCP_CONDITIONS:
        frame, _ = _condition(root, seed, size)
        data = frame.to_numpy(dtype=np.int64)
        if not _result_path(root, GPC_METHOD, seed, size).exists():
            predicted = run_gpc_guess_skeleton(
                data,
                guesses[seed].skeleton,
                alpha=0.05,
                ci_test="chisq",
                rng=np.random.default_rng(stable_seed(STUDY, "gpc", seed, size)),
            )
            marks = np.where(predicted != 0, int(Endpoint.TAIL), int(Endpoint.NULL))
            partial = PartialGraph.from_matrix(marks, "cpdag")
            _record(
                root,
                GPC_METHOD,
                seed,
                size,
                point_prediction_metrics(partial, spec.target),
                graph={"kind": "undirected", "edges": np.argwhere(np.triu(predicted, 1)).tolist()},
                expert_cache=str(guesses[seed].cache_path),
                runtime_manifest=runtime,
                hyperparameters={"ci_test": "chisq", "alpha": 0.05},
            )
        if not _result_path(root, LONG_METHOD, seed, size).exists():
            result = run_long(
                _partial(_statistics(root, seed, size)[0]["pc"]),
                metadata,
                long_expert,
                strategy="srisk",
                tolerance=0.1,
                max_extensions=50_000,
            )
            weights = np.full(len(result.final_mec), 1.0 / len(result.final_mec))
            _record(
                root,
                LONG_METHOD,
                seed,
                size,
                distribution_metrics(result.final_mec, weights, spec.target, "dag"),
                distribution=[graph_edges(graph) for graph in result.final_mec],
                query_count=len(result.expert_records),
                runtime_manifest=runtime,
                hyperparameters={"strategy": "srisk", "tolerance": 0.1, "max_extensions": 50_000},
            )


async def run_shared(output_run: str, base_url: str, runtime_manifest: Path, results_root: Path = RESULTS_ROOT) -> None:
    root = _root(output_run, results_root)
    _runtime_reference(runtime_manifest)
    if not (root / "benchmark" / "evaluation_manifest.json").is_file():
        raise FileNotFoundError("prepare must complete before shared inference")
    fallback_count = await _run_causal_order(root, base_url, runtime_manifest)
    _run_gpc_and_long(root, base_url, runtime_manifest)
    _write_or_validate(
        root / "shared" / "complete.json",
        {
            "status": "passed",
            "gpc_guess_count": len(DATA_SEEDS),
            "causal_order_triplet_count": math.comb(20, 3),
            "causal_order_fallback_count": fallback_count,
            "condition_record_count": 3 * len(SCP_CONDITIONS),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        ("status", "gpc_guess_count", "causal_order_triplet_count", "condition_record_count"),
    )


def run_bfs_stage(output_run: str, base_url: str, runtime_manifest: Path, results_root: Path = RESULTS_ROOT) -> None:
    root = _root(output_run, results_root)
    runtime = _runtime_reference(runtime_manifest)
    spec = build_spec()
    for seed, size in SCP_CONDITIONS:
        if _result_path(root, BFS_METHOD, seed, size).exists():
            continue
        frame, _ = _condition(root, seed, size)
        complete = CachedQwenCompleter(
            root / "caches" / "bfs" / f"seed_{seed:02d}_n{size}.sqlite3",
            base_url,
            stable_seed(STUDY, "bfs", seed, size),
            spec,
        )
        try:
            edges, usage = run_bfs(
                frame,
                complete,
                include_statistics=True,
                descriptions=spec.descriptions,
                domain_context=DOMAIN_CONTEXT,
                opaque_identifiers=True,
                empty_root_policy="empty_graph",
            )
        finally:
            complete.close()
        graph = graph_from_edges(20, directed=edges)
        _record(
            root,
            BFS_METHOD,
            seed,
            size,
            point_prediction_metrics(graph, spec.target),
            graph=graph_edges(graph),
            usage=usage,
            inference_status="empty-root-response" if usage.get("empty_root_response") else "complete",
            runtime_manifest=runtime,
            hyperparameters={
                "temperature": 0.7,
                "include_pearson_correlations": True,
                "empty_root_policy": "empty_graph",
            },
        )


def _collect_scp(
    judge: CachedOpenAIJudge,
    spec: Any,
    size: int,
    bootstrap: PCBootstrapProbabilities,
    pairs: Sequence[tuple[str, str]],
    max_workers: int = 16,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    def query(pair: tuple[str, str]) -> dict[str, Any]:
        source, target = pair
        prompt = build_knowledge_prompt(
            pattern=2,
            source=source,
            target=target,
            nodes=spec.names,
            method="PC with G-squared conditional-independence tests",
            dataset_description=f"{size} binary clinical observations",
            pc_bootstrap_probabilities=bootstrap,
            theme=DOMAIN_CONTEXT,
            variable_descriptions=spec.descriptions,
        )
        audit_model_text(prompt, spec)
        explanation = judge.explanation(prompt)
        decision = build_decision_prompt(
            knowledge_prompt=prompt,
            explanation=explanation,
            source=source,
            target=target,
        )
        _audit_scp_decision_prompt(decision, spec)
        probability, trials = judge.yes_probability(decision)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("SCP returned an invalid probability")
        return {
            "source": source,
            "target": target,
            "mean_yes_probability": probability,
            "trial_probabilities": trials,
            "explanation": explanation,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(query, pairs))
    return records, {
        (str(item["source"]), str(item["target"])): float(item["mean_yes_probability"])
        for item in records
    }


_SCP_DECISION_PREFIX = "An expert was asked the question below:\n"
_SCP_DECISION_MIDDLE = "\n\nThen, the expert replied with its domain knowledge:\n"
_SCP_DECISION_SUFFIX = "\n\nConsidering objectively this discussion above, "


def _split_scp_decision_prompt(decision: str) -> tuple[str, str, str]:
    """Separate supplied SCP text from the model-generated explanation."""
    if not decision.startswith(_SCP_DECISION_PREFIX):
        raise ValueError("SCP decision prompt has an unexpected prefix")
    body = decision[len(_SCP_DECISION_PREFIX):]
    knowledge, middle, remainder = body.partition(_SCP_DECISION_MIDDLE)
    if not middle:
        raise ValueError("SCP decision prompt is missing its explanation marker")
    explanation, suffix, question = remainder.rpartition(_SCP_DECISION_SUFFIX)
    if not suffix:
        raise ValueError("SCP decision prompt is missing its decision marker")
    supplied = (
        _SCP_DECISION_PREFIX
        + knowledge
        + _SCP_DECISION_MIDDLE
        + _SCP_DECISION_SUFFIX
        + question
    )
    return knowledge, explanation, supplied


def _audit_scp_decision_prompt(decision: str, spec: Any) -> str:
    """Audit supplied SCP content while retaining generated text as provenance."""
    knowledge, explanation, supplied = _split_scp_decision_prompt(decision)
    audit_model_text(knowledge, spec)
    audit_model_text(supplied, spec)
    return explanation


def _pag_record(graph: Any) -> dict[str, Any]:
    return {
        "nodes": list(graph.nodes),
        "edges": [
            {
                "left": left,
                "right": right,
                "left_mark": graph.endpoints(left, right)[0],
                "right_mark": graph.endpoints(left, right)[1],
            }
            for left, right in graph.all_edges()
        ],
    }


def run_scp_task(
    output_run: str,
    task_index: int,
    base_url: str,
    runtime_manifest: Path,
    results_root: Path = RESULTS_ROOT,
) -> None:
    if not 0 <= task_index < len(SCP_CONDITIONS):
        raise ValueError("SCP task index is outside 0..19")
    root = _root(output_run, results_root)
    runtime = _runtime_reference(runtime_manifest)
    seed, size = SCP_CONDITIONS[task_index]
    if _result_path(root, SCP_METHOD, seed, size).exists():
        return
    frame, spec = _condition(root, seed, size)
    stats, stats_path, stats_hash = _statistics(root, seed, size)
    judge = CachedOpenAIJudge(
        cache_dir=root / "caches" / "scp" / f"seed_{seed:02d}_n{size}",
        settings=LLMSettings(
            model=QWEN35_MODEL,
            temperature=0.7,
            probability_samples=5,
            top_logprobs=5,
            probability_protocol="paper_logprobs",
            enable_thinking=False,
            request_seed=stable_seed(STUDY, "scp", seed, size),
        ),
        base_url=base_url,
        api_key="EMPTY",
    )
    pairs = tuple((source, target) for source in spec.names for target in spec.names if source != target)
    records, probabilities = _collect_scp(judge, spec, size, _scp_probabilities(stats), pairs)
    knowledge, applications = build_pc_background_knowledge(
        spec.names,
        probabilities,
        forbid_threshold=0.05,
        require_threshold=0.95,
    )
    final = run_discovery(
        frame=frame,
        level=0,
        alpha=0.05,
        fci_depth=-1,
        fci_max_path_length=-1,
        show_progress=False,
        background_knowledge=knowledge,
    )
    _record(
        root,
        SCP_METHOD,
        seed,
        size,
        ceg_fuse_point_metrics(final.graph, final.graph, "dag", truth_graph=spec.target),
        graph=_pag_record(final.graph),
        applications=[asdict(item) for item in applications],
        probability_records=records,
        query_count=len(records) * 6,
        source_statistics={"path": str(stats_path.resolve()), "sha256": stats_hash},
        runtime_manifest=runtime,
        hyperparameters={
            "pattern": 2,
            "pc_bootstrap_count": 1000,
            "temperature": 0.7,
            "probability_samples": 5,
            "probability_protocol": "paper_logprobs",
            "forbid_threshold": 0.05,
            "require_threshold": 0.95,
        },
    )


def run_claim_fuse_stage(
    output_run: str,
    base_url: str,
    runtime_manifest: Path,
    results_root: Path = RESULTS_ROOT,
) -> None:
    root = _root(output_run, results_root)
    runtime = _runtime_reference(runtime_manifest)
    spec = build_spec()
    sources = load_document_corpus(SOURCE_CORPUS_PATH)
    for strategy in CONFIDENCE_STRATEGIES:
        cache = ResponseCache(root / "caches" / "claim_fuse" / f"{strategy}.sqlite3")
        chat = CachedLocalChatClient(
            cache,
            LocalVLLMConfig(
                base_url=base_url,
                seed=stable_seed(STUDY, "claim-fuse", strategy),
                concurrency=1,
                temperature=0.0,
                triplet_max_tokens=256,
                pair_max_tokens=256,
            ),
            prompt_auditor=lambda prompt: audit_model_text(prompt, spec),
        )
        try:
            provider = QwenClaimConfidenceProvider(
                strategy,
                spec.names,
                chat,
                sources=sources if strategy == "grounded" else (),
                descriptions=spec.descriptions,
                domain_context=DOMAIN_CONTEXT,
            )
            for seed, size in SCP_CONDITIONS:
                method = CLAIM_FUSE_METHODS[strategy]
                if _result_path(root, method, seed, size).exists():
                    continue
                frame, _ = _condition(root, seed, size)
                stats, stats_path, stats_hash = _statistics(root, seed, size)
                particles = tuple(_graph(item) for item in stats["particles"])
                weights = tuple(float(value) for value in stats["weights"])
                if len(particles) != SMC_CONFIG.particle_count or not math.isclose(
                    math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-10
                ):
                    raise ValueError("source particle initialization is invalid")
                score = BDeuScore(
                    frame.to_numpy(dtype=np.int64),
                    equivalent_sample_size=1.0,
                    cardinalities=(2,) * 20,
                    max_parents=SMC_CONFIG.max_parents,
                    structure_prior=1.0,
                )
                active = active_query_smc(
                    None,
                    score,
                    provider,
                    budgets=(ACTIVE_BUDGET,),
                    confidence=SELECTION_CONFIDENCE,
                    treatment=EVIDENCE_TREATMENT,
                    soft_confidence_cap=SOFT_CONFIDENCE_CAP,
                    policy="eig",
                    seed=stable_seed("neuropathic_decontamination", "active", seed, size),
                    ess_threshold_fraction=SMC_CONFIG.ess_threshold_fraction,
                    rejuvenation_steps=SMC_CONFIG.rejuvenation_steps,
                    guided_mixture_probability=SMC_CONFIG.guided_mixture_probability,
                    proposal_cache_size=SMC_CONFIG.proposal_cache_size,
                    initial_particles=particles,
                    initial_weights=weights,
                )
                distribution = active.distributions[ACTIVE_BUDGET]
                graphs = tuple(sorted(distribution))
                probabilities = np.asarray([distribution[graph] for graph in graphs])
                trace = _trace(active)
                events = [
                    (int(int(step["answer"]) == _truth(step["query"], spec.target)), float(step["reported_confidence"]))
                    for step in trace
                    if step["valid_response"]
                ]
                _record(
                    root,
                    method,
                    seed,
                    size,
                    distribution_metrics(graphs, probabilities, spec.target, "dag"),
                    distribution=[
                        {"graph": graph_edges(graph), "weight": float(distribution[graph])}
                        for graph in graphs
                    ],
                    strategy=strategy,
                    trace=trace,
                    query_count=ACTIVE_BUDGET,
                    coverage=active.answered_query_count / ACTIVE_BUDGET,
                    calibration=confidence_metrics(events),
                    runtime_manifest=runtime,
                    source_statistics={"path": str(stats_path.resolve()), "sha256": stats_hash},
                    hyperparameters={
                        "claim_group": "I",
                        "confidence_strategy": strategy,
                        "active_budget": ACTIVE_BUDGET,
                        "evidence_treatment": EVIDENCE_TREATMENT,
                        "soft_confidence_cap": SOFT_CONFIDENCE_CAP,
                        "initialization": "GES-only",
                        "particle_count": SMC_CONFIG.particle_count,
                        "initialization_updates_per_particle": 192,
                        "equivalent_sample_size": 1.0,
                        "structure_prior": 1.0,
                    },
                )
        finally:
            cache.close()


async def run_smoke(
    output_run: str,
    base_url: str,
    runtime_manifest: Path,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    root = _root(output_run, results_root)
    runtime = _runtime_reference(runtime_manifest)
    if not (root / "benchmark" / "evaluation_manifest.json").is_file():
        raise FileNotFoundError("prepare must complete before smoke testing")
    spec = build_spec()
    frame, _ = _condition(root, 0, 200)
    stats, _, _ = _statistics(root, 0, 200)
    checks: dict[str, Any] = {}

    guesser = CachedOpenAISkeletonExpert(
        cache_dir=root / "smoke" / "caches" / "guess2graph",
        settings=LLMExpertSettings(
            model=QWEN35_MODEL,
            temperature=0.7,
            max_tokens=2000,
            enable_thinking=False,
            request_seed=stable_seed(STUDY, "smoke", "guess"),
        ),
        base_url=base_url,
        api_key="EMPTY",
        domain_context=DOMAIN_CONTEXT,
        descriptions=spec.descriptions,
    )
    guess = guesser.guess(spec.names, trial_seed=0)
    predicted = run_gpc_guess_skeleton(
        frame.to_numpy(dtype=np.int64),
        guess.skeleton,
        alpha=0.05,
        ci_test="chisq",
        rng=np.random.default_rng(stable_seed(STUDY, "smoke", "gpc")),
    )
    checks["gpc"] = {"node_count": int(predicted.shape[0]), "edge_count": int(np.triu(predicted, 1).sum())}

    judge = CachedOpenAIJudge(
        cache_dir=root / "smoke" / "caches" / "scp",
        settings=LLMSettings(
            model=QWEN35_MODEL,
            temperature=0.7,
            probability_samples=5,
            top_logprobs=5,
            probability_protocol="paper_logprobs",
            enable_thinking=False,
            request_seed=stable_seed(STUDY, "smoke", "scp"),
        ),
        base_url=base_url,
        api_key="EMPTY",
    )
    scp_records, _ = _collect_scp(
        judge,
        spec,
        200,
        _scp_probabilities(stats),
        SMOKE_SCP_PAIRS,
        max_workers=16,
    )
    checks["scp"] = {"pair_count": len(scp_records), "logical_request_count": len(scp_records) * 6}

    bfs_frame = frame.loc[:, spec.names[:3]]
    bfs_complete = CachedQwenCompleter(
        root / "smoke" / "caches" / "bfs.sqlite3",
        base_url,
        stable_seed(STUDY, "smoke", "bfs"),
        spec,
    )
    try:
        bfs_edges, bfs_usage = run_bfs(
            bfs_frame,
            bfs_complete,
            include_statistics=True,
            descriptions={name: spec.descriptions[name] for name in bfs_frame.columns},
            domain_context=DOMAIN_CONTEXT,
            opaque_identifiers=True,
            empty_root_policy="empty_graph",
        )
    finally:
        bfs_complete.close()
    graph_from_edges(3, directed=bfs_edges)
    checks["bfs"] = {"node_count": 3, "edge_count": len(bfs_edges), **bfs_usage}

    checks["causal_order"] = {"triplet_count": await _run_causal_order(root / "smoke", base_url, runtime_manifest, node_limit=3)}

    pc = _partial(stats["pc"])
    pair = ambiguous_pairs(pc)[0]
    metadata = tuple(f"{name} ({spec.descriptions[name]})" for name in spec.names)
    long_expert = openrouter_expert(
        QWEN35_MODEL,
        base_url,
        stable_seed(STUDY, "smoke", "long"),
        root / "smoke" / "caches" / "long.json",
        enable_thinking=False,
        prompt_auditor=lambda prompt: audit_model_text(prompt, spec),
    )
    arc, confidence, _ = long_expert(pair[0], pair[1], metadata[pair[0]], metadata[pair[1]])
    if set(arc) != set(pair) or not 0.5 <= confidence <= 1.0:
        raise ValueError("Long smoke result is invalid")
    checks["long"] = {"pair": list(pair), "arc": list(arc), "confidence": confidence}

    particles = tuple(_graph(item) for item in stats["particles"])
    weights = tuple(float(value) for value in stats["weights"])
    score = BDeuScore(
        frame.to_numpy(dtype=np.int64),
        equivalent_sample_size=1.0,
        cardinalities=(2,) * 20,
        max_parents=SMC_CONFIG.max_parents,
        structure_prior=1.0,
    )
    sources = load_document_corpus(SOURCE_CORPUS_PATH)
    checks["claim_fuse"] = {}
    for strategy in CONFIDENCE_STRATEGIES:
        cache = ResponseCache(root / "smoke" / "caches" / f"claim_fuse_{strategy}.sqlite3")
        chat = CachedLocalChatClient(
            cache,
            LocalVLLMConfig(
                base_url=base_url,
                seed=stable_seed(STUDY, "smoke", "claim-fuse", strategy),
                concurrency=1,
                temperature=0.0,
                triplet_max_tokens=256,
                pair_max_tokens=256,
            ),
            prompt_auditor=lambda prompt: audit_model_text(prompt, spec),
        )
        try:
            provider = QwenClaimConfidenceProvider(
                strategy,
                spec.names,
                chat,
                sources=sources if strategy == "grounded" else (),
                descriptions=spec.descriptions,
                domain_context=DOMAIN_CONTEXT,
            )
            result = active_query_smc(
                None,
                score,
                provider,
                budgets=(1,),
                confidence=SELECTION_CONFIDENCE,
                treatment=EVIDENCE_TREATMENT,
                soft_confidence_cap=SOFT_CONFIDENCE_CAP,
                policy="eig",
                seed=stable_seed("neuropathic_decontamination", "active", 0, 200),
                ess_threshold_fraction=SMC_CONFIG.ess_threshold_fraction,
                rejuvenation_steps=SMC_CONFIG.rejuvenation_steps,
                guided_mixture_probability=SMC_CONFIG.guided_mixture_probability,
                proposal_cache_size=SMC_CONFIG.proposal_cache_size,
                initial_particles=particles,
                initial_weights=weights,
            )
        finally:
            cache.close()
        distribution = result.distributions[1]
        if not math.isclose(math.fsum(distribution.values()), 1.0, abs_tol=1e-10):
            raise ValueError("Claim-Fuse smoke distribution is not normalized")
        checks["claim_fuse"][strategy] = {
            "query_count": len(result.steps),
            "answered_query_count": result.answered_query_count,
            "particle_count": len(distribution),
        }

    audit = audit_saved_requests(root / "smoke", spec)
    summary = {
        "study": STUDY,
        "status": "passed",
        "reportable": False,
        "runtime_manifest": runtime,
        "checks": checks,
        "prompt_audit": audit,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = root / "smoke" / "summary.json"
    _write_or_validate(path, summary, ("study", "status", "reportable", "runtime_manifest", "checks"))
    return path


def audit_saved_requests(root: Path, spec: Any) -> dict[str, Any]:
    prompt_hashes: set[str] = set()
    generated_protected_terms: set[str] = set()

    def generated_hits(text: str) -> None:
        lowered = text.casefold()
        candidates = (*spec.original_names, "neuropathic pain", "simulator", "table45", "benchmark")
        generated_protected_terms.update(value for value in candidates if value.casefold() in lowered)

    def inspect(value: Any, key: str | None = None) -> None:
        if key == "prompt" and isinstance(value, str):
            audit_model_text(value, spec)
            prompt_hashes.add(sha256_text(value))
            return
        if key == "messages" and isinstance(value, list):
            supplied = [
                item for item in value
                if isinstance(item, Mapping) and item.get("role") != "assistant"
            ]
            generated = [
                item for item in value
                if isinstance(item, Mapping) and item.get("role") == "assistant"
            ]
            text = "\n".join(str(item.get("content", "")) for item in supplied)
            audit_model_text(text, spec)
            prompt_hashes.add(sha256_text(text))
            for item in generated:
                generated_hits(str(item.get("content", "")))
            return
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                inspect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for path in sorted((root / "caches").rglob("*.json")):
        payload = read_json(path)
        request = payload.get("request") if isinstance(payload, Mapping) else None
        if "scp" in path.parts and isinstance(request, Mapping):
            prompt = request.get("prompt")
            kind = request.get("kind")
            if not isinstance(prompt, str):
                raise ValueError(f"SCP cache has no prompt: {path}")
            if kind == "explanation":
                audit_model_text(prompt, spec)
                generated_hits(str(payload.get("text", "")))
            elif kind == "yes_probability":
                generated_hits(_audit_scp_decision_prompt(prompt, spec))
            else:
                raise ValueError(f"SCP cache has an unknown request kind: {path}")
            prompt_hashes.add(sha256_text(prompt))
            continue
        inspect(payload)
    sqlite_count = 0
    for path in sorted((root / "caches").rglob("*.sqlite3")):
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT request_json, response_model, content FROM responses").fetchall()
        finally:
            connection.close()
        for request_json, response_model, content in rows:
            request = json.loads(request_json)
            inspect(request)
            generated_hits(str(content))
            if request.get("model") != QWEN35_MODEL:
                raise ValueError(f"cached request model differs in {path}")
            if response_model not in (None, QWEN35_MODEL):
                raise ValueError(f"cached response model differs in {path}")
            sqlite_count += 1
    return {
        "status": "passed",
        "unique_prompt_count": len(prompt_hashes),
        "sqlite_request_count": sqlite_count,
        "generated_protected_terms": sorted(generated_protected_terms),
    }


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _recomputed_metrics(record: Mapping[str, Any], target: Graph) -> dict[str, float]:
    method = str(record["method"])
    if method == GPC_METHOD:
        matrix = np.zeros((20, 20), dtype=np.int8)
        for left, right in record["graph"]["edges"]:
            matrix[int(left), int(right)] = matrix[int(right), int(left)] = int(Endpoint.TAIL)
        return point_prediction_metrics(PartialGraph.from_matrix(matrix, "cpdag"), target)
    if method in (BFS_METHOD, ORDER_METHOD):
        return point_prediction_metrics(_graph(record["graph"]), target)
    if method == LONG_METHOD:
        graphs = tuple(_graph(item) for item in record["distribution"])
        return distribution_metrics(graphs, np.full(len(graphs), 1.0 / len(graphs)), target, "dag")
    if method in CLAIM_FUSE_METHODS.values():
        graphs = tuple(_graph(item["graph"]) for item in record["distribution"])
        weights = np.asarray([float(item["weight"]) for item in record["distribution"]])
        return distribution_metrics(graphs, weights, target, "dag")
    if method == SCP_METHOD:
        from cd_llm.methods.fci.graph import PAGGraph

        graph = PAGGraph(tuple(record["graph"]["nodes"]))
        for edge in record["graph"]["edges"]:
            graph.add_edge(edge["left"], edge["right"], edge["left_mark"], edge["right_mark"])
        return ceg_fuse_point_metrics(graph, graph, "dag", truth_graph=target)
    raise ValueError(f"unknown method: {method}")


def aggregate(output_run: str, results_root: Path = RESULTS_ROOT) -> Path:
    root = _root(output_run, results_root)
    records = [read_json(path) for path in sorted((root / "records").glob("*/*.json"))]
    expected = {(method, seed, size) for method in METHODS for seed, size in SCP_CONDITIONS}
    identities = [(str(item["method"]), int(item["seed"]), int(item["sample_size"])) for item in records]
    actual = set(identities)
    if len(actual) != len(identities):
        raise RuntimeError("duplicate result identities found")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"aggregate is incomplete; missing={missing[:5]}, extra={extra[:5]}")
    spec = build_spec()
    fields = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
    for record in records:
        if (record.get("model"), record.get("model_revision")) != (QWEN35_MODEL, QWEN35_REVISION):
            raise ValueError("record model identity differs")
        if record.get("discovery_input") != {"observational_samples_only": True, "intervention_labels_present": False}:
            raise ValueError("record discovery input policy differs")
        recomputed = _recomputed_metrics(record, spec.target)
        for field in fields:
            if not math.isclose(float(record["metrics"][field]), float(recomputed[field]), abs_tol=1e-10):
                raise ValueError(f"saved metric differs for {record['method']}: {field}")
        if record["method"] in CLAIM_FUSE_METHODS.values():
            if len(record.get("trace", ())) != ACTIVE_BUDGET or record.get("query_count") != ACTIVE_BUDGET:
                raise ValueError("Claim-Fuse trace does not contain exactly 20 queries")

    rows: list[dict[str, Any]] = []
    for size in SAMPLE_SIZES:
        for method in METHODS:
            group = [item for item in records if item["method"] == method and int(item["sample_size"]) == size]
            row: dict[str, Any] = {"sample_size": size, "method": method, "replicate_count": len(group)}
            for field in fields:
                values = [float(item["metrics"][field]) for item in group]
                row[f"{field}_mean"] = statistics.fmean(values)
                row[f"{field}_se"] = statistics.stdev(values) / math.sqrt(len(values))
            rows.append(row)

    calibration_rows: list[dict[str, Any]] = []
    for size in SAMPLE_SIZES:
        for strategy, method in CLAIM_FUSE_METHODS.items():
            group = [item for item in records if item["method"] == method and int(item["sample_size"]) == size]
            events = [
                (int(int(step["answer"]) == _truth(step["query"], spec.target)), float(step["reported_confidence"]))
                for item in group for step in item["trace"] if step["valid_response"]
            ]
            calibration_rows.append({
                "sample_size": size,
                "llm": MODEL_DISPLAY,
                "strategy": strategy,
                "event_count": len(events),
                "coverage": len(events) / (len(group) * ACTIVE_BUDGET),
                **confidence_metrics(events),
            })
    audit = audit_saved_requests(root, spec)
    write_csv(root / "summary.csv", rows)
    write_csv(root / "confidence_summary.csv", calibration_rows)
    write_json(
        root / "summary.json",
        {
            "study": STUDY,
            "record_count": len(records),
            "rows": rows,
            "confidence_rows": calibration_rows,
            "prompt_audit": audit,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    return root / "summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "shared", "bfs", "scp", "claim-fuse", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "prepare":
        prepare(args.run_id, args.results_root)
    elif args.stage == "smoke":
        asyncio.run(run_smoke(args.run_id, args.base_url, args.runtime_manifest, args.results_root))
    elif args.stage == "shared":
        smoke = _root(args.run_id, args.results_root) / "smoke" / "summary.json"
        if not smoke.is_file() or read_json(smoke).get("status") != "passed":
            raise RuntimeError("the non-reportable smoke test must pass before full inference")
        asyncio.run(run_shared(args.run_id, args.base_url, args.runtime_manifest, args.results_root))
    elif args.stage == "bfs":
        run_bfs_stage(args.run_id, args.base_url, args.runtime_manifest, args.results_root)
    elif args.stage == "scp":
        if args.task_index is None:
            raise ValueError("--task-index is required for SCP")
        run_scp_task(args.run_id, args.task_index, args.base_url, args.runtime_manifest, args.results_root)
    elif args.stage == "claim-fuse":
        run_claim_fuse_stage(args.run_id, args.base_url, args.runtime_manifest, args.results_root)
    else:
        aggregate(args.run_id, args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
