"""Opaque-identifier GPT-5.6 Terra study on the locked neuropathic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cd_llm.baselines.causal_order2025.openrouter_backend import (
    INPUT_USD_PER_TOKEN,
    OPENROUTER_BASE_URL,
    OUTPUT_USD_PER_TOKEN,
    OpenRouterConfig,
    ResponseCache,
)
from cd_llm.baselines.guess2graph2026 import (
    build_skeleton_prompt,
    edges_to_skeleton,
    parse_skeleton_response,
    run_gpc_guess_skeleton,
)
from cd_llm.baselines.jiralerspong2024 import Completion, run_bfs
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
from cd_llm.experiments.claim_fuse_smc_eig import _trace
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json
from cd_llm.experiments.neuropathic_active_smc_qwen import SMC_CONFIG, _truth
from cd_llm.experiments.neuropathic_bfs2024 import DESCRIPTIONS as SOURCE_DESCRIPTIONS
from cd_llm.methods.active_query import active_query_smc
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.methods.llm_claim_confidence import (
    CachedOpenRouterChatClient,
    ConfidenceStrategy,
    LLMClaimConfidenceProvider,
    confidence_metrics,
    load_document_corpus,
)
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "neuropathic_decontamination"
STUDY_VERSION = 3
BENCHMARK_ID = "table45-neuropathic-ancestor20-v1"
MODEL = "openai/gpt-5.6-terra"
MODEL_DISPLAY_NAME = "GPT-5.6 Terra"
SOURCE_KIT = WORKSPACE_ROOT / "results" / "20260826T120230Z" / "table45_benchmark_kit"
SOURCE_STATISTICS_ROOT = WORKSPACE_ROOT / "results" / "20260902T201726Z" / STUDY
SOURCE_COMPLETED_ROOT = WORKSPACE_ROOT / "results" / "20260902T234859Z" / STUDY
SOURCE_CORPUS_PATH = (
    WORKSPACE_ROOT
    / "generated"
    / "runtime"
    / "20260826T063414Z"
    / "neuropathic_clinical_reviews.json"
)
RESULTS_ROOT = WORKSPACE_ROOT / "results"
DATA_SEEDS = tuple(range(10))
SAMPLE_SIZES = (200, 1000)
OPAQUE_NAMES = tuple(f"V{index:02d}" for index in range(1, 21))
PERMUTATION_SEED = 2027
DOMAIN_CONTEXT = "a clinical diagnostic system"
GPC_METHOD = f"gPC-Guess + {MODEL_DISPLAY_NAME}"
BFS_METHOD = f"BFS+Pearson + {MODEL_DISPLAY_NAME}"
CLAIM_FUSE_METHOD = f"Claim-Fuse-SMC-Active + {MODEL_DISPLAY_NAME}"
CONFIDENCE_STRATEGIES: tuple[ConfidenceStrategy, ...] = (
    "stated",
    "consistency",
    "grounded",
)
CLAIM_FUSE_METHODS = {
    "stated": CLAIM_FUSE_METHOD,
    "consistency": f"{CLAIM_FUSE_METHOD} (semantic consistency)",
    "grounded": f"{CLAIM_FUSE_METHOD} (document grounded)",
}
METHODS = (GPC_METHOD, BFS_METHOD, *(CLAIM_FUSE_METHODS[value] for value in CONFIDENCE_STRATEGIES))
SOURCE_METHODS = (GPC_METHOD, BFS_METHOD, CLAIM_FUSE_METHOD)
PROTECTED_CONTEXT_TERMS = ("neuropathic pain", "simulator", "table45", "benchmark")
ACTIVE_BUDGET = 20
SELECTION_CONFIDENCE = 0.9
EVIDENCE_TREATMENT = "soft_evidence"
SOFT_CONFIDENCE_CAP = 0.99
DEFAULT_COST_CAP_USD = 5.0
TOTAL_INCREMENTAL_COST_CAP_USD = 10.0
DEFAULT_MAX_ATTEMPTS = 8
GPC_MAX_COMPLETION_TOKENS = 4096
BFS_MAX_COMPLETION_TOKENS = 4096


@dataclass(frozen=True)
class OpaqueSpec:
    names: tuple[str, ...]
    descriptions: Mapping[str, str]
    permutation: tuple[int, ...]
    original_names: tuple[str, ...]
    target: Graph


def _root(output_run: str, results_root: Path = RESULTS_ROOT) -> Path:
    return results_root / run_id(output_run) / STUDY


def _permutation() -> tuple[int, ...]:
    return tuple(int(value) for value in np.random.default_rng(PERMUTATION_SEED).permutation(20))


def _source_manifest(source_kit: Path = SOURCE_KIT) -> dict[str, Any]:
    payload = read_json(source_kit / "manifest.json")
    if payload.get("benchmark_id") != BENCHMARK_ID or payload.get("node_count") != 20:
        raise ValueError("locked neuropathic source kit identity differs")
    return payload


def _source_target(source_kit: Path = SOURCE_KIT) -> Graph:
    payload = read_json(source_kit / "truth_graph.json")["graph"]
    directed = []
    for edge in payload["edges"]:
        pair = (int(edge["node_u"]), int(edge["node_v"]))
        directed.append(pair if (edge["mark_u"], edge["mark_v"]) == ("tail", "arrow") else pair[::-1])
    return graph_from_edges(20, directed=directed)


def build_spec(source_kit: Path = SOURCE_KIT) -> OpaqueSpec:
    manifest = _source_manifest(source_kit)
    originals = tuple(str(node["name"]).replace("\ufffd", "ä") for node in manifest["nodes"])
    if set(originals) != set(SOURCE_DESCRIPTIONS):
        raise ValueError("description coverage differs from the locked node set")
    permutation = _permutation()
    descriptions = {
        opaque: SOURCE_DESCRIPTIONS[originals[source_index]]
        for opaque, source_index in zip(OPAQUE_NAMES, permutation, strict=True)
    }
    inverse = {source: opaque for opaque, source in enumerate(permutation)}
    directed = [
        (inverse[source], inverse[target])
        for source, target in graph_edges(_source_target(source_kit))["directed"]
    ]
    return OpaqueSpec(OPAQUE_NAMES, descriptions, permutation, originals, graph_from_edges(20, directed=directed))


def audit_model_text(text: str, spec: OpaqueSpec) -> None:
    lowered = text.casefold()
    leaked = [name for name in spec.original_names if name.casefold() in lowered]
    leaked.extend(term for term in PROTECTED_CONTEXT_TERMS if term in lowered)
    if leaked:
        raise ValueError(f"model-facing text leaks protected benchmark identity: {sorted(set(leaked))}")


def _write_or_validate(path: Path, payload: Mapping[str, Any], identity: Sequence[str]) -> None:
    if path.exists():
        existing = read_json(path)
        if any(existing.get(key) != payload.get(key) for key in identity):
            raise ValueError(f"existing artifact identity differs: {path}")
        return
    write_json(path, dict(payload), exclusive=True)


def _import_completed_records(root: Path, source_root: Path) -> dict[str, Any]:
    """Import the locked baseline and stated-confidence records without rerunning them."""

    source_summary = source_root / "summary.json"
    source_audit = source_root / "audit.json"
    if not source_summary.is_file() or not source_audit.is_file():
        raise FileNotFoundError("completed decontamination source aggregate is unavailable")
    summary = read_json(source_summary)
    audit = read_json(source_audit)
    if summary.get("study") != STUDY or int(summary.get("record_count", -1)) != 60:
        raise ValueError("completed source aggregate identity differs")
    if audit.get("status") != "passed":
        raise ValueError("completed source prompt audit did not pass")

    expected = {
        (method, seed, sample_size)
        for method in SOURCE_METHODS
        for seed in DATA_SEEDS
        for sample_size in SAMPLE_SIZES
    }
    copied: set[tuple[str, int, int]] = set()
    for source_path in sorted((source_root / "records").glob("*/*.json")):
        record = read_json(source_path)
        identity = (
            str(record.get("method")),
            int(record.get("seed", -1)),
            int(record.get("sample_size", -1)),
        )
        if identity not in expected:
            raise ValueError(f"unexpected completed source record: {identity}")
        destination = root / "records" / source_path.parent.name / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(destination) != sha256(source_path):
                raise ValueError(f"imported source record differs: {destination}")
        else:
            shutil.copy2(source_path, destination)
        copied.add(identity)
    if copied != expected:
        raise ValueError("completed source record set is incomplete")
    return {
        "source_root": str(source_root.resolve()),
        "source_summary_path": str(source_summary.resolve()),
        "source_summary_sha256": sha256(source_summary),
        "source_audit_path": str(source_audit.resolve()),
        "source_audit_sha256": sha256(source_audit),
        "record_count": len(copied),
        "methods": list(SOURCE_METHODS),
    }


def prepare(
    output_run: str,
    results_root: Path = RESULTS_ROOT,
    source_kit: Path = SOURCE_KIT,
    source_statistics_root: Path = SOURCE_STATISTICS_ROOT,
    source_completed_root: Path = SOURCE_COMPLETED_ROOT,
    corpus_path: Path = SOURCE_CORPUS_PATH,
) -> Path:
    root = _root(output_run, results_root)
    benchmark = root / "benchmark"
    benchmark.mkdir(parents=True, exist_ok=True)
    spec = build_spec(source_kit)
    model_metadata = {
        "domain_context": DOMAIN_CONTEXT,
        "nodes": [{"id": name, "description": spec.descriptions[name]} for name in spec.names],
    }
    audit_model_text(json.dumps(model_metadata, ensure_ascii=False), spec)
    _write_or_validate(benchmark / "model_metadata.json", model_metadata, ("domain_context", "nodes"))
    sources = load_document_corpus(corpus_path)
    audit_model_text(corpus_path.read_text(), spec)

    manifest = _source_manifest(source_kit)
    datasets = []
    statistics = []
    for seed in DATA_SEEDS:
        for sample_size in SAMPLE_SIZES:
            source = next(
                item for item in manifest["datasets"]
                if item["seed"] == seed and item["sample_size"] == sample_size
            )
            source_path = source_kit / source["path"]
            if sha256(source_path) != source["sha256"]:
                raise ValueError("source dataset checksum differs")
            frame = pd.read_csv(source_path)
            transformed = frame.iloc[:, list(spec.permutation)].copy()
            transformed.columns = spec.names
            destination = benchmark / "data" / f"seed_{seed:02d}_n{sample_size}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                existing = pd.read_csv(destination)
                if not transformed.equals(existing):
                    raise ValueError(f"existing opaque dataset differs: {destination}")
            else:
                transformed.to_csv(destination, index=False)
            datasets.append({
                "seed": seed,
                "sample_size": sample_size,
                "path": str(destination.relative_to(root)),
                "sha256": sha256(destination),
                "source_sha256": source["sha256"],
            })

            source_data = source_statistics_root / "benchmark" / "data" / destination.name
            source_stats = source_statistics_root / "statistics" / f"seed_{seed:02d}_n{sample_size}.json"
            if not source_data.is_file() or sha256(source_data) != sha256(destination):
                raise ValueError(f"source-statistics dataset differs for seed={seed}, n={sample_size}")
            if not source_stats.is_file():
                raise FileNotFoundError(f"source statistics are unavailable: {source_stats}")
            stats_payload = read_json(source_stats)
            if stats_payload.get("seed") != seed or stats_payload.get("sample_size") != sample_size:
                raise ValueError(f"source statistics identity differs: {source_stats}")
            if len(stats_payload.get("particles", ())) != SMC_CONFIG.particle_count:
                raise ValueError(f"source particle count differs: {source_stats}")
            statistics.append({
                "seed": seed,
                "sample_size": sample_size,
                "path": str(source_stats.resolve()),
                "sha256": sha256(source_stats),
                "dataset_sha256": sha256(source_data),
            })

    for seed in DATA_SEEDS:
        small = pd.read_csv(benchmark / "data" / f"seed_{seed:02d}_n200.csv")
        large = pd.read_csv(benchmark / "data" / f"seed_{seed:02d}_n1000.csv")
        if not small.equals(large.iloc[:200].reset_index(drop=True)):
            raise ValueError(f"nested sample invariant failed for seed {seed}")

    truth = {"usage": "evaluation-only", "graph": graph_edges(spec.target)}
    _write_or_validate(benchmark / "truth_graph.json", truth, ("usage", "graph"))
    evaluation = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "source_benchmark_id": BENCHMARK_ID,
        "permutation_seed": PERMUTATION_SEED,
        "opaque_to_source_local_index": dict(zip(spec.names, spec.permutation, strict=True)),
        "source_names_evaluation_only": list(spec.original_names),
        "data_seeds": list(DATA_SEEDS),
        "sample_sizes": list(SAMPLE_SIZES),
        "datasets": datasets,
        "source_statistics": statistics,
        "model": MODEL,
        "methods": list(METHODS),
        "confidence_strategies": list(CONFIDENCE_STRATEGIES),
        "document_corpus": {
            "path": str(corpus_path.resolve()),
            "sha256": sha256(corpus_path),
            "source_count": len(sources),
        },
        "observational_samples_only": True,
        "intervention_labels_present": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_or_validate(
        benchmark / "evaluation_manifest.json",
        evaluation,
        ("study", "study_version", "permutation_seed", "datasets", "source_statistics", "model", "methods"),
    )
    provenance = _import_completed_records(root, source_completed_root)
    _write_or_validate(
        benchmark / "completed_source.json",
        provenance,
        ("source_summary_sha256", "source_audit_sha256", "record_count", "methods"),
    )
    return benchmark


def load_condition(
    output_run: str,
    seed: int,
    sample_size: int,
    results_root: Path = RESULTS_ROOT,
) -> tuple[pd.DataFrame, OpaqueSpec]:
    if seed not in DATA_SEEDS or sample_size not in SAMPLE_SIZES:
        raise ValueError("unsupported decontamination condition")
    root = _root(output_run, results_root)
    frame = pd.read_csv(root / "benchmark" / "data" / f"seed_{seed:02d}_n{sample_size}.csv")
    spec = build_spec()
    if tuple(frame.columns) != spec.names or frame.shape != (sample_size, 20):
        raise ValueError("opaque dataset layout differs")
    return frame, spec


def _load_statistics(root: Path, seed: int, sample_size: int) -> tuple[dict[str, Any], Path, str]:
    manifest = read_json(root / "benchmark" / "evaluation_manifest.json")
    record = next(
        item for item in manifest["source_statistics"]
        if item["seed"] == seed and item["sample_size"] == sample_size
    )
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"source statistics checksum differs: {path}")
    return read_json(path), path, str(record["sha256"])


def _graph(record: Mapping[str, Any]) -> Graph:
    return graph_from_edges(
        20,
        directed=record.get("directed", ()),
        bidirected=record.get("bidirected", ()),
        undirected=record.get("undirected", ()),
    )


def _result_path(root: Path, method: str, seed: int, sample_size: int) -> Path:
    label = method.lower().replace(" ", "_").replace("+", "plus")
    return root / "records" / label / f"seed_{seed:02d}_n{sample_size}.json"


def _record_result(
    root: Path,
    method: str,
    seed: int,
    sample_size: int,
    metrics: Mapping[str, Any],
    **extra: Any,
) -> None:
    payload = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "method": method,
        "seed": seed,
        "sample_size": sample_size,
        "model": MODEL,
        "metrics": dict(metrics),
        "discovery_input": {
            "observational_samples_only": True,
            "intervention_labels_present": False,
        },
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json(_result_path(root, method, seed, sample_size), payload, exclusive=True)


class CachedTerraCompleter:
    """Restart-safe synchronous OpenRouter calls for BFS and gPC-Guess."""

    def __init__(
        self,
        cache: ResponseCache,
        *,
        model: str = MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        cost_cap_usd: float = DEFAULT_COST_CAP_USD,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        client: Any | None = None,
    ) -> None:
        if cost_cap_usd <= 0 or max_attempts < 1:
            raise ValueError("cost cap and attempt count must be positive")
        self.cache = cache
        self.model = model
        self.cost_cap_usd = float(cost_cap_usd)
        self.max_attempts = int(max_attempts)
        if client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for uncached requests")
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0)
        self.client = client

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        purpose: str,
        seed: int,
        max_completion_tokens: int,
        temperature: float | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "purpose": purpose,
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "seed": int(seed),
            "max_completion_tokens": int(max_completion_tokens),
            "reasoning_effort": "none",
        }
        if temperature is not None:
            request["temperature"] = float(temperature)
        if response_format is not None:
            request["response_format"] = dict(response_format)
        cached = self.cache.get(request)
        if cached is not None:
            return cached

        characters = sum(len(str(message.get("content", ""))) for message in messages)
        reservation = math.ceil(characters / 4) * INPUT_USD_PER_TOKEN
        reservation += max_completion_tokens * OUTPUT_USD_PER_TOKEN
        if self.cache.summary()["cost_usd"] + reservation > self.cost_cap_usd + 1e-12:
            raise RuntimeError(f"OpenRouter safety cap of ${self.cost_cap_usd:.2f} would be exceeded")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "seed": int(seed),
            "max_completion_tokens": int(max_completion_tokens),
            "reasoning_effort": "none",
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if response_format is not None:
            kwargs["response_format"] = dict(response_format)

        response_object = None
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                response_object = self.client.chat.completions.create(**kwargs)
                break
            except Exception as error:
                status = getattr(error, "status_code", None)
                retryable = status in (408, 409, 429) or (
                    isinstance(status, int) and status >= 500
                ) or status is None
                if not retryable or attempt == self.max_attempts:
                    label = f"HTTP {status}" if isinstance(status, int) else type(error).__name__
                    raise RuntimeError(
                        f"OpenRouter request failed after {attempt} attempt(s): {label}"
                    ) from None
                time.sleep(min(60.0, 2.0 ** (attempt - 1)) + random.random())
        if response_object is None:  # pragma: no cover
            raise RuntimeError("OpenRouter request produced no response")

        response = response_object.model_dump()
        content_value = response_object.choices[0].message.content
        content = "" if content_value is None else str(content_value)
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        cost = usage.get("cost")
        if cost is None:
            cost = prompt_tokens * INPUT_USD_PER_TOKEN + completion_tokens * OUTPUT_USD_PER_TOKEN
        self.cache.put(
            request,
            purpose,
            response,
            content,
            provider_request_id=getattr(response_object, "id", None),
            response_model=getattr(response_object, "model", None),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=float(cost),
            latency_seconds=time.perf_counter() - started,
            attempts=attempt,
        )
        result = self.cache.get(request)
        if result is None:  # pragma: no cover
            raise RuntimeError("committed OpenRouter response is unavailable")
        result["cached"] = False
        return result


def _guess_skeleton(
    router: CachedTerraCompleter,
    spec: OpaqueSpec,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    shuffled = list(spec.names)
    random.Random(seed).shuffle(shuffled)
    prompt = build_skeleton_prompt(
        shuffled,
        descriptions=spec.descriptions,
        domain_context=DOMAIN_CONTEXT,
    )
    audit_model_text(prompt, spec)
    response = router.complete(
        [
            {"role": "system", "content": f"You are a careful expert in causal inference and {DOMAIN_CONTEXT}."},
            {"role": "user", "content": prompt},
        ],
        purpose=f"gpc_guess:seed_{seed:02d}",
        seed=seed,
        max_completion_tokens=GPC_MAX_COMPLETION_TOKENS,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response["content"])
    except json.JSONDecodeError as error:
        raise RuntimeError("GPT-5.6 Terra returned invalid JSON for gPC-Guess") from error
    edges = parse_skeleton_response(parsed, spec.names)
    return edges_to_skeleton(spec.names, edges), {
        "request_key": response["request_key"],
        "cache_hit": bool(response["cached"]),
        "input_tokens": int(response.get("prompt_tokens") or 0),
        "output_tokens": int(response.get("completion_tokens") or 0),
    }


def _run_gpc(
    root: Path,
    frame: pd.DataFrame,
    spec: OpaqueSpec,
    seed: int,
    sample_size: int,
    router: CachedTerraCompleter,
) -> None:
    output = _result_path(root, GPC_METHOD, seed, sample_size)
    if output.exists():
        return
    guess, request = _guess_skeleton(router, spec, seed)
    predicted = run_gpc_guess_skeleton(
        frame.to_numpy(dtype=np.int64),
        guess,
        alpha=0.05,
        ci_test="chisq",
        rng=np.random.default_rng(seed),
    )
    partial_marks = np.where(predicted != 0, int(Endpoint.TAIL), int(Endpoint.NULL))
    partial = PartialGraph.from_matrix(partial_marks, "cpdag")
    _record_result(
        root,
        GPC_METHOD,
        seed,
        sample_size,
        point_prediction_metrics(partial, spec.target),
        graph={"kind": "undirected", "edges": np.argwhere(np.triu(predicted, 1)).tolist()},
        request=request,
        hyperparameters={"alpha": 0.05, "ci_test": "chisq", "temperature": 0.7},
    )


def _run_bfs(
    root: Path,
    frame: pd.DataFrame,
    spec: OpaqueSpec,
    seed: int,
    sample_size: int,
    router: CachedTerraCompleter,
) -> None:
    output = _result_path(root, BFS_METHOD, seed, sample_size)
    if output.exists():
        return
    request_keys: list[str] = []
    cache_hits: list[bool] = []

    def complete(messages: Sequence[dict[str, str]]) -> Completion:
        audit_model_text("\n".join(str(item.get("content", "")) for item in messages), spec)
        response = router.complete(
            messages,
            purpose=f"bfs_pearson:seed_{seed:02d}:n{sample_size}",
            seed=seed,
            max_completion_tokens=BFS_MAX_COMPLETION_TOKENS,
            temperature=0.7,
        )
        request_keys.append(str(response["request_key"]))
        cache_hits.append(bool(response["cached"]))
        return Completion(
            text=str(response["content"]),
            input_tokens=int(response.get("prompt_tokens") or 0),
            output_tokens=int(response.get("completion_tokens") or 0),
        )

    edges, usage = run_bfs(
        frame,
        complete,
        include_statistics=True,
        descriptions=spec.descriptions,
        domain_context=DOMAIN_CONTEXT,
        opaque_identifiers=True,
    )
    graph = graph_from_edges(20, directed=edges)
    _record_result(
        root,
        BFS_METHOD,
        seed,
        sample_size,
        point_prediction_metrics(graph, spec.target),
        graph=graph_edges(graph),
        cost=usage,
        request_keys=request_keys,
        cache_hits=cache_hits,
        hyperparameters={"temperature": 0.7, "include_pearson_correlations": True},
    )


def _request_accounting(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[tuple[str, bool, float, str | None, str | None]] = []
    for step in trace:
        audit = step.get("response_audit") or {}
        keys = tuple(str(value) for value in audit.get("request_keys", ()))
        hits = tuple(bool(value) for value in audit.get("cache_hits", ()))
        costs = tuple(float(value) for value in audit.get("cost_usd", ()))
        request_ids = tuple(audit.get("provider_request_ids", ()))
        models = tuple(audit.get("response_models", ()))
        if {len(keys), len(hits), len(costs), len(request_ids), len(models)} != {len(keys)}:
            raise ValueError("response request-audit vectors differ in length")
        records.extend(zip(keys, hits, costs, request_ids, models))
    unique: dict[str, tuple[bool, float, str | None, str | None]] = {}
    for key, hit, cost, request_id, response_model in records:
        value = (hit, cost, request_id, response_model)
        if key in unique and unique[key] != value:
            raise ValueError("one request key has inconsistent request accounting")
        unique[key] = value
    return {
        "logical_request_count": len(records),
        "unique_request_count": len(unique),
        "cache_hit_count": sum(int(hit) for _, hit, _, _, _ in records),
        "paid_cost_usd": math.fsum(cost for hit, cost, _, _ in unique.values() if not hit),
        "request_keys": sorted(unique),
        "provider_request_ids": sorted(
            {str(value) for _, _, value, _ in unique.values() if value}
        ),
        "response_models": sorted(
            {str(value) for _, _, _, value in unique.values() if value}
        ),
    }


def _run_claim_fuse(
    root: Path,
    frame: pd.DataFrame,
    spec: OpaqueSpec,
    seed: int,
    sample_size: int,
    chat: CachedOpenRouterChatClient,
    strategy: ConfidenceStrategy = "stated",
    sources: Sequence[Any] = (),
    corpus_path: Path = SOURCE_CORPUS_PATH,
) -> None:
    if strategy not in CONFIDENCE_STRATEGIES:
        raise ValueError("unknown confidence strategy")
    method = CLAIM_FUSE_METHODS[strategy]
    output = _result_path(root, method, seed, sample_size)
    if output.exists():
        return
    started = time.perf_counter()
    stats, stats_path, stats_hash = _load_statistics(root, seed, sample_size)
    particles = tuple(_graph(item) for item in stats["particles"])
    weights = tuple(float(value) for value in stats["weights"])
    if len(particles) != SMC_CONFIG.particle_count or not math.isclose(
        math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("source particle initialization is invalid")
    data = frame.to_numpy(dtype=np.int64)
    score = BDeuScore(
        data,
        equivalent_sample_size=1.0,
        cardinalities=(2,) * 20,
        max_parents=SMC_CONFIG.max_parents,
        structure_prior=1.0,
    )
    provider = LLMClaimConfidenceProvider(
        strategy,
        spec.names,
        chat,
        sources=sources,
        descriptions=spec.descriptions,
        domain_context=DOMAIN_CONTEXT,
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
        seed=stable_seed(STUDY, "active", seed, sample_size),
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
    valid_events = [
        (
            int(int(step["answer"]) == _truth(step["query"], spec.target)),
            float(step["reported_confidence"]),
        )
        for step in trace
        if step["valid_response"]
    ]
    request_accounting = _request_accounting(trace)
    _record_result(
        root,
        method,
        seed,
        sample_size,
        distribution_metrics(graphs, probabilities, spec.target, "dag"),
        distribution=[
            {"graph": graph_edges(graph), "weight": float(distribution[graph])}
            for graph in graphs
        ],
        strategy=strategy,
        trace=trace,
        query_count=ACTIVE_BUDGET,
        coverage=active.answered_query_count / ACTIVE_BUDGET,
        calibration=confidence_metrics(valid_events),
        request_accounting=request_accounting,
        logical_input_tokens=sum(int(step["prompt_tokens"]) for step in trace),
        logical_output_tokens=sum(int(step["completion_tokens"]) for step in trace),
        runtime_seconds=time.perf_counter() - started,
        document_corpus={
            "path": str(corpus_path.resolve()),
            "sha256": sha256(corpus_path),
        }
        if strategy == "grounded"
        else None,
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


def run_confidence_strategy(
    output_run: str,
    strategy: ConfidenceStrategy,
    conditions: Sequence[tuple[int, int]],
    *,
    results_root: Path = RESULTS_ROOT,
    base_url: str = OPENROUTER_BASE_URL,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    client: Any | None = None,
    corpus_path: Path = SOURCE_CORPUS_PATH,
) -> None:
    if strategy not in ("consistency", "grounded"):
        raise ValueError("only missing confidence strategies may be run")
    root = _root(output_run, results_root)
    if not (root / "benchmark" / "evaluation_manifest.json").is_file():
        raise FileNotFoundError("prepare stage must complete before Terra inference")
    cache = ResponseCache(root / "caches" / f"openrouter_request_cache_{strategy}.sqlite3")
    audit_spec = build_spec()
    sources = load_document_corpus(corpus_path) if strategy == "grounded" else ()
    chat = CachedOpenRouterChatClient(
        cache,
        OpenRouterConfig(
            model=MODEL,
            base_url=base_url,
            seed=0,
            concurrency=1,
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            cost_cap_usd=cost_cap_usd,
            reasoning_effort="none",
        ),
        prompt_auditor=lambda prompt: audit_model_text(prompt, audit_spec),
    )
    try:
        for seed, sample_size in conditions:
            frame, spec = load_condition(output_run, seed, sample_size, results_root)
            _run_claim_fuse(
                root,
                frame,
                spec,
                seed,
                sample_size,
                chat,
                strategy,
                sources,
                corpus_path,
            )
    finally:
        chat.close()
        cache.close()


def audit_saved_model_requests(root: Path, spec: OpaqueSpec) -> dict[str, Any]:
    rows: list[tuple[str, str, str | None, int, int, float, int, str]] = []
    for strategy in ("consistency", "grounded"):
        cache_path = root / "caches" / f"openrouter_request_cache_{strategy}.sqlite3"
        if not cache_path.is_file():
            continue
        connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        try:
            cache_rows = connection.execute(
                "SELECT purpose, request_json, response_model, prompt_tokens, completion_tokens, "
                "cost_usd, attempts FROM responses ORDER BY purpose, request_key"
            ).fetchall()
        finally:
            connection.close()
        rows.extend((*row, strategy) for row in cache_rows)
    prompt_hashes: set[str] = set()
    purposes: dict[str, dict[str, float]] = {
        strategy: {
            "paid_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "attempts": 0,
            "logical_requests": 0,
            "cache_hits": 0,
        }
        for strategy in ("consistency", "grounded")
    }
    response_models: set[str] = set()
    for purpose, request_json, response_model, input_tokens, output_tokens, cost, attempts, strategy in rows:
        request = json.loads(request_json)
        text = "\n".join(str(item.get("content", "")) for item in request.get("messages", ()))
        audit_model_text(text, spec)
        prompt_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        if response_model:
            response_models.add(str(response_model))
        if not str(purpose).startswith(f"{strategy}:"):
            raise ValueError(f"request purpose differs from cache strategy: {purpose}")
        entry = purposes[strategy]
        entry["paid_requests"] += 1
        entry["input_tokens"] += int(input_tokens or 0)
        entry["output_tokens"] += int(output_tokens or 0)
        entry["cost_usd"] += float(cost)
        entry["attempts"] += int(attempts)
    if response_models and response_models != {MODEL}:
        raise ValueError(f"unexpected OpenRouter response models: {sorted(response_models)}")
    for path in sorted((root / "records").glob("*/*.json")):
        record = read_json(path)
        strategy = record.get("strategy")
        if strategy in ("consistency", "grounded"):
            for step in record.get("trace", ()):
                hits = tuple(
                    bool(value)
                    for value in step.get("response_audit", {}).get("cache_hits", ())
                )
                purposes[strategy]["logical_requests"] += len(hits)
                purposes[strategy]["cache_hits"] += sum(hits)
    source = read_json(root / "benchmark" / "completed_source.json")
    return {
        "status": "passed",
        "protected_source_name_count": len(spec.original_names),
        "protected_context_terms": list(PROTECTED_CONTEXT_TERMS),
        "unique_saved_prompt_count": len(prompt_hashes),
        "response_models": sorted(response_models),
        "by_strategy": purposes,
        "total_paid_requests": len(rows),
        "total_cost_usd": math.fsum(float(row[5]) for row in rows),
        "total_attempts": sum(int(row[6]) for row in rows),
        "completed_source": source,
    }


def aggregate(output_run: str, results_root: Path = RESULTS_ROOT) -> Path:
    root = _root(output_run, results_root)
    records = [read_json(path) for path in root.glob("records/*/*.json")]
    expected = {(method, seed, size) for method in METHODS for seed in DATA_SEEDS for size in SAMPLE_SIZES}
    actual = {(record["method"], int(record["seed"]), int(record["sample_size"])) for record in records}
    if actual != expected or len(records) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"decontamination aggregate incomplete; missing={missing[:10]}, extra={extra[:10]}, records={len(records)}"
        )
    spec = build_spec()
    strategy_for_method = {method: strategy for strategy, method in CLAIM_FUSE_METHODS.items()}
    unique_events: dict[str, dict[str, tuple[int, float]]] = {
        strategy: {} for strategy in CONFIDENCE_STRATEGIES
    }
    pooled_events: dict[tuple[int, str], list[tuple[int, float]]] = {
        (sample_size, strategy): []
        for sample_size in SAMPLE_SIZES
        for strategy in CONFIDENCE_STRATEGIES
    }
    for record in records:
        method = str(record["method"])
        if method not in strategy_for_method:
            continue
        strategy = strategy_for_method[method]
        if strategy != "stated":
            if record.get("study_version") != STUDY_VERSION or record.get("strategy") != strategy:
                raise ValueError("new confidence-strategy record identity differs")
        trace = record.get("trace", ())
        if len(trace) != ACTIVE_BUDGET or len({step["query"]["query_id"] for step in trace}) != ACTIVE_BUDGET:
            raise ValueError("Claim-Fuse trajectory lacks 20 unique query events")
        distribution = record.get("distribution", ())
        graphs = tuple(_graph(item["graph"]) for item in distribution)
        probabilities = np.asarray([float(item["weight"]) for item in distribution], dtype=float)
        recomputed = distribution_metrics(graphs, probabilities, spec.target, "dag")
        for field in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
            if not math.isclose(
                float(record["metrics"][field]),
                float(recomputed[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Claim-Fuse trajectory metric differs: {field}")
        events: list[tuple[int, float]] = []
        for step in trace:
            if not step["valid_response"]:
                continue
            event = (
                int(int(step["answer"]) == _truth(step["query"], spec.target)),
                float(step["reported_confidence"]),
            )
            events.append(event)
            query_id = str(step["query"]["query_id"])
            previous = unique_events[strategy].get(query_id)
            if previous is not None and previous != event:
                raise ValueError("one cached predicate has inconsistent confidence evidence")
            unique_events[strategy][query_id] = event
        if "calibration" in record:
            recomputed_calibration = confidence_metrics(events)
            for field in ("valid_count", "accuracy", "brier", "ece"):
                left = float(record["calibration"][field])
                right = float(recomputed_calibration[field])
                if not ((math.isnan(left) and math.isnan(right)) or math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError(f"trajectory calibration differs: {field}")
        pooled_events[(int(record["sample_size"]), strategy)].extend(events)
    rows = []
    fields = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
    for sample_size in SAMPLE_SIZES:
        for method in METHODS:
            group = [
                record for record in records
                if record["method"] == method and record["sample_size"] == sample_size
            ]
            row: dict[str, Any] = {
                "sample_size": sample_size,
                "method": method,
                "display_method": (
                    f"{CLAIM_FUSE_METHOD} (stated confidence)"
                    if method == CLAIM_FUSE_METHOD
                    else method
                ),
                "strategy": strategy_for_method.get(method),
                "replicate_count": len(group),
            }
            for field in fields:
                values = [float(record["metrics"][field]) for record in group]
                row[f"{field}_mean"] = statistics.fmean(values)
                row[f"{field}_se"] = statistics.stdev(values) / math.sqrt(len(values))
            rows.append(row)
    calibration_rows = []
    for sample_size in SAMPLE_SIZES:
        for strategy in CONFIDENCE_STRATEGIES:
            metrics = confidence_metrics(pooled_events[(sample_size, strategy)])
            calibration_rows.append(
                {
                    "sample_size": sample_size,
                    "strategy": strategy,
                    "query_event_count": ACTIVE_BUDGET * len(DATA_SEEDS),
                    "coverage": int(metrics["valid_count"]) / (ACTIVE_BUDGET * len(DATA_SEEDS)),
                    **metrics,
                }
            )
    audit = audit_saved_model_requests(root, build_spec())
    if audit["total_cost_usd"] > TOTAL_INCREMENTAL_COST_CAP_USD + 1e-12:
        raise RuntimeError("recorded OpenRouter cost exceeds the locked safety cap")
    write_csv(root / "summary.csv", rows)
    write_csv(root / "calibration_summary.csv", calibration_rows)
    write_csv(
        root / "unique_predicate_calibration.csv",
        [
            {"strategy": strategy, **confidence_metrics(tuple(events.values()))}
            for strategy, events in unique_events.items()
        ],
    )
    write_json(root / "audit.json", audit, exclusive=True)
    write_json(
        root / "summary.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "model": MODEL,
            "rows": rows,
            "calibration_rows": calibration_rows,
            "record_count": len(records),
            "prompt_audit": audit,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    return root / "summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "pilot", "run", "aggregate", "audit"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--source-kit", type=Path, default=SOURCE_KIT)
    parser.add_argument("--source-statistics-root", type=Path, default=SOURCE_STATISTICS_ROOT)
    parser.add_argument("--source-completed-root", type=Path, default=SOURCE_COMPLETED_ROOT)
    parser.add_argument("--corpus-path", type=Path, default=SOURCE_CORPUS_PATH)
    parser.add_argument("--strategy", choices=("consistency", "grounded"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "prepare":
        print(
            prepare(
                args.run_id,
                args.results_root,
                args.source_kit,
                args.source_statistics_root,
                args.source_completed_root,
                args.corpus_path,
            )
        )
    elif args.stage == "pilot":
        if args.strategy is None:
            raise ValueError("--strategy is required for pilot and run stages")
        run_confidence_strategy(
            args.run_id,
            args.strategy,
            ((0, 200),),
            results_root=args.results_root,
            base_url=args.base_url,
            cost_cap_usd=args.cost_cap_usd,
            corpus_path=args.corpus_path,
        )
        print(json.dumps(audit_saved_model_requests(_root(args.run_id, args.results_root), build_spec()), indent=2))
    elif args.stage == "run":
        if args.strategy is None:
            raise ValueError("--strategy is required for pilot and run stages")
        run_confidence_strategy(
            args.run_id,
            args.strategy,
            tuple((seed, sample_size) for seed in DATA_SEEDS for sample_size in SAMPLE_SIZES),
            results_root=args.results_root,
            base_url=args.base_url,
            cost_cap_usd=args.cost_cap_usd,
            corpus_path=args.corpus_path,
        )
    elif args.stage == "audit":
        print(json.dumps(audit_saved_model_requests(_root(args.run_id, args.results_root), build_spec()), indent=2))
    else:
        print(aggregate(args.run_id, args.results_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
