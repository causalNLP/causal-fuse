"""GES-warm-started active Claim-Fuse-SMC with Qwen confidence elicitation."""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import statistics
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np

from cd_llm.baselines.causal_order2025 import (
    LocalVLLMConfig,
    QWEN35_MODEL,
    QWEN35_REVISION,
)
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.core.finite_support import (
    Graph,
    PartialGraph,
    distribution_metrics,
    graph_edges,
    graph_from_edges,
    has_adjacency,
    has_edge,
    sample_completions_with_diagnostics,
    stable_seed,
)
from cd_llm.experiments.claim_fuse_smc_eig import _serialize, _trace
from cd_llm.experiments.common.artifacts import (
    read_json,
    run_id,
    sha256 as file_sha256,
    write_csv,
    write_json,
)
from cd_llm.experiments.neuropathic_causal_order_subnetwork import (
    BASELINE_SOURCE_FINGERPRINT_V1,
    DEFAULT_MODEL_PATH,
    SAMPLE_SIZES,
    SUBNETWORK_NODE_COUNT,
    _baseline_path,
    _load_data,
    _partial,
    build_subnetwork_spec,
)
from cd_llm.methods.active_query import active_query_smc, group_i_query_pool
from cd_llm.methods.evidence_guided import ClaimFuseSMCConfig, initialize_score_particles
from cd_llm.methods.graph_scores import BDeuScore, parent_masks
from cd_llm.methods.llm_claim_confidence import (
    CachedLocalChatClient,
    ConfidenceStrategy,
    DOCUMENT_URLS,
    QwenClaimConfidenceProvider,
    confidence_metrics,
    load_document_corpus,
)
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "neuropathic_active_smc_qwen"
STUDY_VERSION = 1
BASELINE_RUN_ID = "20260825T082523Z"
DATA_SEEDS = tuple(range(20))
STRATEGIES: tuple[ConfidenceStrategy, ...] = (
    "stated",
    "consistency",
    "grounded",
)
METHOD_LABELS = {
    "stated": "Claim-Fuse-SMC-Active (stated confidence)",
    "consistency": "Claim-Fuse-SMC-Active (semantic consistency)",
    "grounded": "Claim-Fuse-SMC-Active (document grounded)",
}
ACTIVE_BUDGET = 20
SELECTION_CONFIDENCE = 0.9
EVIDENCE_TREATMENT = "soft_evidence"
COMPLETION_LIMIT = 200
STRUCTURE_PRIOR = 1.0
EQUIVALENT_SAMPLE_SIZE = 1.0
SMC_CONFIG = ClaimFuseSMCConfig(
    particle_count=384,
    initialization_updates=192,
    warm_start_score_mass=0.0,
)
QUERY_POOL_SIZE = len(group_i_query_pool(SUBNETWORK_NODE_COUNT))
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"


class _ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.buffer: list[str] = []
        self.passages: list[str] = []
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("p", "h1", "h2", "h3"):
            self.depth += 1
            self.buffer = []
        if tag == "title":
            self.in_title = True
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.depth or self.in_title:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.in_title:
            self.title = re.sub(r"\s+", " ", " ".join(self.buffer)).strip()
            self.in_title = False
            self.buffer = []
        if tag in ("p", "h1", "h2", "h3") and self.depth:
            text = re.sub(r"\s+", " ", " ".join(self.buffer)).strip()
            if 80 <= len(text) <= 2_000:
                self.passages.append(text)
            self.depth -= 1
            self.buffer = []


def prepare_document_corpus(
    output: Path,
    *,
    fetcher: Any | None = None,
) -> Path:
    if fetcher is None:
        def fetcher(url: str) -> bytes:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "cd-llm-research/1.0 (yahang.qi@mail.utoronto.ca)"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
    sources: list[dict[str, Any]] = []
    for url in DOCUMENT_URLS:
        parser = _ParagraphParser()
        parser.feed(fetcher(url).decode("utf-8", errors="replace"))
        if not parser.passages:
            raise ValueError(f"no review passages were extracted from {url}")
        pmcid = re.search(r"PMC\d+", url)
        if pmcid is None:
            raise ValueError("whitelisted review URL lacks a PMCID")
        sources.append(
            {
                "source_id": pmcid.group(0),
                "title": parser.title or pmcid.group(0),
                "url": url,
                "passages": parser.passages,
            }
        )
    write_json(
        output,
        {
            "study": STUDY,
            "source_policy": "independent open clinical reviews; target-derived material excluded",
            "sources": sources,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    return output


def slice_for_task(task_index: int) -> tuple[int, int]:
    if not 0 <= task_index < len(DATA_SEEDS) * len(SAMPLE_SIZES):
        raise ValueError("initialization task index must lie in [0, 39]")
    seed, size_index = divmod(task_index, len(SAMPLE_SIZES))
    return DATA_SEEDS[seed], SAMPLE_SIZES[size_index]


def _root(results_root: Path, output_run: str) -> Path:
    return results_root / output_run / STUDY


def _initialization_path(
    results_root: Path, output_run: str, seed: int, sample_size: int
) -> Path:
    return _root(results_root, output_run) / "initializations" / f"seed_{seed:02d}_n{sample_size}.json"


def _trajectory_path(
    results_root: Path,
    output_run: str,
    strategy: str,
    seed: int,
    sample_size: int,
) -> Path:
    return _root(results_root, output_run) / strategy / "partials" / f"seed_{seed:02d}_n{sample_size}.json"


def _graph(record: Mapping[str, Any]) -> Graph:
    return graph_from_edges(
        SUBNETWORK_NODE_COUNT,
        directed=record.get("directed", ()),
        bidirected=record.get("bidirected", ()),
        undirected=record.get("undirected", ()),
    )


def _graph_fingerprint(graphs: Sequence[Graph]) -> str:
    return __import__("hashlib").sha256(
        json.dumps([graph_edges(graph) for graph in graphs], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _score(data: np.ndarray) -> BDeuScore:
    return BDeuScore(
        data,
        equivalent_sample_size=EQUIVALENT_SAMPLE_SIZE,
        cardinalities=(2,) * SUBNETWORK_NODE_COUNT,
        max_parents=SMC_CONFIG.max_parents,
        structure_prior=STRUCTURE_PRIOR,
    )


def compute_initialization(
    seed: int,
    sample_size: int,
    output_run: str,
    results_root: Path,
    model_path: Path,
) -> Path:
    output_run = run_id(output_run)
    spec = build_subnetwork_spec(model_path)
    baseline_path = _baseline_path(results_root, BASELINE_RUN_ID, seed, sample_size)
    baseline = read_json(baseline_path)
    if baseline.get("source_fingerprint") != BASELINE_SOURCE_FINGERPRINT_V1:
        raise ValueError("Table 4.4 baseline source fingerprint differs")
    if baseline.get("run_id") != BASELINE_RUN_ID or baseline.get("seed") != seed or baseline.get("sample_size") != sample_size:
        raise ValueError("Table 4.4 GES baseline identity differs")
    ges = baseline["methods"]["GES"]
    fit = ges["fit"]
    if fit.get("score") != "local_score_BDeu" or fit.get("equivalent_sample_size") != 1 or fit.get("structure_prior") != 1:
        raise ValueError("Table 4.4 GES settings differ")
    cpdag: PartialGraph = _partial(ges["partial_graph"])
    completion_started = time.perf_counter()
    completions, completion_diagnostics = sample_completions_with_diagnostics(
        cpdag,
        "dag",
        COMPLETION_LIMIT,
        stable_seed(STUDY, "GES-completions", seed, sample_size),
    )
    completion_seconds = time.perf_counter() - completion_started
    if not completions or len(completions) != len(set(completions)):
        raise RuntimeError("GES completion support is empty or duplicated")
    warm_starts = tuple(
        graph
        for graph in completions
        if max(mask.bit_count() for mask in parent_masks(graph)) <= SMC_CONFIG.max_parents
    )
    if not warm_starts:
        raise RuntimeError("no GES completion satisfies the SMC parent cap")
    data, data_path, data_hash, reference_path, reference_hash, _ = _load_data(
        results_root, spec, seed, sample_size
    )
    score = _score(data)
    initialization_started = time.perf_counter()
    initialization = initialize_score_particles(
        score,
        warm_starts,
        stable_seed(STUDY, "initialization", seed, sample_size),
        particle_count=SMC_CONFIG.particle_count,
        updates_per_particle=SMC_CONFIG.initialization_updates,
        warm_start_score_mass=SMC_CONFIG.warm_start_score_mass,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    output = _initialization_path(results_root, output_run, seed, sample_size)
    write_json(
        output,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "initialization",
            "seed": seed,
            "sample_size": sample_size,
            "node_count": SUBNETWORK_NODE_COUNT,
            "support_initialization": "GES completions only",
            "baseline": {
                "run_id": BASELINE_RUN_ID,
                "path": str(baseline_path.resolve()),
                "sha256": file_sha256(baseline_path),
                "source_fingerprint": baseline["source_fingerprint"],
                "ges_partial_graph": ges["partial_graph"],
                "ges_fit": fit,
            },
            "data": {
                "path": str(data_path.resolve()),
                "sha256": data_hash,
                "reference_path": str(reference_path.resolve()),
                "reference_sha256": reference_hash,
            },
            "completion": {
                "requested_count": COMPLETION_LIMIT,
                "available_count": len(completions),
                "compatible_count": len(warm_starts),
                "fingerprint": _graph_fingerprint(warm_starts),
                "seconds": completion_seconds,
                "diagnostics": asdict(completion_diagnostics),
            },
            "score": score.metadata(),
            "smc_config": asdict(SMC_CONFIG),
            "particles": [graph_edges(graph) for graph in initialization.particles],
            "weights": list(initialization.weights),
            "particle_fingerprint": _graph_fingerprint(initialization.particles),
            "initialization": {
                "seconds": initialization_seconds,
                "acceptance_rate": initialization.acceptance_rate,
                "accepted_moves": initialization.accepted_moves,
                "attempted_moves": initialization.attempted_moves,
                "support_size": len(initialization.distribution),
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"initialized GES-only particles seed={seed} n={sample_size}: {output}", flush=True)
    return output


def _load_initialization(
    results_root: Path,
    output_run: str,
    seed: int,
    sample_size: int,
    model_path: Path,
) -> tuple[Path, dict[str, Any], np.ndarray, Graph, tuple[Graph, ...], tuple[float, ...]]:
    path = _initialization_path(results_root, output_run, seed, sample_size)
    payload = read_json(path)
    expected = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "run_id": output_run,
        "stage": "initialization",
        "seed": seed,
        "sample_size": sample_size,
        "node_count": SUBNETWORK_NODE_COUNT,
        "support_initialization": "GES completions only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"initialization {key} differs")
    if payload["baseline"].get("run_id") != BASELINE_RUN_ID or payload["baseline"].get("source_fingerprint") != BASELINE_SOURCE_FINGERPRINT_V1:
        raise ValueError("initialization does not descend from the Table 4.4 GES baseline")
    spec = build_subnetwork_spec(model_path)
    data, data_path, data_hash, _, _, _ = _load_data(results_root, spec, seed, sample_size)
    if payload["data"]["sha256"] != data_hash or payload["data"]["path"] != str(data_path.resolve()):
        raise ValueError("initialization data differs")
    particles = tuple(_graph(record) for record in payload["particles"])
    weights = tuple(float(value) for value in payload["weights"])
    if len(particles) != SMC_CONFIG.particle_count or len(weights) != len(particles):
        raise ValueError("initialization particle-slot count differs")
    if payload["particle_fingerprint"] != _graph_fingerprint(particles):
        raise ValueError("initialization particle fingerprint differs")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("initialization weights do not sum to one")
    return path, payload, data, spec.target, particles, weights


def _truth(query: Mapping[str, Any], target: Graph) -> int:
    source = int(query["source"])
    destination = int(query["target"])
    if query["kind"] == "directed":
        return int(has_edge(target, source, destination))
    return int(has_adjacency(target, source, destination))


def run_strategy(
    strategy: ConfidenceStrategy,
    output_run: str,
    results_root: Path,
    model_path: Path,
    corpus_path: Path,
    base_url: str,
) -> Path:
    output_run = run_id(output_run)
    sources = load_document_corpus(corpus_path) if strategy == "grounded" else ()
    cache_path = _root(results_root, output_run) / strategy / "request_cache.sqlite3"
    cache = ResponseCache(cache_path)
    try:
        config = LocalVLLMConfig(
            base_url=base_url,
            seed=0,
            concurrency=1,
            temperature=0.0,
            triplet_max_tokens=256,
            pair_max_tokens=256,
        )
        chat = CachedLocalChatClient(cache, config)
        node_names = build_subnetwork_spec(model_path).node_names
        provider = QwenClaimConfidenceProvider(strategy, node_names, chat, sources=sources)
        for seed in DATA_SEEDS:
            for sample_size in SAMPLE_SIZES:
                output = _trajectory_path(results_root, output_run, strategy, seed, sample_size)
                if output.is_file():
                    continue
                started = time.perf_counter()
                initialization_path, initialization, data, target, particles, weights = _load_initialization(
                    results_root, output_run, seed, sample_size, model_path
                )
                score = _score(data)
                active = active_query_smc(
                    None,
                    score,
                    provider,
                    budgets=(ACTIVE_BUDGET,),
                    confidence=SELECTION_CONFIDENCE,
                    treatment=EVIDENCE_TREATMENT,
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
                probabilities = np.asarray([distribution[graph] for graph in graphs], dtype=float)
                metrics = distribution_metrics(graphs, probabilities, target, "dag")
                trace = _trace(active)
                valid_events = [
                    (
                        int(int(step["answer"]) == _truth(step["query"], target)),
                        float(step["confidence"]),
                    )
                    for step in trace
                    if step["valid_response"]
                ]
                calibration = confidence_metrics(valid_events)
                prompt_tokens = sum(int(step["prompt_tokens"]) for step in trace)
                completion_tokens = sum(int(step["completion_tokens"]) for step in trace)
                write_json(
                    output,
                    {
                        "study": STUDY,
                        "study_version": STUDY_VERSION,
                        "run_id": output_run,
                        "stage": "trajectory",
                        "strategy": strategy,
                        "method": METHOD_LABELS[strategy],
                        "seed": seed,
                        "sample_size": sample_size,
                        "node_count": SUBNETWORK_NODE_COUNT,
                        "maximum_budget": ACTIVE_BUDGET,
                        "selection_confidence": SELECTION_CONFIDENCE,
                        "evidence_treatment": EVIDENCE_TREATMENT,
                        "support_initialization": "GES completions only",
                        "initialization_path": str(initialization_path.resolve()),
                        "initialization_sha256": file_sha256(initialization_path),
                        "particle_fingerprint": initialization["particle_fingerprint"],
                        "baseline_ges_sha256": initialization["baseline"]["sha256"],
                        "model": QWEN35_MODEL,
                        "model_revision": QWEN35_REVISION,
                        "thinking": False,
                        "temperature": 0.0,
                        "corpus_path": str(corpus_path.resolve()) if strategy == "grounded" else None,
                        "corpus_sha256": file_sha256(corpus_path) if strategy == "grounded" else None,
                        "metrics": metrics,
                        "calibration": calibration,
                        "coverage": active.answered_query_count / ACTIVE_BUDGET,
                        "logical_input_tokens": prompt_tokens,
                        "logical_output_tokens": completion_tokens,
                        "distribution": _serialize(distribution),
                        "trace": trace,
                        "resampling_count": active.resampling_count,
                        "support_recovery_count": active.support_recovery_count,
                        "timing": asdict(active.timing),
                        "runtime_seconds": time.perf_counter() - started,
                        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "command": list(sys.argv),
                    },
                    exclusive=True,
                )
                print(f"completed {strategy} seed={seed} n={sample_size}: {output}", flush=True)
    finally:
        cache.close()
    return _root(results_root, output_run) / strategy


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def _equal_or_both_nan(first: float, second: float) -> bool:
    return (math.isnan(first) and math.isnan(second)) or math.isclose(
        first, second, rel_tol=0.0, abs_tol=1e-12
    )


def aggregate(output_run: str, results_root: Path, model_path: Path) -> Path:
    output_run = run_id(output_run)
    target = build_subnetwork_spec(model_path).target
    raw_rows: list[dict[str, Any]] = []
    unique_by_strategy: dict[str, dict[str, tuple[int, float]]] = {strategy: {} for strategy in STRATEGIES}
    references: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for seed in DATA_SEEDS:
            for sample_size in SAMPLE_SIZES:
                path = _trajectory_path(results_root, output_run, strategy, seed, sample_size)
                payload = read_json(path)
                if payload.get("strategy") != strategy or payload.get("particle_fingerprint") != read_json(_initialization_path(results_root, output_run, seed, sample_size))["particle_fingerprint"]:
                    raise ValueError("trajectory strategy or GES initialization differs")
                trace = payload["trace"]
                if len(trace) != ACTIVE_BUDGET or len({step["query"]["query_id"] for step in trace}) != ACTIVE_BUDGET:
                    raise ValueError("trajectory does not contain 20 unique queries")
                valid = []
                for step in trace:
                    if not step["valid_response"]:
                        continue
                    correctness = int(int(step["answer"]) == _truth(step["query"], target))
                    confidence = float(step["confidence"])
                    valid.append((correctness, confidence))
                    key = str(step["query"]["query_id"])
                    previous = unique_by_strategy[strategy].get(key)
                    value = (correctness, confidence)
                    if previous is not None and previous != value:
                        raise ValueError("cached expert response differs for one predicate")
                    unique_by_strategy[strategy][key] = value
                calibration = confidence_metrics(valid)
                for key in ("accuracy", "brier", "ece"):
                    if not _equal_or_both_nan(
                        float(payload["calibration"][key]),
                        float(calibration[key]),
                    ):
                        raise ValueError(f"trajectory calibration {key} differs")
                raw_rows.append(
                    {
                        "strategy": strategy,
                        "method": METHOD_LABELS[strategy],
                        "seed": seed,
                        "sample_size": sample_size,
                        **{key: float(payload["metrics"][key]) for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")},
                        "valid_count": int(calibration["valid_count"]),
                        "coverage": float(payload["coverage"]),
                        "accuracy": float(calibration["accuracy"]),
                        "brier": float(calibration["brier"]),
                        "ece": float(calibration["ece"]),
                        "logical_input_tokens": int(payload["logical_input_tokens"]),
                        "logical_output_tokens": int(payload["logical_output_tokens"]),
                        "runtime_seconds": float(payload["runtime_seconds"]),
                    }
                )
                references.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    summary: list[dict[str, Any]] = []
    calibration_summary: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        for strategy in STRATEGIES:
            rows = [row for row in raw_rows if row["sample_size"] == sample_size and row["strategy"] == strategy]
            if len(rows) != len(DATA_SEEDS):
                raise ValueError("aggregate lacks a complete 20-seed group")
            record: dict[str, Any] = {"sample_size": sample_size, "strategy": strategy, "method": METHOD_LABELS[strategy], "replicate_count": len(rows)}
            for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
                mean, se = _mean_se([float(row[key]) for row in rows])
                record[f"{key}_mean"] = mean
                record[f"{key}_se"] = se
            record["logical_input_tokens_mean"] = statistics.fmean(float(row["logical_input_tokens"]) for row in rows)
            record["logical_output_tokens_mean"] = statistics.fmean(float(row["logical_output_tokens"]) for row in rows)
            summary.append(record)
            events = [
                (
                    int(int(step["answer"]) == _truth(step["query"], target)),
                    float(step["confidence"]),
                )
                for seed in DATA_SEEDS
                for step in read_json(_trajectory_path(results_root, output_run, strategy, seed, sample_size))["trace"]
                if step["valid_response"]
            ]
            event_metrics = confidence_metrics(events)
            calibration_summary.append(
                {
                    "sample_size": sample_size,
                    "strategy": strategy,
                    "query_event_count": ACTIVE_BUDGET * len(DATA_SEEDS),
                    "coverage": int(event_metrics["valid_count"]) / (ACTIVE_BUDGET * len(DATA_SEEDS)),
                    **event_metrics,
                }
            )
    unique_summary = [
        {"strategy": strategy, **confidence_metrics(tuple(values.values()))}
        for strategy, values in unique_by_strategy.items()
    ]
    root = _root(results_root, output_run)
    write_csv(root / "raw_results.csv", raw_rows)
    write_csv(root / "summary.csv", summary)
    write_csv(root / "calibration_summary.csv", calibration_summary)
    write_csv(root / "unique_predicate_calibration.csv", unique_summary)
    write_json(
        root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "strategies": STRATEGIES,
            "support_initialization": "GES completions only",
            "baseline_run_id": BASELINE_RUN_ID,
            "data_seeds": DATA_SEEDS,
            "sample_sizes": SAMPLE_SIZES,
            "active_budget": ACTIVE_BUDGET,
            "query_pool_size": QUERY_POOL_SIZE,
            "selection_confidence": SELECTION_CONFIDENCE,
            "evidence_treatment": EVIDENCE_TREATMENT,
            "smc_config": asdict(SMC_CONFIG),
            "model": QWEN35_MODEL,
            "model_revision": QWEN35_REVISION,
            "thinking": False,
            "calibration": {"primary": "query-event-weighted", "secondary": "unique-predicate", "bins": 10, "target_labels_used_post_hoc_only": True},
            "artifact_references": references,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"aggregated Qwen active-SMC benchmark: {root}", flush=True)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare-corpus", "initialize", "strategy", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    default_corpus = WORKSPACE_ROOT / "generated" / "runtime" / output_run / "neuropathic_clinical_reviews.json"
    corpus_path = default_corpus if arguments.corpus_path is None else arguments.corpus_path
    if arguments.stage == "prepare-corpus":
        prepare_document_corpus(corpus_path)
    elif arguments.stage == "initialize":
        if arguments.task_index is None:
            raise ValueError("--task-index is required")
        seed, sample_size = slice_for_task(arguments.task_index)
        compute_initialization(seed, sample_size, output_run, arguments.results_root, arguments.model_path)
    elif arguments.stage == "strategy":
        if arguments.strategy is None:
            raise ValueError("--strategy is required")
        run_strategy(arguments.strategy, output_run, arguments.results_root, arguments.model_path, corpus_path, arguments.base_url)
    else:
        aggregate(output_run, arguments.results_root, arguments.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
