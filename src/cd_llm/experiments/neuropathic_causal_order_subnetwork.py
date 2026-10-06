"""Benchmark PC, GES, and Causal Order on a 20-node neuropathic subnetwork."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import resource
import sqlite3
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from cd_llm.baselines.causal_order2025 import (
    CausalOrderInfeasibleError,
    LocalVLLMConfig,
    LocalVLLMGateError,
    QWEN35_MODEL,
    QWEN35_REVISION,
    QWEN_PAIR_MAX_TOKENS,
    QWEN_ROLE_ADAPTER,
    QWEN_TRIPLET_MAX_TOKENS,
    QWEN_TRIPLET_RETRY_MAX_TOKENS,
    elicit_triplet_order_local_vllm,
    integrate_order_into_cpdag,
    query_pair_fallbacks_local_vllm,
)
from cd_llm.baselines.causal_order2025.types import TripletExpertResult
from cd_llm.baselines.causal_order2025.upstream_backend import UPSTREAM_REVISION
from cd_llm.core.finite_support import (
    Graph,
    PartialGraph,
    _causal_learn_partial,
    graph_edges,
    graph_from_edges,
    point_prediction_metrics,
)
from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
from cd_llm.experiments.common.artifacts import (
    read_json,
    run_id,
    sha256,
    write_csv,
    write_json,
)
from cd_llm.experiments.neuropathic_benchmark import (
    DATA_SEEDS,
    SAMPLE_SIZES,
    build_panel_spec,
)
from cd_llm.experiments.neuropathic_panel_point_benchmark import run_pc
from cd_llm.experiments.neuropathic_runtime import run_ges
from cd_llm.paths import CODE_ROOT, WORKSPACE_ROOT


STUDY = "neuropathic_causal_order_subnetwork"
STUDY_VERSION = 1
# The baseline implementation fingerprint is pinned for this run because the
# only post-launch change is retrying transient filesystem reads; PC, GES, the
# data, and every numerical setting are unchanged.
BASELINE_SOURCE_FINGERPRINT_V1 = "59d00637966f04deb85f0ea0ac06767800562094e6c0215913ed6a89f4527340"
METHODS = ("PC", "GES", "Causal Order+PC (Qwen3.5-9B)")
SUBNETWORK_TARGET = "L Nackbesvär"
SUBNETWORK_NODE_COUNT = 20
SUBNETWORK_EDGE_COUNT = 30
SUBNETWORK_LAYER_COUNTS = (7, 12, 1)
SUBNETWORK_PANEL_INDICES = tuple(range(SUBNETWORK_NODE_COUNT))
SUBNETWORK_ORIGINAL_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    80,
)
TRIPLET_COUNT = math.comb(SUBNETWORK_NODE_COUNT, 3)
CONTEXT = "for neuropathic pain diagnosis"
MINIMUM_PARSE_RATE = 0.95
MAX_BENCHMARK_RUNTIME_SECONDS = 3600.0
MAX_GPU_MEMORY_MIB = 38 * 1024
VLLM_MAX_NUM_SEQS = 32
REFERENCE_RUN_ID = "20260821T081942Z"
REFERENCE_STUDY = "neuropathic_benchmark/panel_a"
CONTAINER_PYTHON = "python3"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"
DEFAULT_MODEL_PATH = CODE_ROOT / "resources" / "neuropathic" / "bnm.pickle"


@dataclass(frozen=True)
class SubnetworkSpec:
    panel_indices: tuple[int, ...]
    original_indices: tuple[int, ...]
    node_names: tuple[str, ...]
    target: Graph
    layer_counts: tuple[int, int, int]

    @property
    def node_count(self) -> int:
        return len(self.node_names)

    @property
    def edge_count(self) -> int:
        return len(graph_edges(self.target)["directed"])


def slice_for_task(task_index: int) -> tuple[int, int]:
    if not 0 <= task_index < len(DATA_SEEDS) * len(SAMPLE_SIZES):
        raise ValueError("task index must lie in [0, 39]")
    seed, size_index = divmod(task_index, len(SAMPLE_SIZES))
    return DATA_SEEDS[seed], SAMPLE_SIZES[size_index]


def _induced_graph(target: Graph, indices: Sequence[int]) -> Graph:
    selected = tuple(int(index) for index in indices)
    if len(selected) != len(set(selected)) or any(
        not 0 <= index < target.node_count for index in selected
    ):
        raise ValueError("induced-graph indices must be unique valid nodes")
    mapping = {old: new for new, old in enumerate(selected)}
    directed = [
        (mapping[source], mapping[target_node])
        for source, target_node in graph_edges(target)["directed"]
        if source in mapping and target_node in mapping
    ]
    return graph_from_edges(len(selected), directed=directed)


def build_subnetwork_spec(model_path: Path = DEFAULT_MODEL_PATH) -> SubnetworkSpec:
    panel = build_panel_spec(model_path)
    try:
        target_index = panel.node_names.index(SUBNETWORK_TARGET)
    except ValueError as error:
        raise ValueError("the subnetwork target is absent from Panel A") from error
    panel_graph = nx.DiGraph()
    panel_graph.add_nodes_from(range(panel.node_count))
    panel_graph.add_edges_from(graph_edges(panel.target)["directed"])
    closure = nx.ancestors(panel_graph, target_index) | {target_index}
    panel_indices = tuple(index for index in range(panel.node_count) if index in closure)
    original_indices = tuple(panel.original_indices[index] for index in panel_indices)
    node_names = tuple(panel.node_names[index] for index in panel_indices)
    target = _induced_graph(panel.target, panel_indices)
    layer_counts = (
        sum(index < 27 for index in original_indices),
        sum(27 <= index < 79 for index in original_indices),
        sum(index >= 79 for index in original_indices),
    )
    actual = (
        panel_indices,
        original_indices,
        len(node_names),
        len(graph_edges(target)["directed"]),
        layer_counts,
    )
    expected = (
        SUBNETWORK_PANEL_INDICES,
        SUBNETWORK_ORIGINAL_INDICES,
        SUBNETWORK_NODE_COUNT,
        SUBNETWORK_EDGE_COUNT,
        SUBNETWORK_LAYER_COUNTS,
    )
    if actual != expected:
        raise ValueError(f"neuropathic causal-order subnetwork differs: {actual!r}")
    if any(
        parent not in closure
        for node in closure
        for parent in panel_graph.predecessors(node)
    ):
        raise ValueError("neuropathic causal-order subnetwork is not ancestor closed")
    return SubnetworkSpec(
        panel_indices=panel_indices,
        original_indices=original_indices,
        node_names=node_names,
        target=target,
        layer_counts=layer_counts,
    )


def _root(results_root: Path, output_run: str) -> Path:
    return results_root / output_run / STUDY


def _baseline_path(
    results_root: Path, output_run: str, seed: int, sample_size: int
) -> Path:
    return (
        _root(results_root, output_run)
        / "baseline_partials"
        / f"seed_{seed:02d}_n{sample_size}.json"
    )


def _causal_order_path(
    results_root: Path, output_run: str, seed: int, sample_size: int
) -> Path:
    return (
        _root(results_root, output_run)
        / "causal_order_partials"
        / f"seed_{seed:02d}_n{sample_size}.json"
    )


def _reference_path(results_root: Path, seed: int) -> Path:
    return (
        results_root
        / REFERENCE_RUN_ID
        / REFERENCE_STUDY
        / "partials"
        / f"seed_{seed:02d}.json"
    )


def _runtime_manifest_path(output_run: str) -> Path:
    return (
        WORKSPACE_ROOT
        / "generated"
        / "runtime"
        / output_run
        / "qwen35_neuropathic20_manifest.json"
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_fingerprint() -> str:
    paths = (
        Path(__file__),
        CODE_ROOT / "src" / "cd_llm" / "baselines" / "causal_order2025" / "local_vllm_backend.py",
        CODE_ROOT / "src" / "cd_llm" / "baselines" / "causal_order2025" / "upstream_backend.py",
        CODE_ROOT / "src" / "cd_llm" / "baselines" / "causal_order2025" / "integration.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().relative_to(CODE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _protocol(runtime_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = {} if runtime_manifest is None else dict(runtime_manifest)
    return {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "subnetwork": {
            "construction": "target symptom plus all ancestors",
            "target": SUBNETWORK_TARGET,
            "panel_indices": list(SUBNETWORK_PANEL_INDICES),
            "original_indices": list(SUBNETWORK_ORIGINAL_INDICES),
            "node_count": SUBNETWORK_NODE_COUNT,
            "edge_count": SUBNETWORK_EDGE_COUNT,
            "layer_counts": list(SUBNETWORK_LAYER_COUNTS),
        },
        "sample_sizes": list(SAMPLE_SIZES),
        "data_seeds": list(DATA_SEEDS),
        "pc": {
            "implementation": "causal-learn 0.1.4.5",
            "alpha": 0.01,
            "conditional_independence_test": "gsq",
            "stable": True,
            "uc_rule": 0,
            "uc_priority": 2,
            "mvpc": False,
        },
        "ges": {
            "score": "local_score_BDeu",
            "equivalent_sample_size": 1,
            "structure_prior": 1,
            "max_parents": None,
        },
        "causal_order": {
            "algorithm": "Appendix D Algorithm 1",
            "strategy": "all unordered node triplets",
            "triplet_count": TRIPLET_COUNT,
            "shared_across_data_seeds_and_sample_sizes": True,
            "context": CONTEXT,
            "model": QWEN35_MODEL,
            "model_revision": QWEN35_REVISION,
            "thinking": False,
            "temperature": 0.0,
            "seed": 0,
            "triplet_max_tokens": QWEN_TRIPLET_MAX_TOKENS,
            "triplet_parse_retry_max_tokens": QWEN_TRIPLET_RETRY_MAX_TOKENS,
            "tie_and_fallback_max_tokens": QWEN_PAIR_MAX_TOKENS,
            "minimum_parse_rate": MINIMUM_PARSE_RATE,
            "prompt_role_adapter": QWEN_ROLE_ADAPTER,
            "upstream_revision": UPSTREAM_REVISION,
            "repair": False,
            "completion_retry": "same prompt only after an unparseable tagged response",
            "semantic_retry": False,
        },
        "server": {
            "provider": "local-vllm",
            "max_model_length": 8192,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "gpu_memory_utilization": 0.85,
            "prefix_caching": False,
            "usage_stats": False,
            "runtime_manifest_sha256": None
            if not manifest
            else _fingerprint(manifest),
        },
        "metric": {
            "distance": "normalized_endpoint_hamming",
            "energy_score": "point prediction; equals endpoint distance",
            "adjacency_f1": "raw skeleton",
            "orientation_f1": "explicit tail-arrow edges",
            "truth_representation": "induced target DAG",
        },
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("causal-learn", "networkx", "numpy", "pandas", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _partial_record(graph: PartialGraph) -> dict[str, Any]:
    return {"kind": graph.kind, "node_count": graph.node_count, "marks": list(graph.marks)}


def _partial(record: Mapping[str, Any]) -> PartialGraph:
    return PartialGraph(
        int(record["node_count"]),
        tuple(int(mark) for mark in record["marks"]),
        str(record["kind"]),  # type: ignore[arg-type]
    )


def _graph(record: Mapping[str, Any]) -> Graph:
    return graph_from_edges(
        SUBNETWORK_NODE_COUNT,
        directed=record.get("directed", ()),
        bidirected=record.get("bidirected", ()),
        undirected=record.get("undirected", ()),
    )


def _retry_io(
    operation: Callable[[], Any],
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 1.0,
) -> tuple[Any, int]:
    """Retry transient shared-filesystem failures without masking bad data."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return operation(), attempt
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(initial_delay_seconds * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def _load_data(
    results_root: Path,
    spec: SubnetworkSpec,
    seed: int,
    sample_size: int,
) -> tuple[np.ndarray, Path, str, Path, str, int]:
    reference_path = _reference_path(results_root, seed)
    reference, reference_read_attempts = _retry_io(lambda: read_json(reference_path))
    if reference.get("run_id") != REFERENCE_RUN_ID or reference.get("data_seed") != seed:
        raise ValueError("frozen Panel-A reference identity differs")
    if tuple(reference["panel"]["node_names"])[0:SUBNETWORK_NODE_COUNT] != spec.node_names:
        raise ValueError("frozen Panel-A semantic node order differs")
    sample = reference["sample_artifacts"][str(sample_size)]
    data_path = Path(sample["data_path"])
    data_hash = str(sample["data_sha256"])
    def read_verified_frame() -> pd.DataFrame:
        if sha256(data_path) != data_hash:
            raise ValueError("frozen Panel-A data hash differs")
        return pd.read_csv(data_path, index_col=0)

    frame, data_read_attempts = _retry_io(read_verified_frame)
    selected_columns = tuple(str(index) for index in spec.original_indices)
    if any(column not in frame for column in selected_columns):
        raise ValueError("subnetwork columns are absent from the frozen Panel-A data")
    data = frame.loc[:, list(selected_columns)].to_numpy(dtype=np.int64, copy=True)
    if data.shape != (sample_size, SUBNETWORK_NODE_COUNT):
        raise ValueError("subnetwork data shape differs")
    if np.any((data != 0) & (data != 1)):
        raise ValueError("subnetwork observations must be raw binary integers")
    reference_hash, reference_hash_attempts = _retry_io(lambda: sha256(reference_path))
    total_attempts = reference_read_attempts + data_read_attempts + reference_hash_attempts
    return data, data_path, data_hash, reference_path, reference_hash, total_attempts


def compute_baseline_slice(
    seed: int,
    sample_size: int,
    output_run: str,
    results_root: Path,
    model_path: Path,
    *,
    pc_runner: Callable[..., Any] | None = None,
    ges_runner: Callable[..., Any] | None = None,
) -> Path:
    started = time.perf_counter()
    output_run = run_id(output_run)
    if seed not in DATA_SEEDS or sample_size not in SAMPLE_SIZES:
        raise ValueError("invalid neuropathic benchmark slice")
    spec = build_subnetwork_spec(model_path)
    data, data_path, data_hash, reference_path, reference_hash, data_io_attempts = _load_data(
        results_root, spec, seed, sample_size
    )
    protocol = _protocol()

    fit_started = time.perf_counter()
    raw_pc, pc_settings = run_pc(data, spec.node_names, runner=pc_runner)
    pc_fit = {"runtime_seconds": time.perf_counter() - fit_started, **pc_settings}
    pc = _causal_learn_partial(raw_pc, "cpdag")

    fit_started = time.perf_counter()
    raw_ges, ges_settings = run_ges(
        data, spec.node_names, runner=ges_runner, score_backend="reference"
    )
    ges_fit = {"runtime_seconds": time.perf_counter() - fit_started, **ges_settings}
    ges = _causal_learn_partial(raw_ges, "cpdag")

    methods: dict[str, Any] = {}
    for method, graph, fit in (("PC", pc, pc_fit), ("GES", ges, ges_fit)):
        methods[method] = {
            "partial_graph": _partial_record(graph),
            "fit": fit,
            "metrics": point_prediction_metrics(graph, spec.target),
        }
    output = _baseline_path(results_root, output_run, seed, sample_size)
    write_json(
        output,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "baseline",
            "seed": seed,
            "sample_size": sample_size,
            "protocol": protocol,
            "protocol_fingerprint": _fingerprint(protocol),
            "source_fingerprint": BASELINE_SOURCE_FINGERPRINT_V1,
            "subnetwork": {
                "target": SUBNETWORK_TARGET,
                "panel_indices": list(spec.panel_indices),
                "original_indices": list(spec.original_indices),
                "node_names": list(spec.node_names),
                "layer_counts": list(spec.layer_counts),
            },
            "reference": {
                "path": str(reference_path.resolve()),
                "sha256": reference_hash,
                "data_path": str(data_path.resolve()),
                "data_sha256": data_hash,
                "io_attempts": data_io_attempts,
            },
            "target": graph_edges(spec.target),
            "methods": methods,
            "runtime_seconds": time.perf_counter() - started,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "package_versions": _package_versions(),
            "command": list(sys.argv),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"completed 20-node PC/GES seed={seed} n={sample_size}: {output}", flush=True)
    return output


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def _checkpoint_shards(checkpoint_path: Path) -> tuple[Path, ...]:
    index = json.loads((checkpoint_path / "model.safetensors.index.json").read_text())
    names = tuple(sorted(set(index["weight_map"].values())))
    if len(names) != 4 or any(Path(name).name != name for name in names):
        raise ValueError("the pinned Qwen checkpoint must contain four named shards")
    shards = tuple(checkpoint_path / name for name in names)
    if any(not shard.is_file() for shard in shards):
        raise FileNotFoundError("a pinned Qwen checkpoint shard is missing")
    return shards


def prepare_runtime_manifest(
    output_run: str, image_path: Path, checkpoint_path: Path
) -> Path:
    output_run = run_id(output_run)
    config_path = checkpoint_path / "config.json"
    index_path = checkpoint_path / "model.safetensors.index.json"
    if not image_path.is_file() or not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("the pinned Qwen image or checkpoint is incomplete")
    shards = _checkpoint_shards(checkpoint_path)
    output = _runtime_manifest_path(output_run)
    write_json(
        output,
        {
            "run_id": output_run,
            "study": STUDY,
            "model": QWEN35_MODEL,
            "model_revision": QWEN35_REVISION,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_files": [
                {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (config_path, index_path, *shards)
            ],
            "image_path": str(image_path.resolve()),
            "image_size_bytes": image_path.stat().st_size,
            "image_sha256": sha256(image_path),
            "vllm_version": _command_output(
                ("apptainer", "exec", str(image_path), CONTAINER_PYTHON, "-c", "import vllm; print(vllm.__version__)")
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"prepared 20-node Qwen runtime manifest: {output}", flush=True)
    return output


def _gpu_snapshot() -> dict[str, Any]:
    gpu = _command_output(
        ("nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits")
    ).splitlines()
    used = _command_output(
        ("nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits")
    ).splitlines()
    values = [int(value.strip()) for value in used if value.strip().isdigit()]
    return {
        "gpu": gpu,
        "compute_process_used_memory_mib": values,
        "maximum_compute_process_used_memory_mib": max(values, default=0),
    }


def _expert_record(expert: TripletExpertResult) -> dict[str, Any]:
    return {
        "node_names": list(expert.node_names),
        "raw_edges": [list(edge) for edge in expert.raw_edges],
        "graph": graph_edges(expert.graph),
        "edgewise_distributions": [
            {"pair": list(pair), "probabilities": list(probabilities)}
            for pair, probabilities in sorted(expert.edgewise_distributions.items())
        ],
        "subgroup_list": [list(group) for group in expert.subgroup_list],
        "subgroup_responses": list(expert.subgroup_responses),
        "expert_order": list(expert.expert_order),
        "isolated_nodes": list(expert.isolated_nodes),
        "cycle_removal_threshold": expert.cycle_removal_threshold,
        "upstream_revision": expert.upstream_revision,
        "runtime_seconds": expert.runtime_seconds,
    }


def _cache_purpose_summary(cache_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(cache_path)
    try:
        rows = connection.execute(
            "SELECT purpose, COUNT(*), COALESCE(SUM(prompt_tokens),0), "
            "COALESCE(SUM(completion_tokens),0), COALESCE(SUM(attempts),0), "
            "COALESCE(SUM(latency_seconds),0) FROM responses GROUP BY purpose ORDER BY purpose"
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "purpose": str(purpose),
            "request_count": int(count),
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "attempts": int(attempts),
            "serial_latency_seconds": float(latency),
        }
        for purpose, count, prompt, completion, attempts, latency in rows
    ]


def _validate_baseline(
    payload: Mapping[str, Any], output_run: str, seed: int, sample_size: int
) -> None:
    expected = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "run_id": output_run,
        "stage": "baseline",
        "seed": seed,
        "sample_size": sample_size,
        "source_fingerprint": BASELINE_SOURCE_FINGERPRINT_V1,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"baseline {key} differs for seed={seed}, n={sample_size}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"baseline protocol is absent for seed={seed}, n={sample_size}")
    if payload.get("protocol_fingerprint") != _fingerprint(protocol):
        raise ValueError(f"baseline protocol fingerprint differs for seed={seed}, n={sample_size}")
    current = _protocol()
    for section in ("subnetwork", "sample_sizes", "data_seeds", "pc", "ges", "metric"):
        if protocol.get(section) != current[section]:
            raise ValueError(f"baseline protocol {section} differs for seed={seed}, n={sample_size}")


async def run_qwen(
    output_run: str,
    results_root: Path,
    model_path: Path,
    runtime_manifest_path: Path,
    base_url: str,
) -> Path:
    started = time.perf_counter()
    job_started_raw = os.environ.get("CAUSAL_ORDER_JOB_STARTED_UNIX")
    job_started_unix = (
        float(job_started_raw) if job_started_raw is not None else time.time()
    )
    if not math.isfinite(job_started_unix) or job_started_unix > time.time():
        raise ValueError("CAUSAL_ORDER_JOB_STARTED_UNIX is invalid")
    output_run = run_id(output_run)
    root = _root(results_root, output_run)
    expert_path = root / "expert.json"
    if expert_path.exists() or (root / "gate_failure.json").exists():
        raise FileExistsError("the 20-node Qwen stage already has a terminal artifact")
    runtime_manifest = read_json(runtime_manifest_path)
    if runtime_manifest.get("run_id") != output_run:
        raise ValueError("runtime manifest identity differs")
    if runtime_manifest.get("model_revision") != QWEN35_REVISION:
        raise ValueError("runtime manifest model revision differs")
    protocol = _protocol(runtime_manifest)
    protocol_fingerprint = _fingerprint(protocol)
    source_fingerprint = _source_fingerprint()
    spec = build_subnetwork_spec(model_path)
    baselines: list[tuple[int, int, Path, dict[str, Any], PartialGraph]] = []
    for seed in DATA_SEEDS:
        for sample_size in SAMPLE_SIZES:
            path = _baseline_path(results_root, output_run, seed, sample_size)
            payload = read_json(path)
            _validate_baseline(payload, output_run, seed, sample_size)
            baselines.append(
                (seed, sample_size, path, payload, _partial(payload["methods"]["PC"]["partial_graph"]))
            )
    config = LocalVLLMConfig(
        base_url=base_url,
        triplet_retry_max_tokens=QWEN_TRIPLET_RETRY_MAX_TOKENS,
    )
    cache_path = root / "request_cache.sqlite3"
    gpu_before = _gpu_snapshot()
    try:
        elicitation = await elicit_triplet_order_local_vllm(
            spec.node_names,
            CONTEXT,
            cache_path,
            config=config,
            minimum_parse_rate=MINIMUM_PARSE_RATE,
        )
    except LocalVLLMGateError as error:
        write_json(
            root / "gate_failure.json",
            {
                "study": STUDY,
                "study_version": STUDY_VERSION,
                "run_id": output_run,
                "stage": "gate_failure",
                "protocol": protocol,
                "protocol_fingerprint": protocol_fingerprint,
                "source_fingerprint": source_fingerprint,
                "gate_audit": error.audit,
                "error": str(error),
                "runtime_seconds": time.perf_counter() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
            exclusive=True,
        )
        raise
    fallback_pairs = required_fallback_pairs(
        elicitation.expert, tuple(item[4] for item in baselines)
    )
    fallback_started = time.perf_counter()
    fallbacks, fallback_audit = await query_pair_fallbacks_local_vllm(
        spec.node_names,
        fallback_pairs,
        CONTEXT,
        cache_path,
        config=config,
    )
    fallback_seconds = time.perf_counter() - fallback_started

    def fallback(first: int, second: int) -> tuple[int, int]:
        pair = min(first, second), max(first, second)
        orientation = fallbacks.get(pair)
        if orientation is None:
            raise CausalOrderInfeasibleError(
                "fallback expert returned no relation for an existing PC adjacency"
            )
        return orientation

    references: list[dict[str, Any]] = []
    all_feasible = True
    integration_seconds = 0.0
    for seed, sample_size, baseline_path, baseline, cpdag in baselines:
        integration_started = time.perf_counter()
        try:
            integrated = integrate_order_into_cpdag(
                cpdag, elicitation.expert, fallback_orienter=fallback
            )
            feasible = True
            error = None
            graph = graph_edges(integrated.graph)
            final_order = list(integrated.final_order)
            fallback_orientations = [list(edge) for edge in integrated.fallback_orientations]
            metrics = point_prediction_metrics(integrated.graph, spec.target)
        except CausalOrderInfeasibleError as exc:
            feasible = False
            all_feasible = False
            error = f"{type(exc).__name__}: {exc}"
            graph = None
            final_order = None
            fallback_orientations = None
            metrics = None
        elapsed = time.perf_counter() - integration_started
        integration_seconds += elapsed
        output = _causal_order_path(results_root, output_run, seed, sample_size)
        record = {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "causal_order_integration",
            "seed": seed,
            "sample_size": sample_size,
            "method": METHODS[2],
            "protocol_fingerprint": protocol_fingerprint,
            "source_fingerprint": source_fingerprint,
            "baseline_path": str(baseline_path.resolve()),
            "baseline_sha256": sha256(baseline_path),
            "feasible": feasible,
            "error": error,
            "graph": graph,
            "final_order": final_order,
            "fallback_orientations": fallback_orientations,
            "metrics": metrics,
            "integration_runtime_seconds": elapsed,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        if output.exists():
            existing = read_json(output)
            for key in (
                "study",
                "study_version",
                "run_id",
                "stage",
                "seed",
                "sample_size",
                "method",
                "protocol_fingerprint",
                "source_fingerprint",
                "baseline_path",
                "baseline_sha256",
                "feasible",
                "error",
                "graph",
                "final_order",
                "fallback_orientations",
                "metrics",
            ):
                if existing.get(key) != record.get(key):
                    raise ValueError(
                        f"existing causal-order partial {key} differs: {output}"
                    )
        else:
            write_json(output, record, exclusive=True)
        references.append({"path": str(output.resolve()), "sha256": sha256(output)})

    gpu_after = _gpu_snapshot()
    peak_observed = max(
        int(gpu_before["maximum_compute_process_used_memory_mib"]),
        int(gpu_after["maximum_compute_process_used_memory_mib"]),
    )
    if peak_observed <= 0 or peak_observed >= MAX_GPU_MEMORY_MIB:
        raise RuntimeError("Qwen GPU memory measurement violates the benchmark gate")
    qwen_pipeline_runtime = time.perf_counter() - started
    job_runtime = time.time() - job_started_unix
    if job_runtime >= MAX_BENCHMARK_RUNTIME_SECONDS:
        raise RuntimeError(
            f"20-node Qwen benchmark took {job_runtime:.1f}s, exceeding one hour"
        )
    write_json(
        expert_path,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "stage": "qwen",
            "protocol": protocol,
            "protocol_fingerprint": protocol_fingerprint,
            "source_fingerprint": source_fingerprint,
            "subnetwork": {
                "target": SUBNETWORK_TARGET,
                "panel_indices": list(spec.panel_indices),
                "original_indices": list(spec.original_indices),
                "node_names": list(spec.node_names),
                "truth": graph_edges(spec.target),
            },
            "expert": _expert_record(elicitation.expert),
            "elicitation_audit": dict(elicitation.audit),
            "fallback_pairs": [list(pair) for pair in fallback_pairs],
            "fallback_orientations": [
                {
                    "pair": list(pair),
                    "orientation": None if orientation is None else list(orientation),
                }
                for pair, orientation in sorted(fallbacks.items())
            ],
            "fallback_audit": dict(fallback_audit),
            "fallback_runtime_seconds": fallback_seconds,
            "integration_runtime_seconds": integration_seconds,
            "cache_by_purpose": _cache_purpose_summary(cache_path),
            "request_cache": {"path": str(cache_path.resolve()), "sha256": sha256(cache_path)},
            "runtime_manifest": {
                "path": str(runtime_manifest_path.resolve()),
                "sha256": sha256(runtime_manifest_path),
            },
            "causal_order_partials": references,
            "all_40_feasible": all_feasible,
            "gpu": {
                "before": gpu_before,
                "after": gpu_after,
                "peak_observed_mib": peak_observed,
                "gate_mib": MAX_GPU_MEMORY_MIB,
            },
            "runtime_seconds": qwen_pipeline_runtime,
            "job_runtime_seconds": job_runtime,
            "runtime_gate_seconds": MAX_BENCHMARK_RUNTIME_SECONDS,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "command": list(sys.argv),
            "host": os.uname().nodename,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(
        f"completed 20-node exhaustive Qwen benchmark in {job_runtime:.1f}s: {expert_path}",
        flush=True,
    )
    return expert_path


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def _validate_metrics(
    recorded: Mapping[str, Any], recomputed: Mapping[str, float], label: str
) -> None:
    for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
        if not math.isclose(
            float(recorded[key]), float(recomputed[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} metric {key} differs")


def aggregate(output_run: str, results_root: Path, model_path: Path) -> Path:
    output_run = run_id(output_run)
    root = _root(results_root, output_run)
    expert_path = root / "expert.json"
    expert = read_json(expert_path)
    if not expert.get("all_40_feasible"):
        raise ValueError("Causal Order is not feasible on every benchmark slice")
    spec = build_subnetwork_spec(model_path)
    raw_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for seed in DATA_SEEDS:
        for sample_size in SAMPLE_SIZES:
            baseline_path = _baseline_path(results_root, output_run, seed, sample_size)
            baseline = read_json(baseline_path)
            _validate_baseline(baseline, output_run, seed, sample_size)
            for method in ("PC", "GES"):
                artifact = baseline["methods"][method]
                metrics = point_prediction_metrics(
                    _partial(artifact["partial_graph"]), spec.target
                )
                _validate_metrics(artifact["metrics"], metrics, method)
                raw_rows.append(
                    {
                        "seed": seed,
                        "sample_size": sample_size,
                        "method": method,
                        "feasible": 1,
                        **metrics,
                        "method_runtime_seconds": float(artifact["fit"]["runtime_seconds"]),
                    }
                )
            causal_path = _causal_order_path(results_root, output_run, seed, sample_size)
            causal = read_json(causal_path)
            if not causal.get("feasible"):
                raise ValueError(f"Causal Order is infeasible for seed={seed}, n={sample_size}")
            if causal.get("baseline_sha256") != sha256(baseline_path):
                raise ValueError("Causal Order baseline reference differs")
            metrics = point_prediction_metrics(_graph(causal["graph"]), spec.target)
            _validate_metrics(causal["metrics"], metrics, METHODS[2])
            raw_rows.append(
                {
                    "seed": seed,
                    "sample_size": sample_size,
                    "method": METHODS[2],
                    "feasible": 1,
                    **metrics,
                    "method_runtime_seconds": float(causal["integration_runtime_seconds"]),
                }
            )
            references.extend(
                [
                    {"path": str(baseline_path.resolve()), "sha256": sha256(baseline_path)},
                    {"path": str(causal_path.resolve()), "sha256": sha256(causal_path)},
                ]
            )
    expected = len(DATA_SEEDS) * len(SAMPLE_SIZES) * len(METHODS)
    if len(raw_rows) != expected:
        raise RuntimeError("aggregate does not cover the complete 20x2x3 benchmark grid")
    summary: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        for method in METHODS:
            group = [
                row
                for row in raw_rows
                if row["sample_size"] == sample_size and row["method"] == method
            ]
            record: dict[str, Any] = {
                "sample_size": sample_size,
                "method": method,
                "replicate_count": len(group),
            }
            for key in ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1"):
                mean, standard_error = _mean_se([float(row[key]) for row in group])
                record[f"{key}_mean"] = mean
                record[f"{key}_se"] = standard_error
            record["mean_method_runtime_seconds"] = statistics.fmean(
                float(row["method_runtime_seconds"]) for row in group
            )
            summary.append(record)
    cache_rows = expert["cache_by_purpose"]
    token_summary = {
        "triplet_count": TRIPLET_COUNT,
        "request_count": sum(int(row["request_count"]) for row in cache_rows),
        "input_tokens": sum(int(row["prompt_tokens"]) for row in cache_rows),
        "output_tokens": sum(int(row["completion_tokens"]) for row in cache_rows),
        "attempts": sum(int(row["attempts"]) for row in cache_rows),
        "parse_rate": float(expert["elicitation_audit"]["parse_rate"]),
        "qwen_runtime_seconds": float(expert["runtime_seconds"]),
        "qwen_job_runtime_seconds": float(expert["job_runtime_seconds"]),
        "runtime_gate_seconds": MAX_BENCHMARK_RUNTIME_SECONDS,
        "by_purpose": cache_rows,
    }
    write_csv(root / "raw_results.csv", raw_rows)
    write_csv(root / "summary.csv", summary)
    write_json(root / "token_summary.json", token_summary)
    write_json(
        root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "run_id": output_run,
            "protocol": expert["protocol"],
            "protocol_fingerprint": expert["protocol_fingerprint"],
            "source_fingerprint": expert["source_fingerprint"],
            "expert_path": str(expert_path.resolve()),
            "expert_sha256": sha256(expert_path),
            "token_summary": token_summary,
            "artifact_references": references,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    print(f"aggregated 20-node causal-order benchmark: {root}", flush=True)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare-manifest", "baseline", "qwen", "aggregate"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    if arguments.stage == "prepare-manifest":
        if arguments.image_path is None or arguments.checkpoint_path is None:
            raise ValueError("image and checkpoint paths are required")
        prepare_runtime_manifest(output_run, arguments.image_path, arguments.checkpoint_path)
    elif arguments.stage == "baseline":
        if arguments.task_index is None:
            raise ValueError("--task-index is required for baseline computation")
        seed, sample_size = slice_for_task(arguments.task_index)
        compute_baseline_slice(
            seed, sample_size, output_run, arguments.results_root, arguments.model_path
        )
    elif arguments.stage == "qwen":
        if arguments.runtime_manifest is None:
            raise ValueError("--runtime-manifest is required for Qwen inference")
        asyncio.run(
            run_qwen(
                output_run,
                arguments.results_root,
                arguments.model_path,
                arguments.runtime_manifest,
                arguments.base_url,
            )
        )
    else:
        aggregate(output_run, arguments.results_root, arguments.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
