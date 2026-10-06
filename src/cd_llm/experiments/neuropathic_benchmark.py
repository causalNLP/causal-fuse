from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from cd_llm.core.finite_support import (
    Claim,
    CompletionError,
    DataEnsemble,
    EvidenceGraph,
    Graph,
    PartialGraph,
    _causal_learn_partial,
    build_metric_workspace,
    build_synthetic_evidence,
    claim_value_matrix,
    distribution_metrics,
    graph_edges,
    graph_from_edges,
    preference_fusion,
    sample_completions,
    sample_resolved_claims,
    stable_seed,
)
from cd_llm.data.generation.neuropathic import (
    CANONICAL_VARIABLE_NAMES,
    generate_dataset,
    load_model,
    write_csv as write_neuropathic_csv,
    write_metadata as write_neuropathic_metadata,
)
from cd_llm.data.loaders.neuropathic import load_metadata, load_simulator_edges
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_runtime import run_cpc, run_ges
from cd_llm.paths import CODE_ROOT, WORKSPACE_ROOT


STUDY = "neuropathic_cervical_panel_a"
STUDY_VERSION = "3.0"
PANEL_NAME = "cervical_c2_c3"
PANEL_ANCHOR = "DLS C2-C3"
SAMPLE_SIZES = (200, 1000)
DATA_SEEDS = tuple(range(20))
CPC_ALPHAS = (0.005, 0.01, 0.05)
PRIMARY_CPC_ALPHA = 0.01
COMPLETION_LIMIT = 200
CLAIM_POOL_COUNT = 10
EVIDENCE_COUNT = 5
CLAIM_POOL_SEED = 0
EVIDENCE_SELECTION_SEED = 0
EVIDENCE_CONFIDENCE = 3.7 / 4.7
METHODS = (
    "CPC",
    "GES",
    "Claim-Fuse (CPC)",
    "Claim-Fuse (CPC+GES)",
    "Claim-Fuse (CPC grid)",
)

DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "generated" / "neuropathic"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "results"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    anchor: str
    original_indices: tuple[int, ...]
    node_names: tuple[str, ...]
    directed_edges: tuple[tuple[int, int], ...]
    pathophysiology_count: int
    pattern_count: int
    symptom_count: int

    @property
    def node_count(self) -> int:
        return len(self.original_indices)

    @property
    def edge_count(self) -> int:
        return len(self.directed_edges)

    @property
    def target(self) -> Graph:
        return graph_from_edges(self.node_count, self.directed_edges)


@dataclass(frozen=True)
class BenchmarkConfig:
    panel_name: str = PANEL_NAME
    panel_anchor: str = PANEL_ANCHOR
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES
    data_seeds: tuple[int, ...] = DATA_SEEDS
    cpc_alphas: tuple[float, ...] = CPC_ALPHAS
    primary_cpc_alpha: float = PRIMARY_CPC_ALPHA
    completion_limit: int = COMPLETION_LIMIT
    claim_pool_count: int = CLAIM_POOL_COUNT
    evidence_count: int = EVIDENCE_COUNT
    claim_pool_seed: int = CLAIM_POOL_SEED
    evidence_selection_seed: int = EVIDENCE_SELECTION_SEED
    evidence_confidence: float = EVIDENCE_CONFIDENCE


def config_for_seed_count(seed_count: int) -> BenchmarkConfig:
    if not 1 <= seed_count <= len(DATA_SEEDS):
        raise ValueError(f"seed count must lie in [1,{len(DATA_SEEDS)}]")
    return BenchmarkConfig(data_seeds=DATA_SEEDS[:seed_count])


def build_panel_spec(model_path: Path) -> PanelSpec:
    names = list(CANONICAL_VARIABLE_NAMES)
    index = {name: position for position, name in enumerate(names)}
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(load_simulator_edges(model_path))
    symptom_nodes = set(names[79:])
    selected_symptoms = {
        node for node in nx.descendants(graph, PANEL_ANCHOR) if node in symptom_nodes
    }
    closure = set(selected_symptoms)
    for node in tuple(selected_symptoms):
        closure.update(nx.ancestors(graph, node))
    ordered_names = tuple(name for name in names if name in closure)
    original_indices = tuple(index[name] for name in ordered_names)
    local_index = {name: position for position, name in enumerate(ordered_names)}
    directed_edges = tuple(
        sorted(
            (local_index[source], local_index[target])
            for source, target in graph.edges()
            if source in closure and target in closure
        )
    )
    if any(
        parent not in closure
        for node in closure
        for parent in graph.predecessors(node)
    ):
        raise ValueError("neuropathic pilot panel is not ancestor closed")
    spec = PanelSpec(
        name=PANEL_NAME,
        anchor=PANEL_ANCHOR,
        original_indices=original_indices,
        node_names=ordered_names,
        directed_edges=directed_edges,
        pathophysiology_count=sum(position < 27 for position in original_indices),
        pattern_count=sum(27 <= position < 79 for position in original_indices),
        symptom_count=sum(position >= 79 for position in original_indices),
    )
    expected = (35, 102, 7, 12, 16)
    actual = (
        spec.node_count,
        spec.edge_count,
        spec.pathophysiology_count,
        spec.pattern_count,
        spec.symptom_count,
    )
    if actual != expected:
        raise ValueError(f"neuropathic pilot panel differs: {actual} versus {expected}")
    return spec


def _panel_hash(spec: PanelSpec) -> str:
    encoded = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint() -> str:
    paths = (
        Path(__file__),
        CODE_ROOT / "src" / "cd_llm" / "experiments" / "neuropathic_runtime.py",
        CODE_ROOT / "src" / "cd_llm" / "data" / "generation" / "neuropathic.py",
        CODE_ROOT / "src" / "cd_llm" / "data" / "loaders" / "neuropathic.py",
        CODE_ROOT / "src" / "cd_llm" / "core" / "cpc.py",
        CODE_ROOT / "src" / "cd_llm" / "core" / "finite_support.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().relative_to(CODE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _protocol_fingerprint(config: BenchmarkConfig, panel: PanelSpec) -> str:
    payload = {"study": STUDY, "version": STUDY_VERSION, "config": asdict(config), "panel": asdict(panel)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "causal-learn",
        "JPype1",
        "networkx",
        "numpy",
        "pandas",
        "py-tetrad",
        "scipy",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _dataset_paths(data_root: Path, sample_size: int, seed: int) -> tuple[Path, Path]:
    stem = f"neuropathic_n{sample_size}_level0_seed{seed}"
    return data_root / f"{stem}.csv", data_root / f"{stem}.json"


def _panel_paths(data_root: Path, sample_size: int, seed: int) -> tuple[Path, Path]:
    panel_root = data_root / "panels" / PANEL_NAME
    stem = f"neuropathic_{PANEL_NAME}_n{sample_size}_seed{seed}"
    return panel_root / f"{stem}.csv", panel_root / f"{stem}.json"


def _read_full_dataset(data_path: Path, metadata_path: Path, sample_size: int, seed: int) -> np.ndarray:
    metadata = load_metadata(metadata_path)
    expected = {
        "sample_size": sample_size,
        "level": 0,
        "seed": seed,
        "observed_indices": list(range(222)),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"full dataset metadata {key} differs for seed {seed}")
    frame = pd.read_csv(data_path, index_col=0)
    data = frame.to_numpy(dtype=np.int64, copy=True)
    if data.shape != (sample_size, 222) or np.any((data != 0) & (data != 1)):
        raise ValueError(f"invalid full neuropathic dataset for seed {seed}: {data.shape}")
    return data


def ensure_nested_datasets(
    data_root: Path,
    model_path: Path,
    panel: PanelSpec,
    seed: int,
) -> dict[int, tuple[np.ndarray, Path, Path]]:
    full_1000_path, full_1000_meta = _dataset_paths(data_root, 1000, seed)
    if full_1000_path.exists() and full_1000_meta.exists():
        full_1000 = _read_full_dataset(full_1000_path, full_1000_meta, 1000, seed)
    else:
        model = load_model(model_path)
        full_1000, observed_indices = generate_dataset(model, 1000, 0, seed)
        full_1000_path.parent.mkdir(parents=True, exist_ok=True)
        write_neuropathic_csv(full_1000_path, full_1000, observed_indices)
        write_neuropathic_metadata(
            full_1000_meta,
            sample_size=1000,
            level=0,
            seed=seed,
            observed_indices=observed_indices,
        )
    full_by_size = {200: full_1000[:200].copy(), 1000: full_1000}
    result: dict[int, tuple[np.ndarray, Path, Path]] = {}
    for sample_size, full_data in full_by_size.items():
        full_path, full_meta = _dataset_paths(data_root, sample_size, seed)
        if sample_size == 200:
            write_neuropathic_csv(full_path, full_data, list(range(222)))
            write_neuropathic_metadata(
                full_meta,
                sample_size=200,
                level=0,
                seed=seed,
                observed_indices=list(range(222)),
            )
        panel_data = full_data[:, panel.original_indices]
        panel_path, panel_meta = _panel_paths(data_root, sample_size, seed)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        write_neuropathic_csv(panel_path, panel_data, panel.original_indices)
        write_json(
            panel_meta,
            {
                "study": STUDY,
                "panel": asdict(panel),
                "panel_hash": _panel_hash(panel),
                "sample_size": sample_size,
                "seed": seed,
                "source_data_path": str(full_path.resolve()),
                "source_data_sha256": sha256(full_path),
                "nested_from_n1000": True,
            },
        )
        result[sample_size] = (panel_data, panel_path, panel_meta)
    return result


def select_evidence(panel: PanelSpec) -> tuple[tuple[Claim, ...], EvidenceGraph, dict[str, Any]]:
    pool = tuple(
        claim
        for claim in sample_resolved_claims(
            panel.target,
            "dag",
            CLAIM_POOL_COUNT,
            CLAIM_POOL_SEED,
        )
        if claim.group == "I"
    )
    if len(pool) != CLAIM_POOL_COUNT or len(set(pool)) != CLAIM_POOL_COUNT:
        raise RuntimeError("neuropathic Group-I claim pool must contain ten distinct claims")
    rng = np.random.default_rng(EVIDENCE_SELECTION_SEED)
    indices = tuple(
        int(index)
        for index in rng.choice(len(pool), size=EVIDENCE_COUNT, replace=False)
    )
    if indices != (4, 7, 2, 3, 5):
        raise AssertionError("neuropathic evidence selection differs from locked seed-0 draw")
    claims = tuple(pool[index] for index in indices)
    evidence = build_synthetic_evidence(
        claims,
        ("accurate",),
        EVIDENCE_CONFIDENCE,
        EVIDENCE_SELECTION_SEED,
    )
    if any(not node.correct or not node.context_compatible for node in evidence.nodes):
        raise RuntimeError("neuropathic evidence must be accurate and context compatible")
    return claims, evidence, {
        "claim_pool_seed": CLAIM_POOL_SEED,
        "evidence_selection_seed": EVIDENCE_SELECTION_SEED,
        "selected_pool_indices": indices,
        "confidence": EVIDENCE_CONFIDENCE,
    }


def _claim_records(claims: Sequence[Claim], evidence: EvidenceGraph, panel: PanelSpec) -> list[dict[str, Any]]:
    evidence_by_claim = {node.claim_index: node for node in evidence.nodes}
    result = []
    for index, claim in enumerate(claims):
        node = evidence_by_claim[index]
        result.append(
            {
                "claim_index": index,
                "claim_id": claim.claim_id,
                "group": claim.group,
                "kind": claim.kind,
                "truth": claim.truth,
                "source": claim.source,
                "source_name": panel.node_names[claim.source],
                "target": claim.target,
                "target_name": panel.node_names[claim.target],
                "stance": node.stance,
                "correct": node.correct,
                "context_compatible": node.context_compatible,
                "provenance_family": node.provenance_family,
                "confidence": node.confidence,
                "reported_confidence": node.reported_confidence,
                "effective_confidence": node.resolved_confidence().effective_confidence,
                "treatment": node.treatment,
            }
        )
    return result


def _fit_discovery(data: np.ndarray, panel: PanelSpec) -> tuple[dict[str, PartialGraph], dict[str, Any]]:
    outputs: dict[str, PartialGraph] = {}
    diagnostics: dict[str, Any] = {}
    for alpha in CPC_ALPHAS:
        key = f"cpc_alpha_{alpha:g}"
        started = time.perf_counter()
        graph, settings = run_cpc(
            data,
            panel.node_names,
            alpha=alpha,
        )
        outputs[key] = graph
        diagnostics[key] = {"runtime_seconds": time.perf_counter() - started, **settings}
    started = time.perf_counter()
    graph, settings = run_ges(data, panel.node_names)
    outputs["ges_bdeu_ess1"] = _causal_learn_partial(graph, "cpdag")
    diagnostics["ges_bdeu_ess1"] = {"runtime_seconds": time.perf_counter() - started, **settings}
    return outputs, diagnostics


def _complete_outputs(
    outputs: Mapping[str, PartialGraph],
    seed: int,
    sample_size: int,
) -> tuple[dict[str, tuple[Graph, ...]], dict[str, Any]]:
    completions: dict[str, tuple[Graph, ...]] = {}
    diagnostics: dict[str, Any] = {}
    for key, output in outputs.items():
        started = time.perf_counter()
        try:
            graphs = sample_completions(
                output,
                "dag",
                COMPLETION_LIMIT,
                stable_seed(STUDY, seed, sample_size, key, "completion"),
            )
        except CompletionError as error:
            diagnostics[key] = {
                "available": False,
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            continue
        completions[key] = graphs
        diagnostics[key] = {
            "available": True,
            "runtime_seconds": time.perf_counter() - started,
            "completion_count": len(graphs),
        }
    return completions, diagnostics


def _ensemble(families: Sequence[tuple[Graph, ...]]) -> DataEnsemble:
    if not families or any(not family for family in families):
        raise ValueError("graph-distribution families must be nonempty")
    masses: Counter[Graph] = Counter()
    for family in families:
        family_mass = 1.0 / len(families)
        graph_mass = family_mass / len(family)
        for graph in family:
            masses[graph] += graph_mass
    graphs = tuple(sorted(masses))
    probabilities = np.asarray([[masses[graph] for graph in graphs]], dtype=float)
    probabilities /= probabilities.sum()
    return DataEnsemble(
        "dag",
        graphs,
        probabilities,
        (tuple(len(family) for family in families),),
        (len(families),),
    )


def _method_distribution(
    method: str,
    completions: Mapping[str, tuple[Graph, ...]],
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
) -> tuple[DataEnsemble, np.ndarray] | None:
    primary_key = f"cpc_alpha_{PRIMARY_CPC_ALPHA:g}"
    specifications = {
        "CPC": (primary_key,),
        "GES": ("ges_bdeu_ess1",),
        "Claim-Fuse (CPC)": (primary_key,),
        "Claim-Fuse (CPC+GES)": (primary_key, "ges_bdeu_ess1"),
        "Claim-Fuse (CPC grid)": tuple(
            f"cpc_alpha_{alpha:g}" for alpha in CPC_ALPHAS
        ),
    }
    keys = specifications[method]
    if any(key not in completions for key in keys):
        return None
    ensemble = _ensemble(tuple(completions[key] for key in keys))
    distribution = ensemble.q_data
    if method.startswith("Claim-Fuse"):
        values = claim_value_matrix(ensemble.graphs, claims)
        distribution = preference_fusion(ensemble, values, evidence, "ceg_fuse")
    return ensemble, distribution


def evaluate_sample_size(
    data: np.ndarray,
    panel: PanelSpec,
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    seed: int,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs, fit_diagnostics = _fit_discovery(data, panel)
    completions, completion_diagnostics = _complete_outputs(outputs, seed, sample_size)
    rows: list[dict[str, Any]] = []
    method_artifacts: dict[str, Any] = {}
    for method in METHODS:
        prepared = _method_distribution(method, completions, claims, evidence)
        if prepared is None:
            rows.append(
                {
                    "seed": seed,
                    "sample_size": sample_size,
                    "method": method,
                    "feasible": 0,
                    "expected_distance": None,
                    "energy_score": None,
                    "adjacency_f1": None,
                    "orientation_f1": None,
                }
            )
            method_artifacts[method] = {"available": False}
            continue
        ensemble, distribution = prepared
        metrics = distribution_metrics(
            ensemble.graphs,
            distribution,
            panel.target,
            "dag",
            workspace=build_metric_workspace(ensemble.graphs, panel.target, "dag"),
        )
        rows.append(
            {
                "seed": seed,
                "sample_size": sample_size,
                "method": method,
                "feasible": 1,
                "expected_distance": metrics["expected_distance"],
                "energy_score": metrics["energy_score"],
                "adjacency_f1": metrics["adjacency_f1"],
                "orientation_f1": metrics["orientation_f1"],
            }
        )
        method_artifacts[method] = {
            "available": True,
            "support_size": len(ensemble.graphs),
            "graphs": [graph_edges(graph) for graph in ensemble.graphs],
            "probabilities": distribution.tolist(),
        }
    return rows, {
        "fit_diagnostics": fit_diagnostics,
        "completion_diagnostics": completion_diagnostics,
        "methods": method_artifacts,
    }


def compute_task(
    run_id_value: str,
    task_index: int,
    data_root: Path,
    output_root: Path,
    model_path: Path,
    force: bool,
    seed_count: int,
) -> Path:
    config = config_for_seed_count(seed_count)
    if not 0 <= task_index < len(config.data_seeds):
        raise ValueError(f"task index must lie in [0,{len(config.data_seeds) - 1}]")
    seed = config.data_seeds[task_index]
    result_root = output_root / run_id_value / "neuropathic_benchmark" / "panel_a"
    partial_path = result_root / "partials" / f"seed_{seed:02d}.json"
    if partial_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing partial: {partial_path}")
    panel = build_panel_spec(model_path)
    datasets = ensure_nested_datasets(data_root, model_path, panel, seed)
    claims, evidence, evidence_settings = select_evidence(panel)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    sample_artifacts: dict[str, Any] = {}
    for sample_size in config.sample_sizes:
        data, data_path, metadata_path = datasets[sample_size]
        sample_rows, artifact = evaluate_sample_size(
            data,
            panel,
            claims,
            evidence,
            seed,
            sample_size,
        )
        rows.extend(sample_rows)
        sample_artifacts[str(sample_size)] = {
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256(data_path),
            "metadata_path": str(metadata_path.resolve()),
            **artifact,
        }
    payload = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "run_id": run_id_value,
        "task_index": task_index,
        "data_seed": seed,
        "config": asdict(config),
        "panel": asdict(panel),
        "panel_hash": _panel_hash(panel),
        "target": graph_edges(panel.target),
        "claims": _claim_records(claims, evidence, panel),
        "evidence_settings": evidence_settings,
        "rows": rows,
        "sample_artifacts": sample_artifacts,
        "runtime_seconds": time.perf_counter() - started,
        "source_fingerprint": _source_fingerprint(),
        "protocol_fingerprint": _protocol_fingerprint(config, panel),
        "package_versions": _package_versions(),
        "command": sys.argv,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(partial_path, payload)
    print(f"completed neuropathic Panel A seed={seed}: {partial_path}", flush=True)
    return partial_path


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def aggregate_run(
    run_id_value: str,
    output_root: Path,
    model_path: Path,
    seed_count: int,
) -> Path:
    config = config_for_seed_count(seed_count)
    panel = build_panel_spec(model_path)
    result_root = output_root / run_id_value / "neuropathic_benchmark" / "panel_a"
    partials = [
        read_json(result_root / "partials" / f"seed_{seed:02d}.json")
        for seed in config.data_seeds
    ]
    expected_source = _source_fingerprint()
    expected_protocol = _protocol_fingerprint(config, panel)
    for seed, partial in zip(config.data_seeds, partials):
        if partial.get("data_seed") != seed:
            raise ValueError(f"partial seed differs for seed {seed}")
        if partial.get("source_fingerprint") != expected_source:
            raise ValueError(f"partial source fingerprint differs for seed {seed}")
        if partial.get("protocol_fingerprint") != expected_protocol:
            raise ValueError(f"partial protocol fingerprint differs for seed {seed}")
    raw_rows = [row for partial in partials for row in partial["rows"]]
    availability = []
    summary = []
    common_complete: dict[int, list[int]] = {}
    metrics = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
    for sample_size in config.sample_sizes:
        sample_rows = [row for row in raw_rows if int(row["sample_size"]) == sample_size]
        common = [
            seed
            for seed in config.data_seeds
            if all(
                int(next(row for row in sample_rows if row["seed"] == seed and row["method"] == method)["feasible"])
                for method in METHODS
            )
        ]
        common_complete[sample_size] = common
        for method in METHODS:
            method_rows = [row for row in sample_rows if row["method"] == method]
            availability.append(
                {
                    "sample_size": sample_size,
                    "method": method,
                    "available_count": sum(int(row["feasible"]) for row in method_rows),
                    "total_count": len(method_rows),
                }
            )
            complete_rows = [row for row in method_rows if int(row["seed"]) in common]
            record: dict[str, Any] = {
                "sample_size": sample_size,
                "method": method,
                "replicate_count": len(complete_rows),
            }
            for metric in metrics:
                mean, se = _mean_se([float(row[metric]) for row in complete_rows])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_se"] = se
            summary.append(record)
    pilot_gate_minimum = 16 if seed_count == 20 else seed_count
    pilot_gate_passed = all(
        len(common_complete[size]) >= pilot_gate_minimum
        for size in config.sample_sizes
    )
    write_csv(result_root / "raw_results.csv", raw_rows)
    write_csv(result_root / "availability.csv", availability)
    write_csv(result_root / "summary.csv", summary)
    write_json(result_root / "claims.json", partials[0]["claims"])
    manifest = {
        "study": STUDY,
        "study_version": STUDY_VERSION,
        "run_id": run_id_value,
        "config": asdict(config),
        "panel": asdict(panel),
        "panel_hash": _panel_hash(panel),
        "common_complete_seeds": {str(key): value for key, value in common_complete.items()},
        "pilot_gate_minimum": pilot_gate_minimum,
        "pilot_gate_passed": pilot_gate_passed,
        "source_fingerprint": expected_source,
        "protocol_fingerprint": expected_protocol,
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "python": sys.version,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_root / "run_manifest.json", manifest)
    report_lines = [
        "# Neuropathic Pain Panel A",
        "",
        f"Panel: {panel.name} ({panel.node_count} nodes, {panel.edge_count} edges).",
        f"Common complete seeds: n=200: {len(common_complete[200])}/{len(config.data_seeds)}; n=1000: {len(common_complete[1000])}/{len(config.data_seeds)}.",
        f"Pilot gate passed: {pilot_gate_passed}.",
        "",
        "Panel B was not run.",
    ]
    write_text(result_root / "REPORT.md", "\n".join(report_lines) + "\n")
    print(f"aggregated neuropathic Panel A: {result_root}", flush=True)
    return result_root / "summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compute", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=CODE_ROOT / "resources" / "neuropathic" / "bnm.pickle")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-count", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_run_id = run_id(args.run_id)
    if args.stage == "compute":
        if args.task_index is None:
            raise ValueError("--task-index is required for compute")
        compute_task(
            selected_run_id,
            args.task_index,
            args.data_root,
            args.output_root,
            args.model_path,
            args.force,
            args.seed_count,
        )
    else:
        if args.task_index is not None:
            raise ValueError("--task-index is not valid for aggregate")
        aggregate_run(
            selected_run_id,
            args.output_root,
            args.model_path,
            args.seed_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
