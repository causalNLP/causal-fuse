"""Run Claim-Fuse-SMC beside immutable finite-support benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cd_llm.core.finite_support import (
    Claim,
    EvidenceGraph,
    EvidenceNode,
    Graph,
    build_synthetic_evidence,
    distribution_metrics,
    graph_edges,
    graph_from_edges,
    resolve_evidence_confidence,
    sample_resolved_claims,
    stable_seed,
)
from cd_llm.experiments.common.artifacts import (
    json_value,
    read_json,
    run_id,
    sha256,
    write_csv,
    write_json,
)
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.methods.evidence_guided import (
    ClaimFuseSMCConfig,
    ClaimFuseSMCResult,
    claim_fuse_smc,
)
from cd_llm.methods.graph_scores import BDeuScore, DecomposableDAGScore
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "claim_fuse_smc_sidecar"
STUDY_VERSION = 2
SAMPLE_SIZES = (200, 1_000)
REFERENCE_RUNS = {
    "synthetic": "20260816T161205Z",
    "neuropathic": "20260821T081942Z",
}
REFERENCE_COUNTS = {"synthetic": 50, "neuropathic": 20}
ARCHIVED_EVIDENCE_PROFILE = "archived-5"
PASSIVE_GROUP_I_20_PROFILE = "passive-group-i-20"
EVIDENCE_PROFILES = (ARCHIVED_EVIDENCE_PROFILE, PASSIVE_GROUP_I_20_PROFILE)
PASSIVE_CLAIM_SEED = 0
PASSIVE_CLAIM_COUNT = 20
PASSIVE_CONFIDENCE = 3.7 / 4.7
PRODUCTION_CONFIG = ClaimFuseSMCConfig()
REFERENCE_CONFIG = ClaimFuseSMCConfig(
    particle_count=1_000,
    initialization_updates=500,
    temperatures=tuple(index / 10.0 for index in range(11)),
    adaptive_tempering=False,
    rejuvenation_steps=10,
    rejuvenate_only_after_resampling=False,
    warm_start_score_mass=0.0,
)
CANDIDATE_CONFIGS = {
    "candidate-384-uniform": ClaimFuseSMCConfig(
        particle_count=384,
        initialization_updates=192,
        warm_start_score_mass=0.0,
    ),
    "candidate-384": ClaimFuseSMCConfig(
        particle_count=384,
        initialization_updates=192,
        warm_start_score_mass=0.9,
    ),
    "candidate-256": ClaimFuseSMCConfig(
        particle_count=256,
        initialization_updates=160,
        warm_start_score_mass=0.9,
    ),
    "candidate-160": ClaimFuseSMCConfig(
        particle_count=160,
        initialization_updates=64,
        warm_start_score_mass=0.9,
    ),
    "candidate-128": ClaimFuseSMCConfig(
        particle_count=128,
        initialization_updates=64,
        warm_start_score_mass=0.9,
    ),
    "candidate-96": ClaimFuseSMCConfig(
        particle_count=96,
        initialization_updates=48,
        rejuvenation_steps=1,
        max_tempering_stages=3,
        warm_start_score_mass=0.9,
    ),
}
SMOKE_CONFIG = ClaimFuseSMCConfig(
    particle_count=20,
    initialization_updates=10,
    temperatures=(0.0, 0.5, 1.0),
    adaptive_tempering=False,
    rejuvenation_steps=2,
    rejuvenate_only_after_resampling=False,
    warm_start_score_mass=0.0,
    proposal_cache_size=32,
)
PROFILE_CONFIGS = {
    "production": PRODUCTION_CONFIG,
    "reference": REFERENCE_CONFIG,
    "smoke": SMOKE_CONFIG,
    **CANDIDATE_CONFIGS,
}


@dataclass(frozen=True)
class _SampleProblem:
    sample_size: int
    score: DecomposableDAGScore
    warm_starts: tuple[Graph, ...]
    baseline_runtime_seconds: float | None


@dataclass(frozen=True)
class _ReplicateProblem:
    identifier: int
    reference_path: Path
    target: Graph
    claims: tuple[Claim, ...]
    evidence: EvidenceGraph
    evidence_profile: str
    claim_records: tuple[Mapping[str, Any], ...]
    evidence_settings: Mapping[str, Any]
    samples: tuple[_SampleProblem, ...]

    @property
    def baseline_runtime_seconds(self) -> float | None:
        values = [sample.baseline_runtime_seconds for sample in self.samples]
        if any(value is None for value in values):
            return None
        return math.fsum(float(value) for value in values if value is not None)


def _graph(record: Mapping[str, Any], node_count: int) -> Graph:
    return graph_from_edges(
        node_count,
        record.get("directed", ()),
        record.get("bidirected", ()),
        record.get("undirected", ()),
    )


def _claims(
    records: Sequence[Mapping[str, Any]],
    *,
    locked_confidence: float,
) -> tuple[tuple[Claim, ...], EvidenceGraph]:
    claims: list[Claim] = []
    nodes: list[EvidenceNode] = []
    for index, record in enumerate(records):
        claims.append(
            Claim(
                str(record["claim_id"]),
                str(record["group"]),
                str(record["kind"]),
                int(record["truth"]),
                int(record["source"]),
                int(record["target"]),
            )
        )
        nodes.append(
            EvidenceNode(
                f"e{index:04d}",
                index,
                int(record["stance"]),
                float(record.get("confidence", locked_confidence)),
                str(record["provenance_family"]),
                str(record.get("factor_id", f"factor-{index}")),
                str(record.get("context_id", "target")),
                bool(record["context_compatible"]),
                bool(record["correct"]),
                treatment=record.get("treatment"),
            )
        )
    return tuple(claims), EvidenceGraph("target", tuple(nodes))


def _confidence_records(
    records: Sequence[Mapping[str, Any]],
    locked_confidence: float,
) -> tuple[Mapping[str, Any], ...]:
    """Normalize fixed archived records without retaining deprecated fields."""

    normalized: list[Mapping[str, Any]] = []
    for record in records:
        reported = float(record.get("reported_confidence", record.get("confidence", locked_confidence)))
        resolved = resolve_evidence_confidence(reported, record.get("treatment"))
        normalized.append(
            {
                **{
                    str(key): value
                    for key, value in record.items()
                    if key not in {"uncertainty", "r_max", "preference_ratio"}
                },
                "confidence": resolved.reported_confidence,
                "reported_confidence": resolved.reported_confidence,
                "effective_confidence": resolved.effective_confidence,
                "treatment": resolved.treatment,
            }
        )
    return tuple(normalized)


def _claim_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        json_value(tuple(records)), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _passive_group_i_20(
    target: Graph,
    node_names: Sequence[str],
) -> tuple[tuple[Claim, ...], EvidenceGraph, tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    claims = tuple(
        claim
        for claim in sample_resolved_claims(
            target, "dag", PASSIVE_CLAIM_COUNT, PASSIVE_CLAIM_SEED
        )
        if claim.group == "I"
    )
    kind_counts = {
        kind: sum(claim.kind == kind for claim in claims)
        for kind in ("directed", "directed_exclusion", "adjacency", "non_adjacency")
    }
    if len(claims) != PASSIVE_CLAIM_COUNT or len(set(claims)) != PASSIVE_CLAIM_COUNT:
        raise RuntimeError("passive neuropathic evidence must contain 20 unique claims")
    if sum(claim.truth == 1 for claim in claims) != 10 or sum(
        claim.truth == 0 for claim in claims
    ) != 10:
        raise RuntimeError("passive neuropathic claims must have balanced truth values")
    if kind_counts != {
        "directed": 6,
        "directed_exclusion": 6,
        "adjacency": 4,
        "non_adjacency": 4,
    }:
        raise RuntimeError(f"passive neuropathic claim-kind balance differs: {kind_counts}")
    evidence = build_synthetic_evidence(
        claims,
        ("accurate",),
        PASSIVE_CONFIDENCE,
        PASSIVE_CLAIM_SEED,
    )
    evidence_by_claim = {node.claim_index: node for node in evidence.nodes}
    if len(evidence_by_claim) != PASSIVE_CLAIM_COUNT or any(
        not node.correct or not node.context_compatible for node in evidence.nodes
    ):
        raise RuntimeError("passive neuropathic evidence must be accurate and compatible")
    if len({node.provenance_family for node in evidence.nodes}) != PASSIVE_CLAIM_COUNT:
        raise RuntimeError("passive neuropathic evidence must use independent provenance")
    records: list[Mapping[str, Any]] = []
    for index, claim in enumerate(claims):
        node = evidence_by_claim[index]
        records.append(
            {
                "claim_index": index,
                "claim_id": claim.claim_id,
                "group": claim.group,
                "kind": claim.kind,
                "truth": claim.truth,
                "source": claim.source,
                "source_name": node_names[claim.source],
                "target": claim.target,
                "target_name": node_names[claim.target],
                "stance": node.stance,
                "confidence": node.confidence,
                "reported_confidence": node.reported_confidence,
                "effective_confidence": node.resolved_confidence(
                    PRODUCTION_CONFIG.soft_confidence_cap
                ).effective_confidence,
                "treatment": node.treatment,
                "provenance_family": node.provenance_family,
                "factor_id": node.factor_id,
                "context_id": node.context_id,
                "context_compatible": node.context_compatible,
                "correct": node.correct,
            }
        )
    settings = {
        "active_query": False,
        "claim_seed": PASSIVE_CLAIM_SEED,
        "claim_count": PASSIVE_CLAIM_COUNT,
        "claim_group": "I",
        "truth_counts": {"false": 10, "true": 10},
        "kind_counts": kind_counts,
        "confidence": PASSIVE_CONFIDENCE,
        "reported_confidence": PASSIVE_CONFIDENCE,
        "effective_confidence": min(
            PASSIVE_CONFIDENCE, PRODUCTION_CONFIG.soft_confidence_cap
        ),
        "treatment": "soft_evidence",
        "soft_confidence_cap": PRODUCTION_CONFIG.soft_confidence_cap,
    }
    return claims, evidence, tuple(records), settings


def _validate_evidence_profile(dataset: str, evidence_profile: str) -> None:
    if evidence_profile not in EVIDENCE_PROFILES:
        raise ValueError(f"unknown evidence profile: {evidence_profile}")
    if dataset != "neuropathic" and evidence_profile != ARCHIVED_EVIDENCE_PROFILE:
        raise ValueError("passive-group-i-20 is available only for neuropathic data")


def _output_root(
    results_root: Path,
    output_run: str,
    dataset: str,
    evidence_profile: str,
) -> Path:
    root = results_root / output_run / "claim_fuse_smc" / dataset
    if evidence_profile != ARCHIVED_EVIDENCE_PROFILE:
        root /= evidence_profile
    return root


def _distribution_metrics(distribution: Mapping[Graph, float], truth: Graph) -> dict[str, float]:
    graphs = tuple(distribution)
    probabilities = np.asarray([distribution[graph] for graph in graphs], dtype=float)
    return distribution_metrics(graphs, probabilities, truth, "dag")


def _serialize_distribution(distribution: Mapping[Graph, float]) -> list[dict[str, Any]]:
    return [
        {"graph": graph_edges(graph), "probability": float(distribution[graph])}
        for graph in sorted(distribution, key=lambda item: item.directed)
    ]


def _require_sample_sizes(payload: Mapping[str, Any]) -> None:
    available = tuple(int(value) for value in payload.get("sample_sizes", ()))
    if available and available != SAMPLE_SIZES:
        raise ValueError(f"reference sample sizes {available} do not match {SAMPLE_SIZES}")


def _synthetic_reference_path(
    results_root: Path, reference_run: str, task_index: int
) -> Path:
    candidates = (
        results_root
        / reference_run
        / "partials"
        / f"task_{task_index:03d}.json",
        results_root
        / reference_run
        / "test"
        / "partials"
        / f"task_{task_index:03d}.json",
    )
    matches = tuple(path for path in candidates if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one synthetic reference partial for task {task_index}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_synthetic_problem(
    results_root: Path,
    reference_run: str,
    task_index: int,
    config: ClaimFuseSMCConfig,
) -> _ReplicateProblem:
    reference_path = _synthetic_reference_path(
        results_root, reference_run, task_index
    )
    payload = read_json(reference_path)
    protocol = payload["protocol"]
    _require_sample_sizes(protocol)
    node_count = int(protocol["node_count"])
    if config.max_parents >= node_count:
        raise ValueError("Claim-Fuse-SMC max_parents must be smaller than node_count")
    target = _graph(payload["target"], node_count)
    claims, evidence = _claims(
        payload["evidence"]["claims"],
        locked_confidence=PASSIVE_CONFIDENCE,
    )
    baseline_rows = {
        (int(row["sample_size"]), str(row["method"])): row
        for row in payload["rows"]
        if row["method"] in ("PC", "GES")
    }
    samples: list[_SampleProblem] = []
    for sample_size in SAMPLE_SIZES:
        data = np.asarray(payload["nested_data"][str(sample_size)], dtype=float)
        method = payload["samples"][str(sample_size)]["methods"][
            "CEG-Fuse (PC+GES)"
        ]
        warm_starts = tuple(
            _graph(record, node_count) for record in method["graphs"]
        )
        if not warm_starts:
            raise ValueError(
                "synthetic reference has an empty PC+GES completion support; "
                "single-learner fallback is prohibited"
            )
        baseline_parts = [
            baseline_rows.get((sample_size, method_name))
            for method_name in ("PC", "GES")
        ]
        baseline_runtime = (
            math.fsum(float(row["runtime_seconds"]) for row in baseline_parts if row)
            if all(row and bool(row["feasible"]) for row in baseline_parts)
            else None
        )
        samples.append(
            _SampleProblem(
                sample_size,
                GaussianBICScore(
                    data,
                    standardize_data=True,
                    max_parents=config.max_parents,
                ),
                warm_starts,
                baseline_runtime,
            )
        )
    return _ReplicateProblem(
        int(payload["graph_index"]),
        reference_path,
        target,
        claims,
        evidence,
        ARCHIVED_EVIDENCE_PROFILE,
        _confidence_records(payload["evidence"]["claims"], PASSIVE_CONFIDENCE),
        {
            "active_query": False,
            "source": "locked-archived-run",
            "confidence": PASSIVE_CONFIDENCE,
        },
        tuple(samples),
    )


def _load_neuropathic_problem(
    results_root: Path,
    reference_run: str,
    task_index: int,
    config: ClaimFuseSMCConfig,
    evidence_profile: str,
) -> _ReplicateProblem:
    reference_path = (
        results_root
        / reference_run
        / "neuropathic_benchmark"
        / "panel_a"
        / "partials"
        / f"seed_{task_index:02d}.json"
    )
    payload = read_json(reference_path)
    _require_sample_sizes(payload["config"])
    node_count = len(payload["panel"]["original_indices"])
    if config.max_parents >= node_count:
        raise ValueError("Claim-Fuse-SMC max_parents must be smaller than node_count")
    target = _graph(payload["target"], node_count)
    if evidence_profile == PASSIVE_GROUP_I_20_PROFILE:
        claims, evidence, claim_records, evidence_settings = _passive_group_i_20(
            target, tuple(str(name) for name in payload["panel"]["node_names"])
        )
    else:
        claims, evidence = _claims(
            payload["claims"],
            locked_confidence=PASSIVE_CONFIDENCE,
        )
        claim_records = _confidence_records(payload["claims"], PASSIVE_CONFIDENCE)
        evidence_settings = {
            "active_query": False,
            "source": "locked-archived-run",
            "confidence": PASSIVE_CONFIDENCE,
        }
    samples: list[_SampleProblem] = []
    for sample_size in SAMPLE_SIZES:
        sample = payload["sample_artifacts"][str(sample_size)]
        data_path = Path(sample["data_path"])
        if sha256(data_path) != sample["data_sha256"]:
            raise ValueError(f"neuropathic data checksum mismatch: {data_path}")
        data = pd.read_csv(data_path, index_col=0).to_numpy(dtype=np.int64)
        method = sample["methods"]["Claim-Fuse (PC+GES)"]
        warm_starts = tuple(
            _graph(record, node_count) for record in method["graphs"]
        )
        if not warm_starts:
            raise ValueError("neuropathic reference has an empty PC+GES warm-start support")
        baseline_runtime = math.fsum(
            float(sample["fit_diagnostics"][key]["runtime_seconds"])
            + float(sample["completion_diagnostics"][key]["runtime_seconds"])
            for key in ("pc_alpha_0.01", "ges_bdeu_ess1")
        )
        samples.append(
            _SampleProblem(
                sample_size,
                BDeuScore(
                    data,
                    equivalent_sample_size=1.0,
                    cardinalities=(2,) * node_count,
                    max_parents=config.max_parents,
                ),
                warm_starts,
                baseline_runtime,
            )
        )
    return _ReplicateProblem(
        int(payload["data_seed"]),
        reference_path,
        target,
        claims,
        evidence,
        evidence_profile,
        claim_records,
        evidence_settings,
        tuple(samples),
    )


def _load_problem(
    dataset: str,
    results_root: Path,
    reference_run: str,
    task_index: int,
    config: ClaimFuseSMCConfig,
    evidence_profile: str = ARCHIVED_EVIDENCE_PROFILE,
) -> _ReplicateProblem:
    _validate_evidence_profile(dataset, evidence_profile)
    if not 0 <= task_index < REFERENCE_COUNTS[dataset]:
        raise ValueError(
            f"task index must lie in [0, {REFERENCE_COUNTS[dataset] - 1}] for {dataset}"
        )
    if dataset == "synthetic":
        return _load_synthetic_problem(results_root, reference_run, task_index, config)
    return _load_neuropathic_problem(
        results_root, reference_run, task_index, config, evidence_profile
    )


def _result_artifact(
    result: ClaimFuseSMCResult,
    target: Graph,
    warm_start_count: int,
    baseline_runtime_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_metrics = _distribution_metrics(result.data_distribution, target)
    fused_metrics = _distribution_metrics(result.evidence_distribution, target)
    row = {
        "method": "Claim-Fuse-SMC",
        "expected_distance": fused_metrics["expected_distance"],
        "energy_score": fused_metrics["energy_score"],
        "adjacency_f1": fused_metrics["adjacency_f1"],
        "orientation_f1": fused_metrics["orientation_f1"],
        "data_expected_distance": data_metrics["expected_distance"],
        "data_energy_score": data_metrics["energy_score"],
        "data_adjacency_f1": data_metrics["adjacency_f1"],
        "data_orientation_f1": data_metrics["orientation_f1"],
        "delta_expected_distance": fused_metrics["expected_distance"]
        - data_metrics["expected_distance"],
        "delta_energy_score": fused_metrics["energy_score"]
        - data_metrics["energy_score"],
        "delta_adjacency_f1": fused_metrics["adjacency_f1"]
        - data_metrics["adjacency_f1"],
        "delta_orientation_f1": fused_metrics["orientation_f1"]
        - data_metrics["orientation_f1"],
        "initialization_seconds": result.initialization_seconds,
        "smc_seconds": result.smc_seconds,
        "method_runtime_seconds": result.initialization_seconds
        + result.smc_seconds,
        "baseline_runtime_seconds": baseline_runtime_seconds,
    }
    artifact = {
        "score": result.score_metadata,
        "warm_start_count": warm_start_count,
        "baseline_runtime_seconds": baseline_runtime_seconds,
        "phase_runtime_seconds": {
            "initialization": result.initialization_seconds,
            "smc": result.smc_seconds,
            "method": result.initialization_seconds + result.smc_seconds,
        },
        "data_metrics": data_metrics,
        "fused_metrics": fused_metrics,
        "data_truth_probability": result.data_distribution.get(target, 0.0),
        "fused_truth_probability": result.evidence_distribution.get(target, 0.0),
        "data_distribution": _serialize_distribution(result.data_distribution),
        "fused_distribution": _serialize_distribution(result.evidence_distribution),
        "initialization": {
            "acceptance_rate": result.initialization.acceptance_rate,
            "accepted_moves": result.initialization.accepted_moves,
            "attempted_moves": result.initialization.attempted_moves,
            "support_size": len(result.initialization.distribution),
        },
        "smc": {
            "log_normalizer_ratio": result.smc.log_normalizer_ratio,
            "temperatures": result.smc.temperatures,
            "acceptance_rate": result.smc.acceptance_rate,
            "accepted_moves": result.smc.accepted_moves,
            "attempted_moves": result.smc.attempted_moves,
            "self_proposals": result.smc.self_proposals,
            "guided_branch_proposals": result.smc.guided_branch_proposals,
            "repair_fallbacks": result.smc.repair_fallbacks,
            "resampling_count": result.smc.resampling_count,
            "support_size": len(result.smc.distribution),
            "stages": [asdict(stage) for stage in result.smc.stages],
        },
    }
    return row, artifact


def compute(
    dataset: str,
    reference_run: str,
    task_index: int,
    output_run: str,
    results_root: Path,
    *,
    config: ClaimFuseSMCConfig = PRODUCTION_CONFIG,
    profile: str = "production",
    evidence_profile: str = ARCHIVED_EVIDENCE_PROFILE,
) -> Path:
    problem = _load_problem(
        dataset,
        results_root,
        reference_run,
        task_index,
        config,
        evidence_profile,
    )
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    started = time.perf_counter()
    for sample in problem.samples:
        result = claim_fuse_smc(
            sample.score,
            sample.warm_starts,
            problem.claims,
            problem.evidence,
            seed=stable_seed(
                "claim-fuse-smc",
                dataset,
                problem.identifier,
                sample.sample_size,
            ),
            config=config,
        )
        row, artifact = _result_artifact(
            result,
            problem.target,
            len(sample.warm_starts),
            sample.baseline_runtime_seconds,
        )
        row.update(
            {
                "dataset": dataset,
                "replicate": problem.identifier,
                "sample_size": sample.sample_size,
            }
        )
        rows.append(row)
        artifacts[str(sample.sample_size)] = artifact

    output_path = _output_root(
        results_root, output_run, dataset, problem.evidence_profile
    ) / "partials" / f"task_{task_index:03d}.json"
    elapsed = time.perf_counter() - started
    method_runtime = math.fsum(
        float(row["method_runtime_seconds"]) for row in rows
    )
    baseline_runtime = problem.baseline_runtime_seconds
    write_json(
        output_path,
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "dataset": dataset,
            "profile": profile,
            "evidence_profile": problem.evidence_profile,
            "active_query": False,
            "run_id": output_run,
            "reference_run_id": reference_run,
            "reference_path": str(problem.reference_path),
            "reference_sha256": sha256(problem.reference_path),
            "warm_start_source": "pc+ges-completion-support",
            "task_index": task_index,
            "replicate": problem.identifier,
            "claim_count": len(problem.claims),
            "claim_fingerprint": _claim_fingerprint(problem.claim_records),
            "claims": problem.claim_records,
            "evidence_settings": problem.evidence_settings,
            "config": asdict(config),
            "rows": rows,
            "artifacts": artifacts,
            "method_runtime_seconds": method_runtime,
            "baseline_runtime_seconds": baseline_runtime,
            "runtime_ratio": (
                method_runtime / baseline_runtime if baseline_runtime else None
            ),
            "runtime_seconds": elapsed,
            "command": list(sys.argv),
        },
        exclusive=True,
    )
    print(output_path, flush=True)
    return output_path


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    standard_error = (
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    )
    return mean, standard_error


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _runtime_summary_record(
    dataset: str,
    profile: str,
    method_runtimes: Sequence[float],
    baseline_runtimes: Sequence[float],
) -> dict[str, Any]:
    if not method_runtimes:
        raise ValueError("runtime summary requires at least one method runtime")
    mean_method_runtime = statistics.fmean(method_runtimes)
    mean_baseline_runtime = (
        statistics.fmean(baseline_runtimes) if baseline_runtimes else math.nan
    )
    mean_runtime_ratio = (
        mean_method_runtime / mean_baseline_runtime
        if baseline_runtimes and mean_baseline_runtime > 0.0
        else math.nan
    )
    return {
        "dataset": dataset,
        "profile": profile,
        "task_count": len(method_runtimes),
        "baseline_task_count": len(baseline_runtimes),
        "mean_method_runtime_seconds": mean_method_runtime,
        "mean_baseline_runtime_seconds": mean_baseline_runtime,
        "mean_runtime_ratio": mean_runtime_ratio,
        "method_runtime_p95_seconds": _percentile(method_runtimes, 0.95),
        "runtime_gate_limit": 5.0,
        "runtime_gate_passed": bool(
            math.isfinite(mean_runtime_ratio) and mean_runtime_ratio <= 5.0
        ),
    }


def _summary_records(
    rows: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    metric_keys = (
        "expected_distance",
        "energy_score",
        "adjacency_f1",
        "orientation_f1",
        "data_expected_distance",
        "data_energy_score",
        "data_adjacency_f1",
        "data_orientation_f1",
        "delta_expected_distance",
        "delta_energy_score",
        "delta_adjacency_f1",
        "delta_orientation_f1",
    )
    summary: list[dict[str, Any]] = []
    for sample_size in SAMPLE_SIZES:
        group = [row for row in rows if int(row["sample_size"]) == sample_size]
        if len(group) != expected_counts[sample_size]:
            raise ValueError(
                f"sample size {sample_size} has {len(group)} rows, "
                f"expected {expected_counts[sample_size]}"
            )
        record: dict[str, Any] = {
            "sample_size": sample_size,
            "method": "Claim-Fuse-SMC",
            "replicate_count": len(group),
        }
        for key in metric_keys:
            mean, standard_error = _mean_se([float(row[key]) for row in group])
            record[f"{key}_mean"] = mean
            record[f"{key}_se"] = standard_error
        summary.append(record)
    return summary


def _primary_replicates(
    dataset: str,
    results_root: Path,
    reference_run: str,
) -> tuple[dict[int, set[int]], Path]:
    if dataset == "synthetic":
        manifest_path = results_root / reference_run / "test" / "test_manifest.json"
        manifest = read_json(manifest_path)
        replicates = set(int(value) for value in manifest["common_complete_indices"])
        return {sample_size: replicates for sample_size in SAMPLE_SIZES}, manifest_path
    manifest_path = (
        results_root
        / reference_run
        / "neuropathic_benchmark"
        / "panel_a"
        / "run_manifest.json"
    )
    manifest = read_json(manifest_path)
    return (
        {
            sample_size: set(
                int(value)
                for value in manifest["common_complete_seeds"][str(sample_size)]
            )
            for sample_size in SAMPLE_SIZES
        },
        manifest_path,
    )


def aggregate(
    dataset: str,
    reference_run: str,
    output_run: str,
    results_root: Path,
    count: int,
    *,
    task_indices: Sequence[int] | None = None,
    config: ClaimFuseSMCConfig = PRODUCTION_CONFIG,
    profile: str = "production",
    evidence_profile: str = ARCHIVED_EVIDENCE_PROFILE,
) -> Path:
    _validate_evidence_profile(dataset, evidence_profile)
    if not 1 <= count <= REFERENCE_COUNTS[dataset]:
        raise ValueError(
            f"count must lie in [1, {REFERENCE_COUNTS[dataset]}] for {dataset}"
        )
    selected_tasks = (
        tuple(range(count))
        if task_indices is None
        else tuple(int(value) for value in task_indices)
    )
    if not selected_tasks or len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("task indices must be a nonempty unique sequence")
    if any(
        task_index < 0 or task_index >= REFERENCE_COUNTS[dataset]
        for task_index in selected_tasks
    ):
        raise ValueError("task index is outside the reference dataset")
    count = len(selected_tasks)
    partials: list[Mapping[str, Any]] = []
    output_root = _output_root(results_root, output_run, dataset, evidence_profile)
    partial_root = output_root / "partials"
    expected_config = json_value(asdict(config))
    for task_index in selected_tasks:
        partial = read_json(partial_root / f"task_{task_index:03d}.json")
        expected = {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "dataset": dataset,
            "profile": profile,
            "run_id": output_run,
            "reference_run_id": reference_run,
            "task_index": task_index,
            "config": expected_config,
        }
        for key, value in expected.items():
            if partial.get(key) != value:
                raise ValueError(
                    f"partial {task_index} has incompatible {key}: "
                    f"{partial.get(key)!r} != {value!r}"
                )
        if partial.get("evidence_profile", ARCHIVED_EVIDENCE_PROFILE) != evidence_profile:
            raise ValueError(f"partial {task_index} has incompatible evidence profile")
        if bool(partial.get("active_query", False)):
            raise ValueError(f"partial {task_index} unexpectedly used active querying")
        if sha256(Path(partial["reference_path"])) != partial["reference_sha256"]:
            raise ValueError(f"reference artifact changed for partial {task_index}")
        if len(partial["rows"]) != len(SAMPLE_SIZES):
            raise ValueError(f"partial {task_index} has an incomplete sample-size slice")
        partials.append(partial)

    claim_fingerprints = {str(partial.get("claim_fingerprint")) for partial in partials}
    if dataset == "neuropathic" and len(claim_fingerprints) != 1:
        raise ValueError("partials do not use one fixed causal-claim set")
    claim_counts = {int(partial["claim_count"]) for partial in partials}
    if len(claim_counts) != 1:
        raise ValueError("partials use inconsistent causal-claim counts")

    rows = [dict(row) for partial in partials for row in partial["rows"]]
    diagnostics: list[dict[str, Any]] = []
    available_summary = _summary_records(
        rows, {sample_size: count for sample_size in SAMPLE_SIZES}
    )
    full_task_set = selected_tasks == tuple(range(REFERENCE_COUNTS[dataset]))
    if profile == "production" and full_task_set:
        primary_replicates, analysis_manifest_path = _primary_replicates(
            dataset, results_root, reference_run
        )
        primary_rows = [
            row
            for row in rows
            if int(row["replicate"])
            in primary_replicates[int(row["sample_size"])]
        ]
        expected_primary_counts = {
            sample_size: len(primary_replicates[sample_size])
            for sample_size in SAMPLE_SIZES
        }
        analysis_set = "reference_common_complete"
    else:
        primary_rows = rows
        expected_primary_counts = {
            sample_size: count for sample_size in SAMPLE_SIZES
        }
        analysis_manifest_path = None
        if profile == "production":
            analysis_set = "production_pilot_all"
        elif profile.startswith("candidate-"):
            analysis_set = "calibration_all"
        elif profile == "reference":
            analysis_set = "reference_pilot_all"
        else:
            analysis_set = "smoke_all"
    summary = _summary_records(primary_rows, expected_primary_counts)
    for sample_size in SAMPLE_SIZES:
        sample_artifacts = [partial["artifacts"][str(sample_size)] for partial in partials]
        diagnostic: dict[str, Any] = {
            "sample_size": sample_size,
            "replicate_count": len(sample_artifacts),
        }
        for name, path in (
            ("initialization_acceptance_rate", ("initialization", "acceptance_rate")),
            ("initialization_support_size", ("initialization", "support_size")),
            ("smc_acceptance_rate", ("smc", "acceptance_rate")),
            ("smc_support_size", ("smc", "support_size")),
            ("smc_resampling_count", ("smc", "resampling_count")),
            ("initialization_seconds", ("phase_runtime_seconds", "initialization")),
            ("smc_seconds", ("phase_runtime_seconds", "smc")),
            ("method_runtime_seconds", ("phase_runtime_seconds", "method")),
            ("data_truth_probability", ("data_truth_probability",)),
            ("fused_truth_probability", ("fused_truth_probability",)),
        ):
            values: list[float] = []
            for artifact in sample_artifacts:
                value: Any = artifact
                for key in path:
                    value = value[key]
                values.append(float(value))
            mean, standard_error = _mean_se(values)
            diagnostic[f"{name}_mean"] = mean
            diagnostic[f"{name}_se"] = standard_error
        diagnostics.append(diagnostic)

    method_runtimes = [float(partial["method_runtime_seconds"]) for partial in partials]
    baseline_runtimes = [
        float(partial["baseline_runtime_seconds"])
        for partial in partials
        if partial.get("baseline_runtime_seconds") is not None
    ]
    runtime_summary = [
        _runtime_summary_record(
            dataset, profile, method_runtimes, baseline_runtimes
        )
    ]
    runtime_gate_passed = bool(runtime_summary[0]["runtime_gate_passed"])
    mean_runtime_ratio = float(runtime_summary[0]["mean_runtime_ratio"])

    write_csv(output_root / "raw_results.csv", rows)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "available_case_summary.csv", available_summary)
    write_csv(output_root / "diagnostics.csv", diagnostics)
    write_csv(output_root / "runtime_summary.csv", runtime_summary)
    claims_artifact: Any = (
        {
            str(partial["task_index"]): partial.get("claims", ())
            for partial in partials
        }
        if dataset == "synthetic"
        else partials[0].get("claims", ())
    )
    write_json(output_root / "claims.json", claims_artifact)
    write_json(
        output_root / "run_manifest.json",
        {
            "study": STUDY,
            "study_version": STUDY_VERSION,
            "dataset": dataset,
            "profile": profile,
            "evidence_profile": evidence_profile,
            "active_query": False,
            "claim_count": next(iter(claim_counts)),
            "claim_fingerprint": (
                next(iter(claim_fingerprints))
                if len(claim_fingerprints) == 1
                else None
            ),
            "claim_fingerprints": {
                str(partial["task_index"]): partial["claim_fingerprint"]
                for partial in partials
            },
            "evidence_settings": partials[0].get("evidence_settings", {}),
            "run_id": output_run,
            "reference_run_id": reference_run,
            "count": count,
            "task_indices": selected_tasks,
            "config": expected_config,
            "runtime": runtime_summary[0],
            "analysis_set": analysis_set,
            "analysis_manifest_path": (
                str(analysis_manifest_path) if analysis_manifest_path else None
            ),
            "analysis_manifest_sha256": (
                sha256(analysis_manifest_path) if analysis_manifest_path else None
            ),
            "primary_replicates": (
                {
                    str(sample_size): sorted(primary_replicates[sample_size])
                    for sample_size in SAMPLE_SIZES
                }
                if profile == "production" and full_task_set
                else None
            ),
            "partial_reference_sha256": [
                partial["reference_sha256"] for partial in partials
            ],
        },
    )
    if (
        profile == "production"
        and full_task_set
        and not runtime_gate_passed
    ):
        raise RuntimeError(
            f"Claim-Fuse-SMC mean runtime ratio {mean_runtime_ratio:.3f} exceeds 5"
        )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Claim-Fuse-SMC on archived synthetic or neuropathic tasks."
    )
    parser.add_argument("--stage", choices=("compute", "aggregate"), required=True)
    parser.add_argument(
        "--dataset", choices=("synthetic", "neuropathic"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference-run-id")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--task-indices")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="production")
    parser.add_argument(
        "--evidence-profile",
        choices=EVIDENCE_PROFILES,
        default=ARCHIVED_EVIDENCE_PROFILE,
    )
    parser.add_argument(
        "--results-root", type=Path, default=WORKSPACE_ROOT / "results"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_run = run_id(arguments.run_id)
    reference_run = arguments.reference_run_id or REFERENCE_RUNS[arguments.dataset]
    config = PROFILE_CONFIGS[arguments.profile]
    if arguments.stage == "compute":
        if arguments.task_index is None:
            raise ValueError("--task-index is required for compute")
        compute(
            arguments.dataset,
            reference_run,
            arguments.task_index,
            output_run,
            arguments.results_root,
            config=config,
            profile=arguments.profile,
            evidence_profile=arguments.evidence_profile,
        )
    else:
        task_indices = (
            tuple(int(value) for value in arguments.task_indices.split(","))
            if arguments.task_indices
            else None
        )
        aggregate(
            arguments.dataset,
            reference_run,
            output_run,
            arguments.results_root,
            arguments.count or REFERENCE_COUNTS[arguments.dataset],
            task_indices=task_indices,
            config=config,
            profile=arguments.profile,
            evidence_profile=arguments.evidence_profile,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
