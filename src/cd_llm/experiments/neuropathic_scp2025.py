from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from cd_llm.baselines.scp2025 import (
    CachedOpenAIJudge,
    LLMSettings,
    PCBootstrapProbabilities,
    apply_fci_endpoint_constraints,
    bootstrap_endpoint_frequencies,
    bootstrap_pc_edge_probabilities,
    build_decision_prompt,
    build_knowledge_prompt,
    build_pc_background_knowledge,
    ceg_fuse_point_metrics,
    model_supports_logprobs,
    serialize_bootstrap_frequencies,
    serialize_pc_bootstrap_probabilities,
    skeleton_metrics,
    typed_graph_metrics,
)
from cd_llm.core.finite_support import Graph, graph_from_edges
from cd_llm.data.generation.neuropathic import (
    CANONICAL_VARIABLE_NAMES,
    generate_dataset,
    load_model,
    output_stem,
    write_csv,
    write_metadata,
)
from cd_llm.data.loaders.neuropathic import (
    load_metadata,
    load_simulator_edges,
    observed_module_nodes,
    select_ancestrally_closed_module,
    variable_description,
)
from cd_llm.experiments.common.artifacts import run_id, write_json
from cd_llm.methods.fci.graph import ARROW, CIRCLE, TAIL, PAGGraph
from cd_llm.paths import RESOURCE_ROOT, WORKSPACE_ROOT


DEFAULT_DATA_DIR = WORKSPACE_ROOT / "generated" / "neuropathic"
DEFAULT_MODEL_PATH = RESOURCE_ROOT / "neuropathic" / "bnm.pickle"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"
GRAPH_EVALUATION_FIELDS = (
    "pag_shd",
    "pag_shd_norm",
    "typed_correct",
    "typed_precision",
    "typed_recall",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "skeleton_shd",
    "skeleton_false_positive_rate",
    "skeleton_false_negative_rate",
)


@dataclass(frozen=True)
class DiscoveryRun:
    graph: PAGGraph
    elapsed_seconds: float


@dataclass(frozen=True)
class GraphEvaluation:
    pag_shd: int
    pag_shd_norm: float
    typed_correct: int
    typed_precision: float | None
    typed_recall: float | None
    skeleton_precision: float | None
    skeleton_recall: float | None
    skeleton_f1: float
    skeleton_shd: int
    skeleton_false_positive_rate: float | None
    skeleton_false_negative_rate: float | None
    ceg_fuse_metrics: dict[str, int | float | None | str]


def numeric_summary(values: Sequence[int | float | None]) -> dict[str, int | float | None] | None:
    numeric = np.asarray([value for value in values if value is not None], dtype=float)
    if numeric.size == 0:
        return None
    return {
        "count": int(numeric.size),
        "mean": float(np.mean(numeric)),
        "sample_sd": (
            float(np.std(numeric, ddof=1))
            if numeric.size > 1
            else None
        ),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def summarize_evaluation_payloads(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if not evaluations:
        raise ValueError("At least one graph evaluation is required.")
    metrics = {
        name: numeric_summary(
            [
                evaluation.get(name)
                if isinstance(evaluation.get(name), (int, float))
                else None
                for evaluation in evaluations
            ]
        )
        for name in GRAPH_EVALUATION_FIELDS
    }
    ceg_names = sorted(
        {
            name
            for evaluation in evaluations
            for name, value in dict(
                evaluation.get("ceg_fuse_metrics", {})
            ).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    ceg_metrics = {
        name: numeric_summary(
            [
                value
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None
                for evaluation in evaluations
                for value in [
                    dict(evaluation.get("ceg_fuse_metrics", {})).get(name)
                ]
            ]
        )
        for name in ceg_names
    }
    return {
        "replicate_count": len(evaluations),
        "sd_definition": "sample_standard_deviation_ddof_1",
        "metrics": metrics,
        "ceg_fuse_metrics": ceg_metrics,
    }


def summarize_seed_evaluations(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("status") != "completed" or not isinstance(
            record.get("final_evaluation"),
            Mapping,
        ):
            continue
        key = (
            record.get("sample_size"),
            record.get("level"),
            record.get("pattern"),
            record.get("method"),
            record.get("llm_model"),
            record.get("module_size"),
        )
        grouped.setdefault(key, []).append(record)

    summary_sources = {
        "evaluation_summary": "final_evaluation",
        "initial_evaluation_summary": "initial_evaluation",
        "evaluation_summary_true_graph": "final_evaluation_true_graph",
        "initial_evaluation_summary_true_graph": "initial_evaluation_true_graph",
    }
    summaries = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        sample_size, level, pattern, method, llm_model, module_size = key
        group = grouped[key]
        entry: dict[str, object] = {
            "sample_size": sample_size,
            "level": level,
            "pattern": pattern,
            "method": method,
            "llm_model": llm_model,
            "module_size": module_size,
            "seeds": sorted(int(record["seed"]) for record in group),
        }
        for summary_name, record_key in summary_sources.items():
            payloads = [
                record[record_key]
                for record in group
                if isinstance(record.get(record_key), Mapping)
            ]
            if payloads:
                entry[summary_name] = summarize_evaluation_payloads(payloads)
        summaries.append(entry)
    return summaries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an adapted LLM-and-SCD baseline on a deterministic neuropathic "
            "clinical module. Level 0 uses discrete PC; Levels 1-3 use discrete FCI."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--module-size", type=int, default=11)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--patterns", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--forbid-threshold", type=float, default=0.05)
    parser.add_argument("--require-threshold", type=float, default=0.95)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--theme", default="a clinical system")
    parser.add_argument(
        "--dataset-description",
        help="Target-blind description of the observational input shown to the model.",
    )
    parser.add_argument(
        "--llm-base-url",
        help=(
            "OpenAI-compatible endpoint. Defaults to the Hugging Face router "
            "for repo-style model ids and to the OpenAI API otherwise."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Let a Qwen model emit its reasoning block. Off by default: with "
            "thinking on, a decision call can exhaust its token budget before "
            "emitting <yes>/<no>, leaving no answer token to score."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--probability-samples", type=int, default=5)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument(
        "--probability-protocol",
        choices=("paper_logprobs", "sampled_vote"),
        default="paper_logprobs",
        help=(
            "paper_logprobs reproduces Algorithm 1 and requires first-token "
            "logprobs. sampled_vote is a non-paper adaptation and must be "
            "reported as such."
        ),
    )
    parser.add_argument(
        "--acknowledge-neuropathic-adaptation",
        action="store_true",
        help=(
            "Acknowledge that binary G-squared PC/FCI, latent PAG constraints, "
            "and neuropathic metrics are adaptations rather than an exact paper "
            "reproduction."
        ),
    )
    parser.add_argument("--fci-depth", type=int, default=-1)
    parser.add_argument("--fci-max-path-length", type=int, default=-1)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--overwrite-generated", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select the module and print the run plan without loading data or calling the LLM.",
    )
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.levels or set(args.levels) - {0, 1, 2, 3}:
        parser.error("--levels must contain values from 0, 1, 2, 3.")
    if not args.patterns or set(args.patterns) - {0, 2}:
        parser.error("--patterns supports only paper Patterns 0 and 2.")
    if args.module_size < 2:
        parser.error("--module-size must be at least 2.")
    if args.sample_size < 1:
        parser.error("--sample-size must be positive.")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive.")
    if args.probability_samples < 1:
        parser.error("--probability-samples must be positive.")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1).")
    if not 0.0 <= args.forbid_threshold < args.require_threshold <= 1.0:
        parser.error("Require 0 <= forbid-threshold < require-threshold <= 1.")
    if not args.dry_run and not args.acknowledge_neuropathic_adaptation:
        parser.error(
            "This command is a neuropathic adaptation, not an exact reproduction "
            "of Takayama et al. Pass --acknowledge-neuropathic-adaptation to run "
            "it explicitly."
        )
    if (
        args.probability_protocol == "paper_logprobs"
        and not model_supports_logprobs(args.llm_model)
    ):
        parser.error(
            f"{args.llm_model!r} is configured as not supporting token logprobs. "
            "The paper protocol cannot run with this model. Choose a logprob-"
            "capable model, or explicitly select --probability-protocol "
            "sampled_vote and report the methodological deviation."
        )
    if args.probability_protocol == "paper_logprobs":
        if args.probability_samples != 5:
            parser.error("The paper protocol requires --probability-samples 5.")
        if args.top_logprobs != 5:
            parser.error("The paper protocol requires --top-logprobs 5.")
        if not np.isclose(args.temperature, 0.7):
            parser.error("The paper protocol requires --temperature 0.7.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_RESULTS_ROOT / run_id(args.run_id) / "neuropathic_scp2025"
    simulator_edges = load_simulator_edges(args.model_path)
    module_nodes = select_ancestrally_closed_module(simulator_edges, args.module_size)
    plan = build_run_plan(args, module_nodes)
    print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "llm_cache"
    judge = CachedOpenAIJudge(
        cache_dir=cache_dir,
        settings=LLMSettings(
            model=args.llm_model,
            temperature=args.temperature,
            probability_samples=args.probability_samples,
            top_logprobs=args.top_logprobs,
            probability_protocol=args.probability_protocol,
            enable_thinking=args.enable_thinking,
        ),
        base_url=args.llm_base_url,
    )
    if args.generate_missing:
        generate_missing_datasets(args)

    records = []
    for seed in sorted(set(args.seeds)):
        for level in sorted(set(args.levels)):
            observed_nodes = observed_module_nodes(module_nodes, level)
            if len(observed_nodes) < 2:
                records.append(
                    {
                        "sample_size": args.sample_size,
                        "seed": seed,
                        "level": level,
                        "module_nodes": module_nodes,
                        "observed_nodes": observed_nodes,
                        "status": "skipped",
                        "reason": "fewer than two module nodes remain observed",
                    }
                )
                continue
            records.extend(
                evaluate_dataset(
                    args=args,
                    seed=seed,
                    level=level,
                    module_nodes=module_nodes,
                    observed_nodes=observed_nodes,
                    simulator_edges=simulator_edges,
                    judge=judge,
                )
            )

    aggregate_path = args.output_dir / "neuropathic_llm_scd_aggregate.json"
    write_json(
        aggregate_path,
        {
                "config": serializable_config(args),
                "module_nodes": module_nodes,
                "paper_reproduction_profile": paper_reproduction_profile(args),
                "ceg_fuse_metric_scope": {
                    "distribution_type": "point_mass",
                    "level_0_target": "simulator_dag",
                    "latent_level_target": "reference_pag",
                    "latent_level_caveat": (
                        "No complete neuropathic MAG reference is available; "
                        "circle endpoints are scored without coercion."
                    ),
                },
                "true_graph_target_scope": {
                    "level_0": "simulator_dag",
                    "latent_levels": (
                        "true MAG derived from module-DAG ancestry; the "
                        "reference-PAG metrics are retained alongside because "
                        "the PAG is only the data-identifiability ceiling and "
                        "penalizes causally correct LLM endpoint edits"
                    ),
                },
                "variability_scope": {
                    "bootstrap_sd": (
                        "Sample SD across final constrained graphs learned from "
                        "nonparametric resamples of one dataset."
                    ),
                    "seed_sd": (
                        "Sample SD across final full-data graphs from distinct "
                        "dataset seeds for the same level and pattern."
                    ),
                    "single_replicate_sd": None,
                },
                "records": records,
                "seed_evaluation_summaries": summarize_seed_evaluations(records),
        },
    )
    print(f"Wrote {aggregate_path}")
    return 0


def build_run_plan(
    args: argparse.Namespace,
    module_nodes: Sequence[str],
) -> dict[str, object]:
    level_nodes = {
        str(level): observed_module_nodes(module_nodes, level)
        for level in sorted(set(args.levels))
    }
    calls_per_level = {
        level: {
            "ordered_pairs": len(nodes) * (len(nodes) - 1),
            "calls_per_pattern": len(nodes)
            * (len(nodes) - 1)
            * (1 + args.probability_samples),
        }
        for level, nodes in level_nodes.items()
    }
    return {
        "implementation_source": {
            "branch": "origin/luke/neuropathic-baselines",
            "commit": "e388190",
        },
        "config": serializable_config(args),
        "paper_reproduction_profile": paper_reproduction_profile(args),
        "module_nodes": list(module_nodes),
        "observed_nodes_by_level": level_nodes,
        "estimated_llm_calls_by_level": calls_per_level,
    }


def serializable_config(args: argparse.Namespace) -> dict[str, object]:
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def paper_reproduction_profile(args: argparse.Namespace) -> dict[str, object]:
    probability_description = (
        "M=5 mean of first-token yes probabilities from top_logprobs=5"
        if args.probability_protocol == "paper_logprobs"
        else "M=5 empirical yes/no vote frequency; non-paper adaptation"
    )
    adaptations = [
        {
            "component": "data",
            "paper": "fully standardized continuous, fully observed data",
            "actual": "binary neuropathic Bayesian-network samples",
        },
        {
            "component": "discovery",
            "paper": "Fisher-Z PC with no hidden confounders",
            "actual": "G-squared PC at level 0 and G-squared FCI at levels 1-3",
        },
        {
            "component": "pattern2_latent_levels",
            "paper": "not defined; Pattern 2 uses PC directed/undirected probabilities",
            "actual": "FCI PAG endpoint-state probabilities at levels 1-3",
        },
        {
            "component": "latent_constraints",
            "paper": "not defined",
            "actual": "post-hoc endpoint edits on existing FCI PAG edges",
        },
        {
            "component": "pc_background_knowledge",
            "paper": (
                "independent per-direction thresholding; can assert "
                "contradictory required edges"
            ),
            "actual": (
                "pairs whose two directions both exceed the require threshold "
                "are skipped and recorded"
            ),
        },
        {
            "component": "evaluation",
            "paper": "adjacency SHD/FPR/FNR/precision/F1 plus SEM fit metrics",
            "actual": (
                "adjacency metrics plus typed PAG and CEG-FUSE point metrics; "
                "SEM fit metrics omitted for binary data"
            ),
        },
    ]
    if args.llm_model != "gpt-4-1106-preview":
        adaptations.append(
            {
                "component": "llm_model",
                "paper": "gpt-4-1106-preview",
                "actual": args.llm_model,
            }
        )
    if args.probability_protocol == "sampled_vote":
        adaptations.append(
            {
                "component": "probability_protocol",
                "paper": "M=5 mean first-token yes logprob",
                "actual": "M=5 empirical yes/no vote frequency",
            }
        )
    return {
        "claim": "neuropathic adaptation; not an exact paper reproduction",
        "paper_reference": "Takayama et al., arXiv:2402.01454v5",
        "paper_main_model": "gpt-4-1106-preview",
        "model_used": args.llm_model,
        "probability_protocol": args.probability_protocol,
        "probability_description": probability_description,
        "adaptations": adaptations,
    }


def generate_missing_datasets(args: argparse.Namespace) -> None:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    model = None
    for seed in sorted(set(args.seeds)):
        for level in sorted(set(args.levels)):
            stem = output_stem(args.sample_size, level, seed)
            csv_path = args.data_dir / f"{stem}.csv"
            metadata_path = args.data_dir / f"{stem}.json"
            if csv_path.exists() and metadata_path.exists() and not args.overwrite_generated:
                continue
            if model is None:
                model = load_model(args.model_path)
            data, observed_indices = generate_dataset(model, args.sample_size, level, seed)
            write_csv(csv_path, data, observed_indices)
            write_metadata(
                metadata_path,
                sample_size=args.sample_size,
                level=level,
                seed=seed,
                observed_indices=observed_indices,
            )
            print(f"Wrote {csv_path} and {metadata_path}")


def evaluate_dataset(
    *,
    args: argparse.Namespace,
    seed: int,
    level: int,
    module_nodes: Sequence[str],
    observed_nodes: Sequence[str],
    simulator_edges: Sequence[Tuple[str, str]],
    judge: CachedOpenAIJudge,
) -> list[dict[str, object]]:
    stem = output_stem(args.sample_size, level, seed)
    metadata_path = args.data_dir / f"{stem}.json"
    data_path = args.data_dir / f"{stem}.csv"
    if not metadata_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path} or {metadata_path}. Pass --generate-missing."
        )
    metadata = load_metadata(metadata_path)
    frame = load_module_frame(data_path, metadata, observed_nodes)
    method = (
        "PC (Peter-Clark) algorithm with G-squared independence tests"
        if level == 0
        else "FCI with G-squared independence tests (neuropathic adaptation)"
    )
    target = build_target_graph(
        module_nodes=module_nodes,
        observed_nodes=observed_nodes,
        simulator_edges=simulator_edges,
        level=level,
    )
    ceg_truth = (
        build_ceg_dag_truth(observed_nodes, simulator_edges)
        if level == 0
        else None
    )
    ceg_target_semantics = "simulator_dag" if level == 0 else "reference_pag"
    # The PAG/CPDAG target above is the identifiability ceiling of a data-only
    # method: every endpoint the data cannot decide is a circle, so scoring an
    # LLM edit against it penalizes informative marks even when they are
    # causally correct. Score against the true graph as well (simulator DAG at
    # level 0, true MAG under latent confounding) so knowledge injection can
    # actually win where it is right.
    true_target = build_true_graph_target(
        module_nodes=module_nodes,
        observed_nodes=observed_nodes,
        simulator_edges=simulator_edges,
        level=level,
        reference_pag=target,
    )
    true_target_semantics = "simulator_dag" if level == 0 else "true_mag"

    initial = run_discovery(
        frame=frame,
        level=level,
        alpha=args.alpha,
        fci_depth=args.fci_depth,
        fci_max_path_length=args.fci_max_path_length,
        show_progress=args.show_progress,
    )
    regime = "dag" if level == 0 else "mag"
    initial_evaluation = evaluate_graph(
        initial.graph,
        target,
        regime=regime,
        ceg_truth=ceg_truth,
        ceg_target_semantics=ceg_target_semantics,
    )
    initial_true_evaluation = evaluate_graph(
        initial.graph,
        true_target,
        regime=regime,
        ceg_truth=ceg_truth,
        ceg_target_semantics=true_target_semantics,
    )
    initial_graph_path = (
        args.output_dir / f"{stem}_module{args.module_size}_initial_graph.json"
    )
    target_graph_path = (
        args.output_dir / f"{stem}_module{args.module_size}_target_graph.json"
    )
    true_target_graph_path = (
        args.output_dir / f"{stem}_module{args.module_size}_true_target_graph.json"
    )
    write_graph(initial_graph_path, initial.graph)
    write_graph(target_graph_path, target)
    write_graph(true_target_graph_path, true_target)

    bootstrap_graphs = run_bootstrap(
        frame=frame,
        level=level,
        samples=args.bootstrap_samples,
        seed=seed,
        alpha=args.alpha,
        fci_depth=args.fci_depth,
        fci_max_path_length=args.fci_max_path_length,
        show_progress=False,
    )
    endpoint_frequencies = bootstrap_endpoint_frequencies(
        bootstrap_graphs,
        observed_nodes,
    )
    pc_bootstrap_probabilities = (
        bootstrap_pc_edge_probabilities(bootstrap_graphs, observed_nodes)
        if level == 0
        else None
    )
    records = []
    for pattern in sorted(set(args.patterns)):
        output_path = args.output_dir / (
            f"{stem}_module{args.module_size}_pattern{pattern}_summary.json"
        )
        if output_path.exists() and not args.overwrite:
            records.append(json.loads(output_path.read_text()))
            continue

        probability_records, probabilities = collect_llm_probabilities(
            judge=judge,
            pattern=pattern,
            observed_nodes=observed_nodes,
            method=method,
            # Domain context only: no benchmark meta-information (hidden-
            # confounding level) that a clinician would not have.
            dataset_description=(
                args.dataset_description
                or f"binary clinical observations ({args.sample_size} records)"
            ),
            endpoint_frequencies=endpoint_frequencies,
            pc_bootstrap_probabilities=pc_bootstrap_probabilities,
            theme=args.theme,
        )
        if level == 0:
            background_knowledge, applications = build_pc_background_knowledge(
                observed_nodes,
                probabilities,
                forbid_threshold=args.forbid_threshold,
                require_threshold=args.require_threshold,
            )
            final = run_discovery(
                frame=frame,
                level=level,
                alpha=args.alpha,
                fci_depth=args.fci_depth,
                fci_max_path_length=args.fci_max_path_length,
                show_progress=args.show_progress,
                background_knowledge=background_knowledge,
            )
            final_bootstrap_graphs = run_bootstrap(
                frame=frame,
                level=level,
                samples=args.bootstrap_samples,
                seed=seed,
                alpha=args.alpha,
                fci_depth=args.fci_depth,
                fci_max_path_length=args.fci_max_path_length,
                show_progress=False,
                background_knowledge=background_knowledge,
            )
            bootstrap_evaluation_protocol = (
                "rerun_pc_on_each_resample_with_fixed_llm_background_knowledge"
            )
        else:
            constrained, applications = apply_fci_endpoint_constraints(
                initial.graph,
                probabilities,
                forbid_threshold=args.forbid_threshold,
                require_threshold=args.require_threshold,
            )
            final = DiscoveryRun(graph=constrained, elapsed_seconds=0.0)
            final_bootstrap_graphs = [
                apply_fci_endpoint_constraints(
                    bootstrap_graph,
                    probabilities,
                    forbid_threshold=args.forbid_threshold,
                    require_threshold=args.require_threshold,
                )[0]
                for bootstrap_graph in bootstrap_graphs
            ]
            bootstrap_evaluation_protocol = (
                "apply_fixed_llm_endpoint_constraints_to_each_bootstrap_fci_graph"
            )

        final_evaluation = evaluate_graph(
            final.graph,
            target,
            regime=regime,
            ceg_truth=ceg_truth,
            ceg_target_semantics=ceg_target_semantics,
        )
        final_true_evaluation = evaluate_graph(
            final.graph,
            true_target,
            regime=regime,
            ceg_truth=ceg_truth,
            ceg_target_semantics=true_target_semantics,
        )
        bootstrap_evaluations = [
            asdict(
                evaluate_graph(
                    bootstrap_graph,
                    target,
                    regime=regime,
                    ceg_truth=ceg_truth,
                    ceg_target_semantics=ceg_target_semantics,
                )
            )
            for bootstrap_graph in final_bootstrap_graphs
        ]
        bootstrap_evaluation_summary = summarize_evaluation_payloads(
            bootstrap_evaluations
        )
        bootstrap_evaluation_summary["protocol"] = bootstrap_evaluation_protocol
        graph_path = args.output_dir / (
            f"{stem}_module{args.module_size}_pattern{pattern}_final_graph.json"
        )
        write_graph(graph_path, final.graph)
        record = {
            "dataset": metadata_path.name,
            "sample_size": args.sample_size,
            "seed": seed,
            "level": level,
            "method": method,
            "protocol": "llm_scd_neuropathic_adaptation",
            "probability_protocol": args.probability_protocol,
            "pattern": pattern,
            "module_size": args.module_size,
            "module_nodes": list(module_nodes),
            "observed_nodes": list(observed_nodes),
            "hidden_module_nodes": [
                node for node in module_nodes if node not in set(observed_nodes)
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "llm_model": args.llm_model,
            "initial_evaluation": asdict(initial_evaluation),
            "final_evaluation": asdict(final_evaluation),
            "initial_evaluation_true_graph": asdict(initial_true_evaluation),
            "final_evaluation_true_graph": asdict(final_true_evaluation),
            "true_graph_target_semantics": true_target_semantics,
            "bootstrap_evaluation_summary": bootstrap_evaluation_summary,
            "initial_discovery_seconds": initial.elapsed_seconds,
            "final_discovery_seconds": final.elapsed_seconds,
            "llm_probabilities": probability_records,
            "constraint_applications": [asdict(item) for item in applications],
            "bootstrap_endpoint_frequencies": serialize_bootstrap_frequencies(
                endpoint_frequencies
            ),
            "paper_pattern2_pc_bootstrap_probabilities": (
                serialize_pc_bootstrap_probabilities(pc_bootstrap_probabilities)
                if pc_bootstrap_probabilities is not None
                else None
            ),
            "constraint_protocol": (
                "pc_background_knowledge_rerun"
                if level == 0
                else "fci_post_hoc_endpoint_edit_no_discovery_rerun"
            ),
            "paper_reproduction_profile": paper_reproduction_profile(args),
            "initial_graph_path": str(initial_graph_path),
            "target_graph_path": str(target_graph_path),
            "true_graph_target_path": str(true_target_graph_path),
            "final_graph_path": str(graph_path),
            "status": "completed",
        }
        write_json(output_path, record)
        records.append(record)
    return records


def load_module_frame(
    data_path: Path,
    metadata: Mapping[str, Any],
    observed_nodes: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(data_path, index_col=0)
    observed_indices = [int(index) for index in metadata["observed_indices"]]
    name_by_column = {
        str(index): CANONICAL_VARIABLE_NAMES[index] for index in observed_indices
    }
    frame.columns = [str(column) for column in frame.columns]
    missing_columns = [
        str(CANONICAL_VARIABLE_NAMES.index(name))
        for name in observed_nodes
        if str(CANONICAL_VARIABLE_NAMES.index(name)) not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing module columns: {missing_columns}")
    selected_columns = [
        str(CANONICAL_VARIABLE_NAMES.index(name)) for name in observed_nodes
    ]
    selected = frame.loc[:, selected_columns].rename(columns=name_by_column)
    return selected.astype(int)


def run_discovery(
    *,
    frame: pd.DataFrame,
    level: int,
    alpha: float,
    fci_depth: int,
    fci_max_path_length: int,
    show_progress: bool,
    background_knowledge=None,
) -> DiscoveryRun:
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as error:
        raise RuntimeError("causal-learn is required for this baseline.") from error

    data = frame.to_numpy(dtype=int)
    nodes = list(frame.columns)
    start = time.monotonic()
    if level == 0:
        causal_graph = pc(
            data,
            alpha=alpha,
            indep_test="gsq",
            stable=True,
            background_knowledge=background_knowledge,
            verbose=False,
            show_progress=show_progress,
            node_names=nodes,
        )
        graph = causal_learn_graph_to_pag(causal_graph.G, nodes)
    else:
        causal_graph, _edges = fci(
            data,
            independence_test_method="gsq",
            alpha=alpha,
            depth=fci_depth,
            max_path_length=fci_max_path_length,
            verbose=False,
            background_knowledge=background_knowledge,
            show_progress=show_progress,
            node_names=nodes,
        )
        graph = causal_learn_graph_to_pag(causal_graph, nodes)
    return DiscoveryRun(graph=graph, elapsed_seconds=time.monotonic() - start)


def run_bootstrap(
    *,
    frame: pd.DataFrame,
    level: int,
    samples: int,
    seed: int,
    alpha: float,
    fci_depth: int,
    fci_max_path_length: int,
    show_progress: bool,
    background_knowledge=None,
) -> list[PAGGraph]:
    rng = np.random.default_rng(seed)
    graphs = []
    attempts = 0
    max_attempts = max(samples * 3, samples + 10)
    while len(graphs) < samples and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, frame.shape[0], size=frame.shape[0])
        sample = frame.iloc[indices].reset_index(drop=True)
        try:
            result = run_discovery(
                frame=sample,
                level=level,
                alpha=alpha,
                fci_depth=fci_depth,
                fci_max_path_length=fci_max_path_length,
                show_progress=show_progress,
                background_knowledge=background_knowledge,
            )
        except (ValueError, FloatingPointError):
            continue
        graphs.append(result.graph)
    if len(graphs) != samples:
        raise RuntimeError(
            f"Only {len(graphs)} of {samples} bootstrap graphs succeeded "
            f"after {attempts} attempts."
        )
    return graphs


def collect_llm_probabilities(
    *,
    judge: CachedOpenAIJudge,
    pattern: int,
    observed_nodes: Sequence[str],
    method: str,
    dataset_description: str,
    endpoint_frequencies,
    pc_bootstrap_probabilities: PCBootstrapProbabilities | None,
    variable_descriptions: Mapping[str, str] | None = None,
    theme: str = "a clinical system",
) -> tuple[list[dict[str, object]], dict[Tuple[str, str], float]]:
    records = []
    probabilities: dict[Tuple[str, str], float] = {}
    for source in observed_nodes:
        for target in observed_nodes:
            if source == target:
                continue
            knowledge_prompt = build_knowledge_prompt(
                pattern=pattern,
                source=source,
                target=target,
                nodes=observed_nodes,
                method=method,
                dataset_description=dataset_description,
                bootstrap_frequencies=(
                    endpoint_frequencies
                    if pattern == 2 and pc_bootstrap_probabilities is None
                    else None
                ),
                pc_bootstrap_probabilities=(
                    pc_bootstrap_probabilities
                    if pattern == 2
                    else None
                ),
                theme=theme,
                variable_descriptions=(
                    dict(variable_descriptions)
                    if variable_descriptions is not None
                    else {node: variable_description(node) for node in observed_nodes}
                ),
            )
            explanation = judge.explanation(knowledge_prompt)
            decision_prompt = build_decision_prompt(
                knowledge_prompt=knowledge_prompt,
                explanation=explanation,
                source=source,
                target=target,
            )
            mean_probability, trials = judge.yes_probability(decision_prompt)
            probabilities[(source, target)] = mean_probability
            records.append(
                {
                    "source": source,
                    "target": target,
                    "mean_yes_probability": mean_probability,
                    "trial_probabilities": trials,
                    "explanation": explanation,
                }
            )
            print(
                f"pattern={pattern} {source} -> {target}: "
                f"p_yes={mean_probability:.4f}",
                flush=True,
            )
    return records, probabilities


def build_target_graph(
    *,
    module_nodes: Sequence[str],
    observed_nodes: Sequence[str],
    simulator_edges: Sequence[Tuple[str, str]],
    level: int,
) -> PAGGraph:
    try:
        from causallearn.graph.Dag import Dag
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.DAG2CPDAG import dag2cpdag
        from causallearn.utils.DAG2PAG import dag2pag
    except ImportError as error:
        raise RuntimeError("causal-learn is required to build graph targets.") from error

    graph_nodes = [GraphNode(name) for name in module_nodes]
    node_by_name = {node.get_name(): node for node in graph_nodes}
    dag = Dag(graph_nodes)
    module_set = set(module_nodes)
    for parent, child in simulator_edges:
        if parent in module_set and child in module_set:
            dag.add_directed_edge(node_by_name[parent], node_by_name[child])

    if level == 0:
        target = dag2cpdag(dag)
    else:
        observed_set = set(observed_nodes)
        latent_nodes = [
            node_by_name[name] for name in module_nodes if name not in observed_set
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            target = dag2pag(dag, islatent=latent_nodes)
    return causal_learn_graph_to_pag(target, observed_nodes)


def build_true_graph_target(
    *,
    module_nodes: Sequence[str],
    observed_nodes: Sequence[str],
    simulator_edges: Sequence[Tuple[str, str]],
    level: int,
    reference_pag: PAGGraph,
) -> PAGGraph:
    """Build the true graph over observed nodes: the simulator DAG at level 0,
    the true MAG at latent levels.

    The MAG shares its skeleton with the reference PAG, so adjacencies are
    taken from it; endpoints come from ancestry in the module DAG: X --> Y when
    X is an ancestor of Y (possibly through latent nodes), X <-> Y when
    neither is an ancestor of the other.
    """
    if level == 0:
        observed = set(observed_nodes)
        graph = PAGGraph(tuple(observed_nodes))
        for parent, child in simulator_edges:
            if parent in observed and child in observed:
                graph.add_edge(parent, child, TAIL, ARROW)
        return graph

    module_set = set(module_nodes)
    children: dict[str, set[str]] = {}
    for parent, child in simulator_edges:
        if parent in module_set and child in module_set:
            children.setdefault(parent, set()).add(child)

    def is_ancestor(source: str, target: str) -> bool:
        stack = list(children.get(source, ()))
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(children.get(current, ()))
        return False

    graph = PAGGraph(tuple(observed_nodes))
    for left, right in reference_pag.all_edges():
        if is_ancestor(left, right):
            graph.add_edge(left, right, TAIL, ARROW)
        elif is_ancestor(right, left):
            graph.add_edge(right, left, TAIL, ARROW)
        else:
            graph.add_edge(left, right, ARROW, ARROW)
    return graph


def build_ceg_dag_truth(
    observed_nodes: Sequence[str],
    simulator_edges: Sequence[Tuple[str, str]],
) -> Graph:
    position = {node: index for index, node in enumerate(observed_nodes)}
    directed_edges = [
        (position[parent], position[child])
        for parent, child in simulator_edges
        if parent in position and child in position
    ]
    return graph_from_edges(len(observed_nodes), directed=directed_edges)


def evaluate_graph(
    predicted: PAGGraph,
    target: PAGGraph,
    *,
    regime: Literal["dag", "mag"],
    ceg_truth: Graph | None = None,
    ceg_target_semantics: str = "reference_graph",
) -> GraphEvaluation:
    pag_metrics = typed_graph_metrics(predicted, target)
    adjacency_metrics = skeleton_metrics(predicted, target)
    ceg_metrics = ceg_fuse_point_metrics(
        predicted,
        target,
        regime=regime,
        truth_graph=ceg_truth,
    )
    ceg_metrics["target_semantics"] = ceg_target_semantics
    return GraphEvaluation(
        pag_shd=pag_metrics.shd,
        pag_shd_norm=pag_metrics.shd_norm,
        typed_correct=pag_metrics.typed_correct,
        typed_precision=pag_metrics.typed_precision,
        typed_recall=pag_metrics.typed_recall,
        skeleton_precision=adjacency_metrics.precision,
        skeleton_recall=adjacency_metrics.recall,
        skeleton_f1=adjacency_metrics.f1,
        skeleton_shd=adjacency_metrics.shd,
        skeleton_false_positive_rate=adjacency_metrics.false_positive_rate,
        skeleton_false_negative_rate=adjacency_metrics.false_negative_rate,
        ceg_fuse_metrics=ceg_metrics,
    )


def write_graph(path: Path, graph: PAGGraph) -> None:
    write_json(
        path,
        {
            "nodes": list(graph.nodes),
            "edges": pag_to_records(graph),
            "edge_strings": graph.edge_strings(),
        },
    )


def causal_learn_graph_to_pag(graph: Any, observed_nodes: Sequence[str]) -> PAGGraph:
    endpoint_by_name = {
        "TAIL": TAIL,
        "ARROW": ARROW,
        "CIRCLE": CIRCLE,
    }
    pag = PAGGraph(tuple(observed_nodes))
    observed = set(observed_nodes)
    for edge in graph.get_graph_edges():
        left = str(edge.get_node1().get_name())
        right = str(edge.get_node2().get_name())
        if left not in observed or right not in observed:
            continue
        left_endpoint = endpoint_by_name.get(str(edge.get_endpoint1()))
        right_endpoint = endpoint_by_name.get(str(edge.get_endpoint2()))
        if left_endpoint is None or right_endpoint is None:
            raise ValueError(f"Unsupported causal-learn edge endpoint: {edge}")
        pag.add_edge(left, right, left_endpoint, right_endpoint)
    return pag


def pag_to_records(graph: PAGGraph) -> list[dict[str, str]]:
    records = []
    for left, right in graph.all_edges():
        left_mark, right_mark = graph.endpoints(left, right)
        records.append(
            {
                "left": left,
                "right": right,
                "left_mark": str(left_mark),
                "right_mark": str(right_mark),
            }
        )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
