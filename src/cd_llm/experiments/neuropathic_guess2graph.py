from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_MODULE_SIZE = 11

from cd_llm.baselines.guess2graph2026 import (
    CachedOpenAISkeletonExpert,
    LLMExpertSettings,
    run_gpc_guess_skeleton,
    run_pc_guess_skeleton,
)
from cd_llm.core.finite_support import Endpoint, PartialGraph, graph_from_edges, point_prediction_metrics
from cd_llm.data.generation.neuropathic import CANONICAL_VARIABLE_NAMES
from cd_llm.data.loaders.neuropathic import (
    load_simulator_edges,
    select_ancestrally_closed_module,
)
from cd_llm.experiments.common.artifacts import run_id, write_json
from cd_llm.paths import RESOURCE_ROOT, WORKSPACE_ROOT


DEFAULT_DATA_PATH = (
    WORKSPACE_ROOT / "generated" / "neuropathic" / "neuropathic_n1000_level0_seed0.csv"
)
DEFAULT_MODEL_PATH = RESOURCE_ROOT / "neuropathic" / "bnm.pickle"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results"
METHODS = ("pc_guess", "gpc_guess")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the vendored Guess2Graph PC-Guess and gPC-Guess methods on the "
            "level-0 neuropathic pain benchmark and score their skeletons."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--module-size", type=int, default=DEFAULT_MODULE_SIZE)
    parser.add_argument(
        "--expert-source",
        choices=("llm", "random", "file"),
        default="llm",
        help=(
            "Source of the per-trial skeleton guess. Default: llm "
            "(Qwen3.5-9B, matching the project's configured LLM)."
        ),
    )
    parser.add_argument(
        "--guess-path",
        type=Path,
        help=(
            "Expert graph as .npy, .csv, or JSON for --expert-source file. Directed "
            "inputs are converted to skeletons. Providing this option implies file."
        ),
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Use all 222 variables instead of the default 11-node module.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["pc_guess", "gpc_guess"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--guess-seed", type=int, default=0)
    parser.add_argument("--random-guess-edge-probability", type=float, default=0.5)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B:together")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--ci-test",
        default="chisq",
        help="causal-learn conditional-independence test (default: chisq for binary data).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--allow-expensive-full-gpc",
        action="store_true",
        help=(
            "Acknowledge that full-graph gPC-Guess has exponential worst-case "
            "conditioning-set enumeration."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def canonical_variable_names() -> tuple[str, ...]:
    return tuple(CANONICAL_VARIABLE_NAMES)


def selected_indices(
    full_graph: bool,
    variable_count: int,
    *,
    module_size: int = DEFAULT_MODULE_SIZE,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[int, ...]:
    if full_graph:
        return tuple(range(variable_count))
    names = canonical_variable_names()
    edges = load_simulator_edges(model_path)
    selected_names = select_ancestrally_closed_module(edges, module_size)
    index_by_name = {name: index for index, name in enumerate(names)}
    indices = tuple(index_by_name[name] for name in selected_names)
    if any(index >= variable_count for index in indices):
        raise ValueError(
            f"Data contain {variable_count} variables but the selected module requires "
            f"canonical indices up to {max(indices)}."
        )
    return indices


def load_observations(
    path: Path,
    indices: Sequence[int],
) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    column_by_index: dict[int, object] = {}
    for column in frame.columns:
        try:
            column_by_index[int(str(column))] = column
        except ValueError as error:
            raise ValueError(
                f"Observation column {column!r} is not a canonical integer index."
            ) from error

    missing = [index for index in indices if index not in column_by_index]
    if missing:
        raise ValueError(f"Observation data are missing selected columns: {missing}")

    selected = frame.loc[:, [column_by_index[index] for index in indices]]
    if selected.isna().any().any():
        raise ValueError("Guess2Graph input contains missing values.")
    data = selected.to_numpy()
    values = set(np.unique(data).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Neuropathic input must be binary; found values {sorted(values)}")
    return data.astype(int)


def target_skeleton_from_dag(
    edges: Sequence[tuple[str, str]],
    indices: Sequence[int],
    names: Sequence[str],
) -> np.ndarray:
    selected_names = [names[index] for index in indices]
    position = {name: index for index, name in enumerate(selected_names)}
    skeleton = np.zeros((len(indices), len(indices)), dtype=int)
    for parent, child in edges:
        if parent not in position or child not in position:
            continue
        parent_index = position[parent]
        child_index = position[child]
        skeleton[parent_index, child_index] = 1
        skeleton[child_index, parent_index] = 1
    return skeleton


def target_ceg_graph_from_dag(
    edges: Sequence[tuple[str, str]],
    indices: Sequence[int],
    names: Sequence[str],
):
    selected_names = [names[index] for index in indices]
    position = {name: index for index, name in enumerate(selected_names)}
    directed_edges = [
        (position[parent], position[child])
        for parent, child in edges
        if parent in position and child in position
    ]
    return graph_from_edges(len(indices), directed=directed_edges)


def random_dag(node_count: int, edge_probability: float, seed: int) -> np.ndarray:
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("--random-guess-edge-probability must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(node_count)
    dag = np.zeros((node_count, node_count), dtype=int)
    for earlier in range(node_count):
        for later in range(earlier + 1, node_count):
            if rng.random() < edge_probability:
                dag[order[earlier], order[later]] = 1
    return dag


def load_guess_payload(path: Path) -> tuple[np.ndarray, list[str] | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False)), None
    if suffix == ".csv":
        candidates = (
            pd.read_csv(path, index_col=0),
            pd.read_csv(path),
            pd.read_csv(path, header=None),
        )
        square_matrices = []
        for frame in candidates:
            if frame.shape[0] != frame.shape[1]:
                continue
            try:
                matrix = frame.to_numpy(dtype=float)
            except (TypeError, ValueError):
                continue
            if np.isfinite(matrix).all():
                square_matrices.append(matrix)
        if square_matrices:
            return max(square_matrices, key=lambda matrix: matrix.shape[0]), None
        shapes = [frame.shape for frame in candidates]
        raise ValueError(f"Guess CSV must contain a numeric square matrix; parsed shapes {shapes}.")
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return np.asarray(payload), None
        if not isinstance(payload, Mapping):
            raise ValueError("Guess JSON must be a matrix or an object containing a matrix.")
        matrix = None
        for key in ("adjacency_matrix", "guessed_dag", "matrix"):
            if key in payload:
                matrix = payload[key]
                break
        if matrix is None:
            raise ValueError(
                "Guess JSON must contain 'adjacency_matrix', 'guessed_dag', or 'matrix'."
            )
        nodes = payload.get("nodes", payload.get("variable_names"))
        if nodes is not None and not isinstance(nodes, list):
            raise ValueError("Guess JSON node names must be a list.")
        return np.asarray(matrix), [str(node) for node in nodes] if nodes else None
    raise ValueError(f"Unsupported guess format {suffix!r}; use .npy, .csv, or .json.")


def align_guess(
    matrix: np.ndarray,
    payload_nodes: Sequence[str] | None,
    indices: Sequence[int],
    names: Sequence[str],
    full_variable_count: int,
) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expert guess must be square; got shape {matrix.shape}")

    selected_names = [names[index] for index in indices]
    if payload_nodes is not None:
        if len(payload_nodes) != matrix.shape[0] or len(set(payload_nodes)) != len(payload_nodes):
            raise ValueError("Expert guess node names must be unique and match the matrix size.")
        position = {name: offset for offset, name in enumerate(payload_nodes)}
        if all(name in position for name in selected_names):
            offsets = [position[name] for name in selected_names]
            matrix = matrix[np.ix_(offsets, offsets)]
        elif matrix.shape[0] == len(indices):
            raise ValueError(
                "Named expert guess does not contain every selected canonical variable."
            )
        else:
            raise ValueError("Named expert guess cannot be aligned to the selected module.")
    elif matrix.shape == (full_variable_count, full_variable_count):
        matrix = matrix[np.ix_(indices, indices)]
    elif matrix.shape != (len(indices), len(indices)):
        raise ValueError(
            "Expert guess shape must match either the selected module "
            f"({len(indices)}, {len(indices)}) or full graph "
            f"({full_variable_count}, {full_variable_count}); got {matrix.shape}."
        )

    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError("Expert graph must contain only 0 and 1.")
    graph = matrix.astype(int)
    if np.any(np.diag(graph)):
        raise ValueError("Expert graph must not contain self-loops.")
    return graph


def is_acyclic(dag: np.ndarray) -> bool:
    in_degree = dag.sum(axis=0).astype(int)
    queue = [index for index, degree in enumerate(in_degree) if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(dag[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(int(child))
    return visited == dag.shape[0]


def normalize_skeleton(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Predicted skeleton must be square; got shape {matrix.shape}")
    skeleton = ((matrix != 0) | (matrix.T != 0)).astype(int)
    np.fill_diagonal(skeleton, 0)
    return skeleton


def skeleton_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if predicted.shape != target.shape:
        raise ValueError(
            f"Predicted and target skeleton shapes differ: {predicted.shape} vs {target.shape}"
        )
    upper = np.triu_indices(predicted.shape[0], k=1)
    predicted_edges = predicted[upper].astype(bool)
    target_edges = target[upper].astype(bool)
    true_positive = int(np.sum(predicted_edges & target_edges))
    false_positive = int(np.sum(predicted_edges & ~target_edges))
    false_negative = int(np.sum(~predicted_edges & target_edges))
    true_negative = int(np.sum(~predicted_edges & ~target_edges))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    f1 = (
        2 * true_positive / (2 * true_positive + false_positive + false_negative)
        if true_positive or false_positive or false_negative
        else 1.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skeleton_shd": false_positive + false_negative,
    }


def ceg_fuse_skeleton_metrics(
    predicted: np.ndarray,
    target,
) -> dict[str, int | float | None | str]:
    """Score an undirected prediction against the directed simulator DAG."""
    matrix = np.zeros_like(predicted, dtype=int)
    matrix[predicted != 0] = int(Endpoint.TAIL)
    estimate = PartialGraph.from_matrix(matrix, "cpdag")
    metrics = point_prediction_metrics(estimate, target, normalized=False)
    raw_distance = int(round(metrics["expected_distance"] * 2))
    metrics.update(
        {
            "structural_hamming_distance": raw_distance,
            "entropy": 0.0,
            "high_mass_set_size": 1,
            "high_mass_truth_coverage": float(raw_distance == 0),
            "neighborhood_mass": float(raw_distance == 0),
            "mode_distance": metrics["expected_distance"],
            "endpoint_f1": None,
            "distribution_type": "point_mass",
        }
    )
    metrics["graph_semantics"] = "undirected_skeleton_prediction"
    metrics["target_semantics"] = "simulator_dag"
    return metrics


def matrix_edges(
    skeleton: np.ndarray,
    indices: Sequence[int],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    edges = []
    for left in range(skeleton.shape[0]):
        for right in range(left + 1, skeleton.shape[0]):
            if skeleton[left, right]:
                edges.append(
                    {
                        "left_index": int(indices[left]),
                        "left": names[indices[left]],
                        "right_index": int(indices[right]),
                        "right": names[indices[right]],
                    }
                )
    return edges


def aggregate_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for trial in trials:
        by_method.setdefault(str(trial["method"]), []).append(trial)

    aggregate: dict[str, Any] = {}
    metric_names = (
        "precision",
        "recall",
        "f1",
        "skeleton_shd",
        "runtime_seconds",
        "edge_count",
    )
    ceg_fuse_metric_names = (
        "structural_hamming_distance",
        "expected_distance",
        "energy_score",
        "entropy",
        "high_mass_set_size",
        "high_mass_truth_coverage",
        "neighborhood_mass",
        "mode_distance",
        "adjacency_f1",
        "orientation_f1",
        "endpoint_f1",
    )
    for method, method_trials in by_method.items():
        aggregate[method] = {"trials": len(method_trials)}
        for metric_name in metric_names:
            values = [
                float(trial[metric_name])
                for trial in method_trials
                if trial[metric_name] is not None
            ]
            aggregate[method][metric_name] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                if values
                else None
            )
        aggregate[method]["ceg_fuse_metrics"] = {}
        for metric_name in ceg_fuse_metric_names:
            values = [
                float(trial["ceg_fuse_metrics"][metric_name])
                for trial in method_trials
                if trial["ceg_fuse_metrics"][metric_name] is not None
            ]
            aggregate[method]["ceg_fuse_metrics"][metric_name] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                if values
                else None
            )
    return aggregate


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.seeds:
        parser.error("--seeds requires at least one seed.")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between 0 and 1.")
    if not 0.0 <= args.random_guess_edge_probability <= 1.0:
        parser.error("--random-guess-edge-probability must be between 0 and 1.")
    if args.expert_source == "file" and args.guess_path is None:
        parser.error("--expert-source file requires --guess-path.")
    if not 0.0 <= args.llm_temperature <= 2.0:
        parser.error("--llm-temperature must be between 0 and 2.")
    if args.llm_max_tokens < 1:
        parser.error("--llm-max-tokens must be positive.")
    if args.module_size < 2:
        parser.error("--module-size must be at least 2.")
    if (
        args.full_graph
        and "gpc_guess" in args.methods
        and not args.allow_expensive_full_gpc
        and not args.dry_run
    ):
        parser.error(
            "Full-graph gPC-Guess can enumerate exponentially many conditioning sets. "
            "Pass --allow-expensive-full-gpc to acknowledge the risk."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_RESULTS_ROOT / run_id(args.run_id) / "neuropathic_guess2graph"
        )

    names = canonical_variable_names()
    simulator_edges = load_simulator_edges(args.model_path)
    indices = selected_indices(
        args.full_graph,
        len(names),
        module_size=args.module_size,
        model_path=args.model_path,
    )
    data = load_observations(args.data_path, indices)
    target = target_skeleton_from_dag(simulator_edges, indices, names)
    ceg_target = target_ceg_graph_from_dag(simulator_edges, indices, names)

    expert_source = "file" if args.guess_path is not None else args.expert_source
    fixed_guess: np.ndarray | None = None
    if expert_source == "random":
        guess_source = {
            "kind": "random_control",
            "base_seed": args.guess_seed,
            "edge_probability": args.random_guess_edge_probability,
            "warning": (
                "This is an unguided control, not an expert-guided Guess2Graph result. "
                "Use the default --expert-source llm for the primary baseline."
            ),
        }
    elif expert_source == "file":
        if args.guess_path is None:
            raise AssertionError("Validated file expert is missing --guess-path.")
        matrix, payload_nodes = load_guess_payload(args.guess_path)
        fixed_guess = normalize_skeleton(
            align_guess(matrix, payload_nodes, indices, names, len(names))
        )
        guess_source = {"kind": "file_skeleton", "path": str(args.guess_path)}
    else:
        guess_source = {
            "kind": "llm_skeleton",
            "model": args.llm_model,
            "temperature": args.llm_temperature,
            "one_prediction_per_seed": True,
        }

    plan = {
        "implementation_source": {
            "branch": "origin/luke/neuropathic-baselines",
            "commit": "e388190",
        },
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
        "target_source": "simulator DAG restricted to the selected module",
        "sample_size": int(data.shape[0]),
        "node_count": int(data.shape[1]),
        "module_size": args.module_size,
        "node_indices": list(indices),
        "nodes": [names[index] for index in indices],
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "alpha": args.alpha,
        "ci_test": args.ci_test,
        "guess_source": guess_source,
        "expert_protocol": (
            "One undirected skeleton guess per trial; expert-predicted non-edges are "
            "tested first and conditioning-set order remains random."
        ),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        parser.error(
            f"{summary_path} already exists. Pass --overwrite or choose another output directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = [names[index] for index in indices]
    llm_expert = (
        CachedOpenAISkeletonExpert(
            cache_dir=args.output_dir / "llm_cache",
            settings=LLMExpertSettings(
                model=args.llm_model,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
            ),
        )
        if expert_source == "llm"
        else None
    )

    trials = []
    for seed in args.seeds:
        trial_dir = args.output_dir / f"seed_{seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        if expert_source == "llm":
            if llm_expert is None:
                raise AssertionError("LLM expert was not initialized.")
            llm_guess = llm_expert.guess(selected_names, trial_seed=seed)
            expert_skeleton = llm_guess.skeleton
            trial_expert = {
                **guess_source,
                "trial_seed": seed,
                "shuffled_nodes": list(llm_guess.shuffled_nodes),
                "prompt": llm_guess.prompt,
                "response": llm_guess.response,
                "cache_path": str(llm_guess.cache_path),
            }
        elif expert_source == "random":
            random_seed = args.guess_seed + seed
            expert_skeleton = normalize_skeleton(
                random_dag(
                    len(indices),
                    args.random_guess_edge_probability,
                    random_seed,
                )
            )
            trial_expert = {**guess_source, "trial_seed": random_seed}
        else:
            if fixed_guess is None:
                raise AssertionError("File expert skeleton was not loaded.")
            expert_skeleton = fixed_guess
            trial_expert = dict(guess_source)

        expert_edge_count = int(np.triu(expert_skeleton, k=1).sum())
        np.save(trial_dir / "expert_skeleton.npy", expert_skeleton)
        write_json(
            trial_dir / "expert_skeleton.json",
            {
                **trial_expert,
                "node_indices": list(indices),
                "nodes": selected_names,
                "edges": matrix_edges(expert_skeleton, indices, names),
                "adjacency_matrix": expert_skeleton.tolist(),
            },
        )

        for method in args.methods:
            method_rng = np.random.default_rng(seed)
            if method == "pc_guess":
                start = time.monotonic()
                predicted_raw = run_pc_guess_skeleton(
                    data,
                    expert_skeleton,
                    alpha=args.alpha,
                    ci_test=args.ci_test,
                    rng=method_rng,
                )
                runtime = time.monotonic() - start
            elif method == "gpc_guess":
                start = time.monotonic()
                predicted_raw = run_gpc_guess_skeleton(
                    data,
                    expert_skeleton,
                    alpha=args.alpha,
                    ci_test=args.ci_test,
                    rng=method_rng,
                )
                runtime = time.monotonic() - start
            else:
                raise AssertionError(f"unsupported Guess2Graph method: {method}")
            predicted = normalize_skeleton(predicted_raw)
            metrics = skeleton_metrics(predicted, target)
            ceg_fuse_metrics = ceg_fuse_skeleton_metrics(predicted, ceg_target)
            record = {
                "seed": seed,
                "method": method,
                "expert_source": expert_source,
                "expert_edge_count": expert_edge_count,
                **metrics,
                "ceg_fuse_metrics": ceg_fuse_metrics,
                "runtime_seconds": float(runtime),
                "edge_count": int(np.triu(predicted, k=1).sum()),
            }
            trials.append(record)
            np.save(trial_dir / f"{method}_skeleton.npy", predicted)
            write_json(
                trial_dir / f"{method}_skeleton.json",
                {
                    "seed": seed,
                    "method": method,
                    "node_indices": list(indices),
                    "nodes": selected_names,
                    "edges": matrix_edges(predicted, indices, names),
                    "expert_source": expert_source,
                    "expert_edge_count": expert_edge_count,
                    "metrics": {**metrics, "ceg_fuse_metrics": ceg_fuse_metrics},
                    "runtime_seconds": float(runtime),
                },
            )

    summary = {
        "config": plan,
        "metric_scope": (
            "Skeleton-only. Vendored Guess2Graph methods do not return endpoint orientations, "
            "so PAG typed-edge metrics are intentionally not reported. CEG-FUSE metrics score "
            "a point-mass undirected prediction against the directed simulator DAG; orientation "
            "F1 therefore gives no credit for unoriented edges, and endpoint F1 is unavailable."
        ),
        "trials": trials,
        "aggregate": aggregate_trials(trials),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Saved Guess2Graph neuropathic results to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
