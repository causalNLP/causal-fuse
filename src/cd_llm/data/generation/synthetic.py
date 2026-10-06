from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "generated" / "synthetic_mixed_scm"


@dataclass(frozen=True)
class NodeSpec:
    name: str
    data_type: str
    categories: int | None
    parents: list[str]
    mechanism: str
    intercept: float
    coefficients: list[float]
    interaction_coefficients: list[float]
    thresholds: list[float] | None
    noise_scale: float | None


@dataclass(frozen=True)
class GraphSpec:
    num_nodes: int
    edge_probability: float
    max_parents: int
    topological_order: list[int]
    edges: list[tuple[int, int]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mixed continuous/discrete synthetic SCM datasets with varying DAG structure."
        )
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        required=True,
        help="One or more sample sizes to generate.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20,
        help="Number of variables in each generated DAG before hiding latent variables.",
    )
    parser.add_argument(
        "--hidden-count",
        type=int,
        default=0,
        help=(
            "Number of generated variables to hide from the output CSV. Hidden variables "
            "remain in the metadata graph and can act as latent confounders. "
            "Overridden when --hidden-ratio is set."
        ),
    )
    parser.add_argument(
        "--hidden-ratio",
        type=float,
        default=None,
        help=(
            "Fraction of generated variables to hide from the output CSV. "
            "For positive ratios, at least one variable is hidden."
        ),
    )
    parser.add_argument(
        "--hidden-selection",
        choices=["topological_prefix", "random"],
        default="topological_prefix",
        help=(
            "How to choose hidden variables when hidden variables are requested. "
            "topological_prefix hides early DAG nodes, which are more likely to be upstream causes."
        ),
    )
    parser.add_argument(
        "--min-hidden-observed-children",
        type=int,
        default=2,
        help=(
            "Minimum number of observed children to give each hidden variable when possible. "
            "This makes hidden variables act as latent common causes."
        ),
    )
    parser.add_argument(
        "--edge-probability",
        type=float,
        default=0.15,
        help="Probability of each forward edge under the sampled topological order.",
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        default=4,
        help="Maximum number of parents allowed for each node.",
    )
    parser.add_argument(
        "--discrete-probability",
        type=float,
        default=0.4,
        help="Probability that a generated variable is discrete.",
    )
    parser.add_argument(
        "--min-categories",
        type=int,
        default=2,
        help="Minimum number of categories for discrete variables.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=4,
        help="Maximum number of categories for discrete variables.",
    )
    parser.add_argument(
        "--mechanism",
        choices=["linear", "additive_nonlinear", "interaction"],
        default="additive_nonlinear",
        help="Structural mechanism family used for parent effects.",
    )
    parser.add_argument(
        "--coefficient-scale",
        type=float,
        default=1.0,
        help="Standard deviation for sampled structural coefficients.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Gaussian noise scale for continuous variables.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="One or more random seeds. Each seed produces a distinct DAG and SCM.",
    )
    parser.add_argument(
        "--condition",
        default="mixed_scm",
        help="Condition name embedded in output filenames and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.num_nodes < 2:
        parser.error("--num-nodes must be at least 2.")
    if args.hidden_count < 0:
        parser.error("--hidden-count must be nonnegative.")
    if args.hidden_ratio is not None and not 0.0 <= args.hidden_ratio < 1.0:
        parser.error("--hidden-ratio must be in [0, 1).")
    resolved_hidden_count = resolve_hidden_count(
        num_nodes=args.num_nodes,
        hidden_ratio=args.hidden_ratio,
        hidden_count=args.hidden_count,
    )
    if resolved_hidden_count >= args.num_nodes:
        parser.error("--hidden-count must be smaller than --num-nodes.")
    args.resolved_hidden_count = resolved_hidden_count
    if args.min_hidden_observed_children < 0:
        parser.error("--min-hidden-observed-children must be nonnegative.")
    if not 0.0 <= args.edge_probability <= 1.0:
        parser.error("--edge-probability must be in [0, 1].")
    if args.max_parents < 0:
        parser.error("--max-parents must be nonnegative.")
    if not 0.0 <= args.discrete_probability <= 1.0:
        parser.error("--discrete-probability must be in [0, 1].")
    if args.min_categories < 2:
        parser.error("--min-categories must be at least 2.")
    if args.max_categories < args.min_categories:
        parser.error("--max-categories must be at least --min-categories.")
    if args.coefficient_scale <= 0:
        parser.error("--coefficient-scale must be positive.")
    if args.noise_scale <= 0:
        parser.error("--noise-scale must be positive.")
    if any(sample_size <= 0 for sample_size in args.sample_sizes):
        parser.error("--sample-sizes must all be positive.")
    return args


def resolve_hidden_count(
    *,
    num_nodes: int,
    hidden_ratio: float | None,
    hidden_count: int,
) -> int:
    if hidden_ratio is None:
        return hidden_count
    if hidden_ratio <= 0.0:
        return 0
    return min(num_nodes - 1, max(1, int(round(hidden_ratio * num_nodes))))


def sample_graph(
    *,
    num_nodes: int,
    edge_probability: float,
    max_parents: int,
    rng: np.random.Generator,
) -> GraphSpec:
    topological_order = rng.permutation(num_nodes).tolist()
    candidate_edges: list[tuple[int, int]] = []
    parent_counts = {node: 0 for node in range(num_nodes)}

    for source_position in range(num_nodes):
        source = topological_order[source_position]
        for target_position in range(source_position + 1, num_nodes):
            target = topological_order[target_position]
            if parent_counts[target] >= max_parents:
                continue
            if rng.random() <= edge_probability:
                candidate_edges.append((source, target))
                parent_counts[target] += 1

    return GraphSpec(
        num_nodes=num_nodes,
        edge_probability=edge_probability,
        max_parents=max_parents,
        topological_order=topological_order,
        edges=sorted(candidate_edges),
    )


def assign_node_types(
    *,
    num_nodes: int,
    discrete_probability: float,
    min_categories: int,
    max_categories: int,
    rng: np.random.Generator,
) -> list[tuple[str, int | None]]:
    is_discrete = rng.random(num_nodes) < discrete_probability
    if num_nodes >= 2:
        if not np.any(is_discrete):
            is_discrete[int(rng.integers(num_nodes))] = True
        if np.all(is_discrete):
            is_discrete[int(rng.integers(num_nodes))] = False

    node_types: list[tuple[str, int | None]] = []
    for discrete in is_discrete:
        if discrete:
            categories = int(rng.integers(min_categories, max_categories + 1))
            node_types.append(("disc", categories))
        else:
            node_types.append(("cont", None))
    return node_types


def build_node_specs(
    *,
    graph: GraphSpec,
    node_types: list[tuple[str, int | None]],
    mechanism: str,
    coefficient_scale: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> list[NodeSpec]:
    parents_by_node: dict[int, list[int]] = {node: [] for node in range(graph.num_nodes)}
    for source, target in graph.edges:
        parents_by_node[target].append(source)

    specs: list[NodeSpec] = []
    for node in range(graph.num_nodes):
        parents = sorted(parents_by_node[node])
        coefficients = rng.normal(0.0, coefficient_scale, size=len(parents)).tolist()
        interaction_count = len(parents) * (len(parents) - 1) // 2 if mechanism == "interaction" else 0
        interaction_coefficients = rng.normal(
            0.0, coefficient_scale / 2.0, size=interaction_count
        ).tolist()
        intercept = float(rng.normal(0.0, 0.5))
        data_type, categories = node_types[node]
        thresholds = None
        node_noise_scale = noise_scale if data_type == "cont" else None

        if data_type == "disc" and categories is not None and categories > 2:
            raw_thresholds = rng.normal(0.0, 1.0, size=categories - 1)
            thresholds = sorted(float(value) for value in raw_thresholds)

        specs.append(
            NodeSpec(
                name=variable_name(node),
                data_type=data_type,
                categories=categories,
                parents=[variable_name(parent) for parent in parents],
                mechanism=mechanism,
                intercept=intercept,
                coefficients=[float(value) for value in coefficients],
                interaction_coefficients=[float(value) for value in interaction_coefficients],
                thresholds=thresholds,
                noise_scale=node_noise_scale,
            )
        )
    return specs


def generate_samples(
    *,
    sample_size: int,
    graph: GraphSpec,
    node_specs: list[NodeSpec],
    rng: np.random.Generator,
) -> pd.DataFrame:
    values_by_name: dict[str, np.ndarray] = {}

    for node in graph.topological_order:
        spec = node_specs[node]
        parent_matrix = parent_values(spec.parents, values_by_name, sample_size)
        signal = structural_signal(spec, parent_matrix)
        if spec.data_type == "cont":
            noise = rng.normal(0.0, spec.noise_scale or 1.0, size=sample_size)
            values = signal + noise
        else:
            values = sample_discrete_values(spec, signal, rng)
        values_by_name[spec.name] = values

    ordered_columns = [variable_name(index) for index in range(graph.num_nodes)]
    return pd.DataFrame({name: values_by_name[name] for name in ordered_columns})


def parent_values(
    parent_names: list[str],
    values_by_name: dict[str, np.ndarray],
    sample_size: int,
) -> np.ndarray:
    if not parent_names:
        return np.empty((sample_size, 0), dtype=float)
    columns = [standardize(values_by_name[parent]) for parent in parent_names]
    return np.column_stack(columns)


def structural_signal(spec: NodeSpec, parent_matrix: np.ndarray) -> np.ndarray:
    sample_size = parent_matrix.shape[0]
    signal = np.full(sample_size, spec.intercept, dtype=float)
    if parent_matrix.shape[1] == 0:
        return signal

    coefficients = np.asarray(spec.coefficients, dtype=float)
    if spec.mechanism == "linear":
        return signal + parent_matrix @ coefficients

    nonlinear_terms = np.tanh(parent_matrix) + 0.5 * np.sin(parent_matrix)
    signal = signal + nonlinear_terms @ coefficients

    if spec.mechanism == "interaction" and parent_matrix.shape[1] >= 2:
        interaction_index = 0
        for first in range(parent_matrix.shape[1]):
            for second in range(first + 1, parent_matrix.shape[1]):
                signal = (
                    signal
                    + spec.interaction_coefficients[interaction_index]
                    * parent_matrix[:, first]
                    * parent_matrix[:, second]
                )
                interaction_index += 1
    return signal


def sample_discrete_values(
    spec: NodeSpec,
    signal: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    categories = spec.categories or 2
    if categories == 2:
        probabilities = sigmoid(signal)
        return rng.binomial(1, probabilities).astype(int)

    thresholds = np.asarray(spec.thresholds or [], dtype=float)
    logits = np.column_stack(
        [
            np.zeros_like(signal),
            *[signal - threshold for threshold in thresholds],
        ]
    )
    probabilities = softmax(logits)
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(signal.shape[0])
    return (draws[:, None] > cumulative).sum(axis=1).astype(int)


def standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    std = float(values.std())
    if std < 1e-12:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def variable_name(index: int) -> str:
    return f"X{index + 1}"


def output_stem(
    *,
    condition: str,
    sample_size: int,
    num_nodes: int,
    edge_probability: float,
    discrete_probability: float,
    hidden_count: int,
    seed: int,
) -> str:
    rho = compact_float(edge_probability)
    q = compact_float(discrete_probability)
    hidden = f"_h{hidden_count}" if hidden_count else ""
    return f"{condition}_n{sample_size}_p{num_nodes}{hidden}_rho{rho}_q{q}_seed{seed}"


def compact_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.4g}".replace(".", "p").replace("-", "m")


def write_edges(path: Path, graph: GraphSpec) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        for source, target in graph.edges:
            writer.writerow([variable_name(source), variable_name(target)])


def write_observed_edges(path: Path, graph: GraphSpec, observed_nodes: set[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        for source, target in graph.edges:
            source_name = variable_name(source)
            target_name = variable_name(target)
            if source_name in observed_nodes and target_name in observed_nodes:
                writer.writerow([source_name, target_name])


def write_types(path: Path, node_specs: list[NodeSpec]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "data_type", "categories"])
        for spec in node_specs:
            writer.writerow([spec.name, spec.data_type, spec.categories or ""])


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    sample_size: int,
    seed: int,
    graph: GraphSpec,
    node_specs: list[NodeSpec],
    observed_names: list[str],
    hidden_names: list[str],
) -> None:
    payload: dict[str, Any] = {
        "condition": args.condition,
        "sample_size": sample_size,
        "seed": seed,
        "generator": "mixed_scm",
        "hyperparameters": {
            "num_nodes": args.num_nodes,
            "edge_probability": args.edge_probability,
            "max_parents": args.max_parents,
            "discrete_probability": args.discrete_probability,
            "min_categories": args.min_categories,
            "max_categories": args.max_categories,
            "mechanism": args.mechanism,
            "coefficient_scale": args.coefficient_scale,
            "noise_scale": args.noise_scale,
            "hidden_count": args.resolved_hidden_count,
            "hidden_ratio": args.hidden_ratio,
            "hidden_selection": args.hidden_selection,
            "min_hidden_observed_children": args.min_hidden_observed_children,
        },
        "observed_nodes": observed_names,
        "hidden_nodes": hidden_names,
        "graph": {
            **asdict(graph),
            "edges": [[variable_name(source), variable_name(target)] for source, target in graph.edges],
            "topological_order": [variable_name(index) for index in graph.topological_order],
        },
        "nodes": [asdict(spec) for spec in node_specs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def select_hidden_indices(
    *,
    graph: GraphSpec,
    hidden_count: int,
    hidden_selection: str,
    rng: np.random.Generator,
) -> list[int]:
    if hidden_count == 0:
        return []
    if hidden_selection == "topological_prefix":
        hidden_indices = graph.topological_order[:hidden_count]
    elif hidden_selection == "random":
        hidden_indices = sorted(rng.choice(graph.num_nodes, size=hidden_count, replace=False).tolist())
    else:
        raise ValueError(f"Unsupported hidden selection: {hidden_selection}")
    return list(hidden_indices)


def ensure_hidden_observed_child_edges(
    *,
    graph: GraphSpec,
    hidden_indices: list[int],
    min_hidden_observed_children: int,
    rng: np.random.Generator,
) -> GraphSpec:
    if not hidden_indices or min_hidden_observed_children == 0:
        return graph

    hidden = set(hidden_indices)
    position = {node: index for index, node in enumerate(graph.topological_order)}
    edges = set(graph.edges)
    children_by_hidden = {
        hidden_node: {
            child
            for parent, child in edges
            if parent == hidden_node and child not in hidden
        }
        for hidden_node in hidden_indices
    }

    for hidden_node in hidden_indices:
        candidates = [
            node
            for node in graph.topological_order
            if node not in hidden
            and position[hidden_node] < position[node]
            and (hidden_node, node) not in edges
        ]
        needed = min_hidden_observed_children - len(children_by_hidden[hidden_node])
        if needed <= 0 or not candidates:
            continue
        selected_count = min(needed, len(candidates))
        selected_children = rng.choice(candidates, size=selected_count, replace=False).tolist()
        for child in selected_children:
            edges.add((hidden_node, int(child)))

    return GraphSpec(
        num_nodes=graph.num_nodes,
        edge_probability=graph.edge_probability,
        max_parents=graph.max_parents,
        topological_order=list(graph.topological_order),
        edges=sorted(edges),
    )


def generate_one_dataset(
    *,
    args: argparse.Namespace,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, GraphSpec, list[NodeSpec], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    graph = sample_graph(
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        max_parents=args.max_parents,
        rng=rng,
    )
    hidden_indices = select_hidden_indices(
        graph=graph,
        hidden_count=args.resolved_hidden_count,
        hidden_selection=args.hidden_selection,
        rng=rng,
    )
    graph = ensure_hidden_observed_child_edges(
        graph=graph,
        hidden_indices=hidden_indices,
        min_hidden_observed_children=args.min_hidden_observed_children,
        rng=rng,
    )
    node_types = assign_node_types(
        num_nodes=args.num_nodes,
        discrete_probability=args.discrete_probability,
        min_categories=args.min_categories,
        max_categories=args.max_categories,
        rng=rng,
    )
    node_specs = build_node_specs(
        graph=graph,
        node_types=node_types,
        mechanism=args.mechanism,
        coefficient_scale=args.coefficient_scale,
        noise_scale=args.noise_scale,
        rng=rng,
    )
    data = generate_samples(
        sample_size=sample_size,
        graph=graph,
        node_specs=node_specs,
        rng=rng,
    )
    hidden_names = [variable_name(index) for index in hidden_indices]
    observed_names = [column for column in data.columns if column not in set(hidden_names)]
    return data[observed_names], graph, node_specs, observed_names, hidden_names


def ensure_outputs_are_writable(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing output files:\n{formatted}\nUse --overwrite.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(sample_size, seed) for sample_size in args.sample_sizes for seed in args.seeds]
    start_time = time.monotonic()

    for completed, (sample_size, seed) in enumerate(tasks, start=1):
        stem = output_stem(
            condition=args.condition,
            sample_size=sample_size,
            num_nodes=args.num_nodes,
            edge_probability=args.edge_probability,
            discrete_probability=args.discrete_probability,
            hidden_count=args.resolved_hidden_count,
            seed=seed,
        )
        data_path = args.output_dir / f"{stem}.csv"
        edges_path = args.output_dir / f"{stem}_edges.csv"
        observed_edges_path = args.output_dir / f"{stem}_observed_edges.csv"
        types_path = args.output_dir / f"{stem}_types.csv"
        metadata_path = args.output_dir / f"{stem}.json"
        ensure_outputs_are_writable(
            [data_path, edges_path, observed_edges_path, types_path, metadata_path],
            args.overwrite,
        )

        data, graph, node_specs, observed_names, hidden_names = generate_one_dataset(
            args=args,
            sample_size=sample_size,
            seed=seed,
        )
        data.to_csv(data_path, index=False)
        write_edges(edges_path, graph)
        write_observed_edges(observed_edges_path, graph, set(observed_names))
        write_types(types_path, node_specs)
        write_metadata(
            metadata_path,
            args=args,
            sample_size=sample_size,
            seed=seed,
            graph=graph,
            node_specs=node_specs,
            observed_names=observed_names,
            hidden_names=hidden_names,
        )

        elapsed_seconds = time.monotonic() - start_time
        average_seconds = elapsed_seconds / completed
        remaining_seconds = average_seconds * (len(tasks) - completed)
        print(
            f"[{completed}/{len(tasks)}] Wrote {data_path} "
            f"(ETA: {format_duration(remaining_seconds)})"
        )
    return 0


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
