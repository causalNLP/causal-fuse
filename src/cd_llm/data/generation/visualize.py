from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import networkx as nx


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "generated" / "synthetic_mixed_scm"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize mixed SCM causal graphs from generated metadata JSON files. "
            "Hidden/unobserved variables are drawn with dashed outlines."
        )
    )
    parser.add_argument(
        "metadata",
        nargs="*",
        type=Path,
        help="Metadata JSON files to visualize. If omitted, --input-dir and --pattern are used.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing generated metadata JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="mixed_scm_hidden_*_n500_p20*_seed0.json",
        help="Glob pattern used with --input-dir when no metadata files are passed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PDF figures. Defaults to the metadata file directory.",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=0,
        help="Accepted for backward compatibility; the layered DAG layout is deterministic.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.metadata:
        args.metadata_paths = sorted(args.metadata)
    else:
        args.metadata_paths = sorted(args.input_dir.glob(args.pattern))
    if not args.metadata_paths:
        raise SystemExit("No metadata JSON files found.")
    return args


def load_graph(metadata_path: Path) -> tuple[dict[str, Any], nx.DiGraph]:
    metadata = json.loads(metadata_path.read_text())
    graph = nx.DiGraph()
    nodes = metadata["graph"]["topological_order"]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(tuple(edge) for edge in metadata["graph"]["edges"])
    return metadata, graph


def layered_dag_layout(graph: nx.DiGraph, topological_order: Sequence[str]) -> dict[str, tuple[float, float]]:
    layer_by_node: dict[str, int] = {}
    topological_index = {node: index for index, node in enumerate(topological_order)}

    for node in topological_order:
        parent_layers = [layer_by_node[parent] for parent in graph.predecessors(node)]
        layer_by_node[node] = max(parent_layers, default=-1) + 1

    nodes_by_layer: dict[int, list[str]] = {}
    for node, layer in layer_by_node.items():
        nodes_by_layer.setdefault(layer, []).append(node)

    max_layer_size = max((len(nodes) for nodes in nodes_by_layer.values()), default=1)
    y_gap = 1.35
    x_gap = 2.35
    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in nodes_by_layer.items():
        ordered_nodes = sorted(nodes, key=topological_index.__getitem__)
        layer_height = (len(ordered_nodes) - 1) * y_gap
        sparse_offset = (max_layer_size - len(ordered_nodes)) * y_gap / 2.0
        for index, node in enumerate(ordered_nodes):
            x = layer * x_gap
            y = -index * y_gap - sparse_offset + layer_height / 2.0
            pos[node] = (x, y)
    return pos


def draw_graph(
    *,
    metadata_path: Path,
    output_path: Path,
    layout_seed: int,
) -> None:
    metadata, graph = load_graph(metadata_path)
    hidden_nodes = set(metadata.get("hidden_nodes", []))
    observed_nodes = [node for node in graph.nodes if node not in hidden_nodes]
    hidden_node_list = [node for node in graph.nodes if node in hidden_nodes]
    topological_order = metadata["graph"]["topological_order"]

    pos = layered_dag_layout(graph, topological_order)
    layer_count = len({x for x, _ in pos.values()})
    max_layer_size = max(
        sum(1 for x, _ in pos.values() if x == layer_x)
        for layer_x in {x for x, _ in pos.values()}
    )
    figure_width = max(10.0, 1.8 * layer_count)
    figure_height = max(6.5, 0.8 * max_layer_size + 1.5)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    solid_edges = [
        edge for edge in graph.edges if edge[0] not in hidden_nodes and edge[1] not in hidden_nodes
    ]
    dashed_edges = [
        edge for edge in graph.edges if edge[0] in hidden_nodes or edge[1] in hidden_nodes
    ]

    node_size = 780
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=observed_nodes,
        ax=ax,
        node_color="#dbeafe",
        edgecolors="#1f2937",
        linewidths=1.4,
        node_size=node_size,
    )
    hidden_collection = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=hidden_node_list,
        ax=ax,
        node_color="#f3f4f6",
        edgecolors="#111827",
        linewidths=1.8,
        node_size=node_size,
    )
    hidden_collection.set_linestyle("dashed")

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=solid_edges,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=24,
        edge_color="#111827",
        width=1.8,
        connectionstyle="arc3,rad=0.05",
        node_size=node_size,
        min_source_margin=12,
        min_target_margin=18,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=dashed_edges,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=24,
        edge_color="#374151",
        width=1.8,
        style="dashed",
        connectionstyle="arc3,rad=0.05",
        node_size=node_size,
        min_source_margin=12,
        min_target_margin=18,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        font_size=9,
        font_color="#111827",
        font_weight="bold",
    )

    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for metadata_path in args.metadata_paths:
        output_dir = args.output_dir or metadata_path.parent
        output_path = output_dir / f"{metadata_path.stem}_graph.pdf"
        draw_graph(
            metadata_path=metadata_path,
            output_path=output_path,
            layout_seed=args.layout_seed,
        )
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
