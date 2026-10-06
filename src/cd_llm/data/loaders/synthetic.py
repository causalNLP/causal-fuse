from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_metadata(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    required_keys = {"condition", "sample_size", "seed", "generator", "graph", "nodes"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Metadata file {path} is missing keys: {sorted(missing)}")
    if payload["generator"] != "mixed_scm":
        raise ValueError(f"Metadata file {path} is not a mixed_scm dataset.")
    graph = payload["graph"]
    if not isinstance(graph, dict) or "edges" not in graph:
        raise ValueError(f"Metadata file {path} is missing graph.edges.")
    return payload


def node_names_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    if "observed_nodes" in metadata:
        return [str(name) for name in metadata["observed_nodes"]]

    names = []
    for node in metadata["nodes"]:
        if not isinstance(node, dict) or "name" not in node:
            raise ValueError("Each node metadata entry must contain a name.")
        names.append(str(node["name"]))
    return names


def all_node_names_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    names = []
    for node in metadata["nodes"]:
        if not isinstance(node, dict) or "name" not in node:
            raise ValueError("Each node metadata entry must contain a name.")
        names.append(str(node["name"]))
    return names


def edges_from_metadata(metadata: Dict[str, Any]) -> List[Tuple[str, str]]:
    edges = []
    for edge in metadata["graph"]["edges"]:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Invalid edge entry in synthetic metadata: {edge!r}")
        edges.append((str(edge[0]), str(edge[1])))
    return edges


def find_metadata_for_selection(
    data_dir: Path,
    sample_sizes: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    conditions: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> List[Path]:
    selected_sample_sizes = set(sample_sizes) if sample_sizes is not None else None
    selected_seeds = set(seeds) if seeds is not None else None
    selected_conditions = set(conditions) if conditions is not None else None
    selected_datasets = set(datasets) if datasets is not None else None
    metadata_paths = []
    for path in sorted(data_dir.glob("*.json")):
        metadata = load_metadata(path)
        if selected_sample_sizes is not None and int(metadata["sample_size"]) not in selected_sample_sizes:
            continue
        if selected_seeds is not None and int(metadata["seed"]) not in selected_seeds:
            continue
        if selected_conditions is not None and str(metadata["condition"]) not in selected_conditions:
            continue
        if (
            selected_datasets is not None
            and path.name not in selected_datasets
            and path.stem not in selected_datasets
        ):
            continue
        metadata_paths.append(path)
    if not metadata_paths:
        raise FileNotFoundError(
            f"No synthetic metadata files found in {data_dir}. "
            f"sample_sizes={sorted(selected_sample_sizes) if selected_sample_sizes is not None else 'all'}, "
            f"seeds={sorted(selected_seeds) if selected_seeds is not None else 'all'}, "
            f"conditions={sorted(selected_conditions) if selected_conditions is not None else 'all'}, "
            f"datasets={sorted(selected_datasets) if selected_datasets is not None else 'all'}"
        )
    return metadata_paths
