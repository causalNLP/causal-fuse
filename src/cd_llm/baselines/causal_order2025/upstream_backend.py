"""Pinned, read-only adapter for the authors' causal-order implementation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import networkx as nx

from cd_llm.core.finite_support import graph_from_edges

from .types import (
    CausalOrderInfeasibleError,
    Edge,
    TripletConfig,
    TripletExpertResult,
    UpstreamCausalOrderError,
    normalize_node_names,
)


UPSTREAM_REVISION = "ad609dbd3d93e50d890c9967eb746945546515d1"


@dataclass(frozen=True)
class _UpstreamModules:
    triplet: ModuleType
    cycle_remover: ModuleType
    metrics: ModuleType


def _default_upstream_root() -> Path:
    return Path(__file__).resolve().parents[4] / "vendor" / "Causal_Order_Imperfect_Experts"


def verified_upstream_root(upstream_root: str | Path | None = None) -> Path:
    """Return a clean checkout at the exact revision used by this adapter."""

    root = (
        _default_upstream_root()
        if upstream_root is None
        else Path(upstream_root).expanduser().resolve()
    )
    required = (
        root / "causal_discovery" / "__init__.py",
        root / "causal_discovery" / "strategies" / "triplet.py",
        root / "causal_discovery" / "utils" / "cycle_remover.py",
        root / "causal_discovery" / "utils" / "llm_client.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing pinned causal-order checkout files: "
            + ", ".join(str(path) for path in missing)
        )
    try:
        revision = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise UpstreamCausalOrderError(
            f"cannot verify the causal-order checkout at {root}"
        ) from error
    if revision != UPSTREAM_REVISION:
        raise UpstreamCausalOrderError(
            f"causal-order checkout is at {revision!r}; expected {UPSTREAM_REVISION}"
        )
    if status:
        raise UpstreamCausalOrderError(
            "tracked files in the causal-order checkout differ from the pinned revision"
        )
    return root


def _load_upstream_modules(root: Path) -> _UpstreamModules:
    digest = hashlib.blake2s(str(root).encode("utf-8"), digest_size=6).hexdigest()
    package_name = f"_cd_llm_causal_order_upstream_{UPSTREAM_REVISION[:10]}_{digest}"
    if package_name not in sys.modules:
        package_root = root / "causal_discovery"
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_root / "__init__.py",
            submodule_search_locations=[str(package_root)],
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise UpstreamCausalOrderError(
                f"cannot create an import specification for {package_root}"
            )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            sys.modules.pop(package_name, None)
            raise
    try:
        triplet = importlib.import_module(f"{package_name}.strategies.triplet")
        cycle_remover = importlib.import_module(
            f"{package_name}.utils.cycle_remover"
        )
        metrics = importlib.import_module(f"{package_name}.utils.metrics")
    except (ImportError, ModuleNotFoundError) as error:
        raise UpstreamCausalOrderError(
            "the upstream causal-order backend requires its optional dependencies; "
            "install causal_order2025/requirements-upstream.txt"
        ) from error
    for module, symbol in (
        (triplet, "run_triplet_experiment"),
        (triplet, "_tiebreaker_query"),
        (cycle_remover, "remove_cycles"),
        (metrics, "topological_ordering"),
    ):
        if not callable(getattr(module, symbol, None)):
            raise UpstreamCausalOrderError(
                f"pinned upstream module does not define callable {symbol}"
            )
    return _UpstreamModules(triplet, cycle_remover, metrics)


def _normalized_descriptions(
    node_names: tuple[str, ...], descriptions: Mapping[str, str] | None
) -> dict[str, str] | None:
    if descriptions is None:
        return None
    if not isinstance(descriptions, Mapping):
        raise TypeError("descriptions must be a mapping or None")
    unknown = set(descriptions) - set(node_names)
    if unknown:
        raise ValueError(
            "descriptions contain unknown nodes: " + ", ".join(sorted(unknown))
        )
    normalized: dict[str, str] = {}
    for node, description in descriptions.items():
        if not isinstance(node, str) or not isinstance(description, str):
            raise TypeError("descriptions must map node names to strings")
        normalized[node] = description
    return normalized


def _edge_indices(
    values: object, name_to_index: Mapping[str, int], label: str
) -> tuple[Edge, ...]:
    try:
        raw_edges = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise UpstreamCausalOrderError(f"upstream {label} is not an edge list") from error
    edges: list[Edge] = []
    for value in raw_edges:
        try:
            source_name, target_name = value
        except (TypeError, ValueError) as error:
            raise UpstreamCausalOrderError(
                f"upstream {label} contains a malformed edge"
            ) from error
        if source_name not in name_to_index or target_name not in name_to_index:
            raise UpstreamCausalOrderError(
                f"upstream {label} contains an unknown node"
            )
        edge = name_to_index[source_name], name_to_index[target_name]
        if edge[0] == edge[1]:
            raise UpstreamCausalOrderError(f"upstream {label} contains a self-edge")
        edges.append(edge)
    if len(set(edges)) != len(edges):
        raise UpstreamCausalOrderError(f"upstream {label} contains duplicate edges")
    return tuple(edges)


def _edgewise_indices(
    values: object, name_to_index: Mapping[str, int]
) -> dict[tuple[int, int], tuple[float, float, float]]:
    if not isinstance(values, Mapping):
        raise UpstreamCausalOrderError("upstream edgewise_dist is not a mapping")
    result: dict[tuple[int, int], tuple[float, float, float]] = {}
    for raw_pair, raw_probabilities in values.items():
        try:
            first_name, second_name = raw_pair
            first = name_to_index[first_name]
            second = name_to_index[second_name]
            probabilities = tuple(float(value) for value in raw_probabilities)
        except (KeyError, TypeError, ValueError) as error:
            raise UpstreamCausalOrderError(
                "upstream edgewise_dist contains a malformed entry"
            ) from error
        if first == second or len(probabilities) != 3:
            raise UpstreamCausalOrderError(
                "upstream edgewise_dist contains an invalid pair or distribution"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise UpstreamCausalOrderError(
                "upstream edgewise probabilities must be finite and nonnegative"
            )
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise UpstreamCausalOrderError(
                "upstream edgewise probabilities must sum to one"
            )
        if first < second:
            pair = first, second
            normalized = probabilities
        else:
            pair = second, first
            normalized = probabilities[1], probabilities[0], probabilities[2]
        if pair in result:
            raise UpstreamCausalOrderError("upstream edgewise_dist repeats a node pair")
        result[pair] = normalized  # type: ignore[assignment]
    return result


def _subgroup_indices(
    values: object, name_to_index: Mapping[str, int]
) -> tuple[tuple[int, int, int], ...]:
    try:
        groups = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise UpstreamCausalOrderError("upstream subgroup_list is not iterable") from error
    result: list[tuple[int, int, int]] = []
    for raw_group in groups:
        try:
            group = tuple(name_to_index[name] for name in raw_group)
        except (KeyError, TypeError) as error:
            raise UpstreamCausalOrderError(
                "upstream subgroup_list contains an invalid node"
            ) from error
        if len(group) != 3 or len(set(group)) != 3:
            raise UpstreamCausalOrderError("upstream subgroup_list must contain triplets")
        result.append(group)  # type: ignore[arg-type]
    return tuple(result)


def validated_cycle_threshold(edge_weights):
    """Find the lowest distinct weight cutoff whose retained edges form a DAG."""
    if not edge_weights:
        return (), 0.0
    for edge, weight in edge_weights.items():
        if not isinstance(edge, tuple) or len(edge) != 2 or edge[0] == edge[1]:
            raise ValueError("invalid weighted edge")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("edge weights must be finite and positive")
    cutoffs = sorted(set(edge_weights.values()))
    upper = math.nextafter(cutoffs[-1], math.inf)
    if not math.isfinite(upper):
        raise ValueError("edge weights leave no finite upper cutoff")
    cutoffs.append(upper)

    def edges_at(index):
        return tuple(edge for edge, weight in edge_weights.items() if weight >= cutoffs[index])

    low, high = 0, len(cutoffs) - 1
    # The upper sentinel removes every edge and is always feasible. Search the
    # discrete edge sets, without a floating-point convergence tolerance.
    while low < high:
        mid = (low + high) // 2
        if nx.is_directed_acyclic_graph(nx.DiGraph(edges_at(mid))):
            high = mid
        else:
            low = mid + 1
    retained = edges_at(low)
    if not nx.is_directed_acyclic_graph(nx.DiGraph(retained)):
        raise ValueError("validated threshold did not produce a DAG")
    return retained, cutoffs[low]


def query_triplet_order(
    node_names: Sequence[str],
    context: str,
    descriptions: Mapping[str, str] | None = None,
    *,
    config: TripletConfig | None = None,
    upstream_root: str | Path | None = None,
    cycle_removal_policy: str = "upstream",
    cycle_removal_diagnostics: dict | None = None,
) -> TripletExpertResult:
    """Run pinned voting, optionally correcting an invalid cycle-removal cutoff."""

    if cycle_removal_policy not in {"upstream", "validated_threshold"}:
        raise ValueError("unknown cycle-removal policy")
    names = normalize_node_names(node_names, minimum=3)
    if not isinstance(context, str):
        raise TypeError("context must be a string")
    normalized_descriptions = _normalized_descriptions(names, descriptions)
    normalized_config = TripletConfig() if config is None else config
    if not isinstance(normalized_config, TripletConfig):
        raise TypeError("config must be TripletConfig")
    root = verified_upstream_root(upstream_root)
    modules = _load_upstream_modules(root)
    graph_config = {
        "nodes": list(names),
        "ground_truth_edges": [],
        "descriptions": normalized_descriptions,
        "context": context,
    }
    started = time.perf_counter()
    try:
        payload = modules.triplet.run_triplet_experiment(
            graph_config=graph_config,
            subgraph_model=normalized_config.subgraph_model,
            expert_model=normalized_config.expert_model,
            delay=normalized_config.delay,
            subgroup_size=3,
        )
    except Exception as error:
        raise UpstreamCausalOrderError("upstream triplet elicitation failed") from error
    if not isinstance(payload, Mapping):
        raise UpstreamCausalOrderError("upstream triplet strategy returned a non-mapping")
    name_to_index = {name: index for index, name in enumerate(names)}
    raw_name_edges = tuple(payload.get("predicted_edges", ()))
    raw_edges = _edge_indices(raw_name_edges, name_to_index, "predicted_edges")
    edgewise_raw = payload.get("edgewise_dist", {})
    edgewise = _edgewise_indices(edgewise_raw, name_to_index)

    raw_graph = nx.DiGraph()
    raw_graph.add_edges_from(raw_name_edges)
    threshold: float | None = None
    diagnostic = dict(policy=cycle_removal_policy, correction_applied=False,
                      raw_edges=len(raw_name_edges))
    if nx.is_directed_acyclic_graph(raw_graph):
        acyclic_name_edges = raw_name_edges
    else:
        if not edgewise_raw:
            raise UpstreamCausalOrderError(
                "cyclic upstream output has no edgewise distribution for cycle removal"
            )
        try:
            acyclic_name_edges, threshold, edge_weights = modules.cycle_remover.remove_cycles(
                list(raw_name_edges), edgewise_raw
            )
        except Exception as error:
            raise UpstreamCausalOrderError("upstream cycle removal failed") from error
        acyclic_name_edges = tuple(acyclic_name_edges)
        original_acyclic = nx.is_directed_acyclic_graph(nx.DiGraph(acyclic_name_edges))
        diagnostic.update(original_threshold=threshold, original_edges=len(acyclic_name_edges),
                          original_acyclic=original_acyclic)
        if not original_acyclic and cycle_removal_policy == "validated_threshold":
            original_edges = acyclic_name_edges
            if set(edge_weights) != set(raw_name_edges):
                raise UpstreamCausalOrderError("upstream edge weights do not cover all predicted edges")
            diagnostic["original_cycle"] = list(nx.find_cycle(nx.DiGraph(original_edges)))
            acyclic_name_edges, threshold = validated_cycle_threshold(edge_weights)
            diagnostic.update(correction_applied=True,
                              removed_edges=[edge for edge in original_edges if edge not in acyclic_name_edges])
    diagnostic.update(corrected_threshold=threshold, retained_edges=len(acyclic_name_edges),
                      acyclic=nx.is_directed_acyclic_graph(nx.DiGraph(acyclic_name_edges)))
    if cycle_removal_diagnostics is not None:
        cycle_removal_diagnostics.update(diagnostic)
    acyclic_edges = _edge_indices(
        acyclic_name_edges, name_to_index, "cycle-removed edges"
    )
    try:
        graph = graph_from_edges(len(names), directed=acyclic_edges)
    except ValueError as error:
        raise UpstreamCausalOrderError(
            "upstream cycle-removal output is not a DAG"
        ) from error

    if acyclic_name_edges:
        try:
            order_names = tuple(modules.metrics.topological_ordering(acyclic_name_edges))
        except Exception as error:
            raise UpstreamCausalOrderError(
                "upstream topological ordering failed"
            ) from error
        try:
            expert_order = tuple(name_to_index[name] for name in order_names)
        except KeyError as error:
            raise UpstreamCausalOrderError(
                "upstream topological ordering contains an unknown node"
            ) from error
    else:
        expert_order = ()
    connected = {node for edge in acyclic_edges for node in edge}
    isolated = tuple(node for node in range(len(names)) if node not in connected)
    subgroup_list = _subgroup_indices(
        payload.get("subgroup_list", ()), name_to_index
    )
    responses_raw = payload.get("subgraph_results", ())
    try:
        responses = tuple(responses_raw)
    except TypeError as error:
        raise UpstreamCausalOrderError(
            "upstream subgraph_results is not iterable"
        ) from error
    elapsed = time.perf_counter() - started
    return TripletExpertResult(
        node_names=names,
        raw_edges=raw_edges,
        graph=graph,
        edgewise_distributions=edgewise,
        subgroup_list=subgroup_list,
        subgroup_responses=responses,
        expert_order=expert_order,
        isolated_nodes=isolated,
        cycle_removal_threshold=threshold,
        upstream_revision=UPSTREAM_REVISION,
        config=normalized_config,
        runtime_seconds=elapsed,
    )


def orient_with_upstream_expert(
    node_names: Sequence[str],
    first: int,
    second: int,
    context: str,
    *,
    config: TripletConfig | None = None,
    upstream_root: str | Path | None = None,
) -> Edge:
    """Reuse the authors' strong-model pairwise tie-breaker for Algorithm 1."""

    names = normalize_node_names(node_names, minimum=2)
    if (
        isinstance(first, bool)
        or isinstance(second, bool)
        or not isinstance(first, int)
        or not isinstance(second, int)
    ):
        raise TypeError("fallback endpoints must be integers")
    if first == second or not 0 <= first < len(names) or not 0 <= second < len(names):
        raise ValueError("fallback endpoints must be distinct graph nodes")
    if not isinstance(context, str):
        raise TypeError("context must be a string")
    normalized_config = TripletConfig() if config is None else config
    if not isinstance(normalized_config, TripletConfig):
        raise TypeError("config must be TripletConfig")
    root = verified_upstream_root(upstream_root)
    modules = _load_upstream_modules(root)
    try:
        answer = modules.triplet._tiebreaker_query(
            names[first],
            names[second],
            list(names),
            context,
            model=normalized_config.expert_model,
            delay=normalized_config.delay,
        )
    except Exception as error:
        raise UpstreamCausalOrderError("upstream fallback expert query failed") from error
    if answer == "A_2_B":
        return first, second
    if answer == "B_2_A":
        return second, first
    if answer == "No_Conn":
        raise CausalOrderInfeasibleError(
            "fallback expert returned no relation for an existing PC adjacency"
        )
    raise UpstreamCausalOrderError(
        f"fallback expert returned unsupported answer {answer!r}"
    )


__all__ = [
    "UPSTREAM_REVISION",
    "orient_with_upstream_expert",
    "query_triplet_order",
    "verified_upstream_root",
]
