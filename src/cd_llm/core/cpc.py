"""Certified Tetrad Conservative-PC discovery.

Tetrad is the discovery backend.  This module additionally treats a canonical
CPDAG certificate as part of the backend contract: no malformed or
non-extendible partial graph is returned to downstream completion code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Literal, Sequence

import networkx as nx
import numpy as np

from .finite_support import (
    Endpoint,
    Graph,
    NonExtendableCPDAGError,
    PartialGraph,
    _causal_learn_partial,
    _compile_cpdag,
    graph_from_edges,
    iter_bits,
    py_tetrad_jar,
)


VariableType = Literal["continuous", "discrete"]
Resolution = Literal["identity", "canonicalized", "projected"]

PY_TETRAD_REVISION = "55de999adbb919ae7b2a39a5c0f5931a044395eb"


class CPCBackendError(RuntimeError):
    """Tetrad CPC could not be configured or executed."""


class CPCValidationError(RuntimeError):
    """A supposedly certified CPC result failed an internal invariant."""


@dataclass(frozen=True)
class CPCSettings:
    """Locked settings for one Tetrad CPC fit."""

    alpha: float
    variable_type: VariableType
    node_names: tuple[str, ...]
    independence_test: Literal["fisher_z", "g_square"]
    depth: int = -1
    stable_fas: bool = True
    allow_bidirected: bool = False
    singularity_lambda: float | None = None
    min_count: int | None = None
    cell_table_type: int | None = None


@dataclass(frozen=True)
class CPCBackendProvenance:
    """Version identifiers for the Java-backed discovery result."""

    backend: str = "tetrad-cpc"
    py_tetrad_version: str = "unknown"
    py_tetrad_revision: str = PY_TETRAD_REVISION
    tetrad_jar_sha256: str = "unknown"
    java_version: str = "unknown"


@dataclass(frozen=True)
class CPCRawResult:
    """Raw backend graph plus information that is lost during conversion."""

    graph: PartialGraph
    ambiguous_triples: tuple[tuple[int, int, int], ...] = ()
    provenance: CPCBackendProvenance = CPCBackendProvenance()


@dataclass(frozen=True)
class CPCDiagnostics:
    """Audit trail and validity certificate for one CPC output."""

    settings: CPCSettings
    provenance: CPCBackendProvenance
    ambiguous_triples: tuple[tuple[int, int, int], ...]
    raw_graph_sha256: str
    final_graph_sha256: str
    extension_sha256: str
    raw_edge_count: int
    final_edge_count: int
    resolution: Resolution
    normalization_applied: bool
    projection_applied: bool
    cyclic_sccs: tuple[tuple[int, ...], ...]
    reversed_preferences: tuple[tuple[int, int], ...]
    legal_endpoint_pairs: bool
    skeleton_preserved: bool
    acyclic_extension: bool
    canonical_fixed_point: bool
    certified: bool


@dataclass(frozen=True)
class CPCFit:
    """A certified CPDAG and its complete discovery/certification metadata."""

    graph: PartialGraph
    diagnostics: CPCDiagnostics


CPCBackendRunner = Callable[
    [np.ndarray, CPCSettings, tuple[str, ...]], CPCRawResult | PartialGraph
]


@dataclass(frozen=True)
class _Certification:
    graph: PartialGraph
    extension: Graph
    resolution: Resolution
    cyclic_sccs: tuple[tuple[int, ...], ...]
    reversed_preferences: tuple[tuple[int, int], ...]
    legal_endpoint_pairs: bool
    skeleton_preserved: bool
    canonical_fixed_point: bool


def fit_cpc(
    data: np.ndarray,
    alpha: float = 0.05,
    *,
    variable_type: VariableType = "continuous",
    node_names: Sequence[str] | None = None,
    backend_runner: CPCBackendRunner | None = None,
) -> CPCFit:
    """Fit Tetrad CPC and return only a certified canonical CPDAG.

    ``backend_runner`` is an explicit dependency-injection seam for tests and
    controlled external runners.  Production calls use the pinned py-tetrad
    adapter.
    """

    values, settings = _validated_input(data, alpha, variable_type, node_names)
    runner = _run_tetrad_cpc if backend_runner is None else backend_runner
    try:
        raw_result = runner(values, settings, settings.node_names)
    except CPCBackendError:
        raise
    except Exception as error:
        raise CPCBackendError(
            f"Tetrad CPC failed: {type(error).__name__}: {error}"
        ) from error
    if isinstance(raw_result, PartialGraph):
        raw_result = CPCRawResult(raw_result)
    if not isinstance(raw_result, CPCRawResult):
        raise CPCBackendError(
            "CPC backend runner must return CPCRawResult or PartialGraph"
        )
    raw = raw_result.graph
    if raw.kind != "cpdag" or raw.node_count != values.shape[1]:
        raise CPCBackendError(
            "Tetrad CPC returned a graph with the wrong kind or node count"
        )

    certification = _certify_or_project_cpdag(raw)
    final = certification.graph
    extension = certification.extension
    diagnostics = CPCDiagnostics(
        settings=settings,
        provenance=raw_result.provenance,
        ambiguous_triples=_normalize_ambiguous_triples(
            raw_result.ambiguous_triples, raw.node_count
        ),
        raw_graph_sha256=_partial_graph_sha256(raw),
        final_graph_sha256=_partial_graph_sha256(final),
        extension_sha256=_dag_sha256(extension),
        raw_edge_count=_edge_count(raw),
        final_edge_count=_edge_count(final),
        resolution=certification.resolution,
        normalization_applied=certification.resolution == "canonicalized",
        projection_applied=certification.resolution == "projected",
        cyclic_sccs=certification.cyclic_sccs,
        reversed_preferences=certification.reversed_preferences,
        legal_endpoint_pairs=certification.legal_endpoint_pairs,
        skeleton_preserved=certification.skeleton_preserved,
        acyclic_extension=True,
        canonical_fixed_point=certification.canonical_fixed_point,
        certified=(
            certification.legal_endpoint_pairs
            and certification.skeleton_preserved
            and certification.canonical_fixed_point
        ),
    )
    if not diagnostics.certified:
        raise CPCValidationError("CPC result did not satisfy its validity certificate")
    return CPCFit(final, diagnostics)


def learn_cpc(
    data: np.ndarray,
    alpha: float = 0.05,
    *,
    variable_type: VariableType = "continuous",
    node_names: Sequence[str] | None = None,
) -> PartialGraph:
    """Return the certified graph from :func:`fit_cpc`."""

    return fit_cpc(
        data,
        alpha,
        variable_type=variable_type,
        node_names=node_names,
    ).graph


def learn_pc(data: np.ndarray, alpha: float = 0.05) -> PartialGraph:
    """Compatibility name for continuous-data CPC."""

    return learn_cpc(data, alpha, variable_type="continuous")


def certify_cpdag(partial: PartialGraph) -> PartialGraph:
    """Canonicalize/project a PDAG and return its certified CPDAG.

    This helper is useful when validating stored backend artifacts.  Discovery
    calls should use :func:`fit_cpc` so that the resolution is recorded.
    """

    return _certify_or_project_cpdag(partial).graph


def _validated_input(
    data: np.ndarray,
    alpha: float,
    variable_type: VariableType,
    node_names: Sequence[str] | None,
) -> tuple[np.ndarray, CPCSettings]:
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("CPC data must be a nonempty two-dimensional sample matrix")
    if not np.isfinite(values).all():
        raise ValueError("CPC data cannot contain NaN or infinite values")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("CPC alpha must lie strictly between zero and one")
    if variable_type not in ("continuous", "discrete"):
        raise ValueError("CPC variable_type must be 'continuous' or 'discrete'")
    names = (
        tuple(f"X{index}" for index in range(values.shape[1]))
        if node_names is None
        else tuple(str(name) for name in node_names)
    )
    if len(names) != values.shape[1] or len(set(names)) != len(names) or any(
        not name for name in names
    ):
        raise ValueError("CPC node names must be nonempty, unique, and match the columns")

    if variable_type == "continuous":
        normalized = np.asarray(values, dtype=np.float64)
        settings = CPCSettings(
            alpha=float(alpha),
            variable_type=variable_type,
            node_names=names,
            independence_test="fisher_z",
            singularity_lambda=0.0,
        )
    else:
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise ValueError("discrete CPC data must contain integer-valued categories")
        normalized = np.asarray(rounded, dtype=np.int64)
        settings = CPCSettings(
            alpha=float(alpha),
            variable_type=variable_type,
            node_names=names,
            independence_test="g_square",
            min_count=1,
            cell_table_type=1,
        )
    return normalized, settings


def _run_tetrad_cpc(
    data: np.ndarray,
    settings: CPCSettings,
    node_names: tuple[str, ...],
) -> CPCRawResult:
    try:
        import pandas as pd
        from pytetrad.tools.TetradSearch import TetradSearch
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        raise CPCBackendError(
            "Tetrad CPC requires pinned py-tetrad, JPype, Python >=3.12, and Java >=21"
        ) from error

    frame = pd.DataFrame(data, columns=list(node_names))
    search = TetradSearch(frame)
    if settings.variable_type == "continuous":
        search.use_fisher_z(
            alpha=settings.alpha,
            singularity_lambda=float(settings.singularity_lambda or 0.0),
        )
    else:
        search.use_g_square(
            min_count=int(settings.min_count or 1),
            alpha=settings.alpha,
            cell_table_type=int(settings.cell_table_type or 1),
        )
    search.run_cpc(
        depth=settings.depth,
        stable_fas=settings.stable_fas,
        allow_bidirected=settings.allow_bidirected,
    )
    java_graph = search.get_java()
    if java_graph is None:
        raise CPCBackendError("Tetrad CPC returned no graph")
    observed_names = tuple(str(name) for name in java_graph.getNodeNames())
    if observed_names != node_names:
        raise CPCBackendError(
            f"Tetrad changed the node order: expected {node_names}, got {observed_names}"
        )
    graph = _causal_learn_partial(search.get_causal_learn(), "cpdag")
    ambiguous = _tetrad_ambiguous_triples(java_graph, node_names)
    return CPCRawResult(graph, ambiguous, _tetrad_provenance())


def _tetrad_ambiguous_triples(
    java_graph: object,
    node_names: tuple[str, ...],
) -> tuple[tuple[int, int, int], ...]:
    index = {name: position for position, name in enumerate(node_names)}
    triples: set[tuple[int, int, int]] = set()
    for triple in java_graph.getAmbiguousTriples():
        first = index[str(triple.getX().getName())]
        middle = index[str(triple.getY().getName())]
        last = index[str(triple.getZ().getName())]
        if first > last:
            first, last = last, first
        triples.add((first, middle, last))
    return tuple(sorted(triples))


def _tetrad_provenance() -> CPCBackendProvenance:
    try:
        package_version = version("py-tetrad")
    except PackageNotFoundError:
        package_version = "unknown"
    try:
        import java.lang as java_lang

        java_version = str(java_lang.System.getProperty("java.version"))
    except Exception:
        java_version = "unknown"
    jar = py_tetrad_jar()
    return CPCBackendProvenance(
        py_tetrad_version=package_version,
        tetrad_jar_sha256=_file_sha256(jar) if jar is not None else "unknown",
        java_version=java_version,
    )


@lru_cache(maxsize=4)
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _certify_or_project_cpdag(raw: PartialGraph) -> _Certification:
    if raw.kind != "cpdag":
        raise ValueError("only CPDAG/PDAG outputs can be certified")
    raw_legal = _has_legal_cpdag_endpoint_pairs(raw)
    cyclic_sccs = _cyclic_preference_sccs(raw)
    try:
        if not raw_legal:
            raise NonExtendableCPDAGError("illegal CPDAG endpoint pair")
        raw_extension = _deterministic_cpdag_extension(raw)
        canonical = _dag_to_cpdag(raw_extension)
        resolution: Resolution = "identity" if canonical == raw else "canonicalized"
        reversed_preferences: tuple[tuple[int, int], ...] = ()
    except (NonExtendableCPDAGError, RuntimeError, ValueError):
        raw_extension, reversed_preferences = _same_skeleton_preference_dag(raw)
        canonical = _dag_to_cpdag(raw_extension)
        resolution = "projected"

    skeleton_preserved = _skeleton(raw) == _skeleton(canonical)
    legal = _has_legal_cpdag_endpoint_pairs(canonical)
    try:
        final_extension = _deterministic_cpdag_extension(canonical)
    except (NonExtendableCPDAGError, RuntimeError, ValueError) as error:
        raise CPCValidationError("canonical CPC output is not extendible") from error
    fixed_point = _dag_to_cpdag(final_extension) == canonical
    if not legal or not skeleton_preserved or not fixed_point:
        raise CPCValidationError(
            "CPC canonicalization violated endpoint, skeleton, or fixed-point invariants"
        )
    return _Certification(
        graph=canonical,
        extension=final_extension,
        resolution=resolution,
        cyclic_sccs=cyclic_sccs,
        reversed_preferences=reversed_preferences,
        legal_endpoint_pairs=legal,
        skeleton_preserved=skeleton_preserved,
        canonical_fixed_point=fixed_point,
    )


def _deterministic_cpdag_extension(partial: PartialGraph) -> Graph:
    compiled = _compile_cpdag(partial)
    rows = list(compiled.directed_out)
    active = compiled.all_nodes
    while active:
        eligible: list[int] = []
        for node in iter_bits(active):
            if compiled.directed_out[node] & active:
                continue
            neighbors = compiled.undirected[node] & active
            adjacent_to_node = (compiled.directed_in[node] & active) | neighbors
            if all(
                not (
                    (adjacent_to_node & ~(1 << neighbor))
                    & ~compiled.adjacent[neighbor]
                )
                for neighbor in iter_bits(neighbors)
            ):
                eligible.append(node)
        if not eligible:
            raise NonExtendableCPDAGError("CPDAG has no consistent DAG extension")
        sink = min(eligible)
        for neighbor in iter_bits(compiled.undirected[sink] & active):
            rows[neighbor] |= 1 << sink
        active &= ~(1 << sink)
    return Graph(tuple(rows))


def _dag_to_cpdag(graph: Graph) -> PartialGraph:
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    nodes = [GraphNode(f"X{index}") for index in range(graph.node_count)]
    dag = Dag(nodes)
    for source, row in enumerate(graph.directed):
        for target in iter_bits(row):
            dag.add_directed_edge(nodes[source], nodes[target])
    return _causal_learn_partial(dag2cpdag(dag), "cpdag")


def _same_skeleton_preference_dag(
    partial: PartialGraph,
) -> tuple[Graph, tuple[tuple[int, int], ...]]:
    preferences = _directed_preferences(partial)
    preference_graph = nx.DiGraph()
    preference_graph.add_nodes_from(range(partial.node_count))
    preference_graph.add_edges_from(preferences)
    components = [
        tuple(sorted(component))
        for component in nx.strongly_connected_components(preference_graph)
    ]
    component_by_node = {
        node: component_index
        for component_index, component in enumerate(components)
        for node in component
    }
    condensation = nx.DiGraph()
    condensation.add_nodes_from(range(len(components)))
    condensation.add_edges_from(
        (component_by_node[source], component_by_node[target])
        for source, target in preferences
        if component_by_node[source] != component_by_node[target]
    )
    component_order = nx.lexicographical_topological_sort(
        condensation, key=lambda component_index: components[component_index]
    )
    order: list[int] = []
    for component_index in component_order:
        order.extend(_order_preference_component(components[component_index], preferences))
    position = {node: index for index, node in enumerate(order)}
    directed = [
        (first, second) if position[first] < position[second] else (second, first)
        for first, second in _skeleton(partial)
    ]
    reversed_preferences = tuple(
        sorted(
            (source, target)
            for source, target in preferences
            if position[source] > position[target]
        )
    )
    return graph_from_edges(partial.node_count, directed), reversed_preferences


def _order_preference_component(
    component: tuple[int, ...], preferences: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    remaining = set(component)
    edges = {
        (source, target)
        for source, target in preferences
        if source in remaining and target in remaining
    }
    left: list[int] = []
    right: list[int] = []
    while remaining:
        indegree = {
            node: sum(source in remaining for source, target in edges if target == node)
            for node in remaining
        }
        outdegree = {
            node: sum(target in remaining for source, target in edges if source == node)
            for node in remaining
        }
        sources = [node for node in remaining if indegree[node] == 0]
        if sources:
            selected = min(sources)
            left.append(selected)
        else:
            sinks = [node for node in remaining if outdegree[node] == 0]
            if sinks:
                selected = max(sinks)
                right.append(selected)
            else:
                selected = min(
                    remaining,
                    key=lambda node: (-(outdegree[node] - indegree[node]), node),
                )
                left.append(selected)
        remaining.remove(selected)
    return tuple(left + list(reversed(right)))


def _has_legal_cpdag_endpoint_pairs(partial: PartialGraph) -> bool:
    legal = {
        (Endpoint.NULL, Endpoint.NULL),
        (Endpoint.TAIL, Endpoint.TAIL),
        (Endpoint.TAIL, Endpoint.ARROW),
        (Endpoint.ARROW, Endpoint.TAIL),
    }
    return all(
        (partial.mark(first, second), partial.mark(second, first)) in legal
        for first in range(partial.node_count)
        for second in range(first + 1, partial.node_count)
    )


def _directed_preferences(partial: PartialGraph) -> tuple[tuple[int, int], ...]:
    directed: list[tuple[int, int]] = []
    for first in range(partial.node_count):
        for second in range(first + 1, partial.node_count):
            marks = (partial.mark(first, second), partial.mark(second, first))
            if marks == (Endpoint.TAIL, Endpoint.ARROW):
                directed.append((first, second))
            elif marks == (Endpoint.ARROW, Endpoint.TAIL):
                directed.append((second, first))
    return tuple(directed)


def _cyclic_preference_sccs(partial: PartialGraph) -> tuple[tuple[int, ...], ...]:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(partial.node_count))
    graph.add_edges_from(_directed_preferences(partial))
    return tuple(
        sorted(
            tuple(sorted(component))
            for component in nx.strongly_connected_components(graph)
            if len(component) > 1
        )
    )


def _skeleton(partial: PartialGraph) -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, second)
        for first in range(partial.node_count)
        for second in range(first + 1, partial.node_count)
        if partial.mark(first, second) != Endpoint.NULL
    )


def _edge_count(partial: PartialGraph) -> int:
    return len(_skeleton(partial))


def _normalize_ambiguous_triples(
    triples: Sequence[tuple[int, int, int]], node_count: int
) -> tuple[tuple[int, int, int], ...]:
    normalized: set[tuple[int, int, int]] = set()
    for first, middle, last in triples:
        if any(node < 0 or node >= node_count for node in (first, middle, last)):
            raise CPCBackendError("Tetrad returned an out-of-range ambiguous triple")
        if len({first, middle, last}) != 3:
            raise CPCBackendError("Tetrad returned a degenerate ambiguous triple")
        if first > last:
            first, last = last, first
        normalized.add((first, middle, last))
    return tuple(sorted(normalized))


def _partial_graph_sha256(partial: PartialGraph) -> str:
    digest = hashlib.sha256()
    digest.update(partial.kind.encode("ascii"))
    digest.update(int(partial.node_count).to_bytes(8, "little"))
    digest.update(bytes(partial.marks))
    return digest.hexdigest()


def _dag_sha256(graph: Graph) -> str:
    digest = hashlib.sha256()
    digest.update(int(graph.node_count).to_bytes(8, "little"))
    width = max(1, (graph.node_count + 7) // 8)
    for row in graph.directed:
        digest.update(int(row).to_bytes(width, "little"))
    return digest.hexdigest()


__all__ = [
    "CPCBackendError",
    "CPCBackendProvenance",
    "CPCDiagnostics",
    "CPCFit",
    "CPCRawResult",
    "CPCSettings",
    "CPCValidationError",
    "certify_cpdag",
    "fit_cpc",
    "learn_cpc",
    "learn_pc",
]
