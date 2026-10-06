"""Core CEG-Fuse implementation used by the revised synthetic experiment.

The active paper defines one structural distribution per assumption regime.
Complete graphs therefore retain their endpoint semantics, discovery outputs are
completed inside their own regime, and evidence is normalized separately in
every paired bootstrap replicate before those distributions are averaged.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ceg-fuse-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ceg-fuse-cache")

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


Regime = Literal["dag", "mag"]
ClaimGroup = Literal["I", "II", "III", "IV"]
EvidenceTreatment = Literal["soft_evidence", "hard_constraint"]
DEFAULT_SOFT_CONFIDENCE_CAP = 0.99


class HardClaimInfeasibleError(ValueError):
    """The requested hard-claim event has no support under the target family."""


@dataclass(frozen=True)
class ResolvedEvidenceConfidence:
    """Auditable reported and inference-time confidence semantics for one claim."""

    reported_confidence: float
    effective_confidence: float
    treatment: EvidenceTreatment


def validate_soft_confidence_cap(soft_confidence_cap: float) -> float:
    """Validate and return the symmetric cap used for soft evidence."""

    value = float(soft_confidence_cap)
    if not math.isfinite(value) or not 0.5 <= value < 1.0:
        raise ValueError("soft_confidence_cap must lie in [0.5, 1)")
    return value


def resolve_evidence_confidence(
    reported_confidence: float,
    treatment: EvidenceTreatment | None = None,
    soft_confidence_cap: float = DEFAULT_SOFT_CONFIDENCE_CAP,
) -> ResolvedEvidenceConfidence:
    """Resolve an explicit hard/soft treatment without changing the raw report.

    Soft evidence is symmetrically clipped so neither a report of zero nor one
    removes graphs from the target support. A report of exactly one is
    intentionally ambiguous unless its treatment is explicitly supplied.
    """

    reported = float(reported_confidence)
    if not math.isfinite(reported) or not 0.0 <= reported <= 1.0:
        raise ValueError("reported evidence confidence must lie in [0, 1]")
    cap = validate_soft_confidence_cap(soft_confidence_cap)
    if treatment is None:
        if reported == 1.0:
            raise ValueError(
                "reported confidence 1.0 requires explicit treatment="
                "'soft_evidence' or treatment='hard_constraint'; legacy soft-only "
                "callers must use confidence strictly below one"
            )
        resolved_treatment: EvidenceTreatment = "soft_evidence"
    elif treatment in ("soft_evidence", "hard_constraint"):
        resolved_treatment = treatment
    else:
        raise ValueError(
            "evidence treatment must be 'soft_evidence' or 'hard_constraint'"
        )
    if resolved_treatment == "hard_constraint":
        if reported != 1.0:
            raise ValueError(
                "hard_constraint treatment requires reported confidence exactly 1.0"
            )
        effective = 1.0
    else:
        effective = min(max(reported, 1.0 - cap), cap)
    return ResolvedEvidenceConfidence(reported, effective, resolved_treatment)


class Endpoint(IntEnum):
    NULL = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3


def iter_bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _rows_are_acyclic(rows: tuple[int, ...]) -> bool:
    indegree = [0] * len(rows)
    for row in rows:
        for target in iter_bits(row):
            indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in iter_bits(rows[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return visited == len(rows)


@dataclass(frozen=True, order=True)
class Graph:
    """A complete DAG or MAG represented by directed and symmetric edge rows."""

    directed: tuple[int, ...]
    bidirected: tuple[int, ...] = ()
    undirected: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        node_count = len(self.directed)
        if not self.bidirected:
            object.__setattr__(self, "bidirected", (0,) * node_count)
        if not self.undirected:
            object.__setattr__(self, "undirected", (0,) * node_count)
        if len(self.bidirected) != node_count or len(self.undirected) != node_count:
            raise ValueError("all graph row collections must have the same length")
        limit = 1 << node_count
        incoming = [0] * node_count
        for node in range(node_count):
            rows = (self.directed[node], self.bidirected[node], self.undirected[node])
            if any(row < 0 or row >= limit or row & (1 << node) for row in rows):
                raise ValueError("invalid graph row or self-edge")
            for target in iter_bits(self.directed[node]):
                incoming[target] |= 1 << node
        for node in range(node_count):
            for other in iter_bits(self.bidirected[node]):
                if not self.bidirected[other] & (1 << node):
                    raise ValueError("bidirected rows must be symmetric")
            for other in iter_bits(self.undirected[node]):
                if not self.undirected[other] & (1 << node):
                    raise ValueError("undirected rows must be symmetric")
            directed_adjacencies = self.directed[node] | incoming[node]
            if (
                self.directed[node] & incoming[node]
                or directed_adjacencies & self.bidirected[node]
                or directed_adjacencies & self.undirected[node]
                or self.bidirected[node] & self.undirected[node]
            ):
                raise ValueError("multiple edges between one node pair are unsupported")
        if not _rows_are_acyclic(self.directed):
            raise ValueError("directed component contains a cycle")

    @property
    def node_count(self) -> int:
        return len(self.directed)

    @property
    def regime(self) -> Regime:
        return "mag" if any(self.bidirected) or any(self.undirected) else "dag"


@dataclass(frozen=True, order=True)
class PartialGraph:
    """A CPDAG or PAG with the endpoint mark stored at each matrix row."""

    node_count: int
    marks: tuple[int, ...]
    kind: Literal["cpdag", "pag"]

    def __post_init__(self) -> None:
        if len(self.marks) != self.node_count * self.node_count:
            raise ValueError("endpoint matrix has the wrong size")
        allowed = {int(endpoint) for endpoint in Endpoint}
        for first in range(self.node_count):
            if self.mark(first, first) != Endpoint.NULL:
                raise ValueError("partial graphs cannot contain self-edges")
            for second in range(first + 1, self.node_count):
                left = self.mark(first, second)
                right = self.mark(second, first)
                if int(left) not in allowed or int(right) not in allowed:
                    raise ValueError("unknown endpoint mark")
                if (left == Endpoint.NULL) != (right == Endpoint.NULL):
                    raise ValueError("an adjacency must have two endpoint marks")

    def mark(self, node: int, other: int) -> Endpoint:
        return Endpoint(self.marks[node * self.node_count + other])

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        kind: Literal["cpdag", "pag"],
    ) -> "PartialGraph":
        square = np.asarray(matrix, dtype=int)
        if square.ndim != 2 or square.shape[0] != square.shape[1]:
            raise ValueError("endpoint matrix must be square")
        return cls(square.shape[0], tuple(int(value) for value in square.ravel()), kind)


GraphOutput = Graph | PartialGraph


class CompletionError(RuntimeError):
    """A partial discovery output has no compatible complete-graph extension."""


class NonExtendableCPDAGError(CompletionError):
    """A purported CPDAG has no consistent DAG extension."""


class NonCompletablePAGError(CompletionError):
    """A purported PAG produced no compatible MAG completion."""


@dataclass(frozen=True)
class CompletionSamplingDiagnostics:
    """Outcome counters for one bounded completion-sampling call."""

    requested_limit: int
    returned_count: int
    anchor_source: Literal["point", "causal-learn", "tetrad", "local", "none"]
    random_attempt_count: int
    unique_random_count: int
    duplicate_random_count: int
    rejected_random_count: int
    stopped_for_stagnation: bool


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2s(digest_size=8)
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") & 0xFFFFFFFF


def graph_from_edges(
    node_count: int,
    directed: Iterable[tuple[int, int]] = (),
    bidirected: Iterable[tuple[int, int]] = (),
    undirected: Iterable[tuple[int, int]] = (),
) -> Graph:
    directed_rows = [0] * node_count
    bidirected_rows = [0] * node_count
    undirected_rows = [0] * node_count
    for source, target in directed:
        if source == target:
            raise ValueError("self-edges are not allowed")
        directed_rows[source] |= 1 << target
    for first, second in bidirected:
        if first == second:
            raise ValueError("self-edges are not allowed")
        bidirected_rows[first] |= 1 << second
        bidirected_rows[second] |= 1 << first
    for first, second in undirected:
        if first == second:
            raise ValueError("self-edges are not allowed")
        undirected_rows[first] |= 1 << second
        undirected_rows[second] |= 1 << first
    return Graph(tuple(directed_rows), tuple(bidirected_rows), tuple(undirected_rows))


def edge_count(graph: Graph) -> int:
    return (
        sum(row.bit_count() for row in graph.directed)
        + sum(row.bit_count() for row in graph.bidirected) // 2
        + sum(row.bit_count() for row in graph.undirected) // 2
    )


def has_edge(graph: Graph, source: int, target: int) -> bool:
    return bool(graph.directed[source] & (1 << target))


def has_bidirected(graph: Graph, first: int, second: int) -> bool:
    return bool(graph.bidirected[first] & (1 << second))


def has_undirected(graph: Graph, first: int, second: int) -> bool:
    return bool(graph.undirected[first] & (1 << second))


def has_adjacency(graph: Graph, first: int, second: int) -> bool:
    return (
        has_edge(graph, first, second)
        or has_edge(graph, second, first)
        or has_bidirected(graph, first, second)
        or has_undirected(graph, first, second)
    )


def edge_marks(graph: Graph, first: int, second: int) -> tuple[Endpoint, Endpoint]:
    if has_edge(graph, first, second):
        return Endpoint.TAIL, Endpoint.ARROW
    if has_edge(graph, second, first):
        return Endpoint.ARROW, Endpoint.TAIL
    if has_bidirected(graph, first, second):
        return Endpoint.ARROW, Endpoint.ARROW
    if has_undirected(graph, first, second):
        return Endpoint.TAIL, Endpoint.TAIL
    return Endpoint.NULL, Endpoint.NULL


def topological_order(graph: Graph) -> tuple[int, ...]:
    indegree = [0] * graph.node_count
    for row in graph.directed:
        for target in iter_bits(row):
            indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        node = ready.pop()
        order.append(node)
        for child in iter_bits(graph.directed[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != graph.node_count:
        raise ValueError("graph contains a directed cycle")
    return tuple(order)


def is_acyclic(graph: Graph) -> bool:
    return _rows_are_acyclic(graph.directed)


def transitive_closure(graph: Graph) -> tuple[int, ...]:
    reachability = list(graph.directed)
    for node in reversed(topological_order(graph)):
        for child in iter_bits(graph.directed[node]):
            reachability[node] |= reachability[child]
    return tuple(reachability)


def is_ancestral(graph: Graph) -> bool:
    if not is_acyclic(graph):
        return False
    closure = transitive_closure(graph)
    for first in range(graph.node_count):
        for second in range(first + 1, graph.node_count):
            if has_bidirected(graph, first, second) and (
                closure[first] & (1 << second) or closure[second] & (1 << first)
            ):
                return False
    for node in range(graph.node_count):
        if graph.undirected[node]:
            has_arrowhead = bool(graph.bidirected[node]) or any(
                has_edge(graph, other, node) for other in range(graph.node_count)
            )
            if has_arrowhead:
                return False
    return True


def _has_inducing_path(graph: Graph, first: int, second: int) -> bool:
    """Return whether an observed ancestral graph has an inducing path."""

    closure = transitive_closure(graph)

    def is_arrow_at(other: int, node: int) -> bool:
        return edge_marks(graph, node, other)[0] == Endpoint.ARROW

    def search(path: list[int]) -> bool:
        current = path[-1]
        if current == second:
            return True
        for neighbor in _underlying_neighbors(graph, current):
            if neighbor in path:
                continue
            if len(path) >= 2:
                previous = path[-2]
                collider = is_arrow_at(previous, current) and is_arrow_at(neighbor, current)
                ancestor_endpoint = bool(
                    closure[current] & (1 << first)
                    or closure[current] & (1 << second)
                )
                if not collider or not ancestor_endpoint:
                    continue
            if search(path + [neighbor]):
                return True
        return False

    return search([first])


def is_maximal_ancestral(graph: Graph) -> bool:
    if not is_ancestral(graph):
        return False
    return all(
        not _has_inducing_path(graph, first, second)
        for first in range(graph.node_count)
        for second in range(first + 1, graph.node_count)
        if not has_adjacency(graph, first, second)
    )


def graph_to_adjacency(graph: Graph) -> np.ndarray:
    adjacency = np.zeros((graph.node_count, graph.node_count), dtype=np.int8)
    for source, row in enumerate(graph.directed):
        for target in iter_bits(row):
            adjacency[source, target] = 1
    return adjacency


def generate_random_dag(
    node_count: int,
    expected_degree: float,
    seed: int,
) -> tuple[Graph, np.ndarray]:
    """Use the paper's ordered Erdos--Renyi DAG and coefficient generator."""

    rng = np.random.default_rng(seed)
    order = rng.permutation(node_count)
    probability = expected_degree / (node_count - 1)
    edges: list[tuple[int, int]] = []
    for earlier in range(node_count):
        for later in range(earlier + 1, node_count):
            if rng.random() < probability:
                edges.append((int(order[earlier]), int(order[later])))
    graph = graph_from_edges(node_count, edges)
    coefficients = np.zeros((node_count, node_count), dtype=float)
    for source, target in edges:
        magnitude = rng.uniform(0.5, 2.0)
        coefficients[source, target] = magnitude if rng.random() < 0.5 else -magnitude
    return graph, coefficients


def simulate_linear_sem(
    graph: Graph,
    coefficients: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((sample_size, graph.node_count), dtype=float)
    noise = rng.normal(size=data.shape)
    for node in topological_order(graph):
        parents = [
            parent for parent in range(graph.node_count) if has_edge(graph, parent, node)
        ]
        data[:, node] = noise[:, node]
        if parents:
            data[:, node] += data[:, parents] @ coefficients[parents, node]
    return data


def standardize(data: np.ndarray) -> np.ndarray:
    centered = np.asarray(data, dtype=float) - np.mean(data, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=0, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("constant column in data")
    return centered / scale


def _underlying_neighbors(graph: Graph, node: int) -> tuple[int, ...]:
    return tuple(
        other
        for other in range(graph.node_count)
        if other != node and has_adjacency(graph, node, other)
    )


def _exists_inducing_path(
    dag: Graph,
    first: int,
    second: int,
    observed_count: int,
    closure: tuple[int, ...],
) -> bool:
    """Check the inducing-path criterion in a full DAG with hidden roots."""

    def search(path: list[int]) -> bool:
        current = path[-1]
        if current == second:
            for index in range(1, len(path) - 1):
                previous, node, following = path[index - 1 : index + 2]
                collider = has_edge(dag, previous, node) and has_edge(dag, following, node)
                if collider:
                    if not (
                        closure[node] & (1 << first)
                        or closure[node] & (1 << second)
                    ):
                        return False
                elif node < observed_count:
                    return False
            return True
        for neighbor in _underlying_neighbors(dag, current):
            if neighbor not in path and search(path + [neighbor]):
                return True
        return False

    return search([first])


def latent_projection(dag: Graph, observed_count: int) -> Graph:
    """Return the maximal ancestral projection over the observed prefix."""

    if observed_count > dag.node_count:
        raise ValueError("observed_count exceeds graph size")
    closure = transitive_closure(dag)
    directed: list[tuple[int, int]] = []
    bidirected: list[tuple[int, int]] = []
    for first in range(observed_count):
        for second in range(first + 1, observed_count):
            if not _exists_inducing_path(dag, first, second, observed_count, closure):
                continue
            first_ancestor = bool(closure[first] & (1 << second))
            second_ancestor = bool(closure[second] & (1 << first))
            if first_ancestor:
                directed.append((first, second))
            elif second_ancestor:
                directed.append((second, first))
            else:
                bidirected.append((first, second))
    projection = graph_from_edges(observed_count, directed, bidirected)
    if not is_ancestral(projection):
        raise RuntimeError("latent projection is not ancestral")
    return projection


def generate_latent_model(
    node_count: int,
    expected_degree: float,
    seed: int,
) -> tuple[Graph, Graph, np.ndarray]:
    """Generate two hidden roots and a MAG target as specified by the paper."""

    observed, observed_coefficients = generate_random_dag(
        node_count, expected_degree, stable_seed(seed, "observed")
    )
    rng = np.random.default_rng(stable_seed(seed, "latent"))
    observed_edges = [
        (source, target)
        for source, row in enumerate(observed.directed)
        for target in iter_bits(row)
    ]
    for _ in range(10_000):
        hidden_edges: list[tuple[int, int]] = []
        confounded_pairs: set[tuple[int, int]] = set()
        for hidden in range(node_count, node_count + 2):
            child_count = int(rng.integers(2, 4))
            children = rng.choice(node_count, size=child_count, replace=False)
            hidden_edges.extend((hidden, int(child)) for child in children)
            confounded_pairs.update(
                (min(int(first), int(second)), max(int(first), int(second)))
                for first, second in itertools.combinations(children, 2)
            )
        full = graph_from_edges(node_count + 2, observed_edges + hidden_edges)
        try:
            target = graph_from_edges(node_count, observed_edges, sorted(confounded_pairs))
        except ValueError:
            continue
        if any(target.bidirected) and is_maximal_ancestral(target):
            break
    else:
        raise RuntimeError("failed to generate a latent model with bidirected structure")
    coefficients = np.zeros((node_count + 2, node_count + 2), dtype=float)
    coefficients[:node_count, :node_count] = observed_coefficients
    for source, child in hidden_edges:
        magnitude = rng.uniform(0.5, 2.0)
        coefficients[source, child] = magnitude if rng.random() < 0.5 else -magnitude
    return target, full, coefficients


@dataclass(frozen=True)
class _CompiledCPDAG:
    """Bit-mask representation shared by every Dor--Tarsi draw."""

    node_count: int
    all_nodes: int
    directed_out: tuple[int, ...]
    directed_in: tuple[int, ...]
    undirected: tuple[int, ...]
    adjacent: tuple[int, ...]


def _compile_cpdag(partial: PartialGraph) -> _CompiledCPDAG:
    if partial.kind != "cpdag":
        raise ValueError("DAG completion requires a CPDAG")
    node_count = partial.node_count
    directed_out = [0] * node_count
    directed_in = [0] * node_count
    undirected = [0] * node_count
    adjacent = [0] * node_count
    marks_matrix = partial.marks
    for first in range(node_count):
        for second in range(first + 1, node_count):
            marks = (
                Endpoint(marks_matrix[first * node_count + second]),
                Endpoint(marks_matrix[second * node_count + first]),
            )
            if marks == (Endpoint.TAIL, Endpoint.ARROW):
                directed_out[first] |= 1 << second
                directed_in[second] |= 1 << first
            elif marks == (Endpoint.ARROW, Endpoint.TAIL):
                directed_out[second] |= 1 << first
                directed_in[first] |= 1 << second
            elif marks == (Endpoint.TAIL, Endpoint.TAIL):
                undirected[first] |= 1 << second
                undirected[second] |= 1 << first
            elif marks != (Endpoint.NULL, Endpoint.NULL):
                raise RuntimeError(f"unsupported CPDAG endpoint pair: {marks}")
            if marks != (Endpoint.NULL, Endpoint.NULL):
                adjacent[first] |= 1 << second
                adjacent[second] |= 1 << first
    return _CompiledCPDAG(
        node_count=node_count,
        all_nodes=(1 << node_count) - 1,
        directed_out=tuple(directed_out),
        directed_in=tuple(directed_in),
        undirected=tuple(undirected),
        adjacent=tuple(adjacent),
    )


def _random_compiled_cpdag_extension(
    compiled: _CompiledCPDAG,
    rng: np.random.Generator,
) -> Graph:
    rows = list(compiled.directed_out)
    active = compiled.all_nodes
    while active:
        eligible: list[int] = []
        for node in iter_bits(active):
            if compiled.directed_out[node] & active:
                continue
            neighbors = compiled.undirected[node] & active
            adjacent_to_node = (compiled.directed_in[node] & active) | neighbors
            for neighbor in iter_bits(neighbors):
                required = adjacent_to_node & ~(1 << neighbor)
                if required & ~compiled.adjacent[neighbor]:
                    break
            else:
                eligible.append(node)
        if not eligible:
            raise NonExtendableCPDAGError("CPDAG has no consistent DAG extension")
        sink = int(rng.choice(eligible))
        for neighbor in iter_bits(compiled.undirected[sink] & active):
            rows[neighbor] |= 1 << sink
        active &= ~(1 << sink)
    return Graph(tuple(rows))


def _random_cpdag_extension(partial: PartialGraph, rng: np.random.Generator) -> Graph:
    """Draw one extension; bulk callers should compile the CPDAG only once."""

    return _random_compiled_cpdag_extension(_compile_cpdag(partial), rng)


def _is_compatible_compiled_cpdag(
    compiled: _CompiledCPDAG,
    graph: Graph,
) -> bool:
    if (
        graph.node_count != compiled.node_count
        or any(graph.bidirected)
        or any(graph.undirected)
    ):
        return False
    incoming = [0] * compiled.node_count
    for source, row in enumerate(graph.directed):
        if compiled.directed_out[source] & ~row:
            return False
        for target in iter_bits(row):
            incoming[target] |= 1 << source
    return all(
        graph.directed[node] | incoming[node] == compiled.adjacent[node]
        for node in range(compiled.node_count)
    )


def _pag_anchor_completion(partial: PartialGraph, rng: np.random.Generator) -> Graph:
    """Construct a conservative compatible completion used as a sampler anchor."""

    if partial.kind != "pag":
        raise ValueError("MAG completion requires a PAG")
    directed: list[tuple[int, int]] = []
    bidirected: list[tuple[int, int]] = []
    undirected: list[tuple[int, int]] = []
    circle_pairs: list[tuple[int, int]] = []
    for first in range(partial.node_count):
        for second in range(first + 1, partial.node_count):
            marks = (partial.mark(first, second), partial.mark(second, first))
            if marks == (Endpoint.NULL, Endpoint.NULL):
                continue
            if marks == (Endpoint.CIRCLE, Endpoint.CIRCLE):
                circle_pairs.append((first, second))
            elif marks in (
                (Endpoint.TAIL, Endpoint.ARROW),
                (Endpoint.CIRCLE, Endpoint.ARROW),
                (Endpoint.TAIL, Endpoint.CIRCLE),
            ):
                directed.append((first, second))
            elif marks in (
                (Endpoint.ARROW, Endpoint.TAIL),
                (Endpoint.ARROW, Endpoint.CIRCLE),
                (Endpoint.CIRCLE, Endpoint.TAIL),
            ):
                directed.append((second, first))
            elif marks == (Endpoint.ARROW, Endpoint.ARROW):
                bidirected.append((first, second))
            elif marks == (Endpoint.TAIL, Endpoint.TAIL):
                undirected.append((first, second))
            else:
                raise RuntimeError(f"unsupported PAG endpoint pair: {marks}")
    if circle_pairs:
        matrix = np.zeros((partial.node_count, partial.node_count), dtype=int)
        for first, second in circle_pairs:
            matrix[first, second] = int(Endpoint.TAIL)
            matrix[second, first] = int(Endpoint.TAIL)
        oriented = _random_cpdag_extension(
            PartialGraph.from_matrix(matrix, "cpdag"), rng
        )
        directed.extend(
            (source, target)
            for source, row in enumerate(oriented.directed)
            for target in iter_bits(row)
        )
    graph = graph_from_edges(partial.node_count, directed, bidirected, undirected)
    if not is_maximal_ancestral(graph):
        raise RuntimeError("PAG completion is not a maximal ancestral graph")
    return graph


def _random_pag_completion(partial: PartialGraph, rng: np.random.Generator) -> Graph:
    """Randomize circle marks over compatible complete MAG edge states."""

    if partial.kind != "pag":
        raise ValueError("MAG completion requires a PAG")
    complete_states = (
        (Endpoint.TAIL, Endpoint.ARROW),
        (Endpoint.ARROW, Endpoint.TAIL),
        (Endpoint.ARROW, Endpoint.ARROW),
        (Endpoint.TAIL, Endpoint.TAIL),
    )
    directed: list[tuple[int, int]] = []
    bidirected: list[tuple[int, int]] = []
    undirected: list[tuple[int, int]] = []
    for first in range(partial.node_count):
        for second in range(first + 1, partial.node_count):
            supplied = (partial.mark(first, second), partial.mark(second, first))
            if supplied == (Endpoint.NULL, Endpoint.NULL):
                continue
            compatible = [
                state
                for state in complete_states
                if all(
                    fixed == Endpoint.CIRCLE or fixed == actual
                    for fixed, actual in zip(supplied, state)
                )
            ]
            if not compatible:
                raise RuntimeError(f"PAG edge has no complete state: {supplied}")
            state = compatible[int(rng.integers(len(compatible)))]
            if state == (Endpoint.TAIL, Endpoint.ARROW):
                directed.append((first, second))
            elif state == (Endpoint.ARROW, Endpoint.TAIL):
                directed.append((second, first))
            elif state == (Endpoint.ARROW, Endpoint.ARROW):
                bidirected.append((first, second))
            else:
                undirected.append((first, second))
    try:
        graph = graph_from_edges(partial.node_count, directed, bidirected, undirected)
    except ValueError as error:
        raise RuntimeError(f"random PAG orientation is invalid: {error}") from error
    if not is_maximal_ancestral(graph):
        raise RuntimeError("random PAG orientation is not a maximal ancestral graph")
    return graph


def _validate_pag_fixed_ancestry(partial: PartialGraph) -> None:
    """Reject endpoint constraints that already violate ancestral-graph axioms."""

    if partial.kind != "pag":
        raise ValueError("fixed-ancestry validation requires a PAG")
    directed: list[tuple[int, int]] = []
    bidirected: list[tuple[int, int]] = []
    undirected: list[tuple[int, int]] = []
    for first in range(partial.node_count):
        for second in range(first + 1, partial.node_count):
            marks = (partial.mark(first, second), partial.mark(second, first))
            if marks == (Endpoint.TAIL, Endpoint.ARROW):
                directed.append((first, second))
            elif marks == (Endpoint.ARROW, Endpoint.TAIL):
                directed.append((second, first))
            elif marks == (Endpoint.ARROW, Endpoint.ARROW):
                bidirected.append((first, second))
            elif marks == (Endpoint.TAIL, Endpoint.TAIL):
                undirected.append((first, second))
    try:
        fixed = graph_from_edges(
            partial.node_count, directed, bidirected, undirected
        )
    except ValueError as error:
        raise NonCompletablePAGError(
            f"PAG has no compatible MAG completion: fixed marks cannot form "
            f"an ancestral graph: {error}"
        ) from error
    if not is_ancestral(fixed):
        raise NonCompletablePAGError(
            "PAG has no compatible MAG completion: fixed marks violate "
            "ancestral-graph arrowhead constraints"
        )
    for node in range(partial.node_count):
        if not fixed.undirected[node]:
            continue
        if any(
            partial.mark(node, other) == Endpoint.ARROW
            for other in range(partial.node_count)
        ):
            raise NonCompletablePAGError(
                "PAG has no compatible MAG completion: a fixed undirected "
                "endpoint is incident to a fixed arrowhead"
            )


def is_compatible_completion(partial: PartialGraph, graph: Graph) -> bool:
    """Check skeleton and endpoint compatibility, not Markov equivalence."""

    if partial.node_count != graph.node_count:
        return False
    for first in range(graph.node_count):
        for second in range(first + 1, graph.node_count):
            supplied = (partial.mark(first, second), partial.mark(second, first))
            completed = edge_marks(graph, first, second)
            if (supplied[0] == Endpoint.NULL) != (completed[0] == Endpoint.NULL):
                return False
            if partial.kind == "cpdag" and supplied == (
                Endpoint.TAIL,
                Endpoint.TAIL,
            ):
                if completed not in (
                    (Endpoint.TAIL, Endpoint.ARROW),
                    (Endpoint.ARROW, Endpoint.TAIL),
                ):
                    return False
                continue
            for fixed, actual in zip(supplied, completed):
                if fixed not in (Endpoint.CIRCLE, actual):
                    return False
    return True


@lru_cache(maxsize=4096)
def mag_to_pag(graph: Graph) -> PartialGraph:
    """Return the canonical PAG of a complete MAG using Tetrad's full rules.

    This conversion retains the general ancestral-graph equivalence class.
    Whether to exclude undirected (selection) candidates is a separate support
    restriction and must not alter the PAG used to certify equivalence.
    """

    if not is_maximal_ancestral(graph):
        raise ValueError("canonical PAG conversion requires a maximal ancestral graph")
    _require_tetrad_completion_backend()
    from edu.cmu.tetrad.search.utils import MagToPag
    from pytetrad.tools.translate import (
        causal_learn_graph_to_tetrad,
        tetrad_graph_to_causal_learn,
    )

    matrix = np.zeros((graph.node_count, graph.node_count), dtype=int)
    for first in range(graph.node_count):
        for second in range(first + 1, graph.node_count):
            matrix[first, second], matrix[second, first] = edge_marks(
                graph, first, second
            )
    java_mag = causal_learn_graph_to_tetrad(
        _partial_to_causal_learn(PartialGraph.from_matrix(matrix, "pag"))
    )
    converter = MagToPag(java_mag)
    converter.setVerbose(False)
    converter.setCompleteRuleSetUsed(True)
    converter.setMaxDiscriminatingPathLength(-1)
    java_pag = converter.convert(True, False)
    return _causal_learn_partial(tetrad_graph_to_causal_learn(java_pag), "pag")


def is_equivalent_pag_completion(
    partial: PartialGraph,
    graph: Graph,
    *,
    exclude_selection_bias: bool = False,
) -> bool:
    """Certify full PAG equivalence in addition to endpoint compatibility."""

    if partial.kind != "pag":
        raise ValueError("MAG equivalence certification requires a PAG")
    if exclude_selection_bias and any(graph.undirected):
        return False
    return (
        is_compatible_completion(partial, graph)
        and is_maximal_ancestral(graph)
        and mag_to_pag(graph) == partial
    )


def sample_completions_with_diagnostics(
    output: GraphOutput,
    regime: Regime,
    limit: int,
    seed: int,
    *,
    exclude_selection_bias: bool = False,
) -> tuple[tuple[Graph, ...], CompletionSamplingDiagnostics]:
    """Sample certified completions and report bounded-search counters.

    MAG candidates must round-trip to the input's canonical PAG. Sampling is
    bounded and is not uniform over the entire equivalence class. The optional
    selection restriction excludes undirected edges, never all-directed MAGs.
    """

    if limit < 1:
        raise ValueError("completion limit must be positive")
    if isinstance(output, Graph):
        if regime == "dag" and (any(output.bidirected) or any(output.undirected)):
            raise ValueError("non-DAG point output in the DAG regime")
        if regime == "mag" and not is_maximal_ancestral(output):
            raise ValueError("point output in the MAG regime is not maximal ancestral")
        if regime == "mag" and exclude_selection_bias and any(output.undirected):
            raise ValueError("no-selection MAG support excludes undirected edges")
        return (output,), CompletionSamplingDiagnostics(
            requested_limit=limit,
            returned_count=1,
            anchor_source="point",
            random_attempt_count=0,
            unique_random_count=0,
            duplicate_random_count=0,
            rejected_random_count=0,
            stopped_for_stagnation=False,
        )
    expected_kind = "cpdag" if regime == "dag" else "pag"
    if output.kind != expected_kind:
        raise ValueError(f"{regime} regime requires {expected_kind} outputs")
    rng = np.random.default_rng(seed)
    completions: set[Graph] = set()
    compiled_cpdag: _CompiledCPDAG | None = None
    anchor_source: Literal["point", "causal-learn", "tetrad", "local", "none"] = "none"
    if regime == "dag":
        # causal-learn's pdag2dag has no no-progress guard and loops forever on
        # a non-extendable PDAG.  The bounded local Dor--Tarsi pass is therefore
        # a safety certificate before causal-learn supplies the anchor extension.
        compiled_cpdag = _compile_cpdag(output)
        _random_compiled_cpdag_extension(compiled_cpdag, np.random.default_rng(seed))
        anchor = _causal_learn_cpdag_extension(output)
        if not _is_compatible_compiled_cpdag(compiled_cpdag, anchor):
            raise RuntimeError("causal-learn returned an incompatible DAG extension")
        completions.add(anchor)
        anchor_source = "causal-learn"
    else:
        _validate_pag_fixed_ancestry(output)
        _require_tetrad_completion_backend()
        try:
            anchor = _tetrad_pag_extension(output, seed)
        except RuntimeError:
            pass
        else:
            if is_equivalent_pag_completion(
                output, anchor, exclude_selection_bias=exclude_selection_bias
            ):
                completions.add(anchor)
                anchor_source = "tetrad"
        if not completions:
            try:
                anchor = _pag_anchor_completion(output, rng)
            except RuntimeError:
                pass
            else:
                if is_equivalent_pag_completion(
                    output, anchor, exclude_selection_bias=exclude_selection_bias
                ):
                    completions.add(anchor)
                    anchor_source = "local"
    if len(completions) >= limit:
        graphs = tuple(sorted(completions))
        return graphs, CompletionSamplingDiagnostics(
            requested_limit=limit,
            returned_count=len(graphs),
            anchor_source=anchor_source,
            random_attempt_count=0,
            unique_random_count=0,
            duplicate_random_count=0,
            rejected_random_count=0,
            stopped_for_stagnation=False,
        )
    attempts = max(100, 50 * limit)
    stagnation_limit = max(100, limit)
    stagnant = 0
    random_attempt_count = 0
    unique_random_count = 0
    duplicate_random_count = 0
    rejected_random_count = 0
    stopped_for_stagnation = False
    for _ in range(attempts):
        random_attempt_count += 1
        try:
            graph = (
                _random_compiled_cpdag_extension(compiled_cpdag, rng)
                if regime == "dag"
                else _random_pag_completion(output, rng)
            )
        except RuntimeError:
            rejected_random_count += 1
            stagnant += 1
            if stagnant >= stagnation_limit:
                stopped_for_stagnation = True
                break
            continue
        compatible = (
            _is_compatible_compiled_cpdag(compiled_cpdag, graph)
            if compiled_cpdag is not None
            else is_equivalent_pag_completion(
                output, graph, exclude_selection_bias=exclude_selection_bias
            )
        )
        if compatible:
            previous_count = len(completions)
            completions.add(graph)
            if len(completions) >= limit:
                unique_random_count += 1
                break
            if len(completions) > previous_count:
                unique_random_count += 1
                stagnant = 0
            else:
                duplicate_random_count += 1
                stagnant += 1
        else:
            rejected_random_count += 1
            stagnant += 1
        if stagnant >= stagnation_limit:
            stopped_for_stagnation = True
            break
    if not completions:
        if regime == "dag":
            raise NonExtendableCPDAGError("CPDAG has no consistent DAG extension")
        raise NonCompletablePAGError("PAG yielded no compatible MAG completion")
    graphs = tuple(sorted(completions))
    return graphs, CompletionSamplingDiagnostics(
        requested_limit=limit,
        returned_count=len(graphs),
        anchor_source=anchor_source,
        random_attempt_count=random_attempt_count,
        unique_random_count=unique_random_count,
        duplicate_random_count=duplicate_random_count,
        rejected_random_count=rejected_random_count,
        stopped_for_stagnation=stopped_for_stagnation,
    )


def sample_completions(
    output: GraphOutput,
    regime: Regime,
    limit: int,
    seed: int,
    *,
    exclude_selection_bias: bool = False,
) -> tuple[Graph, ...]:
    """Backward-compatible completion sampler without diagnostic counters."""

    completions, _ = sample_completions_with_diagnostics(
        output, regime, limit, seed,
        exclude_selection_bias=exclude_selection_bias,
    )
    return completions


def _causal_learn_partial(general_graph: object, kind: Literal["cpdag", "pag"]) -> PartialGraph:
    raw = np.asarray(general_graph.graph, dtype=int)
    converted = np.zeros_like(raw)
    endpoint_map = {0: Endpoint.NULL, -1: Endpoint.TAIL, 1: Endpoint.ARROW, 2: Endpoint.CIRCLE}
    for index, value in np.ndenumerate(raw):
        if int(value) not in endpoint_map:
            raise RuntimeError(f"unsupported causal-learn endpoint code: {value}")
        converted[index] = int(endpoint_map[int(value)])
    return PartialGraph.from_matrix(converted, kind)


def _partial_to_causal_learn(partial: PartialGraph) -> object:
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint as CausalEndpoint
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode

    nodes = [GraphNode(f"X{index}") for index in range(partial.node_count)]
    graph = GeneralGraph(nodes)
    endpoint_map = {
        Endpoint.TAIL: CausalEndpoint.TAIL,
        Endpoint.ARROW: CausalEndpoint.ARROW,
        Endpoint.CIRCLE: CausalEndpoint.CIRCLE,
    }
    for first in range(partial.node_count):
        for second in range(first + 1, partial.node_count):
            marks = (partial.mark(first, second), partial.mark(second, first))
            if marks == (Endpoint.NULL, Endpoint.NULL):
                continue
            if marks[0] not in endpoint_map or marks[1] not in endpoint_map:
                raise ValueError(f"unsupported partial-graph endpoint pair: {marks}")
            graph.add_edge(
                Edge(
                    nodes[first],
                    nodes[second],
                    endpoint_map[marks[0]],
                    endpoint_map[marks[1]],
                )
            )
    return graph


def _causal_learn_cpdag_extension(partial: PartialGraph) -> Graph:
    """Use causal-learn for an anchor after the caller proves extendibility."""

    from causallearn.utils.PDAG2DAG import pdag2dag

    if partial.kind != "cpdag":
        raise ValueError("causal-learn DAG completion requires a CPDAG")
    completed = _causal_learn_partial(
        pdag2dag(_partial_to_causal_learn(partial)), "cpdag"
    )
    return _random_cpdag_extension(completed, np.random.default_rng(0))


def _jvm_is_started() -> bool:
    try:
        import jpype
    except ImportError:
        return False
    return bool(jpype.isJVMStarted())


def _require_tetrad_completion_backend() -> None:
    """Load the pinned Java backend; never fall back to uncertified MAGs."""

    try:
        # The pinned translation module starts JPype with its packaged jar.
        import pytetrad.tools.translate  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "certified MAG completion requires the pinned Tetrad Java backend"
        ) from error
    if not _jvm_is_started():
        raise RuntimeError("certified MAG completion could not start the Tetrad JVM")


def _tetrad_pag_extension(partial: PartialGraph, seed: int) -> Graph:
    """Construct a Zhang MAG extension with Tetrad when its JVM is active."""

    if partial.kind != "pag":
        raise ValueError("Tetrad MAG completion requires a PAG")
    if not _jvm_is_started():
        raise RuntimeError("Tetrad JVM is not active")
    try:
        import edu.cmu.tetrad.graph as tetrad_graph
        import java.util as java_util
        from pytetrad.tools.translate import (
            causal_learn_graph_to_tetrad,
            tetrad_graph_to_causal_learn,
        )

        java_pag = causal_learn_graph_to_tetrad(_partial_to_causal_learn(partial))
        java_mag = tetrad_graph.GraphTransforms.randomZhangMagFromPag(
            java_pag, java_util.Random(int(seed))
        )
        if java_mag is None:
            raise RuntimeError("Tetrad returned no Zhang MAG")
        completed = _causal_learn_partial(
            tetrad_graph_to_causal_learn(java_mag), "pag"
        )
        graph = _random_pag_completion(completed, np.random.default_rng(seed))
    except Exception as error:
        raise RuntimeError(
            f"Tetrad Zhang PAG-to-MAG completion failed: {type(error).__name__}: {error}"
        ) from error
    if not is_compatible_completion(partial, graph):
        raise RuntimeError("Tetrad returned a MAG incompatible with the supplied PAG")
    return graph


def learn_pc(data: np.ndarray, alpha: float = 0.05) -> PartialGraph:
    """Compatibility alias for the certified Tetrad CPC backend."""

    from .cpc import learn_cpc

    return learn_cpc(data, alpha, variable_type="continuous")


def learn_ges(
    data: np.ndarray,
    max_parents: int | None = None,
    lambda_value: float | None = None,
) -> PartialGraph:
    from causallearn.search.ScoreBased.GES import ges

    options: dict[str, object] = {"score_func": "local_score_BIC"}
    if max_parents is not None:
        options["maxP"] = max_parents
    if lambda_value is not None:
        if lambda_value <= 0:
            raise ValueError("GES BIC penalty must be positive")
        options["lambda_value"] = lambda_value
    graph = ges(data, **options)["G"]
    return _causal_learn_partial(graph, "cpdag")


def learn_fci(data: np.ndarray, alpha: float = 0.05) -> PartialGraph:
    from causallearn.search.ConstraintBased.FCI import fci

    graph, _ = fci(
        data,
        independence_test_method="fisherz",
        alpha=alpha,
        verbose=False,
        show_progress=False,
    )
    return _causal_learn_partial(graph, "pag")


def learn_tetrad_pag(data: np.ndarray, method: Literal["rfci", "gfci"], alpha: float) -> PartialGraph:
    try:
        import pandas as pd
        from pytetrad.tools.TetradSearch import TetradSearch
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "RFCI/GFCI require the pinned py-tetrad package, Python >=3.12, and Java >=21"
        ) from error
    frame = pd.DataFrame(data, columns=[f"X{index}" for index in range(data.shape[1])])
    search = TetradSearch(frame)
    search.use_fisher_z(alpha=alpha)
    if method == "gfci":
        search.use_sem_bic()
        search.run_gfci()
    else:
        search.run_rfci()
    return _causal_learn_partial(search.get_causal_learn(), "pag")


def learn_notears(
    data: np.ndarray,
    l1_penalty: float = 0.05,
    threshold: float = 0.30,
    max_outer_iterations: int = 12,
) -> Graph:
    """Linear NOTEARS with the original augmented-Lagrangian objective."""

    sample_size, node_count = data.shape
    dimension = node_count * node_count

    def loss(weight_matrix: np.ndarray) -> tuple[float, np.ndarray]:
        residual = data - data @ weight_matrix
        return (
            float(0.5 / sample_size * np.sum(residual * residual)),
            -data.T @ residual / sample_size,
        )

    def acyclicity(weight_matrix: np.ndarray) -> tuple[float, np.ndarray]:
        exponential = expm(weight_matrix * weight_matrix)
        return (
            float(np.trace(exponential) - node_count),
            exponential.T * weight_matrix * 2.0,
        )

    rho = 1.0
    multiplier = 0.0
    previous_h = math.inf
    split_weights = np.zeros(2 * dimension, dtype=float)
    bounds = [
        (0.0, 0.0) if source == target else (0.0, None)
        for _copy in range(2)
        for source in range(node_count)
        for target in range(node_count)
    ]
    for _ in range(max_outer_iterations):
        while rho < 1e16:

            def objective(split: np.ndarray) -> tuple[float, np.ndarray]:
                matrix = (split[:dimension] - split[dimension:]).reshape(
                    node_count, node_count
                )
                loss_value, loss_gradient = loss(matrix)
                h_value, h_gradient = acyclicity(matrix)
                smooth = loss_gradient + (rho * h_value + multiplier) * h_gradient
                value = loss_value + 0.5 * rho * h_value * h_value + multiplier * h_value
                value += l1_penalty * np.sum(split)
                gradient = np.concatenate((smooth.ravel(), -smooth.ravel())) + l1_penalty
                return float(value), gradient

            solution = minimize(
                objective,
                split_weights,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
            )
            candidate = (solution.x[:dimension] - solution.x[dimension:]).reshape(
                node_count, node_count
            )
            candidate_h, _ = acyclicity(candidate)
            if candidate_h > 0.25 * previous_h:
                rho *= 10.0
            else:
                split_weights = solution.x
                break
        previous_h = candidate_h
        multiplier += rho * candidate_h
        if candidate_h <= 1e-8 or rho >= 1e16:
            break
    weights = (split_weights[:dimension] - split_weights[dimension:]).reshape(
        node_count, node_count
    )
    candidates = sorted(
        (
            (abs(weights[source, target]), source, target)
            for source in range(node_count)
            for target in range(node_count)
            if source != target and abs(weights[source, target]) >= threshold
        ),
        reverse=True,
    )
    rows = [0] * node_count
    for _, source, target in candidates:
        proposed = list(rows)
        proposed[source] |= 1 << target
        if _rows_are_acyclic(tuple(proposed)):
            rows = proposed
    return Graph(tuple(rows))


@dataclass(frozen=True)
class DiscoveryResult:
    regime: Regime
    method_names: tuple[str, ...]
    outputs_by_bootstrap: tuple[tuple[GraphOutput, ...], ...]


def run_paired_discovery(
    data: np.ndarray,
    regime: Regime,
    bootstrap_count: int,
    seed: int,
    alpha: float = 0.05,
    notears_l1: float = 0.05,
    notears_threshold: float = 0.30,
) -> DiscoveryResult:
    """Run every regime-compatible method on identical bootstrap samples."""

    rng = np.random.default_rng(seed)
    names = (
        ("cpc", "ges", "notears")
        if regime == "dag"
        else ("fci", "rfci", "gfci")
    )
    bootstraps: list[tuple[GraphOutput, ...]] = []
    for bootstrap in range(bootstrap_count):
        indices = rng.integers(0, data.shape[0], size=data.shape[0])
        sample = standardize(data[indices])
        if regime == "dag":
            from .cpc import learn_cpc

            learners = (
                lambda: learn_cpc(
                    sample,
                    alpha=alpha,
                    variable_type="continuous",
                ),
                lambda: learn_ges(sample),
                lambda: learn_notears(sample, notears_l1, notears_threshold),
            )
        else:
            learners = (
                lambda: learn_fci(sample, alpha),
                lambda: learn_tetrad_pag(sample, "rfci", alpha),
                lambda: learn_tetrad_pag(sample, "gfci", alpha),
            )
        outputs: list[GraphOutput] = []
        for name, learner in zip(names, learners):
            try:
                outputs.append(learner())
            except Exception as error:
                raise RuntimeError(
                    f"{regime} discovery failed for {name} on bootstrap {bootstrap}: "
                    f"{type(error).__name__}: {error}"
                ) from error
        bootstraps.append(tuple(outputs))
    return DiscoveryResult(regime, names, tuple(bootstraps))


@dataclass(frozen=True)
class CompletionFailure:
    bootstrap_index: int
    method_name: str
    output_kind: str
    error_type: str
    message: str
    endpoint_matrix: tuple[tuple[int, ...], ...] | None


@dataclass(frozen=True)
class DataEnsemble:
    regime: Regime
    graphs: tuple[Graph, ...]
    q_bootstrap: np.ndarray
    completion_counts: tuple[tuple[int, ...], ...]
    active_method_counts: tuple[int, ...] = ()
    completion_failures: tuple[CompletionFailure, ...] = ()

    def __post_init__(self) -> None:
        if self.q_bootstrap.shape != (len(self.completion_counts), len(self.graphs)):
            raise ValueError("bootstrap probability matrix has the wrong shape")
        if not np.allclose(np.sum(self.q_bootstrap, axis=1), 1.0):
            raise ValueError("each bootstrap distribution must sum to one")
        if not self.active_method_counts:
            object.__setattr__(
                self,
                "active_method_counts",
                tuple(sum(count > 0 for count in row) for row in self.completion_counts),
            )
        if len(self.active_method_counts) != len(self.completion_counts):
            raise ValueError("active-method counts must align with bootstrap rows")
        if any(count < 1 for count in self.active_method_counts):
            raise ValueError("each bootstrap must retain at least one discovery method")

    @property
    def q_data(self) -> np.ndarray:
        return np.mean(self.q_bootstrap, axis=0)


def build_data_ensemble(
    discovery: DiscoveryResult,
    completion_limit: int,
    seed: int,
) -> DataEnsemble:
    bootstrap_masses: list[Counter[Graph]] = []
    completion_counts: list[tuple[int, ...]] = []
    active_method_counts: list[int] = []
    completion_failures: list[CompletionFailure] = []
    method_count = len(discovery.method_names)
    for bootstrap, outputs in enumerate(discovery.outputs_by_bootstrap):
        if len(outputs) != method_count:
            raise RuntimeError("a paired bootstrap is missing a discovery method")
        masses: Counter[Graph] = Counter()
        counts = [0] * method_count
        successful: list[tuple[Graph, ...]] = []
        for method_index, output in enumerate(outputs):
            try:
                completions = sample_completions(
                    output,
                    discovery.regime,
                    completion_limit,
                    stable_seed(seed, bootstrap, method_index),
                )
            except CompletionError as error:
                endpoint_matrix = None
                if isinstance(output, PartialGraph):
                    endpoint_matrix = tuple(
                        tuple(
                            int(output.mark(first, second))
                            for second in range(output.node_count)
                        )
                        for first in range(output.node_count)
                    )
                completion_failures.append(
                    CompletionFailure(
                        bootstrap,
                        discovery.method_names[method_index],
                        output.kind if isinstance(output, PartialGraph) else "graph",
                        type(error).__name__,
                        str(error),
                        endpoint_matrix,
                    )
                )
                continue
            counts[method_index] = len(completions)
            successful.append(completions)
        if not successful:
            failures = [
                failure
                for failure in completion_failures
                if failure.bootstrap_index == bootstrap
            ]
            details = "; ".join(
                f"{failure.method_name}: {failure.error_type}: {failure.message}"
                for failure in failures
            )
            raise RuntimeError(
                f"all {method_count} {discovery.regime} discovery outputs failed "
                f"completion on bootstrap {bootstrap}: {details}"
            )
        method_mass = 1.0 / len(successful)
        for completions in successful:
            mass = method_mass / len(completions)
            for graph in completions:
                masses[graph] += mass
        bootstrap_masses.append(masses)
        completion_counts.append(tuple(counts))
        active_method_counts.append(len(successful))
    graphs = tuple(sorted(set().union(*(mass.keys() for mass in bootstrap_masses))))
    index = {graph: position for position, graph in enumerate(graphs)}
    probabilities = np.zeros((len(bootstrap_masses), len(graphs)), dtype=float)
    for bootstrap, masses in enumerate(bootstrap_masses):
        for graph, mass in masses.items():
            probabilities[bootstrap, index[graph]] = mass
        probabilities[bootstrap] /= np.sum(probabilities[bootstrap])
    return DataEnsemble(
        discovery.regime,
        graphs,
        probabilities,
        tuple(completion_counts),
        tuple(active_method_counts),
        tuple(completion_failures),
    )


ClaimKind = Literal[
    "directed",
    "directed_exclusion",
    "adjacency",
    "non_adjacency",
    "ancestry",
    "ordering",
    "bidirected",
    "endpoint",
    "whole_graph",
    "skeleton",
]


@dataclass(frozen=True)
class Claim:
    claim_id: str
    group: ClaimGroup
    kind: ClaimKind
    truth: int
    source: int = -1
    target: int = -1
    expected_marks: tuple[int, int] | None = None
    guess: Graph | None = None

    @property
    def label(self) -> str:
        if self.kind in ("whole_graph", "skeleton"):
            return f"{self.kind}:{self.claim_id}"
        operators = {
            "directed": "->",
            "directed_exclusion": "!->",
            "adjacency": "~",
            "non_adjacency": "!~",
            "ancestry": "~>",
            "ordering": "<",
            "bidirected": "<->",
            "endpoint": "endpoint",
        }
        return f"X{self.source}{operators[self.kind]}X{self.target}"


def same_skeleton(first: Graph, second: Graph) -> bool:
    if first.node_count != second.node_count:
        return False
    return all(
        has_adjacency(first, left, right) == has_adjacency(second, left, right)
        for left in range(first.node_count)
        for right in range(left + 1, first.node_count)
    )


def evaluate_claim(
    graph: Graph,
    claim: Claim,
    closure: tuple[int, ...] | None = None,
) -> int | None:
    if claim.kind == "directed":
        return int(has_edge(graph, claim.source, claim.target))
    if claim.kind == "directed_exclusion":
        return int(not has_edge(graph, claim.source, claim.target))
    if claim.kind == "adjacency":
        return int(has_adjacency(graph, claim.source, claim.target))
    if claim.kind == "non_adjacency":
        return int(not has_adjacency(graph, claim.source, claim.target))
    if claim.kind in ("ancestry", "ordering"):
        closure = closure if closure is not None else transitive_closure(graph)
        if claim.kind == "ancestry":
            return int(bool(closure[claim.source] & (1 << claim.target)))
        return int(not bool(closure[claim.target] & (1 << claim.source)))
    if claim.kind == "bidirected":
        return int(has_bidirected(graph, claim.source, claim.target))
    if claim.kind == "endpoint":
        if claim.expected_marks is None:
            raise ValueError("endpoint claim has no expected marks")
        actual = edge_marks(graph, claim.source, claim.target)
        expected = tuple(Endpoint(mark) for mark in claim.expected_marks)
        return int(
            all(
                supplied == observed
                or (supplied == Endpoint.CIRCLE and observed != Endpoint.NULL)
                for supplied, observed in zip(expected, actual)
            )
        )
    if claim.kind == "whole_graph":
        if claim.guess is None:
            raise ValueError("whole-graph claim has no graph")
        return int(graph == claim.guess)
    if claim.kind == "skeleton":
        if claim.guess is None:
            raise ValueError("skeleton claim has no graph")
        return int(same_skeleton(graph, claim.guess))
    raise ValueError(f"unknown claim kind: {claim.kind}")


def claim_value_matrix(graphs: Sequence[Graph], claims: Sequence[Claim]) -> np.ndarray:
    values = np.full((len(graphs), len(claims)), -1, dtype=np.int8)
    for graph_index, graph in enumerate(graphs):
        closure = transitive_closure(graph)
        for claim_index, claim in enumerate(claims):
            value = evaluate_claim(graph, claim, closure)
            values[graph_index, claim_index] = -1 if value is None else value
    return values


def _local_claim(
    claim_id: str,
    group: ClaimGroup,
    kind: ClaimKind,
    graph: Graph,
    source: int,
    target: int,
    expected_marks: tuple[int, int] | None = None,
) -> Claim:
    provisional = Claim(claim_id, group, kind, 0, source, target, expected_marks)
    truth = evaluate_claim(graph, provisional)
    if truth is None:
        raise ValueError("requested claim is unresolved in the target regime")
    return replace(provisional, truth=int(truth))


def _choose_claim(
    pool: Sequence[Claim], desired_truth: int, rng: np.random.Generator
) -> Claim:
    candidates = [claim for claim in pool if claim.truth == desired_truth]
    if not candidates:
        raise RuntimeError("not enough resolved claims to satisfy the balanced design")
    return candidates[int(rng.integers(len(candidates)))]


def _perturb_graph(graph: Graph, rng: np.random.Generator) -> Graph:
    pairs = [
        (first, second)
        for first in range(graph.node_count)
        for second in range(first + 1, graph.node_count)
        if has_adjacency(graph, first, second)
    ]
    if pairs:
        remove = pairs[int(rng.integers(len(pairs)))]
        directed = [
            (source, target)
            for source, row in enumerate(graph.directed)
            for target in iter_bits(row)
            if {source, target} != set(remove)
        ]
        bidirected = [
            (first, second)
            for first in range(graph.node_count)
            for second in range(first + 1, graph.node_count)
            if has_bidirected(graph, first, second) and (first, second) != remove
        ]
        undirected = [
            (first, second)
            for first in range(graph.node_count)
            for second in range(first + 1, graph.node_count)
            if has_undirected(graph, first, second) and (first, second) != remove
        ]
        return graph_from_edges(graph.node_count, directed, bidirected, undirected)
    return graph_from_edges(graph.node_count, [(0, 1)])


def sample_resolved_claims(
    target: Graph,
    regime: Regime,
    count_per_group: int,
    seed: int,
) -> tuple[Claim, ...]:
    if count_per_group % 2:
        raise ValueError("count_per_group must be even")
    rng = np.random.default_rng(seed)
    pair_claims: dict[str, list[Claim]] = {}
    definitions: list[tuple[ClaimGroup, ClaimKind]] = [
        ("I", "directed"),
        ("I", "directed_exclusion"),
        ("I", "adjacency"),
        ("I", "non_adjacency"),
        ("II", "ancestry"),
        ("II", "ordering"),
    ]
    if regime == "mag":
        definitions.append(("III", "bidirected"))
    for group, kind in definitions:
        candidates: list[Claim] = []
        for source in range(target.node_count):
            for destination in range(target.node_count):
                if source == destination:
                    continue
                first, second = (
                    (min(source, destination), max(source, destination))
                    if kind in ("adjacency", "non_adjacency", "bidirected")
                    else (source, destination)
                )
                candidate = _local_claim(
                    f"pool-{group}-{kind}-{first}-{second}",
                    group,
                    kind,
                    target,
                    first,
                    second,
                )
                if candidate not in candidates:
                    candidates.append(candidate)
        pair_claims[f"{group}:{kind}"] = candidates
    if regime == "mag":
        endpoint_candidates: list[Claim] = []
        states = (
            (Endpoint.NULL, Endpoint.NULL),
            (Endpoint.TAIL, Endpoint.ARROW),
            (Endpoint.ARROW, Endpoint.TAIL),
            (Endpoint.ARROW, Endpoint.ARROW),
            (Endpoint.TAIL, Endpoint.TAIL),
            (Endpoint.CIRCLE, Endpoint.CIRCLE),
            (Endpoint.CIRCLE, Endpoint.ARROW),
            (Endpoint.ARROW, Endpoint.CIRCLE),
            (Endpoint.CIRCLE, Endpoint.TAIL),
            (Endpoint.TAIL, Endpoint.CIRCLE),
        )
        for first in range(target.node_count):
            for second in range(first + 1, target.node_count):
                for state in states:
                    endpoint_candidates.append(
                        _local_claim(
                            f"pool-III-endpoint-{first}-{second}-{int(state[0])}{int(state[1])}",
                            "III",
                            "endpoint",
                            target,
                            first,
                            second,
                            tuple(int(mark) for mark in state),
                        )
                    )
        pair_claims["III:endpoint"] = endpoint_candidates

    groups: dict[ClaimGroup, list[Claim]] = {"I": [], "II": [], "III": [], "IV": []}
    kind_cycles: dict[ClaimGroup, tuple[ClaimKind, ...]] = {
        "I": ("directed", "directed_exclusion", "adjacency", "non_adjacency"),
        "II": ("ancestry", "ordering"),
        "III": ("bidirected", "endpoint"),
        "IV": ("whole_graph", "skeleton"),
    }
    applicable: tuple[ClaimGroup, ...] = ("I", "II", "IV") if regime == "dag" else (
        "I",
        "II",
        "III",
        "IV",
    )
    for group in applicable:
        for index in range(count_per_group):
            desired = index % 2
            # Pair false and true instances within each claim kind before
            # advancing through the group's kinds. This avoids coupling kind
            # to truth (for example, sampling only false directed-edge claims).
            kind = kind_cycles[group][(index // 2) % len(kind_cycles[group])]
            if group == "IV":
                guess = target if desired else _perturb_graph(target, rng)
                claim = Claim(
                    f"{group}-{index}", group, kind, desired, guess=guess
                )
            else:
                pool = pair_claims[f"{group}:{kind}"]
                claim = _choose_claim(pool, desired, rng)
                claim = replace(claim, claim_id=f"{group}-{index}")
            groups[group].append(claim)
    return tuple(claim for group in applicable for claim in groups[group])


def expand_composite_claim(
    claim: Claim,
    target: Graph,
    budget: int,
    seed: int,
) -> tuple[Claim, ...]:
    if claim.kind not in ("whole_graph", "skeleton") or claim.guess is None:
        return (claim,)
    primitives: list[Claim] = []
    for first in range(target.node_count):
        for second in range(first + 1, target.node_count):
            if claim.kind == "skeleton":
                kind: ClaimKind = (
                    "adjacency" if has_adjacency(claim.guess, first, second) else "non_adjacency"
                )
                primitive = _local_claim(
                    f"{claim.claim_id}-p{first}-{second}",
                    "IV",
                    kind,
                    target,
                    first,
                    second,
                )
            else:
                expected = tuple(
                    int(mark) for mark in edge_marks(claim.guess, first, second)
                )
                primitive = _local_claim(
                    f"{claim.claim_id}-p{first}-{second}",
                    "IV",
                    "endpoint",
                    target,
                    first,
                    second,
                    expected,
                )
            primitives.append(primitive)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(primitives))
    return tuple(primitives[int(index)] for index in order[:budget])


@dataclass(frozen=True, init=False)
class EvidenceNode:
    evidence_id: str
    claim_index: int
    stance: Literal[-1, 1]
    reported_confidence: float
    provenance_family: str
    factor_id: str
    context_id: str
    context_compatible: bool
    correct: bool
    treatment: EvidenceTreatment

    def __init__(
        self,
        evidence_id: str,
        claim_index: int,
        stance: Literal[-1, 1],
        reported_confidence: float | None = None,
        provenance_family: str | None = None,
        factor_id: str | None = None,
        context_id: str | None = None,
        context_compatible: bool | None = None,
        correct: bool | None = None,
        treatment: EvidenceTreatment | None = None,
        *,
        confidence: float | None = None,
    ) -> None:
        if confidence is not None:
            # ``dataclasses.replace(node, confidence=...)`` supplies every stored
            # field as well as the legacy alias. Let the explicit legacy change
            # override the copied reported value for backward compatibility.
            reported_confidence = confidence
        elif reported_confidence is None:
            raise TypeError("reported_confidence is required")
        required = {
            "provenance_family": provenance_family,
            "factor_id": factor_id,
            "context_id": context_id,
            "context_compatible": context_compatible,
            "correct": correct,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise TypeError(f"missing required evidence fields: {', '.join(missing)}")
        if stance not in (-1, 1):
            raise ValueError("evidence stances must be -1 or 1")
        resolved = resolve_evidence_confidence(reported_confidence, treatment)
        object.__setattr__(self, "evidence_id", str(evidence_id))
        object.__setattr__(self, "claim_index", int(claim_index))
        object.__setattr__(self, "stance", stance)
        object.__setattr__(self, "reported_confidence", resolved.reported_confidence)
        object.__setattr__(self, "provenance_family", provenance_family)
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "context_compatible", context_compatible)
        object.__setattr__(self, "correct", correct)
        object.__setattr__(self, "treatment", resolved.treatment)

    @property
    def confidence(self) -> float:
        """Backward-compatible read-only alias for the reported confidence."""

        return self.reported_confidence

    def resolved_confidence(
        self, soft_confidence_cap: float = DEFAULT_SOFT_CONFIDENCE_CAP
    ) -> ResolvedEvidenceConfidence:
        return resolve_evidence_confidence(
            self.reported_confidence, self.treatment, soft_confidence_cap
        )


@dataclass(frozen=True)
class EvidenceGraph:
    target_context: str
    nodes: tuple[EvidenceNode, ...]

    def __post_init__(self) -> None:
        grouped_treatments: dict[tuple[str, str], set[EvidenceTreatment]] = {}
        for node in self.nodes:
            grouped_treatments.setdefault(
                (node.provenance_family, node.factor_id), set()
            ).add(node.treatment)
        mixed = tuple(key for key, values in grouped_treatments.items() if len(values) > 1)
        if mixed:
            raise ValueError(
                "all evidence in one provenance/factor dependence group must use "
                f"the same treatment; mixed groups: {mixed!r}"
            )

    def representatives(self) -> tuple[EvidenceNode, ...]:
        grouped: dict[tuple[str, str], list[EvidenceNode]] = {}
        for node in self.nodes:
            grouped.setdefault((node.provenance_family, node.factor_id), []).append(node)
        representatives: list[EvidenceNode] = []
        for key in sorted(grouped):
            family = grouped[key]
            representative = min(family, key=lambda node: node.evidence_id)
            representatives.append(
                replace(
                    representative,
                    reported_confidence=min(
                        family,
                        key=lambda node: abs(float(node.confidence) - 0.5),
                    ).reported_confidence,
                )
            )
        return tuple(representatives)


ATOMIC_EVIDENCE_TYPES = (
    "accurate",
    "inaccurate",
    "conflict",
    "repeated",
    "context_incompatible",
)


def build_synthetic_evidence(
    claims: Sequence[Claim],
    evidence_types: Sequence[str],
    confidence: float | None,
    seed: int,
    treatment: EvidenceTreatment | None = None,
) -> EvidenceGraph:
    if not evidence_types:
        return EvidenceGraph("target", ())
    rng = np.random.default_rng(seed)
    levels = (0.9, 0.75, 0.55)
    nodes: list[EvidenceNode] = []
    mixed_profile = (
        len(evidence_types) == len(ATOMIC_EVIDENCE_TYPES)
        and set(evidence_types) == set(ATOMIC_EVIDENCE_TYPES)
    )
    correctness_overrides: list[bool | None]
    if mixed_profile:
        if len(claims) % len(ATOMIC_EVIDENCE_TYPES):
            raise ValueError("the balanced mixed profile requires a multiple of five claims")
        per_type = len(claims) // len(ATOMIC_EVIDENCE_TYPES)
        if per_type % 2:
            raise ValueError("the balanced mixed profile requires an even count per type")
        packages: list[tuple[str, bool | None]] = []
        packages.extend(("accurate", True) for _ in range(per_type))
        packages.extend(("inaccurate", False) for _ in range(per_type))
        packages.extend(("conflict", None) for _ in range(per_type))
        for evidence_type in ("repeated", "context_incompatible"):
            packages.extend((evidence_type, True) for _ in range(per_type // 2))
            packages.extend((evidence_type, False) for _ in range(per_type // 2))
        for _ in range(10_000):
            permutation = rng.permutation(len(packages))
            assigned = [packages[int(index)] for index in permutation]

            def is_supporting(index: int) -> bool:
                _, correct_override = assigned[index]
                if correct_override is None:
                    raise ValueError("conflict packages do not have one stance")
                return bool(claims[index].truth) == correct_override

            repeated_supports = sum(
                is_supporting(index)
                for index, (evidence_type, _) in enumerate(assigned)
                if evidence_type == "repeated"
            )
            nonconflict_supports = sum(
                is_supporting(index)
                for index, (evidence_type, _) in enumerate(assigned)
                if evidence_type != "conflict"
            )
            if (
                repeated_supports == per_type // 2
                and nonconflict_supports == 2 * per_type
            ):
                break
        else:
            raise RuntimeError("failed to balance the mixed evidence profile")
        type_assignments = [evidence_type for evidence_type, _ in assigned]
        correctness_overrides = [correctness for _, correctness in assigned]
    else:
        type_assignments = [
            evidence_types[index % len(evidence_types)] for index in range(len(claims))
        ]
        if len(evidence_types) > 1:
            permutation = rng.permutation(len(type_assignments))
            type_assignments = [type_assignments[int(index)] for index in permutation]
        correctness_overrides = [None] * len(claims)
    if confidence is None:
        confidence_assignments = [levels[index % len(levels)] for index in range(len(claims))]
        permutation = rng.permutation(len(confidence_assignments))
        confidence_assignments = [
            confidence_assignments[int(index)] for index in permutation
        ]
    else:
        confidence_assignments = [confidence] * len(claims)

    def add(
        claim_index: int,
        stance: int,
        value: float,
        provenance: str,
        factor: str,
        context: str = "target",
        compatible: bool = True,
    ) -> None:
        correct_stance = 1 if claims[claim_index].truth else -1
        nodes.append(
            EvidenceNode(
                f"e{len(nodes):04d}",
                claim_index,
                1 if stance > 0 else -1,
                value,
                provenance,
                factor,
                context,
                compatible,
                stance == correct_stance,
                treatment,
            )
        )

    for claim_index, claim in enumerate(claims):
        evidence_type = type_assignments[claim_index]
        value = confidence_assignments[claim_index]
        correct = 1 if claim.truth else -1
        correctness_override = correctness_overrides[claim_index]
        prefix = f"{evidence_type}-{claim_index}"
        if evidence_type in ("accurate", "accurate_one"):
            add(claim_index, correct, value, f"{prefix}-source", f"{prefix}-factor")
        elif evidence_type == "inaccurate":
            add(claim_index, -correct, value, f"{prefix}-source", f"{prefix}-factor")
        elif evidence_type == "conflict":
            add(claim_index, correct, value, f"{prefix}-correct", f"{prefix}-correct")
            add(claim_index, -correct, value, f"{prefix}-wrong", f"{prefix}-wrong")
        elif evidence_type in ("repeated", "repeated_six"):
            stance = correct if correctness_override is not False else -correct
            for copy in range(6):
                add(
                    claim_index,
                    stance,
                    value,
                    f"{prefix}-source",
                    f"{prefix}-factor",
                )
        elif evidence_type == "independent_six":
            for copy in range(6):
                add(
                    claim_index,
                    correct,
                    value,
                    f"{prefix}-source-{copy}",
                    f"{prefix}-factor-{copy}",
                )
        elif evidence_type == "context_incompatible":
            stance = correct if correctness_override is not False else -correct
            add(
                claim_index,
                stance,
                value,
                f"{prefix}-source",
                f"{prefix}-factor",
                context="alternate",
                compatible=False,
            )
        else:
            raise ValueError(f"unknown evidence type: {evidence_type}")
    if nodes:
        permutation = rng.permutation(len(nodes))
        nodes = [nodes[int(index)] for index in permutation]
    return EvidenceGraph("target", tuple(nodes))


def randomize_evidence_signs(evidence: EvidenceGraph, seed: int) -> EvidenceGraph:
    rng = np.random.default_rng(seed)
    return EvidenceGraph(
        evidence.target_context,
        tuple(
            replace(node, stance=1 if rng.random() < 0.5 else -1, correct=False)
            for node in evidence.nodes
        ),
    )


def _evidence_nodes_for_mode(
    evidence: EvidenceGraph,
    mode: Literal["ceg_fuse", "naive_soft", "oracle"],
) -> tuple[tuple[EvidenceNode, ...], bool]:
    if mode == "naive_soft":
        return evidence.nodes, True
    nodes = evidence.representatives()
    if mode == "oracle":
        nodes = tuple(node for node in nodes if node.correct and node.context_compatible)
    return nodes, False


def _hard_evidence_compatibility(
    claim_values: np.ndarray,
    nodes: Sequence[EvidenceNode],
    *,
    ignore_context: bool,
) -> np.ndarray:
    compatible = np.ones(claim_values.shape[0], dtype=bool)
    for node in nodes:
        if node.treatment != "hard_constraint":
            continue
        if not ignore_context and not node.context_compatible:
            continue
        desired = 1 if node.stance == 1 else 0
        compatible &= claim_values[:, node.claim_index] == desired
    return compatible


def evidence_log_preferences(
    claim_values: np.ndarray,
    evidence: EvidenceGraph,
    mode: Literal["ceg_fuse", "naive_soft", "oracle"] = "ceg_fuse",
    *,
    soft_confidence_cap: float = DEFAULT_SOFT_CONFIDENCE_CAP,
) -> np.ndarray:
    validate_soft_confidence_cap(soft_confidence_cap)
    nodes, ignore_context = _evidence_nodes_for_mode(evidence, mode)
    result = np.zeros(claim_values.shape[0], dtype=float)
    for node in nodes:
        if node.treatment == "hard_constraint":
            continue
        values = claim_values[:, node.claim_index]
        if not ignore_context and not node.context_compatible:
            continue
        resolved = values >= 0
        agrees = values == (1 if node.stance == 1 else 0)
        scores = np.where(resolved, np.where(agrees, 1.0, -1.0), 0.0)
        confidence = node.resolved_confidence(soft_confidence_cap).effective_confidence
        result += 0.5 * math.log(confidence / (1.0 - confidence)) * scores
    return result


def graph_evidence_log_preference(
    graph: Graph,
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    soft_confidence_cap: float = DEFAULT_SOFT_CONFIDENCE_CAP,
) -> float:
    """Return the centered claim-response log likelihood for one graph."""

    validate_soft_confidence_cap(soft_confidence_cap)
    result = 0.0
    for node in evidence.representatives():
        if not node.context_compatible:
            continue
        value = evaluate_claim(graph, claims[node.claim_index])
        desired = 1 if node.stance == 1 else 0
        if node.treatment == "hard_constraint":
            if value != desired:
                return -math.inf
            continue
        if value is None:
            continue
        confidence = node.resolved_confidence(soft_confidence_cap).effective_confidence
        agrees = value == desired
        result += (0.5 if agrees else -0.5) * math.log(
            confidence / (1.0 - confidence)
        )
    return result


def _reweight_bootstraps(
    ensemble: DataEnsemble,
    log_preferences: np.ndarray,
    compatible: np.ndarray | None = None,
) -> np.ndarray | None:
    if log_preferences.shape != (len(ensemble.graphs),):
        raise ValueError("preference vector has the wrong shape")
    if compatible is None:
        compatible = np.ones(len(ensemble.graphs), dtype=bool)
    else:
        compatible = np.asarray(compatible, dtype=bool)
        if compatible.shape != (len(ensemble.graphs),):
            raise ValueError("hard-support mask has the wrong shape")
    if not len(log_preferences):
        return ensemble.q_data.copy()
    if np.all(compatible) and np.all(log_preferences == log_preferences[0]):
        return ensemble.q_data.copy()
    if not np.any(compatible):
        return None
    scale = np.zeros(len(log_preferences), dtype=float)
    scale[compatible] = np.exp(
        log_preferences[compatible] - np.max(log_preferences[compatible])
    )
    weighted = ensemble.q_bootstrap * scale[None, :]
    normalizers = np.sum(weighted, axis=1)
    if np.any(normalizers <= 0):
        return None
    return np.mean(weighted / normalizers[:, None], axis=0)


def preference_fusion(
    ensemble: DataEnsemble,
    claim_values: np.ndarray,
    evidence: EvidenceGraph,
    mode: Literal["ceg_fuse", "naive_soft", "oracle"] = "ceg_fuse",
    *,
    soft_confidence_cap: float = DEFAULT_SOFT_CONFIDENCE_CAP,
) -> np.ndarray:
    validate_soft_confidence_cap(soft_confidence_cap)
    if not evidence.nodes:
        return ensemble.q_data.copy()
    nodes, ignore_context = _evidence_nodes_for_mode(evidence, mode)
    compatible = _hard_evidence_compatibility(
        claim_values, nodes, ignore_context=ignore_context
    )
    has_hard_evidence = any(
        node.treatment == "hard_constraint"
        and (ignore_context or node.context_compatible)
        for node in nodes
    )
    result = _reweight_bootstraps(
        ensemble,
        evidence_log_preferences(
            claim_values,
            evidence,
            mode,
            soft_confidence_cap=soft_confidence_cap,
        ),
        compatible,
    )
    if result is None:
        if has_hard_evidence:
            raise HardClaimInfeasibleError(
                "finite candidate support has no positive mass satisfying all hard claims"
            )
        raise RuntimeError("soft preference fusion unexpectedly has zero normalizing mass")
    return result


def _raw_majorities(
    evidence: EvidenceGraph,
    *,
    treatment: EvidenceTreatment | None = None,
) -> dict[int, int]:
    votes: dict[int, int] = {}
    for node in evidence.nodes:
        if treatment is not None and node.treatment != treatment:
            continue
        votes[node.claim_index] = votes.get(node.claim_index, 0) + node.stance
    return {claim: 1 if vote > 0 else 0 for claim, vote in votes.items() if vote != 0}


def hard_distribution(
    ensemble: DataEnsemble,
    claim_values: np.ndarray,
    evidence: EvidenceGraph,
) -> np.ndarray | None:
    majorities = _raw_majorities(evidence)
    if not majorities:
        return ensemble.q_data.copy()
    compatible = np.ones(len(ensemble.graphs), dtype=bool)
    for claim_index, desired in majorities.items():
        values = claim_values[:, claim_index]
        compatible &= (values < 0) | (values == desired)
    weighted = ensemble.q_bootstrap * compatible[None, :]
    normalizers = np.sum(weighted, axis=1)
    if np.any(normalizers <= 0):
        return None
    return np.mean(weighted / normalizers[:, None], axis=0)


def majority_distribution(
    ensemble: DataEnsemble,
    claim_values: np.ndarray,
    evidence: EvidenceGraph,
    *,
    soft_confidence_cap: float = DEFAULT_SOFT_CONFIDENCE_CAP,
) -> np.ndarray:
    validate_soft_confidence_cap(soft_confidence_cap)
    majorities = _raw_majorities(evidence, treatment="soft_evidence")
    representatives = evidence.representatives()
    compatible = _hard_evidence_compatibility(
        claim_values, representatives, ignore_context=False
    )
    has_hard_evidence = any(
        node.treatment == "hard_constraint" and node.context_compatible
        for node in representatives
    )
    if not majorities and not has_hard_evidence:
        return ensemble.q_data.copy()
    log_preferences = np.zeros(len(ensemble.graphs), dtype=float)
    for claim_index, desired in majorities.items():
        values = claim_values[:, claim_index]
        score = np.where(values < 0, 0.0, np.where(values == desired, 1.0, -1.0))
        confidences = [
            node.resolved_confidence(soft_confidence_cap).effective_confidence
            for node in representatives
            if node.claim_index == claim_index
            and node.treatment == "soft_evidence"
            and node.context_compatible
        ]
        confidence = statistics.fmean(confidences) if confidences else 0.5
        log_preferences += 0.5 * math.log(confidence / (1.0 - confidence)) * score
    result = _reweight_bootstraps(ensemble, log_preferences, compatible)
    if result is None:
        if has_hard_evidence:
            raise HardClaimInfeasibleError(
                "finite candidate support has no positive mass satisfying all hard claims"
            )
        raise RuntimeError("majority preference fusion has zero mass")
    return result


def estimator_distributions(
    ensemble: DataEnsemble,
    claim_values: np.ndarray,
    evidence: EvidenceGraph,
    random_seed: int,
) -> dict[str, np.ndarray | None]:
    randomized = randomize_evidence_signs(evidence, random_seed)
    return {
        "data_only": ensemble.q_data.copy(),
        "hard": hard_distribution(ensemble, claim_values, evidence),
        "majority": majority_distribution(ensemble, claim_values, evidence),
        "naive_soft": preference_fusion(
            ensemble, claim_values, evidence, "naive_soft"
        ),
        "ceg_fuse": preference_fusion(
            ensemble, claim_values, evidence, "ceg_fuse"
        ),
        "oracle": preference_fusion(ensemble, claim_values, evidence, "oracle"),
        "random_sign": preference_fusion(
            ensemble, claim_values, randomized, "ceg_fuse"
        ),
    }


def _edge_state(graph: Graph, first: int, second: int) -> int:
    marks = edge_marks(graph, first, second)
    states = {
        (Endpoint.NULL, Endpoint.NULL): 0,
        (Endpoint.TAIL, Endpoint.ARROW): 1,
        (Endpoint.ARROW, Endpoint.TAIL): 2,
        (Endpoint.ARROW, Endpoint.ARROW): 3,
        (Endpoint.TAIL, Endpoint.TAIL): 4,
    }
    return states[marks]


def normalized_structural_distance(estimated: Graph, truth: Graph, regime: Regime) -> float:
    if estimated.node_count != truth.node_count:
        raise ValueError("graphs have different node counts")
    pair_count = math.comb(truth.node_count, 2)
    if not pair_count:
        return 0.0
    if regime == "dag":
        differences = sum(
            _edge_state(estimated, first, second) != _edge_state(truth, first, second)
            for first in range(truth.node_count)
            for second in range(first + 1, truth.node_count)
        )
        return differences / pair_count
    endpoint_differences = sum(
        int(estimated_mark != truth_mark)
        for first in range(truth.node_count)
        for second in range(first + 1, truth.node_count)
        for estimated_mark, truth_mark in zip(
            edge_marks(estimated, first, second), edge_marks(truth, first, second)
        )
    )
    return endpoint_differences / (2 * pair_count)


def structural_hamming_distance(estimated: Graph, truth: Graph) -> int:
    return sum(
        _edge_state(estimated, first, second) != _edge_state(truth, first, second)
        for first in range(truth.node_count)
        for second in range(first + 1, truth.node_count)
    )


def _f1(true_positive: int, predicted: int, actual: int) -> float:
    if predicted == 0 and actual == 0:
        return 1.0
    return 2.0 * true_positive / (predicted + actual)


def adjacency_f1(estimated: Graph, truth: Graph) -> float:
    predicted = actual = true_positive = 0
    for first in range(truth.node_count):
        for second in range(first + 1, truth.node_count):
            estimate = has_adjacency(estimated, first, second)
            target = has_adjacency(truth, first, second)
            predicted += int(estimate)
            actual += int(target)
            true_positive += int(estimate and target)
    return _f1(true_positive, predicted, actual)


def orientation_f1(estimated: Graph, truth: Graph) -> float:
    predicted = sum(row.bit_count() for row in estimated.directed)
    actual = sum(row.bit_count() for row in truth.directed)
    true_positive = sum(
        int(has_edge(estimated, source, target))
        for source, row in enumerate(truth.directed)
        for target in iter_bits(row)
    )
    return _f1(true_positive, predicted, actual)


def endpoint_f1(estimated: Graph, truth: Graph) -> float:
    predicted = actual = true_positive = 0
    for first in range(truth.node_count):
        for second in range(first + 1, truth.node_count):
            for estimate, target in zip(
                edge_marks(estimated, first, second), edge_marks(truth, first, second)
            ):
                predicted += int(estimate != Endpoint.NULL)
                actual += int(target != Endpoint.NULL)
                true_positive += int(estimate == target and target != Endpoint.NULL)
    return _f1(true_positive, predicted, actual)


def _graph_output_edge_marks(
    output: GraphOutput, first: int, second: int
) -> tuple[Endpoint, Endpoint]:
    if isinstance(output, Graph):
        return edge_marks(output, first, second)
    return output.mark(first, second), output.mark(second, first)


def point_prediction_metrics(
    estimated: GraphOutput,
    truth: Graph,
    *,
    normalized: bool = False,
) -> dict[str, float]:
    """Score one graph output directly against a ground-truth DAG.

    Distance is half the endpoint Hamming count. With ``normalized=True`` it
    is additionally divided by the number of unordered node pairs (legacy units).
    Consequently, an unoriented tail--tail
    adjacency against a correctly adjacent tail--arrow edge has distance 1/2
    for that pair.  Orientation F1 treats only exact tail--arrow endpoint pairs
    as directed predictions; partially or wholly unoriented edges abstain.
    """

    if not isinstance(estimated, (Graph, PartialGraph)):
        raise TypeError("estimated output must be a Graph or PartialGraph")
    if not isinstance(truth, Graph):
        raise TypeError("truth must be a Graph")
    if truth.regime != "dag":
        raise ValueError("point-prediction metrics require a ground-truth DAG")
    if estimated.node_count != truth.node_count:
        raise ValueError("graphs have different node counts")

    endpoint_differences = 0
    predicted_adjacencies = 0
    actual_adjacencies = 0
    true_positive_adjacencies = 0
    predicted_directions = 0
    actual_directions = 0
    true_positive_directions = 0
    pair_count = math.comb(truth.node_count, 2)

    for first in range(truth.node_count):
        for second in range(first + 1, truth.node_count):
            estimate_marks = _graph_output_edge_marks(estimated, first, second)
            truth_marks = edge_marks(truth, first, second)
            endpoint_differences += sum(
                estimate_mark != truth_mark
                for estimate_mark, truth_mark in zip(estimate_marks, truth_marks)
            )

            estimate_adjacent = estimate_marks != (Endpoint.NULL, Endpoint.NULL)
            truth_adjacent = truth_marks != (Endpoint.NULL, Endpoint.NULL)
            predicted_adjacencies += int(estimate_adjacent)
            actual_adjacencies += int(truth_adjacent)
            true_positive_adjacencies += int(estimate_adjacent and truth_adjacent)

            estimate_direction = (
                (first, second)
                if estimate_marks == (Endpoint.TAIL, Endpoint.ARROW)
                else (second, first)
                if estimate_marks == (Endpoint.ARROW, Endpoint.TAIL)
                else None
            )
            truth_direction = (
                (first, second)
                if truth_marks == (Endpoint.TAIL, Endpoint.ARROW)
                else (second, first)
                if truth_marks == (Endpoint.ARROW, Endpoint.TAIL)
                else None
            )
            predicted_directions += int(estimate_direction is not None)
            actual_directions += int(truth_direction is not None)
            true_positive_directions += int(
                estimate_direction is not None and estimate_direction == truth_direction
            )

    distance = endpoint_differences / 2
    if normalized and pair_count:
        distance /= pair_count
    return {
        "expected_distance": float(distance),
        "energy_score": float(distance),
        "adjacency_f1": _f1(
            true_positive_adjacencies,
            predicted_adjacencies,
            actual_adjacencies,
        ),
        "orientation_f1": _f1(
            true_positive_directions,
            predicted_directions,
            actual_directions,
        ),
    }


def map_graph(graphs: Sequence[Graph], distribution: np.ndarray) -> Graph:
    probabilities = np.asarray(distribution, dtype=float)
    maximum = float(np.max(probabilities))
    candidates = [
        graph
        for graph, probability in zip(graphs, probabilities)
        if math.isclose(float(probability), maximum, rel_tol=0.0, abs_tol=1e-15)
    ]
    return min(candidates)


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(first) - np.asarray(second))))


@dataclass(frozen=True)
class MetricWorkspace:
    graphs: tuple[Graph, ...]
    truth: Graph
    regime: Regime
    distances: np.ndarray
    states: np.ndarray
    truth_index: int | None


def build_metric_workspace(
    graphs: Sequence[Graph], truth: Graph, regime: Regime
) -> MetricWorkspace:
    graph_tuple = tuple(graphs)
    if not graph_tuple:
        raise ValueError("a graph distribution requires at least one graph")
    if any(graph.node_count != truth.node_count for graph in graph_tuple):
        raise ValueError("graphs have different node counts")
    if regime == "dag":
        pairs = tuple(
            (first, second)
            for first in range(truth.node_count)
            for second in range(first + 1, truth.node_count)
        )
        states = np.asarray(
            [
                [_edge_state(graph, first, second) for first, second in pairs]
                for graph in graph_tuple
            ],
            dtype=np.int8,
        )
        truth_states = np.asarray(
            [_edge_state(truth, first, second) for first, second in pairs],
            dtype=np.int8,
        )
    else:
        states = np.asarray(
            [
                [
                    int(mark)
                    for first in range(truth.node_count)
                    for second in range(first + 1, truth.node_count)
                    for mark in edge_marks(graph, first, second)
                ]
                for graph in graph_tuple
            ],
            dtype=np.int8,
        )
        truth_states = np.asarray(
            [
                int(mark)
                for first in range(truth.node_count)
                for second in range(first + 1, truth.node_count)
                for mark in edge_marks(truth, first, second)
            ],
            dtype=np.int8,
        )
    distances = (np.mean(states != truth_states[None, :], axis=1)
                 if states.shape[1] else np.zeros(len(graph_tuple)))
    truth_index = next(
        (index for index, graph in enumerate(graph_tuple) if graph == truth), None
    )
    return MetricWorkspace(
        graph_tuple,
        truth,
        regime,
        np.asarray(distances, dtype=float),
        states,
        truth_index,
    )


def distribution_metrics(
    graphs: Sequence[Graph],
    distribution: np.ndarray,
    truth: Graph,
    regime: Regime,
    mass_level: float = 0.90,
    neighborhood_radius: float = 0.10,
    workspace: MetricWorkspace | None = None,
    *,
    normalized: bool = False,
) -> dict[str, float]:
    """Evaluate ED/ES in structural-error counts by default.

    ``normalized=True`` reproduces legacy ED/ES. Workspace distances,
    neighborhood radii, and mode_distance retain their normalized units.
    """
    probabilities = np.asarray(distribution, dtype=float)
    if probabilities.shape != (len(graphs),):
        raise ValueError("graph distribution has the wrong shape")
    if not np.isclose(np.sum(probabilities), 1.0) or np.any(probabilities < 0):
        raise ValueError("graph distribution must be nonnegative and sum to one")
    prepared = workspace or build_metric_workspace(graphs, truth, regime)
    if (
        prepared.graphs != tuple(graphs)
        or prepared.truth != truth
        or prepared.regime != regime
    ):
        raise ValueError("metric workspace does not match the requested distribution")
    distances = prepared.distances
    expected_distance = float(probabilities @ distances)
    state_count = 5 if regime == "dag" else 4
    squared_mass = np.zeros(prepared.states.shape[1], dtype=float)
    for state in range(state_count):
        state_mass = (prepared.states == state).T @ probabilities
        squared_mass += state_mass * state_mass
    dispersion = float(np.mean(1.0 - squared_mass)) if squared_mass.size else 0.0
    scale = 1 if normalized else prepared.states.shape[1]
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    order = np.argsort(-probabilities, kind="stable")
    cutoff = int(np.searchsorted(np.cumsum(probabilities[order]), mass_level, side="left")) + 1
    high_mass_indices = set(int(index) for index in order[:cutoff])
    mode = map_graph(graphs, probabilities)
    return {
        "expected_distance": expected_distance * scale,
        "energy_score": (expected_distance - 0.5 * dispersion) * scale,
        "entropy": entropy,
        "high_mass_set_size": float(cutoff),
        "high_mass_truth_coverage": float(
            prepared.truth_index is not None and prepared.truth_index in high_mass_indices
        ),
        "neighborhood_mass": float(np.sum(probabilities[distances <= neighborhood_radius])),
        "mode_distance": normalized_structural_distance(mode, truth, regime),
        "adjacency_f1": adjacency_f1(mode, truth),
        "orientation_f1": (
            orientation_f1(mode, truth) if regime == "dag" else math.nan
        ),
        "endpoint_f1": endpoint_f1(mode, truth) if regime == "mag" else math.nan,
    }


def graph_edges(graph: Graph) -> dict[str, list[list[int]]]:
    return {
        "directed": [
            [source, target]
            for source, row in enumerate(graph.directed)
            for target in iter_bits(row)
        ],
        "bidirected": [
            [first, second]
            for first in range(graph.node_count)
            for second in range(first + 1, graph.node_count)
            if has_bidirected(graph, first, second)
        ],
        "undirected": [
            [first, second]
            for first in range(graph.node_count)
            for second in range(first + 1, graph.node_count)
            if has_undirected(graph, first, second)
        ],
    }


def py_tetrad_jar() -> Path | None:
    try:
        import importlib.resources as resources

        return Path(str(resources.files("pytetrad").joinpath("resources", "tetrad-current.jar")))
    except (ImportError, ModuleNotFoundError):
        return None
