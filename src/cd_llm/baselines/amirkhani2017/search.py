"""Deterministic greedy DAG search used by Amirkhani et al. (2017)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Literal, Mapping

from cd_llm.core.finite_support import Graph, iter_bits

from .accuracy import ExpertAccuracy
from .data_scores import DecomposableLocalScore, graph_data_log_score, parent_masks
from .em import AccuracyEMResult


ExpertScoreName = Literal["marginal", "explicit", "em"]
MoveKind = Literal["add", "delete", "reverse"]


@dataclass(frozen=True)
class AmirkhaniConfig:
    """Controls for scoring and the paper's empty-DAG hill climb."""

    expert_score: ExpertScoreName = "marginal"
    max_parents: int | None = None
    improvement_tolerance: float = 1e-12
    em_tolerance: float = 0.001
    em_max_iterations: int = 1000

    def __post_init__(self) -> None:
        if self.expert_score not in ("marginal", "explicit", "em"):
            raise ValueError("expert_score must be 'marginal', 'explicit', or 'em'")
        if self.max_parents is not None:
            if isinstance(self.max_parents, bool) or not isinstance(
                self.max_parents, Integral
            ):
                raise TypeError("max_parents must be an integer or None")
            if self.max_parents < 0:
                raise ValueError("max_parents must be nonnegative")
            object.__setattr__(self, "max_parents", int(self.max_parents))
        if isinstance(self.improvement_tolerance, bool):
            raise TypeError("improvement_tolerance must be a real number")
        tolerance = float(self.improvement_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("improvement_tolerance must be finite and nonnegative")
        if isinstance(self.em_tolerance, bool):
            raise TypeError("em_tolerance must be a real number")
        em_tolerance = float(self.em_tolerance)
        if not math.isfinite(em_tolerance) or em_tolerance <= 0.0:
            raise ValueError("em_tolerance must be positive and finite")
        if isinstance(self.em_max_iterations, bool) or not isinstance(
            self.em_max_iterations, int
        ):
            raise TypeError("em_max_iterations must be an integer")
        if self.em_max_iterations < 1:
            raise ValueError("em_max_iterations must be positive")
        object.__setattr__(self, "improvement_tolerance", tolerance)
        object.__setattr__(self, "em_tolerance", em_tolerance)


@dataclass(frozen=True)
class SearchMove:
    """One accepted greedy transformation and its resulting log score."""

    kind: MoveKind
    source: int
    target: int
    total_log_score: float
    improvement: float


@dataclass(frozen=True)
class AmirkhaniDiagnostics:
    """Deterministic search counters for reproducibility and profiling."""

    iterations: int
    candidate_moves_considered: int
    graph_scores_computed: int
    graph_score_cache_hits: int
    accepted_moves: int
    local_optimum_reached: bool


@dataclass(frozen=True)
class AmirkhaniResult:
    """Learned DAG, score decomposition, search trace, and EM metadata."""

    graph: Graph
    parent_masks: tuple[int, ...]
    total_log_score: float
    data_log_score: float
    expert_log_score: float
    data_score_name: str
    expert_score_name: ExpertScoreName
    moves: tuple[SearchMove, ...]
    diagnostics: AmirkhaniDiagnostics
    config: AmirkhaniConfig
    estimated_accuracies: Mapping[str, ExpertAccuracy] | None = None
    em_result: AccuracyEMResult | None = None

    def __post_init__(self) -> None:
        if self.graph.node_count != len(self.parent_masks):
            raise ValueError("graph and parent masks have different node counts")
        for value in (
            self.total_log_score,
            self.data_log_score,
            self.expert_log_score,
        ):
            if math.isnan(float(value)) or float(value) == math.inf:
                raise ValueError("result scores cannot be NaN or positive infinity")
        expected = math.fsum((self.data_log_score, self.expert_log_score))
        if self.total_log_score != expected and not math.isclose(
            self.total_log_score, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("total score does not equal its data and expert terms")
        if self.estimated_accuracies is not None and not isinstance(
            self.estimated_accuracies, MappingProxyType
        ):
            object.__setattr__(
                self,
                "estimated_accuracies",
                MappingProxyType(dict(self.estimated_accuracies)),
            )

    @property
    def converged(self) -> bool:
        """Whether hill climbing stopped locally and optional EM converged."""

        return self.diagnostics.local_optimum_reached and (
            self.em_result is None or self.em_result.converged
        )


@dataclass(frozen=True)
class _CandidateMove:
    kind: MoveKind
    source: int
    target: int
    graph: Graph


def _edges(graph: Graph) -> tuple[tuple[int, int], ...]:
    return tuple(
        (source, target)
        for source, row in enumerate(graph.directed)
        for target in iter_bits(row)
    )


def _with_edge(
    graph: Graph, source: int, target: int, *, present: bool
) -> Graph | None:
    rows = list(graph.directed)
    if present:
        rows[source] |= 1 << target
    else:
        rows[source] &= ~(1 << target)
    try:
        return Graph(tuple(rows))
    except ValueError:
        return None


def _legal_moves(graph: Graph, max_parents: int) -> tuple[_CandidateMove, ...]:
    """Generate legal moves in the required deterministic tie-breaking order."""

    node_count = graph.node_count
    masks = parent_masks(graph)
    moves: list[_CandidateMove] = []

    # add < delete < reverse, then source and target.
    for source in range(node_count):
        for target in range(node_count):
            if source == target:
                continue
            adjacent = bool(
                graph.directed[source] & (1 << target)
                or graph.directed[target] & (1 << source)
            )
            if adjacent or masks[target].bit_count() >= max_parents:
                continue
            candidate = _with_edge(graph, source, target, present=True)
            if candidate is not None:
                moves.append(_CandidateMove("add", source, target, candidate))

    for source, target in _edges(graph):
        candidate = _with_edge(graph, source, target, present=False)
        if candidate is None:  # pragma: no cover - deleting cannot create a cycle
            raise RuntimeError("deleting an edge unexpectedly produced an invalid DAG")
        moves.append(_CandidateMove("delete", source, target, candidate))

    for source, target in _edges(graph):
        if masks[source].bit_count() >= max_parents:
            continue
        removed = _with_edge(graph, source, target, present=False)
        if removed is None:  # pragma: no cover - deleting cannot create a cycle
            raise RuntimeError("deleting an edge unexpectedly produced an invalid DAG")
        candidate = _with_edge(removed, target, source, present=True)
        if candidate is not None:
            moves.append(_CandidateMove("reverse", source, target, candidate))
    return tuple(moves)


def _validate_local_score(score: DecomposableLocalScore) -> int:
    if not hasattr(score, "node_count") or not callable(
        getattr(score, "local_score", None)
    ):
        raise TypeError("local_score must expose node_count and local_score(node, mask)")
    node_count = getattr(score, "node_count")
    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("local_score.node_count must be an integer")
    node_count = int(node_count)
    if node_count < 1:
        raise ValueError("local_score.node_count must be positive")
    for node in range(node_count):
        value = float(score.local_score(node, 0))
        if not math.isfinite(value):
            raise ValueError("local_score must return finite values")
    return node_count


def greedy_dag_search(
    local_score: DecomposableLocalScore,
    expert_score: Callable[[Graph], float],
    *,
    config: AmirkhaniConfig,
    data_score_name: str = "precomputed",
    estimated_accuracies: Mapping[str, ExpertAccuracy] | None = None,
    em_result: AccuracyEMResult | None = None,
) -> AmirkhaniResult:
    """Run exhaustive-neighborhood hill climbing from the empty DAG.

    Each iteration evaluates every legal single-edge addition, deletion, and
    reversal. Equal-scoring candidates retain the first generated move, which
    implements the documented ``add < delete < reverse``/endpoint tie break.
    """

    if not isinstance(config, AmirkhaniConfig):
        raise TypeError("config must be AmirkhaniConfig")
    node_count = _validate_local_score(local_score)
    max_parents = (
        node_count - 1
        if config.max_parents is None
        else min(config.max_parents, node_count - 1)
    )
    current = Graph((0,) * node_count)
    score_cache: dict[Graph, tuple[float, float, float]] = {}
    computed = 0
    cache_hits = 0

    def evaluate(graph: Graph) -> tuple[float, float, float]:
        nonlocal computed, cache_hits
        cached = score_cache.get(graph)
        if cached is not None:
            cache_hits += 1
            return cached
        data_value = graph_data_log_score(graph, local_score)
        expert_value = float(expert_score(graph))
        if math.isnan(expert_value) or expert_value == math.inf:
            raise ValueError(
                "expert score cannot be NaN or positive infinity for a candidate DAG"
            )
        result = (math.fsum((data_value, expert_value)), data_value, expert_value)
        score_cache[graph] = result
        computed += 1
        return result

    current_scores = evaluate(current)
    accepted: list[SearchMove] = []
    candidates_considered = 0
    iterations = 0
    while True:
        iterations += 1
        best_move: _CandidateMove | None = None
        best_scores: tuple[float, float, float] | None = None
        for move in _legal_moves(current, max_parents):
            candidates_considered += 1
            candidate_scores = evaluate(move.graph)
            if best_scores is None or candidate_scores[0] > best_scores[0]:
                best_move = move
                best_scores = candidate_scores
        if best_move is None or best_scores is None:
            break
        if current_scores[0] == -math.inf:
            improves = best_scores[0] > -math.inf
            improvement = math.inf if improves else 0.0
        else:
            improvement = best_scores[0] - current_scores[0]
            improves = improvement > config.improvement_tolerance
        if not improves:
            break
        current = best_move.graph
        current_scores = best_scores
        accepted.append(
            SearchMove(
                best_move.kind,
                best_move.source,
                best_move.target,
                best_scores[0],
                improvement,
            )
        )

    if current_scores[0] == -math.inf:
        raise ValueError(
            "strict greedy search stopped on a zero-expert-likelihood plateau; "
            "boundary accuracies may require multiple simultaneous corrections, or "
            "the opinions may be incompatible"
        )
    diagnostics = AmirkhaniDiagnostics(
        iterations=iterations,
        candidate_moves_considered=candidates_considered,
        graph_scores_computed=computed,
        graph_score_cache_hits=cache_hits,
        accepted_moves=len(accepted),
        local_optimum_reached=True,
    )
    return AmirkhaniResult(
        graph=current,
        parent_masks=parent_masks(current),
        total_log_score=current_scores[0],
        data_log_score=current_scores[1],
        expert_log_score=current_scores[2],
        data_score_name=str(data_score_name),
        expert_score_name=config.expert_score,
        moves=tuple(accepted),
        diagnostics=diagnostics,
        config=config,
        estimated_accuracies=estimated_accuracies,
        em_result=em_result,
    )


__all__ = [
    "AmirkhaniConfig",
    "AmirkhaniDiagnostics",
    "AmirkhaniResult",
    "ExpertScoreName",
    "MoveKind",
    "SearchMove",
    "greedy_dag_search",
]
