"""Confidence-aware active causal-claim acquisition for DAG particles."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from cd_llm.core.finite_support import (
    Claim,
    EvidenceGraph,
    EvidenceNode,
    EvidenceTreatment,
    Graph,
    HardClaimInfeasibleError,
    evaluate_claim,
    has_adjacency,
    has_edge,
    iter_bits,
    resolve_evidence_confidence,
    transitive_closure,
    validate_soft_confidence_cap,
)
from cd_llm.methods.adaptive_pruning import normalize_distribution
from cd_llm.methods.evidence_guided import EvidenceGuidedDAGProposal
from cd_llm.methods.graph_scores import DecomposableDAGScore
from cd_llm.methods.source_pair_evidence import SourcePairEvidence


QueryKind = Literal["directed", "adjacency", "ancestry", "ordering"]
QueryFamily = Literal["local_edge", "causal_order", "ancestry"]
QueryPolicy = Literal["eig", "random", "answerability_eig", "alternating", "fixed"]
GraphDistribution = dict[Graph, float]


@dataclass(frozen=True, order=True)
class ClaimQuery:
    kind: QueryKind
    source: int
    target: int
    query_id: str


@dataclass(frozen=True)
class QueryResponse:
    """One auditable expert response to a selected causal predicate."""

    answer: int | None
    reported_confidence: float | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    abstention_reason: str | None = None
    audit: Mapping[str, Any] = field(default_factory=dict)
    treatment: EvidenceTreatment | None = None

    def __post_init__(self) -> None:
        if self.answer is None:
            if self.reported_confidence is not None:
                raise ValueError("an abstention must not carry confidence")
        elif not isinstance(self.answer, (bool, int, np.integer)) or int(
            self.answer
        ) not in (0, 1):
            raise ValueError("query answer must be zero, one, or None")
        elif self.reported_confidence is None or not math.isfinite(
            float(self.reported_confidence)
        ) or not 0.5 <= float(self.reported_confidence) <= 1.0:
            raise ValueError("answered-query confidence must lie in [0.5, 1]")
        else:
            resolve_evidence_confidence(
                float(self.reported_confidence), self.treatment
            )
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        if self.attempts < 1:
            raise ValueError("attempt count must be positive")

    @property
    def valid(self) -> bool:
        return self.answer is not None

    @property
    def confidence(self) -> float | None:
        """Backward-compatible alias for the reported confidence."""

        return self.reported_confidence


AnswerProvider = Callable[[ClaimQuery], int | QueryResponse]


@dataclass(frozen=True)
class AnswerabilityEstimate:
    """A pre-response estimate that a claim will yield usable evidence.

    This object is used only for acquisition.  It is never converted into a
    causal claim and therefore cannot update particle weights.
    """

    probability: float
    can_assess: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    physical_requests: int = 1
    audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("answerability probability must lie in [0, 1]")
        if not isinstance(self.can_assess, bool):
            raise TypeError("can_assess must be boolean")
        if not self.can_assess and probability != 0.0:
            raise ValueError("a negative answerability gate must have probability zero")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("answerability token counts must be nonnegative")
        if self.attempts < 1 or self.physical_requests < 0:
            raise ValueError("answerability request counts are invalid")


AnswerabilityProvider = Callable[[ClaimQuery], AnswerabilityEstimate]


@dataclass(frozen=True)
class QueryStep:
    budget: int
    query: ClaimQuery
    predicate_probability: float
    positive_response_probability: float
    observed_response_probability: float | None
    expected_information_gain: float
    answer: int | None
    reported_confidence: float | None
    ess_before: float
    ess_after: float
    resampled: bool
    support_recovered: bool
    cumulative_seconds: float
    applied_claim: Claim | None = None
    applied_likelihood_normalizer: float | None = None
    unique_positive_graphs_before: int = 0
    unique_positive_graphs_after: int = 0
    graph_mass_ess_before: float = 0.0
    graph_mass_ess_after: float = 0.0
    unique_resampling_ancestors: int | None = None
    rejuvenation_attempted_moves: int = 0
    rejuvenation_accepted_moves: int = 0
    rejuvenation_self_proposals: int = 0
    rejuvenation_guided_proposals: int = 0
    rejuvenation_repair_fallbacks: int = 0
    rejuvenation_acceptance_rate: float = 0.0
    rejuvenation_hard_constraint_rejections: int = 0
    support_recovery_strategy: str | None = None
    support_recovery_projection_attempts: int = 0
    support_recovery_projected_particles: int = 0
    support_recovery_unique_graphs: int = 0
    selection_reported_confidence: float = 0.9
    valid_response: bool = True
    abstention_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_attempts: int = 1
    response_audit: Mapping[str, Any] = field(default_factory=dict)
    effective_confidence: float | None = None
    treatment: EvidenceTreatment | None = None
    selection_effective_confidence: float = 0.9
    selection_treatment: EvidenceTreatment = "soft_evidence"
    answerability_probability: float = 1.0
    answerability_adjusted_information_gain: float | None = None
    answerability_screened_query_ids: tuple[str, ...] = ()
    answerability_screening_batches: tuple[tuple[str, ...], ...] = ()
    answerability_assessment_count: int = 0
    answerability_physical_requests: int = 0
    answerability_prompt_tokens: int = 0
    answerability_completion_tokens: int = 0

    # None in archived traces; populated by the current SMC implementation.
    selection_rule: str | None = None
    source_pair_update: Mapping[str, Any] | None = None

    @property
    def confidence(self) -> float | None:
        """Backward-compatible alias for the reported response confidence."""

        return self.reported_confidence

    @property
    def selection_confidence(self) -> float:
        """Backward-compatible alias for the reported selection confidence."""

        return self.selection_reported_confidence


@dataclass(frozen=True)
class ActiveQueryResult:
    distributions: dict[int, GraphDistribution]
    cumulative_seconds: dict[int, float]
    steps: tuple[QueryStep, ...]
    particles: tuple[Graph, ...]
    weights: tuple[float, ...]
    resampling_count: int
    support_recovery_count: int
    timing: "ActiveQueryTiming"
    query_count: int = 0
    answered_query_count: int = 0
    abstention_count: int = 0
    answerability_assessment_count: int = 0
    answerability_physical_requests: int = 0
    answerability_prompt_tokens: int = 0
    answerability_completion_tokens: int = 0
    attempted_causal_queries: int = 0
    unused_query_budget: int = 0
    early_stop_reason: str | None = None
    budget_unit: Literal["attempted", "answered"] = "attempted"


@dataclass(frozen=True)
class ActiveQueryTiming:
    """Exclusive wall-clock accounting for one active-query call."""

    setup_seconds: float
    selection_seconds: float
    oracle_seconds: float
    conditioning_seconds: float
    support_recovery_seconds: float
    resampling_seconds: float
    rejuvenation_seconds: float
    aggregation_seconds: float
    bookkeeping_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class RejuvenationDiagnostics:
    """Move diagnostics accumulated by one population rejuvenation call."""

    attempted_moves: int = 0
    accepted_moves: int = 0
    self_proposals: int = 0
    guided_branch_proposals: int = 0
    repair_fallbacks: int = 0
    hard_constraint_rejections: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.attempted_moves == 0:
            return 0.0
        return self.accepted_moves / self.attempted_moves


@dataclass(frozen=True)
class _RejuvenationOutcome:
    particles: tuple[Graph, ...]
    diagnostics: RejuvenationDiagnostics


@dataclass(frozen=True)
class _HardConstraintSpecification:
    required_edges: tuple[tuple[int, int], ...]
    flexible_adjacencies: tuple[tuple[int, int], ...]
    forbidden_edges: frozenset[tuple[int, int]]
    forbidden_adjacencies: frozenset[tuple[int, int]]
    required_ancestries: frozenset[tuple[int, int]]
    forbidden_ancestries: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class _HardSupportRecoveryOutcome:
    particles: tuple[Graph, ...]
    projection_attempts: int
    projected_particles: int
    unique_graphs: int


@dataclass(frozen=True)
class _QueryIndex:
    sources: np.ndarray
    targets: np.ndarray
    kinds: np.ndarray


_QUERY_KIND_CODE: dict[QueryKind, int] = {
    "directed": 0,
    "adjacency": 1,
    "ancestry": 2,
    "ordering": 3,
}


def dag_query_pool(
    node_count: int, families: Sequence[QueryFamily]
) -> tuple[ClaimQuery, ...]:
    """Return the complete predicate pool for selected observable DAG families."""

    if node_count < 2:
        raise ValueError("active querying requires at least two nodes")
    selected = tuple(families)
    allowed = ("local_edge", "causal_order", "ancestry")
    if not selected or len(selected) != len(set(selected)) or any(
        family not in allowed for family in selected
    ):
        raise ValueError(f"query families must be unique values from {allowed}")
    queries: list[ClaimQuery] = []
    if "local_edge" in selected:
        queries.extend(
            ClaimQuery("directed", source, target, f"directed-{source}-{target}")
            for source in range(node_count)
            for target in range(node_count)
            if source != target
        )
        queries.extend(
            ClaimQuery("adjacency", first, second, f"adjacency-{first}-{second}")
            for first in range(node_count)
            for second in range(first + 1, node_count)
        )
    if "causal_order" in selected:
        queries.extend(
            ClaimQuery("ordering", source, target, f"ordering-{source}-{target}")
            for source in range(node_count)
            for target in range(node_count)
            if source != target
        )
    if "ancestry" in selected:
        queries.extend(
            ClaimQuery("ancestry", source, target, f"ancestry-{source}-{target}")
            for source in range(node_count)
            for target in range(node_count)
            if source != target
        )
    return tuple(sorted(queries))


def ancestry_ordering_query_links(
    queries: Sequence[ClaimQuery],
) -> dict[str, str]:
    """Link ordering predicates to their reverse-ancestry complements."""

    by_semantic = {
        (query.kind, query.source, query.target): query.query_id for query in queries
    }
    links: dict[str, str] = {}
    for query in queries:
        if query.kind == "ancestry":
            other = by_semantic.get(("ordering", query.target, query.source))
        elif query.kind == "ordering":
            other = by_semantic.get(("ancestry", query.target, query.source))
        else:
            other = None
        if other is not None:
            links[query.query_id] = other
    return links


def _nonredundant_query_indices(
    queries: Sequence[ClaimQuery], linked_query_ids: Mapping[str, str]
) -> np.ndarray:
    """Return one deterministic representative of every linked query component."""

    index_by_id = {query.query_id: index for index, query in enumerate(queries)}
    representatives = [
        index
        for index, query in enumerate(queries)
        if (
            (linked_id := linked_query_ids.get(query.query_id)) is None
            or query < queries[index_by_id[linked_id]]
        )
    ]
    return np.asarray(representatives, dtype=np.int64)


def _select_uniform_candidate_index(
    candidate_indices: np.ndarray, rng: np.random.Generator
) -> int:
    """Map a uniform integer draw bijectively to a candidate query index."""

    if not len(candidate_indices):
        raise ValueError("uniform query selection requires at least one candidate")
    return int(candidate_indices[int(rng.integers(len(candidate_indices)))])


def group_i_query_pool(node_count: int) -> tuple[ClaimQuery, ...]:
    return dag_query_pool(node_count, ("local_edge",))


def binary_entropy(probability: float) -> float:
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if value in (0.0, 1.0):
        return 0.0
    return -value * math.log(value) - (1.0 - value) * math.log(1.0 - value)


def expected_information_gain(truth_probability: float, confidence: float) -> float:
    theta = float(truth_probability)
    p = float(confidence)
    tolerance = 1e-12
    if not -tolerance <= theta <= 1.0 + tolerance:
        raise ValueError("query truth probability must lie in [0, 1]")
    theta = float(np.clip(theta, 0.0, 1.0))
    if not 0.5 <= p <= 1.0:
        raise ValueError("query confidence must lie in [0.5, 1]")
    response_probability = p * theta + (1.0 - p) * (1.0 - theta)
    return binary_entropy(response_probability) - binary_entropy(p)


def query_values(particles: Sequence[Graph], query: ClaimQuery) -> np.ndarray:
    if query.kind == "directed":
        return np.asarray(
            [has_edge(graph, query.source, query.target) for graph in particles],
            dtype=np.int8,
        )
    if query.kind == "adjacency":
        return np.asarray(
            [has_adjacency(graph, query.source, query.target) for graph in particles],
            dtype=np.int8,
        )
    if query.kind in ("ancestry", "ordering"):
        values: list[int] = []
        for graph in particles:
            closure = transitive_closure(graph)
            if query.kind == "ancestry":
                value = bool(closure[query.source] & (1 << query.target))
            else:
                value = not bool(closure[query.target] & (1 << query.source))
            values.append(int(value))
        return np.asarray(values, dtype=np.int8)
    raise ValueError("unknown query kind")


def _index_queries(queries: Sequence[ClaimQuery]) -> _QueryIndex:
    if any(query.kind not in _QUERY_KIND_CODE for query in queries):
        raise ValueError("unknown query kind")
    return _QueryIndex(
        sources=np.fromiter(
            (query.source for query in queries), dtype=np.int64, count=len(queries)
        ),
        targets=np.fromiter(
            (query.target for query in queries), dtype=np.int64, count=len(queries)
        ),
        kinds=np.fromiter(
            (_QUERY_KIND_CODE[query.kind] for query in queries),
            dtype=np.int8,
            count=len(queries),
        ),
    )


def _directed_edge_marginals(
    particles: Sequence[Graph], weights: np.ndarray, node_count: int
) -> np.ndarray:
    marginals = np.zeros(node_count * node_count, dtype=float)
    graph_masses: dict[Graph, float] = {}
    for graph, weight in zip(particles, weights):
        graph_masses[graph] = graph_masses.get(graph, 0.0) + float(weight)
    for graph, weight in graph_masses.items():
        for source, row in enumerate(graph.directed):
            offset = source * node_count
            for target in iter_bits(row):
                marginals[offset + target] += weight
    return marginals


def _ancestry_marginals(
    particles: Sequence[Graph], weights: np.ndarray, node_count: int
) -> np.ndarray:
    marginals = np.zeros(node_count * node_count, dtype=float)
    graph_masses: dict[Graph, float] = {}
    for graph, weight in zip(particles, weights):
        graph_masses[graph] = graph_masses.get(graph, 0.0) + float(weight)
    for graph, weight in graph_masses.items():
        for source, descendants in enumerate(transitive_closure(graph)):
            offset = source * node_count
            for target in iter_bits(descendants):
                marginals[offset + target] += weight
    return marginals


def _indexed_query_truth_probabilities(
    particles: Sequence[Graph],
    weights: np.ndarray,
    query_index: _QueryIndex,
    available_indices: np.ndarray,
    node_count: int,
) -> np.ndarray:
    directed_marginals = _directed_edge_marginals(particles, weights, node_count)
    sources = query_index.sources[available_indices]
    targets = query_index.targets[available_indices]
    kinds = query_index.kinds[available_indices]
    result = directed_marginals[sources * node_count + targets].copy()
    adjacency = kinds == _QUERY_KIND_CODE["adjacency"]
    if np.any(adjacency):
        result[adjacency] += directed_marginals[
            targets[adjacency] * node_count + sources[adjacency]
        ]
    needs_closure = np.any(kinds >= _QUERY_KIND_CODE["ancestry"])
    if needs_closure:
        ancestry_marginals = _ancestry_marginals(
            particles, weights, node_count
        )
        ancestry = kinds == _QUERY_KIND_CODE["ancestry"]
        if np.any(ancestry):
            result[ancestry] = ancestry_marginals[
                sources[ancestry] * node_count + targets[ancestry]
            ]
        ordering = kinds == _QUERY_KIND_CODE["ordering"]
        if np.any(ordering):
            result[ordering] = 1.0 - ancestry_marginals[
                targets[ordering] * node_count + sources[ordering]
            ]
    return np.clip(result, 0.0, 1.0)


def _query_truth_probabilities(
    particles: Sequence[Graph],
    weights: np.ndarray,
    queries: Sequence[ClaimQuery],
    node_count: int,
) -> np.ndarray:
    """Return Group-I--II predicate marginals without rescanning per query."""

    index = _index_queries(queries)
    available = np.arange(len(queries), dtype=np.int64)
    return _indexed_query_truth_probabilities(
        particles, weights, index, available, node_count
    )


def _information_gains(
    probabilities: np.ndarray, confidence: float
) -> np.ndarray:
    """Evaluate scalar EIG once for each exactly distinct predicate marginal."""

    cache: dict[float, float] = {}
    result = np.empty(len(probabilities), dtype=float)
    for index, probability in enumerate(probabilities):
        theta = float(probability)
        if theta not in cache:
            cache[theta] = expected_information_gain(theta, confidence)
        result[index] = cache[theta]
    return result


def _normalized_weights(weights: Sequence[float], count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,) or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("particle weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("particle weights must have positive mass")
    return values / total


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    positions = (float(rng.random()) + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _aggregate(particles: Sequence[Graph], weights: np.ndarray) -> GraphDistribution:
    masses: dict[Graph, float] = {}
    for graph, weight in zip(particles, weights):
        masses[graph] = masses.get(graph, 0.0) + float(weight)
    return normalize_distribution(masses)


def _graph_mass_diagnostics(
    particles: Sequence[Graph], weights: np.ndarray
) -> tuple[int, float]:
    masses: dict[Graph, float] = {}
    for graph, weight in zip(particles, weights):
        value = float(weight)
        if value > 0.0:
            masses[graph] = masses.get(graph, 0.0) + value
    if not masses:
        return 0, 0.0
    graph_ess = 1.0 / math.fsum(mass * mass for mass in masses.values())
    return len(masses), graph_ess


def _add_rejuvenation_diagnostics(
    first: RejuvenationDiagnostics, second: RejuvenationDiagnostics
) -> RejuvenationDiagnostics:
    return RejuvenationDiagnostics(
        attempted_moves=first.attempted_moves + second.attempted_moves,
        accepted_moves=first.accepted_moves + second.accepted_moves,
        self_proposals=first.self_proposals + second.self_proposals,
        guided_branch_proposals=(
            first.guided_branch_proposals + second.guided_branch_proposals
        ),
        repair_fallbacks=first.repair_fallbacks + second.repair_fallbacks,
        hard_constraint_rejections=(
            first.hard_constraint_rejections
            + second.hard_constraint_rejections
        ),
    )


def _resolved_claim(query: ClaimQuery, answer: int, index: int) -> Claim:
    return Claim(
        f"AQ-{index}-{query.query_id}",
        "I" if query.kind in ("directed", "adjacency") else "II",
        query.kind,
        int(answer),
        query.source,
        query.target,
    )


def _coerce_query_response(
    raw_response: int | QueryResponse,
    default_confidence: float,
    default_treatment: EvidenceTreatment,
) -> QueryResponse:
    if isinstance(raw_response, QueryResponse):
        return raw_response
    if not isinstance(raw_response, (bool, int, np.integer)) or int(
        raw_response
    ) not in (0, 1):
        raise ValueError(
            "answer_provider must return zero or one, or a QueryResponse"
        )
    return QueryResponse(
        int(raw_response),
        float(default_confidence),
        treatment=default_treatment,
    )


def _evidence_graph(
    claims: Sequence[Claim],
    confidence: float | Sequence[float],
    treatments: Sequence[EvidenceTreatment],
) -> EvidenceGraph:
    if isinstance(confidence, (int, float, np.floating)):
        confidences = (float(confidence),) * len(claims)
    else:
        confidences = tuple(float(value) for value in confidence)
    if len(confidences) != len(claims):
        raise ValueError("claim and confidence counts differ")
    if len(treatments) != len(claims):
        raise ValueError("claim and treatment counts differ")
    return EvidenceGraph(
        "target",
        tuple(
            EvidenceNode(
                f"aq-evidence-{index}",
                index,
                1 if claim.truth else -1,
                confidences[index],
                f"aq-source-{index}",
                f"aq-factor-{index}",
                "target",
                True,
                True,
                treatment=treatments[index],
            )
            for index, claim in enumerate(claims)
        ),
    )


def _claims_hold(graph: Graph, claims: Sequence[Claim]) -> bool:
    """Return whether a DAG lies in every resolved Group-I--II claim event."""

    closure = (
        transitive_closure(graph)
        if any(claim.kind in ("ancestry", "ordering") for claim in claims)
        else None
    )
    return all(
        evaluate_claim(graph, claim, closure) == claim.truth for claim in claims
    )


def _hard_constraint_specification(
    claims: Sequence[Claim], node_count: int
) -> _HardConstraintSpecification:
    """Compile exact Group-I--II answers into DAG constraints.

    A positive adjacency with one explicitly excluded direction becomes a
    required directed edge.  Adjacencies whose two orientations remain legal
    are left flexible and are oriented when a particle is projected.
    """

    required_edges: set[tuple[int, int]] = set()
    forbidden_edges: set[tuple[int, int]] = set()
    required_adjacencies: set[tuple[int, int]] = set()
    forbidden_adjacencies: set[tuple[int, int]] = set()
    required_ancestries: set[tuple[int, int]] = set()
    forbidden_ancestries: set[tuple[int, int]] = set()
    for claim in claims:
        if claim.group not in ("I", "II") or claim.kind not in (
            "directed",
            "directed_exclusion",
            "adjacency",
            "non_adjacency",
            "ancestry",
            "ordering",
        ):
            raise ValueError(
                "hard active-query recovery accepts resolved Group-I--II claims only"
            )
        if claim.truth not in (0, 1):
            raise ValueError("resolved claim truth must be zero or one")
        if not 0 <= claim.source < node_count or not 0 <= claim.target < node_count:
            raise ValueError("resolved claim endpoint is outside the DAG")
        if claim.source == claim.target:
            raise ValueError("resolved claim endpoints must be distinct")
        pair = tuple(sorted((claim.source, claim.target)))
        if claim.kind in ("directed", "directed_exclusion"):
            edge = (claim.source, claim.target)
            requires_edge = bool(claim.truth) == (claim.kind == "directed")
            (required_edges if requires_edge else forbidden_edges).add(edge)
        elif claim.truth:
            if claim.kind in ("adjacency", "non_adjacency"):
                if claim.kind == "non_adjacency":
                    forbidden_adjacencies.add(pair)
                    continue
                required_adjacencies.add(pair)
            elif claim.kind == "ancestry":
                required_ancestries.add((claim.source, claim.target))
            else:
                forbidden_ancestries.add((claim.target, claim.source))
        elif claim.kind in ("adjacency", "non_adjacency"):
            if claim.kind == "non_adjacency":
                required_adjacencies.add(pair)
            else:
                forbidden_adjacencies.add(pair)
        elif claim.kind == "ancestry":
            forbidden_ancestries.add((claim.source, claim.target))
        else:
            required_ancestries.add((claim.target, claim.source))

    forbidden_edges.update(
        edge
        for first, second in forbidden_adjacencies
        for edge in ((first, second), (second, first))
    )
    if required_adjacencies & forbidden_adjacencies:
        raise HardClaimInfeasibleError(
            "hard constraints require and exclude the same adjacency"
        )
    conflict = required_edges & forbidden_edges
    if conflict:
        raise HardClaimInfeasibleError(
            "hard constraints require and exclude the same direction"
        )

    flexible: list[tuple[int, int]] = []
    for first, second in sorted(required_adjacencies):
        if (first, second) in required_edges or (second, first) in required_edges:
            continue
        allowed = tuple(
            edge
            for edge in ((first, second), (second, first))
            if edge not in forbidden_edges
        )
        if not allowed:
            raise HardClaimInfeasibleError(
                "hard constraints exclude both orientations of a "
                "required adjacency"
            )
        if len(allowed) == 1:
            required_edges.add(allowed[0])
        else:
            flexible.append((first, second))

    if any(
        (target, source) in required_edges for source, target in required_edges
    ):
        raise HardClaimInfeasibleError(
            "hard constraints require both directions of an adjacency"
        )
    if any(
        tuple(sorted(edge)) in forbidden_adjacencies for edge in required_edges
    ):
        raise HardClaimInfeasibleError(
            "hard constraints require an excluded adjacency"
        )
    if required_ancestries & forbidden_ancestries:
        raise HardClaimInfeasibleError(
            "hard constraints require and forbid the same ancestry"
        )
    return _HardConstraintSpecification(
        tuple(sorted(required_edges)),
        tuple(flexible),
        frozenset(forbidden_edges),
        frozenset(forbidden_adjacencies),
        frozenset(required_ancestries),
        frozenset(forbidden_ancestries),
    )


def _directed_path_exists(
    rows: Sequence[int], source: int, target: int
) -> bool:
    """Check reachability using Python integer bitsets."""

    frontier = 1 << source
    visited = 0
    target_bit = 1 << target
    while frontier:
        if frontier & target_bit:
            return True
        node_bit = frontier & -frontier
        frontier ^= node_bit
        if visited & node_bit:
            continue
        visited |= node_bit
        node = node_bit.bit_length() - 1
        frontier |= rows[node] & ~visited
    return False


def _can_add_edge(
    rows: Sequence[int],
    parent_masks: Sequence[int],
    source: int,
    target: int,
    max_parents: int,
) -> bool:
    if rows[source] & (1 << target):
        return True
    if rows[target] & (1 << source):
        return False
    if parent_masks[target].bit_count() >= max_parents:
        return False
    return not _directed_path_exists(rows, target, source)


def _add_edge(
    rows: list[int], parent_masks: list[int], source: int, target: int
) -> None:
    rows[source] |= 1 << target
    parent_masks[target] |= 1 << source


def _orient_required_adjacencies(
    rows: list[int],
    parent_masks: list[int],
    flexible: Sequence[tuple[int, int]],
    reference: Graph,
    max_parents: int,
) -> tuple[list[int], list[int]] | None:
    """Deterministically orient disjunctive adjacency constraints.

    The depth-first fallback is bounded because this is an emergency support
    expansion path.  In the benchmark there are at most 50 acquired claims,
    while the parent cap is 12, so the reference-preferred first branch is
    normally sufficient.
    """

    search_limit = max(4_096, 128 * len(flexible))
    visited_states = 0

    def search(
        index: int, current_rows: list[int], current_parents: list[int]
    ) -> tuple[list[int], list[int]] | None:
        nonlocal visited_states
        visited_states += 1
        if visited_states > search_limit:
            return None
        if index == len(flexible):
            return current_rows, current_parents
        first, second = flexible[index]
        if current_rows[first] & (1 << second) or current_rows[second] & (
            1 << first
        ):
            return search(index + 1, current_rows, current_parents)
        candidates = ((first, second), (second, first))
        candidates = tuple(
            sorted(
                candidates,
                key=lambda edge: (
                    0 if has_edge(reference, edge[0], edge[1]) else 1,
                    current_parents[edge[1]].bit_count(),
                    edge,
                ),
            )
        )
        for source, target in candidates:
            if not _can_add_edge(
                current_rows,
                current_parents,
                source,
                target,
                max_parents,
            ):
                continue
            following_rows = current_rows.copy()
            following_parents = current_parents.copy()
            _add_edge(following_rows, following_parents, source, target)
            result = search(index + 1, following_rows, following_parents)
            if result is not None:
                return result
        return None

    return search(0, rows, parent_masks)


def _project_graph_to_hard_claims(
    graph: Graph,
    claims: Sequence[Claim],
    specification: _HardConstraintSpecification,
    max_parents: int,
) -> Graph:
    """Project one data-posterior DAG into the exact claim support.

    Required edges are installed before optional edges.  Consequently a
    positive addition blocked by a directed path or a full parent set keeps the
    claim and drops the conflicting optional structure rather than failing.
    Reference edges are then restored whenever acyclicity, the parent cap, and
    all exclusions allow them.  This makes the support-entry proposal local to
    the data particle while remaining deterministic.
    """

    rows = [0] * graph.node_count
    parents = [0] * graph.node_count
    for source, target in specification.required_edges:
        rows[source] |= 1 << target
        parents[target] |= 1 << source
    if any(mask.bit_count() > max_parents for mask in parents):
        raise HardClaimInfeasibleError(
            "hard constraints exceed the DAG parent cap"
        )
    try:
        Graph(tuple(rows))
    except ValueError as error:
        raise HardClaimInfeasibleError(
            "hard directed constraints contain a cycle"
        ) from error

    oriented = _orient_required_adjacencies(
        rows,
        parents,
        specification.flexible_adjacencies,
        graph,
        max_parents,
    )
    if oriented is None:
        raise HardClaimInfeasibleError(
            "required adjacencies could not be oriented within the DAG parent cap"
        )
    rows, parents = oriented

    for source, row in enumerate(graph.directed):
        for target in iter_bits(row):
            pair = tuple(sorted((source, target)))
            if (
                (source, target) in specification.forbidden_edges
                or pair in specification.forbidden_adjacencies
                or rows[source] & (1 << target)
                or rows[target] & (1 << source)
            ):
                continue
            if _can_add_edge(rows, parents, source, target, max_parents):
                _add_edge(rows, parents, source, target)

    projected = Graph(tuple(rows))
    if any(mask.bit_count() > max_parents for mask in parents) or not _claims_hold(
        projected, claims
    ):
        raise RuntimeError("hard-claim projection produced an invalid support point")
    return projected


def _project_graph_to_mixed_hard_claims(
    graph: Graph,
    claims: Sequence[Claim],
    specification: _HardConstraintSpecification,
    max_parents: int,
) -> Graph:
    """Project a DAG into mixed local and reachability constraints by MILP.

    The optimization minimizes edge Hamming distance from ``graph``. Required
    ancestry events receive unit-flow witnesses; forbidden ancestry events use
    forward reachability propagation from their sources. The target graph is
    neither an input nor a fallback.
    """

    node_count = graph.node_count
    edges = tuple(
        (source, target)
        for source in range(node_count)
        for target in range(node_count)
        if source != target
    )
    edge_index = {edge: index for index, edge in enumerate(edges)}
    edge_count = len(edges)
    order_offset = edge_count
    required_paths = tuple(sorted(specification.required_ancestries))
    flow_offset = order_offset + node_count
    forbidden_by_source: dict[int, set[int]] = {}
    for source, target in specification.forbidden_ancestries:
        forbidden_by_source.setdefault(source, set()).add(target)
    forbidden_sources = tuple(sorted(forbidden_by_source))
    reach_offset = flow_offset + len(required_paths) * edge_count
    variable_count = reach_offset + len(forbidden_sources) * node_count

    objective = np.zeros(variable_count, dtype=float)
    tie_scale = 1e-7 / max(1, edge_count)
    for index, (source, target) in enumerate(edges):
        objective[index] = (
            -1.0 if has_edge(graph, source, target) else 1.0
        ) + tie_scale * (index + 1)
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    upper[order_offset : order_offset + node_count] = node_count - 1
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:edge_count] = 1

    for edge in specification.required_edges:
        lower[edge_index[edge]] = 1.0
        upper[edge_index[edge]] = 1.0
    for edge in specification.forbidden_edges:
        lower[edge_index[edge]] = 0.0
        upper[edge_index[edge]] = 0.0

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float], minimum: float, maximum: float
    ) -> None:
        rows.append(dict(coefficients))
        row_lower.append(float(minimum))
        row_upper.append(float(maximum))

    for first in range(node_count):
        for second in range(first + 1, node_count):
            add_constraint(
                {
                    edge_index[(first, second)]: 1.0,
                    edge_index[(second, first)]: 1.0,
                },
                -np.inf,
                1.0,
            )
    for target in range(node_count):
        add_constraint(
            {
                edge_index[(source, target)]: 1.0
                for source in range(node_count)
                if source != target
            },
            -np.inf,
            float(max_parents),
        )
    for source, target in edges:
        add_constraint(
            {
                order_offset + source: 1.0,
                order_offset + target: -1.0,
                edge_index[(source, target)]: float(node_count),
            },
            -np.inf,
            float(node_count - 1),
        )
    for first, second in specification.flexible_adjacencies:
        add_constraint(
            {
                edge_index[(first, second)]: 1.0,
                edge_index[(second, first)]: 1.0,
            },
            1.0,
            1.0,
        )

    for path_index, (path_source, path_target) in enumerate(required_paths):
        start = flow_offset + path_index * edge_count
        for edge_position, edge in enumerate(edges):
            add_constraint(
                {start + edge_position: 1.0, edge_index[edge]: -1.0},
                -np.inf,
                0.0,
            )
        for node in range(node_count):
            coefficients: dict[int, float] = {}
            for edge_position, (source, target) in enumerate(edges):
                if source == node:
                    coefficients[start + edge_position] = 1.0
                elif target == node:
                    coefficients[start + edge_position] = -1.0
            value = 1.0 if node == path_source else -1.0 if node == path_target else 0.0
            add_constraint(coefficients, value, value)

    for source_position, source in enumerate(forbidden_sources):
        start = reach_offset + source_position * node_count
        lower[start + source] = 1.0
        upper[start + source] = 1.0
        for target in forbidden_by_source[source]:
            lower[start + target] = 0.0
            upper[start + target] = 0.0
        for edge_source, edge_target in edges:
            add_constraint(
                {
                    start + edge_source: 1.0,
                    edge_index[(edge_source, edge_target)]: 1.0,
                    start + edge_target: -1.0,
                },
                -np.inf,
                1.0,
            )

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix.tocsr(),
            np.asarray(row_lower, dtype=float),
            np.asarray(row_upper, dtype=float),
        ),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise HardClaimInfeasibleError(
            f"mixed hard constraints have no recovered DAG: {result.message}"
        )
    directed = tuple(
        edge for edge, index in edge_index.items() if float(result.x[index]) > 0.5
    )
    projected = Graph(
        tuple(
            sum(1 << target for source, target in directed if source == node)
            for node in range(node_count)
        )
    )
    if any(
        sum(has_edge(projected, source, target) for source in range(node_count))
        > max_parents
        for target in range(node_count)
    ) or not _claims_hold(projected, claims):
        raise RuntimeError("mixed hard-claim projection produced an invalid DAG")
    return projected


def _recover_hard_constraint_support(
    particles: Sequence[Graph],
    claims: Sequence[Claim],
    *,
    node_count: int,
    max_parents: int,
) -> _HardSupportRecoveryOutcome:
    """Deterministically regularize a zero-mass population into hard support.

    This is a support-expansion operation, not an exact importance update: a
    finite empirical distribution with zero event mass cannot be conditioned
    directly.  Each ancestor is instead projected locally, after which the
    ordinary MH rejuvenation kernel targets the score posterior restricted to
    all acquired claims.
    """

    specification = _hard_constraint_specification(claims, node_count)
    cache: dict[Graph, Graph] = {}
    attempts = 0
    repaired: list[Graph] = []
    for graph in particles:
        attempts += 1
        projected = cache.get(graph)
        if projected is None:
            if specification.required_ancestries or specification.forbidden_ancestries:
                projected = _project_graph_to_mixed_hard_claims(
                    graph, claims, specification, max_parents
                )
            else:
                projected = _project_graph_to_hard_claims(
                    graph, claims, specification, max_parents
                )
            cache[graph] = projected
        repaired.append(projected)
    if not repaired or not all(_claims_hold(graph, claims) for graph in repaired):
        raise RuntimeError("hard-constraint recovery did not restore particle support")
    return _HardSupportRecoveryOutcome(
        tuple(repaired), attempts, len(repaired), len(set(repaired))
    )


def _rejuvenate(
    score: DecomposableDAGScore,
    particles: Sequence[Graph],
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    seed: int,
    steps: int,
    gamma: float,
    cache_size: int,
    ordinary_kernel: str = "uniform_pair",
    screened_additions: Sequence[tuple[int, int]] = (),
    hard_constraints: bool = False,
    hard_claims: Sequence[Claim] | None = None,
    soft_confidence_cap: float = 0.99,
) -> _RejuvenationOutcome:
    constrained_claims = tuple(
        claims if hard_claims is None and hard_constraints else hard_claims or ()
    )
    if constrained_claims and any(
        not _claims_hold(graph, constrained_claims) for graph in particles
    ):
        raise ValueError(
            "hard-constraint rejuvenation requires particles satisfying all claims"
        )
    if steps <= 0:
        return _RejuvenationOutcome(tuple(particles), RejuvenationDiagnostics())
    proposal = EvidenceGuidedDAGProposal(
        claims,
        evidence,
        node_count=score.node_count,
        gamma=gamma,
        cache_size=cache_size,
        max_parents=score.max_parents,
        ordinary_kernel=ordinary_kernel,
        screened_additions=screened_additions,
        soft_confidence_cap=soft_confidence_cap,
    )
    tiny = np.finfo(float).tiny
    result: list[Graph] = []
    attempted_moves = 0
    accepted_moves = 0
    self_proposals = 0
    guided_branch_proposals = 0
    repair_fallbacks = 0
    hard_constraint_rejections = 0
    for particle_index, starting in enumerate(particles):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0xA17E, particle_index])
        )
        current = starting
        current_score = score.graph_score(current)
        current_preference = proposal.evidence_score(current)
        for _ in range(steps):
            candidate, used_guided, repair_fallback = proposal._sample_candidate(
                current, rng
            )
            attempted_moves += 1
            guided_branch_proposals += int(used_guided)
            repair_fallbacks += int(repair_fallback)
            if candidate == current:
                self_proposals += 1
                continue
            if constrained_claims and not _claims_hold(candidate, constrained_claims):
                hard_constraint_rejections += 1
                continue
            forward = proposal.proposal_probability(candidate, current)
            reverse = proposal.proposal_probability(current, candidate)
            if reverse <= 0.0:
                continue
            candidate_score = score.graph_score(candidate)
            candidate_preference = proposal.evidence_score(candidate)
            log_ratio = candidate_score - current_score
            log_ratio += candidate_preference - current_preference
            log_ratio += math.log(reverse) - math.log(forward)
            if math.log(max(float(rng.random()), tiny)) < min(0.0, log_ratio):
                current = candidate
                current_score = candidate_score
                current_preference = candidate_preference
                accepted_moves += 1
        result.append(current)
    return _RejuvenationOutcome(
        tuple(result),
        RejuvenationDiagnostics(
            attempted_moves=attempted_moves,
            accepted_moves=accepted_moves,
            self_proposals=self_proposals,
            guided_branch_proposals=guided_branch_proposals,
            repair_fallbacks=repair_fallbacks,
            hard_constraint_rejections=hard_constraint_rejections,
        ),
    )


def _coerce_rejuvenation_outcome(
    value: _RejuvenationOutcome | Sequence[Graph],
) -> _RejuvenationOutcome:
    """Accept legacy test/custom hooks that return only particle slots."""

    if isinstance(value, _RejuvenationOutcome):
        return value
    return _RejuvenationOutcome(tuple(value), RejuvenationDiagnostics())


def active_query_smc(
    initial_data_distribution: Mapping[Graph, float] | None,
    score: DecomposableDAGScore,
    answer_provider: AnswerProvider,
    *,
    query_pool: Sequence[ClaimQuery] | None = None,
    linked_query_ids: Mapping[str, str] | None = None,
    budgets: Sequence[int] = (1, 3, 5, 10),
    budget_unit: Literal["attempted", "answered"] = "attempted",
    confidence: float = 0.9,
    treatment: EvidenceTreatment | None = None,
    soft_confidence_cap: float = 0.99,
    policy: QueryPolicy = "eig",
    exploration_period: int = 2,
    acquisition_seed: int | None = None,
    answerability_provider: AnswerabilityProvider | None = None,
    answerability_batch_size: int = 5,
    answerability_assessment_budget: int = 50,
    answerability_probability_upper_bound: float = 1.0,
    seed: int = 0,
    ess_threshold_fraction: float = 0.5,
    rejuvenation_steps: int = 2,
    rejuvenate_after_every_query: bool = False,
    rejuvenate_at_final_budget: bool = True,
    guided_mixture_probability: float = 0.5,
    proposal_cache_size: int = 256,
    initial_particles: Sequence[Graph] | None = None,
    initial_weights: Sequence[float] | None = None,
    ordinary_kernel: Literal["uniform_pair", "balanced"] = "uniform_pair",
    screened_additions: Sequence[tuple[int, int]] = (),
    direct_yes_interpretation: Literal["literal", "ancestry"] = "literal",
    evidence_aggregation: Literal["per_answer", "per_source_pair"] = "per_answer",
    evidence_source_family: str | None = None,
) -> ActiveQueryResult:
    """Acquire binary Group-I--II claims from a data-score particle posterior.

    ``answer_provider`` is the only component that communicates with an expert or
    simulated oracle.  The selector itself has no access to a target graph. Pass
    a legacy graph-to-mass mapping as ``initial_data_distribution``, or pass
    ``None`` together with explicit ``initial_particles`` and ``initial_weights``
    to retain duplicate particle slots. Explicit ``hard_constraint`` responses
    restrict the target support and invoke deterministic joint projection when
    the empirical population misses that support. ``soft_evidence`` responses
    contribute capped log-likelihood odds. Optional symmetric
    ``linked_query_ids`` remove logically redundant counterparts when either
    predicate is selected. ``alternating`` samples uniformly on attempted slots
    1, 1 + ``exploration_period``, and so on, using EIG on other slots.
    ``acquisition_seed`` separates uniform selection from resampling randomness;
    alternating acquisition always uses a separate stream even when omitted.
    ``direct_yes_interpretation='ancestry'`` weakens every affirmative directed
    response to positive ancestry, including one-edge paths. Negative directed
    responses retain edge-absence semantics. Selection and its response
    probabilities remain the original working acquisition model; the applied
    claim and its likelihood normalizer are logged separately. This asymmetric
    interpretation is an experimental evidence mapping, not a newly calibrated
    binary response channel.
    ``evidence_aggregation='per_source_pair'`` is an experimental soft Group I
    pooling rule. Compatible same-source answers become one intersected event
    at minimum confidence; an inconsistent pair becomes neutral. Updates use
    the new factor divided by the old factor, and mutation uses the same
    pooled evidence. The original binary EIG remains an acquisition heuristic.
    ``evidence_source_family`` must identify the fixed model and context.
    ``budget_unit='answered'`` charges only valid responses, including neutral
    confidence. Abstained predicates remain excluded. Trace ``budget`` values
    still identify attempted slots; snapshots use the selected budget unit.
    Exhausting the pool leaves unachieved answered-budget snapshots absent.
    ``policy='fixed'`` consumes the supplied pool in its original order without
    acquisition randomness or information-gain selection, for passive replay.
    ``rejuvenate_at_final_budget=False`` leaves MH moves to resampling (or the
    explicit every-query setting) at the final requested budget.
    """

    function_started = time.perf_counter()
    if budget_unit not in ("attempted", "answered"):
        raise ValueError("unknown query budget unit")
    if evidence_aggregation not in ("per_answer", "per_source_pair"):
        raise ValueError("unknown evidence aggregation")
    if evidence_aggregation == "per_source_pair" and (
        not evidence_source_family or direct_yes_interpretation != "literal"
        or treatment == "hard_constraint"
    ):
        raise ValueError("pair aggregation requires a source family and literal soft Group I answers")
    if direct_yes_interpretation not in ("literal", "ancestry"):
        raise ValueError("unknown direct yes interpretation")
    requested = tuple(sorted(set(int(value) for value in budgets)))
    if not requested or requested[0] < 1:
        raise ValueError("query budgets must be positive")
    if not callable(answer_provider):
        raise TypeError("answer_provider must be callable")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if not math.isfinite(float(confidence)) or not 0.5 <= float(confidence) <= 1.0:
        raise ValueError("active-query confidence must lie in [0.5, 1]")
    soft_confidence_cap = validate_soft_confidence_cap(soft_confidence_cap)
    pair_evidence = (
        SourcePairEvidence(evidence_source_family, soft_confidence_cap=soft_confidence_cap)
        if evidence_aggregation == "per_source_pair" else None
    )
    selection_confidence = resolve_evidence_confidence(
        float(confidence),
        treatment,
        soft_confidence_cap=soft_confidence_cap,
    )
    if not math.isfinite(float(ess_threshold_fraction)) or not (
        0.0 < float(ess_threshold_fraction) <= 1.0
    ):
        raise ValueError("ESS threshold fraction must lie in (0, 1]")
    if rejuvenation_steps < 0:
        raise ValueError("rejuvenation_steps must be nonnegative")
    if not isinstance(rejuvenate_after_every_query, bool):
        raise TypeError("rejuvenate_after_every_query must be boolean")
    if not isinstance(rejuvenate_at_final_budget, bool):
        raise TypeError("rejuvenate_at_final_budget must be boolean")
    if not math.isfinite(float(guided_mixture_probability)) or not (
        0.0 <= float(guided_mixture_probability) < 1.0
    ):
        raise ValueError("guided mixture probability must lie in [0, 1)")
    if proposal_cache_size < 1:
        raise ValueError("proposal cache size must be positive")
    if ordinary_kernel not in ("uniform_pair", "balanced"):
        raise ValueError("ordinary_kernel must be 'uniform_pair' or 'balanced'")
    raw_screened_additions = tuple(screened_additions)
    if any(
        not isinstance(addition, (tuple, list)) or len(addition) != 2
        for addition in raw_screened_additions
    ):
        raise TypeError("each screened addition must be a (source, target) pair")
    if any(
        not isinstance(source, (int, np.integer))
        or not isinstance(target, (int, np.integer))
        for source, target in raw_screened_additions
    ):
        raise TypeError("screened-addition endpoints must be integers")
    screened_additions = tuple(
        (int(source), int(target))
        for source, target in raw_screened_additions
    )
    if len(screened_additions) != len(set(screened_additions)):
        raise ValueError("screened additions must be unique")
    if any(
        source == target
        or not 0 <= source < score.node_count
        or not 0 <= target < score.node_count
        for source, target in screened_additions
    ):
        raise ValueError("screened addition endpoint is outside the DAG")

    explicit_population = initial_particles is not None or initial_weights is not None
    if explicit_population:
        if initial_particles is None or initial_weights is None:
            raise ValueError(
                "initial_particles and initial_weights must be provided together"
            )
        if initial_data_distribution is not None:
            raise ValueError(
                "provide either initial_data_distribution or explicit particle slots"
            )
        particles = tuple(initial_particles)
        weights = _normalized_weights(initial_weights, len(particles))
    else:
        if initial_data_distribution is None:
            raise ValueError("an initial particle population is required")
        distribution_items = tuple(
            (graph, float(mass))
            for graph, mass in initial_data_distribution.items()
        )
        if any(
            not math.isfinite(mass) or mass < 0.0
            for _, mass in distribution_items
        ):
            raise ValueError(
                "initial data distribution must have finite nonnegative mass"
            )
        positive_items = tuple(
            (graph, mass) for graph, mass in distribution_items if mass > 0.0
        )
        particles = tuple(graph for graph, _ in positive_items)
        weights = _normalized_weights(
            [mass for _, mass in positive_items], len(particles)
        )
    if len(particles) < 2:
        raise ValueError("active querying needs at least two particles")
    for graph in particles:
        if graph.node_count != score.node_count:
            raise ValueError("particle and score node counts differ")
        if graph.regime != "dag":
            raise ValueError("active querying currently accepts DAG particles only")
    pool = tuple(
        group_i_query_pool(score.node_count) if query_pool is None else query_pool
    )
    if not pool:
        raise ValueError("query pool must not be empty")
    if len(pool) != len(set(pool)):
        raise ValueError("query pool contains duplicate queries")
    query_ids = [query.query_id for query in pool]
    if any(not query_id for query_id in query_ids) or len(query_ids) != len(
        set(query_ids)
    ):
        raise ValueError("query identifiers must be nonempty and unique")
    semantic_queries: set[tuple[str, int, int]] = set()
    for query in pool:
        if query.kind not in _QUERY_KIND_CODE:
            raise ValueError("active querying supports Group-I--II DAG queries only")
        if not 0 <= query.source < score.node_count or not (
            0 <= query.target < score.node_count
        ):
            raise ValueError("query endpoint is outside the DAG")
        if query.source == query.target:
            raise ValueError("query endpoints must be distinct")
        if query.kind == "adjacency" and query.source > query.target:
            raise ValueError("adjacency query endpoints must be in canonical order")
        semantic = (query.kind, query.source, query.target)
        if semantic in semantic_queries:
            raise ValueError("query pool contains duplicate predicates")
        semantic_queries.add(semantic)
    query_index_by_id = {query.query_id: index for index, query in enumerate(pool)}
    links = dict(linked_query_ids or {})
    for query_id, linked_id in links.items():
        if query_id not in query_index_by_id or linked_id not in query_index_by_id:
            raise ValueError("linked query identifier is outside the query pool")
        if query_id == linked_id or links.get(linked_id) != query_id:
            raise ValueError("linked query identifiers must form symmetric pairs")
    available_indices = _nonredundant_query_indices(pool, links)
    if pair_evidence is not None and any(query.kind not in ("directed", "adjacency") for query in pool):
        raise ValueError("pair aggregation requires a Group I query pool")
    effective_pool_size = len(available_indices)
    if requested[-1] > effective_pool_size:
        raise ValueError("query budget exceeds the nonredundant candidate pool")
    if policy not in ("eig", "random", "answerability_eig", "alternating", "fixed"):
        raise ValueError("unknown query policy")
    if type(exploration_period) is not int or exploration_period < 2:
        raise ValueError("exploration period must be an integer of at least two")
    if acquisition_seed is not None and (
        not isinstance(acquisition_seed, (int, np.integer)) or int(acquisition_seed) < 0
    ):
        raise ValueError("acquisition seed must be a nonnegative integer")
    if policy == "answerability_eig" and not callable(answerability_provider):
        raise TypeError(
            "answerability_provider must be callable for answerability_eig"
        )
    if not isinstance(answerability_batch_size, (int, np.integer)) or int(
        answerability_batch_size
    ) < 1:
        raise ValueError("answerability batch size must be a positive integer")
    answerability_batch_size = int(answerability_batch_size)
    if not isinstance(answerability_assessment_budget, (int, np.integer)) or int(
        answerability_assessment_budget
    ) < 1:
        raise ValueError("answerability assessment budget must be positive")
    answerability_assessment_budget = int(answerability_assessment_budget)
    if not math.isfinite(float(answerability_probability_upper_bound)) or not (
        0.0 <= float(answerability_probability_upper_bound) <= 1.0
    ):
        raise ValueError("answerability probability upper bound must lie in [0, 1]")
    answerability_probability_upper_bound = float(
        answerability_probability_upper_bound
    )
    indexed_queries = _index_queries(pool)
    excluded_query_ids: set[str] = set()
    rng = np.random.default_rng(seed)
    # Leave legacy random runs unchanged unless an independent stream is
    # requested. Alternating acquisition always uses an independent stream.
    query_rng = (
        np.random.default_rng(acquisition_seed)
        if acquisition_seed is not None
        else np.random.default_rng(np.random.SeedSequence([seed, 71309]))
        if policy == "alternating"
        else rng
    )
    claims: list[Claim] = []
    claim_confidences: list[float] = []
    claim_treatments: list[EvidenceTreatment] = []
    hard_claims: list[Claim] = []
    traces: list[QueryStep] = []
    snapshots: dict[int, GraphDistribution] = {}
    cumulative_seconds: dict[int, float] = {}
    resampling_count = 0
    support_recovery_count = 0
    answered_query_count = 0
    answerability_cache: dict[str, AnswerabilityEstimate] = {}
    answerability_assessment_count = 0
    answerability_physical_requests = 0
    answerability_prompt_tokens = 0
    answerability_completion_tokens = 0
    early_stop_reason: str | None = None
    started = time.perf_counter()
    setup_seconds = started - function_started
    selection_seconds = 0.0
    oracle_seconds = 0.0
    conditioning_seconds = 0.0
    support_recovery_seconds = 0.0
    resampling_seconds = 0.0
    rejuvenation_seconds = 0.0
    aggregation_seconds = 0.0

    attempt_limit = requested[-1] if budget_unit == "attempted" else effective_pool_size + 1
    for query_index in range(1, attempt_limit + 1):
        if budget_unit == "answered" and answered_query_count == requested[-1]:
            break
        available_indices = np.asarray(
            [
                index
                for index in available_indices
                if pool[int(index)].query_id not in excluded_query_ids
            ],
            dtype=np.int64,
        )
        if not len(available_indices):
            if budget_unit == "answered":
                early_stop_reason = "query-pool-exhausted"
                break
            raise RuntimeError("nonredundant query pool was exhausted prematurely")
        if hard_claims and any(
            weight > 0.0 and not _claims_hold(graph, hard_claims)
            for graph, weight in zip(particles, weights)
        ):
            raise RuntimeError(
                "positive-mass particle violates an acquired hard constraint"
            )
        unique_positive_graphs_before, graph_mass_ess_before = (
            _graph_mass_diagnostics(particles, weights)
        )
        query_rejuvenation = RejuvenationDiagnostics()
        support_recovery_strategy: str | None = None
        support_recovery_projection_attempts = 0
        support_recovery_projected_particles = 0
        support_recovery_unique_graphs = 0
        selection_started = time.perf_counter()
        step_answerability_probability = 1.0
        step_adjusted_information_gain: float | None = None
        step_screened_query_ids: list[str] = []
        step_screening_batches: list[tuple[str, ...]] = []
        step_answerability_assessment_count = 0
        step_answerability_physical_requests = 0
        step_answerability_prompt_tokens = 0
        step_answerability_completion_tokens = 0
        selection_rule = (
            "random" if (query_index - 1) % exploration_period == 0 else "eig"
        ) if policy == "alternating" else policy
        if selection_rule in ("random", "fixed"):
            selected_index = (
                int(available_indices[0]) if selection_rule == "fixed"
                else _select_uniform_candidate_index(available_indices, query_rng)
            )
            selected = pool[selected_index]
            values = query_values(particles, selected)
            theta = float(np.clip(weights @ values, 0.0, 1.0))
            eig = expected_information_gain(
                theta, selection_confidence.effective_confidence
            )
        else:
            probabilities = _indexed_query_truth_probabilities(
                particles,
                weights,
                indexed_queries,
                available_indices,
                score.node_count,
            )
            information_gains = _information_gains(
                probabilities, selection_confidence.effective_confidence
            )
            if selection_rule == "eig":
                maximum = max(float(value) for value in information_gains)
                tied_positions = (
                    index
                    for index, value in enumerate(information_gains)
                    if math.isclose(
                        float(value), maximum, rel_tol=0.0, abs_tol=1e-15
                    )
                )
                selected_position = min(
                    tied_positions,
                    key=lambda index: pool[int(available_indices[index])],
                )
                step_adjusted_information_gain = maximum
            else:
                ranked_positions = sorted(
                    range(len(available_indices)),
                    key=lambda position: (
                        -float(information_gains[position]),
                        pool[int(available_indices[position])],
                    ),
                )
                assessed_positions: set[int] = {
                    position
                    for position in ranked_positions
                    if pool[int(available_indices[position])].query_id
                    in answerability_cache
                }
                cursor = 0
                while cursor < len(ranked_positions):
                    batch = ranked_positions[
                        cursor : cursor + answerability_batch_size
                    ]
                    batch_screened_query_ids: list[str] = []
                    for position in batch:
                        candidate = pool[int(available_indices[position])]
                        if candidate.query_id in answerability_cache:
                            assessed_positions.add(position)
                            continue
                        if (
                            answerability_assessment_count
                            >= answerability_assessment_budget
                        ):
                            break
                        assert answerability_provider is not None
                        estimate = answerability_provider(candidate)
                        if not isinstance(estimate, AnswerabilityEstimate):
                            raise TypeError(
                                "answerability_provider must return AnswerabilityEstimate"
                            )
                        if (
                            estimate.probability
                            > answerability_probability_upper_bound + 1e-15
                        ):
                            raise ValueError(
                                "answerability estimate exceeds its declared upper bound"
                            )
                        answerability_cache[candidate.query_id] = estimate
                        assessed_positions.add(position)
                        step_screened_query_ids.append(candidate.query_id)
                        batch_screened_query_ids.append(candidate.query_id)
                        answerability_assessment_count += 1
                        step_answerability_assessment_count += 1
                        answerability_physical_requests += estimate.physical_requests
                        step_answerability_physical_requests += estimate.physical_requests
                        answerability_prompt_tokens += estimate.prompt_tokens
                        step_answerability_prompt_tokens += estimate.prompt_tokens
                        answerability_completion_tokens += estimate.completion_tokens
                        step_answerability_completion_tokens += estimate.completion_tokens
                    if batch_screened_query_ids:
                        step_screening_batches.append(
                            tuple(batch_screened_query_ids)
                        )
                    cursor += len(batch)
                    usable_positions = [
                        position
                        for position in assessed_positions
                        if answerability_cache[
                            pool[int(available_indices[position])].query_id
                        ].can_assess
                    ]
                    if usable_positions:
                        best_position = min(
                            usable_positions,
                            key=lambda position: (
                                -float(information_gains[position])
                                * answerability_cache[
                                    pool[int(available_indices[position])].query_id
                                ].probability,
                                -answerability_cache[
                                    pool[int(available_indices[position])].query_id
                                ].probability,
                                -float(information_gains[position]),
                                pool[int(available_indices[position])],
                            ),
                        )
                        best_score = (
                            float(information_gains[best_position])
                            * answerability_cache[
                                pool[int(available_indices[best_position])].query_id
                            ].probability
                        )
                        unseen = [
                            position
                            for position in ranked_positions[cursor:]
                            if position not in assessed_positions
                        ]
                        unseen_upper_bound = (
                            max(float(information_gains[position]) for position in unseen)
                            * answerability_probability_upper_bound
                            if unseen
                            else -math.inf
                        )
                        if best_score > unseen_upper_bound + 1e-15:
                            break
                    if (
                        answerability_assessment_count
                        >= answerability_assessment_budget
                    ):
                        break
                usable_positions = [
                    position
                    for position in assessed_positions
                    if answerability_cache[
                        pool[int(available_indices[position])].query_id
                    ].can_assess
                ]
                if not usable_positions:
                    early_stop_reason = "no-assessable-candidate-within-screening-budget"
                    selection_seconds += time.perf_counter() - selection_started
                    break
                selected_position = min(
                    usable_positions,
                    key=lambda position: (
                        -float(information_gains[position])
                        * answerability_cache[
                            pool[int(available_indices[position])].query_id
                        ].probability,
                        -answerability_cache[
                            pool[int(available_indices[position])].query_id
                        ].probability,
                        -float(information_gains[position]),
                        pool[int(available_indices[position])],
                    ),
                )
                selected_estimate = answerability_cache[
                    pool[int(available_indices[selected_position])].query_id
                ]
                step_answerability_probability = selected_estimate.probability
                step_adjusted_information_gain = (
                    float(information_gains[selected_position])
                    * step_answerability_probability
                )
            selected_index = int(available_indices[selected_position])
            selected = pool[selected_index]
            theta = float(probabilities[selected_position])
            values = query_values(particles, selected)
            eig = float(information_gains[selected_position])
        if selected.query_id in excluded_query_ids:
            raise RuntimeError("active selector revisited an excluded predicate")
        excluded_query_ids.add(selected.query_id)
        linked_id = links.get(selected.query_id)
        if linked_id is not None:
            excluded_query_ids.add(linked_id)
        available_indices = np.asarray(
            [
                index
                for index in available_indices
                if pool[int(index)].query_id not in excluded_query_ids
            ],
            dtype=np.int64,
        )
        selection_seconds += time.perf_counter() - selection_started

        oracle_started = time.perf_counter()
        response = _coerce_query_response(
            answer_provider(selected),
            selection_confidence.reported_confidence,
            selection_confidence.treatment,
        )
        oracle_seconds += time.perf_counter() - oracle_started

        ess_before = float(1.0 / np.sum(weights * weights))
        if not response.valid:
            elapsed = time.perf_counter() - started
            traces.append(
                QueryStep(
                    budget=query_index,
                    selection_rule=selection_rule,
                    query=selected,
                    predicate_probability=theta,
                    positive_response_probability=(
                        selection_confidence.effective_confidence * theta
                        + (1.0 - selection_confidence.effective_confidence)
                        * (1.0 - theta)
                    ),
                    observed_response_probability=None,
                    expected_information_gain=eig,
                    answer=None,
                    reported_confidence=None,
                    ess_before=ess_before,
                    ess_after=ess_before,
                    resampled=False,
                    support_recovered=False,
                    cumulative_seconds=elapsed,
                    unique_positive_graphs_before=unique_positive_graphs_before,
                    unique_positive_graphs_after=unique_positive_graphs_before,
                    graph_mass_ess_before=graph_mass_ess_before,
                    graph_mass_ess_after=graph_mass_ess_before,
                    selection_reported_confidence=(
                        selection_confidence.reported_confidence
                    ),
                    valid_response=False,
                    abstention_reason=response.abstention_reason,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    response_attempts=response.attempts,
                    response_audit=response.audit,
                    selection_effective_confidence=(
                        selection_confidence.effective_confidence
                    ),
                    selection_treatment=selection_confidence.treatment,
                    answerability_probability=step_answerability_probability,
                    answerability_adjusted_information_gain=(
                        step_adjusted_information_gain
                    ),
                    answerability_screened_query_ids=tuple(
                        step_screened_query_ids
                    ),
                    answerability_screening_batches=tuple(
                        step_screening_batches
                    ),
                    answerability_assessment_count=(
                        step_answerability_assessment_count
                    ),
                    answerability_physical_requests=(
                        step_answerability_physical_requests
                    ),
                    answerability_prompt_tokens=step_answerability_prompt_tokens,
                    answerability_completion_tokens=(
                        step_answerability_completion_tokens
                    ),
                )
            )
            if budget_unit == "attempted" and query_index in requested:
                aggregation_started = time.perf_counter()
                snapshots[query_index] = _aggregate(particles, weights)
                cumulative_seconds[query_index] = elapsed
                aggregation_seconds += time.perf_counter() - aggregation_started
            continue

        conditioning_started = time.perf_counter()
        answer = int(response.answer)
        response_confidence = resolve_evidence_confidence(
            float(response.reported_confidence),
            response.treatment,
            soft_confidence_cap=soft_confidence_cap,
        )
        if pair_evidence is not None and response_confidence.treatment != "soft_evidence":
            raise ValueError("pair aggregation does not support hard constraints")
        positive_response_probability = (
            response_confidence.effective_confidence * theta
            + (1.0 - response_confidence.effective_confidence) * (1.0 - theta)
        )
        observed_response_probability = (
            positive_response_probability
            if answer == 1
            else 1.0 - positive_response_probability
        )
        applied_query = selected
        if direct_yes_interpretation == "ancestry" and selected.kind == "directed" and answer == 1:
            applied_query = ClaimQuery("ancestry", selected.source, selected.target, selected.query_id)
            values = query_values(particles, applied_query)
        claim = _resolved_claim(applied_query, answer, query_index - 1)
        claims.append(claim)
        claim_confidences.append(response_confidence.reported_confidence)
        claim_treatments.append(response_confidence.treatment)
        if response_confidence.treatment == "hard_constraint":
            hard_claims.append(claim)
        answered_query_count += 1
        query_cost = answered_query_count if budget_unit == "answered" else query_index
        likelihood = np.where(
            values == answer,
            response_confidence.effective_confidence,
            1.0 - response_confidence.effective_confidence,
        )
        pair_update = None
        applied_claim = claim
        if pair_evidence is not None:
            previous_group, current_group = pair_evidence.add(claim, response_confidence.reported_confidence)
            changes = {graph: current_group.log_value(graph)-previous_group.log_value(graph) for graph in set(particles)}
            likelihood = np.exp(np.asarray([changes[graph] for graph in particles]))
            pair_update = dict(before=previous_group.audit(), after=current_group.audit(),
                               update="new factor divided by previous factor", acquisition="original binary EIG heuristic")
            applied_claim = current_group.claim
        unnormalized = weights * likelihood
        applied_likelihood_normalizer = float(np.sum(unnormalized))
        conditioning_seconds += time.perf_counter() - conditioning_started
        support_recovered = False
        if float(np.sum(unnormalized)) <= 0.0:
            recovery_started = time.perf_counter()
            if response_confidence.treatment != "hard_constraint":
                raise RuntimeError(
                    "soft evidence unexpectedly produced zero particle mass"
                )
            ancestors = _systematic_resample(weights, rng)
            particles = tuple(particles[int(index)] for index in ancestors)
            weights = np.full(len(particles), 1.0 / len(particles), dtype=float)
            recovery_outcome = _recover_hard_constraint_support(
                particles,
                hard_claims,
                node_count=score.node_count,
                max_parents=score.max_parents,
            )
            particles = recovery_outcome.particles
            support_recovery_strategy = "deterministic-hard-constraint-projection"
            support_recovery_projection_attempts = (
                recovery_outcome.projection_attempts
            )
            support_recovery_projected_particles = (
                recovery_outcome.projected_particles
            )
            support_recovery_unique_graphs = recovery_outcome.unique_graphs
            values = query_values(particles, applied_query)
            likelihood = np.where(values == answer, 1.0, 0.0)
            unnormalized = weights * likelihood
            if float(np.sum(unnormalized)) <= 0.0 or any(
                not _claims_hold(graph, hard_claims) for graph in particles
            ):
                raise RuntimeError(
                    "hard-constraint recovery did not restore all acquired claims"
                )
            support_recovered = True
            support_recovery_count += 1
            support_recovery_seconds += time.perf_counter() - recovery_started

        conditioning_started = time.perf_counter()
        weights = _normalized_weights(unnormalized, len(particles))
        ess_after = float(1.0 / np.sum(weights * weights))
        resampled = support_recovered or (
            ess_after < ess_threshold_fraction * len(particles)
        )
        conditioning_seconds += time.perf_counter() - conditioning_started
        if resampled:
            resampling_started = time.perf_counter()
            ancestors = _systematic_resample(weights, rng)
            unique_resampling_ancestors = len(
                {int(index) for index in ancestors}
            )
            particles = tuple(particles[int(index)] for index in ancestors)
            weights = np.full(len(particles), 1.0 / len(particles), dtype=float)
            resampling_count += 1
            resampling_seconds += time.perf_counter() - resampling_started
        else:
            unique_resampling_ancestors = None
        if (
            rejuvenate_after_every_query
            or resampled
            or (rejuvenate_at_final_budget and query_cost == requested[-1])
        ):
            rejuvenation_started = time.perf_counter()
            if pair_evidence is None:
                mutation_claims = claims
                mutation_evidence = _evidence_graph(claims, claim_confidences, claim_treatments)
            else:
                mutation_claims, mutation_evidence = pair_evidence.effective_evidence()
            if hard_claims:
                rejuvenated_indices = tuple(
                    index
                    for index, weight in enumerate(weights)
                    if weight > 0.0
                )
            else:
                rejuvenated_indices = tuple(range(len(particles)))
            rejuvenation_outcome = _coerce_rejuvenation_outcome(
                _rejuvenate(
                    score,
                    tuple(particles[index] for index in rejuvenated_indices),
                    mutation_claims,
                    mutation_evidence,
                    seed + query_index,
                    rejuvenation_steps,
                    guided_mixture_probability,
                    proposal_cache_size,
                    ordinary_kernel,
                    screened_additions,
                    False,
                    hard_claims,
                    soft_confidence_cap,
                )
            )
            if len(rejuvenation_outcome.particles) != len(rejuvenated_indices):
                raise RuntimeError("rejuvenation changed the particle-slot count")
            updated_particles = list(particles)
            for index, graph in zip(
                rejuvenated_indices, rejuvenation_outcome.particles
            ):
                updated_particles[index] = graph
            particles = tuple(updated_particles)
            if hard_claims and any(
                weights[index] > 0.0 and not _claims_hold(graph, hard_claims)
                for index, graph in enumerate(particles)
            ):
                raise RuntimeError(
                    "hard-constraint rejuvenation left positive mass outside support"
                )
            query_rejuvenation = _add_rejuvenation_diagnostics(
                query_rejuvenation, rejuvenation_outcome.diagnostics
            )
            rejuvenation_seconds += time.perf_counter() - rejuvenation_started
        unique_positive_graphs_after, graph_mass_ess_after = (
            _graph_mass_diagnostics(particles, weights)
        )
        elapsed = time.perf_counter() - started
        traces.append(
            QueryStep(
                budget=query_index,
                selection_rule=selection_rule,
                query=selected,
                predicate_probability=theta,
                positive_response_probability=positive_response_probability,
                observed_response_probability=observed_response_probability,
                expected_information_gain=eig,
                answer=answer,
                reported_confidence=response_confidence.reported_confidence,
                applied_claim=applied_claim,
                source_pair_update=pair_update,
                applied_likelihood_normalizer=applied_likelihood_normalizer,
                ess_before=ess_before,
                ess_after=ess_after,
                resampled=resampled,
                support_recovered=support_recovered,
                cumulative_seconds=elapsed,
                unique_positive_graphs_before=unique_positive_graphs_before,
                unique_positive_graphs_after=unique_positive_graphs_after,
                graph_mass_ess_before=graph_mass_ess_before,
                graph_mass_ess_after=graph_mass_ess_after,
                unique_resampling_ancestors=unique_resampling_ancestors,
                rejuvenation_attempted_moves=query_rejuvenation.attempted_moves,
                rejuvenation_accepted_moves=query_rejuvenation.accepted_moves,
                rejuvenation_self_proposals=query_rejuvenation.self_proposals,
                rejuvenation_guided_proposals=query_rejuvenation.guided_branch_proposals,
                rejuvenation_repair_fallbacks=query_rejuvenation.repair_fallbacks,
                rejuvenation_acceptance_rate=query_rejuvenation.acceptance_rate,
                rejuvenation_hard_constraint_rejections=query_rejuvenation.hard_constraint_rejections,
                support_recovery_strategy=support_recovery_strategy,
                support_recovery_projection_attempts=support_recovery_projection_attempts,
                support_recovery_projected_particles=support_recovery_projected_particles,
                support_recovery_unique_graphs=support_recovery_unique_graphs,
                selection_reported_confidence=(
                    selection_confidence.reported_confidence
                ),
                valid_response=True,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                response_attempts=response.attempts,
                response_audit=response.audit,
                effective_confidence=response_confidence.effective_confidence,
                treatment=response_confidence.treatment,
                selection_effective_confidence=(
                    selection_confidence.effective_confidence
                ),
                selection_treatment=selection_confidence.treatment,
                answerability_probability=step_answerability_probability,
                answerability_adjusted_information_gain=(
                    step_adjusted_information_gain
                ),
                answerability_screened_query_ids=tuple(step_screened_query_ids),
                answerability_screening_batches=tuple(step_screening_batches),
                answerability_assessment_count=(
                    step_answerability_assessment_count
                ),
                answerability_physical_requests=(
                    step_answerability_physical_requests
                ),
                answerability_prompt_tokens=step_answerability_prompt_tokens,
                answerability_completion_tokens=(
                    step_answerability_completion_tokens
                ),
            )
        )
        if query_cost in requested:
            aggregation_started = time.perf_counter()
            snapshots[query_cost] = _aggregate(particles, weights)
            cumulative_seconds[query_cost] = elapsed
            aggregation_seconds += time.perf_counter() - aggregation_started

    if early_stop_reason is not None and budget_unit == "attempted":
        aggregation_started = time.perf_counter()
        elapsed = time.perf_counter() - started
        current = _aggregate(particles, weights)
        for budget in requested:
            if budget not in snapshots:
                snapshots[budget] = current
                cumulative_seconds[budget] = elapsed
        aggregation_seconds += time.perf_counter() - aggregation_started

    total_seconds = time.perf_counter() - function_started
    named_seconds = (
        setup_seconds
        + selection_seconds
        + oracle_seconds
        + conditioning_seconds
        + support_recovery_seconds
        + resampling_seconds
        + rejuvenation_seconds
        + aggregation_seconds
    )
    bookkeeping_seconds = max(0.0, total_seconds - named_seconds)

    return ActiveQueryResult(
        distributions=snapshots,
        cumulative_seconds=cumulative_seconds,
        steps=tuple(traces),
        particles=tuple(particles),
        weights=tuple(float(value) for value in weights),
        resampling_count=resampling_count,
        support_recovery_count=support_recovery_count,
        timing=ActiveQueryTiming(
            setup_seconds,
            selection_seconds,
            oracle_seconds,
            conditioning_seconds,
            support_recovery_seconds,
            resampling_seconds,
            rejuvenation_seconds,
            aggregation_seconds,
            bookkeeping_seconds,
            total_seconds,
        ),
        query_count=len(traces),
        answered_query_count=answered_query_count,
        abstention_count=len(traces) - answered_query_count,
        answerability_assessment_count=answerability_assessment_count,
        answerability_physical_requests=answerability_physical_requests,
        answerability_prompt_tokens=answerability_prompt_tokens,
        answerability_completion_tokens=answerability_completion_tokens,
        attempted_causal_queries=len(traces),
        unused_query_budget=requested[-1] - (
            answered_query_count if budget_unit == "answered" else len(traces)
        ),
        early_stop_reason=early_stop_reason,
        budget_unit=budget_unit,
    )


__all__ = [
    "AnswerabilityEstimate",
    "AnswerabilityProvider",
    "AnswerProvider",
    "ActiveQueryResult",
    "ActiveQueryTiming",
    "ClaimQuery",
    "HardClaimInfeasibleError",
    "QueryFamily",
    "QueryResponse",
    "QueryStep",
    "RejuvenationDiagnostics",
    "active_query_smc",
    "ancestry_ordering_query_links",
    "binary_entropy",
    "dag_query_pool",
    "expected_information_gain",
    "group_i_query_pool",
    "query_values",
]
