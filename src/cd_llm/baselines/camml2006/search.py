"""Seeded Metropolis sampling over Totally Ordered Models."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from cd_llm.core.finite_support import Graph, graph_from_edges, iter_bits

from .data_scores import DecomposableLocalScore
from .priors import StructurePriorScorer, closest_topological_order
from .types import CaMMLConfig, CaMMLDiagnostics, CaMMLPrior, CaMMLResult, TOM


class CaMMLSearchError(RuntimeError):
    """The bounded TOM search could not produce a valid posterior sample."""


@dataclass(frozen=True)
class _State:
    order: tuple[int, ...]
    rows: tuple[int, ...]
    parents: tuple[int, ...]
    data_score: float
    local_prior: float
    nonlocal_prior: float

    @property
    def total(self) -> float:
        return self.data_score + self.local_prior + self.nonlocal_prior

    @property
    def graph(self) -> Graph:
        return Graph(self.rows)

    @property
    def tom(self) -> TOM:
        return TOM(self.order, self.graph)


@dataclass(frozen=True)
class _Proposal:
    order: tuple[int, ...]
    rows: tuple[int, ...]
    parents: tuple[int, ...]
    children: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    valid: bool = True


def _parent_masks(rows: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(rows)
    for source, row in enumerate(rows):
        for target in iter_bits(row):
            result[target] |= 1 << source
    return tuple(result)


def _state(
    order: tuple[int, ...],
    rows: tuple[int, ...],
    score: DecomposableLocalScore,
    prior: StructurePriorScorer,
) -> _State:
    graph = Graph(rows)
    tom = TOM(order, graph)
    parents = _parent_masks(rows)
    data = sum(score.local_score(node, parents[node]) for node in range(len(rows)))
    return _State(
        order,
        rows,
        parents,
        data,
        prior.local_log_score(tom),
        prior.nonlocal_log_score(tom),
    )


def _sample_initial_tom(
    node_count: int,
    max_parents: int,
    rng: np.random.Generator,
    prior: StructurePriorScorer,
) -> TOM | None:
    order = tuple(int(value) for value in rng.permutation(node_count))
    position = {node: index for index, node in enumerate(order)}
    rows = [0] * node_count
    indegree = [0] * node_count
    for left in range(node_count):
        for right in range(left + 1, node_count):
            table = prior.local_tables[(left, right)]
            if position[left] < position[right]:
                parent, child = left, right
                edge_weight, absent_weight = table[0], table[2]
            else:
                parent, child = right, left
                edge_weight, absent_weight = table[1], table[3]
            total = edge_weight + absent_weight
            if total == 0.0:
                return None
            take_edge = rng.random() < edge_weight / total
            if take_edge:
                if indegree[child] >= max_parents:
                    if absent_weight == 0.0:
                        return None
                    continue
                rows[parent] |= 1 << child
                indegree[child] += 1
    return TOM(order, Graph(tuple(rows)))


def _initial_state(
    score: DecomposableLocalScore,
    prior: StructurePriorScorer,
    config: CaMMLConfig,
    rng: np.random.Generator,
) -> _State:
    for graph_prior in prior.prior.graph_priors:
        if graph_prior.confidence == 1.0:
            order = closest_topological_order(
                graph_prior.graph,
                tuple(int(value) for value in rng.permutation(prior.node_count)),
            )
            candidate = _state(order, graph_prior.graph.directed, score, prior)
            if math.isfinite(candidate.total):
                return candidate
    for _ in range(config.initialization_attempts):
        tom = _sample_initial_tom(
            prior.node_count, config.max_parents, rng, prior
        )
        if tom is None:
            continue
        candidate = _state(tom.order, tom.graph.directed, score, prior)
        if math.isfinite(candidate.total):
            return candidate
    raise CaMMLSearchError(
        "no finite TOM found within initialization_attempts; priors may be infeasible"
    )


def _toggle_arc(state: _State, rng: np.random.Generator, max_parents: int) -> _Proposal:
    n = len(state.order)
    first = int(rng.integers(n))
    second = int(rng.integers(n - 1))
    if second >= first:
        second += 1
    early, late = sorted((first, second))
    parent, child = state.order[early], state.order[late]
    rows = list(state.rows)
    parents = list(state.parents)
    bit = 1 << child
    if rows[parent] & bit:
        rows[parent] ^= bit
        parents[child] ^= 1 << parent
    else:
        if parents[child].bit_count() >= max_parents:
            return _Proposal(state.order, state.rows, state.parents, (), (), False)
        rows[parent] |= bit
        parents[child] |= 1 << parent
    pair = (min(parent, child), max(parent, child))
    return _Proposal(state.order, tuple(rows), tuple(parents), (child,), (pair,))


def _swap_order(state: _State, rng: np.random.Generator, max_parents: int) -> _Proposal:
    n = len(state.order)
    if n < 2:
        return _Proposal(state.order, state.rows, state.parents, (), (), False)
    index = int(rng.integers(n - 1))
    left, right = state.order[index], state.order[index + 1]
    order = list(state.order)
    order[index], order[index + 1] = right, left
    rows = list(state.rows)
    parents = list(state.parents)
    children: tuple[int, ...] = ()
    if rows[left] & (1 << right):
        if parents[left].bit_count() >= max_parents:
            return _Proposal(state.order, state.rows, state.parents, (), (), False)
        rows[left] ^= 1 << right
        parents[right] ^= 1 << left
        rows[right] |= 1 << left
        parents[left] |= 1 << right
        children = (left, right)
    pair = min(left, right), max(left, right)
    return _Proposal(tuple(order), tuple(rows), tuple(parents), children, (pair,))


def _double_toggle(state: _State, rng: np.random.Generator, max_parents: int) -> _Proposal:
    n = len(state.order)
    if n < 3:
        return _Proposal(state.order, state.rows, state.parents, (), (), False)
    positions = sorted(int(value) for value in rng.choice(n, size=3, replace=False))
    parent1, parent2, child = (
        state.order[positions[0]],
        state.order[positions[1]],
        state.order[positions[2]],
    )
    rows = list(state.rows)
    parents = list(state.parents)
    new_parent_mask = parents[child]
    for parent in (parent1, parent2):
        new_parent_mask ^= 1 << parent
    if new_parent_mask.bit_count() > max_parents:
        return _Proposal(state.order, state.rows, state.parents, (), (), False)
    for parent in (parent1, parent2):
        rows[parent] ^= 1 << child
    parents[child] = new_parent_mask
    pairs = tuple((min(parent, child), max(parent, child)) for parent in (parent1, parent2))
    return _Proposal(state.order, tuple(rows), tuple(parents), (child,), pairs)


def _parent_swap(state: _State, rng: np.random.Generator, max_parents: int) -> _Proposal:
    del max_parents
    position = {node: index for index, node in enumerate(state.order)}
    eligible: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for child in state.order:
        earlier = state.order[: position[child]]
        current = tuple(iter_bits(state.parents[child]))
        nonparents = tuple(node for node in earlier if not state.parents[child] & (1 << node))
        if current and nonparents:
            eligible.append((child, current, nonparents))
    if not eligible:
        return _Proposal(state.order, state.rows, state.parents, (), (), False)
    child, current, nonparents = eligible[int(rng.integers(len(eligible)))]
    removed = current[int(rng.integers(len(current)))]
    added = nonparents[int(rng.integers(len(nonparents)))]
    rows = list(state.rows)
    parents = list(state.parents)
    rows[removed] ^= 1 << child
    rows[added] |= 1 << child
    parents[child] ^= 1 << removed
    parents[child] |= 1 << added
    pairs = (
        (min(removed, child), max(removed, child)),
        (min(added, child), max(added, child)),
    )
    return _Proposal(state.order, tuple(rows), tuple(parents), (child,), pairs)


def _propose(
    state: _State, rng: np.random.Generator, max_parents: int
) -> tuple[str, _Proposal]:
    value = rng.random()
    if value < 1.0 / 6.0:
        return "parent_swap", _parent_swap(state, rng, max_parents)
    if value < 1.0 / 3.0:
        return "double_toggle", _double_toggle(state, rng, max_parents)
    if value < 2.0 / 3.0:
        return "arc_toggle", _toggle_arc(state, rng, max_parents)
    return "order_swap", _swap_order(state, rng, max_parents)


def _updated_state(
    state: _State,
    proposal: _Proposal,
    score: DecomposableLocalScore,
    prior: StructurePriorScorer,
) -> _State:
    data = state.data_score
    for child in set(proposal.children):
        data -= score.local_score(child, state.parents[child])
        data += score.local_score(child, proposal.parents[child])
    old_tom = state.tom
    new_tom = TOM(proposal.order, Graph(proposal.rows))
    local = state.local_prior
    for left, right in set(proposal.pairs):
        local -= prior.local_pair_log_score(old_tom, left, right)
        local += prior.local_pair_log_score(new_tom, left, right)
    nonlocal_score = (
        prior.nonlocal_log_score(new_tom)
        if prior.global_tables or prior.prior.graph_priors
        else state.nonlocal_prior
    )
    return _State(
        proposal.order,
        proposal.rows,
        proposal.parents,
        data,
        local,
        nonlocal_score,
    )


def _split_rhat(trace: np.ndarray) -> float:
    chains, draws = trace.shape
    half = draws // 2
    if chains < 2 or half < 2:
        return math.nan
    split = np.concatenate((trace[:, :half], trace[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = float(half * np.var(np.mean(split, axis=1), ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else math.inf
    variance = ((half - 1) / half) * within + between / half
    return float(math.sqrt(max(variance / within, 0.0)))


def _ess(trace: np.ndarray) -> float:
    chains, draws = trace.shape
    centered = trace - np.mean(trace, axis=1, keepdims=True)
    variance = float(np.mean(centered * centered))
    if variance <= 0.0:
        return float(chains * draws)
    correlation_sum = 0.0
    previous = math.inf
    for lag in range(1, draws - 1, 2):
        pair = 0.0
        for offset in (lag, lag + 1):
            pair += float(
                np.mean(
                    np.sum(centered[:, :-offset] * centered[:, offset:], axis=1)
                    / (draws - offset)
                )
            ) / variance
        if pair < 0.0:
            break
        pair = min(pair, previous)
        correlation_sum += pair
        previous = pair
    return float(chains * draws / max(1.0 + 2.0 * correlation_sum, 1.0))


def sample_toms(
    local_score: DecomposableLocalScore,
    prior: CaMMLPrior,
    *,
    config: CaMMLConfig | None = None,
    data_score_name: str | None = None,
    evidence_adaptation: object | None = None,
) -> CaMMLResult:
    """Sample TOM posterior mass and aggregate it to project-native DAG support."""

    started = time.perf_counter()
    config = CaMMLConfig() if config is None else config
    if not isinstance(config, CaMMLConfig):
        raise TypeError("config must be CaMMLConfig")
    if not isinstance(prior, CaMMLPrior):
        raise TypeError("prior must be CaMMLPrior")
    node_count = getattr(local_score, "node_count", None)
    if node_count != prior.node_count:
        raise ValueError("local score and prior have different node counts")
    if config.max_parents >= prior.node_count:
        effective_max_parents = prior.node_count - 1
    else:
        effective_max_parents = config.max_parents
    scorer = StructurePriorScorer(prior, arc_probability=config.arc_probability)
    retained_per_chain = (config.iterations - config.burn_in) // config.thin
    score_trace = np.empty((config.chain_count, retained_per_chain), dtype=float)
    edge_trace = np.empty((config.chain_count, retained_per_chain), dtype=float)
    graph_counts: Counter[Graph] = Counter()
    tom_counts: Counter[TOM] = Counter()
    tom_scores: dict[TOM, float] = {}
    attempts: defaultdict[str, int] = defaultdict(int)
    accepts: defaultdict[str, int] = defaultdict(int)
    master = np.random.default_rng(config.seed)
    for chain in range(config.chain_count):
        rng = np.random.default_rng(int(master.integers(0, 2**32)))
        current = _initial_state(local_score, scorer, config, rng)
        retained = 0
        for iteration in range(config.iterations):
            kind, proposal = _propose(current, rng, effective_max_parents)
            attempts[kind] += 1
            accepted = False
            if proposal.valid:
                candidate = _updated_state(current, proposal, local_score, scorer)
                difference = (candidate.total - current.total) / config.temperature
                if math.isfinite(candidate.total) and (
                    difference >= 0.0 or math.log(rng.random()) < difference
                ):
                    current = candidate
                    accepted = True
            accepts[kind] += int(accepted)
            if (
                iteration >= config.burn_in
                and (iteration - config.burn_in) % config.thin == 0
                and retained < retained_per_chain
            ):
                tom = current.tom
                graph_counts[tom.graph] += 1
                tom_counts[tom] += 1
                tom_scores[tom] = current.total
                score_trace[chain, retained] = current.total
                edge_trace[chain, retained] = sum(row.bit_count() for row in current.rows)
                retained += 1
        if retained != retained_per_chain:
            raise CaMMLSearchError("chain retained an unexpected number of samples")
    total_samples = sum(graph_counts.values())
    if total_samples == 0:
        raise CaMMLSearchError("TOM sampler retained no finite samples")
    support = {graph: count / total_samples for graph, count in graph_counts.items()}
    max_count = max(graph_counts.values())
    map_graph = min(graph for graph, count in graph_counts.items() if count == max_count)
    map_toms = [tom for tom in tom_counts if tom.graph == map_graph]
    map_tom = min(map_toms, key=lambda tom: (-tom_counts[tom], -tom_scores[tom], tom.order))
    map_state = _state(map_tom.order, map_tom.graph.directed, local_score, scorer)
    attempted = sum(attempts.values())
    accepted = sum(accepts.values())
    diagnostics = CaMMLDiagnostics(
        attempted_moves=attempted,
        accepted_moves=accepted,
        acceptance_rate=accepted / attempted,
        proposal_attempts=dict(attempts),
        proposal_accepts=dict(accepts),
        retained_samples=total_samples,
        unique_toms=len(tom_counts),
        unique_dags=len(graph_counts),
        score_ess=_ess(score_trace),
        edge_count_ess=_ess(edge_trace),
        score_rhat=_split_rhat(score_trace),
        edge_count_rhat=_split_rhat(edge_trace),
        runtime_seconds=time.perf_counter() - started,
    )
    name = data_score_name or str(getattr(local_score, "name", type(local_score).__name__))
    return CaMMLResult(
        graph=map_graph,
        support=support,
        map_tom=map_tom,
        data_log_score=map_state.data_score,
        structure_log_prior=map_state.local_prior + map_state.nonlocal_prior,
        total_log_score=map_state.total,
        diagnostics=diagnostics,
        config=config,
        prior=prior,
        data_score_name=name,
        evidence_adaptation=evidence_adaptation,
    )


__all__ = ["CaMMLSearchError", "sample_toms"]
