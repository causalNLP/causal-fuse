"""DAG candidate-support diagnostics for the fixed synthetic stress test.

The main experiment deliberately uses the finite union of discovery outputs.
This module contains the more expensive, DAG-only support repairs evaluated in
the appendix without changing that main estimator.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from cd_llm.core.finite_support import (
    Endpoint,
    Graph,
    GraphOutput,
    NonExtendableCPDAGError,
    PartialGraph,
    edge_marks,
    graph_from_edges,
    has_adjacency,
    has_edge,
    is_compatible_completion,
    iter_bits,
    standardize,
)


GraphDistribution = dict[Graph, float]


class CompletionEnumerationLimitError(RuntimeError):
    """A CPDAG has more extensions than the declared exact-enumeration limit."""


@dataclass(frozen=True)
class PruningResult:
    """A mass-pruned distribution and its exact approximation diagnostics."""

    distribution: GraphDistribution
    removed_graphs: tuple[Graph, ...]
    mass_limit: float
    removed_mass: float
    retained_mass: float
    total_variation: float
    original_support_size: int
    retained_support_size: int


def _partial_adjacent(partial: PartialGraph, first: int, second: int) -> bool:
    return partial.mark(first, second) != Endpoint.NULL


def _partial_directed(partial: PartialGraph, source: int, target: int) -> bool:
    return (
        partial.mark(source, target) == Endpoint.TAIL
        and partial.mark(target, source) == Endpoint.ARROW
    )


def _partial_undirected(partial: PartialGraph, first: int, second: int) -> bool:
    return (
        partial.mark(first, second) == Endpoint.TAIL
        and partial.mark(second, first) == Endpoint.TAIL
    )


def enumerate_cpdag_extensions(
    output: GraphOutput,
    max_extensions: int = 50_000,
) -> tuple[Graph, ...]:
    """Enumerate all distinct Dor--Tarsi consistent extensions of a CPDAG."""

    if max_extensions < 1:
        raise ValueError("maximum extension count must be positive")
    if isinstance(output, Graph):
        if any(output.bidirected) or any(output.undirected):
            raise ValueError("DAG completion received a non-DAG point graph")
        return (output,)
    if output.kind != "cpdag":
        raise ValueError("exact DAG completion requires a CPDAG")

    fixed_rows = [0] * output.node_count
    for first in range(output.node_count):
        for second in range(first + 1, output.node_count):
            marks = (output.mark(first, second), output.mark(second, first))
            if marks == (Endpoint.TAIL, Endpoint.ARROW):
                fixed_rows[first] |= 1 << second
            elif marks == (Endpoint.ARROW, Endpoint.TAIL):
                fixed_rows[second] |= 1 << first
            elif marks not in (
                (Endpoint.NULL, Endpoint.NULL),
                (Endpoint.TAIL, Endpoint.TAIL),
            ):
                raise ValueError(f"unsupported CPDAG endpoint pair: {marks}")

    completions: set[Graph] = set()
    visited: set[tuple[int, tuple[int, ...]]] = set()

    def visit(active_mask: int, rows: tuple[int, ...]) -> None:
        state = (active_mask, rows)
        if state in visited:
            return
        visited.add(state)
        if active_mask == 0:
            graph = Graph(rows)
            if not is_compatible_completion(output, graph):
                raise RuntimeError("Dor--Tarsi produced an incompatible extension")
            completions.add(graph)
            if len(completions) > max_extensions:
                raise CompletionEnumerationLimitError(
                    f"CPDAG has more than {max_extensions} consistent extensions"
                )
            return

        active = tuple(iter_bits(active_mask))
        eligible: list[int] = []
        for node in active:
            if any(_partial_directed(output, node, other) for other in active):
                continue
            parents = {
                other for other in active if _partial_directed(output, other, node)
            }
            neighbors = {
                other for other in active if _partial_undirected(output, node, other)
            }
            adjacent = parents | neighbors
            if all(
                _partial_adjacent(output, neighbor, other)
                for neighbor in neighbors
                for other in adjacent - {neighbor}
            ):
                eligible.append(node)
        if not eligible:
            return
        for sink in eligible:
            oriented = list(rows)
            for neighbor in active:
                if _partial_undirected(output, sink, neighbor):
                    oriented[neighbor] |= 1 << sink
            visit(active_mask & ~(1 << sink), tuple(oriented))

    visit((1 << output.node_count) - 1, tuple(fixed_rows))
    if not completions:
        raise NonExtendableCPDAGError("CPDAG has no consistent DAG extension")
    return tuple(sorted(completions))


def dag_one_pair_neighbors(graph: Graph) -> tuple[Graph, ...]:
    """Return every acyclic DAG differing in exactly one unordered-pair state."""

    neighbors: set[Graph] = set()
    edges = tuple(
        (source, target)
        for source, row in enumerate(graph.directed)
        for target in iter_bits(row)
    )
    for first in range(graph.node_count):
        for second in range(first + 1, graph.node_count):
            if has_edge(graph, first, second):
                current = 1
            elif has_edge(graph, second, first):
                current = 2
            else:
                current = 0
            retained = [
                edge for edge in edges if set(edge) != {first, second}
            ]
            for proposed in (0, 1, 2):
                if proposed == current:
                    continue
                candidate_edges = list(retained)
                if proposed == 1:
                    candidate_edges.append((first, second))
                elif proposed == 2:
                    candidate_edges.append((second, first))
                try:
                    neighbors.add(graph_from_edges(graph.node_count, candidate_edges))
                except ValueError:
                    continue
    return tuple(sorted(neighbors))


def markov_equivalent_dag(first: Graph, second: Graph) -> bool:
    """Return whether two DAGs have the same skeleton and unshielded colliders."""

    if first.node_count != second.node_count:
        return False
    for left in range(first.node_count):
        for right in range(left + 1, first.node_count):
            if has_adjacency(first, left, right) != has_adjacency(second, left, right):
                return False

    def colliders(graph: Graph) -> set[tuple[int, int, int]]:
        result: set[tuple[int, int, int]] = set()
        for middle in range(graph.node_count):
            parents = [
                node
                for node in range(graph.node_count)
                if node != middle and has_edge(graph, node, middle)
            ]
            for index, left in enumerate(parents):
                for right in parents[index + 1 :]:
                    if not has_adjacency(graph, left, right):
                        result.add((min(left, right), middle, max(left, right)))
        return result

    return colliders(first) == colliders(second)


class GaussianBICScore:
    """Cached decomposable linear-Gaussian BIC log score for one dataset."""

    name = "gaussian-bic"

    def __init__(
        self,
        data: np.ndarray,
        *,
        standardize_data: bool = True,
        max_parents: int | None = None,
    ) -> None:
        values = standardize(data) if standardize_data else np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
            raise ValueError("BIC scoring requires a two-dimensional dataset")
        if not np.all(np.isfinite(values)):
            raise ValueError("BIC scoring requires finite observations")
        self.data = values
        self.sample_size, self.node_count = values.shape
        self.max_parents = min(12, self.node_count - 1) if max_parents is None else int(max_parents)
        if not 0 <= self.max_parents < self.node_count:
            raise ValueError("max_parents must lie in [0, node_count)")
        self.covariance = np.cov(values.T, ddof=0)
        self._local_cache: dict[tuple[int, int], float] = {}
        self._graph_cache: dict[Graph, float] = {}

    def local_score(self, node: int, parent_mask: int) -> float:
        if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
            raise TypeError("node must be an integer")
        if not 0 <= int(node) < self.node_count:
            raise IndexError("node is outside the score")
        if isinstance(parent_mask, bool) or not isinstance(
            parent_mask, (int, np.integer)
        ):
            raise TypeError("parent_mask must be an integer")
        node = int(node)
        parent_mask = int(parent_mask)
        if not 0 <= parent_mask < 1 << self.node_count:
            raise ValueError("parent_mask is outside the score")
        if parent_mask & (1 << node):
            raise ValueError("a node cannot be its own parent")
        if parent_mask.bit_count() > self.max_parents:
            raise ValueError("parent count exceeds score cap")
        key = (node, parent_mask)
        cached = self._local_cache.get(key)
        if cached is not None:
            return cached
        parents = list(iter_bits(parent_mask))
        variance = float(self.covariance[node, node])
        if parents:
            cross = self.covariance[np.ix_([node], parents)]
            parent_covariance = self.covariance[np.ix_(parents, parents)]
            try:
                solved = np.linalg.solve(parent_covariance, cross.T)
            except np.linalg.LinAlgError:
                solved = np.linalg.pinv(parent_covariance) @ cross.T
            variance -= float((cross @ solved)[0, 0])
        variance = max(variance, np.finfo(float).eps)
        result = -0.5 * self.sample_size * (1.0 + math.log(variance))
        result -= 0.5 * (len(parents) + 1) * math.log(self.sample_size)
        self._local_cache[key] = result
        return result

    def parent_masks(self, graph: Graph) -> tuple[int, ...]:
        parents = [0] * graph.node_count
        for source, row in enumerate(graph.directed):
            for target in iter_bits(row):
                parents[target] |= 1 << source
        result = tuple(parents)
        if any(mask.bit_count() > self.max_parents for mask in result):
            raise ValueError("graph exceeds score parent cap")
        return result

    def graph_score(self, graph: Graph) -> float:
        cached = self._graph_cache.get(graph)
        if cached is not None:
            return cached
        result = sum(
            self.local_score(node, mask)
            for node, mask in enumerate(self.parent_masks(graph))
        )
        self._graph_cache[graph] = result
        return result

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "max_parents": self.max_parents, "graph_prior": "uniform"}


def normalize_distribution(masses: Mapping[Graph, float]) -> GraphDistribution:
    total = math.fsum(float(value) for value in masses.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("graph masses must have a positive finite sum")
    result = {
        graph: float(value) / total
        for graph, value in masses.items()
        if float(value) > 0
    }
    if not result:
        raise ValueError("graph distribution has empty support")
    return result


def prune_by_mass(
    distribution: Mapping[Graph, float],
    max_removed_mass: float,
) -> PruningResult:
    """Remove the smallest masses subject to a cumulative probability bound.

    The input is normalized before pruning.  Retained probabilities are then
    renormalized, so the total-variation distance from the normalized input is
    exactly the removed mass (up to floating-point roundoff).  Graph ordering
    breaks probability ties and makes pruning deterministic.
    """

    mass_limit = float(max_removed_mass)
    if not math.isfinite(mass_limit) or not 0.0 <= mass_limit < 1.0:
        raise ValueError("maximum removed mass must lie in [0, 1)")
    if not distribution:
        raise ValueError("mass pruning needs a nonempty distribution")
    for probability in distribution.values():
        value = float(probability)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("graph probabilities must be finite and nonnegative")

    original = normalize_distribution(distribution)
    ordered = sorted(original.items(), key=lambda item: (item[1], item[0]))
    removed: list[Graph] = []
    removed_mass = 0.0
    for graph, probability in ordered:
        candidate_mass = math.fsum((removed_mass, probability))
        if candidate_mass > mass_limit:
            break
        removed.append(graph)
        removed_mass = candidate_mass
    removed_mass = math.fsum(original[graph] for graph in removed)
    while removed_mass > mass_limit:
        removed.pop()
        removed_mass = math.fsum(original[graph] for graph in removed)

    removed_set = set(removed)
    retained = {
        graph: probability
        for graph, probability in original.items()
        if graph not in removed_set
    }
    retained_mass = math.fsum(retained.values())
    pruned = normalize_distribution(retained)
    support = set(original) | set(pruned)
    total_variation = 0.5 * math.fsum(
        abs(original.get(graph, 0.0) - pruned.get(graph, 0.0))
        for graph in support
    )
    return PruningResult(
        distribution=pruned,
        removed_graphs=tuple(removed),
        mass_limit=mass_limit,
        removed_mass=removed_mass,
        retained_mass=retained_mass,
        total_variation=total_variation,
        original_support_size=len(original),
        retained_support_size=len(pruned),
    )


def score_distribution(
    graphs: Iterable[Graph], score: GaussianBICScore
) -> GraphDistribution:
    unique = tuple(set(graphs))
    if not unique:
        raise ValueError("score distribution needs at least one graph")
    log_scores = np.asarray([score.graph_score(graph) for graph in unique])
    weights = np.exp(log_scores - np.max(log_scores))
    return normalize_distribution(dict(zip(unique, weights)))


def aggregate_bootstraps(
    distributions: Sequence[Mapping[Graph, float]],
) -> GraphDistribution:
    if not distributions:
        raise ValueError("bootstrap aggregation needs at least one distribution")
    mass: Counter[Graph] = Counter()
    scale = 1.0 / len(distributions)
    for distribution in distributions:
        for graph, probability in distribution.items():
            mass[graph] += scale * float(probability)
    return normalize_distribution(mass)


def hierarchical_dag_ensemble(
    outputs_by_bootstrap: Sequence[Mapping[str, Sequence[GraphOutput]]],
    max_extensions: int = 50_000,
) -> tuple[list[GraphDistribution], list[dict[str, object]]]:
    """Weight families, successful variants, and completions hierarchically."""

    rows: list[GraphDistribution] = []
    diagnostics: list[dict[str, object]] = []
    for bootstrap, families in enumerate(outputs_by_bootstrap):
        family_completions: dict[str, list[tuple[Graph, ...]]] = {}
        failures: list[dict[str, str]] = []
        for family, outputs in families.items():
            successful: list[tuple[Graph, ...]] = []
            for variant, output in enumerate(outputs):
                try:
                    successful.append(
                        enumerate_cpdag_extensions(output, max_extensions)
                    )
                except NonExtendableCPDAGError as error:
                    failures.append(
                        {
                            "family": family,
                            "variant": str(variant),
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
            if successful:
                family_completions[family] = successful
        if not family_completions:
            raise RuntimeError(f"all DAG families failed on bootstrap {bootstrap}")
        masses: Counter[Graph] = Counter()
        family_mass = 1.0 / len(family_completions)
        for variants in family_completions.values():
            variant_mass = family_mass / len(variants)
            for completions in variants:
                completion_mass = variant_mass / len(completions)
                for graph in completions:
                    masses[graph] += completion_mass
        rows.append(normalize_distribution(masses))
        diagnostics.append(
            {
                "bootstrap": bootstrap,
                "active_families": sorted(family_completions),
                "successful_variants": {
                    family: len(variants)
                    for family, variants in family_completions.items()
                },
                "support_size": len(masses),
                "failures": failures,
            }
        )
    return rows, diagnostics


def reweight_distribution(
    distribution: Mapping[Graph, float],
    log_preference: Callable[[Graph], float],
) -> tuple[GraphDistribution, float]:
    log_values = {
        graph: math.log(probability) + log_preference(graph)
        for graph, probability in distribution.items()
        if probability > 0
    }
    maximum = max(log_values.values())
    scaled = {graph: math.exp(value - maximum) for graph, value in log_values.items()}
    scaled_sum = math.fsum(scaled.values())
    result = {graph: value / scaled_sum for graph, value in scaled.items()}
    log_normalizer = maximum + math.log(scaled_sum)
    return result, log_normalizer


def defensive_mixture(
    core: Mapping[Graph, float],
    exploration: Mapping[Graph, float],
    epsilon: float = 0.1,
) -> GraphDistribution:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("mixture epsilon must lie in [0, 1]")
    mass: Counter[Graph] = Counter()
    for graph, probability in core.items():
        mass[graph] += (1.0 - epsilon) * float(probability)
    for graph, probability in exploration.items():
        mass[graph] += epsilon * float(probability)
    return normalize_distribution(mass)


def defensive_evidence_update(
    core: Mapping[Graph, float],
    exploration: Mapping[Graph, float],
    log_preference: Callable[[Graph], float],
    epsilon: float = 0.1,
    *,
    exploration_posterior: Mapping[Graph, float] | None = None,
    exploration_log_normalizer: float | None = None,
) -> tuple[GraphDistribution, dict[str, float]]:
    """Fuse evidence while preserving the two defensive-mixture components."""

    core_posterior, core_log_z = reweight_distribution(core, log_preference)
    if exploration_posterior is None:
        exploration_posterior, exploration_log_z = reweight_distribution(
            exploration, log_preference
        )
    else:
        exploration_posterior = normalize_distribution(exploration_posterior)
        if exploration_log_normalizer is None:
            raise ValueError("an external exploration posterior needs its log normalizer")
        exploration_log_z = float(exploration_log_normalizer)
    log_core_mass = (
        -math.inf if epsilon == 1.0 else math.log1p(-epsilon) + core_log_z
    )
    log_exploration_mass = (
        -math.inf if epsilon == 0.0 else math.log(epsilon) + exploration_log_z
    )
    maximum = max(log_core_mass, log_exploration_mass)
    core_mass = 0.0 if not math.isfinite(log_core_mass) else math.exp(log_core_mass - maximum)
    exploration_mass = (
        0.0
        if not math.isfinite(log_exploration_mass)
        else math.exp(log_exploration_mass - maximum)
    )
    exploration_share = exploration_mass / (core_mass + exploration_mass)
    result = defensive_mixture(
        core_posterior, exploration_posterior, exploration_share
    )
    return result, {
        "core_log_evidence_normalizer": core_log_z,
        "exploration_log_evidence_normalizer": exploration_log_z,
        "posterior_exploration_share": exploration_share,
    }


def _rows_are_acyclic(rows: Sequence[int]) -> bool:
    indegree = [0] * len(rows)
    for row in rows:
        for target in iter_bits(row):
            indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in iter_bits(rows[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == len(rows)


def _state_from_graph(graph: Graph, score: GaussianBICScore) -> tuple[list[int], list[int], float]:
    rows = list(graph.directed)
    parents = list(score.parent_masks(graph))
    return rows, parents, sum(
        score.local_score(node, mask) for node, mask in enumerate(parents)
    )


def _mh_step(
    rows: list[int],
    parents: list[int],
    current_score: float,
    score: GaussianBICScore,
    rng: np.random.Generator,
    log_preference: Callable[[Graph], float] | None = None,
    preference_scale: float = 0.0,
    current_preference: float = 0.0,
) -> tuple[list[int], list[int], float, float, bool]:
    first, second = sorted(rng.choice(len(rows), size=2, replace=False).tolist())
    if rows[first] & (1 << second):
        current_state = 1
    elif rows[second] & (1 << first):
        current_state = 2
    else:
        current_state = 0
    choices = [state for state in (0, 1, 2) if state != current_state]
    proposed_state = choices[int(rng.integers(2))]
    proposed_rows = list(rows)
    proposed_rows[first] &= ~(1 << second)
    proposed_rows[second] &= ~(1 << first)
    if proposed_state == 1:
        proposed_rows[first] |= 1 << second
    elif proposed_state == 2:
        proposed_rows[second] |= 1 << first
    if not _rows_are_acyclic(proposed_rows):
        return rows, parents, current_score, current_preference, False

    proposed_parents = list(parents)
    old_first = proposed_parents[first]
    old_second = proposed_parents[second]
    proposed_parents[first] &= ~(1 << second)
    proposed_parents[second] &= ~(1 << first)
    if proposed_state == 1:
        proposed_parents[second] |= 1 << first
    elif proposed_state == 2:
        proposed_parents[first] |= 1 << second
    proposed_score = current_score
    proposed_score += score.local_score(first, proposed_parents[first])
    proposed_score -= score.local_score(first, old_first)
    proposed_score += score.local_score(second, proposed_parents[second])
    proposed_score -= score.local_score(second, old_second)
    proposed_preference = current_preference
    graph: Graph | None = None
    if log_preference is not None and preference_scale != 0.0:
        graph = Graph(tuple(proposed_rows))
        proposed_preference = log_preference(graph)
    log_ratio = proposed_score - current_score
    log_ratio += preference_scale * (proposed_preference - current_preference)
    if math.log(max(rng.random(), np.finfo(float).tiny)) >= min(0.0, log_ratio):
        return rows, parents, current_score, current_preference, False
    return proposed_rows, proposed_parents, proposed_score, proposed_preference, True


def _split_rhat(chains: np.ndarray) -> float:
    if chains.ndim != 2 or chains.shape[0] < 2 or chains.shape[1] < 4:
        return math.nan
    half = chains.shape[1] // 2
    split = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = half * float(np.var(np.mean(split, axis=1), ddof=1))
    if within == 0:
        return 1.0 if between == 0 else math.inf
    variance = ((half - 1) / half) * within + between / half
    return float(math.sqrt(variance / within))


def _effective_sample_size(chains: np.ndarray) -> float:
    if chains.ndim != 2 or chains.shape[1] < 3:
        return math.nan
    chain_count, sample_count = chains.shape
    centered = chains - np.mean(chains, axis=1, keepdims=True)
    variance = float(np.mean(np.sum(centered * centered, axis=1) / sample_count))
    if variance <= 0:
        return float(chain_count * sample_count)
    correlation_sum = 0.0
    previous_pair = math.inf
    for lag in range(1, sample_count - 1, 2):
        pair = 0.0
        for offset in (lag, lag + 1):
            covariance = float(
                np.mean(
                    np.sum(centered[:, :-offset] * centered[:, offset:], axis=1)
                    / (sample_count - offset)
                )
            )
            pair += covariance / variance
        if pair < 0:
            break
        pair = min(pair, previous_pair)
        correlation_sum += pair
        previous_pair = pair
    return float(chain_count * sample_count / max(1.0 + 2.0 * correlation_sum, 1.0))


@dataclass(frozen=True)
class MCMCResult:
    samples: tuple[Graph, ...]
    distribution: GraphDistribution
    acceptance_rate: float
    score_ess: float
    edge_count_ess: float
    score_rhat: float
    edge_count_rhat: float


def sample_score_dags_mcmc(
    score: GaussianBICScore,
    initial_graphs: Sequence[Graph],
    seed: int,
    *,
    chain_count: int = 4,
    iterations: int = 25_000,
    burn_in: int = 5_000,
    thin: int = 10,
) -> MCMCResult:
    if not initial_graphs:
        raise ValueError("MCMC needs at least one initial DAG")
    if chain_count < 2 or not 0 <= burn_in < iterations or thin < 1:
        raise ValueError("invalid MCMC configuration")
    per_chain = (iterations - burn_in) // thin
    if per_chain < 2:
        raise ValueError("MCMC configuration retains too few samples")
    all_samples: list[Graph] = []
    score_traces = np.zeros((chain_count, per_chain), dtype=float)
    edge_traces = np.zeros((chain_count, per_chain), dtype=float)
    accepted = attempted = 0
    master = np.random.default_rng(seed)
    for chain in range(chain_count):
        rng = np.random.default_rng(int(master.integers(2**32)))
        initial = initial_graphs[chain % len(initial_graphs)]
        rows, parents, current_score = _state_from_graph(initial, score)
        retained = 0
        for iteration in range(iterations):
            rows, parents, current_score, _, changed = _mh_step(
                rows, parents, current_score, score, rng
            )
            accepted += int(changed)
            attempted += 1
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                graph = Graph(tuple(rows))
                all_samples.append(graph)
                score_traces[chain, retained] = current_score
                edge_traces[chain, retained] = sum(row.bit_count() for row in rows)
                retained += 1
                if retained == per_chain:
                    break
    counts = Counter(all_samples)
    return MCMCResult(
        tuple(all_samples),
        normalize_distribution(counts),
        accepted / attempted,
        _effective_sample_size(score_traces),
        _effective_sample_size(edge_traces),
        _split_rhat(score_traces),
        _split_rhat(edge_traces),
    )


def _conditional_ess(log_preferences: np.ndarray, delta: float) -> float:
    log_weights = delta * log_preferences
    weights = np.exp(log_weights - np.max(log_weights))
    return float(np.sum(weights) ** 2 / np.sum(weights * weights))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


@dataclass(frozen=True)
class SMCResult:
    distribution: GraphDistribution
    log_normalizer_ratio: float
    temperatures: tuple[float, ...]
    conditional_ess: tuple[float, ...]
    acceptance_rate: float


@dataclass(frozen=True)
class AdaptiveSupportResult:
    """DAG-only defensive support expansion, evidence update, and pruning."""

    data_distribution: GraphDistribution
    evidence_distribution: GraphDistribution
    unpruned_data_distribution: GraphDistribution
    unpruned_evidence_distribution: GraphDistribution
    mcmc_result: MCMCResult
    smc_result: SMCResult
    data_pruning: PruningResult
    evidence_pruning: PruningResult
    evidence_diagnostics: dict[str, float]
    original_support_size: int
    created_data_support_size: int
    created_evidence_support_size: int


def temper_evidence_smc(
    score: GaussianBICScore,
    initial_samples: Sequence[Graph],
    log_preference: Callable[[Graph], float],
    seed: int,
    *,
    particle_count: int = 2_000,
    conditional_ess_fraction: float = 0.8,
    mh_sweeps: int = 10,
) -> SMCResult:
    if not initial_samples or particle_count < 2:
        raise ValueError("SMC needs score-posterior initial samples")
    if not 0 < conditional_ess_fraction < 1 or mh_sweeps < 1:
        raise ValueError("invalid SMC configuration")
    rng = np.random.default_rng(seed)
    replace = len(initial_samples) < particle_count
    indices = rng.choice(len(initial_samples), size=particle_count, replace=replace)
    particles = [initial_samples[int(index)] for index in indices]
    preference_cache: dict[Graph, float] = {}

    def preference(graph: Graph) -> float:
        if graph not in preference_cache:
            preference_cache[graph] = float(log_preference(graph))
        return preference_cache[graph]

    temperatures = [0.0]
    ess_values: list[float] = []
    log_z = 0.0
    accepted = attempted = 0
    while temperatures[-1] < 1.0 - 1e-12:
        current = temperatures[-1]
        log_values = np.asarray([preference(graph) for graph in particles])
        remaining = 1.0 - current
        threshold = conditional_ess_fraction * particle_count
        if _conditional_ess(log_values, remaining) >= threshold:
            following = 1.0
        else:
            low, high = current, 1.0
            for _ in range(50):
                middle = 0.5 * (low + high)
                if _conditional_ess(log_values, middle - current) < threshold:
                    high = middle
                else:
                    low = middle
            following = low
            if following <= current + 1e-10:
                following = min(1.0, current + 1e-6)
        delta = following - current
        incremental = delta * log_values
        maximum = float(np.max(incremental))
        unnormalized = np.exp(incremental - maximum)
        mean_weight = float(np.mean(unnormalized))
        log_z += maximum + math.log(mean_weight)
        weights = unnormalized / np.sum(unnormalized)
        ess_values.append(float(1.0 / np.sum(weights * weights)))
        resampled = _systematic_resample(weights, rng)
        particles = [particles[int(index)] for index in resampled]
        temperatures.append(following)

        rejuvenated: list[Graph] = []
        for graph in particles:
            rows, parents, current_score = _state_from_graph(graph, score)
            current_preference = preference(graph)
            for _ in range(mh_sweeps):
                rows, parents, current_score, current_preference, changed = _mh_step(
                    rows,
                    parents,
                    current_score,
                    score,
                    rng,
                    preference,
                    following,
                    current_preference,
                )
                accepted += int(changed)
                attempted += 1
            rejuvenated.append(Graph(tuple(rows)))
        particles = rejuvenated
    return SMCResult(
        normalize_distribution(Counter(particles)),
        log_z,
        tuple(temperatures),
        tuple(ess_values),
        accepted / attempted,
    )


def adaptive_support_ceg_fuse(
    core: Mapping[Graph, float],
    score: GaussianBICScore,
    log_preference: Callable[[Graph], float],
    seed: int,
    *,
    epsilon: float = 0.1,
    max_removed_mass: float = 0.0,
    initial_graphs: Sequence[Graph] | None = None,
    chain_count: int = 4,
    iterations: int = 25_000,
    burn_in: int = 5_000,
    thin: int = 10,
    particle_count: int = 2_000,
    conditional_ess_fraction: float = 0.8,
    mh_sweeps: int = 10,
) -> AdaptiveSupportResult:
    """Expand a DAG support by score MCMC, fuse evidence by SMC, and prune.

    ``core`` is the empirical discovery distribution for one bootstrap.  The
    score-MCMC distribution supplies the defensive exploration component, and
    evidence-tempered SMC approximates its evidence-updated counterpart.  Both
    final mixtures are pruned independently under the same total-mass bound.
    """

    if not core:
        raise ValueError("adaptive support needs a nonempty core distribution")
    for probability in core.values():
        value = float(probability)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("core probabilities must be finite and nonnegative")
    normalized_core = normalize_distribution(core)
    if initial_graphs is None:
        starts = tuple(
            graph
            for graph, _ in sorted(
                normalized_core.items(), key=lambda item: (-item[1], item[0])
            )[:chain_count]
        )
    else:
        starts = tuple(initial_graphs)
    if not starts:
        raise ValueError("adaptive support needs at least one initial DAG")

    graphs_to_validate = set(normalized_core) | set(starts)
    if any(graph.regime != "dag" for graph in graphs_to_validate):
        raise ValueError("adaptive support is available only for DAGs")
    if any(graph.node_count != score.node_count for graph in graphs_to_validate):
        raise ValueError("all DAGs must have the same node count as the score")

    master = np.random.default_rng(seed)
    mcmc_seed = int(master.integers(2**32))
    smc_seed = int(master.integers(2**32))
    mcmc_result = sample_score_dags_mcmc(
        score,
        starts,
        mcmc_seed,
        chain_count=chain_count,
        iterations=iterations,
        burn_in=burn_in,
        thin=thin,
    )
    unpruned_data = defensive_mixture(
        normalized_core, mcmc_result.distribution, epsilon
    )
    preference_cache: dict[Graph, float] = {}

    def tracked_preference(graph: Graph) -> float:
        if graph not in preference_cache:
            value = float(log_preference(graph))
            if not math.isfinite(value):
                raise ValueError("log preferences must be finite")
            preference_cache[graph] = value
        return preference_cache[graph]

    for graph in set(normalized_core) | set(mcmc_result.distribution):
        tracked_preference(graph)
    smc_result = temper_evidence_smc(
        score,
        mcmc_result.samples,
        tracked_preference,
        smc_seed,
        particle_count=particle_count,
        conditional_ess_fraction=conditional_ess_fraction,
        mh_sweeps=mh_sweeps,
    )
    preference_values = tuple(preference_cache.values())
    constant_preference = all(
        value == preference_values[0] for value in preference_values[1:]
    )
    if constant_preference:
        constant = preference_values[0]
        smc_result = SMCResult(
            distribution=dict(mcmc_result.distribution),
            log_normalizer_ratio=constant,
            temperatures=smc_result.temperatures,
            conditional_ess=smc_result.conditional_ess,
            acceptance_rate=smc_result.acceptance_rate,
        )
        unpruned_evidence = dict(unpruned_data)
        evidence_diagnostics = {
            "core_log_evidence_normalizer": constant,
            "exploration_log_evidence_normalizer": constant,
            "posterior_exploration_share": float(epsilon),
        }
    else:
        unpruned_evidence, evidence_diagnostics = defensive_evidence_update(
            normalized_core,
            mcmc_result.distribution,
            tracked_preference,
            epsilon,
            exploration_posterior=smc_result.distribution,
            exploration_log_normalizer=smc_result.log_normalizer_ratio,
        )
    data_pruning = prune_by_mass(unpruned_data, max_removed_mass)
    evidence_pruning = prune_by_mass(unpruned_evidence, max_removed_mass)
    core_support = set(normalized_core)
    return AdaptiveSupportResult(
        data_distribution=data_pruning.distribution,
        evidence_distribution=evidence_pruning.distribution,
        unpruned_data_distribution=unpruned_data,
        unpruned_evidence_distribution=unpruned_evidence,
        mcmc_result=mcmc_result,
        smc_result=smc_result,
        data_pruning=data_pruning,
        evidence_pruning=evidence_pruning,
        evidence_diagnostics=evidence_diagnostics,
        original_support_size=len(normalized_core),
        created_data_support_size=len(set(unpruned_data) - core_support),
        created_evidence_support_size=len(set(unpruned_evidence) - core_support),
    )
