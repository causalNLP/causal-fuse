"""EM estimation of expert accuracies from Amirkhani et al. (2017)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from .accuracy import ExpertAccuracy
from .expert_scores import opinion_probability
from .opinions import (
    EdgeState,
    ExpertOpinion,
    normalize_expert_ids,
    normalize_opinions,
)


_STATES = (
    EdgeState.LEFT_TO_RIGHT,
    EdgeState.RIGHT_TO_LEFT,
    EdgeState.ABSENT,
)


@dataclass(frozen=True)
class AccuracyEMConfig:
    """Convergence controls for the paper's EM procedure."""

    tolerance: float = 0.001
    max_iterations: int = 1000

    def __post_init__(self) -> None:
        if isinstance(self.tolerance, bool) or not isinstance(
            self.tolerance, (int, float)
        ):
            raise TypeError("tolerance must be a real number")
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, int
        ):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class AccuracyEMResult:
    """Final EM parameters and convergence diagnostics."""

    edge_prior: tuple[float, float, float]
    accuracies: Mapping[str, ExpertAccuracy]
    iterations: int
    converged: bool
    log_likelihoods: tuple[float, ...]
    max_parameter_change: float
    unidentified_parameters: tuple[tuple[str, str], ...]
    pair_posteriors: tuple[
        tuple[int, int, tuple[float, float, float]], ...
    ]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accuracies",
            MappingProxyType(dict(sorted(self.accuracies.items()))),
        )
        object.__setattr__(
            self,
            "unidentified_parameters",
            tuple(sorted(set(self.unidentified_parameters))),
        )


def _validate_node_count(node_count: int) -> None:
    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise TypeError("node_count must be an integer")
    if node_count < 2:
        raise ValueError("node_count must be at least two")


def _normalize_edge_prior(
    edge_prior: tuple[float, float, float] | Iterable[float],
) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in edge_prior)
    except (TypeError, ValueError) as error:
        raise TypeError("initial_edge_prior must contain three real numbers") from error
    if len(values) != 3:
        raise ValueError("initial_edge_prior must contain three probabilities")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("initial_edge_prior probabilities must be finite and nonnegative")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("initial_edge_prior probabilities must sum to one")
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _initial_accuracy_map(
    expert_ids: tuple[str, ...],
    initial: ExpertAccuracy | Mapping[str, ExpertAccuracy] | None,
) -> dict[str, ExpertAccuracy]:
    if initial is None:
        shared = ExpertAccuracy.neutral()
        return {expert_id: shared for expert_id in expert_ids}
    if isinstance(initial, ExpertAccuracy):
        return {expert_id: initial for expert_id in expert_ids}
    if not isinstance(initial, Mapping):
        raise TypeError(
            "initial_accuracies must be an ExpertAccuracy, a mapping, or None"
        )
    result: dict[str, ExpertAccuracy] = {}
    for expert_id in expert_ids:
        try:
            accuracy = initial[expert_id]
        except KeyError as error:
            raise ValueError(
                f"missing initial accuracy for expert {expert_id!r}"
            ) from error
        if not isinstance(accuracy, ExpertAccuracy):
            raise TypeError("initial_accuracies must contain ExpertAccuracy values")
        result[expert_id] = accuracy
    return result


def _expectation(
    pairs: tuple[tuple[int, int], ...],
    by_pair: Mapping[tuple[int, int], tuple[ExpertOpinion, ...]],
    edge_prior: tuple[float, float, float],
    accuracies: Mapping[str, ExpertAccuracy],
) -> tuple[dict[tuple[int, int], np.ndarray], float]:
    posteriors: dict[tuple[int, int], np.ndarray] = {}
    log_likelihood = 0.0
    for pair in pairs:
        log_weights = np.asarray(
            [math.log(value) if value > 0.0 else -math.inf for value in edge_prior],
            dtype=float,
        )
        for opinion in by_pair.get(pair, ()):
            accuracy = accuracies[opinion.expert_id]
            for index, true_state in enumerate(_STATES):
                probability = opinion_probability(
                    true_state, opinion.state, accuracy
                )
                if probability <= 0.0:
                    log_weights[index] = -math.inf
                elif math.isfinite(log_weights[index]):
                    log_weights[index] += math.log(probability)
        normalizer = float(logsumexp(log_weights))
        if not math.isfinite(normalizer):
            raise ValueError(
                f"opinions for pair {pair} have zero likelihood under all edge states"
            )
        posterior = np.exp(log_weights - normalizer)
        posteriors[pair] = posterior
        log_likelihood += normalizer
    return posteriors, log_likelihood


def _parameter_change(
    previous_prior: tuple[float, float, float],
    next_prior: tuple[float, float, float],
    previous_accuracies: Mapping[str, ExpertAccuracy],
    next_accuracies: Mapping[str, ExpertAccuracy],
) -> float:
    changes = [
        abs(previous - current)
        for previous, current in zip(previous_prior, next_prior)
    ]
    for expert_id in previous_accuracies:
        previous = previous_accuracies[expert_id]
        current = next_accuracies[expert_id]
        changes.extend(
            (
                abs(previous.gamma1 - current.gamma1),
                abs(previous.gamma2 - current.gamma2),
                abs(previous.gamma3 - current.gamma3),
            )
        )
    return max(changes, default=0.0)


def estimate_accuracies_em(
    opinions: Iterable[ExpertOpinion],
    *,
    node_count: int,
    expert_ids: Iterable[str] | None = None,
    initial_edge_prior: tuple[float, float, float] = (0.1, 0.1, 0.8),
    initial_accuracies: ExpertAccuracy | Mapping[str, ExpertAccuracy] | None = None,
    config: AccuracyEMConfig = AccuracyEMConfig(),
) -> AccuracyEMResult:
    """Estimate edge-type frequencies and per-expert accuracies by EM.

    Missing responses make no likelihood contribution.  If an expert has zero
    posterior mass in a parameter's conditioning class, that parameter is not
    identifiable: its preceding value is retained and its name is reported.
    """

    _validate_node_count(node_count)
    if not isinstance(config, AccuracyEMConfig):
        raise TypeError("config must be an AccuracyEMConfig")
    normalized = normalize_opinions(opinions, node_count=node_count)
    observed_experts = tuple(sorted({opinion.expert_id for opinion in normalized}))
    if expert_ids is None:
        normalized_experts = observed_experts
    else:
        normalized_experts = normalize_expert_ids(expert_ids)
        missing = set(observed_experts) - set(normalized_experts)
        if missing:
            raise ValueError(
                "expert_ids omits experts represented in opinions: "
                + ", ".join(sorted(missing))
            )
    edge_prior = _normalize_edge_prior(initial_edge_prior)
    accuracies = _initial_accuracy_map(normalized_experts, initial_accuracies)

    pairs = tuple(
        (left, right)
        for left in range(node_count)
        for right in range(left + 1, node_count)
    )
    grouped: dict[tuple[int, int], list[ExpertOpinion]] = {}
    for opinion in normalized:
        grouped.setdefault(opinion.pair, []).append(opinion)
    by_pair = {pair: tuple(values) for pair, values in grouped.items()}

    posteriors, likelihood = _expectation(
        pairs, by_pair, edge_prior, accuracies
    )
    likelihoods = [likelihood]
    unidentified: set[tuple[str, str]] = set()
    converged = False
    last_change = math.inf
    iterations = 0

    for iterations in range(1, config.max_iterations + 1):
        mean = np.mean(np.stack(tuple(posteriors.values())), axis=0)
        next_prior = tuple(float(value) for value in mean)
        next_accuracies: dict[str, ExpertAccuracy] = {}
        opinions_by_expert: dict[str, list[ExpertOpinion]] = {
            expert_id: [] for expert_id in normalized_experts
        }
        for opinion in normalized:
            opinions_by_expert[opinion.expert_id].append(opinion)

        for expert_id in normalized_experts:
            previous = accuracies[expert_id]
            existing_denominator = 0.0
            correct_numerator = 0.0
            reverse_numerator = 0.0
            absent_denominator = 0.0
            absent_correct_numerator = 0.0
            for opinion in opinions_by_expert[expert_id]:
                posterior = posteriors[opinion.pair]
                existing_denominator += float(posterior[0] + posterior[1])
                if opinion.state == EdgeState.LEFT_TO_RIGHT:
                    correct_numerator += float(posterior[0])
                    reverse_numerator += float(posterior[1])
                elif opinion.state == EdgeState.RIGHT_TO_LEFT:
                    correct_numerator += float(posterior[1])
                    reverse_numerator += float(posterior[0])
                absent_denominator += float(posterior[2])
                if opinion.state == EdgeState.ABSENT:
                    absent_correct_numerator += float(posterior[2])

            if existing_denominator <= 1e-15:
                gamma1, gamma2 = previous.gamma1, previous.gamma2
                unidentified.update(
                    ((expert_id, "gamma1"), (expert_id, "gamma2"))
                )
            else:
                gamma1 = max(
                    0.0, min(1.0, correct_numerator / existing_denominator)
                )
                gamma2 = max(
                    0.0, min(1.0, reverse_numerator / existing_denominator)
                )
                if gamma1 + gamma2 > 1.0:
                    scale = gamma1 + gamma2
                    gamma1 /= scale
                    gamma2 /= scale
            if absent_denominator <= 1e-15:
                gamma3 = previous.gamma3
                unidentified.add((expert_id, "gamma3"))
            else:
                gamma3 = max(
                    0.0,
                    min(1.0, absent_correct_numerator / absent_denominator),
                )
            next_accuracies[expert_id] = ExpertAccuracy(
                gamma1, gamma2, gamma3
            )

        last_change = _parameter_change(
            edge_prior, next_prior, accuracies, next_accuracies
        )
        edge_prior = next_prior  # type: ignore[assignment]
        accuracies = next_accuracies
        posteriors, likelihood = _expectation(
            pairs, by_pair, edge_prior, accuracies
        )
        likelihoods.append(likelihood)
        if last_change < config.tolerance:
            converged = True
            break

    pair_posteriors = tuple(
        (
            left,
            right,
            tuple(float(value) for value in posteriors[(left, right)]),
        )
        for left, right in pairs
    )
    return AccuracyEMResult(
        edge_prior=edge_prior,
        accuracies=accuracies,
        iterations=iterations,
        converged=converged,
        log_likelihoods=tuple(likelihoods),
        max_parameter_change=last_change,
        unidentified_parameters=tuple(unidentified),
        pair_posteriors=pair_posteriors,  # type: ignore[arg-type]
    )
