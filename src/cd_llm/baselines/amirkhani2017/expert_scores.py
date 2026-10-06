"""Expert-opinion terms in the two Amirkhani et al. (2017) scores."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from scipy.special import betaln, gammaln

from cd_llm.core.finite_support import Graph

from .accuracy import AccuracyPrior, ExpertAccuracy
from .opinions import EdgeState, ExpertOpinion, normalize_opinions


def _validate_dag(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a cd_llm.core.finite_support.Graph")
    if graph.regime != "dag":
        raise ValueError("the Amirkhani baseline supports DAGs only")


def edge_state_in_graph(graph: Graph, left: int, right: int) -> EdgeState:
    """Return the state of a canonical node pair in a DAG."""

    _validate_dag(graph)
    if not 0 <= left < right < graph.node_count:
        raise ValueError("edge-state endpoints must satisfy 0 <= left < right < p")
    if graph.directed[left] & (1 << right):
        return EdgeState.LEFT_TO_RIGHT
    if graph.directed[right] & (1 << left):
        return EdgeState.RIGHT_TO_LEFT
    return EdgeState.ABSENT


def opinion_probability(
    true_state: EdgeState,
    observed_state: EdgeState,
    accuracy: ExpertAccuracy,
) -> float:
    """Compute one entry of the paper's three-state confusion matrix."""

    try:
        truth = EdgeState(true_state)
        observed = EdgeState(observed_state)
    except (TypeError, ValueError) as error:
        raise ValueError("true_state and observed_state must be EdgeState values") from error
    if not isinstance(accuracy, ExpertAccuracy):
        raise TypeError("accuracy must be an ExpertAccuracy")
    if truth == EdgeState.ABSENT:
        if observed == EdgeState.ABSENT:
            return accuracy.gamma3
        return accuracy.absent_as_each_direction
    if observed == truth:
        return accuracy.gamma1
    if observed == EdgeState.ABSENT:
        return accuracy.existing_as_absent
    return accuracy.gamma2


def _log_probability(probability: float) -> float:
    return math.log(probability) if probability > 0.0 else -math.inf


def explicit_expert_log_score(
    graph: Graph,
    opinions: Iterable[ExpertOpinion],
    accuracies: Mapping[str, ExpertAccuracy],
) -> float:
    """Return ``log P(O | G, gamma)`` from equations (1)--(2)."""

    _validate_dag(graph)
    normalized = normalize_opinions(opinions, node_count=graph.node_count)
    if not isinstance(accuracies, Mapping):
        raise TypeError("accuracies must map expert identifiers to ExpertAccuracy values")
    total = 0.0
    for opinion in normalized:
        try:
            accuracy = accuracies[opinion.expert_id]
        except KeyError as error:
            raise ValueError(
                f"missing accuracy for expert {opinion.expert_id!r}"
            ) from error
        if not isinstance(accuracy, ExpertAccuracy):
            raise TypeError("accuracies must contain ExpertAccuracy values")
        truth = edge_state_in_graph(graph, opinion.left, opinion.right)
        total += _log_probability(
            opinion_probability(truth, opinion.state, accuracy)
        )
    return total


def _log_multivariate_beta(shapes: tuple[float, ...]) -> float:
    return float(sum(gammaln(shape) for shape in shapes) - gammaln(sum(shapes)))


def _prior_for_expert(
    expert_id: str,
    priors: AccuracyPrior | Mapping[str, AccuracyPrior] | None,
) -> AccuracyPrior:
    if priors is None:
        return AccuracyPrior.neutral()
    if isinstance(priors, AccuracyPrior):
        return priors
    if not isinstance(priors, Mapping):
        raise TypeError(
            "priors must be an AccuracyPrior, a mapping, or None"
        )
    try:
        prior = priors[expert_id]
    except KeyError as error:
        raise ValueError(f"missing accuracy prior for expert {expert_id!r}") from error
    if not isinstance(prior, AccuracyPrior):
        raise TypeError("priors must contain AccuracyPrior values")
    return prior


def marginal_expert_log_score(
    graph: Graph,
    opinions: Iterable[ExpertOpinion],
    priors: AccuracyPrior | Mapping[str, AccuracyPrior] | None = None,
) -> float:
    """Return ``log P(O | G)`` from equations (9), (16), and (17)."""

    _validate_dag(graph)
    normalized = normalize_opinions(opinions, node_count=graph.node_count)
    by_expert: dict[str, list[ExpertOpinion]] = defaultdict(list)
    for opinion in normalized:
        by_expert[opinion.expert_id].append(opinion)

    total = 0.0
    for expert_id in sorted(by_expert):
        n1 = n2 = n3 = m1 = m2 = 0
        for opinion in by_expert[expert_id]:
            truth = edge_state_in_graph(graph, opinion.left, opinion.right)
            if truth == EdgeState.ABSENT:
                if opinion.state == EdgeState.ABSENT:
                    m1 += 1
                else:
                    m2 += 1
            elif opinion.state == truth:
                n1 += 1
            elif opinion.state == EdgeState.ABSENT:
                n3 += 1
            else:
                n2 += 1

        prior = _prior_for_expert(expert_id, priors)
        beta_term = (
            float(betaln(prior.beta1 + m1, prior.beta2 + m2))
            - m2 * math.log(2.0)
            - float(betaln(prior.beta1, prior.beta2))
        )
        dirichlet_term = _log_multivariate_beta(
            (prior.alpha1 + n1, prior.alpha2 + n2, prior.alpha3 + n3)
        ) - _log_multivariate_beta(
            (prior.alpha1, prior.alpha2, prior.alpha3)
        )
        total += beta_term + dirichlet_term
    return total
