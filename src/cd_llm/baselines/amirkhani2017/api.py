"""Public data-to-DAG API for the Amirkhani et al. (2017) baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Literal

import numpy as np

from .accuracy import AccuracyPrior, ExpertAccuracy
from .data_scores import (
    BDeuLocalScore,
    DecomposableLocalScore,
    GaussianBICLocalScore,
)
from .em import AccuracyEMConfig, estimate_accuracies_em
from .expert_scores import explicit_expert_log_score, marginal_expert_log_score
from .opinions import (
    ExpertOpinion,
    normalize_expert_ids,
    normalize_opinions,
)
from .search import AmirkhaniConfig, AmirkhaniResult, greedy_dag_search


DataScoreName = Literal["bdeu", "gaussian-bic"]
_DEFAULT_EDGE_PRIOR = (0.1, 0.1, 0.8)


def _normalized_expert_ids(
    opinions: tuple[ExpertOpinion, ...], expert_ids: Iterable[str] | None
) -> tuple[str, ...]:
    observed = {opinion.expert_id for opinion in opinions}
    if expert_ids is None:
        return tuple(sorted(observed))
    supplied = normalize_expert_ids(expert_ids)
    missing = observed - set(supplied)
    if missing:
        raise ValueError(
            "expert_ids omits opinion providers: " + ", ".join(sorted(missing))
        )
    return supplied


def _reject_nondefault_em_prior(
    values: Sequence[float], expert_score: str
) -> None:
    if expert_score == "em":
        return
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError("initial_edge_prior must contain three real numbers") from error
    if normalized != _DEFAULT_EDGE_PRIOR:
        raise ValueError("initial_edge_prior applies only to EM scoring")


def fit_amirkhani_from_local_score(
    local_score: DecomposableLocalScore,
    opinions: Iterable[ExpertOpinion] = (),
    *,
    config: AmirkhaniConfig | None = None,
    accuracies: ExpertAccuracy | Mapping[str, ExpertAccuracy] | None = None,
    priors: AccuracyPrior | Mapping[str, AccuracyPrior] | None = None,
    expert_ids: Iterable[str] | None = None,
    initial_edge_prior: Sequence[float] = _DEFAULT_EDGE_PRIOR,
    initial_accuracies: ExpertAccuracy | Mapping[str, ExpertAccuracy] | None = None,
    data_score_name: str | None = None,
) -> AmirkhaniResult:
    """Learn a DAG from a reusable decomposable observational score.

    The local score and both expert scores are higher-is-better log scores.
    ``expert_score='em'`` first estimates each expert's accuracy from opinions
    alone, as in the paper, and then invokes the explicit expert score.
    """

    if config is None:
        config = AmirkhaniConfig()
    if not isinstance(config, AmirkhaniConfig):
        raise TypeError("config must be AmirkhaniConfig")
    _reject_nondefault_em_prior(initial_edge_prior, config.expert_score)
    node_count = getattr(local_score, "node_count", None)
    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("local_score.node_count must be an integer")
    node_count = int(node_count)
    normalized_opinions = normalize_opinions(opinions, node_count=node_count)
    normalized_ids = _normalized_expert_ids(normalized_opinions, expert_ids)

    estimated_accuracies: Mapping[str, ExpertAccuracy] | None = None
    em_result = None
    if config.expert_score == "marginal":
        if accuracies is not None or initial_accuracies is not None:
            raise ValueError(
                "accuracies apply to explicit scoring and initial_accuracies to EM"
            )

        def expert_score(graph):  # type: ignore[no-untyped-def]
            return marginal_expert_log_score(graph, normalized_opinions, priors)

    elif config.expert_score == "explicit":
        if accuracies is None:
            raise ValueError("explicit expert scoring requires accuracies")
        if priors is not None or initial_accuracies is not None:
            raise ValueError("priors and initial_accuracies do not apply to explicit scoring")
        # Freeze the objective so a caller cannot change it mid-search. A
        # single triple reproduces the paper's shared-mean scoring condition.
        if isinstance(accuracies, ExpertAccuracy):
            explicit_accuracies = {
                expert_id: accuracies for expert_id in normalized_ids
            }
        elif isinstance(accuracies, Mapping):
            explicit_accuracies = dict(accuracies)
        else:
            raise TypeError(
                "accuracies must be an ExpertAccuracy or a mapping"
            )

        def expert_score(graph):  # type: ignore[no-untyped-def]
            return explicit_expert_log_score(
                graph, normalized_opinions, explicit_accuracies
            )

    else:
        if accuracies is not None or priors is not None:
            raise ValueError("accuracies and priors do not apply to EM scoring")
        em_result = estimate_accuracies_em(
            normalized_opinions,
            node_count=node_count,
            expert_ids=normalized_ids,
            initial_edge_prior=initial_edge_prior,
            initial_accuracies=initial_accuracies,
            config=AccuracyEMConfig(
                tolerance=config.em_tolerance,
                max_iterations=config.em_max_iterations,
            ),
        )
        estimated_accuracies = em_result.accuracies

        def expert_score(graph):  # type: ignore[no-untyped-def]
            return explicit_expert_log_score(
                graph, normalized_opinions, estimated_accuracies  # type: ignore[arg-type]
            )

    name = (
        str(data_score_name)
        if data_score_name is not None
        else str(getattr(local_score, "name", type(local_score).__name__))
    )
    if not name.strip():
        raise ValueError("data_score_name must be nonempty")
    return greedy_dag_search(
        local_score,
        expert_score,
        config=config,
        data_score_name=name,
        estimated_accuracies=estimated_accuracies,
        em_result=em_result,
    )


def fit_amirkhani(
    data: np.ndarray | Sequence[Sequence[object]],
    opinions: Iterable[ExpertOpinion] = (),
    *,
    data_score: DataScoreName = "bdeu",
    config: AmirkhaniConfig | None = None,
    accuracies: ExpertAccuracy | Mapping[str, ExpertAccuracy] | None = None,
    priors: AccuracyPrior | Mapping[str, AccuracyPrior] | None = None,
    expert_ids: Iterable[str] | None = None,
    initial_edge_prior: Sequence[float] = _DEFAULT_EDGE_PRIOR,
    initial_accuracies: ExpertAccuracy | Mapping[str, ExpertAccuracy] | None = None,
    equivalent_sample_size: float = 1.0,
    cardinalities: Sequence[int] | None = None,
    standardize_data: bool = True,
) -> AmirkhaniResult:
    """Learn a DAG from raw discrete or continuous observations.

    ``data_score='bdeu'`` is the paper-native complete-discrete method.
    ``data_score='gaussian-bic'`` is this project's continuous adaptation and
    leaves the expert-opinion term unchanged.
    """

    if data_score == "bdeu":
        local_score: DecomposableLocalScore = BDeuLocalScore(
            data,
            equivalent_sample_size=equivalent_sample_size,
            cardinalities=cardinalities,
        )
    elif data_score == "gaussian-bic":
        if cardinalities is not None:
            raise ValueError("cardinalities apply only to BDeu scoring")
        if equivalent_sample_size != 1.0:
            raise ValueError("equivalent_sample_size applies only to BDeu scoring")
        local_score = GaussianBICLocalScore(
            data,  # type: ignore[arg-type]
            standardize_data=standardize_data,
        )
    else:
        raise ValueError("data_score must be 'bdeu' or 'gaussian-bic'")
    return fit_amirkhani_from_local_score(
        local_score,
        opinions,
        config=config,
        accuracies=accuracies,
        priors=priors,
        expert_ids=expert_ids,
        initial_edge_prior=initial_edge_prior,
        initial_accuracies=initial_accuracies,
        data_score_name=data_score,
    )


__all__ = ["DataScoreName", "fit_amirkhani", "fit_amirkhani_from_local_score"]
