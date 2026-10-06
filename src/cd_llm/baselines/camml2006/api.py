"""Public raw-data and reusable-score APIs for the CaMML adaptation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cd_llm.core.finite_support import Claim, EvidenceGraph

from .adapters import adapt_evidence_to_prior
from .data_scores import (
    AdaptiveMMLLocalScore,
    BDeuLocalScore,
    DecomposableLocalScore,
    GaussianBICLocalScore,
)
from .search import sample_toms
from .types import CaMMLConfig, CaMMLPrior, CaMMLResult, DataScoreName


def fit_camml_from_local_score(
    local_score: DecomposableLocalScore,
    prior: CaMMLPrior | None = None,
    *,
    config: CaMMLConfig | None = None,
    data_score_name: str | None = None,
    evidence_adaptation: object | None = None,
) -> CaMMLResult:
    node_count = getattr(local_score, "node_count", None)
    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count < 1:
        raise TypeError("local_score.node_count must be a positive integer")
    normalized_prior = CaMMLPrior.empty(node_count) if prior is None else prior
    return sample_toms(
        local_score,
        normalized_prior,
        config=config,
        data_score_name=data_score_name,
        evidence_adaptation=evidence_adaptation,
    )


def _build_score(
    data: np.ndarray | Sequence[Sequence[object]],
    data_score: DataScoreName,
    *,
    equivalent_sample_size: float,
    cardinalities: Sequence[int] | None,
    standardize_data: bool,
) -> DecomposableLocalScore:
    if data_score == "adaptive-mml":
        if equivalent_sample_size != 1.0 or standardize_data is not True:
            raise ValueError("equivalent_sample_size and standardize_data do not apply to adaptive-mml")
        return AdaptiveMMLLocalScore(data, cardinalities=cardinalities)
    if data_score == "bdeu":
        if standardize_data is not True:
            raise ValueError("standardize_data does not apply to bdeu")
        return BDeuLocalScore(
            data,
            equivalent_sample_size=equivalent_sample_size,
            cardinalities=cardinalities,
        )
    if data_score == "gaussian-bic":
        if equivalent_sample_size != 1.0 or cardinalities is not None:
            raise ValueError("equivalent_sample_size and cardinalities apply only to discrete scores")
        return GaussianBICLocalScore(data, standardize_data=standardize_data)  # type: ignore[arg-type]
    raise ValueError("data_score must be 'adaptive-mml', 'bdeu', or 'gaussian-bic'")


def fit_camml(
    data: np.ndarray | Sequence[Sequence[object]],
    prior: CaMMLPrior | None = None,
    *,
    data_score: DataScoreName = "adaptive-mml",
    config: CaMMLConfig | None = None,
    equivalent_sample_size: float = 1.0,
    cardinalities: Sequence[int] | None = None,
    standardize_data: bool = True,
) -> CaMMLResult:
    score = _build_score(
        data,
        data_score,
        equivalent_sample_size=equivalent_sample_size,
        cardinalities=cardinalities,
        standardize_data=standardize_data,
    )
    return fit_camml_from_local_score(
        score,
        prior,
        config=config,
        data_score_name=data_score,
    )


def fit_camml_with_evidence(
    data: np.ndarray | Sequence[Sequence[object]],
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    data_score: DataScoreName = "adaptive-mml",
    config: CaMMLConfig | None = None,
    equivalent_sample_size: float = 1.0,
    cardinalities: Sequence[int] | None = None,
    standardize_data: bool = True,
) -> CaMMLResult:
    score = _build_score(
        data,
        data_score,
        equivalent_sample_size=equivalent_sample_size,
        cardinalities=cardinalities,
        standardize_data=standardize_data,
    )
    normalized_config = CaMMLConfig() if config is None else config
    adaptation = adapt_evidence_to_prior(
        claims,
        evidence,
        node_count=score.node_count,
        arc_probability=normalized_config.arc_probability,
        global_prior_samples=normalized_config.global_prior_samples,
    )
    return fit_camml_from_local_score(
        score,
        adaptation.prior,
        config=normalized_config,
        data_score_name=data_score,
        evidence_adaptation=adaptation,
    )


__all__ = ["fit_camml", "fit_camml_from_local_score", "fit_camml_with_evidence"]
