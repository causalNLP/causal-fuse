"""Public APIs for Causal Order + PC and Causal Order + CaMML."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cd_llm.baselines.camml2006 import (
    CaMMLConfig,
    DataScoreName,
    fit_camml,
)
from cd_llm.core.cpc import certify_cpdag

from .integration import (
    causal_learn_cpdag,
    integrate_order_into_cpdag,
    level_order_from_edges,
    level_order_prior,
)
from .types import (
    CaMMLOrderResult,
    FallbackOrienter,
    PCOrderConfig,
    PCOrderResult,
    TripletConfig,
    TripletExpertResult,
    normalize_node_names,
)
from .upstream_backend import (
    orient_with_upstream_expert,
    query_triplet_order,
)


PCRunner = Callable[..., Any]


def _validated_data(data: object, node_count: int) -> np.ndarray:
    observations = np.asarray(data)
    if observations.ndim != 2 or observations.shape[0] < 1:
        raise ValueError("data must be a nonempty two-dimensional array")
    if observations.shape[1] != node_count:
        raise ValueError("data column count must match node_names")
    if observations.dtype.kind not in "biufc":
        raise TypeError("data must contain numeric observations")
    if not np.all(np.isfinite(observations)):
        raise ValueError("data must contain only finite values")
    return observations


def _run_pc(
    data: np.ndarray,
    node_names: tuple[str, ...],
    config: PCOrderConfig,
    runner: PCRunner | None,
) -> object:
    if runner is None:
        from causallearn.search.ConstraintBased.PC import pc as runner

    result = runner(
        data,
        alpha=config.alpha,
        indep_test=config.indep_test,
        stable=config.stable,
        uc_rule=config.uc_rule,
        uc_priority=config.uc_priority,
        mvpc=config.mvpc,
        correction_name="MV_Crtn_Fisher_Z",
        background_knowledge=None,
        verbose=False,
        show_progress=False,
        node_names=list(node_names),
    )
    return result.G if hasattr(result, "G") else result


def fit_pc_with_causal_order(
    data: object,
    node_names: Sequence[str],
    expert: TripletExpertResult,
    *,
    config: PCOrderConfig | None = None,
    fallback_orienter: FallbackOrienter | None = None,
    runner: PCRunner | None = None,
) -> PCOrderResult:
    """Fit PC and apply Appendix-D Algorithm 1 using a cached expert result."""

    names = normalize_node_names(node_names)
    if not isinstance(expert, TripletExpertResult):
        raise TypeError("expert must be TripletExpertResult")
    if names != expert.node_names:
        raise ValueError("node_names must exactly match the cached expert result")
    observations = _validated_data(data, len(names))
    normalized_config = PCOrderConfig() if config is None else config
    if not isinstance(normalized_config, PCOrderConfig):
        raise TypeError("config must be PCOrderConfig")
    started = time.perf_counter()
    raw_graph = _run_pc(observations, names, normalized_config, runner)
    pc_runtime = time.perf_counter() - started
    # Finite-sample orientation conflicts can make causal-learn's nominal PC
    # output cyclic and therefore non-extendable.  Canonical certification
    # preserves its skeleton while projecting the endpoint preferences onto a
    # valid CPDAG before the expert order is integrated.
    cpdag = certify_cpdag(causal_learn_cpdag(raw_graph))
    integration_started = time.perf_counter()
    integrated = integrate_order_into_cpdag(
        cpdag,
        expert,
        fallback_orienter=fallback_orienter,
    )
    integration_runtime = time.perf_counter() - integration_started
    return PCOrderResult(
        graph=integrated.graph,
        final_order=integrated.final_order,
        cpdag=cpdag,
        expert=expert,
        fallback_orientations=integrated.fallback_orientations,
        config=normalized_config,
        pc_runtime_seconds=pc_runtime,
        integration_runtime_seconds=integration_runtime,
    )


def fit_pc_with_llm(
    data: object,
    node_names: Sequence[str],
    context: str,
    descriptions: Mapping[str, str] | None = None,
    *,
    triplet_config: TripletConfig | None = None,
    pc_config: PCOrderConfig | None = None,
    upstream_root: str | Path | None = None,
    runner: PCRunner | None = None,
) -> PCOrderResult:
    """Run the authors' triplet provider once, then fit Causal Order + PC."""

    names = normalize_node_names(node_names, minimum=3)
    normalized_triplet = TripletConfig() if triplet_config is None else triplet_config
    expert = query_triplet_order(
        names,
        context,
        descriptions,
        config=normalized_triplet,
        upstream_root=upstream_root,
    )

    def fallback(first: int, second: int) -> tuple[int, int]:
        return orient_with_upstream_expert(
            names,
            first,
            second,
            context,
            config=normalized_triplet,
            upstream_root=upstream_root,
        )

    return fit_pc_with_causal_order(
        data,
        names,
        expert,
        config=pc_config,
        fallback_orienter=fallback,
        runner=runner,
    )


def fit_camml_with_causal_order(
    data: object,
    node_names: Sequence[str],
    expert: TripletExpertResult,
    *,
    prior_probability: float = 0.9,
    data_score: DataScoreName = "adaptive-mml",
    config: CaMMLConfig | None = None,
    equivalent_sample_size: float = 1.0,
    cardinalities: Sequence[int] | None = None,
    standardize_data: bool = True,
) -> CaMMLOrderResult:
    """Apply Appendix-D Algorithm 2 using a cached expert graph."""

    names = normalize_node_names(node_names)
    if not isinstance(expert, TripletExpertResult):
        raise TypeError("expert must be TripletExpertResult")
    if names != expert.node_names:
        raise ValueError("node_names must exactly match the cached expert result")
    observations = _validated_data(data, len(names))
    normalized_config = CaMMLConfig() if config is None else config
    if not isinstance(normalized_config, CaMMLConfig):
        raise TypeError("config must be CaMMLConfig")
    levels = level_order_from_edges(len(names), expert.raw_edges)
    prior = level_order_prior(
        len(names),
        levels,
        prior_probability=prior_probability,
        arc_probability=normalized_config.arc_probability,
    )
    result = fit_camml(
        observations,
        prior,
        data_score=data_score,
        config=normalized_config,
        equivalent_sample_size=equivalent_sample_size,
        cardinalities=cardinalities,
        standardize_data=standardize_data,
    )
    return CaMMLOrderResult(
        camml=result,
        expert=expert,
        level_order=levels,
        prior_probability=prior_probability,
        prior=prior,
    )


def fit_camml_with_llm(
    data: object,
    node_names: Sequence[str],
    context: str,
    descriptions: Mapping[str, str] | None = None,
    *,
    triplet_config: TripletConfig | None = None,
    upstream_root: str | Path | None = None,
    prior_probability: float = 0.9,
    data_score: DataScoreName = "adaptive-mml",
    config: CaMMLConfig | None = None,
    equivalent_sample_size: float = 1.0,
    cardinalities: Sequence[int] | None = None,
    standardize_data: bool = True,
) -> CaMMLOrderResult:
    """Run the authors' triplet provider once, then fit Causal Order + CaMML."""

    names = normalize_node_names(node_names, minimum=3)
    expert = query_triplet_order(
        names,
        context,
        descriptions,
        config=triplet_config,
        upstream_root=upstream_root,
    )
    return fit_camml_with_causal_order(
        data,
        names,
        expert,
        prior_probability=prior_probability,
        data_score=data_score,
        config=config,
        equivalent_sample_size=equivalent_sample_size,
        cardinalities=cardinalities,
        standardize_data=standardize_data,
    )


__all__ = [
    "fit_camml_with_causal_order",
    "fit_camml_with_llm",
    "fit_pc_with_causal_order",
    "fit_pc_with_llm",
    "query_triplet_order",
]
