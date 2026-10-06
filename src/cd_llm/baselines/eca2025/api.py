"""Project-facing linear ECA API."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from cd_llm.core.finite_support import Claim, EvidenceGraph

from .adapters import adapt_evidence_to_prior
from .paper_backend import fit_paper_eca
from .types import ECAConfig, ECADiagnostics, ECAPrior, ECAResult
from .upstream_backend import fit_upstream_eca


def _data_node_count(data: np.ndarray) -> int:
    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("data must contain real numbers") from error
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("data must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must be finite")
    return int(values.shape[1])


def _fit_eca(
    data: np.ndarray,
    prior: ECAPrior | None = None,
    *,
    config: ECAConfig | None = None,
    upstream_root: str | Path | None = None,
    evidence_adaptation=None,
) -> ECAResult:
    """Fit linear ECA and return a project-native point-mass DAG result.

    ``backend='paper'`` is the scientifically preferred implementation of the
    published equations.  ``backend='upstream'`` reproduces the archived c980
    source behavior after repairing only its missing prior-loading call and
    threshold forwarding.
    """

    if config is None:
        config = ECAConfig()
    if not isinstance(config, ECAConfig):
        raise TypeError("config must be ECAConfig")
    node_count = _data_node_count(data)
    if prior is None:
        prior = ECAPrior.empty(node_count)
    if not isinstance(prior, ECAPrior):
        raise TypeError("prior must be ECAPrior or None")
    if prior.node_count != node_count:
        raise ValueError("prior node count does not match the data")
    mask, probabilities = prior.as_matrices()

    if config.backend == "paper":
        if upstream_root is not None:
            raise ValueError("upstream_root applies only to backend='upstream'")
        backend = fit_paper_eca(
            data,
            mask,
            probabilities,
            lambda1=config.lambda1,
            sigma=config.sigma,
            threshold=config.threshold,
            standardize_data=config.standardize_data,
            max_outer_iterations=config.max_iterations,
            max_inner_iterations=config.max_iterations,
            h_tolerance=config.h_tolerance,
            rho_max=config.rho_max,
            rho_multiplier=config.rho_multiplier,
        )
        backend_diagnostics = backend.diagnostics
        diagnostics = ECADiagnostics(
            iterations=backend_diagnostics.outer_iterations,
            inner_optimizations=backend_diagnostics.inner_optimizations,
            objective_evaluations=backend_diagnostics.objective_evaluations,
            gradient_evaluations=backend_diagnostics.gradient_evaluations,
            converged=backend_diagnostics.converged,
            acyclicity=backend_diagnostics.acyclicity,
            rho=backend_diagnostics.rho,
            alpha=backend_diagnostics.alpha,
            runtime_seconds=backend_diagnostics.runtime_seconds,
            optimizer_success=backend_diagnostics.optimizer_success,
            optimizer_status=backend_diagnostics.optimizer_status,
            optimizer_message=backend_diagnostics.optimizer_message,
        )
        terms = backend.terms
        prior_penalty = -(config.sigma * config.sigma) * terms.prior_log_score
        return ECAResult(
            graph=backend.graph,
            weighted_adjacency=backend.weighted_adjacency,
            binary_adjacency=backend.binary_adjacency,
            objective=terms.objective,
            data_loss=terms.data_loss,
            prior_log_likelihood=terms.prior_log_score,
            prior_penalty=prior_penalty,
            l1_penalty=terms.l1_penalty,
            diagnostics=diagnostics,
            config=config,
            prior=prior,
            evidence_adaptation=evidence_adaptation,
        )

    if not math.isclose(config.rho_multiplier, 10.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "the upstream c980 source hard-codes rho_multiplier=10; "
            "use backend='paper' to configure it"
        )
    backend = fit_upstream_eca(
        data,
        mask,
        probabilities,
        lambda1=config.lambda1,
        sigma=config.sigma,
        threshold=config.threshold,
        standardize_data=config.standardize_data,
        max_outer_iterations=config.max_iterations,
        h_tolerance=config.h_tolerance,
        rho_max=config.rho_max,
        upstream_root=upstream_root,
    )
    converged = backend.diagnostics.acyclicity <= config.h_tolerance
    diagnostics = ECADiagnostics(
        iterations=0,
        inner_optimizations=0,
        objective_evaluations=0,
        gradient_evaluations=0,
        converged=converged,
        acyclicity=backend.diagnostics.acyclicity,
        rho=0.0,
        alpha=0.0,
        runtime_seconds=backend.diagnostics.fit_seconds,
        optimizer_success=True,
        optimizer_status=None,
        optimizer_message=(
            "upstream c980 returned normally; optimization counters, final rho, "
            "and alpha are not exposed, so their numeric fields are zero"
        ),
    )
    terms = backend.terms
    return ECAResult(
        graph=backend.graph,
        weighted_adjacency=backend.weighted_adjacency,
        binary_adjacency=backend.binary_adjacency,
        objective=terms.objective,
        data_loss=terms.data_loss,
        prior_log_likelihood=terms.prior_log_score,
        prior_penalty=terms.prior_penalty,
        l1_penalty=terms.l1_penalty,
        diagnostics=diagnostics,
        config=config,
        prior=prior,
        evidence_adaptation=evidence_adaptation,
    )


def fit_eca(
    data: np.ndarray,
    prior: ECAPrior | None = None,
    *,
    config: ECAConfig | None = None,
    upstream_root: str | Path | None = None,
) -> ECAResult:
    """Fit linear ECA from a native edge-probability prior."""

    return _fit_eca(
        data,
        prior,
        config=config,
        upstream_root=upstream_root,
        evidence_adaptation=None,
    )


def fit_eca_with_evidence(
    data: np.ndarray,
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    config: ECAConfig | None = None,
    upstream_root: str | Path | None = None,
) -> ECAResult:
    """Adapt project evidence and fit ECA without consulting target truth."""

    if config is None:
        config = ECAConfig()
    if not isinstance(config, ECAConfig):
        raise TypeError("config must be ECAConfig")
    adaptation = adapt_evidence_to_prior(
        claims,
        evidence,
        node_count=_data_node_count(data),
        sigma=config.sigma,
    )
    return _fit_eca(
        data,
        adaptation.prior,
        config=config,
        upstream_root=upstream_root,
        evidence_adaptation=adaptation,
    )


__all__ = ["fit_eca", "fit_eca_with_evidence"]
