"""Modular project interface for Chen et al.'s linear ECA method."""

from .adapters import (
    EvidenceAdaptation,
    EvidenceContribution,
    EvidenceExclusion,
    adapt_evidence_to_prior,
    prior_from_evidence,
)
from .api import fit_eca, fit_eca_with_evidence
from .paper_backend import (
    CyclicThresholdedGraphError,
    PaperECAOptimizationError,
    acyclicity_and_gradient,
    data_loss_and_gradient,
    objective_and_subgradient,
    objective_terms,
    prior_log_score_and_gradient,
    smooth_objective_and_gradient,
)
from .prior import (
    EdgeLogOdds,
    aggregate_edge_log_odds,
    confidence_log_odds,
    confidence_from_log_odds,
)
from .types import (
    Backend,
    ECAConfig,
    ECADiagnostics,
    ECAPrior,
    ECAResult,
    EdgeConstraint,
)
from .upstream_backend import UPSTREAM_REVISION


__all__ = [
    "Backend",
    "CyclicThresholdedGraphError",
    "ECAConfig",
    "ECADiagnostics",
    "ECAPrior",
    "ECAResult",
    "EdgeConstraint",
    "EdgeLogOdds",
    "EvidenceAdaptation",
    "EvidenceContribution",
    "EvidenceExclusion",
    "PaperECAOptimizationError",
    "UPSTREAM_REVISION",
    "acyclicity_and_gradient",
    "adapt_evidence_to_prior",
    "aggregate_edge_log_odds",
    "confidence_log_odds",
    "confidence_from_log_odds",
    "data_loss_and_gradient",
    "fit_eca",
    "fit_eca_with_evidence",
    "objective_and_subgradient",
    "objective_terms",
    "prior_from_evidence",
    "prior_log_score_and_gradient",
    "smooth_objective_and_gradient",
]
