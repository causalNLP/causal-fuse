"""Target-independent adapter from project evidence to ECA edge priors."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Sequence

from cd_llm.core.finite_support import Claim, EvidenceGraph, EvidenceNode

from .prior import EdgeLogOdds, aggregate_edge_log_odds, confidence_log_odds
from .types import ECAPrior


ExclusionReason = Literal["provenance_duplicate", "context_incompatible"]


@dataclass(frozen=True, order=True)
class EvidenceContribution:
    """One representative's signed ordered-edge factor."""

    evidence_id: str
    claim_id: str
    source: int
    target: int
    signed_log_odds: float


@dataclass(frozen=True, order=True)
class EvidenceExclusion:
    evidence_id: str
    reason: ExclusionReason


@dataclass(frozen=True)
class EvidenceAdaptation:
    """An ECA prior plus an auditable account of evidence conversion."""

    prior: ECAPrior
    contributions: tuple[EvidenceContribution, ...]
    edge_log_odds: tuple[EdgeLogOdds, ...]
    used_evidence_ids: tuple[str, ...]
    exclusions: tuple[EvidenceExclusion, ...]

    @property
    def excluded_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.exclusions)


def _directed_factors(claim: Claim, stance: int) -> tuple[tuple[int, int, int], ...]:
    """Return ``(source, target, presence_sign)`` factors for one claim."""

    source, target = claim.source, claim.target
    if claim.kind == "directed":
        return ((source, target, stance),)
    if claim.kind == "directed_exclusion":
        return ((source, target, -stance),)
    if claim.kind == "adjacency":
        return ((source, target, stance), (target, source, stance))
    if claim.kind == "non_adjacency":
        return ((source, target, -stance), (target, source, -stance))
    raise ValueError(
        f"Group-I claim kind {claim.kind!r} is not supported by ECA"
    )


def _validate_evidence_nodes(
    evidence: EvidenceGraph, claim_count: int
) -> tuple[EvidenceNode, ...]:
    nodes = tuple(evidence.nodes)
    identifiers: set[str] = set()
    for node in nodes:
        if not isinstance(node, EvidenceNode):
            raise TypeError("evidence.nodes must contain EvidenceNode values")
        if not isinstance(node.evidence_id, str) or not node.evidence_id:
            raise ValueError("evidence identifiers must be nonempty strings")
        if node.evidence_id in identifiers:
            raise ValueError("evidence identifiers must be unique")
        identifiers.add(node.evidence_id)
        if isinstance(node.claim_index, bool) or not isinstance(
            node.claim_index, Integral
        ):
            raise TypeError("evidence claim_index values must be integers")
        if not 0 <= int(node.claim_index) < claim_count:
            raise ValueError("evidence claim_index is outside the claim collection")
        if isinstance(node.stance, bool) or not isinstance(node.stance, Integral):
            raise TypeError("evidence stances must be -1 or +1")
        if int(node.stance) not in (-1, 1):
            raise ValueError("evidence stances must be -1 or +1")
    return nodes


def adapt_evidence_to_prior(
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    node_count: int,
    sigma: float = 1.0,
) -> EvidenceAdaptation:
    """Convert context-compatible provenance representatives to an ECA prior.

    Only the supplied stance, confidence, provenance/factor identity, and
    context-compatibility annotation influence the conversion.  In particular,
    the evaluation-only ``Claim.truth`` and ``EvidenceNode.correct`` fields are
    deliberately never consulted.

    Positive adjacency factors are represented by equal existence factors in
    both directions.  A valid DAG can contain at most one, giving adjacency
    versus absence the requested likelihood ratio without choosing an
    orientation from the target graph.
    """

    try:
        normalized_claims = tuple(claims)
    except TypeError as error:
        raise TypeError("claims must be a sequence of Claim values") from error
    if any(not isinstance(claim, Claim) for claim in normalized_claims):
        raise TypeError("claims must contain Claim values")
    if not isinstance(evidence, EvidenceGraph):
        raise TypeError("evidence must be a cd_llm.core.finite_support.EvidenceGraph")
    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    node_count = int(node_count)
    if node_count <= 0:
        raise ValueError("node_count must be positive")

    raw_nodes = _validate_evidence_nodes(evidence, len(normalized_claims))
    representatives = evidence.representatives()
    representative_ids = {node.evidence_id for node in representatives}
    compatible = tuple(node for node in representatives if node.context_compatible)

    exclusions = [
        EvidenceExclusion(node.evidence_id, "provenance_duplicate")
        for node in raw_nodes
        if node.evidence_id not in representative_ids
    ]
    exclusions.extend(
        EvidenceExclusion(node.evidence_id, "context_incompatible")
        for node in representatives
        if not node.context_compatible
    )

    contributions: list[EvidenceContribution] = []
    for node in compatible:
        claim = normalized_claims[int(node.claim_index)]
        if claim.group != "I":
            raise ValueError(
                f"ECA supports Group-I claims only; evidence {node.evidence_id!r} "
                f"references Group-{claim.group}"
            )
        if claim.source < 0 or claim.target < 0 or claim.source == claim.target:
            raise ValueError("supported claims require distinct nonnegative endpoints")
        if claim.source >= node_count or claim.target >= node_count:
            raise ValueError(f"claim endpoint is outside [0, {node_count})")
        magnitude = confidence_log_odds(node.confidence, sigma=sigma)
        for source, target, presence_sign in _directed_factors(
            claim, int(node.stance)
        ):
            contributions.append(
                EvidenceContribution(
                    node.evidence_id,
                    claim.claim_id,
                    source,
                    target,
                    presence_sign * magnitude,
                )
            )

    prior, aggregated = aggregate_edge_log_odds(
        node_count,
        (
            (item.source, item.target, item.signed_log_odds)
            for item in contributions
        ),
    )
    return EvidenceAdaptation(
        prior=prior,
        contributions=tuple(contributions),
        edge_log_odds=aggregated,
        used_evidence_ids=tuple(node.evidence_id for node in compatible),
        exclusions=tuple(sorted(exclusions)),
    )


def prior_from_evidence(
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    node_count: int,
    sigma: float = 1.0,
) -> ECAPrior:
    """Convenience wrapper returning only the adapted edge prior."""

    return adapt_evidence_to_prior(
        claims,
        evidence,
        node_count=node_count,
        sigma=sigma,
    ).prior


__all__ = [
    "EvidenceAdaptation",
    "EvidenceContribution",
    "EvidenceExclusion",
    "ExclusionReason",
    "adapt_evidence_to_prior",
    "prior_from_evidence",
]
