"""Target-independent adapters from project evidence to CaMML priors."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

from cd_llm.core.finite_support import Claim, EvidenceGraph, EvidenceNode, Graph

from .priors import (
    default_global_probabilities,
    global_pair_prior,
    local_pair_prior,
)
from .types import CaMMLPrior, ExpertGraphPrior, GlobalPairPrior, LocalPairPrior


@dataclass(frozen=True, order=True)
class EvidenceContribution:
    evidence_id: str
    claim_id: str
    relation: str
    signed_log_odds: float


@dataclass(frozen=True, order=True)
class EvidenceExclusion:
    evidence_id: str
    reason: Literal["provenance_duplicate", "context_incompatible"]


@dataclass(frozen=True)
class EvidenceAdaptation:
    prior: CaMMLPrior
    contributions: tuple[EvidenceContribution, ...]
    used_evidence_ids: tuple[str, ...]
    exclusions: tuple[EvidenceExclusion, ...]


def _logistic(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate_claim(claim: Claim, node_count: int) -> None:
    if not isinstance(claim, Claim):
        raise TypeError("claims must contain Claim values")
    if claim.kind == "whole_graph":
        if claim.guess is None or not isinstance(claim.guess, Graph) or claim.guess.regime != "dag":
            raise ValueError("whole_graph evidence requires a DAG guess")
        if claim.guess.node_count != node_count:
            raise ValueError("whole_graph guess has the wrong node count")
        return
    if claim.source < 0 or claim.target < 0 or claim.source == claim.target:
        raise ValueError("pairwise claims require distinct nonnegative endpoints")
    if claim.source >= node_count or claim.target >= node_count:
        raise ValueError("claim endpoint is outside the graph")


def adapt_evidence_to_prior(
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    node_count: int,
    arc_probability: float = 0.5,
    global_prior_samples: int = 10_000,
    global_prior_seed: int = 123_456,
    whole_graph_distance: Literal["dag", "tom"] = "dag",
) -> EvidenceAdaptation:
    """Map representative evidence confidence to elicited marginal probabilities."""

    normalized_claims = tuple(claims)
    if not isinstance(evidence, EvidenceGraph):
        raise TypeError("evidence must be an EvidenceGraph")
    if node_count < 1:
        raise ValueError("node_count must be positive")
    raw_nodes = tuple(evidence.nodes)
    if len({node.evidence_id for node in raw_nodes}) != len(raw_nodes):
        raise ValueError("evidence identifiers must be unique")
    representatives = evidence.representatives()
    representative_ids = {node.evidence_id for node in representatives}
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
    compatible = tuple(node for node in representatives if node.context_compatible)

    relation_odds: defaultdict[tuple[object, ...], float] = defaultdict(float)
    contributions: list[EvidenceContribution] = []
    graph_priors: list[ExpertGraphPrior] = []
    for node in compatible:
        if not isinstance(node, EvidenceNode):
            raise TypeError("evidence nodes must be EvidenceNode values")
        if not 0 <= node.claim_index < len(normalized_claims):
            raise ValueError("evidence claim_index is outside the claim collection")
        if node.stance not in (-1, 1):
            raise ValueError("evidence stance must be -1 or +1")
        claim = normalized_claims[node.claim_index]
        _validate_claim(claim, node_count)
        confidence = min(max(float(node.confidence), 1e-12), 1.0 - 1e-12)
        magnitude = math.log(confidence / (1.0 - confidence))
        sign = int(node.stance)
        if claim.kind == "directed":
            key = ("directed", claim.source, claim.target)
        elif claim.kind == "directed_exclusion":
            key = ("directed", claim.source, claim.target)
            sign *= -1
        elif claim.kind == "adjacency":
            key = ("adjacency", min(claim.source, claim.target), max(claim.source, claim.target))
        elif claim.kind == "non_adjacency":
            key = ("adjacency", min(claim.source, claim.target), max(claim.source, claim.target))
            sign *= -1
        elif claim.kind == "ancestry":
            key = ("ancestry", claim.source, claim.target)
        elif claim.kind == "ordering":
            key = ("ordering", claim.source, claim.target)
        elif claim.kind == "whole_graph":
            assert claim.guess is not None
            confidence = _logistic(sign * magnitude)
            graph_priors.append(ExpertGraphPrior(claim.guess, confidence, whole_graph_distance))
            contributions.append(
                EvidenceContribution(node.evidence_id, claim.claim_id, "whole_graph", sign * magnitude)
            )
            continue
        else:
            raise ValueError(
                f"claim kind {claim.kind!r} has no CaMML DAG-prior representation"
            )
        relation_odds[key] += sign * magnitude
        contributions.append(
            EvidenceContribution(
                node.evidence_id,
                claim.claim_id,
                ":".join(str(value) for value in key),
                sign * magnitude,
            )
        )

    local_inputs: defaultdict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    global_inputs: defaultdict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for key, log_odds in sorted(relation_odds.items(), key=lambda item: repr(item[0])):
        kind = str(key[0])
        source, target = int(key[1]), int(key[2])
        probability = _logistic(log_odds)
        left, right = min(source, target), max(source, target)
        if kind == "directed":
            field = "left_to_right" if source == left else "right_to_left"
            local_inputs[(left, right)][field] = probability
        elif kind == "adjacency":
            local_inputs[(left, right)]["adjacency"] = probability
        elif kind == "ordering":
            local_inputs[(left, right)]["left_before_right"] = (
                probability if source == left else 1.0 - probability
            )
        elif kind == "ancestry":
            field = "left_ancestor_right" if source == left else "right_ancestor_left"
            global_inputs[(left, right)][field] = probability
        else:
            raise AssertionError("unknown relation kind")

    local: list[LocalPairPrior] = []
    for (left, right), values in sorted(local_inputs.items()):
        local.append(
            local_pair_prior(
                left,
                right,
                default_arc_probability=arc_probability,
                **values,
            )
        )
    defaults = default_global_probabilities(
        node_count,
        arc_probability,
        global_prior_samples,
        global_prior_seed,
    ) if global_inputs else (0.25, 0.25, 0.25, 0.25)
    global_values: list[GlobalPairPrior] = []
    for (left, right), values in sorted(global_inputs.items()):
        global_values.append(
            global_pair_prior(left, right, defaults=defaults, **values)
        )
    prior = CaMMLPrior(
        node_count,
        tuple(local),
        tuple(global_values),
        tuple(graph_priors),
    )
    return EvidenceAdaptation(
        prior,
        tuple(contributions),
        tuple(node.evidence_id for node in compatible),
        tuple(sorted(exclusions)),
    )


def prior_from_evidence(
    claims: Sequence[Claim],
    evidence: EvidenceGraph,
    *,
    node_count: int,
    arc_probability: float = 0.5,
    global_prior_samples: int = 10_000,
) -> CaMMLPrior:
    return adapt_evidence_to_prior(
        claims,
        evidence,
        node_count=node_count,
        arc_probability=arc_probability,
        global_prior_samples=global_prior_samples,
    ).prior


__all__ = [
    "EvidenceAdaptation",
    "EvidenceContribution",
    "EvidenceExclusion",
    "adapt_evidence_to_prior",
    "prior_from_evidence",
]
