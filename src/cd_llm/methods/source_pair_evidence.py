"""Experimental dependence pooling of Group I answers from one model/context.

The three states describe adjacency only. A missing edge can still coexist
with an indirect causal path. Minimum confidence is a heuristic, not a
calibrated probability of the intersection of several uncertain statements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from cd_llm.core.finite_support import (
    Claim, EvidenceGraph, EvidenceNode, Graph, evaluate_claim,
    validate_soft_confidence_cap,
)

STATES = frozenset((0, 1, 2))  # no edge, forward edge, reverse edge


@dataclass(frozen=True)
class SourcePairGroup:
    source: int
    target: int
    source_family: str
    allowed_states: frozenset[int] = STATES
    confidence: float = 1.0
    response_count: int = 0

    @property
    def claim(self) -> Claim | None:
        if not self.allowed_states or self.allowed_states == STATES:
            return None
        a, b = self.source, self.target
        kind, truth, source, target = {
            frozenset((0,)): ('adjacency', 0, a, b),
            frozenset((1,)): ('directed', 1, a, b),
            frozenset((2,)): ('directed', 1, b, a),
            frozenset((0, 1)): ('directed', 0, b, a),
            frozenset((0, 2)): ('directed', 0, a, b),
            frozenset((1, 2)): ('adjacency', 1, a, b),
        }[self.allowed_states]
        return Claim(f'pair-{a}-{b}', 'I', kind, truth, source, target)

    def log_value(self, graph: Graph) -> float:
        claim = self.claim
        if claim is None:
            return 0.0
        agrees = evaluate_claim(graph, claim) == bool(claim.truth)
        return math.log(self.confidence if agrees else 1.0 - self.confidence)

    def audit(self) -> dict:
        return dict(source=self.source, target=self.target, source_family=self.source_family,
                    allowed_states=sorted(self.allowed_states), confidence=self.confidence,
                    response_count=self.response_count, contradiction=not self.allowed_states,
                    claim=asdict(self.claim) if self.claim else None)


class SourcePairEvidence:
    def __init__(self, source_family: str, *, soft_confidence_cap: float = .99):
        if not isinstance(source_family, str) or not source_family.strip():
            raise ValueError('grouped evidence requires a model/context source family')
        self.source_family = source_family
        self.soft_confidence_cap = validate_soft_confidence_cap(soft_confidence_cap)
        self.groups: dict[tuple[int, int], SourcePairGroup] = {}

    def add(self, claim: Claim, confidence: float) -> tuple[SourcePairGroup, SourcePairGroup]:
        if claim.kind not in ('directed', 'adjacency') or claim.group != 'I':
            raise ValueError('source pair pooling requires literal Group I claims')
        if claim.source == claim.target or not math.isfinite(confidence) or not .5 <= confidence <= 1:
            raise ValueError('invalid pair or confidence')
        a, b = sorted((claim.source, claim.target))
        state = 1 if claim.source == a else 2
        allowed = frozenset((1, 2)) if claim.kind == 'adjacency' else frozenset((state,))
        if not claim.truth:
            allowed = STATES - allowed
        old = self.groups.get((a, b), SourcePairGroup(a, b, f'{self.source_family}/pair-{a}-{b}'))
        new = replace(old, allowed_states=old.allowed_states & allowed,
                      confidence=min(old.confidence, confidence, self.soft_confidence_cap),
                      response_count=old.response_count + 1)
        self.groups[a, b] = new
        return old, new

    def effective_evidence(self) -> tuple[tuple[Claim, ...], EvidenceGraph]:
        retained = [self.groups[key] for key in sorted(self.groups) if self.groups[key].claim is not None]
        claims = tuple(group.claim for group in retained)
        nodes = tuple(EvidenceNode(
            f'pair-evidence-{group.source}-{group.target}', index,
            1 if claims[index].truth else -1, group.confidence,
            group.source_family, 'pooled-pair', 'target', True, True,
            treatment='soft_evidence') for index, group in enumerate(retained))
        return claims, EvidenceGraph('target', nodes)

    def log_value(self, graph: Graph) -> float:
        return math.fsum(group.log_value(graph) for group in self.groups.values())
