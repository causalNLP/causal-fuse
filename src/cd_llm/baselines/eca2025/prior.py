"""Calibration and aggregation utilities for paper-native ECA priors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

from .types import ECAPrior, EdgeConstraint


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def confidence_log_odds(confidence: float, *, sigma: float = 1.0) -> float:
    """Map conditional response confidence to the ECA prior scale.

    The returned value satisfies ``sigma**2 * logit(c_eca) = logit(confidence)``.
    Finite evidence aggregation requires an interior confidence.
    """

    normalized_confidence = _finite_real(confidence, "confidence")
    if not 0.0 < normalized_confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    normalized_sigma = _finite_real(sigma, "sigma")
    if normalized_sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return math.log(normalized_confidence / (1.0 - normalized_confidence)) / (
        normalized_sigma * normalized_sigma
    )


def confidence_from_log_odds(log_odds: float) -> float:
    """Return a numerically stable logistic transform."""

    value = _finite_real(log_odds, "log_odds")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, order=True)
class EdgeLogOdds:
    """Aggregated signed log odds for one ordered edge."""

    source: int
    target: int
    log_odds: float

    def __post_init__(self) -> None:
        for label in ("source", "target"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{label} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{label} must be nonnegative")
            object.__setattr__(self, label, value)
        if self.source == self.target:
            raise ValueError("edge log odds cannot describe a self-edge")
        object.__setattr__(
            self, "log_odds", _finite_real(self.log_odds, "log_odds")
        )

    @property
    def probability(self) -> float:
        return confidence_from_log_odds(self.log_odds)


def aggregate_edge_log_odds(
    node_count: int,
    contributions: Iterable[tuple[int, int, float]],
) -> tuple[ECAPrior, tuple[EdgeLogOdds, ...]]:
    """Aggregate independent evidence factors by adding signed log odds.

    A total of exactly zero is retained in the diagnostics but omitted from
    the prior mask, so equal conflicting evidence has no optimization effect.
    """

    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    node_count = int(node_count)
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    grouped: dict[tuple[int, int], list[float]] = {}
    try:
        normalized = tuple(contributions)
    except TypeError as error:
        raise TypeError("contributions must be iterable") from error
    for item in normalized:
        try:
            source, target, log_odds = item
        except (TypeError, ValueError) as error:
            raise TypeError(
                "each contribution must be (source, target, signed_log_odds)"
            ) from error
        edge = EdgeLogOdds(source, target, log_odds)
        if edge.source >= node_count or edge.target >= node_count:
            raise ValueError(f"contribution endpoint is outside [0, {node_count})")
        grouped.setdefault((edge.source, edge.target), []).append(edge.log_odds)

    aggregated = tuple(
        EdgeLogOdds(source, target, math.fsum(grouped[(source, target)]))
        for source, target in sorted(grouped)
    )
    constraints = tuple(
        EdgeConstraint(item.source, item.target, item.probability)
        for item in aggregated
        if item.log_odds != 0.0
    )
    return ECAPrior(node_count, constraints), aggregated


__all__ = [
    "EdgeLogOdds",
    "aggregate_edge_log_odds",
    "confidence_log_odds",
    "confidence_from_log_odds",
]
