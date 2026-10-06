"""Accuracy parameters and priors from Amirkhani et al. (2017)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_probability(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _validate_shape(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True)
class ExpertAccuracy:
    """The paper's three accuracy parameters ``(gamma1, gamma2, gamma3)``."""

    gamma1: float
    gamma2: float
    gamma3: float

    def __post_init__(self) -> None:
        gamma1 = _validate_probability(self.gamma1, "gamma1")
        gamma2 = _validate_probability(self.gamma2, "gamma2")
        gamma3 = _validate_probability(self.gamma3, "gamma3")
        total = gamma1 + gamma2
        if total > 1.0 + 1e-15:
            raise ValueError("gamma1 + gamma2 must not exceed one")
        if total > 1.0:
            gamma2 = 1.0 - gamma1
        object.__setattr__(self, "gamma1", gamma1)
        object.__setattr__(self, "gamma2", gamma2)
        object.__setattr__(self, "gamma3", gamma3)

    @classmethod
    def neutral(cls) -> "ExpertAccuracy":
        """Return the plan's uninformative EM initialization."""

        return cls(1.0 / 3.0, 1.0 / 3.0, 0.5)

    @property
    def existing_as_absent(self) -> float:
        return max(0.0, 1.0 - (self.gamma1 + self.gamma2))

    @property
    def absent_as_each_direction(self) -> float:
        return 0.5 * (1.0 - self.gamma3)


@dataclass(frozen=True)
class AccuracyPrior:
    """Dirichlet/Beta shapes for marginalizing an expert's accuracy."""

    alpha1: float
    alpha2: float
    alpha3: float
    beta1: float
    beta2: float

    def __post_init__(self) -> None:
        for label in ("alpha1", "alpha2", "alpha3", "beta1", "beta2"):
            object.__setattr__(self, label, _validate_shape(getattr(self, label), label))

    @classmethod
    def neutral(cls) -> "AccuracyPrior":
        """Return uniform Dirichlet(1,1,1) and Beta(1,1) priors."""

        return cls(1.0, 1.0, 1.0, 1.0, 1.0)

    @classmethod
    def from_mean(
        cls, mean_accuracy: ExpertAccuracy, concentration: float = 10.0
    ) -> "AccuracyPrior":
        """Construct the paper's equation-(18) prior from a mean accuracy."""

        if not isinstance(mean_accuracy, ExpertAccuracy):
            raise TypeError("mean_accuracy must be an ExpertAccuracy")
        concentration = _validate_shape(concentration, "concentration")
        residual = mean_accuracy.existing_as_absent
        shapes = (
            concentration * mean_accuracy.gamma1,
            concentration * mean_accuracy.gamma2,
            concentration * residual,
            concentration * mean_accuracy.gamma3,
            concentration * (1.0 - mean_accuracy.gamma3),
        )
        if any(shape <= 0.0 for shape in shapes):
            raise ValueError(
                "mean accuracy must be strictly inside both prior simplexes"
            )
        return cls(*shapes)
