"""Canonical expert opinions for the Amirkhani et al. (2017) baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class EdgeState(str, Enum):
    """One of the paper's three possible states for an ordered node pair."""

    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    ABSENT = "absent"


def _validate_endpoint(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer node index")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _validate_expert_id(expert_id: str) -> None:
    if not isinstance(expert_id, str):
        raise TypeError("expert_id must be a string")
    if not expert_id:
        raise ValueError("expert_id must not be empty")


@dataclass(frozen=True, order=True)
class ExpertOpinion:
    """An expert's categorical label for one canonical node pair.

    Direct construction is normalized so that ``left < right``.  Reordering
    the endpoints also reverses a directed state, preserving its meaning.
    """

    expert_id: str
    left: int
    right: int
    state: EdgeState

    def __post_init__(self) -> None:
        _validate_expert_id(self.expert_id)
        _validate_endpoint(self.left, "left")
        _validate_endpoint(self.right, "right")
        if self.left == self.right:
            raise ValueError("an opinion cannot relate a node to itself")
        try:
            state = EdgeState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown edge state {self.state!r}") from error
        left, right = self.left, self.right
        if right < left:
            left, right = right, left
            if state == EdgeState.LEFT_TO_RIGHT:
                state = EdgeState.RIGHT_TO_LEFT
            elif state == EdgeState.RIGHT_TO_LEFT:
                state = EdgeState.LEFT_TO_RIGHT
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "state", state)

    @classmethod
    def directed(
        cls, expert_id: str, source: int, target: int
    ) -> "ExpertOpinion":
        """Construct the opinion ``source -> target``."""

        _validate_endpoint(source, "source")
        _validate_endpoint(target, "target")
        if source == target:
            raise ValueError("an opinion cannot relate a node to itself")
        if source < target:
            return cls(expert_id, source, target, EdgeState.LEFT_TO_RIGHT)
        return cls(expert_id, target, source, EdgeState.RIGHT_TO_LEFT)

    @classmethod
    def absent(cls, expert_id: str, left: int, right: int) -> "ExpertOpinion":
        """Construct an opinion that the pair has no edge."""

        return cls(expert_id, left, right, EdgeState.ABSENT)

    @property
    def pair(self) -> tuple[int, int]:
        return self.left, self.right

    @property
    def directed_edge(self) -> tuple[int, int] | None:
        if self.state == EdgeState.LEFT_TO_RIGHT:
            return self.left, self.right
        if self.state == EdgeState.RIGHT_TO_LEFT:
            return self.right, self.left
        return None


def normalize_expert_ids(expert_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate, deduplicate, and deterministically order expert identifiers."""

    try:
        normalized = tuple(expert_ids)
    except TypeError as error:
        raise TypeError("expert_ids must be an iterable of strings") from error
    for expert_id in normalized:
        _validate_expert_id(expert_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must not contain duplicates")
    return tuple(sorted(normalized))


def normalize_opinions(
    opinions: Iterable[ExpertOpinion], *, node_count: int | None = None
) -> tuple[ExpertOpinion, ...]:
    """Validate opinions and reject multiple responses for one expert/pair."""

    if node_count is not None:
        if isinstance(node_count, bool) or not isinstance(node_count, int):
            raise TypeError("node_count must be an integer")
        if node_count <= 0:
            raise ValueError("node_count must be positive")
    try:
        normalized = tuple(opinions)
    except TypeError as error:
        raise TypeError("opinions must be an iterable of ExpertOpinion values") from error
    seen: set[tuple[str, int, int]] = set()
    for opinion in normalized:
        if not isinstance(opinion, ExpertOpinion):
            raise TypeError("opinions must contain ExpertOpinion values")
        if node_count is not None and opinion.right >= node_count:
            raise ValueError(
                f"opinion endpoint is outside [0, {node_count})"
            )
        key = opinion.expert_id, opinion.left, opinion.right
        if key in seen:
            raise ValueError(
                "an expert may provide at most one opinion for each node pair"
            )
        seen.add(key)
    return tuple(sorted(normalized))
