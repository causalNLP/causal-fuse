from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

CIRCLE = "circle"
TAIL = "tail"
ARROW = "arrow"


@dataclass
class PAGGraph:
    nodes: Tuple[str, ...]
    _marks: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._marks:
            self._marks = {
                node: {other: None for other in self.nodes if other != node}
                for node in self.nodes
            }

    @classmethod
    def complete(cls, nodes: Sequence[str], mark: str = TAIL) -> "PAGGraph":
        graph = cls(tuple(nodes))
        for i, left in enumerate(graph.nodes):
            for right in graph.nodes[i + 1 :]:
                graph.add_edge(left, right, mark, mark)
        return graph

    def copy(self) -> "PAGGraph":
        marks = {
            node: {other: mark for other, mark in row.items()}
            for node, row in self._marks.items()
        }
        return PAGGraph(self.nodes, marks)

    def add_edge(self, left: str, right: str, left_mark: str, right_mark: str) -> None:
        self._marks[left][right] = left_mark
        self._marks[right][left] = right_mark

    def remove_edge(self, left: str, right: str) -> None:
        self._marks[left][right] = None
        self._marks[right][left] = None

    def has_edge(self, left: str, right: str) -> bool:
        return self._marks[left][right] is not None

    def neighbors(self, node: str) -> List[str]:
        return [other for other, mark in self._marks[node].items() if mark is not None]

    def degree(self, node: str) -> int:
        return len(self.neighbors(node))

    def endpoints(self, left: str, right: str) -> Tuple[Optional[str], Optional[str]]:
        return self._marks[left][right], self._marks[right][left]

    def mark_at(self, node: str, other: str) -> Optional[str]:
        return self._marks[node][other]

    def set_mark(self, node: str, other: str, mark: str) -> bool:
        if not self.has_edge(node, other):
            raise ValueError(f"Cannot mark a missing edge: {node}-{other}")
        if self._marks[node][other] == mark:
            return False
        self._marks[node][other] = mark
        return True

    def set_edge(self, left: str, right: str, left_mark: str, right_mark: str) -> bool:
        changed = False
        changed |= self.set_mark(left, right, left_mark)
        changed |= self.set_mark(right, left, right_mark)
        return changed

    def orient_into(self, source: str, target: str) -> bool:
        return self.set_mark(target, source, ARROW)

    def orient_from_to(self, source: str, target: str) -> bool:
        changed = self.set_mark(source, target, TAIL)
        changed |= self.set_mark(target, source, ARROW)
        return changed

    def all_edges(self) -> List[Tuple[str, str]]:
        edges: List[Tuple[str, str]] = []
        for i, left in enumerate(self.nodes):
            for right in self.nodes[i + 1 :]:
                if self.has_edge(left, right):
                    edges.append((left, right))
        return edges

    def ordered_edges(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for left, right in self.all_edges():
            pairs.append((left, right))
            pairs.append((right, left))
        return pairs

    def unshielded_triples(self) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        for middle in self.nodes:
            neighbors = sorted(self.neighbors(middle))
            for i, left in enumerate(neighbors):
                for right in neighbors[i + 1 :]:
                    if not self.has_edge(left, right):
                        triples.append((left, middle, right))
        return triples

    def is_triangle(self, left: str, middle: str, right: str) -> bool:
        return (
            self.has_edge(left, middle)
            and self.has_edge(middle, right)
            and self.has_edge(left, right)
        )

    def is_collider(self, left: str, middle: str, right: str) -> bool:
        if not (self.has_edge(left, middle) and self.has_edge(middle, right)):
            return False
        return self.mark_at(middle, left) == ARROW and self.mark_at(middle, right) == ARROW

    def is_directed_edge(self, source: str, target: str) -> bool:
        return self.has_edge(source, target) and self.endpoints(source, target) == (TAIL, ARROW)

    def has_directed_path(self, source: str, target: str) -> bool:
        stack = [source]
        visited: Set[str] = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.neighbors(current):
                if self.is_directed_edge(current, neighbor):
                    stack.append(neighbor)
        return False

    def reset_all_marks(self, mark: str) -> None:
        for left, right in self.all_edges():
            self._marks[left][right] = mark
            self._marks[right][left] = mark

    def edge_strings(self) -> List[str]:
        marker = {CIRCLE: "o", TAIL: "-", ARROW: ">"}
        edges: List[str] = []
        for left, right in self.all_edges():
            left_mark, right_mark = self.endpoints(left, right)
            edges.append(f"{left} {marker[left_mark]}-{marker[right_mark]} {right}")
        return sorted(edges)
