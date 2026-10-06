from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from cd_llm.methods.fci.graph import ARROW, PAGGraph, TAIL


@dataclass(frozen=True)
class ConstraintApplication:
    source: str
    target: str
    probability: float
    action: str
    status: str
    reason: str | None = None


def apply_fci_endpoint_constraints(
    graph: PAGGraph,
    probabilities: Mapping[Tuple[str, str], float],
    *,
    forbid_threshold: float = 0.05,
    require_threshold: float = 0.95,
) -> tuple[PAGGraph, list[ConstraintApplication]]:
    """Apply ordered ancestry judgments only to endpoints of existing PAG edges."""
    constrained = graph.copy()
    applications: list[ConstraintApplication] = []

    required = sorted(
        (
            (probability, source, target)
            for (source, target), probability in probabilities.items()
            if probability >= require_threshold and constrained.has_edge(source, target)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    handled_pairs: set[frozenset[str]] = set()
    for probability, source, target in required:
        pair = frozenset((source, target))
        reverse_probability = float(probabilities.get((target, source), 0.5))
        if reverse_probability >= require_threshold:
            if pair not in handled_pairs:
                applications.append(
                    ConstraintApplication(
                        source=source,
                        target=target,
                        probability=probability,
                        action="require_ancestry",
                        status="skipped",
                        reason="both directions exceeded the require threshold",
                    )
                )
                handled_pairs.add(pair)
            continue
        if constrained.has_directed_path(target, source):
            applications.append(
                ConstraintApplication(
                    source=source,
                    target=target,
                    probability=probability,
                    action="require_ancestry",
                    status="skipped",
                    reason="orientation would create a directed cycle",
                )
            )
            continue
        constrained.set_edge(source, target, TAIL, ARROW)
        applications.append(
            ConstraintApplication(
                source=source,
                target=target,
                probability=probability,
                action="require_ancestry",
                status="applied",
            )
        )

    forbidden = sorted(
        (
            (probability, source, target)
            for (source, target), probability in probabilities.items()
            if probability < forbid_threshold and constrained.has_edge(source, target)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for probability, source, target in forbidden:
        constrained.set_mark(source, target, ARROW)
        applications.append(
            ConstraintApplication(
                source=source,
                target=target,
                probability=probability,
                action="forbid_ancestry",
                status="applied",
            )
        )
    return constrained, applications


def build_pc_background_knowledge(
    nodes: Sequence[str],
    probabilities: Mapping[Tuple[str, str], float],
    *,
    forbid_threshold: float = 0.05,
    require_threshold: float = 0.95,
):
    try:
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    except ImportError as error:
        raise RuntimeError("causal-learn is required for the PC baseline.") from error

    knowledge = BackgroundKnowledge()
    graph_nodes = {name: GraphNode(name) for name in nodes}
    applications: list[ConstraintApplication] = []
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            probability = float(probabilities[(source, target)])
            if probability >= require_threshold:
                # Guard against contradictory background knowledge: upstream
                # thresholds each direction independently, so a saturated LLM
                # can demand X -> Y and Y -> X at once and causal-learn
                # resolves the conflict arbitrarily. Skip such pairs, as the
                # FCI constraint path already does.
                reverse_probability = float(probabilities.get((target, source), 0.0))
                if reverse_probability >= require_threshold:
                    applications.append(
                        ConstraintApplication(
                            source=source,
                            target=target,
                            probability=probability,
                            action="require_edge",
                            status="skipped",
                            reason="both directions exceeded the require threshold",
                        )
                    )
                    continue
                knowledge.add_required_by_node(graph_nodes[source], graph_nodes[target])
                applications.append(
                    ConstraintApplication(
                        source=source,
                        target=target,
                        probability=probability,
                        action="require_edge",
                        status="applied",
                    )
                )
            elif probability < forbid_threshold:
                knowledge.add_forbidden_by_node(graph_nodes[source], graph_nodes[target])
                applications.append(
                    ConstraintApplication(
                        source=source,
                        target=target,
                        probability=probability,
                        action="forbid_edge",
                        status="applied",
                    )
                )
    return knowledge, applications
