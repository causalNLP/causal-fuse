from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from cd_llm.core.finite_support import (
    Endpoint,
    Graph,
    PartialGraph,
    point_prediction_metrics,
)
from cd_llm.methods.fci.graph import ARROW, CIRCLE, PAGGraph, TAIL


Regime = str


ABSENT = "absent"
ENDPOINT_STATE_LABELS = {
    (TAIL, TAIL): "tail-tail",
    (TAIL, ARROW): "tail-arrow",
    (ARROW, TAIL): "arrow-tail",
    (ARROW, ARROW): "arrow-arrow",
    (CIRCLE, CIRCLE): "circle-circle",
    (CIRCLE, ARROW): "circle-arrow",
    (ARROW, CIRCLE): "arrow-circle",
    (TAIL, CIRCLE): "tail-circle",
    (CIRCLE, TAIL): "circle-tail",
}


@dataclass(frozen=True)
class SkeletonMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    shd: int
    false_positive_rate: float | None
    false_negative_rate: float | None
    precision: float | None
    recall: float | None
    f1: float


@dataclass(frozen=True)
class TypedGraphMetrics:
    shd: int
    shd_norm: float
    typed_correct: int
    typed_precision: float | None
    typed_recall: float | None


@dataclass(frozen=True)
class PCBootstrapProbabilities:
    directed: Dict[Tuple[str, str], float]
    undirected: Dict[Tuple[str, str], float]
    bidirected: Dict[Tuple[str, str], float]


def ceg_fuse_point_metrics(
    predicted: PAGGraph,
    target: PAGGraph,
    regime: Regime,
    *,
    truth_graph: Graph | None = None,
) -> dict[str, int | float | None | str]:
    """Score one PAG/CPDAG as a CEG-FUSE point-mass distribution."""
    if set(predicted.nodes) != set(target.nodes):
        raise ValueError("Predicted and target graphs must have the same node set.")
    nodes = tuple(target.nodes)
    if regime == "dag" and truth_graph is not None:
        metrics = point_prediction_metrics(_to_finite_partial(predicted, nodes), truth_graph)
        distance = metrics["expected_distance"]
        return {
            "structural_hamming_distance": int(round(distance * 2)),
            **metrics,
            "entropy": 0.0,
            "high_mass_set_size": 1,
            "high_mass_truth_coverage": float(distance == 0.0),
            "neighborhood_mass": float(distance == 0.0),
            "mode_distance": distance / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0.0,
            "endpoint_f1": None,
        }
    typed = typed_graph_metrics(predicted, target)
    adjacency = skeleton_metrics(predicted, target)
    orientation_f1 = _f1_from_precision_recall(
        typed.typed_precision, typed.typed_recall
    )
    return {
        "structural_hamming_distance": typed.shd,
        "expected_distance": float(typed.shd) / (2 if regime == "dag" else 1),
        "energy_score": float(typed.shd) / (2 if regime == "dag" else 1),
        "entropy": 0.0,
        "high_mass_set_size": 1,
        "high_mass_truth_coverage": float(typed.shd == 0),
        "neighborhood_mass": float(typed.shd == 0),
        "mode_distance": typed.shd_norm,
        "adjacency_f1": adjacency.f1,
        "orientation_f1": orientation_f1,
        "endpoint_f1": orientation_f1,
    }


def _to_finite_partial(
    graph: PAGGraph,
    nodes: Sequence[str],
) -> PartialGraph:
    endpoint_by_mark = {
        TAIL: Endpoint.TAIL,
        ARROW: Endpoint.ARROW,
        CIRCLE: Endpoint.CIRCLE,
    }
    node_count = len(nodes)
    marks = np.zeros((node_count, node_count), dtype=int)
    for first, left in enumerate(nodes):
        for second in range(first + 1, node_count):
            right = nodes[second]
            if not graph.has_edge(left, right):
                continue
            left_mark, right_mark = graph.endpoints(left, right)
            try:
                marks[first, second] = int(endpoint_by_mark[left_mark])
                marks[second, first] = int(endpoint_by_mark[right_mark])
            except KeyError as error:
                raise ValueError(
                    f"Unsupported endpoint state for {left}-{right}: "
                    f"{(left_mark, right_mark)}"
                ) from error
    return PartialGraph.from_matrix(marks, "pag")


def _f1_from_precision_recall(
    precision: float | None, recall: float | None
) -> float:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def endpoint_state(graph: PAGGraph, left: str, right: str) -> str:
    if not graph.has_edge(left, right):
        return ABSENT
    endpoints = graph.endpoints(left, right)
    try:
        return ENDPOINT_STATE_LABELS[endpoints]
    except KeyError as error:
        raise ValueError(f"Unsupported endpoint state for {left}-{right}: {endpoints}") from error


def bootstrap_endpoint_frequencies(
    graphs: Sequence[PAGGraph],
    nodes: Sequence[str],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not graphs:
        raise ValueError("At least one bootstrap graph is required.")
    expected_nodes = set(nodes)
    for graph in graphs:
        if set(graph.nodes) != expected_nodes:
            raise ValueError("Every bootstrap graph must contain the requested node set.")

    frequencies: Dict[Tuple[str, str], Dict[str, float]] = {}
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            counts: Dict[str, int] = {}
            for graph in graphs:
                state = endpoint_state(graph, left, right)
                counts[state] = counts.get(state, 0) + 1
            frequencies[(left, right)] = {
                state: count / len(graphs)
                for state, count in sorted(counts.items())
            }
    return frequencies


def bootstrap_pc_edge_probabilities(
    graphs: Sequence[PAGGraph],
    nodes: Sequence[str],
) -> PCBootstrapProbabilities:
    """Paper Pattern-2 PC probabilities for directed and undirected edges."""
    if not graphs:
        raise ValueError("At least one bootstrap graph is required.")
    expected_nodes = set(nodes)
    if any(set(graph.nodes) != expected_nodes for graph in graphs):
        raise ValueError("Every bootstrap graph must contain the requested node set.")

    directed_counts: Dict[Tuple[str, str], int] = {}
    undirected_counts: Dict[Tuple[str, str], int] = {}
    bidirected_counts: Dict[Tuple[str, str], int] = {}
    for graph in graphs:
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                if not graph.has_edge(left, right):
                    continue
                endpoints = graph.endpoints(left, right)
                if endpoints == (TAIL, ARROW):
                    key = (left, right)
                    directed_counts[key] = directed_counts.get(key, 0) + 1
                elif endpoints == (ARROW, TAIL):
                    key = (right, left)
                    directed_counts[key] = directed_counts.get(key, 0) + 1
                elif endpoints == (TAIL, TAIL):
                    key = tuple(sorted((left, right)))
                    undirected_counts[key] = undirected_counts.get(key, 0) + 1
                elif endpoints == (ARROW, ARROW):
                    key = tuple(sorted((left, right)))
                    bidirected_counts[key] = bidirected_counts.get(key, 0) + 1
                else:
                    raise ValueError(
                        "Paper PC bootstrap probabilities support only directed, "
                        f"undirected, and bidirected edges; found {endpoints} for "
                        f"{left!r}-{right!r}."
                    )

    denominator = float(len(graphs))
    return PCBootstrapProbabilities(
        directed={
            key: count / denominator
            for key, count in sorted(directed_counts.items())
        },
        undirected={
            key: count / denominator
            for key, count in sorted(undirected_counts.items())
        },
        bidirected={
            key: count / denominator
            for key, count in sorted(bidirected_counts.items())
        },
    )


def skeleton_metrics(predicted: PAGGraph, target: PAGGraph) -> SkeletonMetrics:
    if set(predicted.nodes) != set(target.nodes):
        raise ValueError("Predicted and target graphs must have the same node set.")
    true_positive = false_positive = false_negative = true_negative = 0
    nodes = tuple(target.nodes)
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            predicted_edge = predicted.has_edge(left, right)
            target_edge = target.has_edge(left, right)
            if predicted_edge and target_edge:
                true_positive += 1
            elif predicted_edge:
                false_positive += 1
            elif target_edge:
                false_negative += 1
            else:
                true_negative += 1
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    f1 = (
        2 * true_positive / (2 * true_positive + false_positive + false_negative)
        if true_positive or false_positive or false_negative
        else 1.0
    )
    return SkeletonMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        shd=false_positive + false_negative,
        false_positive_rate=(
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else None
        ),
        false_negative_rate=(
            false_negative / (false_negative + true_positive)
            if false_negative + true_positive
            else None
        ),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def typed_graph_metrics(predicted: PAGGraph, target: PAGGraph) -> TypedGraphMetrics:
    """Compute endpoint-aware SHD and exact typed-edge precision/recall."""
    if set(predicted.nodes) != set(target.nodes):
        raise ValueError("Predicted and target graphs must have the same node set.")
    shd = 0
    typed_correct = 0
    predicted_edges = 0
    target_edges = 0
    nodes = tuple(target.nodes)
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            predicted_edge = predicted.has_edge(left, right)
            target_edge = target.has_edge(left, right)
            predicted_edges += int(predicted_edge)
            target_edges += int(target_edge)
            if predicted_edge != target_edge:
                shd += 2
                continue
            if not predicted_edge:
                continue
            predicted_endpoints = predicted.endpoints(left, right)
            target_endpoints = target.endpoints(left, right)
            shd += int(predicted_endpoints[0] != target_endpoints[0])
            shd += int(predicted_endpoints[1] != target_endpoints[1])
            typed_correct += int(predicted_endpoints == target_endpoints)

    max_shd = len(nodes) * (len(nodes) - 1)
    return TypedGraphMetrics(
        shd=shd,
        shd_norm=(shd / max_shd if max_shd else 0.0),
        typed_correct=typed_correct,
        typed_precision=(
            typed_correct / predicted_edges if predicted_edges else None
        ),
        typed_recall=typed_correct / target_edges if target_edges else None,
    )


def serialize_bootstrap_frequencies(
    frequencies: Mapping[Tuple[str, str], Mapping[str, float]],
) -> list[dict[str, object]]:
    return [
        {
            "left": left,
            "right": right,
            "states": dict(sorted(states.items())),
        }
        for (left, right), states in sorted(frequencies.items())
    ]


def serialize_pc_bootstrap_probabilities(
    probabilities: PCBootstrapProbabilities,
) -> dict[str, list[dict[str, object]]]:
    def records(values: Mapping[Tuple[str, str], float]) -> list[dict[str, object]]:
        return [
            {
                "left": left,
                "right": right,
                "probability": probability,
            }
            for (left, right), probability in sorted(values.items())
        ]

    return {
        "directed": records(probabilities.directed),
        "undirected": records(probabilities.undirected),
        "bidirected": records(probabilities.bidirected),
    }
