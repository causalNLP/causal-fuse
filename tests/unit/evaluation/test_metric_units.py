import math

import numpy as np
import pytest

from cd_llm.core.finite_support import (
    Graph, PartialGraph, build_metric_workspace, distribution_metrics,
    graph_from_edges, normalized_structural_distance, point_prediction_metrics,
)
from cd_llm.metrics.units import convert_metrics, distance_factor, read_metric_csv


@pytest.mark.parametrize("p", [3, 20])
@pytest.mark.parametrize("edges", [[], [(1, 0)], [(0, 1), (1, 2)]])
def test_one_dag_error_is_one_regardless_of_node_count(p, edges):
    truth = graph_from_edges(p, [(0, 1)])
    estimate = graph_from_edges(p, edges)
    point = point_prediction_metrics(estimate, truth)
    assert point["expected_distance"] == point["energy_score"] == 1
    old = point_prediction_metrics(estimate, truth, normalized=True)
    assert old["expected_distance"] == pytest.approx(1 / math.comb(p, 2))
    distribution = distribution_metrics([estimate], np.array([1.]), truth, "dag")
    assert distribution["expected_distance"] == distribution["energy_score"] == 1


def test_unresolved_direction_is_half_an_error_on_twenty_nodes():
    truth = graph_from_edges(20, [(0, 1)])
    matrix = np.zeros((20, 20), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 1
    result = point_prediction_metrics(PartialGraph.from_matrix(matrix, "cpdag"), truth)
    assert result["expected_distance"] == .5
    assert result["adjacency_f1"] == 1
    assert result["orientation_f1"] == 0


@pytest.mark.parametrize("regime", ["dag", "mag"])
def test_energy_score_matches_explicit_pairwise_sum(regime):
    truth = graph_from_edges(4, [(0, 1)])
    graphs = [truth, graph_from_edges(4), graph_from_edges(4, [(1, 0)])]
    if regime == "mag":
        graphs.append(graph_from_edges(4, bidirected=[(0, 1)]))
    weights = np.arange(1, len(graphs) + 1, dtype=float)
    weights /= weights.sum()
    factor = distance_factor(4, regime)
    distance = lambda a, b: normalized_structural_distance(a, b, regime) * factor
    ed = sum(w * distance(g, truth) for w, g in zip(weights, graphs))
    es = ed - .5 * sum(w * v * distance(g, h) for w, g in zip(weights, graphs)
                      for v, h in zip(weights, graphs))
    workspace = build_metric_workspace(graphs, truth, regime)
    actual = distribution_metrics(graphs, weights, truth, regime, workspace=workspace)
    old = distribution_metrics(graphs, weights, truth, regime, normalized=True, workspace=workspace)
    assert actual["expected_distance"] == pytest.approx(ed)
    assert actual["energy_score"] == pytest.approx(es)
    for name in actual:
        assert actual[name] == pytest.approx(old[name] * (factor if name in ("expected_distance", "energy_score") else 1), nan_ok=True)


def test_mag_missing_edge_counts_two_endpoints():
    g = graph_from_edges(10, bidirected=[(0, 1)])
    assert distribution_metrics([graph_from_edges(10)], np.array([1.]), g, "mag")["expected_distance"] == 2


def test_single_node_distribution_is_zero():
    g = Graph((0,))
    result = distribution_metrics([g], np.array([1.]), g, "dag")
    assert result["expected_distance"] == result["energy_score"] == 0


def test_conversion_scales_uncertainty_but_preserves_other_quantities():
    data = dict(expected_distance_mean=.1, expected_distance_se=.01,
                energy_score=dict(mean=.05, paired_95_interval=[-.02, .1]),
                pd=.2, adjacency_f1=.8, brier=.1, count=10,
                contrasts=[dict(metric="expected_distance", mean=-.01, low=-.02, high=-.005, paired_seeds=10)])
    converted = convert_metrics(data, 190, source_scale="normalized")
    assert converted["expected_distance_mean"] == 19
    assert converted["expected_distance_se"] == pytest.approx(1.9)
    assert converted["energy_score"]["paired_95_interval"] == pytest.approx([-3.8, 19])
    assert converted["contrasts"][0]["low"] == pytest.approx(-3.8)
    assert converted["contrasts"][0]["paired_seeds"] == 10
    assert all(converted[k] == data[k] for k in ("pd", "adjacency_f1", "brier", "count"))
    assert convert_metrics(converted, 190, source_scale="unnormalized") == converted
    assert data["expected_distance_mean"] == .1


def test_reader_rejects_ambiguous_units_and_mixed_metadata(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("expected_distance_mean\n0.1\n")
    with pytest.raises(ValueError, match="missing metric units"):
        read_metric_csv(path, node_count=20)
    assert float(read_metric_csv(path, node_count=20, legacy_normalized=True)[0]["expected_distance_mean"]) == 19
    with pytest.raises(ValueError, match="disagrees"):
        convert_metrics(dict(metric_scale="unnormalized", expected_distance=19), 190, source_scale="normalized")
