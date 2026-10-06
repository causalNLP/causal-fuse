from __future__ import annotations

import unittest

import numpy as np

from cd_llm.baselines.causal_order2025 import TripletConfig, TripletExpertResult
from cd_llm.baselines.causal_order2025.upstream_backend import UPSTREAM_REVISION
from cd_llm.core.finite_support import Endpoint, PartialGraph, graph_from_edges
from cd_llm.experiments.causal_order2025_table42 import (
    _expert,
    _expert_record,
    required_fallback_pairs,
    slice_for_task,
)


def _fixture_expert() -> TripletExpertResult:
    return TripletExpertResult(
        node_names=("A", "B", "C"),
        raw_edges=((0, 1),),
        graph=graph_from_edges(3, directed=((0, 1),)),
        edgewise_distributions={(0, 1): (1.0, 0.0, 0.0)},
        subgroup_list=((0, 1, 2),),
        subgroup_responses=("[('A','B')]",),
        expert_order=(0, 1),
        isolated_nodes=(2,),
        cycle_removal_threshold=None,
        upstream_revision=UPSTREAM_REVISION,
        config=TripletConfig(delay=0),
        runtime_seconds=1.0,
    )


class Table42CausalOrderTests(unittest.TestCase):
    def test_task_grid(self) -> None:
        self.assertEqual(slice_for_task(0), (0, 200))
        self.assertEqual(slice_for_task(39), (19, 1000))
        with self.assertRaises(ValueError):
            slice_for_task(40)

    def test_expert_artifact_round_trip(self) -> None:
        original = _fixture_expert()
        restored = _expert(_expert_record(original))
        self.assertEqual(restored, original)

    def test_only_unranked_undirected_endpoints_need_fallback(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[0, 1] = matrix[1, 0] = int(Endpoint.TAIL)
        matrix[1, 2] = matrix[2, 1] = int(Endpoint.TAIL)
        cpdag = PartialGraph.from_matrix(matrix, "cpdag")
        self.assertEqual(required_fallback_pairs(_fixture_expert(), [cpdag]), ((1, 2),))


if __name__ == "__main__":
    unittest.main()
