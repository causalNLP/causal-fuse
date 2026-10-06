from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import cd_llm.baselines.causal_order2025.api as api_module
import cd_llm.baselines.causal_order2025.upstream_backend as upstream_module
from cd_llm.baselines.camml2006 import CaMMLConfig
from cd_llm.baselines.causal_order2025 import (
    CausalOrderInfeasibleError,
    PCOrderConfig,
    TripletConfig,
    TripletExpertResult,
    UPSTREAM_REVISION,
    causal_learn_cpdag,
    fit_camml_with_causal_order,
    fit_pc_with_causal_order,
    integrate_order_into_cpdag,
    level_order_from_edges,
    level_order_prior,
    orient_with_upstream_expert,
    query_triplet_order,
    verified_upstream_root,
)
from cd_llm.core.finite_support import (
    Endpoint,
    PartialGraph,
    graph_from_edges,
    has_edge,
)
from cd_llm.paths import CODE_ROOT


def _expert(
    *,
    raw_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2)),
    graph_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2)),
    order: tuple[int, ...] = (0, 1, 2),
) -> TripletExpertResult:
    connected = {node for edge in graph_edges for node in edge}
    return TripletExpertResult(
        node_names=("A", "B", "C"),
        raw_edges=raw_edges,
        graph=graph_from_edges(3, graph_edges),
        edgewise_distributions={},
        subgroup_list=((0, 1, 2),),
        subgroup_responses=("<Answer>[]</Answer>",),
        expert_order=order,
        isolated_nodes=tuple(node for node in range(3) if node not in connected),
        cycle_removal_threshold=None,
        upstream_revision=UPSTREAM_REVISION,
        config=TripletConfig(delay=0),
        runtime_seconds=0.01,
    )


def _partial(matrix: np.ndarray) -> PartialGraph:
    return PartialGraph.from_matrix(matrix, "cpdag")


class AlgorithmOneTests(unittest.TestCase):
    def test_orients_undirected_edges_symmetrically_by_expert_order(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[0, 1] = matrix[1, 0] = int(Endpoint.TAIL)
        matrix[1, 2] = matrix[2, 1] = int(Endpoint.TAIL)
        result = integrate_order_into_cpdag(_partial(matrix), _expert())
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertTrue(has_edge(result.graph, 1, 2))
        self.assertEqual(result.fallback_orientations, ())

        reverse = _expert(
            raw_edges=((2, 1), (1, 0)),
            graph_edges=((2, 1), (1, 0)),
            order=(2, 1, 0),
        )
        reversed_result = integrate_order_into_cpdag(_partial(matrix), reverse)
        self.assertTrue(has_edge(reversed_result.graph, 1, 0))
        self.assertTrue(has_edge(reversed_result.graph, 2, 1))

    def test_preserves_directed_edges_and_uses_fallback_for_isolates(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[0, 1] = int(Endpoint.TAIL)
        matrix[1, 0] = int(Endpoint.ARROW)
        matrix[1, 2] = matrix[2, 1] = int(Endpoint.TAIL)
        expert = _expert(
            raw_edges=((0, 1),),
            graph_edges=((0, 1),),
            order=(0, 1),
        )
        result = integrate_order_into_cpdag(
            _partial(matrix), expert, fallback_orienter=lambda first, second: (second, first)
        )
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertTrue(has_edge(result.graph, 2, 1))
        self.assertEqual(result.fallback_orientations, ((2, 1),))

    def test_cycle_guard_uses_expert_direction(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[0, 1] = int(Endpoint.TAIL)
        matrix[1, 0] = int(Endpoint.ARROW)
        matrix[1, 2] = int(Endpoint.TAIL)
        matrix[2, 1] = int(Endpoint.ARROW)
        matrix[0, 2] = matrix[2, 0] = int(Endpoint.TAIL)
        expert = _expert(
            raw_edges=((2, 0),),
            graph_edges=((2, 0),),
            order=(2, 0),
        )
        result = integrate_order_into_cpdag(
            _partial(matrix), expert, fallback_orienter=lambda first, second: (first, second)
        )
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertTrue(has_edge(result.graph, 1, 2))
        self.assertTrue(has_edge(result.graph, 0, 2))
        self.assertEqual(result.expert_fallback_pairs, ((0, 2),))
        self.assertEqual(result.cycle_guard_pairs, ((0, 2),))
        self.assertEqual(result.fallback_orientations, ((0, 2),))
        self.assertEqual(result.forced_safe_orientations, ())

    def test_cycle_guard_forces_unique_safe_direction_for_unsafe_expert(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[0, 1] = int(Endpoint.TAIL)
        matrix[1, 0] = int(Endpoint.ARROW)
        matrix[1, 2] = int(Endpoint.TAIL)
        matrix[2, 1] = int(Endpoint.ARROW)
        matrix[0, 2] = matrix[2, 0] = int(Endpoint.TAIL)
        expert = _expert(raw_edges=((2, 0),), graph_edges=((2, 0),), order=(2, 0))
        for answer in ((2, 0), None):
            with self.subTest(answer=answer):
                result = integrate_order_into_cpdag(
                    _partial(matrix), expert,
                    fallback_orienter=lambda first, second, answer=answer: answer,
                )
                self.assertTrue(has_edge(result.graph, 0, 2))
                self.assertEqual(result.forced_safe_orientations, ((0, 2),))

    def test_cycle_guard_requires_expert_when_order_direction_is_unsafe(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[0, 1] = int(Endpoint.TAIL)
        matrix[1, 0] = int(Endpoint.ARROW)
        matrix[1, 2] = int(Endpoint.TAIL)
        matrix[2, 1] = int(Endpoint.ARROW)
        matrix[0, 2] = matrix[2, 0] = int(Endpoint.TAIL)
        expert = _expert(raw_edges=((2, 0),), graph_edges=((2, 0),), order=(2, 0))
        with self.assertRaisesRegex(CausalOrderInfeasibleError, "no superior-expert"):
            integrate_order_into_cpdag(_partial(matrix), expert)

    def test_missing_order_and_unusable_expert_is_infeasible_when_both_directions_safe(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[1, 2] = matrix[2, 1] = int(Endpoint.TAIL)
        expert = _expert(raw_edges=((0, 1),), graph_edges=((0, 1),), order=(0, 1))
        with self.assertRaisesRegex(CausalOrderInfeasibleError, "no unique acyclic direction"):
            integrate_order_into_cpdag(
                _partial(matrix), expert, fallback_orienter=lambda first, second: None
            )

    def test_requires_a_direction_from_fallback(self) -> None:
        matrix = np.zeros((3, 3), dtype=int)
        matrix[1, 2] = matrix[2, 1] = int(Endpoint.TAIL)
        expert = _expert(
            raw_edges=((0, 1),),
            graph_edges=((0, 1),),
            order=(0, 1),
        )
        with self.assertRaisesRegex(CausalOrderInfeasibleError, "no unique acyclic direction"):
            integrate_order_into_cpdag(
                _partial(matrix), expert, fallback_orienter=lambda first, second: (0, 1)
            )

    def test_causal_learn_endpoint_conversion(self) -> None:
        raw = np.asarray([[0, -1], [1, 0]])
        converted = causal_learn_cpdag(SimpleNamespace(graph=raw))
        self.assertEqual(converted.mark(0, 1), Endpoint.TAIL)
        self.assertEqual(converted.mark(1, 0), Endpoint.ARROW)

    def test_pc_api_returns_point_mass_and_forwards_config(self) -> None:
        raw = np.asarray(
            [
                [0, -1, 0],
                [-1, 0, -1],
                [0, -1, 0],
            ]
        )
        runner = Mock(return_value=SimpleNamespace(G=SimpleNamespace(graph=raw)))
        data = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]])
        config = PCOrderConfig(alpha=0.01, indep_test="fisherz")
        result = fit_pc_with_causal_order(
            data, ("A", "B", "C"), _expert(), config=config, runner=runner
        )
        self.assertEqual(dict(result.support), {result.graph: 1.0})
        self.assertEqual(result.final_order, (0, 1, 2))
        self.assertEqual(runner.call_args.kwargs["alpha"], 0.01)
        self.assertEqual(runner.call_args.kwargs["node_names"], ["A", "B", "C"])

    def test_pc_api_projects_cyclic_endpoint_preferences(self) -> None:
        raw = np.asarray(
            [
                [0, -1, 1],
                [1, 0, -1],
                [-1, 1, 0],
            ]
        )
        runner = Mock(return_value=SimpleNamespace(G=SimpleNamespace(graph=raw)))
        result = fit_pc_with_causal_order(
            np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
            ("A", "B", "C"),
            _expert(),
            runner=runner,
        )
        self.assertEqual(result.graph.regime, "dag")
        self.assertEqual(
            sum(result.cpdag.mark(i, j) != Endpoint.NULL for i in range(3) for j in range(i + 1, 3)),
            3,
        )


class AlgorithmTwoTests(unittest.TestCase):
    def test_sccs_share_a_level_and_isolated_nodes_are_omitted(self) -> None:
        levels = level_order_from_edges(4, ((0, 1), (1, 0), (1, 2)))
        self.assertEqual(levels, {0: 0, 1: 0, 2: 1})
        self.assertNotIn(3, levels)

    def test_level_prior_constrains_order_not_edge_existence(self) -> None:
        prior = level_order_prior(
            3,
            {0: 0, 1: 0, 2: 1},
            prior_probability=0.9,
            arc_probability=0.5,
        )
        self.assertEqual([(value.left, value.right) for value in prior.local], [(0, 2), (1, 2)])
        for value in prior.local:
            np.testing.assert_allclose(value.probabilities, (0.45, 0.05, 0.45, 0.05))

    def test_camml_api_uses_default_point_nine_prior(self) -> None:
        data = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        config = CaMMLConfig(
            seed=3,
            chain_count=1,
            iterations=8,
            burn_in=2,
            thin=1,
            max_parents=2,
            global_prior_samples=10,
            initialization_attempts=100,
        )
        result = fit_camml_with_causal_order(
            data,
            ("A", "B", "C"),
            _expert(),
            data_score="bdeu",
            config=config,
        )
        self.assertEqual(result.prior_probability, 0.9)
        self.assertEqual(result.graph.node_count, 3)
        self.assertEqual(set(result.final_order), {0, 1, 2})
        self.assertEqual(result.prior, result.camml.prior)


class UpstreamAdapterTests(unittest.TestCase):
    def test_invalid_cutoff_is_corrected_only_when_requested(self) -> None:
        import networkx as nx
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        weights = {edges[0]: .7497329206785424, edges[1]: 1.6514549991527705,
                   edges[2]: .7441093767989729}
        payload = dict(predicted_edges=edges, edgewise_dist={edge: [.5, .3, .2] for edge in edges},
                       subgroup_list=[("A", "B", "C")], subgraph_results=["[]"])
        modules = SimpleNamespace(triplet=SimpleNamespace(run_triplet_experiment=Mock(return_value=payload)),
            cycle_remover=SimpleNamespace(remove_cycles=Mock(return_value=(edges, .7441070886119618, weights))),
            metrics=SimpleNamespace(topological_ordering=Mock(return_value=["A", "B", "C"])))
        with patch.object(upstream_module, "verified_upstream_root", return_value=Path("/tmp/upstream")), patch.object(
                upstream_module, "_load_upstream_modules", return_value=modules):
            with self.assertRaisesRegex(upstream_module.UpstreamCausalOrderError, "not a DAG"):
                query_triplet_order(("A", "B", "C"), "fixture")
            diagnostic = {}
            result = query_triplet_order(("A", "B", "C"), "fixture", cycle_removal_policy="validated_threshold",
                                         cycle_removal_diagnostics=diagnostic)
        self.assertEqual(result.raw_edges, ((0, 1), (1, 2), (2, 0)))
        self.assertEqual(diagnostic["removed_edges"], [("C", "A")])
        self.assertEqual(diagnostic["original_threshold"], .7441070886119618)
        self.assertTrue(diagnostic["acyclic"])
        self.assertEqual(result.cycle_removal_threshold, weights[edges[0]])
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertFalse(has_edge(result.graph, 2, 0))
        # Once upstream returns a valid output, the opt-in policy leaves it alone.
        modules.cycle_remover.remove_cycles.return_value = (edges[:2], .74411, weights)
        with patch.object(upstream_module, "verified_upstream_root", return_value=Path("/tmp/upstream")), patch.object(
                upstream_module, "_load_upstream_modules", return_value=modules):
            diagnostic = {}
            valid = query_triplet_order(("A", "B", "C"), "fixture", cycle_removal_policy="validated_threshold",
                                        cycle_removal_diagnostics=diagnostic)
        self.assertEqual(valid.cycle_removal_threshold, .74411)
        self.assertEqual(valid.graph, result.graph)
        self.assertFalse(diagnostic["correction_applied"])

    def test_discrete_cutoffs_handle_ties_empty_and_acyclic_graphs(self) -> None:
        import networkx as nx
        fn = upstream_module.validated_cycle_threshold
        equal_cycle = {("A", "B"): 1., ("B", "C"): 1., ("C", "A"): 1.}
        edges, threshold = fn(equal_cycle)
        self.assertEqual(edges, ())
        self.assertGreater(threshold, 1.)
        self.assertEqual(fn({}), ((), 0.))
        dag = {("A", "B"): .5, ("B", "C"): 1.}
        self.assertEqual(fn(dag), (tuple(dag), .5))
        weights = dict(equal_cycle, **{})
        weights[("A", "D")] = 2.
        edges, threshold = fn(weights)
        self.assertEqual(edges, (("A", "D"),))
        self.assertEqual(threshold, 2.)
        self.assertTrue(nx.is_directed_acyclic_graph(nx.DiGraph(edges)))
        self.assertEqual(fn(weights), (edges, threshold))

    def test_discrete_cutoff_rejects_malformed_weights(self) -> None:
        fn = upstream_module.validated_cycle_threshold
        for value in (float("nan"), float("inf"), -1., 0., True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fn({("A", "B"): value})
        with self.assertRaises(ValueError):
            fn({("A", "A"): 1.})

    def test_translates_upstream_payload_and_uses_upstream_cycle_remover(self) -> None:
        triplet = SimpleNamespace()
        triplet.run_triplet_experiment = Mock(
            return_value={
                "predicted_edges": [("A", "B"), ("B", "A"), ("B", "C")],
                "edgewise_dist": {
                    ("A", "B"): [0.4, 0.4, 0.2],
                    ("A", "C"): [0.0, 0.0, 1.0],
                    ("B", "C"): [0.8, 0.1, 0.1],
                },
                "subgroup_list": [("A", "B", "C")],
                "subgraph_results": ["<Answer>[('A','B')]</Answer>"],
            }
        )
        cycle_remover = SimpleNamespace()
        cycle_remover.remove_cycles = Mock(
            return_value=([("A", "B"), ("B", "C")], 0.25, {})
        )
        metrics = SimpleNamespace(topological_ordering=Mock(return_value=["A", "B", "C"]))
        modules = SimpleNamespace(
            triplet=triplet,
            cycle_remover=cycle_remover,
            metrics=metrics,
        )
        with patch.object(
            upstream_module, "verified_upstream_root", return_value=Path("/tmp/upstream")
        ), patch.object(upstream_module, "_load_upstream_modules", return_value=modules):
            result = query_triplet_order(
                ("A", "B", "C"), "fixture", config=TripletConfig(delay=0)
            )
        self.assertEqual(result.raw_edges, ((0, 1), (1, 0), (1, 2)))
        self.assertEqual(result.expert_order, (0, 1, 2))
        self.assertEqual(result.cycle_removal_threshold, 0.25)
        self.assertTrue(has_edge(result.graph, 0, 1))
        self.assertTrue(has_edge(result.graph, 1, 2))
        cycle_remover.remove_cycles.assert_called_once()

    def test_no_relation_fallback_is_reported_as_infeasible(self) -> None:
        modules = SimpleNamespace(
            triplet=SimpleNamespace(_tiebreaker_query=Mock(return_value="No_Conn"))
        )
        with patch.object(
            upstream_module, "verified_upstream_root", return_value=Path("/tmp/upstream")
        ), patch.object(upstream_module, "_load_upstream_modules", return_value=modules):
            with self.assertRaisesRegex(CausalOrderInfeasibleError, "no relation"):
                orient_with_upstream_expert(
                    ("A", "B", "C"), 0, 1, "fixture", config=TripletConfig(delay=0)
                )

    def test_local_pinned_checkout_is_clean_when_present(self) -> None:
        checkout = CODE_ROOT / "vendor" / "Causal_Order_Imperfect_Experts"
        if not checkout.is_dir():
            self.skipTest("ignored upstream checkout is not installed")
        self.assertEqual(verified_upstream_root(checkout), checkout.resolve())


class ValidationTests(unittest.TestCase):
    def test_duplicate_names_and_mismatched_cached_result_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            query_triplet_order(("A", "A", "C"), "fixture")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            fit_pc_with_causal_order(
                np.ones((2, 3)),
                ("A", "C", "B"),
                _expert(),
                runner=Mock(),
            )

    def test_configs_reject_ambiguous_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly"):
            PCOrderConfig(alpha=1.0)
        with self.assertRaises(TypeError):
            TripletConfig(delay=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            level_order_prior(
                2,
                {0: 0, 1: 1},
                prior_probability=True,  # type: ignore[arg-type]
                arc_probability=0.5,
            )
        with self.assertRaisesRegex(ValueError, r"\[0.5, 1\]"):
            level_order_prior(
                2,
                {0: 0, 1: 1},
                prior_probability=0.49,
                arc_probability=0.5,
            )


if __name__ == "__main__":
    unittest.main()
