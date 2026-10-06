from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

import numpy as np

import cd_llm.core.finite_support as finite_support_core
from cd_llm.core.finite_support import (
    Claim,
    DataEnsemble,
    DiscoveryResult,
    DEFAULT_SOFT_CONFIDENCE_CAP,
    Endpoint,
    EvidenceGraph,
    EvidenceNode,
    Graph,
    HardClaimInfeasibleError,
    NonCompletablePAGError,
    NonExtendableCPDAGError,
    PartialGraph,
    adjacency_f1,
    build_data_ensemble,
    build_synthetic_evidence,
    claim_value_matrix,
    distribution_metrics,
    edge_marks,
    estimator_distributions,
    evaluate_claim,
    generate_latent_model,
    generate_random_dag,
    graph_from_edges,
    has_bidirected,
    has_edge,
    is_ancestral,
    is_compatible_completion,
    is_equivalent_pag_completion,
    is_maximal_ancestral,
    learn_fci,
    learn_ges,
    learn_pc,
    learn_tetrad_pag,
    map_graph,
    mag_to_pag,
    normalized_structural_distance,
    orientation_f1,
    preference_fusion,
    resolve_evidence_confidence,
    sample_completions,
    sample_completions_with_diagnostics,
    sample_resolved_claims,
    simulate_linear_sem,
    total_variation,
    transitive_closure,
    validate_soft_confidence_cap,
)


def partial(matrix: list[list[int]], kind: str) -> PartialGraph:
    return PartialGraph.from_matrix(np.asarray(matrix), kind)  # type: ignore[arg-type]


class GraphTests(unittest.TestCase):
    def test_sparse_validation_rejects_every_conflicting_edge_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "bidirected rows must be symmetric"):
            Graph((0, 0, 0), (1 << 1, 0, 0))
        with self.assertRaisesRegex(ValueError, "undirected rows must be symmetric"):
            Graph((0, 0, 0), (), (1 << 2, 0, 0))
        with self.assertRaisesRegex(ValueError, "multiple edges"):
            Graph((1 << 1, 1 << 0), (0, 0))
        with self.assertRaisesRegex(ValueError, "multiple edges"):
            Graph((1 << 1, 0), (1 << 1, 1 << 0))
        with self.assertRaisesRegex(ValueError, "directed component contains a cycle"):
            Graph((1 << 1, 1 << 2, 1 << 0))

    def test_dag_closure_and_distance(self) -> None:
        chain = graph_from_edges(3, [(0, 1), (1, 2)])
        reversed_first = graph_from_edges(3, [(1, 0), (1, 2)])
        closure = transitive_closure(chain)
        self.assertTrue(closure[0] & (1 << 2))
        self.assertAlmostEqual(normalized_structural_distance(chain, chain, "dag"), 0.0)
        self.assertAlmostEqual(
            normalized_structural_distance(reversed_first, chain, "dag"), 1.0 / 3.0
        )

    def test_mag_endpoint_semantics_and_validity(self) -> None:
        mag = graph_from_edges(3, [(0, 2)], [(0, 1)])
        self.assertEqual(edge_marks(mag, 0, 1), (Endpoint.ARROW, Endpoint.ARROW))
        self.assertTrue(has_bidirected(mag, 0, 1))
        self.assertTrue(is_ancestral(mag))
        self.assertTrue(is_maximal_ancestral(mag))
        with self.assertRaises(ValueError):
            graph_from_edges(3, [(0, 1)], [(0, 1)])

    def test_latent_generator_has_a_genuine_bidirected_edge(self) -> None:
        target, full, coefficients = generate_latent_model(10, 2.0, seed=19)
        self.assertEqual(full.node_count, 12)
        self.assertEqual(coefficients.shape, (12, 12))
        self.assertTrue(any(target.bidirected))
        self.assertTrue(is_maximal_ancestral(target))

    def test_claim_groups_are_balanced_and_regime_specific(self) -> None:
        dag, _ = generate_random_dag(10, 2.0, seed=9)
        dag_claims = sample_resolved_claims(dag, "dag", 10, seed=3)
        self.assertEqual({claim.group for claim in dag_claims}, {"I", "II", "IV"})
        for group in ("I", "II", "IV"):
            selected = [claim for claim in dag_claims if claim.group == group]
            self.assertEqual(len(selected), 10)
            self.assertEqual(sum(claim.truth for claim in selected), 5)
            for kind in {claim.kind for claim in selected}:
                self.assertEqual(
                    {claim.truth for claim in selected if claim.kind == kind}, {0, 1}
                )

        mag, _, _ = generate_latent_model(10, 2.0, seed=11)
        mag_claims = sample_resolved_claims(mag, "mag", 10, seed=4)
        self.assertEqual({claim.group for claim in mag_claims}, {"I", "II", "III", "IV"})
        self.assertEqual(sum(claim.group == "III" for claim in mag_claims), 10)
        for kind in {claim.kind for claim in mag_claims if claim.group == "III"}:
            self.assertEqual(
                {claim.truth for claim in mag_claims if claim.group == "III" and claim.kind == kind},
                {0, 1},
            )

    def test_circle_endpoint_claims_preserve_partial_information(self) -> None:
        forward = graph_from_edges(2, [(0, 1)])
        reverse = graph_from_edges(2, [(1, 0)])
        empty = graph_from_edges(2)
        into_one = Claim(
            "circle-arrow", "III", "endpoint", 1, 0, 1,
            (int(Endpoint.CIRCLE), int(Endpoint.ARROW)),
        )
        adjacent = Claim(
            "circle-circle", "III", "endpoint", 1, 0, 1,
            (int(Endpoint.CIRCLE), int(Endpoint.CIRCLE)),
        )
        self.assertEqual(evaluate_claim(forward, into_one), 1)
        self.assertEqual(evaluate_claim(reverse, into_one), 0)
        self.assertEqual(evaluate_claim(forward, adjacent), 1)
        self.assertEqual(evaluate_claim(empty, adjacent), 0)


class CompletionTests(unittest.TestCase):
    @staticmethod
    def non_extendable_cpdag() -> PartialGraph:
        # Neither orientation of 0--3 is consistent: 3->0 closes the
        # directed cycle, while 0->3 creates the unshielded collider
        # 2->3<-0 because 0 and 2 are not adjacent.
        return partial(
            [
                [0, Endpoint.TAIL, 0, Endpoint.TAIL],
                [Endpoint.ARROW, 0, Endpoint.TAIL, 0],
                [0, Endpoint.ARROW, 0, Endpoint.TAIL],
                [Endpoint.TAIL, 0, Endpoint.ARROW, 0],
            ],
            "cpdag",
        )

    @staticmethod
    def non_completable_pag() -> PartialGraph:
        # The fixed marks require 0->1->2 together with 0<->2, which violates
        # the ancestral-graph prohibition on an arrowhead into an ancestor.
        return partial(
            [
                [0, Endpoint.TAIL, Endpoint.ARROW],
                [Endpoint.ARROW, 0, Endpoint.TAIL],
                [Endpoint.ARROW, Endpoint.ARROW, 0],
            ],
            "pag",
        )

    def test_cpdag_completions_are_distinct_compatible_dags(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, 0],
                [Endpoint.TAIL, 0, Endpoint.TAIL],
                [0, Endpoint.TAIL, 0],
            ],
            "cpdag",
        )
        completions = sample_completions(cpdag, "dag", limit=10, seed=5)
        self.assertGreaterEqual(len(completions), 2)
        self.assertEqual(len(completions), len(set(completions)))
        self.assertTrue(all(is_compatible_completion(cpdag, graph) for graph in completions))

    def test_compiled_cpdag_sampler_preserves_legacy_seeded_draws(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, Endpoint.TAIL, Endpoint.TAIL],
                [Endpoint.TAIL, 0, Endpoint.TAIL, Endpoint.TAIL],
                [Endpoint.TAIL, Endpoint.TAIL, 0, Endpoint.TAIL],
                [Endpoint.TAIL, Endpoint.TAIL, Endpoint.TAIL, 0],
            ],
            "cpdag",
        )
        expected_rows = (
            (0, 1, 3, 7),
            (0, 5, 1, 7),
            (0, 13, 1, 5),
            (10, 8, 11, 0),
            (12, 13, 0, 4),
        )

        completions = sample_completions(cpdag, "dag", limit=5, seed=17)

        self.assertEqual(tuple(graph.directed for graph in completions), expected_rows)
        self.assertTrue(all(is_compatible_completion(cpdag, graph) for graph in completions))

    def test_completion_diagnostics_share_one_compilation_across_draws(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, Endpoint.TAIL, Endpoint.TAIL],
                [Endpoint.TAIL, 0, Endpoint.TAIL, Endpoint.TAIL],
                [Endpoint.TAIL, Endpoint.TAIL, 0, Endpoint.TAIL],
                [Endpoint.TAIL, Endpoint.TAIL, Endpoint.TAIL, 0],
            ],
            "cpdag",
        )
        with patch.object(
            finite_support_core,
            "_compile_cpdag",
            wraps=finite_support_core._compile_cpdag,
        ) as compile_cpdag:
            completions, diagnostics = sample_completions_with_diagnostics(
                cpdag, "dag", limit=10, seed=0
            )

        # The input is compiled once and causal-learn's fully directed anchor
        # is compiled once; the twelve randomized draws reuse the input state.
        self.assertEqual(compile_cpdag.call_count, 2)
        self.assertEqual(diagnostics.requested_limit, 10)
        self.assertEqual(diagnostics.returned_count, len(completions))
        self.assertEqual(diagnostics.anchor_source, "causal-learn")
        self.assertGreater(diagnostics.random_attempt_count, 2)
        self.assertEqual(
            diagnostics.random_attempt_count,
            diagnostics.unique_random_count
            + diagnostics.duplicate_random_count
            + diagnostics.rejected_random_count,
        )
        self.assertEqual(diagnostics.unique_random_count, len(completions) - 1)
        self.assertFalse(diagnostics.stopped_for_stagnation)
        self.assertEqual(
            completions, sample_completions(cpdag, "dag", limit=10, seed=0)
        )

    def test_completion_diagnostics_respect_limit_and_nonextendable_error(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, 0],
                [Endpoint.TAIL, 0, Endpoint.TAIL],
                [0, Endpoint.TAIL, 0],
            ],
            "cpdag",
        )
        completions, diagnostics = sample_completions_with_diagnostics(
            cpdag, "dag", limit=1, seed=5
        )
        self.assertEqual(len(completions), 1)
        self.assertEqual(diagnostics.returned_count, 1)
        self.assertEqual(diagnostics.random_attempt_count, 0)
        with self.assertRaisesRegex(
            NonExtendableCPDAGError, "no consistent DAG extension"
        ):
            sample_completions_with_diagnostics(
                self.non_extendable_cpdag(), "dag", limit=5, seed=17
            )

    def test_pag_completion_preserves_arrowheads(self) -> None:
        # Incoming arrows at both ends force a genuine bidirected middle edge.
        target = graph_from_edges(4, [(0, 1), (3, 2)], [(1, 2)])
        pag = mag_to_pag(target)
        completions = sample_completions(pag, "mag", limit=10, seed=7)
        self.assertTrue(completions)
        for graph in completions:
            self.assertTrue(is_maximal_ancestral(graph))
            self.assertTrue(has_bidirected(graph, 1, 2))
            self.assertTrue(is_compatible_completion(pag, graph))
            self.assertTrue(is_equivalent_pag_completion(pag, graph))

    def test_pag_circle_arrow_samples_both_compatible_states(self) -> None:
        # An isolated o-> is not a complete PAG. This unshielded collider
        # supplies the context that makes its arrowheads invariant.
        pag = mag_to_pag(graph_from_edges(3, [(0, 1), (2, 1)]))
        completions = sample_completions(pag, "mag", limit=10, seed=13)
        self.assertEqual(len(completions), 4)
        self.assertTrue(any(has_edge(graph, 0, 1) for graph in completions))
        self.assertTrue(any(has_bidirected(graph, 0, 1) for graph in completions))
        self.assertEqual(len(sample_completions(pag, "mag", limit=1, seed=13)), 1)

    def test_pag_chain_rejects_endpoint_compatible_new_collider(self) -> None:
        pag = partial(
            [[0, Endpoint.CIRCLE, 0],
             [Endpoint.CIRCLE, 0, Endpoint.CIRCLE],
             [0, Endpoint.CIRCLE, 0]],
            "pag",
        )
        chain = graph_from_edges(3, [(0, 1), (1, 2)])
        collider = graph_from_edges(3, [(0, 1), (2, 1)])
        self.assertEqual(mag_to_pag(chain), pag)
        self.assertTrue(is_compatible_completion(pag, collider))
        self.assertFalse(is_equivalent_pag_completion(pag, collider))
        completions = sample_completions(pag, "mag", limit=20, seed=29)
        self.assertNotIn(collider, completions)
        self.assertTrue(all(is_equivalent_pag_completion(pag, g) for g in completions))

    def test_noncanonical_endpoint_pattern_is_not_silently_completed(self) -> None:
        pag = partial([[0, Endpoint.CIRCLE], [Endpoint.ARROW, 0]], "pag")
        with self.assertRaises(NonCompletablePAGError):
            sample_completions(pag, "mag", limit=10, seed=13)

    def test_no_selection_support_retains_directed_and_bidirected_graphs(self) -> None:
        pag = partial([[0, Endpoint.CIRCLE], [Endpoint.CIRCLE, 0]], "pag")
        general = sample_completions(pag, "mag", limit=10, seed=17)
        restricted = sample_completions(
            pag, "mag", limit=10, seed=17, exclude_selection_bias=True
        )
        self.assertEqual(len(general), 4)
        self.assertEqual(len(restricted), 3)
        self.assertTrue(any(any(g.undirected) for g in general))
        self.assertTrue(all(not any(g.undirected) for g in restricted))
        self.assertIn(graph_from_edges(2, [(0, 1)]), restricted)
        self.assertIn(graph_from_edges(2, [(1, 0)]), restricted)
        self.assertIn(graph_from_edges(2, bidirected=[(0, 1)]), restricted)
        self.assertEqual(
            restricted,
            sample_completions(
                pag, "mag", limit=10, seed=17, exclude_selection_bias=True
            ),
        )
        with self.assertRaisesRegex(ValueError, "excludes undirected"):
            sample_completions(
                graph_from_edges(2, undirected=[(0, 1)]), "mag", 1, 17,
                exclude_selection_bias=True,
            )

    def test_certified_small_mags_preserve_all_observed_independences(self) -> None:
        import itertools
        import networkx as nx
        from networkx.algorithms.d_separation import is_d_separator

        def signature(graph: Graph) -> tuple[bool, ...]:
            # Replace each bidirected edge with an independent hidden root.
            # Exhaustive observed d-separation is independent of the PAG
            # round-trip certificate used by the production sampler.
            dag = nx.DiGraph()
            dag.add_nodes_from(range(graph.node_count))
            for source, row in enumerate(graph.directed):
                for target in finite_support_core.iter_bits(row):
                    dag.add_edge(source, target)
            hidden = graph.node_count
            for first in range(graph.node_count):
                for second in range(first + 1, graph.node_count):
                    if has_bidirected(graph, first, second):
                        dag.add_edges_from([(hidden, first), (hidden, second)])
                        hidden += 1
            values = []
            for first, second in itertools.combinations(range(graph.node_count), 2):
                others = [n for n in range(graph.node_count) if n not in (first, second)]
                for size in range(len(others) + 1):
                    for subset in itertools.combinations(others, size):
                        values.append(is_d_separator(dag, {first}, {second}, set(subset)))
            return tuple(values)

        targets = (
            graph_from_edges(3, [(0, 1), (1, 2)]),
            graph_from_edges(3, [(0, 1), (2, 1)]),
            graph_from_edges(4, [(0, 1), (3, 2)], [(1, 2)]),
        )
        for target in targets:
            with self.subTest(target=target):
                expected = signature(target)
                completions = sample_completions(
                    mag_to_pag(target), "mag", limit=20, seed=37,
                    exclude_selection_bias=True,
                )
                self.assertTrue(completions)
                for graph in completions:
                    self.assertEqual(signature(graph), expected)

    def test_non_extendable_cpdag_raises_typed_error(self) -> None:
        with self.assertRaisesRegex(
            NonExtendableCPDAGError, "no consistent DAG extension"
        ):
            sample_completions(
                self.non_extendable_cpdag(), "dag", limit=200, seed=17
            )

    def test_non_completable_pag_raises_typed_error(self) -> None:
        with self.assertRaisesRegex(
            NonCompletablePAGError, "no compatible MAG completion"
        ):
            sample_completions(
                self.non_completable_pag(), "mag", limit=1, seed=19
            )

    def test_data_ensemble_weights_methods_equally_inside_bootstrap(self) -> None:
        empty = graph_from_edges(2)
        edge = graph_from_edges(2, [(0, 1)])
        discovery = DiscoveryResult(
            "dag",
            ("m1", "m2"),
            ((empty, edge), (edge, edge)),
        )
        ensemble = build_data_ensemble(discovery, completion_limit=5, seed=2)
        index = {graph: i for i, graph in enumerate(ensemble.graphs)}
        self.assertAlmostEqual(ensemble.q_bootstrap[0, index[empty]], 0.5)
        self.assertAlmostEqual(ensemble.q_bootstrap[0, index[edge]], 0.5)
        self.assertAlmostEqual(ensemble.q_bootstrap[1, index[edge]], 1.0)
        self.assertTrue(np.allclose(np.sum(ensemble.q_bootstrap, axis=1), 1.0))

    def test_data_ensemble_records_and_reweights_one_completion_failure(self) -> None:
        empty = graph_from_edges(2)
        edge = graph_from_edges(2, [(0, 1)])
        discovery = DiscoveryResult(
            "dag",
            ("broken", "empty", "edge"),
            ((self.non_extendable_cpdag(), empty, edge),),
        )
        ensemble = build_data_ensemble(discovery, completion_limit=5, seed=29)
        index = {graph: position for position, graph in enumerate(ensemble.graphs)}

        self.assertEqual(ensemble.completion_counts, ((0, 1, 1),))
        self.assertEqual(ensemble.active_method_counts, (2,))
        self.assertEqual(len(ensemble.completion_failures), 1)
        failure = ensemble.completion_failures[0]
        self.assertEqual(failure.bootstrap_index, 0)
        self.assertEqual(failure.method_name, "broken")
        self.assertEqual(failure.output_kind, "cpdag")
        self.assertEqual(failure.error_type, "NonExtendableCPDAGError")
        self.assertIn("no consistent DAG extension", failure.message)
        self.assertIsNotNone(failure.endpoint_matrix)
        self.assertAlmostEqual(ensemble.q_bootstrap[0, index[empty]], 0.5)
        self.assertAlmostEqual(ensemble.q_bootstrap[0, index[edge]], 0.5)

    def test_data_ensemble_raises_when_all_completions_fail(self) -> None:
        invalid = self.non_extendable_cpdag()
        discovery = DiscoveryResult(
            "dag", ("first", "second"), ((invalid, invalid),)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "all 2 dag discovery outputs failed completion on bootstrap 0",
        ):
            build_data_ensemble(discovery, completion_limit=5, seed=31)


class PairedDiscoveryTests(unittest.TestCase):
    def test_dag_discovery_explicitly_uses_and_emits_cpc(self) -> None:
        data = np.column_stack(
            (
                np.linspace(-1.0, 1.0, 12),
                np.linspace(-1.0, 1.0, 12) ** 2,
            )
        )
        cpc_output = partial([[0, 0], [0, 0]], "cpdag")
        ges_output = partial([[0, 0], [0, 0]], "cpdag")
        notears_output = graph_from_edges(2)

        with (
            patch(
                "cd_llm.core.cpc.learn_cpc",
                return_value=cpc_output,
            ) as learn_cpc,
            patch.object(
                finite_support_core,
                "learn_pc",
                side_effect=AssertionError("legacy PC alias must not be called"),
            ),
            patch.object(finite_support_core, "learn_ges", return_value=ges_output),
            patch.object(
                finite_support_core,
                "learn_notears",
                return_value=notears_output,
            ),
        ):
            result = finite_support_core.run_paired_discovery(
                data,
                "dag",
                bootstrap_count=2,
                seed=7,
                alpha=0.02,
            )

        self.assertEqual(result.method_names, ("cpc", "ges", "notears"))
        self.assertEqual(
            result.outputs_by_bootstrap,
            (
                (cpc_output, ges_output, notears_output),
                (cpc_output, ges_output, notears_output),
            ),
        )
        self.assertEqual(learn_cpc.call_count, 2)
        for call in learn_cpc.call_args_list:
            self.assertEqual(call.kwargs["alpha"], 0.02)
            self.assertEqual(call.kwargs["variable_type"], "continuous")


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.empty = graph_from_edges(2)
        self.edge = graph_from_edges(2, [(0, 1)])
        self.graphs = (self.empty, self.edge)
        self.claims = (Claim("c", "I", "directed", 1, 0, 1),)
        self.values = claim_value_matrix(self.graphs, self.claims)

    def ensemble(self, rows: list[list[float]]) -> DataEnsemble:
        return DataEnsemble(
            "dag", self.graphs, np.asarray(rows, dtype=float), tuple((1,) for _ in rows)
        )

    def evidence_node(
        self,
        identifier: str,
        claim_index: int,
        stance: int,
        confidence: float,
        treatment: str | None = None,
        *,
        family: str | None = None,
        factor: str | None = None,
    ) -> EvidenceNode:
        return EvidenceNode(
            identifier,
            claim_index,
            stance,  # type: ignore[arg-type]
            confidence,
            identifier if family is None else family,
            identifier if factor is None else factor,
            "target",
            True,
            True,
            treatment,  # type: ignore[arg-type]
        )

    def test_explicit_treatment_resolution_and_legacy_confidence_alias(self) -> None:
        implicit = self.evidence_node("implicit", 0, 1, 0.8)
        self.assertEqual(implicit.treatment, "soft_evidence")
        self.assertEqual(implicit.reported_confidence, 0.8)
        self.assertEqual(implicit.confidence, 0.8)
        self.assertAlmostEqual(
            implicit.resolved_confidence().effective_confidence, 0.8
        )

        capped = self.evidence_node("capped", 0, 1, 1.0, "soft_evidence")
        self.assertEqual(capped.reported_confidence, 1.0)
        self.assertEqual(
            capped.resolved_confidence().effective_confidence,
            DEFAULT_SOFT_CONFIDENCE_CAP,
        )
        self.assertEqual(
            capped.resolved_confidence(0.95).effective_confidence, 0.95
        )

        legacy = EvidenceNode(
            evidence_id="legacy",
            claim_index=0,
            stance=1,
            confidence=0.7,
            provenance_family="legacy",
            factor_id="legacy",
            context_id="target",
            context_compatible=True,
            correct=True,
        )
        self.assertEqual(legacy.reported_confidence, 0.7)

    def test_ambiguous_or_invalid_hard_treatments_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires explicit treatment"):
            self.evidence_node("ambiguous", 0, 1, 1.0)
        with self.assertRaisesRegex(ValueError, "requires reported confidence exactly"):
            self.evidence_node("invalid-hard", 0, 1, 0.99, "hard_constraint")

    def test_soft_confidence_cap_is_symmetric_and_validated(self) -> None:
        self.assertAlmostEqual(
            resolve_evidence_confidence(0.0, "soft_evidence").effective_confidence,
            0.01,
        )
        self.assertAlmostEqual(
            resolve_evidence_confidence(1.0, "soft_evidence").effective_confidence,
            0.99,
        )
        self.assertEqual(validate_soft_confidence_cap(0.5), 0.5)
        for invalid in (0.49, 1.0, math.inf, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_soft_confidence_cap(invalid)

    def test_dependence_group_cannot_mix_hard_and_soft_treatments(self) -> None:
        soft = self.evidence_node(
            "soft", 0, 1, 1.0, "soft_evidence", family="shared", factor="shared"
        )
        hard = self.evidence_node(
            "hard", 0, 1, 1.0, "hard_constraint", family="shared", factor="shared"
        )
        with self.assertRaisesRegex(ValueError, "same treatment"):
            EvidenceGraph("target", (soft, hard))

    def test_soft_reported_one_caps_without_removing_finite_support(self) -> None:
        ensemble = self.ensemble([[0.5, 0.5]])
        evidence = EvidenceGraph(
            "target", (self.evidence_node("soft-one", 0, 1, 1.0, "soft_evidence"),)
        )
        updated = preference_fusion(ensemble, self.values, evidence)
        self.assertAlmostEqual(updated[1], 0.99)
        self.assertGreater(updated[0], 0.0)

    def test_hard_evidence_conditions_support_and_empty_support_raises(self) -> None:
        evidence = EvidenceGraph(
            "target", (self.evidence_node("hard", 0, 1, 1.0, "hard_constraint"),)
        )
        updated = preference_fusion(
            self.ensemble([[0.5, 0.5]]), self.values, evidence
        )
        self.assertTrue(np.array_equal(updated, np.asarray([0.0, 1.0])))

        unsupported = DataEnsemble("dag", (self.empty,), np.asarray([[1.0]]), ((1,),))
        unsupported_values = claim_value_matrix((self.empty,), self.claims)
        with self.assertRaises(HardClaimInfeasibleError):
            preference_fusion(unsupported, unsupported_values, evidence)

    def test_mixed_hard_and_soft_evidence_uses_constrained_soft_target(self) -> None:
        first = graph_from_edges(3, [(0, 1)])
        second = graph_from_edges(3, [(0, 1), (1, 2)])
        claims = (
            Claim("hard", "I", "directed", 1, 0, 1),
            Claim("soft", "I", "directed", 1, 1, 2),
        )
        values = claim_value_matrix((first, second), claims)
        ensemble = DataEnsemble(
            "dag", (first, second), np.asarray([[0.5, 0.5]]), ((1,),)
        )
        evidence = EvidenceGraph(
            "target",
            (
                self.evidence_node("hard", 0, 1, 1.0, "hard_constraint"),
                self.evidence_node("soft", 1, 1, 0.8, "soft_evidence"),
            ),
        )
        updated = preference_fusion(ensemble, values, evidence)
        self.assertTrue(np.allclose(updated, np.asarray([0.2, 0.8])))

    def test_confidence_is_exact_relative_likelihood(self) -> None:
        ensemble = self.ensemble([[0.5, 0.5]])
        evidence = build_synthetic_evidence(self.claims, ("accurate",), 0.8, seed=1)
        updated = preference_fusion(ensemble, self.values, evidence)
        self.assertAlmostEqual(updated[1], 0.8)
        self.assertAlmostEqual(updated[1] / updated[0], 4.0)

    def test_fusion_normalizes_before_bootstrap_average(self) -> None:
        ensemble = self.ensemble([[0.9, 0.1], [0.1, 0.9]])
        evidence = build_synthetic_evidence(self.claims, ("accurate",), 0.8, seed=2)
        updated = preference_fusion(ensemble, self.values, evidence)
        aggregate_then_fuse = 0.8
        self.assertNotAlmostEqual(float(updated[1]), aggregate_then_fuse)
        first = (0.1 * 2.0) / (0.9 * 0.5 + 0.1 * 2.0)
        second = (0.9 * 2.0) / (0.1 * 0.5 + 0.9 * 2.0)
        self.assertAlmostEqual(float(updated[1]), 0.5 * (first + second))

    def test_empty_duplicate_and_context_invariances(self) -> None:
        ensemble = self.ensemble([[0.4, 0.6], [0.7, 0.3]])
        empty_evidence = EvidenceGraph("target", ())
        identity = preference_fusion(ensemble, self.values, empty_evidence)
        self.assertTrue(np.array_equal(identity, ensemble.q_data))

        one = build_synthetic_evidence(self.claims, ("accurate_one",), 0.9, seed=1)
        repeated = build_synthetic_evidence(self.claims, ("repeated_six",), 0.9, seed=1)
        one_q = preference_fusion(ensemble, self.values, one)
        repeated_q = preference_fusion(ensemble, self.values, repeated)
        self.assertTrue(np.allclose(one_q, repeated_q))

        incompatible = build_synthetic_evidence(
            self.claims, ("context_incompatible",), 0.9, seed=1
        )
        context_q = preference_fusion(ensemble, self.values, incompatible)
        self.assertTrue(np.array_equal(context_q, ensemble.q_data))

    def test_independent_sources_accumulate_and_confidence_controls_strength(self) -> None:
        ensemble = self.ensemble([[0.5, 0.5]])
        one = build_synthetic_evidence(self.claims, ("accurate_one",), 0.9, seed=3)
        six = build_synthetic_evidence(self.claims, ("independent_six",), 0.9, seed=3)
        self.assertGreater(
            preference_fusion(ensemble, self.values, six)[1],
            preference_fusion(ensemble, self.values, one)[1],
        )
        shifts = []
        for confidence in (0.9, 0.75, 0.55):
            evidence = build_synthetic_evidence(
                self.claims, ("accurate",), confidence, seed=4
            )
            updated = preference_fusion(ensemble, self.values, evidence)
            shifts.append(total_variation(updated, ensemble.q_data))
        self.assertGreater(shifts[0], shifts[1])
        self.assertGreater(shifts[1], shifts[2])

    def test_all_estimators_preserve_no_evidence_identity(self) -> None:
        ensemble = self.ensemble([[0.25, 0.75]])
        estimates = estimator_distributions(
            ensemble, self.values, EvidenceGraph("target", ()), random_seed=1
        )
        for distribution in estimates.values():
            self.assertIsNotNone(distribution)
            self.assertTrue(np.array_equal(distribution, ensemble.q_data))

    def test_mixed_evidence_seeds_change_balanced_assignments(self) -> None:
        graph, _ = generate_random_dag(10, 2.0, seed=17)
        claims = sample_resolved_claims(graph, "dag", 10, seed=18)
        first = build_synthetic_evidence(
            claims,
            ("accurate", "inaccurate", "conflict", "repeated", "context_incompatible"),
            None,
            seed=1,
        )
        second = build_synthetic_evidence(
            claims,
            ("accurate", "inaccurate", "conflict", "repeated", "context_incompatible"),
            None,
            seed=2,
        )
        signature = lambda evidence: sorted(
            (
                node.claim_index,
                node.stance,
                node.confidence,
                node.context_compatible,
                node.provenance_family,
            )
            for node in evidence.nodes
        )
        self.assertNotEqual(signature(first), signature(second))
        self.assertEqual(len(first.nodes), len(second.nodes))
        for evidence in (first, second):
            representatives = evidence.representatives()
            self.assertEqual(
                sum(node.correct for node in representatives), len(representatives) // 2
            )
            self.assertEqual(
                sum(node.stance > 0 for node in representatives), len(representatives) // 2
            )
            self.assertEqual(sum(node.correct for node in evidence.nodes), len(evidence.nodes) // 2)
            self.assertEqual(sum(node.stance > 0 for node in evidence.nodes), len(evidence.nodes) // 2)


class MetricTests(unittest.TestCase):
    def test_map_graph_uses_canonical_tie_break(self) -> None:
        first = graph_from_edges(3, [(1, 2)])
        second = graph_from_edges(3, [(0, 2)])
        expected = min(first, second)
        self.assertEqual(map_graph((first, second), np.asarray([0.5, 0.5])), expected)
        self.assertEqual(map_graph((second, first), np.asarray([0.5, 0.5])), expected)

    def test_orientation_f1_is_zero_for_disjoint_nonempty_edges(self) -> None:
        truth = graph_from_edges(3, [(0, 1)])
        estimated = graph_from_edges(3, [(1, 2)])
        self.assertEqual(orientation_f1(estimated, truth), 0.0)

    def test_energy_score_uses_full_distribution(self) -> None:
        truth = graph_from_edges(2, [(0, 1)])
        wrong = graph_from_edges(2)
        metrics = distribution_metrics(
            (wrong, truth), np.asarray([0.5, 0.5]), truth, "dag"
        )
        self.assertAlmostEqual(metrics["expected_distance"], 0.5)
        self.assertAlmostEqual(metrics["energy_score"], 0.25)
        self.assertAlmostEqual(metrics["neighborhood_mass"], 0.5)
        self.assertAlmostEqual(adjacency_f1(truth, truth), 1.0)

    def test_mag_endpoint_distance_counts_marks(self) -> None:
        directed = graph_from_edges(2, [(0, 1)])
        confounded = graph_from_edges(2, bidirected=[(0, 1)])
        self.assertAlmostEqual(
            normalized_structural_distance(directed, confounded, "mag"), 0.5
        )


@unittest.skipUnless(os.environ.get("RUN_DISCOVERY_SMOKE") == "1", "backend smoke is opt-in")
class BackendSmokeTests(unittest.TestCase):
    def test_causal_learn_backends(self) -> None:
        graph = graph_from_edges(3, [(0, 1), (1, 2)])
        coefficients = np.zeros((3, 3))
        coefficients[0, 1] = 1.0
        coefficients[1, 2] = 1.0
        data = simulate_linear_sem(graph, coefficients, 200, seed=3)
        self.assertEqual(learn_pc(data).kind, "cpdag")
        self.assertEqual(learn_ges(data).kind, "cpdag")
        self.assertEqual(learn_fci(data).kind, "pag")


@unittest.skipUnless(os.environ.get("RUN_TETRAD_SMOKE") == "1", "Tetrad smoke is opt-in")
class TetradBackendSmokeTests(unittest.TestCase):
    def test_tetrad_backends(self) -> None:
        graph = graph_from_edges(3, [(0, 1), (1, 2)])
        coefficients = np.zeros((3, 3))
        coefficients[0, 1] = 1.0
        coefficients[1, 2] = 1.0
        data = simulate_linear_sem(graph, coefficients, 120, seed=7)
        self.assertEqual(learn_tetrad_pag(data, "rfci", 0.05).kind, "pag")
        self.assertEqual(learn_tetrad_pag(data, "gfci", 0.05).kind, "pag")
        pag = mag_to_pag(graph_from_edges(3, [(0, 1), (2, 1)]))
        completions = sample_completions(pag, "mag", limit=5, seed=11)
        self.assertTrue(completions)
        self.assertTrue(
            all(
                is_compatible_completion(pag, completion)
                and is_maximal_ancestral(completion)
                and is_equivalent_pag_completion(pag, completion)
                for completion in completions
            )
        )


if __name__ == "__main__":
    unittest.main()
