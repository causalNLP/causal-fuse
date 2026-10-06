from __future__ import annotations

import itertools
import math
import unittest

import numpy as np
from causallearn.score.LocalScoreFunction import local_score_BIC

from cd_llm.core.finite_support import (
    Claim,
    Endpoint,
    Graph,
    NonExtendableCPDAGError,
    PartialGraph,
    graph_from_edges,
    is_compatible_completion,
    structural_hamming_distance,
)
from cd_llm.methods.adaptive_pruning import (
    GaussianBICScore,
    adaptive_support_ceg_fuse,
    aggregate_bootstraps,
    dag_one_pair_neighbors,
    defensive_evidence_update,
    defensive_mixture,
    enumerate_cpdag_extensions,
    hierarchical_dag_ensemble,
    markov_equivalent_dag,
    prune_by_mass,
    reweight_distribution,
    sample_score_dags_mcmc,
    score_distribution,
    temper_evidence_smc,
)


def partial(matrix: list[list[int]]) -> PartialGraph:
    return PartialGraph.from_matrix(np.asarray(matrix), "cpdag")


def all_dags(node_count: int) -> tuple[Graph, ...]:
    pairs = tuple(itertools.combinations(range(node_count), 2))
    result: list[Graph] = []
    for states in itertools.product((0, 1, 2), repeat=len(pairs)):
        edges = []
        for (first, second), state in zip(pairs, states):
            if state == 1:
                edges.append((first, second))
            elif state == 2:
                edges.append((second, first))
        try:
            result.append(graph_from_edges(node_count, edges))
        except ValueError:
            pass
    return tuple(result)


class CompletionEnumerationTests(unittest.TestCase):
    def test_chain_has_all_three_consistent_extensions(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, 0],
                [Endpoint.TAIL, 0, Endpoint.TAIL],
                [0, Endpoint.TAIL, 0],
            ]
        )
        extensions = enumerate_cpdag_extensions(cpdag)
        self.assertEqual(len(extensions), 3)
        self.assertTrue(all(is_compatible_completion(cpdag, graph) for graph in extensions))
        collider = graph_from_edges(3, [(0, 1), (2, 1)])
        self.assertNotIn(collider, extensions)

    def test_triangle_has_six_extensions(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, Endpoint.TAIL],
                [Endpoint.TAIL, 0, Endpoint.TAIL],
                [Endpoint.TAIL, Endpoint.TAIL, 0],
            ]
        )
        self.assertEqual(len(enumerate_cpdag_extensions(cpdag)), 6)

    def test_fixed_arrow_is_preserved_without_new_collider(self) -> None:
        cpdag = partial(
            [
                [0, Endpoint.TAIL, 0],
                [Endpoint.ARROW, 0, Endpoint.TAIL],
                [0, Endpoint.TAIL, 0],
            ]
        )
        self.assertEqual(
            enumerate_cpdag_extensions(cpdag),
            (graph_from_edges(3, [(0, 1), (1, 2)]),),
        )

    def test_nonextendable_cpdag_fails_promptly(self) -> None:
        cpdag = partial(
            [
                [0, 0, 0, Endpoint.TAIL],
                [0, 0, 0, 0],
                [0, 0, 0, Endpoint.TAIL],
                [Endpoint.ARROW, 0, Endpoint.ARROW, 0],
            ]
        )
        # Add 0->1->2->3 and the problematic undirected 0--3 relation.
        cpdag = partial(
            [
                [0, Endpoint.TAIL, 0, Endpoint.TAIL],
                [Endpoint.ARROW, 0, Endpoint.TAIL, 0],
                [0, Endpoint.ARROW, 0, Endpoint.TAIL],
                [Endpoint.TAIL, 0, Endpoint.ARROW, 0],
            ]
        )
        with self.assertRaises(NonExtendableCPDAGError):
            enumerate_cpdag_extensions(cpdag)


class ScoreAndSupportTests(unittest.TestCase):
    def test_one_pair_neighborhood_is_exhaustive_and_acyclic(self) -> None:
        graph = graph_from_edges(3, [(0, 1), (1, 2)])
        neighbors = dag_one_pair_neighbors(graph)
        self.assertEqual(len(neighbors), len(set(neighbors)))
        self.assertTrue(neighbors)
        self.assertTrue(
            all(structural_hamming_distance(graph, neighbor) == 1 for neighbor in neighbors)
        )

    def test_cached_bic_matches_causal_learn_local_score(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(80, 4))
        score = GaussianBICScore(data)
        standardized = score.data
        for node, parents in ((0, []), (1, [0]), (2, [0, 1])):
            mask = sum(1 << parent for parent in parents)
            expected = local_score_BIC(
                standardized, node, parents, {"lambda_value": 0.5}
            )
            self.assertAlmostEqual(score.local_score(node, mask), expected, places=8)

    def test_hierarchical_family_and_variant_weights(self) -> None:
        empty = graph_from_edges(2)
        forward = graph_from_edges(2, [(0, 1)])
        reverse = graph_from_edges(2, [(1, 0)])
        rows, _ = hierarchical_dag_ensemble(
            [{"pc": [empty], "ges": [forward, reverse]}]
        )
        self.assertAlmostEqual(rows[0][empty], 0.5)
        self.assertAlmostEqual(rows[0][forward], 0.25)
        self.assertAlmostEqual(rows[0][reverse], 0.25)

    def test_defensive_update_uses_component_normalizers(self) -> None:
        empty = graph_from_edges(2)
        edge = graph_from_edges(2, [(0, 1)])
        core = {empty: 1.0}
        exploration = {edge: 1.0}
        preference = lambda graph: math.log(4.0) if graph == edge else 0.0
        data = defensive_mixture(core, exploration, 0.1)
        evidence, diagnostic = defensive_evidence_update(
            core, exploration, preference, 0.1
        )
        self.assertAlmostEqual(data[edge], 0.1)
        self.assertAlmostEqual(evidence[edge], 0.4 / 1.3)
        self.assertAlmostEqual(
            diagnostic["posterior_exploration_share"], 0.4 / 1.3
        )

    def test_markov_equivalence_uses_skeleton_and_v_structures(self) -> None:
        chain = graph_from_edges(3, [(0, 1), (1, 2)])
        fork = graph_from_edges(3, [(1, 0), (1, 2)])
        collider = graph_from_edges(3, [(0, 1), (2, 1)])
        self.assertTrue(markov_equivalent_dag(chain, fork))
        self.assertFalse(markov_equivalent_dag(chain, collider))

    def test_mass_pruning_has_declared_cumulative_mass_and_tv_bound(self) -> None:
        graphs = (
            graph_from_edges(3),
            graph_from_edges(3, [(0, 1)]),
            graph_from_edges(3, [(1, 2)]),
            graph_from_edges(3, [(2, 0)]),
        )
        original = dict(zip(graphs, (0.50, 0.30, 0.12, 0.08)))
        result = prune_by_mass(original, 0.20)

        self.assertEqual(set(result.removed_graphs), {graphs[2], graphs[3]})
        self.assertEqual(result.original_support_size, 4)
        self.assertEqual(result.retained_support_size, 2)
        self.assertAlmostEqual(result.removed_mass, 0.20)
        self.assertAlmostEqual(result.retained_mass, 0.80)
        self.assertAlmostEqual(result.total_variation, result.removed_mass)
        self.assertLessEqual(result.total_variation, result.mass_limit + 1e-15)
        self.assertAlmostEqual(sum(result.distribution.values()), 1.0)
        self.assertAlmostEqual(result.distribution[graphs[0]], 0.50 / 0.80)

    def test_mass_pruning_rejects_invalid_bounds_and_probabilities(self) -> None:
        graph = graph_from_edges(2)
        with self.assertRaises(ValueError):
            prune_by_mass({graph: 1.0}, 1.0)
        with self.assertRaises(ValueError):
            prune_by_mass({graph: -1.0}, 0.1)


class SamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(19)
        self.data = rng.normal(size=(25, 3))
        self.score = GaussianBICScore(self.data)
        self.graphs = all_dags(3)
        self.exact_data = score_distribution(self.graphs, self.score)

    def test_mcmc_matches_exact_three_node_score_distribution(self) -> None:
        initial = tuple(sorted(self.exact_data, key=self.exact_data.get, reverse=True)[:4])
        result = sample_score_dags_mcmc(
            self.score,
            initial,
            seed=23,
            chain_count=4,
            iterations=8_000,
            burn_in=2_000,
            thin=3,
        )
        union = set(self.exact_data) | set(result.distribution)
        tv = 0.5 * sum(
            abs(self.exact_data.get(graph, 0.0) - result.distribution.get(graph, 0.0))
            for graph in union
        )
        self.assertLess(tv, 0.10)
        self.assertGreater(result.acceptance_rate, 0.0)
        self.assertLess(result.score_rhat, 1.10)

    def test_tempered_smc_matches_exact_evidence_update(self) -> None:
        initial = tuple(sorted(self.exact_data, key=self.exact_data.get, reverse=True)[:4])
        mcmc = sample_score_dags_mcmc(
            self.score,
            initial,
            seed=29,
            chain_count=4,
            iterations=8_000,
            burn_in=2_000,
            thin=3,
        )
        preference = lambda graph: math.log(3.0) if graph.directed[0] & (1 << 1) else 0.0
        exact, exact_log_z = reweight_distribution(self.exact_data, preference)
        smc = temper_evidence_smc(
            self.score,
            mcmc.samples,
            preference,
            seed=31,
            particle_count=2_000,
            conditional_ess_fraction=0.8,
            mh_sweeps=10,
        )
        union = set(exact) | set(smc.distribution)
        tv = 0.5 * sum(
            abs(exact.get(graph, 0.0) - smc.distribution.get(graph, 0.0))
            for graph in union
        )
        self.assertLess(tv, 0.12)
        self.assertAlmostEqual(smc.log_normalizer_ratio, exact_log_z, delta=0.10)
        self.assertEqual(smc.temperatures[0], 0.0)
        self.assertEqual(smc.temperatures[-1], 1.0)

    def test_adaptive_support_expands_prunes_and_preserves_neutral_evidence(self) -> None:
        core = {graph_from_edges(3): 1.0}
        constant = math.log(2.0)
        result = adaptive_support_ceg_fuse(
            core,
            self.score,
            lambda graph: constant,
            seed=37,
            epsilon=0.2,
            max_removed_mass=0.03,
            chain_count=2,
            iterations=600,
            burn_in=100,
            thin=2,
            particle_count=300,
            conditional_ess_fraction=0.8,
            mh_sweeps=2,
        )

        self.assertGreater(result.created_data_support_size, 0)
        self.assertGreater(result.created_evidence_support_size, 0)
        self.assertLess(
            result.data_pruning.retained_support_size,
            result.data_pruning.original_support_size,
        )
        self.assertEqual(
            result.unpruned_evidence_distribution,
            result.unpruned_data_distribution,
        )
        self.assertEqual(result.evidence_distribution, result.data_distribution)
        self.assertEqual(
            result.smc_result.distribution, result.mcmc_result.distribution
        )
        self.assertEqual(result.smc_result.log_normalizer_ratio, constant)
        self.assertEqual(
            result.evidence_diagnostics["posterior_exploration_share"], 0.2
        )
        for pruning in (result.data_pruning, result.evidence_pruning):
            self.assertLessEqual(pruning.removed_mass, 0.03)
            self.assertAlmostEqual(pruning.total_variation, pruning.removed_mass)
            self.assertAlmostEqual(sum(pruning.distribution.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
