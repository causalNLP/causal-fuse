from __future__ import annotations

import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import (
    Claim,
    EvidenceGraph,
    EvidenceNode,
    Graph,
    evaluate_claim,
    graph_from_edges,
    structural_hamming_distance,
    transitive_closure,
)
from cd_llm.methods.adaptive_pruning import (
    GaussianBICScore,
    normalize_distribution,
    score_distribution,
)
from cd_llm.methods.evidence_guided import (
    BALANCED_KERNEL,
    ClaimFuseSMCConfig,
    DAGBaseProposal,
    EvidenceGuidedDAGProposal,
    UNIFORM_PAIR_KERNEL,
    aggregate_weighted_particles,
    claim_fuse_smc,
    evidence_guided_tempered_smc,
    initialize_score_particles,
)


def all_dags(node_count: int) -> tuple[Graph, ...]:
    pairs = tuple(itertools.combinations(range(node_count), 2))
    result: list[Graph] = []
    for states in itertools.product((0, 1, 2), repeat=len(pairs)):
        edges: list[tuple[int, int]] = []
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


def local_claim(
    claim_id: str,
    group: str,
    kind: str,
    source: int,
    target: int,
    truth: int = 1,
) -> Claim:
    return Claim(claim_id, group, kind, truth, source=source, target=target)


def evidence_for(
    claims: tuple[Claim, ...],
    stances: tuple[int, ...],
    confidence: float = 0.8,
) -> EvidenceGraph:
    return EvidenceGraph(
        "target",
        tuple(
            EvidenceNode(
                f"e{index}",
                index,
                stance,
                confidence,
                f"source-{index}",
                f"factor-{index}",
                "target",
                True,
                True,
            )
            for index, stance in enumerate(stances)
        ),
    )


def total_variation(
    first: dict[Graph, float], second: dict[Graph, float]
) -> float:
    support = set(first) | set(second)
    return 0.5 * math.fsum(
        abs(first.get(graph, 0.0) - second.get(graph, 0.0))
        for graph in support
    )


class ProposalKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = all_dags(3)
        self.claims = (
            local_claim("edge", "I", "directed", 0, 1),
            local_claim("ancestor", "II", "ancestry", 0, 2),
        )
        self.evidence = evidence_for(self.claims, (1, 1))
        self.proposal = EvidenceGuidedDAGProposal(
            self.claims,
            self.evidence,
            node_count=3,
            gamma=0.5,
            cache_size=8,
        )

    def test_ordinary_and_complete_proposals_normalize_over_valid_dags(self) -> None:
        graph_set = set(self.graphs)
        for graph in self.graphs:
            ordinary = self.proposal.ordinary_distribution(graph)
            complete = self.proposal.proposal_distribution(graph)
            self.assertAlmostEqual(math.fsum(ordinary.values()), 1.0, places=14)
            self.assertAlmostEqual(math.fsum(complete.values()), 1.0, places=14)
            self.assertTrue(set(ordinary).issubset(graph_set))
            self.assertTrue(set(complete).issubset(graph_set))
            self.assertTrue(all(candidate.regime == "dag" for candidate in complete))

    def test_repairs_resolve_the_selected_violation_in_one_pair(self) -> None:
        empty = graph_from_edges(3)
        self.assertIn("e0", self.proposal.violated_evidence_ids(empty))
        repairs = self.proposal.repair_distribution(empty, "e0")
        self.assertAlmostEqual(math.fsum(repairs.values()), 1.0)
        for graph in repairs:
            self.assertEqual(evaluate_claim(graph, self.claims[0]), 1)
            self.assertEqual(structural_hamming_distance(empty, graph), 1)

    def test_empty_repair_set_falls_back_to_q0(self) -> None:
        graph = graph_from_edges(3, [(2, 1), (1, 0)])
        claim = (local_claim("ancestor", "II", "ancestry", 0, 2),)
        proposal = EvidenceGuidedDAGProposal(
            claim,
            evidence_for(claim, (1,)),
            node_count=3,
            gamma=0.5,
        )
        self.assertEqual(
            proposal.repair_distribution(graph, "e0"),
            proposal.ordinary_distribution(graph),
        )

    def test_indexed_group_ii_repairs_match_exhaustive_definition(self) -> None:
        for node_count in (2, 3, 4):
            graphs = all_dags(node_count)
            claims: list[Claim] = []
            stances: list[int] = []
            for kind in ("ancestry", "ordering"):
                for source in range(node_count):
                    for target in range(node_count):
                        if source == target:
                            continue
                        for stance in (-1, 1):
                            claims.append(
                                local_claim(
                                    f"{kind}-{source}-{target}-{stance}",
                                    "II",
                                    kind,
                                    source,
                                    target,
                                )
                            )
                            stances.append(stance)
            frozen_claims = tuple(claims)
            frozen_stances = tuple(stances)
            for max_parents in sorted({max(0, node_count - 2), node_count - 1}):
                proposal = EvidenceGuidedDAGProposal(
                    frozen_claims,
                    evidence_for(frozen_claims, frozen_stances),
                    node_count=node_count,
                    gamma=0.5,
                    max_parents=max_parents,
                    cache_size=64,
                )
                for graph in graphs:
                    parent_counts = [0] * node_count
                    for row in graph.directed:
                        for target in range(node_count):
                            parent_counts[target] += int(bool(row & (1 << target)))
                    if max(parent_counts, default=0) > max_parents:
                        continue
                    states = proposal.evidence_states(graph)
                    for representative_index, state in enumerate(states):
                        if state != -1:
                            continue
                        expected: list[Graph] = []
                        for first in range(node_count):
                            for second in range(first + 1, node_count):
                                current_state = (
                                    1
                                    if graph.directed[first] & (1 << second)
                                    else 2
                                    if graph.directed[second] & (1 << first)
                                    else 0
                                )
                                for proposed_state in (0, 1, 2):
                                    if proposed_state == current_state:
                                        continue
                                    rows = list(graph.directed)
                                    rows[first] &= ~(1 << second)
                                    rows[second] &= ~(1 << first)
                                    if proposed_state == 1:
                                        rows[first] |= 1 << second
                                    elif proposed_state == 2:
                                        rows[second] |= 1 << first
                                    try:
                                        candidate = Graph(tuple(rows))
                                    except ValueError:
                                        continue
                                    candidate_parent_counts = [0] * node_count
                                    for row in candidate.directed:
                                        for target in range(node_count):
                                            candidate_parent_counts[target] += int(
                                                bool(row & (1 << target))
                                            )
                                    if max(candidate_parent_counts, default=0) > max_parents:
                                        continue
                                    representative = proposal.representatives[
                                        representative_index
                                    ]
                                    claim = frozen_claims[
                                        representative.claim_index
                                    ]
                                    desired = 1 if representative.stance == 1 else 0
                                    if evaluate_claim(candidate, claim) == desired:
                                        expected.append(candidate)
                        with self.subTest(
                            node_count=node_count,
                            max_parents=max_parents,
                            graph=graph,
                            claim=frozen_claims[
                                proposal.representatives[
                                    representative_index
                                ].claim_index
                            ].claim_id,
                        ):
                            self.assertEqual(
                                proposal._repair_candidates(
                                    graph, representative_index
                                ),
                                tuple(expected),
                            )

    def test_score_cache_miss_is_safe_when_state_cache_still_contains_graph(self) -> None:
        proposal = EvidenceGuidedDAGProposal(
            self.claims,
            self.evidence,
            node_count=3,
            gamma=0.5,
            cache_size=1,
        )
        target = self.graphs[0]
        expected = proposal.evidence_score(target)
        for graph in self.graphs[1:4]:
            proposal.evidence_score(graph)
        proposal.evidence_states(target)
        proposal.evidence_score(self.graphs[4])
        self.assertEqual(proposal.evidence_score(target), expected)

    def test_multiple_path_claims_share_one_transitive_closure(self) -> None:
        claims = (
            local_claim("ancestor-02", "II", "ancestry", 0, 2),
            local_claim("ancestor-01", "II", "ancestry", 0, 1),
            local_claim("ordering-20", "II", "ordering", 2, 0),
        )
        proposal = EvidenceGuidedDAGProposal(
            claims,
            evidence_for(claims, (1, 1, 1)),
            node_count=3,
            gamma=0.5,
        )
        graph = graph_from_edges(3, [(0, 1), (1, 2)])
        with patch(
            "cd_llm.methods.evidence_guided.transitive_closure",
            wraps=transitive_closure,
        ) as closure:
            self.assertEqual(proposal.evidence_states(graph), (1, 1, -1))
        self.assertEqual(closure.call_count, 1)

    def test_analytical_self_mass_matches_exhaustive_enumeration(self) -> None:
        for max_parents in range(3):
            proposal = EvidenceGuidedDAGProposal(
                self.claims,
                self.evidence,
                node_count=3,
                gamma=0.5,
                max_parents=max_parents,
            )
            for graph in self.graphs:
                with self.subTest(max_parents=max_parents, graph=graph):
                    self.assertEqual(
                        proposal._invalid_ordinary_count(graph),
                        proposal._invalid_ordinary_count_enumerated(graph),
                    )

    def test_internal_self_draw_defers_unused_transition_mass(self) -> None:
        proposal = EvidenceGuidedDAGProposal(
            (), EvidenceGraph("target", ()), node_count=3, gamma=0.0
        )
        empty = graph_from_edges(3)
        with patch.object(
            proposal, "_sample_ordinary", return_value=empty
        ), patch.object(
            proposal,
            "proposal_probability",
            side_effect=AssertionError("self mass should be deferred"),
        ):
            candidate, used_guided, fallback = proposal._sample_candidate(
                empty, np.random.default_rng(7)
            )
        self.assertEqual(candidate, empty)
        self.assertFalse(used_guided)
        self.assertFalse(fallback)

    def test_exhaustive_three_node_mh_kernel_satisfies_detailed_balance(self) -> None:
        rng = np.random.default_rng(81)
        scorer = GaussianBICScore(rng.normal(size=(20, 3)))
        temperature = 0.7
        log_targets = {
            graph: scorer.graph_score(graph)
            + temperature * self.proposal.evidence_score(graph)
            for graph in self.graphs
        }
        maximum = max(log_targets.values())
        target = normalize_distribution(
            {
                graph: math.exp(log_target - maximum)
                for graph, log_target in log_targets.items()
            }
        )
        for current in self.graphs:
            for proposed in self.graphs:
                forward = self.proposal.proposal_probability(proposed, current)
                reverse = self.proposal.proposal_probability(current, proposed)
                if current == proposed:
                    transition_forward = forward
                    transition_reverse = reverse
                else:
                    transition_forward = forward * min(
                        1.0,
                        math.exp(log_targets[proposed] - log_targets[current])
                        * reverse
                        / forward,
                    ) if forward else 0.0
                    transition_reverse = reverse * min(
                        1.0,
                        math.exp(log_targets[current] - log_targets[proposed])
                        * forward
                        / reverse,
                    ) if reverse else 0.0
                self.assertAlmostEqual(
                    target[current] * transition_forward,
                    target[proposed] * transition_reverse,
                    places=13,
                )


class BalancedProposalKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs3 = all_dags(3)
        self.screen3 = ((0, 1), (1, 2), (2, 0), (0, 2))

    def test_legacy_default_is_unchanged_by_the_opt_in_kernel(self) -> None:
        default = DAGBaseProposal(node_count=3, max_parents=2)
        explicit = DAGBaseProposal(
            node_count=3,
            max_parents=2,
            kernel=UNIFORM_PAIR_KERNEL,
            screened_additions=self.screen3,
        )
        for graph in self.graphs3:
            self.assertEqual(default.distribution(graph), explicit.distribution(graph))

    def test_balanced_component_masses_and_invalid_self_transitions(self) -> None:
        empty = graph_from_edges(3)
        edge = graph_from_edges(3, [(0, 1)])
        reverse = graph_from_edges(3, [(1, 0)])
        proposal = DAGBaseProposal(
            node_count=3,
            max_parents=2,
            kernel=BALANCED_KERNEL,
            screened_additions=((0, 1),),
        )
        self.assertAlmostEqual(proposal.probability(empty, empty), 0.45)
        self.assertAlmostEqual(
            proposal.probability(edge, empty), 0.45 + 0.10 / 6.0
        )
        self.assertAlmostEqual(proposal.probability(reverse, empty), 0.10 / 6.0)
        self.assertAlmostEqual(
            proposal.probability(empty, edge), 0.45 / 2.0 + 0.10 / 6.0
        )
        self.assertAlmostEqual(
            proposal.probability(reverse, edge), 0.45 / 2.0 + 0.10 / 6.0
        )
        self.assertGreaterEqual(proposal.probability(edge, edge), 0.45)

        cyclic = graph_from_edges(3, [(0, 1), (1, 2)])
        cyclic_screen = DAGBaseProposal(
            node_count=3,
            max_parents=2,
            kernel=BALANCED_KERNEL,
            screened_additions=((2, 0),),
        )
        self.assertGreaterEqual(cyclic_screen.probability(cyclic, cyclic), 0.45)

        capped = DAGBaseProposal(
            node_count=3,
            max_parents=0,
            kernel=BALANCED_KERNEL,
            screened_additions=((0, 1),),
        )
        self.assertEqual(capped.distribution(empty), {empty: 1.0})

    def test_balanced_screen_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            DAGBaseProposal(
                node_count=3,
                kernel=BALANCED_KERNEL,
                screened_additions=((0, 1), (0, 1)),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            DAGBaseProposal(
                node_count=3,
                kernel=BALANCED_KERNEL,
                screened_additions=((0, 3),),
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            DAGBaseProposal(
                node_count=3,
                kernel=BALANCED_KERNEL,
                screened_additions=((1, 1),),
            )
        with self.assertRaisesRegex(ValueError, "kernel"):
            DAGBaseProposal(node_count=3, kernel="unknown")

    def test_seeded_balanced_draws_are_deterministic_and_exactly_diagnosed(self) -> None:
        graph = graph_from_edges(3, [(0, 1)])
        proposal = DAGBaseProposal(
            node_count=3,
            max_parents=2,
            kernel=BALANCED_KERNEL,
            screened_additions=self.screen3,
        )
        first_rng = np.random.default_rng(773)
        second_rng = np.random.default_rng(773)
        first = tuple(proposal.sample(graph, first_rng) for _ in range(200))
        second = tuple(proposal.sample(graph, second_rng) for _ in range(200))
        self.assertEqual(first, second)
        self.assertEqual(
            {draw.component for draw in first},
            {UNIFORM_PAIR_KERNEL, "existing_edge", "screened_addition"},
        )
        for draw in first:
            self.assertEqual(draw.self_transition, draw.graph == graph)
            self.assertAlmostEqual(
                draw.probability, proposal.probability(draw.graph, graph), places=15
            )
            self.assertEqual(draw.graph.regime, "dag")

    def test_every_four_node_balanced_distribution_normalizes(self) -> None:
        proposal = DAGBaseProposal(
            node_count=4,
            max_parents=3,
            kernel=BALANCED_KERNEL,
            screened_additions=((0, 1), (1, 2), (2, 0), (3, 0), (3, 2)),
        )
        for graph in all_dags(4):
            distribution = proposal.distribution(graph)
            self.assertAlmostEqual(math.fsum(distribution.values()), 1.0, places=14)
            self.assertTrue(all(candidate.regime == "dag" for candidate in distribution))

    def test_exhaustive_four_node_balanced_mh_satisfies_detailed_balance(self) -> None:
        graphs = all_dags(4)
        proposal = DAGBaseProposal(
            node_count=4,
            max_parents=3,
            kernel=BALANCED_KERNEL,
            screened_additions=((0, 1), (1, 2), (2, 0), (3, 0), (3, 2)),
        )
        log_targets = {
            graph: -0.37 * sum(row.bit_count() for row in graph.directed)
            + 0.11 * sum((node + 1) * row for node, row in enumerate(graph.directed))
            for graph in graphs
        }
        maximum = max(log_targets.values())
        target = normalize_distribution(
            {
                graph: math.exp(log_target - maximum)
                for graph, log_target in log_targets.items()
            }
        )
        for current in graphs:
            for proposed in graphs:
                forward = proposal.probability(proposed, current)
                reverse = proposal.probability(current, proposed)
                if current == proposed:
                    transition_forward = forward
                    transition_reverse = reverse
                elif forward > 0.0 and reverse > 0.0:
                    transition_forward = forward * min(
                        1.0,
                        math.exp(log_targets[proposed] - log_targets[current])
                        * reverse
                        / forward,
                    )
                    transition_reverse = reverse * min(
                        1.0,
                        math.exp(log_targets[current] - log_targets[proposed])
                        * forward
                        / reverse,
                    )
                else:
                    transition_forward = 0.0
                    transition_reverse = 0.0
                self.assertAlmostEqual(
                    target[current] * transition_forward,
                    target[proposed] * transition_reverse,
                    places=13,
                )

    def test_balanced_evidence_mixture_normalizes_and_balances(self) -> None:
        claims = (
            local_claim("edge", "I", "directed", 0, 1),
            local_claim("ancestor", "II", "ancestry", 0, 2),
        )
        proposal = EvidenceGuidedDAGProposal(
            claims,
            evidence_for(claims, (1, 1)),
            node_count=3,
            gamma=0.5,
            max_parents=2,
            ordinary_kernel=BALANCED_KERNEL,
            screened_additions=self.screen3,
        )
        rng = np.random.default_rng(811)
        scorer = GaussianBICScore(rng.normal(size=(20, 3)))
        temperature = 0.7
        log_targets = {
            graph: scorer.graph_score(graph)
            + temperature * proposal.evidence_score(graph)
            for graph in self.graphs3
        }
        maximum = max(log_targets.values())
        target = normalize_distribution(
            {
                graph: math.exp(log_target - maximum)
                for graph, log_target in log_targets.items()
            }
        )
        for current in self.graphs3:
            distribution = proposal.proposal_distribution(current)
            self.assertAlmostEqual(math.fsum(distribution.values()), 1.0, places=14)
            for proposed in self.graphs3:
                forward = proposal.proposal_probability(proposed, current)
                reverse = proposal.proposal_probability(current, proposed)
                if current == proposed:
                    transition_forward = forward
                    transition_reverse = reverse
                elif forward > 0.0 and reverse > 0.0:
                    transition_forward = forward * min(
                        1.0,
                        math.exp(log_targets[proposed] - log_targets[current])
                        * reverse
                        / forward,
                    )
                    transition_reverse = reverse * min(
                        1.0,
                        math.exp(log_targets[current] - log_targets[proposed])
                        * forward
                        / reverse,
                    )
                else:
                    transition_forward = 0.0
                    transition_reverse = 0.0
                self.assertAlmostEqual(
                    target[current] * transition_forward,
                    target[proposed] * transition_reverse,
                    places=13,
                )


class ParticleAndSMCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(103)
        self.scorer = GaussianBICScore(self.rng.normal(size=(18, 3)))
        self.empty = graph_from_edges(3)
        self.edge = graph_from_edges(3, [(0, 1)])
        self.claims = (local_claim("edge", "I", "directed", 0, 1),)
        self.evidence = evidence_for(self.claims, (1,))

    def test_weight_retention_and_ess_triggered_resampling(self) -> None:
        particles = (self.empty, self.empty, self.edge, self.edge)
        strong_evidence = evidence_for(self.claims, (1,), confidence=16.0 / 17.0)
        retained = evidence_guided_tempered_smc(
            self.scorer,
            particles,
            self.claims,
            strong_evidence,
            seed=7,
            temperatures=(0.0, 1.0),
            gamma=0.0,
            ess_threshold_fraction=0.1,
            rejuvenation_steps=0,
        )
        preference = np.asarray(
            [
                EvidenceGuidedDAGProposal(
                    self.claims,
                    strong_evidence,
                    node_count=3,
                ).evidence_score(graph)
                for graph in particles
            ]
        )
        expected = np.exp(preference - np.max(preference))
        expected /= np.sum(expected)
        self.assertFalse(retained.stages[0].resampled)
        np.testing.assert_allclose(retained.weights, expected, atol=1e-15)
        self.assertEqual(
            retained.distribution,
            aggregate_weighted_particles(particles, expected),
        )

        resampled = evidence_guided_tempered_smc(
            self.scorer,
            particles,
            self.claims,
            strong_evidence,
            seed=7,
            temperatures=(0.0, 1.0),
            gamma=0.0,
            ess_threshold_fraction=0.75,
            rejuvenation_steps=0,
        )
        self.assertTrue(resampled.stages[0].resampled)
        self.assertEqual(resampled.resampling_count, 1)
        np.testing.assert_allclose(resampled.weights, np.full(4, 0.25))

    def test_initializer_is_deterministic(self) -> None:
        first = initialize_score_particles(
            self.scorer,
            (self.empty, self.edge),
            seed=29,
            particle_count=30,
            updates_per_particle=25,
        )
        second = initialize_score_particles(
            self.scorer,
            (self.empty, self.edge),
            seed=29,
            particle_count=30,
            updates_per_particle=25,
        )
        self.assertEqual(first.particles, second.particles)
        self.assertEqual(first.distribution, second.distribution)
        self.assertEqual(first.accepted_moves, second.accepted_moves)

        weighted_first = initialize_score_particles(
            self.scorer,
            (self.empty, self.edge),
            seed=31,
            particle_count=30,
            updates_per_particle=5,
            warm_start_score_mass=0.9,
        )
        weighted_second = initialize_score_particles(
            self.scorer,
            (self.empty, self.edge),
            seed=31,
            particle_count=30,
            updates_per_particle=5,
            warm_start_score_mass=0.9,
        )
        self.assertEqual(weighted_first.particles, weighted_second.particles)

    def test_initializer_accepts_weighted_duplicate_warm_starts(self) -> None:
        with patch(
            "cd_llm.methods.evidence_guided._score_only_step",
            side_effect=lambda rows, parents, score, scorer, pairs, rng: (
                rows,
                parents,
                score,
                False,
            ),
        ):
            result = initialize_score_particles(
                self.scorer,
                (self.empty, self.edge, self.empty),
                seed=41,
                particle_count=20,
                updates_per_particle=1,
                warm_start_score_mass=0.0,
                initial_graph_probabilities=(0.25, 0.5, 0.25),
            )
        self.assertEqual(result.particles.count(self.empty), 10)
        self.assertEqual(result.particles.count(self.edge), 10)
        self.assertEqual(result.starting_particles, result.particles)

    def test_initializer_rejects_invalid_warm_start_probabilities(self) -> None:
        for probabilities in ((1.0,), (0.0, 0.0), (0.5, -0.5)):
            with self.subTest(probabilities=probabilities), self.assertRaises(
                ValueError
            ):
                initialize_score_particles(
                    self.scorer,
                    (self.empty, self.edge),
                    seed=43,
                    particle_count=4,
                    updates_per_particle=1,
                    initial_graph_probabilities=probabilities,
                )

    def test_balanced_initializer_uses_exact_proposal_and_reports_components(self) -> None:
        arguments = {
            "seed": 37,
            "particle_count": 24,
            "updates_per_particle": 20,
            "ordinary_kernel": BALANCED_KERNEL,
            "screened_additions": ((0, 1), (1, 2), (0, 2)),
            "proposal_cache_size": 16,
        }
        first = initialize_score_particles(
            self.scorer, (self.empty, self.edge), **arguments
        )
        second = initialize_score_particles(
            self.scorer, (self.empty, self.edge), **arguments
        )
        self.assertEqual(first.particles, second.particles)
        self.assertEqual(first.accepted_moves, second.accepted_moves)
        self.assertEqual(
            first.proposal_component_counts, second.proposal_component_counts
        )
        self.assertEqual(
            sum(count for _, count in first.proposal_component_counts),
            first.attempted_moves,
        )
        self.assertGreater(first.self_proposals, 0)
        self.assertGreater(first.accepted_moves, 0)

    def test_adaptive_tempering_is_monotone_and_skips_stable_intermediate_mutation(self) -> None:
        particles = (self.empty, self.empty, self.edge, self.edge)
        result = evidence_guided_tempered_smc(
            self.scorer,
            particles,
            self.claims,
            evidence_for(self.claims, (1,), confidence=16.0 / 17.0),
            seed=17,
            adaptive_tempering=True,
            target_ess_fraction=0.7,
            max_tempering_stages=4,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=2,
            rejuvenate_only_after_resampling=True,
        )
        self.assertEqual(result.temperatures[0], 0.0)
        self.assertEqual(result.temperatures[-1], 1.0)
        self.assertLessEqual(len(result.stages), 4)
        self.assertTrue(
            all(
                following > current
                for current, following in zip(
                    result.temperatures, result.temperatures[1:]
                )
            )
        )
        for stage in result.stages[:-1]:
            if not stage.resampled:
                self.assertEqual(stage.attempted_moves, 0)
        self.assertGreater(result.stages[-1].attempted_moves, 0)

    def test_seeded_three_node_result_approximates_exact_target(self) -> None:
        graphs = all_dags(3)
        exact_data = score_distribution(graphs, self.scorer)
        config = ClaimFuseSMCConfig(
            particle_count=1_000,
            initialization_updates=500,
            temperatures=tuple(index / 10.0 for index in range(11)),
            adaptive_tempering=False,
            rejuvenation_steps=10,
            rejuvenate_only_after_resampling=False,
            warm_start_score_mass=0.0,
            max_parents=2,
        )
        result = claim_fuse_smc(
            self.scorer,
            tuple(sorted(exact_data, key=exact_data.get, reverse=True)[:4]),
            self.claims,
            self.evidence,
            seed=41,
            config=config,
        )
        evaluator = EvidenceGuidedDAGProposal(
            self.claims,
            self.evidence,
            node_count=3,
            gamma=0.5,
        )
        log_masses = {
            graph: self.scorer.graph_score(graph) + evaluator.evidence_score(graph)
            for graph in graphs
        }
        maximum = max(log_masses.values())
        exact = normalize_distribution(
            {
                graph: math.exp(value - maximum)
                for graph, value in log_masses.items()
            }
        )
        self.assertLessEqual(total_variation(exact, result.evidence_distribution), 0.08)
        self.assertGreater(result.smc.attempted_moves, 0)
        self.assertEqual(result.smc.temperatures[0], 0.0)
        self.assertEqual(result.smc.temperatures[-1], 1.0)

        fast = claim_fuse_smc(
            self.scorer,
            tuple(sorted(exact_data, key=exact_data.get, reverse=True)[:4]),
            self.claims,
            self.evidence,
            seed=41,
            config=ClaimFuseSMCConfig(max_parents=2),
        )
        self.assertLessEqual(
            total_variation(exact, fast.evidence_distribution), 0.15
        )
        self.assertLessEqual(len(fast.smc.stages), 4)

    def test_claim_fuse_smc_is_deterministic_and_no_evidence_is_identity(self) -> None:
        config = ClaimFuseSMCConfig(
            particle_count=30,
            initialization_updates=25,
            temperatures=(0.0, 0.5, 1.0),
            adaptive_tempering=False,
            rejuvenation_steps=2,
            rejuvenate_only_after_resampling=False,
            proposal_cache_size=16,
            max_parents=2,
        )
        first = claim_fuse_smc(
            self.scorer,
            (self.empty, self.edge),
            self.claims,
            self.evidence,
            seed=91,
            config=config,
        )
        second = claim_fuse_smc(
            self.scorer,
            (self.empty, self.edge),
            self.claims,
            self.evidence,
            seed=91,
            config=config,
        )
        self.assertEqual(first.data_distribution, second.data_distribution)
        self.assertEqual(first.evidence_distribution, second.evidence_distribution)
        self.assertEqual(first.smc.stages, second.smc.stages)

        identity = claim_fuse_smc(
            self.scorer,
            (self.empty, self.edge),
            (),
            EvidenceGraph("target", ()),
            seed=91,
            config=config,
        )
        self.assertEqual(identity.data_distribution, identity.evidence_distribution)
        self.assertEqual(identity.smc.stages, ())

    def test_claim_fuse_smc_rejects_a_mismatched_parent_cap(self) -> None:
        config = ClaimFuseSMCConfig(
            particle_count=10,
            initialization_updates=5,
            max_parents=1,
        )
        with self.assertRaisesRegex(ValueError, "max-parent caps differ"):
            claim_fuse_smc(
                self.scorer,
                (self.empty,),
                self.claims,
                self.evidence,
                seed=1,
                config=config,
            )

    def test_none_parent_cap_matches_the_scores_full_dag_space(self) -> None:
        result = claim_fuse_smc(
            self.scorer,
            (self.empty, self.edge),
            self.claims,
            self.evidence,
            seed=5,
            config=ClaimFuseSMCConfig(
                particle_count=12,
                initialization_updates=4,
                max_parents=None,
            ),
        )
        self.assertIsNone(result.config.max_parents)
        self.assertEqual(result.score_metadata["max_parents"], 2)


if __name__ == "__main__":
    unittest.main()
