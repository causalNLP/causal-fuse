from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core import finite_support as fs
from cd_llm.experiments import latent_mag_validation as study


def dag_separated_by_moralization(graph, first, second, conditioned):
    """Independent d-separation implementation using the ancestral moral graph."""
    parents = {node: {other for other in range(graph.node_count)
                      if graph.directed[other] & (1 << node)}
               for node in range(graph.node_count)}
    ancestors = {first, second, *conditioned}
    while True:
        extended = ancestors | set().union(*(parents[node] for node in ancestors))
        if extended == ancestors:
            break
        ancestors = extended
    moral = {node: set() for node in ancestors}
    for node in ancestors:
        family = parents[node] & ancestors
        for parent in family:
            moral[node].add(parent)
            moral[parent].add(node)
        for left, right in itertools.combinations(family, 2):
            moral[left].add(right)
            moral[right].add(left)
    reached = {first}
    stack = [first]
    while stack:
        node = stack.pop()
        for neighbor in moral[node] - reached - set(conditioned):
            reached.add(neighbor)
            stack.append(neighbor)
    return second not in reached


def mag_separated_by_paths(graph, first, second, conditioned):
    """Independent m-separation implementation enumerating simple paths."""
    conditioned = set(conditioned)
    activated = set(conditioned)
    while True:
        ancestors = {parent for parent in range(graph.node_count)
                     for child in activated if graph.directed[parent] & (1 << child)}
        if ancestors <= activated:
            break
        activated |= ancestors

    def arrow_at(node, neighbor):
        return bool(graph.directed[neighbor] & (1 << node)
                    or graph.bidirected[node] & (1 << neighbor))

    def adjacent(node, neighbor):
        return bool(graph.directed[node] & (1 << neighbor)
                    or graph.directed[neighbor] & (1 << node)
                    or graph.bidirected[node] & (1 << neighbor))

    stack = [[first]]
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == second:
            return False
        for neighbor in range(graph.node_count):
            if neighbor in path or not adjacent(node, neighbor):
                continue
            if len(path) > 1:
                collider = arrow_at(node, path[-2]) and arrow_at(node, neighbor)
                if (collider and node not in activated) or (not collider and node in conditioned):
                    continue
            stack.append(path + [neighbor])
    return True


class LatentMAGValidationTests(unittest.TestCase):
    def test_latent_projection_matches_independent_conditional_independences(self):
        # Hidden common cause, observed collider and collider descendant.
        full = fs.graph_from_edges(6, [(4, 0), (4, 1), (0, 2), (1, 2), (2, 3), (5, 1), (5, 3)])
        projected = fs.latent_projection(full, 4)
        self.assertTrue(fs.is_maximal_ancestral(projected))
        for first, second in itertools.combinations(range(4), 2):
            others = set(range(4)) - {first, second}
            for size in range(3):
                for conditioned in itertools.combinations(others, size):
                    with self.subTest(pair=(first, second), conditioned=conditioned):
                        self.assertEqual(
                            dag_separated_by_moralization(full, first, second, conditioned),
                            mag_separated_by_paths(projected, first, second, conditioned),
                        )

    def test_independent_check_distinguishes_collider_from_chain(self):
        collider = fs.graph_from_edges(3, [(0, 1), (2, 1)])
        chain = fs.graph_from_edges(3, [(0, 1), (1, 2)])
        self.assertTrue(mag_separated_by_paths(collider, 0, 2, ()))
        self.assertFalse(mag_separated_by_paths(chain, 0, 2, ()))
        self.assertFalse(mag_separated_by_paths(collider, 0, 2, (1,)))
        self.assertTrue(mag_separated_by_paths(chain, 0, 2, (1,)))

    def test_worked_example_exact_posterior_and_uninformative_evidence(self):
        example = study.worked_example()
        np.testing.assert_allclose(example["after_endpoint_claim"], [4 / 9, 4 / 9, 1 / 9])
        np.testing.assert_allclose(example["after_both_claims"], [16 / 21, 4 / 21, 1 / 21])
        graphs = [study.graph_from_record(row) for row in example["graphs"]]
        ensemble = study.uniform_ensemble(graphs)
        claims = [fs.Claim("bidirected", "III", "bidirected", 1, 0, 1)]
        values = fs.claim_value_matrix(ensemble.graphs, claims)
        for evidence in (fs.EvidenceGraph("latent-validation", ()),
                         study.independent_evidence(claims, [1], 0.5)):
            np.testing.assert_array_equal(fs.preference_fusion(ensemble, values, evidence), ensemble.q_data)

    def test_generation_is_nested_balanced_and_uses_two_hidden_variables(self):
        config = study.Config()
        target, full, coefficients, observed, claims, seeds = study.model_and_observations(config, 0)
        self.assertEqual(full.node_count, 12)
        self.assertEqual(observed.shape, (1000, 10))
        self.assertEqual(coefficients.shape, (12, 12))
        self.assertEqual(target, fs.latent_projection(full, 10))
        full_data = fs.simulate_linear_sem(full, coefficients, 1000, seeds["data"])
        np.testing.assert_array_equal(observed[:200], full_data[:200, :10])
        self.assertEqual(len(claims), 30)
        for group in ("I", "II", "III"):
            selected = [claim for claim in claims if claim.group == group]
            self.assertEqual(len(selected), 10)
            self.assertEqual(sum(claim.truth for claim in selected), 5)

    def test_uniform_support_deduplicates_and_does_not_insert_truth(self):
        first = fs.graph_from_edges(3, [(0, 1)])
        second = fs.graph_from_edges(3, [(1, 0)])
        truth = fs.graph_from_edges(3, [(1, 2)])
        ensemble = study.uniform_ensemble([first, first, second])
        self.assertEqual(len(ensemble.graphs), 2)
        self.assertNotIn(truth, ensemble.graphs)
        np.testing.assert_allclose(ensemble.q_data, [0.5, 0.5])

    def test_bidirected_metric_and_paired_interval(self):
        truth = fs.graph_from_edges(3, [], [(0, 1)])
        estimate = fs.graph_from_edges(3, [], [(0, 1), (1, 2)])
        self.assertAlmostEqual(study.bidirected_f1(estimate, truth), 2 / 3)
        paired = study.bootstrap_paired([0.1, 0.8, 0.3], [0.0, 0.7, 0.2], 2000, 9)
        np.testing.assert_allclose(paired["paired_95_interval"], [-0.1, -0.1])

    def test_manifest_rejects_implementation_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study.initialize(root)
            with patch.object(study, "VERSION", "incompatible-version"):
                with self.assertRaisesRegex(RuntimeError, "implementation or configuration changed"):
                    study.load_manifest(root)

    def test_one_fit_resumes_and_preserves_shared_response_draws(self):
        target = fs.graph_from_edges(3, [(1, 2)], [(0, 1)])
        alternative = fs.graph_from_edges(3, [(0, 1), (1, 2)])
        claims = [fs.Claim("bidirected", "III", "bidirected", 1, 0, 1)]
        pag = fs.PartialGraph.from_matrix(np.array([[0, 3, 0], [3, 0, 3], [0, 3, 0]]), "pag")
        diagnostics = fs.CompletionSamplingDiagnostics(200, 2, "tetrad", 3, 1, 1, 1, True)
        config = study.Config()
        observations = np.random.default_rng(7).normal(size=(1000, 3))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = study.initialize(root, config)
            with patch.object(fs, "learn_fci", return_value=pag) as learn, patch.object(
                fs, "sample_completions_with_diagnostics", return_value=((target, alternative), diagnostics)
            ) as complete:
                first = study.run_dataset(root, config, manifest, 0, 200, target, observations, claims)
                self.assertEqual(first["status"], "complete", first.get("error"))
                study.run_dataset(root, config, manifest, 0, 200, target, observations, claims)
                self.assertEqual(learn.call_count, 1)
                self.assertEqual(complete.call_count, 1)
                self.assertTrue(complete.call_args.kwargs["exclude_selection_bias"])
                second = study.run_dataset(root, config, manifest, 0, 1000, target, observations, claims)
                self.assertEqual(learn.call_count, 2)
                for before, after in zip(first["responses"], second["responses"]):
                    np.testing.assert_array_equal(before["answers"], after["answers"])
                    self.assertEqual(before["seed"], after["seed"])
                np.testing.assert_array_equal(learn.call_args_list[0].args[0], observations[:200])
                self.assertEqual(first["informative_claims"]["III"], 1)

    def test_completion_failure_is_explicit_without_truth_fallback(self):
        target = fs.graph_from_edges(2, [], [(0, 1)])
        pag = fs.PartialGraph.from_matrix(np.array([[0, 3], [3, 0]]), "pag")
        config = replace(study.Config(), graph_count=1, sample_sizes=(200,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = study.initialize(root, config)
            with patch.object(fs, "learn_fci", return_value=pag), patch.object(
                fs, "sample_completions_with_diagnostics", side_effect=fs.NonCompletablePAGError("no certified candidates")
            ):
                result = study.run_dataset(root, config, manifest, 0, 200, target, np.ones((200, 2)), [])
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("data_only", result)
            summary = study.aggregate(root)
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["all_successful"])
            self.assertEqual(len(summary["panels"][0]["failed_scms"]), 1)
            self.assertEqual(summary["panels"][0]["successful_scms"], 0)


if __name__ == "__main__":
    unittest.main()
