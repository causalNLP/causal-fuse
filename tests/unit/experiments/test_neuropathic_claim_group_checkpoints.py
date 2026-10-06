import copy
from dataclasses import asdict
import unittest

import numpy as np

from cd_llm.core.finite_support import graph_edges, graph_from_edges
from cd_llm.experiments import neuropathic_claim_group_checkpoints as study
from cd_llm.experiments.common.artifacts import json_value
from cd_llm.methods.active_query import ClaimQuery, QueryResponse, active_query_smc
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class CheckpointStudyTests(unittest.TestCase):
    def sample(self, budgets):
        distribution = {graph_from_edges(3, edges): 1 / 3 for edges in ([], [(0, 1)], [(1, 0)])}
        score = GaussianBICScore(np.random.default_rng(4).normal(size=(30, 3)), max_parents=2)
        replies = iter(((1, 0.99), (None, None), (0, 0.8), (None, None),
                        (1, 0.5), (1, 0.5), (0, 0.8), (None, None), (1, 0.8)))

        def provider(query):
            answer, confidence = next(replies)
            return QueryResponse(answer, confidence, treatment="soft_evidence")

        return active_query_smc(distribution, score, provider, budgets=budgets,
            budget_unit="attempted", seed=17, ess_threshold_fraction=0.9, rejuvenation_steps=1)

    def test_snapshots_preserve_selection_resampling_mutation_and_terminal_state(self):
        terminal = self.sample((9,))
        extended = self.sample((1, 2, 3, 4, 6, 9))
        for left, right in zip(terminal.steps, extended.steps, strict=True):
            a, b = asdict(left), asdict(right)
            a.pop("cumulative_seconds")
            b.pop("cumulative_seconds")
            self.assertEqual(a, b)
        self.assertTrue(any(s.resampled for s in extended.steps))
        self.assertEqual(terminal.distributions[9], extended.distributions[9])
        self.assertEqual(terminal.particles, extended.particles)
        self.assertEqual(terminal.weights, extended.weights)
        self.assertEqual(extended.distributions[1], extended.distributions[2])
        self.assertEqual(extended.distributions[3], extended.distributions[4])
        self.assertEqual(extended.steps[5].rejuvenation_attempted_moves, 0)
        self.assertGreater(self.sample((6,)).steps[5].rejuvenation_attempted_moves, 0)

    def test_replay_verification_checks_effective_confidence_and_response_keys(self):
        sampled = self.sample((9,))
        record = json_value(dict(trace=[asdict(s) for s in sampled.steps],
            distribution=[dict(graph=graph_edges(g), weight=w) for g, w in sampled.distributions[9].items()]))
        changed = copy.deepcopy(record)
        changed["trace"][0]["effective_confidence"] = 0.5
        with self.assertRaisesRegex(ValueError, "effective_confidence"):
            study.assert_replay(changed, record)
        changed = copy.deepcopy(record)
        changed["trace"][0]["response_audit"]["request_keys"] = ["different"]
        with self.assertRaisesRegex(ValueError, "request_keys"):
            study.assert_replay(changed, record)

    def checkpoint_record(self):
        empty = graph_edges(graph_from_edges(20))
        entries = [dict(graph=empty, weight=1.)]
        return dict(distribution=entries, distributions={str(b): copy.deepcopy(entries) for b in study.BUDGETS})

    def test_checkpoint_validation_rejects_missing_negative_and_duplicate_mass(self):
        record = self.checkpoint_record()
        study.validate_checkpoints(record)
        del record["distributions"]["20"]
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            study.validate_checkpoints(record)
        record = self.checkpoint_record()
        record["distributions"]["10"][0]["weight"] = -1
        with self.assertRaisesRegex(ValueError, "weights"):
            study.validate_checkpoints(record)
        record = self.checkpoint_record()
        record["distributions"]["10"] *= 2
        for entry in record["distributions"]["10"]:
            entry["weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "duplicate"):
            study.validate_checkpoints(record)

    def test_checkpoint_validation_rejects_cyclic_graph(self):
        record = self.checkpoint_record()
        record["distributions"]["10"][0]["graph"] = dict(directed=[[0, 1], [1, 0]], bidirected=[])
        with self.assertRaises(ValueError):
            study.validate_checkpoints(record)

    def test_prefix_metrics_use_earlier_graphs_and_answers_only(self):
        truth = graph_from_edges(20, directed=[(0, 1)])
        trace = [dict(query=asdict(ClaimQuery("directed", 0, 1, "directed-0-1")),
            answer=int(i < 10), reported_confidence=0.8, valid_response=True, resampled=False,
            rejuvenation_attempted_moves=0, rejuvenation_accepted_moves=0, prompt_tokens=1,
            completion_tokens=1, response_audit=dict(request_keys=["cached"])) for i in range(100)]
        record = self.checkpoint_record()
        record.update(trace=trace, checkpoint_seconds={str(b): b / 10 for b in study.BUDGETS})
        record["distributions"]["10"] = [dict(graph=graph_edges(truth), weight=1.)]
        early, _ = study.original.metrics_for(study.prefix_record(record, 10), truth)
        final, _ = study.original.metrics_for(study.prefix_record(record, 100), truth)
        self.assertEqual(early["attempted"], 10)
        self.assertEqual(early["accuracy"], 1)
        self.assertAlmostEqual(early["brier"], 0.04)
        self.assertAlmostEqual(early["expected_distance"], 0)
        self.assertEqual(final["accuracy"], 0.1)
        self.assertGreater(final["expected_distance"], 0)


if __name__ == "__main__":
    unittest.main()
