import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import graph_edges, graph_from_edges
from cd_llm.experiments import neuropathic_confidence_cap as study
from cd_llm.methods import active_query as active
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class ConfidenceCapTests(unittest.TestCase):
    def test_cap_boundaries_preserve_raw_answers_and_audit(self):
        for raw, expected in ((.5, .5), (.7, .7), (.8, .8), (.95, .8), (1., .8), (None, None)):
            response = active.QueryResponse(None if raw is None else 1, raw,
                abstention_reason="uncertain" if raw is None else None,
                audit=dict(request_keys=["immutable"], votes=[1]*8), treatment="soft_evidence")
            clipped = study.cap_response(response)
            self.assertEqual(clipped.answer, response.answer)
            self.assertEqual(clipped.reported_confidence, expected)
            self.assertEqual(clipped.audit["raw_reported_confidence"], raw)
            self.assertEqual(clipped.audit["request_keys"], response.audit["request_keys"])
            self.assertEqual(clipped.abstention_reason, response.abstention_reason)
            self.assertEqual(clipped.treatment, "soft_evidence")
            self.assertNotIn("raw_reported_confidence", response.audit)
            self.assertIs(study.cap_response(response, None), response)

    def run_small(self, cap=.8, budgets=(1, 2, 3), moves=1):
        empty, edge = (graph_from_edges(3, directed=e) for e in ([], [(0, 1)]))
        distribution = {empty: .5, edge: .5}
        score = GaussianBICScore(np.random.default_rng(7).normal(size=(30, 3)), max_parents=2)
        responses = iter([(1, .95), (None, None), (0, 1.)])
        def provider(q):
            answer, confidence = next(responses)
            return study.cap_response(active.QueryResponse(answer, confidence, treatment="soft_evidence"), cap)
        pool = (active.ClaimQuery("directed", 0, 1, "a"),
                active.ClaimQuery("directed", 1, 0, "b"), active.ClaimQuery("ancestry", 0, 2, "c"))
        return active.active_query_smc(distribution, score, provider, query_pool=pool,
            budgets=budgets, budget_unit="attempted", confidence=.9, treatment="soft_evidence",
            soft_confidence_cap=.99, ess_threshold_fraction=.01, rejuvenation_steps=moves, seed=31)

    def test_particle_weighting_and_mh_use_cap_but_acquisition_uses_point_nine(self):
        with patch.object(active, "_rejuvenate", wraps=active._rejuvenate) as mutation:
            result = self.run_small()
        self.assertTrue(all(s.selection_effective_confidence == .9 for s in result.steps))
        self.assertEqual([s.effective_confidence for s in result.steps], [.8, None, .8])
        edge = graph_from_edges(3, directed=[(0, 1)])
        self.assertAlmostEqual(result.distributions[1][edge], .8)
        self.assertEqual(result.distributions[1], result.distributions[2])
        self.assertEqual(result.query_count, 3)
        self.assertEqual(result.answered_query_count, 2)
        self.assertEqual(mutation.call_count, 1)
        evidence = mutation.call_args.args[3]
        self.assertEqual([n.reported_confidence for n in evidence.nodes], [.8, .8])
        self.assertEqual(result.steps[0].rejuvenation_attempted_moves, 0)
        self.assertGreater(result.steps[-1].rejuvenation_attempted_moves, 0)

    def test_checkpoints_do_not_change_trajectory_and_replay_is_deterministic(self):
        terminal, prefixes, repeated = self.run_small(budgets=(3,)), self.run_small(), self.run_small()
        for a, b, c in zip(terminal.steps, prefixes.steps, repeated.steps):
            rows = [asdict(s) for s in (a, b, c)]
            for r in rows:
                r.pop("cumulative_seconds")
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[1], rows[2])
        self.assertEqual(terminal.distributions[3], prefixes.distributions[3])

    def test_disabled_cap_preserves_baseline_weights(self):
        result = self.run_small(cap=None, moves=0)
        edge = graph_from_edges(3, directed=[(0, 1)])
        self.assertAlmostEqual(result.distributions[1][edge], .95)
        self.assertEqual(result.steps[0].reported_confidence, .95)
        self.assertEqual(result.steps[-1].reported_confidence, 1.)
        self.assertEqual(result.steps[-1].effective_confidence, .99)

    def consistency_record(self, count=8, tie=False):
        task = next(t for t in study.original.tasks() if t["strategy"] == "consistency")
        valid = count >= 6
        votes = ([1]*(count//2)+[0]*(count//2) if tie else [1]*count) + [None]*(8-count)
        confidence = .5 if tie else min(.8, max(.5, (count+.5)/9))
        trace = []
        for slot, q in enumerate(active.dag_query_pool(20, study.original.GROUPS[task["group"]])[:100], 1):
            trace.append(dict(query=asdict(q), budget=slot, cumulative_query_cost=slot,
                valid_response=valid, answer=int(not tie) if valid else None,
                reported_confidence=confidence if valid else None,
                effective_confidence=confidence if valid else None, treatment="soft_evidence",
                selection_effective_confidence=.9, support_recovered=False, resampled=False,
                ess_before=384., ess_after=384., rejuvenation_attempted_moves=768 if valid and slot==100 else 0,
                response_audit=dict(votes=votes, request_keys=list("abcdefgh"), valid_response_count=count,
                                   minimum_valid_response_count=6)))
        return dict(task=task,manifest_sha256="m",initialization_sha256="p",budget_unit="attempted",
            trace=trace, attempted_queries=100, answered_queries=100 if valid else 0,
            unused_query_budget=0, early_stop_reason=None, distribution=[dict(weight=1.)])

    def test_consistency_validation_accepts_capped_votes_and_preserves_threshold_ties(self):
        for count, tie in ((5, False), (6, False), (8, False), (6, True), (8, True)):
            record = self.consistency_record(count, tie)
            study.validate_core(record, record["task"], "m", "p")
        record = self.consistency_record()
        record["trace"][0]["reported_confidence"] = .7
        record["trace"][0]["effective_confidence"] = .7
        with self.assertRaisesRegex(ValueError, "consistency answer or confidence"):
            study.validate_core(record, record["task"], "m", "p")

    def test_raw_and_effective_calibration_are_separate_and_missing_stays_missing(self):
        result = study.calibration([(1,.95),(0,.95)], [(1,.8),(0,.8)])
        self.assertAlmostEqual(result["raw_brier"], .4525)
        self.assertAlmostEqual(result["effective_brier"], .34)
        self.assertEqual(result["accuracy"], .5)
        empty = study.calibration([], [])
        self.assertIsNone(empty["accuracy"])
        self.assertIsNone(empty["raw_brier"])
        self.assertIsNone(empty["effective_brier"])

    def test_paired_cap_comparisons_match_seeds_and_direction(self):
        left = {s: {m: s+.2 for m in study.CONTRAST_METRICS} for s in reversed(range(10))}
        right = {s: {m: s for m in study.CONTRAST_METRICS} for s in range(10)}
        left[0]["raw_brier"] = None
        for row in study.contrast(left, right, dict(comparison="cap_08_minus_099")):
            self.assertAlmostEqual(row["mean"], .2)
            self.assertAlmostEqual(row["low"], .2)
            self.assertAlmostEqual(row["high"], .2)
            self.assertEqual(row["paired_seeds"], 9 if row["metric"] == "raw_brier" else 10)


if __name__ == "__main__":
    unittest.main()
