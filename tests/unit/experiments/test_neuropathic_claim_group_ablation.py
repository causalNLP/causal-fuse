import copy
from dataclasses import asdict
import socket
import unittest

from cd_llm.core.finite_support import graph_edges, graph_from_edges
from cd_llm.experiments import neuropathic_claim_group_ablation as study
from cd_llm.methods.active_query import ClaimQuery, dag_query_pool


class ClaimGroupStudyTests(unittest.TestCase):
    def record(self, *, terminal_abstention=False):
        task = next(t for t in study.tasks() if t["strategy"] == "consistency" and t["group"] == "group_i")
        trace = []
        for slot, query in enumerate(dag_query_pool(20, ("local_edge",))[:100], 1):
            valid = not (terminal_abstention and slot == 100)
            count = 6 if valid else 5
            trace.append(dict(budget=slot, cumulative_query_cost=slot, query=asdict(query),
                valid_response=valid, answer=1 if valid else None,
                reported_confidence=6.5 / 9 if valid else None,
                effective_confidence=6.5 / 9 if valid else None, treatment="soft_evidence",
                selection_effective_confidence=0.9, support_recovered=False,
                ess_before=384., ess_after=384., resampled=False,
                rejuvenation_attempted_moves=768 if valid and slot == 100 else 0,
                rejuvenation_accepted_moves=0, predicate_probability=0.5, expected_information_gain=0.1,
                response_audit=dict(votes=[1] * count + [None] * (8 - count),
                    valid_response_count=count, minimum_valid_response_count=6, request_keys=list("abcdefgh"))))
        return dict(task=task, manifest_sha256="manifest", initialization_sha256="population",
            budget_unit="attempted", trace=trace, attempted_queries=100,
            answered_queries=99 if terminal_abstention else 100, unused_query_budget=0,
            early_stop_reason=None, distribution=[dict(graph=dict(directed=[]), weight=1.)])

    def validate(self, record):
        study.validate_record(record, record["task"], "manifest", "population")

    def test_full_grid_and_disjoint_canonical_pools(self):
        tasks = study.tasks()
        self.assertEqual(len(tasks), 120)
        self.assertEqual(len({t["id"] for t in tasks}), 120)
        self.assertEqual({t["seed"] for t in tasks[:12]}, {0})
        for block in range(20):
            self.assertEqual(sum(t["block"] == block for t in tasks), 6)
        pools = {g: set(dag_query_pool(20, families)) for g, families in study.GROUPS.items()}
        self.assertEqual({g: len(q) for g, q in pools.items()}, dict(group_i=570, group_ii=380, all=950))
        self.assertFalse(pools["group_i"] & pools["group_ii"])
        self.assertEqual(pools["all"], pools["group_i"] | pools["group_ii"])

    def test_six_valid_answers_accepted_but_five_rejected(self):
        record = self.record()
        self.validate(record)
        record["trace"][0]["response_audit"].update(votes=[1] * 5 + [None] * 3, valid_response_count=5)
        with self.assertRaisesRegex(ValueError, "six-response"):
            self.validate(record)

    def test_terminal_abstention_costs_one_without_mutation(self):
        record = self.record(terminal_abstention=True)
        self.validate(record)
        record["trace"][-1]["cumulative_query_cost"] = 99
        with self.assertRaisesRegex(ValueError, "attempt accounting"):
            self.validate(record)
        record["trace"][-1]["cumulative_query_cost"] = 100
        record["trace"][-1]["rejuvenation_attempted_moves"] = 768
        with self.assertRaisesRegex(ValueError, "abstention changed"):
            self.validate(record)

    def test_group_leakage_and_repeated_queries_are_rejected(self):
        record = self.record()
        record["trace"][0]["query"] = asdict(ClaimQuery("ancestry", 0, 1, "ancestry-0-1"))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.validate(record)
        record = self.record()
        record["trace"][0]["query"] = record["trace"][1]["query"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.validate(record)

    def test_ancestry_metrics_use_paths_and_exclude_abstentions(self):
        truth = graph_from_edges(20, directed=[(0, 1), (1, 2)])
        trace = []
        for kind, answer, confidence in (("ancestry", 1, 0.9), ("directed", 0, 0.8), ("adjacency", None, None)):
            trace.append(dict(query=asdict(ClaimQuery(kind, 0, 2, kind)), answer=answer,
                reported_confidence=confidence, valid_response=answer is not None, resampled=False,
                rejuvenation_attempted_moves=0, rejuvenation_accepted_moves=0,
                response_audit=dict(request_keys=["a"]), prompt_tokens=2, completion_tokens=1))
        metrics, responses = study.metrics_for(dict(trace=trace, elapsed_seconds=0,
            distribution=[dict(graph=graph_edges(truth), weight=1.)]), truth)
        self.assertEqual(responses, [(1, 0.9), (1, 0.8)])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["brier"], 0.025)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["abstentions"], 1)
        self.assertEqual(metrics["ancestry_queries"], 1)
        self.assertAlmostEqual(metrics["expected_distance"], 0)

    def test_reproduction_ignores_timing_but_detects_inference_changes(self):
        original = self.record()
        replay = copy.deepcopy(original)
        replay["trace"][0]["cumulative_seconds"] = 42
        study.assert_same_inference(original, replay)
        replay["trace"][0]["answer"] = 0
        with self.assertRaisesRegex(ValueError, "replay differs"):
            study.assert_same_inference(original, replay)

    def test_offline_guard_prevents_network_connections(self):
        with study.offline_guard(), self.assertRaisesRegex(AssertionError, "network access"):
            socket.socket()

    def test_paired_intervals_match_by_seed_and_have_explicit_direction(self):
        rows = []
        for task in reversed(study.tasks()):
            value = task["seed"] + {"group_i": 0, "group_ii": 0.2, "all": -0.1}[task["group"]]
            rows.append(dict(sample_size=task["sample_size"], strategy=task["strategy"],
                group=task["group"], seed=task["seed"],
                **{m: value for m in (*study.METRICS, "accuracy", "brier", "coverage")}))
        effects = study.paired_differences(rows)
        self.assertEqual(len(effects), 108)
        for effect in effects:
            if effect["left_group"] == "all" and effect["right_group"] == "group_i":
                for key in ("mean", "low", "high"):
                    self.assertAlmostEqual(effect[key], -0.1)


if __name__ == "__main__":
    unittest.main()
