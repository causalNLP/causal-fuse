import copy
from itertools import product
import unittest

import numpy as np

from cd_llm.core.finite_support import graph_from_edges, resolve_evidence_confidence
from cd_llm.experiments.synthetic_pc_grid_ablation import (
    ACCURACIES, BUDGETS, DIAGNOSTICS, GROUPS, METRICS, SCMS, SIZES,
    Expert, pc_grid_distribution, summarize, tasks,
)
from cd_llm.experiments.claim_fuse_smc_eig import _initial_particle_population
from cd_llm.methods.active_query import ClaimQuery, active_query_smc, dag_query_pool
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class PCGridAblationTests(unittest.TestCase):
    def test_archived_posterior_is_not_loaded_and_duplicates_combine(self):
        payload = {"protocol": {"node_count": 20, "pc_alphas": [0.01, .05, .1]},
                   "samples": {"200": {"methods": {"CEG-Fuse (PC grid)": {
                       "feasible": True, "graphs": [{"directed": [[0, 1]]}, {"directed": []},
                                                       {"directed": [[0, 1]]}],
                       "probabilities": [.01, .98, .01],
                       "diagnostics": {"data_probabilities": [.3, .5, .2]}}}}}}
        distribution = pc_grid_distribution(payload, 200)
        self.assertEqual(distribution[graph_from_edges(20, [(0, 1)])], .5)
        first = _initial_particle_population(distribution, 71)
        second = _initial_particle_population(distribution, 71)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        bad = copy.deepcopy(payload)
        del bad["samples"]["200"]["methods"]["CEG-Fuse (PC grid)"]["diagnostics"]["data_probabilities"]
        with self.assertRaises(KeyError):
            pc_grid_distribution(bad, 200)

    def test_pools_and_task_grid_are_matched(self):
        self.assertEqual(len(tasks()), 180)
        self.assertEqual(sum(t["graph_index"] == 47 for t in tasks()), 18)
        pools = {group: dag_query_pool(20, families) for group, families in GROUPS.items()}
        self.assertEqual({g: len(p) for g, p in pools.items()}, {"group_i": 570, "group_ii": 380, "all": 950})
        self.assertEqual(set(pools["all"]), set(pools["group_i"]) | set(pools["group_ii"]))
        self.assertFalse(any(q.kind == "ordering" for q in pools["all"]))

    def test_answers_are_shared_nested_and_one_remains_soft(self):
        truth = graph_from_edges(3, [(0, 1), (1, 2)])
        for query in dag_query_pool(3, GROUPS["all"]):
            sources = [Expert(truth, 1, 200, a) for a in ACCURACIES]
            responses = [source(query) for source in sources]
            self.assertEqual(len({s.records[0]["uniform"] for s in sources}), 1)
            correctness = [s.records[0]["correct"] for s in sources]
            self.assertEqual(correctness, sorted(correctness))
            self.assertTrue(correctness[-1])
            effective = [resolve_evidence_confidence(r.reported_confidence, r.treatment).effective_confidence
                         for r in responses]
            self.assertEqual(effective, [.6, .8, .99])

    def test_soft_smc_records_checkpoints_without_hard_projection(self):
        data = np.random.default_rng(3).normal(size=(20, 4))
        truth = graph_from_edges(4, [(0, 1), (1, 2)])
        initial = graph_from_edges(4, [])
        expert = Expert(truth, 1, 200, 1.0)
        result = active_query_smc(None, GaussianBICScore(data, max_parents=3), expert,
                                  query_pool=dag_query_pool(4, GROUPS["all"]), budgets=(1, 3, 5),
                                  confidence=1.0, treatment="soft_evidence", seed=11,
                                  initial_particles=(initial,) * 8, initial_weights=np.full(8, 1/8),
                                  rejuvenation_steps=2, rejuvenate_after_every_query=False)
        self.assertEqual(set(result.distributions), {1, 3, 5})
        self.assertEqual(result.support_recovery_count, 0)
        for step in result.steps:
            self.assertEqual(step.effective_confidence, .99)
            self.assertEqual(step.rejuvenation_hard_constraint_rejections, 0)
        self.assertEqual(result.steps[-1].rejuvenation_attempted_moves, 16)
        for distribution in result.distributions.values():
            self.assertAlmostEqual(sum(distribution.values()), 1)

    def test_aggregation_rejects_missing_or_duplicate_scms(self):
        rows = [dict(sample_size=n, accuracy=a, group=group, budget=b, graph_index=g,
                     **{m: float(g) for m in (*METRICS, *DIAGNOSTICS)})
                for n, a, group, b, g in product(SIZES, ACCURACIES, GROUPS, BUDGETS, SCMS)]
        controls = [dict(sample_size=n, graph_index=g, **{m: float(g) for m in METRICS})
                    for n, g in product(SIZES, SCMS)]
        with self.assertRaises(ValueError):
            summarize(rows[:-1], controls)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], controls)
        summary, contrasts = summarize(rows, controls)
        self.assertEqual(len(summary), 54)
        self.assertEqual(len(contrasts), 432)
        self.assertTrue(all(c["mean_difference"] == c["ci_lower"] == c["ci_upper"] == 0 for c in contrasts))


if __name__ == "__main__":
    unittest.main()
