import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cd_llm.baselines.causal_order2025.openrouter_backend import _canonical_request, _request_key
from cd_llm.core.finite_support import graph_edges, graph_from_edges
from cd_llm.experiments.neuropathic_active_interpretation import HistoricalChat, LayeredCache, check_literal
from cd_llm.experiments.neuropathic_active_interpretation_report import interval, summarize_distribution


class InterpretationExperimentTests(unittest.TestCase):
    def test_cache_miss_fails_without_fabricating_abstention(self):
        config=SimpleNamespace(model="test",seed=0,reasoning_effort="none")
        chat=HistoricalChat("terra",config,LayeredCache({},None))
        with self.assertRaisesRegex(RuntimeError,"no abstention substituted"):
            chat.request([dict(role="user",content="question")],purpose="stated:q")

    def test_legacy_medical_wording_hits_original_cache(self):
        config=SimpleNamespace(model="test",seed=0,reasoning_effort="none")
        old=[dict(role="system",content="You are a careful medical causal expert.")]
        request=_canonical_request(old,"test",256,0,"none")
        record=dict(content='{"answer":"yes"}',cached=True)
        cache=LayeredCache({_request_key(request):dict(record=record)},None)
        actual=HistoricalChat("terra",config,cache).request(
            [dict(role="system",content="You are a careful causal domain expert.")],purpose="stated:q")
        self.assertEqual(actual,record)

    def test_historical_cost_does_not_enter_new_budget(self):
        class NewCache:
            def summary(self): return {"cost_usd":0.1}
        cache=LayeredCache({"old":dict(record={"cost_usd":99.0})},NewCache())
        self.assertEqual(cache.summary()["cost_usd"],0.1)

    def test_path_recovery_distinguishes_shortcut(self):
        path=graph_from_edges(20,directed=[(0,1),(1,2)])
        shortcut=graph_from_edges(20,directed=[(0,2)])
        entries=[dict(graph=graph_edges(path),weight=0.7),dict(graph=graph_edges(shortcut),weight=0.3)]
        metrics,rows=summarize_distribution(entries,path,[(0,2)])
        self.assertAlmostEqual(rows[0]["ancestry_mass"],1.0)
        self.assertAlmostEqual(rows[0]["indirect_path_mass"],0.7)
        self.assertAlmostEqual(rows[0]["shortcut_mass"],0.3)
        self.assertEqual(metrics["map_indirect_paths"],1)

    def test_literal_gate_rejects_changed_graph_mass(self):
        empty=graph_edges(graph_from_edges(20))
        edge=graph_edges(graph_from_edges(20,directed=[(0,1)]))
        source=dict(trace=[],distribution=[dict(graph=empty,weight=1.0)])
        result=dict(trace=[],distribution=[dict(graph=edge,weight=1.0)])
        with self.assertRaisesRegex(ValueError,"literal posterior differs"):
            check_literal(result,source)

    def test_paired_interval_uses_signed_differences(self):
        result=interval([-0.1]*10,4)
        self.assertAlmostEqual(result["mean"],-0.1)
        self.assertAlmostEqual(result["high"],-0.1)


if __name__ == "__main__":
    unittest.main()
