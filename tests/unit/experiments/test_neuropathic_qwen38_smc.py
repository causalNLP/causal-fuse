import asyncio
import copy
from dataclasses import asdict
import json
import socket
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.experiments import neuropathic_qwen38_smc as study
from cd_llm.experiments import neuropathic_new_prompt_cache as cache
from cd_llm.experiments import neuropathic_ancestry_prompt_api as api
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache, _request_key
from cd_llm.core.finite_support import graph_from_edges, distribution_metrics
from cd_llm.methods import active_query as active
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class Client:
    def __init__(self, outputs=None):
        self.outputs = iter(outputs or ['{"answer":"yes","confidence":1.0}'])
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        response = dict(id=str(len(self.calls)), provider="DeepInfra", model=kwargs["model"],
            choices=[dict(message=dict(content=next(self.outputs)), finish_reason="stop")],
            usage=dict(prompt_tokens=100, completion_tokens=10, cost=.00004,
                       completion_tokens_details=dict(reasoning_tokens=0)))
        return SimpleNamespace(model_dump=lambda: response)

    async def close(self):
        pass


class Qwen38Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "responses.sqlite3"
        self.db = ResponseCache(self.path)
        self.manifest = dict(settings=study.settings(), seeds=dict(stated=19, consistency=28),
            response_provider="DeepInfra", require_token_usage=True, cost_limit_usd=10.,
            pricing=dict(prompt="0.00000015", completion="0.000001875"))
        self.inputs = dict(metadata=dict(domain_context="clinical context",
            nodes=[dict(id=f"V{i:02}", description=f"description {i}") for i in range(20)]),
            templates=cache.prompt_templates())

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    async def test_fresh_model_cache_and_offline_replay(self):
        query = next(q for q, s in study.tasks() if q.kind == "ancestry")
        client = Client()
        chat = cache.APIChat(self.db, self.manifest, client=client)
        row = await cache.elicit(query, "stated", chat, self.inputs)
        self.assertEqual(client.calls[0]["model"], study.MODEL)
        self.assertEqual(cache.response_for(row, self.manifest).audit["model"], study.MODEL)
        self.assertEqual(cache.response_for(row, self.manifest).audit["source"], "fresh-model-cache")
        self.assertEqual(row["reported_confidence"], 1.)
        with study.original.offline_guard():
            replay = await cache.elicit(query, "stated", api.OfflineChat(study.sqlite_snapshot(self.path), self.manifest), self.inputs)
        self.assertEqual(row, replay)
        self.assertEqual(len(client.calls), 1)

    def test_model_and_provider_isolate_cache_keys(self):
        message = [{"role":"user", "content":"same question"}]
        def key(manifest):
            return _request_key(api.request_record(message, "stated", "new", manifest))
        original = key(self.manifest)
        for field, value in (("model", api.MODEL), ("provider", dict(only=["parasail/fp8"]))):
            other = copy.deepcopy(self.manifest)
            other["settings"][field] = value
            self.assertNotEqual(original, key(other))

    def test_provider_model_reasoning_and_usage_validation(self):
        response = dict(provider="DeepInfra", model=study.MODEL,
            choices=[dict(message=dict(content="yes"))], usage=dict(prompt_tokens=10, completion_tokens=5))
        api.validate_response(response, self.manifest)
        for change in (dict(model=api.MODEL), dict(provider="Other"), dict(usage={}),
                       dict(choices=[dict(message=dict(reasoning="hidden"))])):
            with self.assertRaises(RuntimeError):
                api.validate_response(dict(response, **change), self.manifest)

    async def test_ten_dollar_limit_counts_pending_reservations(self):
        client = Client()
        chat = cache.APIChat(self.db, self.manifest, client=client)
        self.db.connection.execute("INSERT INTO api_attempts(request_key,reserved_usd,state) VALUES ('unknown',9.99999,'uncertain')")
        self.db.connection.commit()
        with self.assertRaisesRegex(RuntimeError, r"\$10 run spending limit"):
            await chat.request([dict(role="user", content="a question")], strategy="stated", condition="new", purpose="test")
        self.assertEqual(client.calls, [])

    async def test_consistency_six_valid_and_full_token_cost(self):
        queries = [q for q, s in study.tasks() if s == "consistency"]
        for query, count in zip(queries, (5,6,8)):
            outputs = ['{"answer":"yes"}']*count + ['{"answer":"abstain"}']*(8-count)
            row = await cache.elicit(query, "consistency", cache.APIChat(self.db, self.manifest, client=Client(outputs)), self.inputs)
            self.assertEqual(row["answer"], 1 if count >= 6 else None)
            self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (800,80))
            self.assertEqual(len(row["request_keys"]), 8)
            if count >= 6:
                self.assertEqual(row["reported_confidence"], (count+.5)/9)

    async def test_stated_repair_keeps_both_response_costs(self):
        query = study.tasks()[0][0]
        row = await cache.elicit(query, "stated", cache.APIChat(self.db, self.manifest,
            client=Client(['invalid', '{"answer":"yes","confidence":0.95}'])), self.inputs)
        self.assertEqual(row["reported_confidence"], .95)
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (200,20))
        self.assertEqual(len(row["request_keys"]), 2)

    def test_full_grid_and_unchanged_prompt_templates(self):
        tasks = study.tasks()
        self.assertEqual(len(tasks), 1900)
        self.assertEqual(sum(1 if s == "stated" else 8 for q,s in tasks), 8550)
        self.assertEqual({q.kind for q,s in tasks}, {"directed", "adjacency", "ancestry"})
        self.assertEqual(cache.prompt_templates(), study.read_json(study.SOURCE / "prompts.json"))
        self.assertEqual(len(study.original.tasks()), 120)

    def run_small(self, budgets=(1,2,3)):
        empty, edge = (graph_from_edges(3, directed=e) for e in ([], [(0,1)]))
        score = GaussianBICScore(np.random.default_rng(7).normal(size=(30,3)), max_parents=2)
        responses = iter(((1,1.),(None,None),(0,1.)))
        def provider(query):
            answer, confidence = next(responses)
            return active.QueryResponse(answer, confidence, treatment="soft_evidence")
        pool = tuple(active.ClaimQuery("directed", a,b,str(i)) for i,(a,b) in enumerate(((0,1),(1,0),(0,2))))
        return active.active_query_smc({empty:.5, edge:.5}, score, provider, query_pool=pool, budgets=budgets,
            budget_unit="attempted", confidence=.9, treatment="soft_evidence", soft_confidence_cap=.99,
            ess_threshold_fraction=.01, rejuvenation_steps=1, seed=31)

    def test_point_99_weighting_mh_and_abstention(self):
        with patch.object(active, "_rejuvenate", wraps=active._rejuvenate) as mutation:
            result = self.run_small()
        self.assertEqual([s.effective_confidence for s in result.steps], [.99,None,.99])
        self.assertTrue(all(s.selection_effective_confidence == .9 for s in result.steps))
        self.assertEqual(result.steps[0].reported_confidence, 1.)
        self.assertAlmostEqual(result.distributions[1][graph_from_edges(3,directed=[(0,1)])], .99)
        self.assertEqual(result.distributions[1], result.distributions[2])
        self.assertEqual((result.query_count,result.answered_query_count), (3,2))
        self.assertEqual(mutation.call_count, 1)
        evidence = mutation.call_args.args[3]
        self.assertEqual([n.reported_confidence for n in evidence.nodes], [1.,1.])
        args = mutation.call_args.args
        self.assertEqual(args[12], .99)
        proposal = active.EvidenceGuidedDAGProposal(args[2], evidence,
            node_count=args[0].node_count, soft_confidence_cap=args[12])
        self.assertEqual(proposal.effective_confidences, (.99,.99))

    def test_intermediate_checkpoints_do_not_change_inference(self):
        terminal, prefixes = self.run_small((3,)), self.run_small()
        self.assertEqual(terminal.distributions[3], prefixes.distributions[3])
        for left, right in zip(terminal.steps, prefixes.steps):
            a,b = asdict(left), asdict(right)
            a.pop("cumulative_seconds"); b.pop("cumulative_seconds")
            self.assertEqual(a,b)

    def test_ed_es_units_are_raw(self):
        empty, edge = (graph_from_edges(3, directed=e) for e in ([], [(0,1)]))
        raw = distribution_metrics((empty,), np.asarray([1.]), edge, "dag")
        old = distribution_metrics((empty,), np.asarray([1.]), edge, "dag", normalized=True)
        for name in ("expected_distance", "energy_score"):
            self.assertAlmostEqual(raw[name], old[name]*3)


class OfflineAuditLoopTests(unittest.TestCase):
    def test_sync_audit_initializes_loop_before_guard_and_replays_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"responses.sqlite3"
            db = ResponseCache(path)
            manifest = dict(settings=study.settings(), seeds=dict(stated=19, consistency=28),
                response_provider="DeepInfra", require_token_usage=True, cost_limit_usd=10.,
                pricing=dict(prompt="0.00000015", completion="0.000001875"))
            inputs = dict(metadata=dict(domain_context="clinical context",
                nodes=[dict(id=f"V{i:02}", description=f"description {i}") for i in range(20)]),
                templates=cache.prompt_templates())
            query = study.tasks()[0][0]
            try:
                row = asyncio.run(cache.elicit(query, "stated", cache.APIChat(db, manifest, client=Client()), inputs))
                raw = study.sqlite_snapshot(path)
                original_elicit = cache.elicit
                async def guarded_elicit(*args, **kwargs):
                    with self.assertRaisesRegex(AssertionError, "offline experiment"):
                        socket.socket()
                    return await original_elicit(*args, **kwargs)
                with patch.object(study, "tasks", return_value=[(query,"stated")]), \
                     patch.object(cache, "elicit", side_effect=guarded_elicit):
                    self.assertEqual(study.replay_cached_outcomes(raw, manifest, inputs), [row])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
