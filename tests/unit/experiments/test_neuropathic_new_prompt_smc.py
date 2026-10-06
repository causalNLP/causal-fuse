import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.experiments import neuropathic_new_prompt_cache as cache
from cd_llm.experiments import neuropathic_new_prompt_smc as study
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot


class Client:
    def __init__(self, outputs=None):
        self.outputs = iter(outputs) if outputs is not None else None
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        value = next(self.outputs) if self.outputs is not None else '{"answer":"yes","confidence":0.8}'
        if isinstance(value, Exception):
            raise value
        response = dict(id=str(len(self.calls)), provider="DeepInfra", model=cache.api.MODEL,
            choices=[dict(message=dict(content=value), finish_reason="stop")],
            usage=dict(prompt_tokens=100, completion_tokens=10, cost=0.00002,
                       completion_tokens_details=dict(reasoning_tokens=0)))
        return SimpleNamespace(model_dump=lambda: response)

    async def close(self):
        pass


class RateLimit(Exception):
    status_code = 429
    def __init__(self, value=None):
        self.response = SimpleNamespace(headers={} if value is None else {"retry-after": value})


class NewPromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "responses.sqlite3"
        self.db = ResponseCache(self.path)
        self.manifest = dict(settings=copy.deepcopy(cache.api.SETTINGS), seeds=dict(stated=19, consistency=28),
            pricing=dict(prompt="0.0000001", completion="0.00000015"), cost_limit_usd=1.)
        self.inputs = dict(metadata=dict(domain_context="clinical context",
            nodes=[dict(id=f"X{i}", description=f"description {i}") for i in range(20)]),
            templates=cache.prompt_templates())
        self.q = cache.tasks()[0][0]
        self.now, self.delays = 0., []

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    async def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay

    def chat(self, client):
        return cache.APIChat(self.db, self.manifest, client=client, sleep=self.sleep, clock=lambda: self.now)

    def test_prompt_semantics_and_all_variants(self):
        queries = cache.dag_query_pool(20, cache.original.GROUPS["all"])
        for kind in ("directed", "adjacency", "ancestry"):
            query = next(q for q in queries if q.kind == kind)
            for variant in range(8):
                prompt = cache.messages(query, "consistency", self.inputs, variant=variant)
                text = prompt[-1]["content"]
                labels = "yes, no" if variant % 2 == 0 else "no, yes"
                self.assertIn("Allowed labels in this order: " + labels, text)
                self.assertNotIn("{", text.split("Return JSON only")[0])
                if kind == "ancestry":
                    self.assertEqual(prompt, cache.local.messages(query, "consistency", self.inputs["metadata"],
                        cache.local.prompt_templates(), variant=variant))
                else:
                    self.assertIn("intermediate variable alone does not establish\na direct causal edge", text)
                    self.assertNotIn("causal-path proposition", text)
                if kind == "adjacency":
                    self.assertIn("in at least one direction", text)
                    self.assertIn("in both directions", text)
                    self.assertNotIn("reverse causation", text)
                if kind == "directed":
                    self.assertIn("directly causes", text)
        self.assertEqual(len(cache.tasks()), 1140)
        self.assertEqual(len(queries), 950)
        self.assertEqual(len(cache.original.tasks()), 120)

    async def test_stated_abstain_immediate_and_offline_replay(self):
        client = Client(['{"answer":"abstain","confidence":null}'])
        row = await cache.elicit(self.q, "stated", self.chat(client), self.inputs)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(row["answer"])
        offline = cache.api.OfflineChat(sqlite_snapshot(self.path), self.manifest)
        with cache.original.offline_guard():
            self.assertEqual(row, await cache.elicit(self.q, "stated", offline, self.inputs))
        self.assertEqual(cache.response_for(row).treatment, "soft_evidence")

    async def test_malformed_retry_and_soft_one(self):
        client = Client(['bad', '{"answer":"yes","confidence":1}'])
        row = await cache.elicit(self.q, "stated", self.chat(client), self.inputs)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("previous response was malformed", client.calls[1]["messages"][-1]["content"])
        self.assertEqual(cache.response_for(row).reported_confidence, 1.)
        self.assertEqual(cache.response_for(row).treatment, "soft_evidence")

    async def test_five_six_valid_and_ties(self):
        queries = cache.dag_query_pool(20, ("local_edge",))
        for query, values, answer, confidence in zip(queries, ([1]*5, [1]*6, [1]*3+[0]*3),
                                                    (None, 1, 0), (None, 6.5/9, .5)):
            outputs = [json.dumps(dict(answer="yes" if v else "no")) for v in values]
            outputs += ['{"answer":"abstain"}'] * (8-len(values))
            row = await cache.elicit(query, "consistency", self.chat(Client(outputs)), self.inputs)
            self.assertEqual(len(row["decisions"]), 8)
            self.assertEqual(row["answer"], answer)
            self.assertEqual(row["reported_confidence"], confidence)
            self.assertEqual(cache.response_for(row).audit["valid_response_count"], len(values))

    async def test_retry_after_honored_and_cached_resume(self):
        client = Client([RateLimit("73"), '{"answer":"yes","confidence":0.8}'])
        chat = self.chat(client)
        row = await cache.elicit(self.q, "stated", chat, self.inputs)
        self.assertEqual(self.delays, [73.])
        self.assertEqual(chat.accounting()["attempts"], 2)
        self.assertGreater(chat.accounting()["unresolved_reservations_usd"], 0)
        self.assertEqual(row, await cache.elicit(self.q, "stated", self.chat(Client([])), self.inputs))

    async def test_shared_cooldown_backoff_and_twenty_attempt_limit(self):
        chat = self.chat(Client([RateLimit()] * 20))
        with self.assertRaisesRegex(RuntimeError, "cache incomplete"):
            await cache.elicit(self.q, "stated", chat, self.inputs)
        self.assertEqual(chat.accounting()["attempts"], 20)
        self.assertEqual(self.delays[:5], [30, 60, 120, 240, 300])
        self.assertEqual(self.delays[-1], 300)
        chat.defer(RateLimit("42"), 1)
        await chat.wait_cooldown()  # Any worker on this transport observes the same deadline.
        self.assertEqual(self.delays[-1], 42)
        self.assertEqual(sqlite_snapshot(self.path), {})

    async def test_cost_cap_blocks_request(self):
        self.manifest["cost_limit_usd"] = 1e-9
        client = Client()
        with self.assertRaisesRegex(RuntimeError, "spending limit"):
            await cache.elicit(self.q, "stated", self.chat(client), self.inputs)
        self.assertEqual(client.calls, [])

    def test_http_date_retry_after_and_invalid_value(self):
        self.assertEqual(cache.retry_after(RateLimit("Thu, 01 Jan 1970 00:01:00 GMT"), now=10), 50)
        self.assertIsNone(cache.retry_after(RateLimit("invalid")))
        self.assertIsNone(cache.retry_after(RateLimit("nan")))

    def test_missing_calibration_is_not_zero(self):
        left = {s: dict(expected_distance=2., adjacency_f1=.5, orientation_f1=.4,
                       accuracy=None, brier=float("nan"), coverage=0.) for s in range(10)}
        right = {s: dict(expected_distance=1., adjacency_f1=.5, orientation_f1=.4,
                        accuracy=.8, brier=.1, coverage=1.) for s in range(10)}
        rows = study.contrast(left, right, dict(budget=10, sample_size=200))
        for r in rows:
            if r["metric"] in ("accuracy", "brier"):
                self.assertIsNone(r["mean"])
                self.assertIsNone(r["low"])
                self.assertEqual(r["paired_seeds"], 0)
            elif r["metric"] == "expected_distance":
                self.assertEqual(r["mean"], 1.)


if __name__ == "__main__":
    unittest.main()
