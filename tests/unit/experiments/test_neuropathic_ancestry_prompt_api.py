import asyncio
import copy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache, _request_key
from cd_llm.experiments import neuropathic_ancestry_prompt_api as api
from cd_llm.experiments import neuropathic_ancestry_prompt_ablation as local
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot


def manifest():
    return dict(settings=copy.deepcopy(api.SETTINGS), seeds=dict(stated=19, consistency=28),
                pricing=dict(prompt="0.0000001", completion="0.00000015"), cost_limit_usd=1.0)


class Client:
    def __init__(self, outputs=None):
        self.outputs = iter(outputs) if outputs is not None else None
        self.calls, self.active, self.peak = [], 0, 0
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.peak = max(self.active, self.peak)
        await asyncio.sleep(0)
        self.active -= 1
        value = next(self.outputs) if self.outputs is not None else '{"answer":"yes","confidence":0.8}'
        if isinstance(value, Exception):
            raise value
        response = dict(id=f"test-{len(self.calls)}", provider="DeepInfra", model=api.MODEL,
            choices=[dict(message=dict(content=value), finish_reason="stop")],
            usage=dict(prompt_tokens=100, completion_tokens=10, cost=0.00002,
                       completion_tokens_details=dict(reasoning_tokens=0)))
        return SimpleNamespace(model_dump=lambda: response)

    async def close(self):
        pass


class APITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "cache.sqlite3"
        self.cache = ResponseCache(self.path)
        self.manifest = manifest()
        self.query = local.questions()[0]
        self.inputs = dict(metadata=dict(domain_context="a clinical diagnostic system",
            nodes=[dict(id=f"V{i+1:02d}", description=f"description {i}") for i in range(20)]),
            templates=local.prompt_templates(), old={})
        for s in local.STRATEGIES:
            self.inputs["old"][s] = {self.query.query_id: [
                [dict(role="system", content="old system"), dict(role="user", content=f"old {s} question {i}")]
                for i in range(1 if s == "stated" else 8)]}

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    async def test_exact_routing_cache_separation_and_resume(self):
        client = Client()
        chat = api.APIChat(self.cache, self.manifest, client=client)
        prompt = [dict(role="user", content="same message")]
        kwargs = dict(strategy="stated", condition="old", purpose="test")
        first = await chat.request(prompt, **kwargs)
        self.assertEqual(first, await chat.request(prompt, **kwargs))
        self.assertEqual(len(client.calls), 1)
        await chat.request(prompt, **dict(kwargs, condition="new"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["extra_body"], dict(provider=api.SETTINGS["provider"], reasoning=dict(enabled=False)))
        self.assertEqual(client.calls[0]["temperature"], 0)
        self.assertEqual(client.calls[0]["max_tokens"], 256)
        self.assertEqual(chat.accounting()["attempts"], 2)
        self.assertAlmostEqual(chat.accounting()["accounted_cost_usd"], 0.00004)
        resumed = api.APIChat(self.cache, self.manifest, client=Client([]))
        self.assertEqual(await resumed.request(prompt, **kwargs), first)

    async def test_four_request_concurrency_limit(self):
        client = Client()
        chat = api.APIChat(self.cache, self.manifest, client=client)
        await asyncio.gather(*(chat.request([dict(role="user", content=str(i))],
            strategy="stated", condition="new", purpose=str(i)) for i in range(12)))
        self.assertEqual(client.peak, 4)

    async def test_budget_prevents_calls_and_tracks_uncertain_reservation(self):
        client = Client()
        chat = api.APIChat(self.cache, self.manifest, client=client)
        prompt = [dict(role="user", content="hello")]
        request = api.request_record(prompt, "stated", "new", self.manifest)
        chat.reserve(request)
        self.assertGreater(chat.accounting()["unresolved_reservations_usd"], 0)
        self.manifest["cost_limit_usd"] = 0.0000001
        with self.assertRaisesRegex(RuntimeError, "spending limit"):
            await chat.request(prompt, strategy="stated", condition="new", purpose="test")
        self.assertEqual(client.calls, [])
        restored = api.APIChat(self.cache, self.manifest, client=client)
        self.assertEqual(restored.accounting()["attempts"], 1)

    async def test_retries_account_each_attempt(self):
        chat = api.APIChat(self.cache, self.manifest, client=Client([RuntimeError("outage"), '{"answer":"abstain"}']))
        async def no_sleep(_):
            pass
        with patch.object(api.asyncio, "sleep", no_sleep):
            row = await chat.request([dict(role="user", content="test")], strategy="stated", condition="old", purpose="test")
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(chat.accounting()["attempts"], 2)
        self.assertGreater(chat.accounting()["unresolved_reservations_usd"], 0)

    async def test_stated_abstains_immediately_for_both_conditions(self):
        for condition in api.CONDITIONS:
            client = Client(['{"answer":"abstain","confidence":null}'])
            chat = api.APIChat(self.cache, self.manifest, client=client)
            row = await api.elicit(self.query, "stated", condition, chat, self.inputs)
            self.assertIsNone(row["answer"])
            self.assertEqual(len(client.calls), 1)

    async def test_invalid_stated_retries_and_consistency_threshold(self):
        client = Client(['bad', '{"answer":"no","confidence":0.8}'])
        chat = api.APIChat(self.cache, self.manifest, client=client)
        row = await api.elicit(self.query, "stated", "new", chat, self.inputs)
        self.assertEqual(row["answer"], 0)
        self.assertIn("previous response was malformed", client.calls[1]["messages"][-1]["content"])
        for condition, count in (("old", 5), ("new", 6)):
            client = Client(['{"answer":"yes"}'] * count + ['{"answer":"abstain"}'] * (8 - count))
            chat = api.APIChat(self.cache, self.manifest, client=client)
            row = await api.elicit(self.query, "consistency", condition, chat, self.inputs)
            self.assertEqual(len(client.calls), 8)
            self.assertEqual(row["answer"], 1 if count == 6 else None)

    async def test_offline_replay_without_api_key_or_network(self):
        chat = api.APIChat(self.cache, self.manifest, client=Client())
        row = await api.elicit(self.query, "consistency", "new", chat, self.inputs)
        offline = api.OfflineChat(sqlite_snapshot(self.path), self.manifest)
        with patch.dict("os.environ", {}, clear=True), patch.object(socket, "socket", side_effect=AssertionError("network")):
            replay = await api.elicit(self.query, "consistency", "new", offline, self.inputs)
        self.assertEqual(replay, row)

    def test_response_identity_and_reasoning_gate(self):
        response = dict(provider="DeepInfra", model=api.MODEL, choices=[dict(message=dict(content="yes"))])
        api.validate_response(response)
        for invalid in (dict(response, provider="Other"), dict(response, model="other"),
                        dict(response, choices=[dict(message=dict(reasoning="thinking"))]),
                        dict(response, usage=dict(completion_tokens_details=dict(reasoning_tokens=5)))):
            with self.assertRaises(RuntimeError):
                api.validate_response(invalid)

    def test_seed_and_provider_are_in_cache_identity(self):
        messages = [dict(role="user", content="test")]
        before = api.request_record(messages, "stated", "new", self.manifest)
        self.manifest["seeds"]["stated"] += 1
        after = api.request_record(messages, "stated", "new", self.manifest)
        self.assertNotEqual(_request_key(before), _request_key(after))
        changed = copy.deepcopy(self.manifest)
        changed["settings"]["provider"]["only"] = ["other"]
        self.assertNotEqual(_request_key(after), _request_key(api.request_record(messages, "stated", "new", changed)))


if __name__ == "__main__":
    unittest.main()
