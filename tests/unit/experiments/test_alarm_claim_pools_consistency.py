import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from cd_llm.experiments import alarm_claim_pools_consistency as study
from cd_llm.experiments import alarm_claim_pools_consistency_transport as transport


def decision(answer=None, status="valid", key="key"):
    return dict(answer=answer, reported_confidence=None, justification="x", status=status,
                finish_reason="stop", request_key=key)


def row(decisions):
    answer, confidence = study.prompts.local.pool_votes(decisions)
    return dict(
        query=dict(kind="ancestry", source=0, target=1, query_id="q"),
        strategy="consistency", condition="new", answer=answer,
        reported_confidence=confidence, valid=answer is not None,
        abstention_reason=None if answer is not None else "fewer-than-six-valid-variants",
        decisions=decisions, request_keys=[item["request_key"] for item in decisions],
        prompt_tokens=800, completion_tokens=80, cost_usd=0.0,
    )


class Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = dict(
            id="response", model=study.MODEL,
            choices=[dict(message=dict(content='{"answer":"yes","justification":"x"}'),
                          finish_reason="stop")],
            usage=dict(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        )
        return SimpleNamespace(model_dump=lambda: response)

    async def close(self):
        pass


class ConsistencyRuleTests(unittest.TestCase):
    def test_fewer_than_six_valid_votes_abstains(self):
        decisions = [decision(1, key=str(index)) for index in range(5)]
        decisions += [decision(None, "abstain", str(index)) for index in range(5, 8)]
        value = study.validate_consistency_row(row(decisions))
        self.assertIsNone(value["answer"])
        self.assertEqual(value["abstention_reason"], "fewer-than-six-valid-variants")

    def test_six_valid_votes_answer_and_exact_tie_is_no(self):
        majority = [decision(1, key=str(index)) for index in range(4)]
        majority += [decision(0, key=str(index)) for index in range(4, 6)]
        majority += [decision(None, "malformed", str(index)) for index in range(6, 8)]
        value = study.validate_consistency_row(row(majority))
        self.assertEqual(value["answer"], 1)
        self.assertEqual(value["reported_confidence"], 0.5)

        tie = [decision(1, key=str(index)) for index in range(3)]
        tie += [decision(0, key=str(index)) for index in range(3, 6)]
        tie += [decision(None, "truncated", str(index)) for index in range(6, 8)]
        value = study.validate_consistency_row(row(tie))
        self.assertEqual((value["answer"], value["reported_confidence"]), (0, 0.5))

    def test_full_vote_confidence_and_cap(self):
        decisions = [decision(1, key=str(index)) for index in range(8)]
        value = study.validate_consistency_row(row(decisions))
        self.assertAlmostEqual(value["reported_confidence"], 8.5 / 9)
        manifest = dict(runtime=dict(model_revision="revision"), runtime_fingerprint="runtime")
        response = study.response_for(value, manifest)
        self.assertEqual(response.reported_confidence, 0.8)
        self.assertAlmostEqual(response.audit["raw_reported_confidence"], 8.5 / 9)
        self.assertEqual(response.audit["minimum_valid_response_count"], 6)


class GridTests(unittest.TestCase):
    def test_task_grid_and_shards_are_exact(self):
        self.assertEqual(len(study.tasks()), 60)
        self.assertEqual({task["strategy"] for task in study.tasks()}, {"consistency"})
        self.assertEqual({task["group"] for task in study.tasks()}, set(study.GROUPS))
        pilots = study.pilot_queries()
        shards = [query for index in range(study.SHARDS) for query in study.shard_queries(index)]
        self.assertEqual(len(pilots), 6)
        self.assertEqual(len(shards), study.EXPECTED_OUTCOMES - 6)
        self.assertEqual(len({query.query_id for query in pilots + shards}), study.EXPECTED_OUTCOMES)
        self.assertEqual({query.kind for query in pilots}, {"directed", "adjacency", "ancestry"})

    def test_inference_configuration_is_unchanged(self):
        arguments = study.arguments(1000, 200, (), ())
        self.assertEqual(arguments["budgets"], (10, 20, 50, 100))
        self.assertEqual(arguments["budget_unit"], "attempted")
        self.assertEqual((arguments["confidence"], arguments["soft_confidence_cap"]), (0.9, 0.99))
        self.assertFalse(arguments["rejuvenate_after_every_query"])


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_consistency_only_transport_and_cache_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = study.ResponseCache(Path(temporary) / "cache.sqlite3")
            manifest = dict(
                settings=study.settings(), runtime=dict(model_revision="revision"),
                runtime_fingerprint="runtime",
            )
            client = Client()
            chat = transport.LocalChat(
                cache, manifest, "http://127.0.0.1:22000/v1", client=client)
            messages = [dict(role="user", content="question")]
            first = await chat.request(
                messages, strategy="consistency", condition="new", purpose="test")
            second = await chat.request(
                messages, strategy="consistency", condition="new", purpose="test")
            self.assertEqual(first, second)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.calls[0]["temperature"], 0.0)
            self.assertEqual(client.calls[0]["extra_body"],
                             dict(chat_template_kwargs=dict(enable_thinking=False)))
            with self.assertRaises(ValueError):
                await chat.request(messages, strategy="stated", condition="new", purpose="test")
            snapshot = study.sqlite_snapshot(Path(temporary) / "cache.sqlite3")
            replay = await transport.OfflineChat(snapshot, manifest).request(
                messages, strategy="consistency", condition="new", purpose="test")
            self.assertEqual(first, replay)
            await chat.close()
            cache.close()


if __name__ == "__main__":
    unittest.main()
