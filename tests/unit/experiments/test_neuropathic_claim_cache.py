import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.local_vllm_backend import LocalVLLMConfig
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.experiments import neuropathic_claim_cache as study
from cd_llm.experiments.common.artifacts import write_json
from cd_llm.methods.active_query import ClaimQuery
from cd_llm.methods.llm_claim_confidence import CachedLocalChatClient, QwenClaimConfidenceProvider


class FakeClient:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        content = next(self.contents)
        self.calls += 1
        if isinstance(content, Exception):
            raise content
        payload = dict(id=f"r{self.calls}", model=kwargs["model"],
                       choices=[dict(message=dict(content=content))],
                       usage=dict(prompt_tokens=10, completion_tokens=4, total_tokens=14))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               id=payload["id"], model=kwargs["model"], model_dump=lambda: payload)


class ClaimCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = dict(nodes=[dict(id=f"V{i:02d}", description=f"clinical node {i}") for i in range(20)],
                             domain_context="neuropathic pain diagnosis")
        self.query = ClaimQuery("directed", 0, 1, "directed-0-1")

    def tearDown(self):
        self.temp.cleanup()

    def expert(self, cache, client, strategy, minimum=5):
        config = LocalVLLMConfig(seed=123, max_attempts=1)
        delegate = CachedLocalChatClient(cache, config, client=client)
        chat = study.HistoricalChat("qwen", config, cache, delegate)
        return QwenClaimConfidenceProvider(strategy, [n["id"] for n in self.metadata["nodes"]], chat,
            descriptions={n["id"]: n["description"] for n in self.metadata["nodes"]},
            domain_context=self.metadata["domain_context"], minimum_valid_consistency_responses=minimum)

    def test_question_pool(self):
        queries = study.questions()
        self.assertEqual(len(queries), 950)
        self.assertEqual(len(set(queries)), 950)
        self.assertEqual(sum(q.kind == "directed" for q in queries), 380)
        self.assertEqual(sum(q.kind == "adjacency" for q in queries), 190)
        self.assertTrue(all(q.kind == "ancestry" for q in queries[570:]))

    def test_stated_retry_and_resume_preserve_raw_content(self):
        path = self.root / "cache.sqlite3"
        cache = ResponseCache(path)
        client = FakeClient(['{"answer":"abstain"}', '{"answer":"yes","confidence":0.8}'])
        expert = self.expert(cache, client, "stated")
        first = expert(self.query)
        self.assertEqual(first.confidence, 0.8)
        self.assertEqual(client.calls, 2)
        cached = expert(self.query)
        self.assertEqual(client.calls, 2)
        self.assertEqual(study.parsed_response(self.query, "stated", first),
                         study.parsed_response(self.query, "stated", cached))
        snapshot = study.sqlite_snapshot(path)
        cache.close()
        offline, _ = study.provider(self.metadata, snapshot, "stated")
        self.assertEqual(offline(self.query).confidence, 0.8)
        self.assertEqual(snapshot, study.merge_records(snapshot, copy.deepcopy(snapshot)))
        conflict = copy.deepcopy(snapshot)
        next(iter(conflict.values()))["record"]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            study.merge_records(snapshot, conflict)

    def test_partial_consistency_transport_failure_then_resume(self):
        path = self.root / "cache.sqlite3"
        cache = ResponseCache(path)
        first = FakeClient(['{"answer":"yes"}'] * 3 + [RuntimeError("outage")])
        with self.assertRaisesRegex(RuntimeError, "failed after"):
            self.expert(cache, first, "consistency")(self.query)
        self.assertEqual(len(study.sqlite_snapshot(path)), 3)
        second = FakeClient(['{"answer":"yes"}'] * 2 + ['{"answer":"abstain"}'] * 3)
        completed = self.expert(cache, second, "consistency")(self.query)
        self.assertEqual(second.calls, 5)
        self.assertTrue(completed.valid)
        self.assertAlmostEqual(completed.confidence, 5.5 / 9)
        self.assertEqual(len(completed.audit["votes"]), 8)
        snapshot = study.sqlite_snapshot(path)
        cache.close()
        self.assertEqual(len(snapshot), 8)
        six, _ = study.provider(self.metadata, snapshot, "consistency", minimum_valid=6)
        self.assertFalse(six(self.query).valid)
        five, _ = study.provider(self.metadata, snapshot, "consistency", minimum_valid=5)
        self.assertTrue(five(self.query).valid)

    def test_abstention_is_complete_and_offline_replay_blocks_network(self):
        records, outcomes = {}, []
        for strategy, contents in (("stated", ['{"answer":"abstain"}'] * 2),
                                   ("consistency", ['{"answer":"abstain"}'] * 8)):
            path = self.root / f"{strategy}.sqlite3"
            cache = ResponseCache(path)
            response = self.expert(cache, FakeClient(contents), strategy)(self.query)
            outcomes.append(study.parsed_response(self.query, strategy, response))
            records[strategy] = study.sqlite_snapshot(path)
            cache.close()
        write_json(self.root / "inputs/model_metadata.json", self.metadata)
        with patch.object(study, "questions", return_value=(self.query,)):
            actual, six = study.replay(self.root, records, expected=outcomes)
        self.assertTrue(all(not r["valid"] for r in actual + six))
        self.assertEqual(len(actual), 2)

    def test_full_portable_export_and_relocated_offline_loader(self):
        """Exercise the actual archive path without a model or cluster files."""
        records, outcomes = {}, {}
        for strategy, count, content in (
            ("stated", 950, '{"answer":"yes","confidence":0.8}'),
            ("consistency", 7600, '{"answer":"yes"}'),
        ):
            path = self.root / f"{strategy}.sqlite3"
            cache = ResponseCache(path)
            expert = self.expert(cache, FakeClient([content] * count), strategy)
            for query in study.questions():
                outcomes[query, strategy] = study.parsed_response(query, strategy, expert(query))
            records[strategy] = study.sqlite_snapshot(path)
            cache.close()
            write_json(self.root / f"inputs/caches/qwen_{strategy}.json", records[strategy])
        manifest = dict(request_settings={s: study.request_settings(records[s]) for s in study.STRATEGIES},
                        imported={s: dict(raw_requests=len(records[s])) for s in study.STRATEGIES})
        write_json(self.root / "manifest.json", manifest)
        write_json(self.root / "questions.json", [study.asdict(q) for q in study.questions()])
        write_json(self.root / "inputs/model_metadata.json", self.metadata)
        write_json(self.root / "inputs/runtime.json", {})
        write_json(self.root / "responses.json", [outcomes[q, s] for q in study.questions() for s in study.STRATEGIES])
        write_json(self.root / "status.json", dict(elapsed_seconds=0, new_requests=dict(
            request_count=0, prompt_tokens=0, completion_tokens=0)))
        with patch.object(study, "verify", return_value=manifest):
            study.export(self.root)
        completion = json.loads((self.root / "completion.json").read_text())
        self.assertTrue(completion["complete"])
        self.assertEqual(completion["relocated_audit"]["offline_outcomes"], 1900)
        self.assertEqual(completion["relocated_audit"]["six_response_outcomes"], 950)
        loader = study.load_offline_provider(self.root / "portable_cache", "stated")
        self.assertEqual(loader(self.query).confidence, 0.8)
        with self.assertRaisesRegex(ValueError, "outside"):
            loader(ClaimQuery("ordering", 0, 1, "ordering-0-1"))


if __name__ == "__main__":
    unittest.main()
