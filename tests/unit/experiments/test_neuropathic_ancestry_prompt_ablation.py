import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.local_vllm_backend import LocalVLLMConfig
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.experiments import neuropathic_ancestry_prompt_ablation as study
from cd_llm.experiments.neuropathic_active_interpretation import HistoricalChat, LayeredCache
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot
from cd_llm.methods.llm_claim_confidence import CachedLocalChatClient


class FakeChat:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def request(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        content = next(self.contents)
        if isinstance(content, Exception):
            raise content
        return dict(request_key=str(len(self.calls)), content=content, response={},
                    prompt_tokens=10, completion_tokens=5)


class FakeClient:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        content = next(self.contents)
        if isinstance(content, Exception):
            raise content
        payload = dict(id=f"fake-{self.calls}", model=kwargs["model"],
            choices=[dict(message=dict(content=content), finish_reason="stop")],
            usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        return SimpleNamespace(id=payload["id"], model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model_dump=lambda: payload)


class AncestryPromptTests(unittest.TestCase):
    def setUp(self):
        self.metadata = dict(domain_context="a clinical diagnostic system",
            nodes=[dict(id=f"V{i+1:02d}", description=f"clinical description {i}") for i in range(20)])
        self.query = study.questions()[0]
        self.templates = study.prompt_templates()

    def elicit(self, contents, strategy="stated"):
        chat = FakeChat(contents)
        result = study.elicit(self.query, strategy, chat, self.metadata, self.templates)
        return result, chat

    def test_pool_and_prompt_variants(self):
        queries = study.questions()
        self.assertEqual(len(queries), 380)
        self.assertEqual(len({(q.source, q.target) for q in queries}), 380)
        self.assertTrue(all(q.source != q.target and q.kind == "ancestry" for q in queries))
        prompts = [study.messages(self.query, "consistency", self.metadata, self.templates, variant=i)[1]["content"] for i in range(8)]
        self.assertEqual(len(set(prompts)), 8)
        for i, prompt in enumerate(prompts):
            self.assertIn("Do not invent intermediate mechanisms.", prompt)
            self.assertIn("Failure to recall a mechanism is not sufficient evidence for NO.", prompt)
            self.assertNotIn("{", prompt.replace('{"answer":"yes|no|abstain","justification":"..."}', ""))
            self.assertNotIn("clinical description 2\n", prompt)
            self.assertIn("yes, no, abstain" if i % 2 == 0 else "no, yes, abstain", prompt)

    def test_valid_abstention_does_not_retry(self):
        row, chat = self.elicit(['{"answer":"abstain","confidence":null,"justification":"Insufficient knowledge."}'])
        self.assertEqual(len(chat.calls), 1)
        self.assertIsNone(row["answer"])
        self.assertEqual(row["abstention_reason"], "abstain")

    def test_malformed_retries_once_and_preserves_audit(self):
        row, chat = self.elicit(['not JSON', '{"answer":"yes","confidence":0.8,"justification":"A mechanism exists."}'])
        self.assertEqual(row["answer"], 1)
        self.assertEqual([r["status"] for r in row["decisions"]], ["malformed", "valid"])
        self.assertIn("previous response was malformed", chat.calls[1][0][1]["content"])
        self.assertEqual(row["prompt_tokens"], 20)

    def test_invalid_confidence_retries_and_missing_justification_is_not_a_vote_filter(self):
        for confidence in (0.4, 1.1, True, None):
            row, chat = self.elicit([json.dumps(dict(answer="yes", confidence=confidence)), '{"answer":"no","confidence":0.7}'])
            self.assertEqual(len(chat.calls), 2)
            self.assertEqual(row["answer"], 0)
        row, chat = self.elicit(['bad', 'bad'])
        self.assertIsNone(row["answer"])
        self.assertEqual(row["abstention_reason"], "malformed")

    def test_truncation_is_distinct_from_explicit_abstention(self):
        record = dict(request_key="key", content='{"answer":"yes","confidence":0.9}',
                      response=dict(choices=[dict(finish_reason="length")]))
        parsed = study.parse_record(record, "stated")
        self.assertEqual(parsed["status"], "truncated")
        self.assertIsNone(parsed["answer"])

    def test_consistency_five_six_boundary_and_no_format_retry(self):
        for valid in (5, 6):
            row, chat = self.elicit(['{"answer":"yes"}'] * valid + ['{"answer":"abstain"}'] * (7 - valid) + ['bad'], "consistency")
            self.assertEqual(len(chat.calls), 8)
            self.assertEqual(row["valid_response_count"], valid)
            if valid == 5:
                self.assertIsNone(row["answer"])
            else:
                self.assertEqual(row["answer"], 1)
                self.assertAlmostEqual(row["reported_confidence"], 6.5 / 9)

    def test_consistency_ties_and_majority_confidence(self):
        for yes, no, abstain, answer, confidence in ((4, 4, 0, 0, 0.5), (3, 3, 2, 0, 0.5), (5, 2, 1, 1, 5.5 / 9)):
            row, _ = self.elicit(['{"answer":"yes"}'] * yes + ['{"answer":"no"}'] * no + ['{"answer":"abstain"}'] * abstain, "consistency")
            self.assertEqual(row["answer"], answer)
            self.assertAlmostEqual(row["reported_confidence"], confidence)

    def test_transport_error_is_not_an_abstention(self):
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            self.elicit([RuntimeError("service unavailable")])

    def test_partial_cache_resume_and_network_disabled_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.sqlite3"
            cache = ResponseCache(path)
            config = LocalVLLMConfig(seed=123, max_attempts=1)
            failed = FakeClient(['{"answer":"yes"}'] * 3 + [RuntimeError("outage")])
            first = CachedLocalChatClient(cache, config, client=failed)
            with self.assertRaises(RuntimeError):
                study.elicit(self.query, "consistency", first, self.metadata, self.templates)
            self.assertEqual(len(sqlite_snapshot(path)), 3)
            resumed = FakeClient(['{"answer":"yes"}'] * 3 + ['{"answer":"abstain"}'] * 2)
            second = CachedLocalChatClient(cache, config, client=resumed)
            row = study.elicit(self.query, "consistency", second, self.metadata, self.templates)
            self.assertEqual(resumed.calls, 5)
            raw = sqlite_snapshot(path)
            cache.close()
            offline = HistoricalChat("qwen", config, LayeredCache(raw, None))
            with patch.object(socket, "socket", side_effect=AssertionError("network disabled")):
                replayed = study.elicit(self.query, "consistency", offline, self.metadata, self.templates)
            self.assertEqual(row, replayed)

    def test_transitive_truth_and_cycle_rejection(self):
        labels = study.truth_labels(dict(graph=dict(directed=[[0, 1], [1, 2]])))
        self.assertEqual(labels["ancestry-0-2"], 1)
        self.assertEqual(labels["ancestry-2-0"], 0)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            study.truth_labels(dict(graph=dict(directed=[[0, 1], [1, 0]])))

    def test_metrics_keep_abstentions_in_population_denominators(self):
        answers, truths = [1, 0, None, 1, 0, None], [1, 1, 1, 0, 0, 0]
        rows = [dict(query=dict(query_id=str(i)), answer=a, reported_confidence=0.8 if a is not None else None)
                for i, a in enumerate(answers)]
        labels = {str(i): t for i, t in enumerate(truths)}
        m = study.metrics(rows, labels)
        self.assertEqual((m["TP"], m["FN"], m["FP"], m["TN"]), (1, 1, 1, 1))
        self.assertEqual(m["fpr_all_negatives"], 1 / 3)
        self.assertEqual(m["fpr_answered_negatives"], 0.5)
        self.assertEqual(m["recall"], 1 / 3)
        self.assertEqual(m["coverage"], 2 / 3)
        self.assertAlmostEqual(m["brier"], 0.34)
        for row in rows:
            row.update(answer=None, reported_confidence=None)
        m = study.metrics(rows, labels)
        self.assertEqual(m["recall"], 0)
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["brier"])


if __name__ == "__main__":
    unittest.main()
