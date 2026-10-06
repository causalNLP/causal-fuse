import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments.alarm_active import (
    MODEL, Provider, Transport, accounted, distribution_scores, fingerprint,
    messages_for, parse_response, request_for, reservation,
)
from cd_llm.experiments.common.artifacts import write_json
from cd_llm.methods.active_query import ClaimQuery, QueryResponse, active_query_smc, group_i_query_pool
from cd_llm.methods.fast_bdeu import FastBDeuLocalScore
from cd_llm.methods.graph_scores import BDeuScore


def response(answer="yes", confidence=.8, finish="stop"):
    return dict(model=MODEL, provider="DeepInfra", choices=[dict(finish_reason=finish,
        message=dict(content=json.dumps(dict(answer=answer, confidence=confidence))))],
        usage=dict(prompt_tokens=10, completion_tokens=5))


class FakeTransport:
    def __init__(self, responses):
        self.responses, self.messages = iter(responses), []

    def query(self, messages):
        self.messages.append(messages)
        return next(self.responses), f"requests/{len(self.messages)}.json"


class AlarmActiveTests(unittest.TestCase):
    def test_local_pool_and_prompt_semantics(self):
        pool = group_i_query_pool(37)
        self.assertEqual(sum(q.kind=="directed" for q in pool), 1332)
        self.assertEqual(sum(q.kind=="adjacency" for q in pool), 666)
        meta = dict(context="Clinical setting", nodes=[dict(id="V02", description="Exposure"), dict(id="V01", description="Measurement")])
        prompt = messages_for(ClaimQuery("directed", 0, 1, "x"), meta)
        self.assertTrue(prompt[1]["content"].endswith("Does V02 directly cause V01?"))
        self.assertIn("not mediated by another variable", prompt[0]["content"])
        self.assertIn("already know", prompt[0]["content"])
        self.assertIn("either direction", messages_for(ClaimQuery("adjacency", 0, 1, "y"), meta)[1]["content"])

    def test_parser_rejects_truncation_invalid_confidence_and_ambiguity(self):
        self.assertEqual(parse_response(response()).answer, 1)
        self.assertEqual(parse_response(response("no", 1)).treatment, "soft_evidence")
        self.assertFalse(parse_response(response("abstain", None)).valid)
        for confidence in (.49, 1.01, True, "0.8", float("nan")):
            with self.assertRaises((ValueError, TypeError)):
                parse_response(response(confidence=confidence))
        with self.assertRaises(ValueError):
            parse_response(response(finish="length"))
        ambiguous = response()
        ambiguous["choices"][0]["message"]["content"] *= 2
        with self.assertRaises(ValueError):
            parse_response(ambiguous)

    def test_abstention_uses_one_call_and_truncation_has_one_repair(self):
        meta = dict(context="Clinical setting", nodes=[dict(id="V02", description="Exposure"), dict(id="V01", description="Measurement")])
        query = ClaimQuery("directed", 0, 1, "q")
        with tempfile.TemporaryDirectory() as temporary:
            transport = FakeTransport([response("abstain", None)])
            provider = Provider(Path(temporary), 0, meta, transport)
            self.assertFalse(provider(query).valid)
            self.assertEqual(len(transport.messages), 1)
            transport = FakeTransport([response(finish="length"), response("no", .9)])
            provider = Provider(Path(temporary), 1, meta, transport)
            result = provider(query)
            self.assertEqual(result.answer, 0)
            self.assertEqual(result.attempts, 2)
            self.assertEqual(transport.messages[1][-2]["role"], "assistant")
            self.assertEqual(result.prompt_tokens, 20)

    def test_twenty_abstentions_preserve_distribution_and_consume_slots(self):
        score = BDeuScore(np.zeros((10, 5)), equivalent_sample_size=1, structure_prior=1, max_parents=4)
        a, b = graph_from_edges(5, directed=[(0, 1)]), graph_from_edges(5, directed=[(1, 0)])
        result = active_query_smc({a:.5, b:.5}, score, lambda q: QueryResponse(None, None),
            budgets=(5, 10, 20), confidence=.9, treatment="soft_evidence", seed=2)
        self.assertEqual(len(result.steps), 20)
        self.assertEqual(len({s.query.query_id for s in result.steps}), 20)
        self.assertEqual(result.abstention_count, 20)
        for distribution in result.distributions.values():
            self.assertEqual(distribution, {a:.5,b:.5})

    def test_soft_answer_updates_weights_and_replay_is_deterministic(self):
        score = BDeuScore(np.zeros((10, 3)), structure_prior=1)
        a, b = graph_from_edges(3, directed=[(0, 1)]), graph_from_edges(3, directed=[(1, 0)])
        pool = [ClaimQuery("directed", 0, 1, "edge"), ClaimQuery("adjacency", 0, 1, "adj")]
        def provider(q):
            return QueryResponse(1, .8, treatment="soft_evidence") if q.kind=="directed" else QueryResponse(None, None)
        def replay():
            return active_query_smc({a:.5,b:.5}, score, provider, query_pool=pool, budgets=(2,),
                                    confidence=.9, treatment="soft_evidence", ess_threshold_fraction=.01, seed=42)
        result = replay()
        self.assertAlmostEqual(result.distributions[2][a], .8)
        self.assertEqual(result.distributions, replay().distributions)

    def test_bdeu_agrees_with_ges_and_keeps_constant_columns(self):
        rng = np.random.default_rng(32)
        data = rng.integers(0, 3, (60, 5))
        data[:, 4] = 0
        score = BDeuScore(data, equivalent_sample_size=1, structure_prior=1, max_parents=4)
        ges_score = FastBDeuLocalScore(data)
        for child, parents in [(0, []), (0, [1, 2]), (4, [0, 1]), (2, [4])]:
            self.assertAlmostEqual(score.local_score(child, sum(1 << p for p in parents)), ges_score.local_score(child, parents))

    def test_distribution_scoring_maps_nodes_and_uses_truth_blind_ties(self):
        records = [dict(edges=[[0, 1]], weight=.5), dict(edges=[[1, 0]], weight=.5)]
        scores = distribution_scores(records, [1, 0, 2], dict(names=["a", "b", "c"], edges=[[0, 1]]))
        self.assertEqual(scores["directed_f1"], 1)
        self.assertAlmostEqual(scores["posterior_expected_distance"], .5)

    def test_cache_reuses_exact_requests_and_retains_uncertain_reservations(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"OPENROUTER_API_KEY":"sk-or-test"}):
            directory = Path(temporary)
            messages = [dict(role="user", content="A question")]
            request = request_for(messages, 0)
            digest = fingerprint(request)
            saved = dict(request=request, reserved_usd=reservation(request), response=response())
            write_json(directory / "requests" / f"{digest}_0.json", saved)
            with patch("urllib.request.urlopen", side_effect=AssertionError("cache must avoid network")):
                self.assertEqual(Transport(directory, 0).query(messages)[0], saved["response"])
            self.assertEqual(accounted(dict(reserved_usd=.4)), .4)
            write_json(directory / "requests/block.json", dict(reserved_usd=1.0))
            with self.assertRaisesRegex(RuntimeError, "ceiling"):
                Transport(directory, 0).query([dict(role="user", content="different")])


if __name__ == "__main__":
    unittest.main()
