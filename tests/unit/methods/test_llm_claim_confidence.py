from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from cd_llm.baselines.causal_order2025.openrouter_backend import (
    OpenRouterConfig,
    ResponseCache,
)
from cd_llm.methods.active_query import ClaimQuery
from cd_llm.methods.llm_claim_confidence import (
    CachedOpenRouterChatClient,
    DOCUMENT_URLS,
    DocumentSource,
    GroundedEvidencePassage,
    QwenClaimConfidenceProvider,
    confidence_metrics,
    load_document_corpus,
    retrieve_passages,
)


class _AsyncTransport:
    def __init__(self) -> None:
        self.requests = []

    async def request(self, messages, *, max_completion_tokens, purpose):
        self.requests.append((messages, max_completion_tokens, purpose))
        return {"content": '{"answer":"yes","confidence":0.8}'}


class _Chat:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.requests = []

    def request(self, messages, *, purpose, max_tokens=256):
        self.requests.append((messages, purpose, max_tokens))
        index = len(self.requests)
        return {
            "content": next(self.contents),
            "request_key": f"key-{index}",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "attempts": 1,
        }


def _sources() -> tuple[DocumentSource, ...]:
    return tuple(
        DocumentSource(
            f"PMC{index}",
            f"Review {index}",
            url,
            ("Cervical radiculopathy may cause neck symptoms and arm pain.",),
        )
        for index, url in enumerate(DOCUMENT_URLS, start=1)
    )


class LLMClaimConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = ClaimQuery("directed", 0, 1, "directed-0-1")
        self.names = ("L C2 Radikulopati", "L Nackbesvär")

    def test_stated_probability_retries_one_malformed_response(self) -> None:
        chat = _Chat(["not json", '{"answer":"yes","confidence":0.8}'])
        response = QwenClaimConfidenceProvider("stated", self.names, chat)(self.query)
        self.assertEqual(response.answer, 1)
        self.assertEqual(response.confidence, 0.8)
        self.assertEqual(response.prompt_tokens, 20)
        self.assertEqual(response.completion_tokens, 4)
        self.assertEqual(len(chat.requests), 2)

    def test_opaque_descriptions_replace_builtin_clinical_names(self) -> None:
        chat = _Chat(['{"answer":"yes","confidence":0.8}'])
        provider = QwenClaimConfidenceProvider(
            "stated",
            ("V01", "V02"),
            chat,
            descriptions={"V01": "first finding", "V02": "second finding"},
            domain_context="a clinical diagnostic system",
        )
        provider(ClaimQuery("directed", 0, 1, "directed-0-1"))
        prompt = chat.requests[0][0][1]["content"]
        self.assertIn("V01 = first finding", prompt)
        self.assertIn("V02 = second finding", prompt)
        self.assertIn("clinical diagnostic system", prompt)
        self.assertNotIn("Radikulopati", prompt)

    def test_group_ii_prompts_distinguish_ancestry_and_ordering(self) -> None:
        ancestry_chat = _Chat(['{"answer":"yes","confidence":0.8}'])
        ancestry = ClaimQuery("ancestry", 0, 1, "ancestry-0-1")
        response = QwenClaimConfidenceProvider(
            "stated", self.names, ancestry_chat
        )(ancestry)
        self.assertTrue(response.valid)
        ancestry_prompt = ancestry_chat.requests[0][0][1]["content"]
        self.assertIn("directly or indirectly", ancestry_prompt)
        self.assertIn("directed causal pathway", ancestry_prompt)

        ordering_chat = _Chat(['{"answer":"no","confidence":0.7}'])
        ordering = ClaimQuery("ordering", 0, 1, "ordering-0-1")
        response = QwenClaimConfidenceProvider(
            "stated", self.names, ordering_chat
        )(ordering)
        self.assertTrue(response.valid)
        ordering_prompt = ordering_chat.requests[0][0][1]["content"]
        self.assertIn("causally before", ordering_prompt)
        self.assertIn("directly or indirectly", ordering_prompt)

    def test_synchronous_openrouter_adapter_preserves_request_metadata(self) -> None:
        transport = _AsyncTransport()
        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(Path(directory) / "responses.sqlite3")
            client = CachedOpenRouterChatClient(
                cache,
                OpenRouterConfig(cost_cap_usd=1.0),
                transport=transport,
            )
            try:
                result = client.request(
                    ({"role": "user", "content": "question"},),
                    purpose="test-request",
                    max_tokens=123,
                )
            finally:
                client.close()
                cache.close()
            self.assertIn("answer", result["content"])
            self.assertEqual(transport.requests[0][1:], (123, "test-request"))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                client.request((), purpose="closed")

    def test_stated_probability_abstains_after_two_failures(self) -> None:
        response = QwenClaimConfidenceProvider(
            "stated", self.names, _Chat(["bad", '{"answer":"abstain","confidence":0.7}'])
        )(self.query)
        self.assertFalse(response.valid)
        self.assertEqual(response.abstention_reason, "malformed-or-abstained-twice")

    def test_v2_stated_probability_separates_knowledge_and_probability(self) -> None:
        unknown_chat = _Chat(['{"can_assess":false}'])
        unknown = QwenClaimConfidenceProvider(
            "stated", self.names, unknown_chat, protocol="separated_v2"
        )(self.query)
        self.assertFalse(unknown.valid)
        self.assertEqual(unknown.abstention_reason, "self-reported-insufficient-knowledge")
        self.assertEqual(len(unknown_chat.requests), 1)

        chat = _Chat(['{"can_assess":true}', '{"probability_claim_true":0.2}'])
        response = QwenClaimConfidenceProvider(
            "stated", self.names, chat, protocol="separated_v2"
        )(self.query)
        self.assertEqual(response.answer, 0)
        self.assertAlmostEqual(response.confidence, 0.8)
        self.assertEqual(response.audit["probability_claim_true"], 0.2)
        self.assertEqual(response.treatment, "soft_evidence")

        indifferent = QwenClaimConfidenceProvider(
            "stated",
            self.names,
            _Chat(['{"can_assess":true}', '{"probability_claim_true":0.5}']),
            protocol="separated_v2",
        )(self.query)
        self.assertFalse(indifferent.valid)
        self.assertEqual(indifferent.abstention_reason, "reported-indifference")

    def test_v2_answerability_gate_is_reused_by_stated_probability(self) -> None:
        chat = _Chat(
            ['{"can_assess":true}', '{"probability_claim_true":0.8}']
        )
        provider = QwenClaimConfidenceProvider(
            "stated", self.names, chat, protocol="separated_v2"
        )
        assessment = provider.assess_answerability(self.query, 0.6)
        response = provider.answer_after_assessment(self.query, assessment)
        self.assertTrue(assessment.can_assess)
        self.assertEqual(assessment.probability, 0.6)
        self.assertEqual(response.answer, 1)
        self.assertEqual(len(chat.requests), 2)
        self.assertIn(":knowledge-1", chat.requests[0][1])
        self.assertIn(":probability-1", chat.requests[1][1])
        self.assertTrue(response.audit["reused_answerability_gate"])

    def test_semantic_consistency_unanimous_and_tie(self) -> None:
        unanimous = QwenClaimConfidenceProvider(
            "consistency",
            self.names,
            _Chat(['{"answer":"yes"}'] * 8),
        )(self.query)
        self.assertEqual(unanimous.answer, 1)
        self.assertAlmostEqual(unanimous.confidence, 8.5 / 9.0)
        tie = QwenClaimConfidenceProvider(
            "consistency",
            self.names,
            _Chat(['{"answer":"yes"}'] * 4 + ['{"answer":"no"}'] * 4),
        )(self.query)
        self.assertEqual(tie.answer, 0)
        self.assertEqual(tie.confidence, 0.5)

    def test_semantic_consistency_abstains_below_six_valid_variants(self) -> None:
        response = QwenClaimConfidenceProvider(
            "consistency",
            self.names,
            _Chat(['{"answer":"yes"}'] * 5 + ["bad"] * 3),
        )(self.query)
        self.assertFalse(response.valid)

    def test_five_response_consistency_threshold_and_confidence_floor(self) -> None:
        for positive, negative, expected in ((4, 0, None), (3, 2, 0.5), (4, 1, 0.5), (5, 0, 5.5 / 9), (3, 3, 0.5), (8, 0, 8.5 / 9)):
            with self.subTest(positive=positive, negative=negative):
                contents = ['{"answer":"yes"}'] * positive + ['{"answer":"no"}'] * negative
                contents += ['{"answer":"abstain"}'] * (8 - len(contents))
                chat = _Chat(contents)
                response = QwenClaimConfidenceProvider(
                    "consistency", self.names, chat, minimum_valid_consistency_responses=5,
                )(self.query)
                self.assertEqual(len(chat.requests), 8)
                self.assertEqual(response.confidence, expected)
                self.assertEqual(response.valid, expected is not None)
                self.assertEqual(response.audit["valid_response_count"], positive + negative)
                self.assertEqual(response.audit["minimum_valid_response_count"], 5)

    def test_v2_semantic_consistency_decodes_negation_and_order_pairs(self) -> None:
        contents = (
            ['{"judgment":"supported"}'] * 4
            + ['{"judgment":"contradicted"}'] * 4
        )
        response = QwenClaimConfidenceProvider(
            "consistency", self.names, _Chat(contents), protocol="separated_v2"
        )(self.query)
        self.assertEqual(response.answer, 1)
        self.assertEqual(response.confidence, 1.0)
        self.assertEqual(response.audit["pair_votes"], [1, 1, 1, 1])
        self.assertEqual(response.audit["semantic_votes"], [1, 1])
        self.assertEqual(response.audit["order_sensitive_pair_count"], 0)

    def test_v2_semantic_requires_matched_affirmative_and_negated_evidence(self) -> None:
        response = QwenClaimConfidenceProvider(
            "consistency",
            self.names,
            _Chat(
                ['{"judgment":"unknown"}'] * 4
                + ['{"judgment":"supported"}'] * 4
            ),
            protocol="separated_v2",
        )(self.query)
        self.assertFalse(response.valid)
        self.assertEqual(response.audit["pair_votes"], [0, 0, -1, -1])
        self.assertEqual(response.audit["semantic_votes"], [0, 0])

    def test_v2_semantic_order_bias_becomes_abstention(self) -> None:
        response = QwenClaimConfidenceProvider(
            "consistency",
            self.names,
            _Chat(
                [
                    value
                    for _ in range(4)
                    for value in ('{"judgment":"unknown"}', '{"judgment":"contradicted"}')
                ]
            ),
            protocol="separated_v2",
        )(self.query)
        self.assertFalse(response.valid)
        self.assertEqual(response.audit["order_sensitive_pair_count"], 4)
        self.assertEqual(response.audit["signed_consistency_score"], 0.0)

    def test_grounded_response_requires_a_retrieved_support_identifier(self) -> None:
        sources = _sources()
        chat = _Chat(['{"answer":"yes","confidence":0.75,"support":["PMC1"]}'])
        response = QwenClaimConfidenceProvider(
            "grounded", self.names, chat, sources=sources
        )(self.query)
        self.assertEqual(response.answer, 1)
        self.assertEqual(response.confidence, 0.75)
        self.assertIn("PMC1", response.audit["support_source_ids"])
        unsupported = QwenClaimConfidenceProvider(
            "grounded",
            self.names,
            _Chat(['{"answer":"yes","confidence":0.9,"support":["TARGET-DAG"]}']),
            sources=sources,
        )(self.query)
        self.assertFalse(unsupported.valid)

    def test_v2_grounding_requires_cited_direct_pair_evidence(self) -> None:
        evidence = {
            "L C2 Radikulopati__L Nackbesvär": (
                GroundedEvidencePassage(
                    "PMID1",
                    "Pair study",
                    "https://pubmed.ncbi.nlm.nih.gov/1/",
                    2021,
                    ("The first variable directly regulates the second.",),
                ),
                GroundedEvidencePassage(
                    "PMID2",
                    "Unrelated study",
                    "https://pubmed.ncbi.nlm.nih.gov/2/",
                    2020,
                    ("The variables have broad cellular functions.",),
                ),
            )
        }
        response = QwenClaimConfidenceProvider(
            "grounded",
            self.names,
            _Chat(
                [
                    '{"relation":"supports_claim","directness":"direct","evidence_sentence_ids":[1]}',
                    '{"relation":"irrelevant","directness":"unclear","evidence_sentence_ids":[]}',
                ]
            ),
            protocol="separated_v2",
            pair_evidence=evidence,
        )(self.query)
        self.assertEqual(response.answer, 1)
        self.assertEqual(response.confidence, 0.75)
        self.assertEqual(response.audit["supporting_source_ids"], ["PMID1"])

        indirect = QwenClaimConfidenceProvider(
            "grounded",
            self.names,
            _Chat(
                [
                    '{"relation":"supports_claim","directness":"indirect","evidence_sentence_ids":[1]}',
                    '{"relation":"irrelevant","directness":"unclear","evidence_sentence_ids":[]}',
                ]
            ),
            protocol="separated_v2",
            pair_evidence=evidence,
        )(self.query)
        self.assertFalse(indirect.valid)

    def test_retrieval_returns_at_most_one_passage_per_whitelisted_source(self) -> None:
        passages = retrieve_passages(_sources(), "cervical radiculopathy")
        self.assertEqual(len(passages), 4)
        self.assertEqual(len({source for source, _ in passages}), 4)

    def test_corpus_loader_rejects_target_or_nonwhitelisted_sources(self) -> None:
        sources = _sources()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source_id": source.source_id,
                                "title": source.title,
                                "url": source.url,
                                "passages": list(source.passages),
                            }
                            for source in sources
                        ]
                    }
                )
            )
            self.assertEqual(len(load_document_corpus(path)), 4)
            payload = json.loads(path.read_text())
            payload["sources"][0]["url"] = "https://example.com/target-dag"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "whitelist"):
                load_document_corpus(path)

    def test_confidence_metrics_match_hand_calculation(self) -> None:
        metrics = confidence_metrics(((1, 0.8), (0, 0.6)))
        self.assertEqual(metrics["valid_count"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["brier"], ((0.8 - 1) ** 2 + 0.6**2) / 2)
        self.assertTrue(math.isfinite(float(metrics["ece"])))
        empty = confidence_metrics(())
        self.assertEqual(empty["valid_count"], 0)
        self.assertTrue(math.isnan(float(empty["brier"])))


if __name__ == "__main__":
    unittest.main()
