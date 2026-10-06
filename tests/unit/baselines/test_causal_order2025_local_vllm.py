from __future__ import annotations

import asyncio
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cd_llm.baselines.causal_order2025.local_vllm_backend import (
    LocalVLLMConfig,
    LocalVLLMTransport,
    _answer_candidates,
    _context_with_metadata,
    _effective_completion_limit,
    _passes_parse_gate,
    _pair_prompt,
    _qwen_chat_messages,
    _robust_answer_tag,
    _triplet_prompt,
    _validated_triplet_answer_tag,
)
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache


class _Response:
    id = "cmpl-local-1"
    model = "Qwen/Qwen3.5-9B"
    choices = [SimpleNamespace(message=SimpleNamespace(content="<Answer>[]</Answer>"))]

    def model_dump(self):
        return {
            "id": self.id,
            "model": self.model,
            "choices": [{"message": {"content": "<Answer>[]</Answer>"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }


class _Completions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response()


class LocalVLLMTransportTests(unittest.TestCase):
    def test_metadata_context_uses_opaque_ids_and_descriptions(self) -> None:
        context = _context_with_metadata(
            "to model a clinical diagnostic system",
            ("V01", "V02"),
            {"V01": "first finding", "V02": "second finding"},
        )
        self.assertIn("V01 means first finding", context)
        self.assertIn("V02 means second finding", context)

    def test_triplet_prompt_excludes_unqueried_metadata(self) -> None:
        module = SimpleNamespace(
            generate_subgraph_with_descr_prompt=lambda **kwargs: kwargs,
            generate_subgraph_prompt=lambda **kwargs: kwargs,
        )
        prompt = _triplet_prompt(
            module,
            ("X01", "X02", "X03"),
            "a system",
            {
                "X01": "first",
                "X02": "second",
                "X03": "third",
                "X04": "unrelated",
            },
        )
        self.assertEqual(
            prompt["descr_nodes"],
            {"X01": "first", "X02": "second", "X03": "third"},
        )
        self.assertNotIn("X04", repr(prompt))

    def test_pair_prompt_excludes_unqueried_metadata(self) -> None:
        def builder(first, second, names, context):
            return {"first": first, "second": second, "names": names, "context": context}

        prompt = _pair_prompt(
            builder,
            "X01",
            "X02",
            ("X01", "X02", "X03"),
            "a system",
            {"X01": "first", "X02": "second", "X03": "unrelated"},
        )
        self.assertIn("X01 means first", prompt["context"])
        self.assertIn("X02 means second", prompt["context"])
        self.assertNotIn("unrelated", prompt["context"])

    def test_nonthinking_greedy_request_is_cached_without_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(Path(directory) / "cache.sqlite3")
            completions = _Completions()
            client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
            transport = LocalVLLMTransport(cache, client=client)
            messages = [
                {"role": "system", "content": "system text"},
                {"role": "assistant", "content": "prompt text"},
            ]
            first = asyncio.run(
                transport.query(messages, max_tokens=300, purpose="triplet")
            )
            second = asyncio.run(
                transport.query(messages, max_tokens=300, purpose="triplet")
            )
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(len(completions.calls), 1)
            call = completions.calls[0]
            self.assertEqual(call["temperature"], 0.0)
            self.assertEqual(call["seed"], 0)
            self.assertEqual(call["max_tokens"], 300)
            self.assertEqual(
                call["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )
            self.assertEqual(
                [message["role"] for message in call["messages"]],
                ["system", "user"],
            )
            self.assertEqual(
                [message["content"] for message in call["messages"]],
                ["system text", "prompt text"],
            )
            self.assertEqual(messages[-1]["role"], "assistant")
            self.assertEqual(cache.summary()["cost_usd"], 0.0)
            cache.close()

    def test_existing_user_message_is_unchanged(self) -> None:
        messages = [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "prompt text"},
        ]
        self.assertEqual(_qwen_chat_messages(messages), messages)

    def test_missing_user_and_final_assistant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a user query"):
            _qwen_chat_messages([{"role": "system", "content": "test"}])

    def test_upstream_limits_map_to_qwen_specific_budgets(self) -> None:
        config = LocalVLLMConfig()
        self.assertEqual(_effective_completion_limit(300, config), 4096)
        self.assertEqual(_effective_completion_limit(400, config), 2048)

    def test_configuration_rejects_remote_or_sampling_server(self) -> None:
        with self.assertRaises(ValueError):
            LocalVLLMConfig(base_url="https://example.com/v1")
        with self.assertRaises(ValueError):
            LocalVLLMConfig(temperature=0.7)
        with self.assertRaises(ValueError):
            LocalVLLMConfig(triplet_retry_max_tokens=4096)

    def test_multiline_parser_uses_final_nonempty_answer(self) -> None:
        content = (
            "<Answer>...</Answer>\nreasoning\n"
            "<Answer>\n[('A','B'), ('C')]\n</Answer>"
        )
        self.assertEqual(
            _answer_candidates(content),
            ("...", "[('A','B'), ('C')]")
        )
        self.assertEqual(_robust_answer_tag(content), "[('A','B'), ('C')]")

    def test_parse_gate_uses_the_configured_aggregate_rate(self) -> None:
        self.assertTrue(_passes_parse_gate(1111 / 1140, 0.95))
        self.assertFalse(_passes_parse_gate(0.949, 0.95))

    def test_cached_replay_turns_an_unparseable_tag_into_an_abstention(self) -> None:
        modules = SimpleNamespace(
            triplet=SimpleNamespace(str_2_lst=ast.literal_eval)
        )
        self.assertEqual(
            _validated_triplet_answer_tag(
                modules,
                "reasoning <Answer>[('A','B')]</Answer>",
            ),
            "[('A','B')]",
        )
        self.assertEqual(
            _validated_triplet_answer_tag(
                modules,
                "<Answer>[('A','B')]\nLet's write it.</Answer>",
            ),
            "[]",
        )


if __name__ == "__main__":
    unittest.main()
