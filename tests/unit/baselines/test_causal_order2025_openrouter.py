from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cd_llm.baselines.causal_order2025.openrouter_backend import (
    OpenRouterConfig,
    OpenRouterTransport,
    ResponseCache,
)


class _Response:
    id = "generation-123"
    model = "openai/gpt-5.6-terra"
    choices = [SimpleNamespace(message=SimpleNamespace(content="<Answer>A</Answer>"))]

    def model_dump(self):
        return {
            "id": self.id,
            "model": self.model,
            "choices": [{"message": {"content": self.choices[0].message.content}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cost": 0.000068,
            },
        }


class _Completions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response()


class _Client:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())


class _DeniedCompletions:
    async def create(self, **kwargs):
        error = RuntimeError("provider body containing sensitive account metadata")
        error.status_code = 403
        raise error


class OpenRouterTransportTests(unittest.TestCase):
    def test_request_is_seeded_audited_and_reused_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(Path(directory) / "responses.sqlite3")
            client = _Client()
            transport = OpenRouterTransport(
                cache,
                OpenRouterConfig(concurrency=2, cost_cap_usd=1.0),
                client=client,
            )
            messages = [{"role": "system", "content": "test"}]
            first = asyncio.run(
                transport.request(messages, max_completion_tokens=20, purpose="triplet")
            )
            second = asyncio.run(
                transport.request(messages, max_completion_tokens=20, purpose="triplet")
            )
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(len(client.chat.completions.calls), 1)
            kwargs = client.chat.completions.calls[0]
            self.assertEqual(kwargs["model"], "openai/gpt-5.6-terra")
            self.assertEqual(kwargs["seed"], 0)
            self.assertNotIn("temperature", kwargs)
            self.assertEqual(cache.summary()["request_count"], 1)
            self.assertAlmostEqual(cache.summary()["cost_usd"], 0.000068)
            cache.close()

    def test_cost_cap_blocks_before_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(Path(directory) / "responses.sqlite3")
            client = _Client()
            transport = OpenRouterTransport(
                cache,
                OpenRouterConfig(cost_cap_usd=1e-8),
                client=client,
            )
            with self.assertRaisesRegex(RuntimeError, "safety cap"):
                asyncio.run(
                    transport.request(
                        [{"role": "system", "content": "test"}],
                        max_completion_tokens=20,
                        purpose="triplet",
                    )
                )
            self.assertEqual(client.chat.completions.calls, [])
            cache.close()

    def test_provider_error_body_is_not_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(Path(directory) / "responses.sqlite3")
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=_DeniedCompletions())
            )
            transport = OpenRouterTransport(
                cache,
                OpenRouterConfig(cost_cap_usd=1.0),
                client=client,
            )
            with self.assertRaisesRegex(RuntimeError, "HTTP 403") as caught:
                asyncio.run(
                    transport.request(
                        [{"role": "system", "content": "test"}],
                        max_completion_tokens=20,
                        purpose="triplet",
                    )
                )
            self.assertNotIn("sensitive account metadata", str(caught.exception))
            cache.close()


if __name__ == "__main__":
    unittest.main()
