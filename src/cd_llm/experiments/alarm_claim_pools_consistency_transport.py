"""Revision-isolated local transport for the ALARM consistency extension."""
from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlsplit

from cd_llm.baselines.causal_order2025.openrouter_backend import _request_key
from cd_llm.experiments.alarm_claim_pools_local_transport import (
    CheckpointRequested,
    request_identity,
    validate_response,
)


class LocalChat:
    def __init__(self, cache, manifest, base_url, *, deadline=float("inf"), client=None):
        url = urlsplit(base_url)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
                or url.path != "/v1" or url.query or url.fragment or url.username):
            raise ValueError("only a loopback vLLM endpoint is allowed")
        self.cache, self.manifest, self.deadline = cache, manifest, deadline
        self.stopped = False
        self.semaphore = asyncio.Semaphore(4)
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, max_retries=0, timeout=120.0)
        self.client = client
        cache.connection.execute("""CREATE TABLE IF NOT EXISTS local_attempts (
            id INTEGER PRIMARY KEY, request_key TEXT NOT NULL, purpose TEXT NOT NULL,
            state TEXT NOT NULL, response_json TEXT, error TEXT,
            started_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        cache.connection.commit()

    def check_deadline(self):
        if self.stopped or time.monotonic() >= self.deadline:
            raise CheckpointRequested("collection checkpoint reached")

    def accounting(self):
        states = dict(self.cache.connection.execute(
            "SELECT state,COUNT(*) FROM local_attempts GROUP BY state"))
        responses = [json.loads(row[0]) for row in self.cache.connection.execute(
            "SELECT response_json FROM local_attempts WHERE response_json IS NOT NULL")]
        return dict(
            self.cache.summary(), backend="local-vllm", actual_api_spending_usd=0.0,
            attempts=sum(states.values()), attempt_states=states,
            returned_prompt_tokens=sum((r.get("usage") or {}).get("prompt_tokens", 0) or 0 for r in responses),
            returned_completion_tokens=sum((r.get("usage") or {}).get("completion_tokens", 0) or 0 for r in responses),
            unresolved_attempts=states.get("pending", 0) + states.get("uncertain", 0),
        )

    async def request(self, messages, *, strategy, condition, purpose):
        if strategy != "consistency" or condition != "new":
            raise ValueError("only the frozen consistency protocol is allowed")
        request = request_identity(messages, self.manifest)
        async with self.semaphore:
            cached = self.cache.get(request)
            if cached is not None:
                validate_response(cached["response"], self.manifest)
                return cached
            recovered = self.cache.connection.execute(
                "SELECT response_json FROM local_attempts WHERE request_key=? "
                "AND response_json IS NOT NULL ORDER BY id DESC LIMIT 1",
                (_request_key(request),),
            ).fetchone()
            if recovered:
                response = json.loads(recovered[0])
                validate_response(response, self.manifest)
                self.store(request, purpose, response, 0.0, 1)
                return self.cache.get(request)
            for attempt in range(1, 6):
                self.check_deadline()
                cursor = self.cache.connection.execute(
                    "INSERT INTO local_attempts(request_key,purpose,state) VALUES (?,?,?)",
                    (_request_key(request), purpose, "pending"),
                )
                event = cursor.lastrowid
                self.cache.connection.commit()
                started = time.monotonic()
                try:
                    value = await self.client.chat.completions.create(**request["body"])
                except Exception as exc:
                    self.cache.connection.execute(
                        "UPDATE local_attempts SET state='uncertain',error=? WHERE id=?",
                        (type(exc).__name__, event),
                    )
                    self.cache.connection.commit()
                    status = getattr(exc, "status_code", None)
                    if attempt == 5 or (status is not None and status < 500 and status not in (408, 409, 429)):
                        raise RuntimeError(
                            f"local request failed ({status or type(exc).__name__})") from exc
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
                    continue
                response = value.model_dump()
                self.cache.connection.execute(
                    "UPDATE local_attempts SET state='returned',response_json=? WHERE id=?",
                    (json.dumps(response), event),
                )
                self.cache.connection.commit()
                validate_response(response, self.manifest)
                self.store(request, purpose, response, time.monotonic() - started, attempt)
                return self.cache.get(request)
        raise AssertionError("unreachable")

    def store(self, request, purpose, response, elapsed, attempts):
        usage = response["usage"]
        self.cache.put(
            request, purpose, response, response["choices"][0]["message"]["content"],
            provider_request_id=response.get("id"), response_model=response["model"],
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"], cost_usd=0.0,
            latency_seconds=elapsed, attempts=attempts,
        )

    async def close(self):
        await self.client.close()


class OfflineChat:
    def __init__(self, records, manifest):
        self.records, self.manifest = records, manifest

    async def request(self, messages, *, strategy, condition, purpose):
        if strategy != "consistency" or condition != "new":
            raise ValueError("unexpected replay protocol")
        request = request_identity(messages, self.manifest)
        item = self.records[_request_key(request)]
        if item["request"] != request or item["purpose"] != purpose:
            raise ValueError("replay request differs")
        validate_response(item["record"]["response"], self.manifest)
        return item["record"]
