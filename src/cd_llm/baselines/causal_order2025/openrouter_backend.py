"""Resumable OpenRouter transport for the pinned triplet-order baseline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import TripletConfig, TripletExpertResult, UpstreamCausalOrderError
from .upstream_backend import (
    _load_upstream_modules,
    query_triplet_order,
    verified_upstream_root,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.6-terra"
INPUT_USD_PER_TOKEN = 2.0e-6
OUTPUT_USD_PER_TOKEN = 12.0e-6


@dataclass(frozen=True)
class OpenRouterConfig:
    """Auditable transport settings for one elicitation run."""

    model: str = DEFAULT_MODEL
    base_url: str = OPENROUTER_BASE_URL
    seed: int = 0
    concurrency: int = 16
    max_attempts: int = 8
    cost_cap_usd: float = 50.0
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("model and base_url must be nonempty")
        if self.concurrency < 1 or self.max_attempts < 1:
            raise ValueError("concurrency and max_attempts must be positive")
        if not math.isfinite(self.cost_cap_usd) or self.cost_cap_usd <= 0:
            raise ValueError("cost_cap_usd must be finite and positive")


@dataclass(frozen=True)
class OpenRouterElicitation:
    expert: TripletExpertResult
    audit: Mapping[str, Any]


def _canonical_request(
    messages: Sequence[Mapping[str, str]],
    model: str,
    max_completion_tokens: int,
    seed: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    request = {
        "messages": [dict(message) for message in messages],
        "model": model,
        "max_completion_tokens": int(max_completion_tokens),
        "seed": int(seed),
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    return request


def _request_key(request: Mapping[str, Any]) -> str:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ResponseCache:
    """Single-file, transactionally committed request/response cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=120.0)
        self.connection.execute("PRAGMA busy_timeout=120000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                request_key TEXT PRIMARY KEY,
                purpose TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                content TEXT NOT NULL,
                provider_request_id TEXT,
                response_model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL NOT NULL,
                latency_seconds REAL NOT NULL,
                attempts INTEGER NOT NULL,
                completed_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        key = _request_key(request)
        row = self.connection.execute(
            "SELECT content, response_json, provider_request_id, response_model, "
            "prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_seconds, attempts "
            "FROM responses WHERE request_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "request_key": key,
            "content": row[0],
            "response": json.loads(row[1]),
            "provider_request_id": row[2],
            "response_model": row[3],
            "prompt_tokens": row[4],
            "completion_tokens": row[5],
            "total_tokens": row[6],
            "cost_usd": float(row[7]),
            "latency_seconds": float(row[8]),
            "attempts": int(row[9]),
            "cached": True,
        }

    def put(
        self,
        request: Mapping[str, Any],
        purpose: str,
        response: Mapping[str, Any],
        content: str,
        *,
        provider_request_id: str | None,
        response_model: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        cost_usd: float,
        latency_seconds: float,
        attempts: int,
    ) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO responses "
            "(request_key, purpose, request_json, response_json, content, "
            "provider_request_id, response_model, prompt_tokens, completion_tokens, "
            "total_tokens, cost_usd, latency_seconds, attempts, completed_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (
                _request_key(request),
                purpose,
                json.dumps(request, sort_keys=True, separators=(",", ":")),
                json.dumps(response, sort_keys=True, separators=(",", ":")),
                content,
                provider_request_id,
                response_model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                float(cost_usd),
                float(latency_seconds),
                int(attempts),
            ),
        )
        self.connection.commit()

    def summary(self) -> dict[str, Any]:
        count, cost, prompt, completion, attempts = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), "
            "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
            "COALESCE(SUM(attempts), 0) FROM responses"
        ).fetchone()
        return {
            "request_count": int(count),
            "cost_usd": float(cost),
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_attempts": int(attempts),
        }


class OpenRouterTransport:
    """Concurrent OpenAI-compatible client with bounded retries and spending."""

    def __init__(
        self,
        cache: ResponseCache,
        config: OpenRouterConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.cache = cache
        self.config = config or OpenRouterConfig()
        if client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for uncached requests")
            try:
                from openai import AsyncOpenAI
            except ImportError as error:  # pragma: no cover - environment validation
                raise RuntimeError("install openai==3.3.1 for OpenRouter elicitation") from error
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                max_retries=0,
                timeout=120.0,
            )
        self.client = client
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._budget_lock = asyncio.Lock()
        self._reserved_usd = 0.0

    @staticmethod
    def _reservation(messages: Sequence[Mapping[str, str]], max_tokens: int) -> float:
        characters = sum(len(str(message.get("content", ""))) for message in messages)
        prompt_tokens = math.ceil(characters / 4)
        return prompt_tokens * INPUT_USD_PER_TOKEN + max_tokens * OUTPUT_USD_PER_TOKEN

    async def _reserve(self, amount: float) -> None:
        async with self._budget_lock:
            spent = float(self.cache.summary()["cost_usd"])
            if spent + self._reserved_usd + amount > self.config.cost_cap_usd + 1e-12:
                raise RuntimeError(
                    f"OpenRouter safety cap of ${self.config.cost_cap_usd:.2f} would be exceeded"
                )
            self._reserved_usd += amount

    async def _release(self, amount: float) -> None:
        async with self._budget_lock:
            self._reserved_usd -= amount

    async def request(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_completion_tokens: int,
        purpose: str,
    ) -> dict[str, Any]:
        request = _canonical_request(
            messages,
            self.config.model,
            max_completion_tokens,
            self.config.seed,
            self.config.reasoning_effort,
        )
        cached = self.cache.get(request)
        if cached is not None:
            return cached
        reservation = self._reservation(messages, max_completion_tokens)
        await self._reserve(reservation)
        try:
            async with self._semaphore:
                cached = self.cache.get(request)
                if cached is not None:
                    return cached
                started = time.perf_counter()
                last_error: Exception | None = None
                response_object: Any | None = None
                for attempt in range(1, self.config.max_attempts + 1):
                    try:
                        kwargs: dict[str, Any] = {
                            "model": self.config.model,
                            "messages": [dict(message) for message in messages],
                            "max_completion_tokens": max_completion_tokens,
                            "seed": self.config.seed,
                        }
                        if self.config.reasoning_effort is not None:
                            kwargs["reasoning_effort"] = self.config.reasoning_effort
                        response_object = await self.client.chat.completions.create(
                            **kwargs,
                        )
                        break
                    except Exception as error:
                        last_error = error
                        status = getattr(error, "status_code", None)
                        retryable = status in (408, 409, 429) or (
                            isinstance(status, int) and status >= 500
                        ) or status is None
                        if not retryable or attempt == self.config.max_attempts:
                            label = (
                                f"HTTP {status}"
                                if isinstance(status, int)
                                else type(error).__name__
                            )
                            raise RuntimeError(
                                f"OpenRouter request failed after {attempt} attempt(s): {label}"
                            ) from None
                        delay = min(60.0, 2.0 ** (attempt - 1)) + random.random()
                        await asyncio.sleep(delay)
                if response_object is None:  # pragma: no cover - defensive
                    raise RuntimeError("OpenRouter request produced no response") from last_error
                try:
                    response = response_object.model_dump()
                    content_value = response_object.choices[0].message.content
                    content = "" if content_value is None else str(content_value)
                    usage = response.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
                    cost = usage.get("cost")
                    if cost is None:
                        cost = (
                            int(prompt_tokens or 0) * INPUT_USD_PER_TOKEN
                            + int(completion_tokens or 0) * OUTPUT_USD_PER_TOKEN
                        )
                    latency = time.perf_counter() - started
                    self.cache.put(
                        request,
                        purpose,
                        response,
                        content,
                        provider_request_id=getattr(response_object, "id", None),
                        response_model=getattr(response_object, "model", None),
                        prompt_tokens=None if prompt_tokens is None else int(prompt_tokens),
                        completion_tokens=None if completion_tokens is None else int(completion_tokens),
                        total_tokens=None if total_tokens is None else int(total_tokens),
                        cost_usd=float(cost),
                        latency_seconds=latency,
                        attempts=attempt,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"OpenRouter returned an invalid response object: {type(error).__name__}"
                    ) from None
                result = self.cache.get(request)
                if result is None:  # pragma: no cover - defensive
                    raise RuntimeError("committed OpenRouter response is unavailable")
                result["cached"] = False
                return result
        finally:
            await self._release(reservation)

    async def request_many(
        self,
        requests: Sequence[tuple[Sequence[Mapping[str, str]], int, str]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        # One request validates credentials/model before launching the full batch.
        first = await self.request(
            requests[0][0], max_completion_tokens=requests[0][1], purpose=requests[0][2]
        )
        remaining = [
            asyncio.create_task(
                self.request(messages, max_completion_tokens=max_tokens, purpose=purpose)
            )
            for messages, max_tokens, purpose in requests[1:]
        ]
        return [first, *(await asyncio.gather(*remaining))]


def _cache_query(cache: ResponseCache, config: OpenRouterConfig):
    def query(messages: Sequence[Mapping[str, str]], model: str, max_completion_tokens: int) -> str:
        if model != config.model:
            raise UpstreamCausalOrderError(f"unexpected upstream model {model!r}")
        request = _canonical_request(messages, model, max_completion_tokens, config.seed)
        record = cache.get(request)
        if record is None:
            raise UpstreamCausalOrderError("upstream requested an uncached LLM response")
        return str(record["content"])

    return query


async def elicit_triplet_order_openrouter(
    node_names: Sequence[str],
    context: str,
    cache_path: str | Path,
    *,
    config: OpenRouterConfig | None = None,
    upstream_root: str | Path | None = None,
    client: Any | None = None,
) -> OpenRouterElicitation:
    """Query every upstream triplet, then run its merge/cycle code cache-only."""

    normalized = config or OpenRouterConfig()
    root = verified_upstream_root(upstream_root)
    modules = _load_upstream_modules(root)
    names = tuple(node_names)
    subgroups = modules.triplet.generate_all_subgroups(list(names), subgroup_size=3)
    triplet_requests = [
        (
            modules.triplet.generate_subgraph_prompt(nodes=list(group), context=context),
            300,
            "triplet",
        )
        for group in subgroups
    ]
    cache = ResponseCache(cache_path)
    transport = OpenRouterTransport(cache, normalized, client=client)
    started = time.perf_counter()
    try:
        responses = await transport.request_many(triplet_requests)
        parsed = []
        for response in responses:
            answer = modules.triplet.parse_answer_tag(response["content"])
            parsed.append(modules.triplet.str_2_lst(answer if answer is not None else "[]"))

        tie_pairs: list[tuple[str, str]] = []
        original_tiebreaker = modules.triplet._tiebreaker_query

        def collect_tie(first: str, second: str, *_args: Any, **_kwargs: Any) -> str:
            tie_pairs.append((first, second))
            return "No_Conn"

        modules.triplet._tiebreaker_query = collect_tie
        try:
            modules.triplet.merge_triplet_votes(
                subgroup_list=subgroups,
                subgraph_list=parsed,
                nodes=list(names),
                context=context,
                expert_model=normalized.model,
                delay=0,
            )
        finally:
            modules.triplet._tiebreaker_query = original_tiebreaker

        tie_requests = [
            (
                modules.triplet.cot_tiebreaker_prompt(first, second, list(names), context),
                400,
                "vote_tiebreaker",
            )
            for first, second in tie_pairs
        ]
        await transport.request_many(tie_requests)

        original_query = modules.triplet.query_llm
        modules.triplet.query_llm = _cache_query(cache, normalized)
        try:
            expert = query_triplet_order(
                names,
                context,
                descriptions=None,
                config=TripletConfig(
                    subgraph_model=normalized.model,
                    expert_model=normalized.model,
                    delay=0,
                ),
                upstream_root=root,
            )
        finally:
            modules.triplet.query_llm = original_query
        audit = {
            "provider": "OpenRouter",
            "base_url": normalized.base_url,
            "model": normalized.model,
            "seed": normalized.seed,
            "temperature_supplied": False,
            "concurrency": normalized.concurrency,
            "max_attempts": normalized.max_attempts,
            "cost_cap_usd": normalized.cost_cap_usd,
            "triplet_count": len(subgroups),
            "vote_tiebreaker_count": len(tie_pairs),
            "cache": cache.summary(),
            "runtime_seconds": time.perf_counter() - started,
        }
        return OpenRouterElicitation(expert=expert, audit=audit)
    finally:
        cache.close()


async def query_pair_fallbacks_openrouter(
    node_names: Sequence[str],
    pairs: Sequence[tuple[int, int]],
    context: str,
    cache_path: str | Path,
    *,
    config: OpenRouterConfig | None = None,
    upstream_root: str | Path | None = None,
    client: Any | None = None,
) -> tuple[dict[tuple[int, int], tuple[int, int] | None], dict[str, Any]]:
    """Query Algorithm-1 superior-expert fallbacks once per unique pair."""

    normalized = config or OpenRouterConfig()
    names = tuple(node_names)
    root = verified_upstream_root(upstream_root)
    modules = _load_upstream_modules(root)
    normalized_pairs = tuple(sorted(set((min(a, b), max(a, b)) for a, b in pairs)))
    requests = [
        (
            modules.triplet.cot_tiebreaker_prompt(
                names[first], names[second], list(names), context
            ),
            400,
            "algorithm1_fallback",
        )
        for first, second in normalized_pairs
    ]
    cache = ResponseCache(cache_path)
    transport = OpenRouterTransport(cache, normalized, client=client)
    try:
        responses = await transport.request_many(requests)
        orientations: dict[tuple[int, int], tuple[int, int] | None] = {}
        for pair, response in zip(normalized_pairs, responses, strict=True):
            answer = modules.triplet.parse_answer_tag(response["content"])
            if answer == "A":
                orientations[pair] = pair
            elif answer == "B":
                orientations[pair] = pair[1], pair[0]
            else:
                orientations[pair] = None
        return orientations, {
            "fallback_pair_count": len(normalized_pairs),
            "no_connection_count": sum(value is None for value in orientations.values()),
            "cache": cache.summary(),
        }
    finally:
        cache.close()


__all__ = [
    "DEFAULT_MODEL",
    "OPENROUTER_BASE_URL",
    "OpenRouterConfig",
    "OpenRouterElicitation",
    "OpenRouterTransport",
    "ResponseCache",
    "elicit_triplet_order_openrouter",
    "query_pair_fallbacks_openrouter",
]
