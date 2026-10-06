"""Local vLLM transport for the pinned causal-order elicitation pipeline."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openrouter_backend import ResponseCache
from .types import TripletConfig, TripletExpertResult, UpstreamCausalOrderError
from .upstream_backend import (
    _load_upstream_modules,
    query_triplet_order,
    verified_upstream_root,
)


QWEN35_MODEL = "Qwen/Qwen3.5-9B"
QWEN35_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN_ROLE_ADAPTER = "final_assistant_to_user_when_user_is_absent"
QWEN_TRIPLET_MAX_TOKENS = 4096
QWEN_TRIPLET_RETRY_MAX_TOKENS = 7680
QWEN_PAIR_MAX_TOKENS = 2048


@dataclass(frozen=True)
class LocalVLLMConfig:
    """Deterministic local-server settings used by the Qwen POC."""

    model: str = QWEN35_MODEL
    base_url: str = "http://127.0.0.1:8000/v1"
    seed: int = 0
    concurrency: int = 16
    max_attempts: int = 5
    temperature: float = 0.0
    triplet_max_tokens: int = QWEN_TRIPLET_MAX_TOKENS
    triplet_retry_max_tokens: int | None = None
    pair_max_tokens: int = QWEN_PAIR_MAX_TOKENS

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.startswith("http://127.0.0.1:"):
            raise ValueError("local vLLM requires a nonempty model and loopback base URL")
        if self.concurrency < 1 or self.max_attempts < 1:
            raise ValueError("concurrency and max_attempts must be positive")
        if self.triplet_max_tokens < 1 or self.pair_max_tokens < 1:
            raise ValueError("local Qwen completion limits must be positive")
        if (
            self.triplet_retry_max_tokens is not None
            and self.triplet_retry_max_tokens <= self.triplet_max_tokens
        ):
            raise ValueError("triplet retry limit must exceed the initial limit")
        if self.temperature != 0.0:
            raise ValueError("the Qwen proof of concept requires greedy temperature zero")


@dataclass(frozen=True)
class LocalVLLMElicitation:
    expert: TripletExpertResult
    audit: Mapping[str, Any]


class LocalVLLMGateError(RuntimeError):
    """The fixed response-quality gate failed before vote aggregation."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def _context_with_metadata(
    context: str,
    node_names: Sequence[str],
    descriptions: Mapping[str, str],
) -> str:
    """Embed separate node metadata in every upstream prompt path."""
    if not descriptions:
        return context
    glossary = "; ".join(
        f"{name} means {descriptions[name]}"
        for name in node_names
        if name in descriptions
    )
    return f"{context}. Variable metadata: {glossary}"


def _triplet_prompt(
    triplet_module: Any,
    group: Sequence[str],
    context: str,
    descriptions: Mapping[str, str],
) -> Sequence[Mapping[str, str]]:
    """Build the upstream triplet prompt with only the queried metadata."""

    if descriptions:
        return triplet_module.generate_subgraph_with_descr_prompt(
            nodes=list(group),
            context=context,
            descr_nodes={name: descriptions[name] for name in group},
        )
    return triplet_module.generate_subgraph_prompt(
        nodes=list(group), context=context
    )


def _pair_prompt(
    prompt_builder: Any,
    first: str,
    second: str,
    all_names: Sequence[str],
    context: str,
    descriptions: Mapping[str, str],
) -> Sequence[Mapping[str, str]]:
    """Build an upstream pair prompt without exposing unrelated metadata."""

    pair_context = _context_with_metadata(
        context, (first, second), descriptions
    )
    return prompt_builder(
        first, second, list(all_names), pair_context
    )


def _qwen_chat_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Adapt the upstream prompt roles without changing any prompt text."""

    adapted = [dict(message) for message in messages]
    if any(message.get("role") == "user" for message in adapted):
        return adapted
    if not adapted or adapted[-1].get("role") != "assistant":
        raise ValueError(
            "Qwen requires a user query; expected the upstream prompt to end in assistant"
        )
    adapted[-1]["role"] = "user"
    return adapted


def _request(
    messages: Sequence[Mapping[str, str]], config: LocalVLLMConfig, max_tokens: int
) -> dict[str, Any]:
    return {
        "provider": "local-vllm",
        "messages": _qwen_chat_messages(messages),
        "role_adapter": QWEN_ROLE_ADAPTER,
        "model": config.model,
        "max_tokens": int(max_tokens),
        "seed": config.seed,
        "temperature": config.temperature,
        "thinking": False,
    }


class LocalVLLMTransport:
    """Concurrent, resumable client for a loopback OpenAI-compatible server."""

    def __init__(
        self,
        cache: ResponseCache,
        config: LocalVLLMConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.cache = cache
        self.config = config or LocalVLLMConfig()
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:  # pragma: no cover - runtime validation
                raise RuntimeError("openai is required for the local vLLM client") from error
            client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=self.config.base_url,
                max_retries=0,
                timeout=300.0,
            )
        self.client = client
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    async def query(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        purpose: str,
    ) -> dict[str, Any]:
        request = _request(messages, self.config, max_tokens)
        cached = self.cache.get(request)
        if cached is not None:
            return cached
        async with self._semaphore:
            cached = self.cache.get(request)
            if cached is not None:
                return cached
            started = time.perf_counter()
            response_object: Any | None = None
            for attempt in range(1, self.config.max_attempts + 1):
                try:
                    response_object = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=request["messages"],
                        max_tokens=max_tokens,
                        seed=self.config.seed,
                        temperature=self.config.temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    )
                    break
                except Exception as error:
                    status = getattr(error, "status_code", None)
                    retryable = status in (408, 409, 429) or (
                        isinstance(status, int) and status >= 500
                    ) or status is None
                    if not retryable or attempt == self.config.max_attempts:
                        label = f"HTTP {status}" if isinstance(status, int) else type(error).__name__
                        raise RuntimeError(
                            f"local vLLM request failed after {attempt} attempt(s): {label}"
                        ) from None
                    await asyncio.sleep(min(8.0, 2.0 ** (attempt - 1)))
            if response_object is None:  # pragma: no cover - defensive
                raise RuntimeError("local vLLM returned no response")
            try:
                response = response_object.model_dump()
                content_value = response_object.choices[0].message.content
                content = "" if content_value is None else str(content_value)
                usage = response.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
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
                    cost_usd=0.0,
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempt,
                )
            except Exception as error:
                raise RuntimeError(
                    f"local vLLM returned an invalid response object: {type(error).__name__}"
                ) from None
            result = self.cache.get(request)
            if result is None:  # pragma: no cover - defensive
                raise RuntimeError("committed local response is unavailable")
            result["cached"] = False
            return result

    async def query_many(
        self,
        requests: Sequence[tuple[Sequence[Mapping[str, str]], int, str]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        first = await self.query(
            requests[0][0], max_tokens=requests[0][1], purpose=requests[0][2]
        )
        remaining = [
            asyncio.create_task(self.query(messages, max_tokens=limit, purpose=purpose))
            for messages, limit, purpose in requests[1:]
        ]
        return [first, *(await asyncio.gather(*remaining))]


def _effective_completion_limit(
    upstream_limit: int, config: LocalVLLMConfig
) -> int:
    if upstream_limit == 300:
        return config.triplet_max_tokens
    if upstream_limit == 400:
        return config.pair_max_tokens
    raise UpstreamCausalOrderError(
        f"unexpected upstream completion limit {upstream_limit}"
    )


def _answer_candidates(content: str) -> tuple[str, ...]:
    """Return tagged answers, including multiline answers, in response order."""

    return tuple(
        match.strip()
        for match in re.findall(r"<Answer>(.*?)</Answer>", content, flags=re.DOTALL)
    )


def _robust_answer_tag(content: str) -> str | None:
    """Use the final nonempty answer tag, tolerating line breaks inside it."""

    for candidate in reversed(_answer_candidates(content)):
        if candidate:
            return candidate
    return None


def _parse_triplet_content(modules: Any, content: str) -> tuple[str, list[Any]] | None:
    """Find the final syntactically valid tagged graph in a model response."""

    for candidate in reversed(_answer_candidates(content)):
        try:
            parsed = modules.triplet.str_2_lst(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            return candidate, parsed
    return None


def _validated_triplet_answer_tag(modules: Any, content: str) -> str:
    """Return a valid cached graph string or an explicit abstention."""

    result = _parse_triplet_content(modules, content)
    return "[]" if result is None else result[0]


def _passes_parse_gate(parse_rate: float, minimum_parse_rate: float) -> bool:
    """Return whether aggregate parsing meets the configured quality gate."""

    return parse_rate >= minimum_parse_rate


def _cache_query(cache: ResponseCache, config: LocalVLLMConfig):
    def query(messages: Sequence[Mapping[str, str]], model: str, max_completion_tokens: int) -> str:
        if model != config.model:
            raise UpstreamCausalOrderError(f"unexpected upstream model {model!r}")
        effective_limit = _effective_completion_limit(max_completion_tokens, config)
        limits = [effective_limit]
        if max_completion_tokens == 300 and config.triplet_retry_max_tokens is not None:
            limits.insert(0, config.triplet_retry_max_tokens)
        for limit in limits:
            record = cache.get(_request(messages, config, limit))
            if record is not None:
                return str(record["content"])
        raise UpstreamCausalOrderError("upstream requested an uncached local response")

    return query


async def elicit_triplet_order_local_vllm(
    node_names: Sequence[str],
    context: str,
    cache_path: str | Path,
    *,
    descriptions: Mapping[str, str] | None = None,
    config: LocalVLLMConfig | None = None,
    minimum_parse_rate: float = 0.95,
    upstream_root: str | Path | None = None,
    client: Any | None = None,
) -> LocalVLLMElicitation:
    """Run exhaustive upstream triplets through a non-thinking local server."""

    normalized = config or LocalVLLMConfig()
    if not math.isfinite(minimum_parse_rate) or not 0.0 <= minimum_parse_rate <= 1.0:
        raise ValueError("minimum_parse_rate must lie in [0, 1]")
    root = verified_upstream_root(upstream_root)
    modules = _load_upstream_modules(root)
    names = tuple(node_names)
    metadata = dict(descriptions or {})
    if set(metadata) - set(names):
        raise ValueError("descriptions contain unknown nodes")
    subgroups = modules.triplet.generate_all_subgroups(list(names), subgroup_size=3)

    requests = [
        (
            _triplet_prompt(modules.triplet, group, context, metadata),
            normalized.triplet_max_tokens,
            "triplet",
        )
        for group in subgroups
    ]
    cache = ResponseCache(cache_path)
    transport = LocalVLLMTransport(cache, normalized, client=client)
    started = time.perf_counter()
    try:
        responses = list(await transport.query_many(requests))
        initial_parsed = [
            _parse_triplet_content(modules, response["content"])
            for response in responses
        ]
        retry_indices = [
            index for index, result in enumerate(initial_parsed) if result is None
        ]
        if retry_indices and normalized.triplet_retry_max_tokens is not None:
            retried = await transport.query_many(
                [
                    (
                        requests[index][0],
                        normalized.triplet_retry_max_tokens,
                        "triplet_parse_retry",
                    )
                    for index in retry_indices
                ]
            )
            for index, response in zip(retry_indices, retried, strict=True):
                responses[index] = response

        final_parsed = [
            _parse_triplet_content(modules, response["content"])
            for response in responses
        ]
        parsed: list[Any] = []
        tagged = 0
        malformed = 0
        finish_reason_counts: dict[str, int] = {}
        for response, result in zip(responses, final_parsed, strict=True):
            try:
                finish_reason = str(response["response"]["choices"][0]["finish_reason"])
            except (IndexError, KeyError, TypeError):
                finish_reason = "unknown"
            finish_reason_counts[finish_reason] = (
                finish_reason_counts.get(finish_reason, 0) + 1
            )
            candidates = _answer_candidates(response["content"])
            if candidates:
                tagged += 1
            if result is None:
                if candidates:
                    malformed += 1
                parsed.append([])
                continue
            parsed.append(result[1])
        parseable = sum(result is not None for result in final_parsed)
        parse_rate = parseable / len(subgroups)
        gate_audit = {
            "triplet_count": len(subgroups),
            "initial_parseable_count": sum(result is not None for result in initial_parsed),
            "parse_retry_count": len(retry_indices)
            if normalized.triplet_retry_max_tokens is not None
            else 0,
            "answer_tag_count": tagged,
            "malformed_answer_count": malformed,
            "parseable_count": parseable,
            "unparseable_count": len(subgroups) - parseable,
            "parse_rate": parse_rate,
            "minimum_parse_rate": minimum_parse_rate,
            "initial_completion_tokens": normalized.triplet_max_tokens,
            "retry_completion_tokens": normalized.triplet_retry_max_tokens,
            "responses_at_completion_limit": sum(
                response.get("completion_tokens")
                == (
                    normalized.triplet_retry_max_tokens
                    if normalized.triplet_retry_max_tokens is not None
                    else normalized.triplet_max_tokens
                )
                for response in responses
            ),
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "cache": cache.summary(),
        }
        # Tagged-but-malformed completions are already counted as unparseable.
        # The configured parse-rate tolerance deliberately permits a small
        # number of such abstentions, so malformed tags must not impose an
        # undocumented zero-tolerance gate of their own.
        if not _passes_parse_gate(parse_rate, minimum_parse_rate):
            raise LocalVLLMGateError("local Qwen triplet parse gate failed", gate_audit)

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
        original_tiebreaker_prompt = modules.triplet.cot_tiebreaker_prompt

        def localized_tiebreaker_prompt(
            first: str,
            second: str,
            all_names: Sequence[str],
            _supplied_context: str,
        ) -> Sequence[Mapping[str, str]]:
            return _pair_prompt(
                original_tiebreaker_prompt,
                first,
                second,
                all_names,
                context,
                metadata,
            )

        await transport.query_many(
            [
                (
                    localized_tiebreaker_prompt(first, second, names, context),
                    normalized.pair_max_tokens,
                    "vote_tiebreaker",
                )
                for first, second in tie_pairs
            ]
        )

        original_query = modules.triplet.query_llm
        original_parser = modules.triplet.parse_answer_tag
        modules.triplet.query_llm = _cache_query(cache, normalized)
        modules.triplet.parse_answer_tag = lambda content: _validated_triplet_answer_tag(
            modules, content
        )
        modules.triplet.cot_tiebreaker_prompt = localized_tiebreaker_prompt
        try:
            expert = query_triplet_order(
                names,
                context,
                descriptions=metadata or None,
                config=TripletConfig(
                    subgraph_model=normalized.model,
                    expert_model=normalized.model,
                    delay=0,
                ),
                upstream_root=root,
            )
        finally:
            modules.triplet.query_llm = original_query
            modules.triplet.parse_answer_tag = original_parser
            modules.triplet.cot_tiebreaker_prompt = original_tiebreaker_prompt
        return LocalVLLMElicitation(
            expert=expert,
            audit={
                "provider": "local-vllm",
                "model": normalized.model,
                "seed": normalized.seed,
                "temperature": normalized.temperature,
                "thinking": False,
                "concurrency": normalized.concurrency,
                "vote_tiebreaker_count": len(tie_pairs),
                "runtime_seconds": time.perf_counter() - started,
                **gate_audit,
                "cache": cache.summary(),
            },
        )
    finally:
        cache.close()


async def query_pair_fallbacks_local_vllm(
    node_names: Sequence[str],
    pairs: Sequence[tuple[int, int]],
    context: str,
    cache_path: str | Path,
    *,
    descriptions: Mapping[str, str] | None = None,
    config: LocalVLLMConfig | None = None,
    upstream_root: str | Path | None = None,
    client: Any | None = None,
) -> tuple[dict[tuple[int, int], tuple[int, int] | None], dict[str, Any]]:
    normalized = config or LocalVLLMConfig()
    root = verified_upstream_root(upstream_root)
    modules = _load_upstream_modules(root)
    names = tuple(node_names)
    metadata = dict(descriptions or {})
    if set(metadata) - set(names):
        raise ValueError("descriptions contain unknown nodes")
    normalized_pairs = tuple(sorted(set((min(a, b), max(a, b)) for a, b in pairs)))
    cache = ResponseCache(cache_path)
    transport = LocalVLLMTransport(cache, normalized, client=client)
    try:
        responses = await transport.query_many(
            [
                (
                    _pair_prompt(
                        modules.triplet.cot_tiebreaker_prompt,
                        names[a],
                        names[b],
                        names,
                        context,
                        metadata,
                    ),
                    normalized.pair_max_tokens,
                    "algorithm1_fallback",
                )
                for a, b in normalized_pairs
            ]
        )
        result: dict[tuple[int, int], tuple[int, int] | None] = {}
        for pair, response in zip(normalized_pairs, responses, strict=True):
            answer = modules.triplet.parse_answer_tag(response["content"])
            result[pair] = pair if answer == "A" else (pair[1], pair[0]) if answer == "B" else None
        return result, {
            "fallback_pair_count": len(normalized_pairs),
            "no_connection_count": sum(value is None for value in result.values()),
            "cache": cache.summary(),
        }
    finally:
        cache.close()


__all__ = [
    "LocalVLLMConfig",
    "LocalVLLMElicitation",
    "LocalVLLMGateError",
    "LocalVLLMTransport",
    "QWEN35_MODEL",
    "QWEN35_REVISION",
    "QWEN_PAIR_MAX_TOKENS",
    "QWEN_ROLE_ADAPTER",
    "QWEN_TRIPLET_MAX_TOKENS",
    "QWEN_TRIPLET_RETRY_MAX_TOKENS",
    "elicit_triplet_order_local_vllm",
    "query_pair_fallbacks_local_vllm",
]
