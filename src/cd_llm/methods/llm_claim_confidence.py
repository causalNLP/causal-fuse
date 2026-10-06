"""Auditable confidence elicitation for active binary causal claims."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from cd_llm.baselines.causal_order2025.local_vllm_backend import (
    LocalVLLMConfig,
    _request,
)
from cd_llm.baselines.causal_order2025.openrouter_backend import (
    OpenRouterConfig,
    OpenRouterTransport,
    ResponseCache,
)
from cd_llm.methods.active_query import (
    AnswerabilityEstimate,
    ClaimQuery,
    QueryResponse,
)


ConfidenceStrategy = Literal["stated", "consistency", "grounded"]
PromptProtocol = Literal["legacy_v1", "separated_v2"]
DOCUMENT_URLS = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3192889/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4958381/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7788378/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10982913/",
)
SEMANTIC_VARIANT_COUNT = 8
MINIMUM_VALID_SEMANTIC_RESPONSES = 6
DEFAULT_MAX_TOKENS = 256


@dataclass(frozen=True)
class DocumentSource:
    source_id: str
    title: str
    url: str
    passages: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("document source URL must use HTTP or HTTPS")
        if not self.source_id or not self.title or not self.passages:
            raise ValueError("document source metadata and passages must be nonempty")


@dataclass(frozen=True)
class GroundedEvidencePassage:
    """One dated, independently citable passage for a variable pair."""

    source_id: str
    title: str
    url: str
    publication_year: int
    sentences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id or not self.title or not self.sentences:
            raise ValueError("grounded evidence metadata and sentences must be nonempty")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("grounded evidence URL must use HTTP or HTTPS")
        if self.publication_year < 1800:
            raise ValueError("grounded evidence publication year is invalid")
        if any(not sentence.strip() for sentence in self.sentences):
            raise ValueError("grounded evidence sentences must be nonempty")


def load_document_corpus(path: str | Path) -> tuple[DocumentSource, ...]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("sources")
    if not isinstance(records, list):
        raise ValueError("document corpus must contain a source list")
    sources = tuple(
        DocumentSource(
            source_id=str(record["source_id"]),
            title=str(record["title"]),
            url=str(record["url"]),
            passages=tuple(str(value) for value in record["passages"]),
        )
        for record in records
    )
    if tuple(source.url for source in sources) != DOCUMENT_URLS:
        raise ValueError("document corpus must contain the four whitelisted reviews")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("document source identifiers must be unique")
    return sources


def corpus_fingerprint(sources: Sequence[DocumentSource]) -> str:
    value = [
        {
            "source_id": source.source_id,
            "title": source.title,
            "url": source.url,
            "passages": list(source.passages),
        }
        for source in sources
    ]
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _terms(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_passages(
    sources: Sequence[DocumentSource], query_text: str
) -> tuple[tuple[str, str], ...]:
    """Return at most one deterministic lexical match from each source."""

    query_terms = set(_terms(query_text))
    selected: list[tuple[str, str]] = []
    for source in sources:
        scored = []
        for index, passage in enumerate(source.passages):
            terms = set(_terms(passage))
            overlap = sum(1.0 + math.log1p(len(term)) for term in query_terms & terms)
            scored.append((overlap, -index, passage))
        score, _, passage = max(scored)
        if score > 0.0:
            selected.append((source.source_id, passage))
    return tuple(selected)


def _last_json_object(content: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates: list[Mapping[str, Any]] = []
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            candidates.append(value)
    return candidates[-1] if candidates else None


def _answer(value: Any) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return 1
    if normalized in {"no", "false", "0"}:
        return 0
    if normalized in {"abstain", "unknown", "insufficient evidence"}:
        return None
    raise ValueError("unrecognized answer label")


def _query_text(query: ClaimQuery, node_names: Sequence[str]) -> str:
    source = node_names[query.source]
    target = node_names[query.target]
    if query.kind == "directed":
        return f"Does {source} directly cause {target}?"
    if query.kind == "adjacency":
        return f"Are {source} and {target} directly causally related in either direction?"
    if query.kind == "ancestry":
        return (
            f"Can {source} cause {target}, either directly or indirectly through "
            "a directed causal pathway?"
        )
    if query.kind == "ordering":
        return (
            f"Is {source} causally before {target}, meaning that {target} does not "
            f"cause {source} directly or indirectly?"
        )
    raise ValueError(f"unsupported LLM query kind: {query.kind}")


def _claim_statement(query: ClaimQuery, node_names: Sequence[str]) -> str:
    source = node_names[query.source]
    target = node_names[query.target]
    if query.kind == "directed":
        return f"{source} directly causes {target}"
    if query.kind == "adjacency":
        return f"{source} and {target} are directly causally related in at least one direction"
    if query.kind == "ancestry":
        return f"{source} causes {target}, directly or through a directed causal pathway"
    if query.kind == "ordering":
        return f"{source} is causally before {target}, so {target} does not cause {source} directly or indirectly"
    raise ValueError(f"unsupported LLM query kind: {query.kind}")


def _glossary(
    node_names: Sequence[str], descriptions: Mapping[str, str] | None = None
) -> str:
    entries: list[str] = []
    for name in node_names:
        if descriptions is not None and name in descriptions:
            translated = descriptions[name]
        else:
            translated = name
            translated = re.sub(r"^DLS ", "cervical disc lesion at ", translated)
            translated = re.sub(r"^L C(\d) Radikulopati$", r"left C\1 radiculopathy", translated)
            translated = re.sub(r"^R C(\d) Radikulopati$", r"right C\1 radiculopathy", translated)
            translated = translated.replace("Kraniocervikal ledskada", "craniocervical joint injury")
            translated = translated.replace("L Nackbesvär", "left-sided neck symptoms")
        entries.append(f"{name} = {translated}")
    return "; ".join(entries)


class CachedLocalChatClient:
    """Synchronous cached client for a loopback OpenAI-compatible server."""

    def __init__(
        self,
        cache: ResponseCache,
        config: LocalVLLMConfig,
        *,
        client: Any | None = None,
        prompt_auditor: Callable[[str], None] | None = None,
        cache_namespace: str | None = None,
    ) -> None:
        self.cache = cache
        self.config = config
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - runtime validation
                raise RuntimeError("openai is required for local Qwen inference") from error
            client = OpenAI(
                api_key="EMPTY",
                base_url=config.base_url,
                max_retries=0,
                timeout=300.0,
            )
        self.client = client
        self.prompt_auditor = prompt_auditor
        self.cache_namespace = cache_namespace

    def request(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        purpose: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        if self.prompt_auditor is not None:
            self.prompt_auditor(
                "\n".join(str(message.get("content", "")) for message in messages)
            )
        request = _request(messages, self.config, max_tokens)
        if self.cache_namespace is not None:
            request["cache_namespace"] = self.cache_namespace
        cached = self.cache.get(request)
        if cached is not None:
            return cached
        started = time.perf_counter()
        response_object: Any | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response_object = self.client.chat.completions.create(
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
                        f"local Qwen request failed after {attempt} attempt(s): {label}"
                    ) from None
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
        if response_object is None:  # pragma: no cover
            raise RuntimeError("local Qwen returned no response")
        response = response_object.model_dump()
        content_value = response_object.choices[0].message.content
        content = "" if content_value is None else str(content_value)
        usage = response.get("usage") or {}
        self.cache.put(
            request,
            purpose,
            response,
            content,
            provider_request_id=getattr(response_object, "id", None),
            response_model=getattr(response_object, "model", None),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cost_usd=0.0,
            latency_seconds=time.perf_counter() - started,
            attempts=attempt,
        )
        result = self.cache.get(request)
        if result is None:  # pragma: no cover
            raise RuntimeError("committed local Qwen response is unavailable")
        result["cached"] = False
        return result


class CachedOpenRouterChatClient:
    """Synchronous adapter around the resumable OpenRouter transport."""

    def __init__(
        self,
        cache: ResponseCache,
        config: OpenRouterConfig,
        *,
        transport: Any | None = None,
        prompt_auditor: Callable[[str], None] | None = None,
    ) -> None:
        self.cache = cache
        self.config = config
        self.prompt_auditor = prompt_auditor
        self._runner = asyncio.Runner()
        self.transport = transport or OpenRouterTransport(cache, config)
        self._closed = False

    def request(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        purpose: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("OpenRouter chat client is closed")
        if self.prompt_auditor is not None:
            self.prompt_auditor(
                "\n".join(str(message.get("content", "")) for message in messages)
            )
        return self._runner.run(
            self.transport.request(
                messages,
                max_completion_tokens=max_tokens,
                purpose=purpose,
            )
        )

    def close(self) -> None:
        if not self._closed:
            self._runner.close()
            self._closed = True

    def __enter__(self) -> "CachedOpenRouterChatClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class QwenClaimConfidenceProvider:
    """Answer active claims using one prespecified confidence strategy."""

    def __init__(
        self,
        strategy: ConfidenceStrategy,
        node_names: Sequence[str],
        chat: CachedLocalChatClient | CachedOpenRouterChatClient,
        *,
        sources: Sequence[DocumentSource] = (),
        descriptions: Mapping[str, str] | None = None,
        domain_context: str = "neuropathic pain diagnosis",
        allow_unlisted_sources: bool = False,
        protocol: PromptProtocol = "legacy_v1",
        pair_evidence: Mapping[str, Sequence[GroundedEvidencePassage]] | None = None,
        minimum_valid_consistency_responses: int = MINIMUM_VALID_SEMANTIC_RESPONSES,
    ) -> None:
        if strategy not in ("stated", "consistency", "grounded"):
            raise ValueError("unknown confidence strategy")
        if protocol not in ("legacy_v1", "separated_v2"):
            raise ValueError("unknown prompt protocol")
        if type(minimum_valid_consistency_responses) is not int or not (
            1 <= minimum_valid_consistency_responses <= SEMANTIC_VARIANT_COUNT
        ):
            raise ValueError("minimum valid consistency responses must be in [1, 8]")
        if protocol != "legacy_v1" and minimum_valid_consistency_responses != MINIMUM_VALID_SEMANTIC_RESPONSES:
            raise ValueError("custom consistency minimum requires legacy_v1")
        self.minimum_valid_consistency_responses = minimum_valid_consistency_responses
        if (
            strategy == "grounded"
            and protocol == "legacy_v1"
            and not allow_unlisted_sources
            and tuple(source.url for source in sources) != DOCUMENT_URLS
        ):
            raise ValueError("grounded elicitation requires the whitelisted corpus")
        if protocol == "separated_v2" and strategy == "grounded" and pair_evidence is None:
            raise ValueError("separated_v2 grounded elicitation requires pair evidence")
        self.strategy = strategy
        self.protocol = protocol
        self.node_names = tuple(node_names)
        self.chat = chat
        self.sources = tuple(sources)
        self.pair_evidence = {
            str(key): tuple(value) for key, value in (pair_evidence or {}).items()
        }
        self.descriptions = dict(descriptions or {})
        unknown = set(self.descriptions) - set(self.node_names)
        if unknown:
            raise ValueError("descriptions contain unknown nodes")
        self.domain_context = str(domain_context).strip()
        if not self.domain_context:
            raise ValueError("domain_context must be nonempty")
        self.corpus_hash = corpus_fingerprint(sources) if sources else None

    def _audit(self, records: Sequence[Mapping[str, Any]], **values: Any) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "protocol": self.protocol,
            **values,
            **self._request_audit(records),
        }

    @staticmethod
    def _usage(records: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
        return (
            sum(int(record.get("prompt_tokens") or 0) for record in records),
            sum(int(record.get("completion_tokens") or 0) for record in records),
            sum(int(record.get("attempts") or 1) for record in records),
        )

    @staticmethod
    def _request_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "request_keys": [str(item["request_key"]) for item in records],
            "cache_hits": [bool(item.get("cached", False)) for item in records],
            "provider_request_ids": [item.get("provider_request_id") for item in records],
            "response_models": [item.get("response_model") for item in records],
            "cost_usd": [float(item.get("cost_usd") or 0.0) for item in records],
            "latency_seconds": [float(item.get("latency_seconds") or 0.0) for item in records],
        }

    def _base(self, query: ClaimQuery) -> tuple[str, str]:
        names = (self.node_names[query.source], self.node_names[query.target])
        return _query_text(query, self.node_names), _glossary(names, self.descriptions)

    @staticmethod
    def _boolean(value: Any) -> bool:
        if value is True or str(value).strip().lower() == "true":
            return True
        if value is False or str(value).strip().lower() == "false":
            return False
        raise ValueError("expected a JSON boolean")

    def assess_answerability(
        self, query: ClaimQuery, calibrated_probability: float
    ) -> AnswerabilityEstimate:
        """Run only the lightweight knowledge gate used for acquisition."""

        if self.protocol != "separated_v2" or self.strategy not in (
            "stated",
            "consistency",
        ):
            raise ValueError(
                "knowledge-gate answerability is defined only for separated "
                "stated and consistency protocols"
            )
        if not math.isfinite(float(calibrated_probability)) or not (
            0.0 <= float(calibrated_probability) <= 1.0
        ):
            raise ValueError("calibrated answerability must lie in [0, 1]")
        statement = _claim_statement(query, self.node_names)
        _, glossary = self._base(query)
        records: list[Mapping[str, Any]] = []
        can_assess: bool | None = None
        for prompt_attempt in range(2):
            retry = (
                " Your previous response was malformed; return JSON only."
                if prompt_attempt
                else ""
            )
            messages = (
                {
                    "role": "system",
                    "content": (
                        "You are a careful causal domain expert. Decide only whether you "
                        "have enough specific causal knowledge to assess the claim. Lack "
                        "of knowledge is not evidence that a causal claim is false."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context: {self.domain_context}. Glossary: {glossary}\nClaim: {statement}.\n"
                        "Do you have enough specific causal knowledge to assess this claim? "
                        "Return JSON only as {\"can_assess\":true} or "
                        f"{{\"can_assess\":false}}.{retry}"
                    ),
                },
            )
            record = self.chat.request(
                messages,
                purpose=(
                    f"v2:{self.strategy}:{query.query_id}:knowledge-"
                    f"{prompt_attempt + 1}"
                ),
            )
            records.append(record)
            parsed = _last_json_object(str(record["content"]))
            try:
                if parsed is None:
                    raise ValueError("missing JSON")
                can_assess = self._boolean(parsed["can_assess"])
            except (KeyError, ValueError):
                continue
            break
        prompt_tokens, completion_tokens, attempts = self._usage(records)
        audit = self._audit(
            records,
            knowledge_gate=can_assess,
            calibrated_probability=(
                float(calibrated_probability) if can_assess else 0.0
            ),
        )
        return AnswerabilityEstimate(
            probability=(float(calibrated_probability) if can_assess else 0.0),
            can_assess=bool(can_assess),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            attempts=attempts,
            physical_requests=len(records),
            audit=audit,
        )

    def _stated_v2(
        self,
        query: ClaimQuery,
        assessment: AnswerabilityEstimate | None = None,
    ) -> QueryResponse:
        statement = _claim_statement(query, self.node_names)
        _, glossary = self._base(query)
        records: list[Mapping[str, Any]] = []
        if assessment is None:
            assessment = self.assess_answerability(query, 1.0)
            gate_tokens = assessment.prompt_tokens
            gate_completion_tokens = assessment.completion_tokens
            gate_attempts = assessment.attempts
        else:
            gate_tokens = gate_completion_tokens = gate_attempts = 0
        if not assessment.can_assess:
            return QueryResponse(
                None,
                None,
                gate_tokens,
                gate_completion_tokens,
                max(1, gate_attempts),
                (
                    "malformed-knowledge-gate-twice"
                    if assessment.audit.get("knowledge_gate") is None
                    else "self-reported-insufficient-knowledge"
                ),
                dict(assessment.audit),
            )
        probability: float | None = None
        for prompt_attempt in range(2):
            retry = " Your previous response was malformed; return JSON only." if prompt_attempt else ""
            messages = (
                {
                    "role": "system",
                    "content": (
                        "You are a careful causal domain expert. You have already indicated that you "
                        "can assess the claim. Report the probability that the claim is true. Do not "
                        "turn lack of recalled evidence into a probability below one half."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context: {self.domain_context}. Glossary: {glossary}\nClaim: {statement}.\n"
                        "Return JSON only as {\"probability_claim_true\":0.73}. The probability must "
                        f"lie in [0,1], and 0.5 means that your knowledge favors neither conclusion.{retry}"
                    ),
                },
            )
            record = self.chat.request(
                messages, purpose=f"v2:stated:{query.query_id}:probability-{prompt_attempt + 1}"
            )
            records.append(record)
            parsed = _last_json_object(str(record["content"]))
            try:
                if parsed is None:
                    raise ValueError("missing JSON")
                value = float(parsed["probability_claim_true"])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("probability outside [0,1]")
                probability = value
            except (KeyError, TypeError, ValueError):
                continue
            break
        prompt_tokens, completion_tokens, attempts = self._usage(records)
        prompt_tokens += gate_tokens
        completion_tokens += gate_completion_tokens
        attempts += gate_attempts
        audit = self._audit(
            records,
            knowledge_gate=True,
            probability_claim_true=probability,
            reused_answerability_gate=gate_attempts == 0,
            answerability_gate_audit=dict(assessment.audit),
        )
        if probability is None:
            return QueryResponse(
                None,
                None,
                prompt_tokens,
                completion_tokens,
                attempts,
                "malformed-probability-twice",
                audit,
            )
        if probability == 0.5:
            return QueryResponse(
                None,
                None,
                prompt_tokens,
                completion_tokens,
                attempts,
                "reported-indifference",
                audit,
            )
        return QueryResponse(
            int(probability > 0.5),
            max(probability, 1.0 - probability),
            prompt_tokens,
            completion_tokens,
            attempts,
            audit=audit,
            treatment="soft_evidence",
        )

    def answer_after_assessment(
        self, query: ClaimQuery, assessment: AnswerabilityEstimate
    ) -> QueryResponse:
        """Run the full protocol without repeating a cached acquisition gate."""

        if not assessment.can_assess:
            return QueryResponse(
                None,
                None,
                abstention_reason="answerability-gate-negative",
                audit={"answerability_gate_audit": dict(assessment.audit)},
            )
        if self.protocol != "separated_v2":
            raise ValueError("gate reuse requires separated_v2")
        if self.strategy == "stated":
            return self._stated_v2(query, assessment)
        if self.strategy == "consistency":
            return self._consistency_v2(query)
        raise ValueError("document grounding does not use a knowledge gate")

    @staticmethod
    def _semantic_judgment(value: Any) -> int:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"supported", "supports", "support"}:
            return 1
        if normalized in {"contradicted", "contradicts", "contradict"}:
            return -1
        if normalized in {"unknown", "insufficient", "insufficient_knowledge", "abstain"}:
            return 0
        raise ValueError("unrecognized semantic judgment")

    def _consistency_v2(self, query: ClaimQuery) -> QueryResponse:
        statement = _claim_statement(query, self.node_names)
        _, glossary = self._base(query)
        contexts = (
            (statement, 1, "original-direct"),
            (f"The following causal relation holds: {statement}", 1, "original-paraphrase"),
            (f"It is not the case that {statement}", -1, "negated-direct"),
            (f"The following causal relation does not hold: {statement}", -1, "negated-paraphrase"),
        )
        orders = (
            ("supported", "contradicted", "unknown"),
            ("contradicted", "supported", "unknown"),
        )
        records: list[Mapping[str, Any]] = []
        decoded: list[int] = []
        raw_judgments: list[str | None] = []
        variant_metadata: list[dict[str, Any]] = []
        for context_index, (proposition, polarity, context_name) in enumerate(contexts):
            for order_index, order in enumerate(orders):
                messages = (
                    {
                        "role": "system",
                        "content": (
                            "You are a careful causal domain expert. Judge whether the proposition is "
                            "supported or contradicted by your causal knowledge. If you lack enough "
                            "knowledge, return unknown. Lack of knowledge is not contradiction."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context: {self.domain_context}. Glossary: {glossary}\nProposition: {proposition}\n"
                            f"The permitted judgments, presented in this order, are {order[0]}, "
                            f"{order[1]}, and {order[2]}. Return JSON only as "
                            '{"judgment":"supported|contradicted|unknown"}.'
                        ),
                    },
                )
                record = self.chat.request(
                    messages,
                    purpose=f"v2:consistency:{query.query_id}:c{context_index}:o{order_index}",
                )
                records.append(record)
                parsed = _last_json_object(str(record["content"]))
                raw: str | None = None
                value = 0
                try:
                    if parsed is None:
                        raise ValueError("missing JSON")
                    raw = str(parsed["judgment"])
                    value = self._semantic_judgment(raw) * polarity
                except (KeyError, ValueError):
                    raw = None
                    value = 0
                raw_judgments.append(raw)
                decoded.append(value)
                variant_metadata.append(
                    {
                        "context": context_name,
                        "polarity": polarity,
                        "order": list(order),
                        "raw_judgment": raw,
                        "decoded_relative_to_original": value,
                    }
                )
        pair_votes: list[int] = []
        pair_stability: list[bool] = []
        for index in range(0, len(decoded), 2):
            first, second = decoded[index : index + 2]
            stable = first == second
            pair_stability.append(stable)
            pair_votes.append(first if stable and first != 0 else 0)
        semantic_votes: list[int] = []
        polarity_pair_stability: list[bool] = []
        for original_index, negated_index in ((0, 2), (1, 3)):
            original = pair_votes[original_index]
            negated = pair_votes[negated_index]
            stable = original == negated and original != 0
            polarity_pair_stability.append(stable)
            semantic_votes.append(original if stable else 0)
        score = sum(semantic_votes) / len(semantic_votes)
        prompt_tokens, completion_tokens, attempts = self._usage(records)
        audit = self._audit(
            records,
            variant_count=len(records),
            variants=variant_metadata,
            raw_judgments=raw_judgments,
            decoded_responses=decoded,
            order_pair_stability=pair_stability,
            order_sensitive_pair_count=sum(not value for value in pair_stability),
            stable_nonunknown_pair_count=sum(value != 0 for value in pair_votes),
            pair_votes=pair_votes,
            polarity_pair_stability=polarity_pair_stability,
            semantic_votes=semantic_votes,
            signed_consistency_score=score,
            dependent_source_count=1,
        )
        if score == 0.0:
            return QueryResponse(
                None,
                None,
                prompt_tokens,
                completion_tokens,
                attempts,
                "zero-order-and-negation-stable-consistency-score",
                audit,
            )
        return QueryResponse(
            int(score > 0.0),
            0.5 + 0.5 * abs(score),
            prompt_tokens,
            completion_tokens,
            attempts,
            audit=audit,
            treatment="soft_evidence",
        )

    def _pair_key(self, query: ClaimQuery) -> str:
        names = sorted((self.node_names[query.source], self.node_names[query.target]))
        return "__".join(names)

    def _grounded_v2(self, query: ClaimQuery) -> QueryResponse:
        statement = _claim_statement(query, self.node_names)
        _, glossary = self._base(query)
        passages = self.pair_evidence.get(self._pair_key(query), ())
        records: list[Mapping[str, Any]] = []
        judgments: list[dict[str, Any]] = []
        supporting_sources: set[str] = set()
        contradicting_sources: set[str] = set()
        for passage in passages:
            sentence_block = "\n".join(
                f"[{index}] {sentence}" for index, sentence in enumerate(passage.sentences, start=1)
            )
            messages = (
                {
                    "role": "system",
                    "content": (
                        "You classify causal evidence in one supplied document. Use only the document. "
                        "Lack of mention is irrelevant, not contradiction. Contradiction requires "
                        "explicit evidence against the stated relation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context: {self.domain_context}. Glossary: {glossary}\nClaim: {statement}.\n"
                        f"Document {passage.source_id}, published {passage.publication_year}:\n{sentence_block}\n"
                        "Classify the document as supports_claim, contradicts_claim, or irrelevant. "
                        "For a direct causal claim, support requires direct causal or regulatory evidence; "
                        "functional similarity and indirect pathways are insufficient. Reverse-direction "
                        "evidence alone does not contradict the claim. Return JSON only as "
                        '{"relation":"supports_claim|contradicts_claim|irrelevant",'
                        '"directness":"direct|indirect|explicit_absence|unclear",'
                        '"evidence_sentence_ids":[1]}.'
                    ),
                },
            )
            record = self.chat.request(
                messages,
                purpose=f"v2:grounded:{query.query_id}:{passage.source_id}",
            )
            records.append(record)
            parsed = _last_json_object(str(record["content"]))
            accepted_relation = "irrelevant"
            directness = "unclear"
            evidence_ids: tuple[int, ...] = ()
            valid = False
            try:
                if parsed is None:
                    raise ValueError("missing JSON")
                relation = str(parsed["relation"]).strip().lower()
                directness = str(parsed["directness"]).strip().lower()
                raw_ids = parsed["evidence_sentence_ids"]
                if relation not in {"supports_claim", "contradicts_claim", "irrelevant"}:
                    raise ValueError("invalid relation")
                if directness not in {"direct", "indirect", "explicit_absence", "unclear"}:
                    raise ValueError("invalid directness")
                if not isinstance(raw_ids, list):
                    raise ValueError("sentence identifiers must be a list")
                evidence_ids = tuple(int(value) for value in raw_ids)
                if any(value < 1 or value > len(passage.sentences) for value in evidence_ids):
                    raise ValueError("invalid sentence identifier")
                if relation == "supports_claim":
                    valid = directness == "direct" and bool(evidence_ids)
                    accepted_relation = relation if valid else "irrelevant"
                elif relation == "contradicts_claim":
                    valid = directness == "explicit_absence" and bool(evidence_ids)
                    accepted_relation = relation if valid else "irrelevant"
                else:
                    valid = not evidence_ids
                    accepted_relation = "irrelevant"
            except (KeyError, TypeError, ValueError):
                relation = "malformed"
            if accepted_relation == "supports_claim":
                supporting_sources.add(passage.source_id)
            elif accepted_relation == "contradicts_claim":
                contradicting_sources.add(passage.source_id)
            judgments.append(
                {
                    "source_id": passage.source_id,
                    "raw_relation": relation,
                    "accepted_relation": accepted_relation,
                    "directness": directness,
                    "evidence_sentence_ids": list(evidence_ids),
                    "valid": valid,
                }
            )
        positive = len(supporting_sources)
        negative = len(contradicting_sources)
        score = (positive - negative) / (positive + negative + 1.0)
        prompt_tokens, completion_tokens, attempts = self._usage(records)
        audit = self._audit(
            records,
            pair_key=self._pair_key(query),
            retrieved_source_count=len(passages),
            judgments=judgments,
            supporting_source_ids=sorted(supporting_sources),
            contradicting_source_ids=sorted(contradicting_sources),
            shrinkage=1.0,
            signed_document_score=score,
        )
        if score == 0.0:
            reason = "no-pair-specific-passage" if not passages else "no-net-cited-direct-evidence"
            return QueryResponse(
                None,
                None,
                prompt_tokens,
                completion_tokens,
                max(1, attempts),
                reason,
                audit,
            )
        return QueryResponse(
            int(score > 0.0),
            0.5 + 0.5 * abs(score),
            prompt_tokens,
            completion_tokens,
            attempts,
            audit=audit,
            treatment="soft_evidence",
        )

    def _stated(self, query: ClaimQuery) -> QueryResponse:
        question, glossary = self._base(query)
        records: list[Mapping[str, Any]] = []
        for prompt_attempt in range(2):
            retry = " Your previous response was malformed; return JSON only." if prompt_attempt else ""
            messages = (
                {"role": "system", "content": "You are a careful causal domain expert. Do not infer from variable ordering. Answer the causal question and state the probability that your selected yes/no answer is correct. Abstain when you cannot make a defensible judgment."},
                {"role": "user", "content": f"Context: {self.domain_context}. Glossary: {glossary}\nQuestion: {question}\nReturn JSON only as {{\"answer\":\"yes|no|abstain\",\"confidence\":0.5}}. Confidence must be in [0.5,1].{retry}"},
            )
            record = self.chat.request(messages, purpose=f"stated:{query.query_id}:attempt-{prompt_attempt + 1}")
            records.append(record)
            parsed = _last_json_object(str(record["content"]))
            try:
                if parsed is None:
                    raise ValueError("missing JSON")
                answer = _answer(parsed.get("answer"))
                if answer is None:
                    raise ValueError("model abstained")
                confidence = float(parsed["confidence"])
                response = QueryResponse(
                    answer, confidence, treatment="soft_evidence"
                )
            except (KeyError, TypeError, ValueError):
                continue
            prompt_tokens, completion_tokens, attempts = self._usage(records)
            return QueryResponse(
                response.answer,
                response.confidence,
                prompt_tokens,
                completion_tokens,
                attempts,
                audit={"strategy": self.strategy, **self._request_audit(records)},
                treatment="soft_evidence",
            )
        prompt_tokens, completion_tokens, attempts = self._usage(records)
        return QueryResponse(
            None,
            None,
            prompt_tokens,
            completion_tokens,
            attempts,
            "malformed-or-abstained-twice",
            {"strategy": self.strategy, **self._request_audit(records)},
        )

    def _consistency(self, query: ClaimQuery) -> QueryResponse:
        question, glossary = self._base(query)
        proposition = (
            "direct causal proposition"
            if query.kind in ("directed", "adjacency")
            else "causal-path proposition"
        )
        phrasings = (
            question,
            f"Assess whether this causal relation is true: {question}",
            f"Judge this {proposition}: {question}",
            f"In {self.domain_context}, decide whether the following relation holds: {question}",
        )
        records: list[Mapping[str, Any]] = []
        votes: list[int] = []
        parsed_votes: list[int | None] = []
        for phrasing_index, phrasing in enumerate(phrasings):
            for order_index, labels in enumerate((("yes", "no"), ("no", "yes"))):
                messages = (
                    {"role": "system", "content": "You are a careful causal domain expert. Do not infer from variable ordering. Make one independent binary judgment and abstain if uncertain."},
                    {"role": "user", "content": f"Context: {self.domain_context}. Glossary: {glossary}\n{phrasing}\nAllowed labels in this order: {labels[0]}, {labels[1]}, abstain. Return JSON only as {{\"answer\":\"yes|no|abstain\"}}."},
                )
                record = self.chat.request(
                    messages,
                    purpose=f"consistency:{query.query_id}:p{phrasing_index}:o{order_index}",
                )
                records.append(record)
                parsed = _last_json_object(str(record["content"]))
                try:
                    vote = None if parsed is None else _answer(parsed.get("answer"))
                except ValueError:
                    vote = None
                parsed_votes.append(vote)
                if vote is not None:
                    votes.append(vote)
        prompt_tokens, completion_tokens, attempts = self._usage(records)
        audit = {
            "strategy": self.strategy,
            "variant_count": SEMANTIC_VARIANT_COUNT,
            "votes": parsed_votes,
            "valid_response_count": len(votes),
            "minimum_valid_response_count": self.minimum_valid_consistency_responses,
            "confidence_rule": "max(0.5, (majority_count + 0.5) / 9); ties=0.5",
            **self._request_audit(records),
            "dependent_source_count": 1,
        }
        if len(votes) < self.minimum_valid_consistency_responses:
            reason = (
                "fewer-than-six-valid-variants"
                if self.minimum_valid_consistency_responses == 6
                else f"fewer-than-{self.minimum_valid_consistency_responses}-valid-variants"
            )
            return QueryResponse(None, None, prompt_tokens, completion_tokens, attempts, reason, audit)
        positive = sum(votes)
        answer = int(positive > len(votes) / 2)
        if positive * 2 == len(votes):
            confidence = 0.5
        else:
            majority_count = max(positive, len(votes) - positive)
            confidence = max(0.5, (majority_count + 0.5) / 9.0)
        return QueryResponse(
            answer,
            confidence,
            prompt_tokens,
            completion_tokens,
            attempts,
            audit=audit,
            treatment="soft_evidence",
        )

    def _grounded(self, query: ClaimQuery) -> QueryResponse:
        question, glossary = self._base(query)
        passages = retrieve_passages(self.sources, f"{question} {glossary}")
        if not passages:
            return QueryResponse(None, None, abstention_reason="no-retrieved-passage", audit={"strategy": self.strategy, "corpus_hash": self.corpus_hash})
        context = "\n\n".join(f"[{source_id}] {passage}" for source_id, passage in passages)
        messages = (
            {"role": "system", "content": "You extract causal knowledge from supplied domain text. Use only the passages, do not use unstated background knowledge, and abstain when the passages do not support a yes/no judgment."},
            {"role": "user", "content": f"Glossary: {glossary}\nQuestion: {question}\nPassages:\n{context}\nReturn JSON only as {{\"answer\":\"yes|no|abstain\",\"confidence\":0.5,\"support\":[\"source-id\"]}}. Confidence is the probability that the selected answer is correct and must lie in [0.5,1]."},
        )
        record = self.chat.request(messages, purpose=f"grounded:{query.query_id}:{self.corpus_hash}")
        parsed = _last_json_object(str(record["content"]))
        prompt_tokens, completion_tokens, attempts = self._usage((record,))
        audit = {
            "strategy": self.strategy,
            "corpus_hash": self.corpus_hash,
            "retrieved_source_ids": [source_id for source_id, _ in passages],
            **self._request_audit((record,)),
        }
        try:
            if parsed is None:
                raise ValueError("missing JSON")
            answer = _answer(parsed.get("answer"))
            support = tuple(str(value) for value in parsed.get("support", ()))
            allowed = {source_id for source_id, _ in passages}
            if answer is None or not support or any(value not in allowed for value in support):
                raise ValueError("unsupported answer")
            confidence = float(parsed["confidence"])
            response = QueryResponse(
                answer, confidence, treatment="soft_evidence"
            )
        except (KeyError, TypeError, ValueError):
            return QueryResponse(None, None, prompt_tokens, completion_tokens, attempts, "malformed-abstained-or-unsupported", audit)
        audit["support_source_ids"] = list(support)
        return QueryResponse(
            response.answer,
            response.confidence,
            prompt_tokens,
            completion_tokens,
            attempts,
            audit=audit,
            treatment="soft_evidence",
        )

    def __call__(self, query: ClaimQuery) -> QueryResponse:
        if self.protocol == "separated_v2":
            if self.strategy == "stated":
                return self._stated_v2(query)
            if self.strategy == "consistency":
                return self._consistency_v2(query)
            return self._grounded_v2(query)
        if self.strategy == "stated":
            return self._stated(query)
        if self.strategy == "consistency":
            return self._consistency(query)
        return self._grounded(query)


# The elicitation prompts and parsing are model-independent. Keep the historical
# Qwen name as a compatibility alias for archived experiments.
LLMClaimConfidenceProvider = QwenClaimConfidenceProvider


def confidence_metrics(
    responses: Sequence[tuple[int, float]], *, bins: int = 10
) -> dict[str, float | int]:
    """Post-hoc event-weighted accuracy, Brier score, and equal-width ECE."""

    if bins < 1:
        raise ValueError("bin count must be positive")
    if not responses:
        return {"valid_count": 0, "accuracy": math.nan, "brier": math.nan, "ece": math.nan}
    correctness = [int(value) for value, _ in responses]
    confidences = [float(confidence) for _, confidence in responses]
    if any(value not in (0, 1) for value in correctness) or any(
        not math.isfinite(value) or not 0.5 <= value <= 1.0 for value in confidences
    ):
        raise ValueError("confidence metric inputs are invalid")
    accuracy = sum(correctness) / len(correctness)
    brier = sum((confidence - correct) ** 2 for correct, confidence in zip(correctness, confidences)) / len(correctness)
    ece = 0.0
    for index in range(bins):
        lower = 0.5 + 0.5 * index / bins
        upper = 0.5 + 0.5 * (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(confidences)
            if confidence >= lower and (confidence < upper or (index == bins - 1 and confidence <= upper))
        ]
        if members:
            bin_accuracy = sum(correctness[item] for item in members) / len(members)
            bin_confidence = sum(confidences[item] for item in members) / len(members)
            ece += len(members) / len(correctness) * abs(bin_accuracy - bin_confidence)
    return {"valid_count": len(responses), "accuracy": accuracy, "brier": brier, "ece": ece}


__all__ = [
    "CachedLocalChatClient",
    "CachedOpenRouterChatClient",
    "ConfidenceStrategy",
    "DOCUMENT_URLS",
    "DocumentSource",
    "GroundedEvidencePassage",
    "LLMClaimConfidenceProvider",
    "PromptProtocol",
    "QwenClaimConfidenceProvider",
    "confidence_metrics",
    "corpus_fingerprint",
    "load_document_corpus",
    "retrieve_passages",
]
