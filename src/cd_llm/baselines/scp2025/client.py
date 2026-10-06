from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path
from typing import Literal, Mapping

from cd_llm.experiments.common.artifacts import read_json, write_json


class _RetryableYesNoResponse(RuntimeError):
    pass


ProbabilityProtocol = Literal["paper_logprobs", "sampled_vote"]

# Open-weight models are reached through Hugging Face's OpenAI-compatible
# router, so the only thing that changes is which endpoint the same client
# talks to. A repo-style id ("Qwen/Qwen3.5-9B:together", pinned to the
# Together backend) selects the router; a bare OpenAI id
# ("gpt-4-1106-preview") keeps the paper's original endpoint, so the reference
# arm stays runnable.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
_HF_TOKEN_VARIABLES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def is_router_model(model: str) -> bool:
    return "/" in model.split(":", 1)[0]


def build_chat_client(*, model: str, base_url: str | None = None, api_key: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required for LLM evaluation. Install project requirements."
        ) from error

    if base_url is None and is_router_model(model):
        base_url = os.environ.get("LLM_BASE_URL", HF_ROUTER_BASE_URL)
    if base_url is None:
        return OpenAI()
    if api_key is None:
        api_key = next(
            (os.environ[name] for name in _HF_TOKEN_VARIABLES if os.environ.get(name)),
            None,
        ) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No API token for {base_url}. Set HF_TOKEN (or pass api_key=) to reach "
            f"{model!r} through the Hugging Face router."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def _thinking_kwargs(model: str, enable_thinking: bool) -> dict[str, object]:
    """Suppress Qwen's reasoning block on router-served models.

    Left on, the decision call spends its whole token budget reasoning before
    it ever emits <yes>/<no>, so the logprob scan finds no answer token and
    burns all four retries. OpenAI models get nothing extra.
    """
    if enable_thinking or not is_router_model(model):
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


@dataclass(frozen=True)
class LLMSettings:
    model: str = "Qwen/Qwen3.5-9B:together"
    temperature: float = 0.7
    probability_samples: int = 5
    top_logprobs: int = 5
    explanation_max_tokens: int = 3000
    decision_max_tokens: int = 1500
    system_message: str = "You are a helpful assistant for causal inference."
    probability_protocol: ProbabilityProtocol = "paper_logprobs"
    enable_thinking: bool = False
    request_seed: int | None = None


class CachedOpenAIJudge:
    def __init__(
        self,
        *,
        cache_dir: Path,
        settings: LLMSettings,
        client=None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        if (
            settings.probability_protocol == "paper_logprobs"
            and not model_supports_logprobs(settings.model)
        ):
            raise ValueError(
                f"Model {settings.model!r} is configured as not supporting token "
                "log probabilities. The paper protocol requires first-token "
                "logprobs; choose a logprob-capable model. To run the non-paper "
                "binary-vote adaptation, explicitly set "
                "probability_protocol='sampled_vote'."
            )
        if client is None:
            client = build_chat_client(
                model=settings.model,
                base_url=base_url,
                api_key=api_key,
            )
        self.client = client

    @staticmethod
    def _usage(response) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def explanation(self, prompt: str) -> str:
        payload = {
            "kind": "explanation",
            "prompt": prompt,
            "settings": asdict(self.settings),
        }
        cached = self._read_cache(payload)
        if cached is not None:
            return str(cached["text"])
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": self.settings.system_message},
                {"role": "user", "content": prompt},
            ],
            **_temperature_kwargs(
                self.settings.model,
                self.settings.temperature,
            ),
            **_token_limit_kwargs(
                self.settings.model,
                self.settings.explanation_max_tokens,
            ),
            **_thinking_kwargs(self.settings.model, self.settings.enable_thinking),
            **({"seed": self.settings.request_seed} if self.settings.request_seed is not None else {}),
        )
        text = str(response.choices[0].message.content or "")
        input_tokens, output_tokens = self._usage(response)
        self._write_cache(
            payload,
            {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens},
        )
        return text

    def yes_probability(self, prompt: str) -> tuple[float, list[float]]:
        probabilities = []
        for trial in range(self.settings.probability_samples):
            payload = {
                "kind": "yes_probability",
                "prompt": prompt,
                "trial": trial,
                "settings": asdict(self.settings),
            }
            cached = self._read_cache(payload)
            if cached is not None:
                probabilities.append(float(cached["probability"]))
                continue
            if self.settings.probability_protocol == "paper_logprobs":
                probability, input_tokens, output_tokens = self._sample_logprob_probability(prompt, trial)
            else:
                probability, input_tokens, output_tokens = self._sample_yes_vote(prompt, trial)
            self._write_cache(
                payload,
                {
                    "probability": probability,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
            probabilities.append(probability)
        return sum(probabilities) / len(probabilities), probabilities

    _LOGPROB_ATTEMPTS = 4

    def _sample_logprob_probability(self, prompt: str, trial: int) -> tuple[float, int, int]:
        # At temperature 0.7 the model occasionally ignores the <yes>/<no>
        # instruction (e.g. answering just "Indirect"). Such a response has no
        # token whose log probabilities answer the question, so resample it
        # instead of scoring an unrelated distribution or killing the run.
        input_tokens = 0
        output_tokens = 0
        for attempt in range(self._LOGPROB_ATTEMPTS):
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.settings.system_message,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
                logprobs=True,
                top_logprobs=self.settings.top_logprobs,
                max_tokens=self.settings.decision_max_tokens,
                **_thinking_kwargs(
                    self.settings.model, self.settings.enable_thinking
                ),
                **(
                    {"seed": self.settings.request_seed + 10 * trial + attempt}
                    if self.settings.request_seed is not None
                    else {}
                ),
            )
            current_input, current_output = self._usage(response)
            input_tokens += current_input
            output_tokens += current_output
            try:
                return _yes_probability_from_logprobs(response), input_tokens, output_tokens
            except _RetryableYesNoResponse:
                if attempt == self._LOGPROB_ATTEMPTS - 1:
                    raise
        raise AssertionError("logprob retry loop ended without a result")

    def _sample_yes_vote(self, prompt: str, trial: int) -> tuple[float, int, int]:
        token_limit = _decision_token_limit(
            self.settings.model,
            self.settings.decision_max_tokens,
        )
        attempt_limits = (
            token_limit,
            token_limit * 4,
            token_limit * 16,
            token_limit * 64,
        )
        input_tokens = 0
        output_tokens = 0
        for attempt, attempt_limit in enumerate(attempt_limits):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.settings.system_message,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    **_temperature_kwargs(
                        self.settings.model,
                        self.settings.temperature,
                    ),
                    **_token_limit_kwargs(
                        self.settings.model,
                        attempt_limit,
                    ),
                    **_thinking_kwargs(
                        self.settings.model, self.settings.enable_thinking
                    ),
                    **(
                        {"seed": self.settings.request_seed + 10 * trial + attempt}
                        if self.settings.request_seed is not None
                        else {}
                    ),
                )
            except Exception as error:
                if (
                    attempt < len(attempt_limits) - 1
                    and _is_output_limit_error(error)
                ):
                    continue
                raise
            current_input, current_output = self._usage(response)
            input_tokens += current_input
            output_tokens += current_output
            try:
                return _yes_vote_from_response(response), input_tokens, output_tokens
            except _RetryableYesNoResponse:
                if attempt == len(attempt_limits) - 1:
                    raise
        raise AssertionError("sampled-vote retry loop ended without a result")

    def _cache_path(self, payload: Mapping[str, object]) -> Path:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, payload: Mapping[str, object]) -> dict[str, object] | None:
        path = self._cache_path(payload)
        if not path.exists():
            return None
        return read_json(path)

    def _write_cache(
        self,
        payload: Mapping[str, object],
        result: Mapping[str, object],
    ) -> None:
        path = self._cache_path(payload)
        write_json(path, {"request": dict(payload), **dict(result)})


def _temperature_kwargs(model: str, temperature: float) -> dict[str, float]:
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": temperature}


def model_supports_logprobs(model: str) -> bool:
    model_name = model.rsplit("/", 1)[-1].lower()
    return not model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def _token_limit_kwargs(model: str, token_limit: int) -> dict[str, int]:
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": token_limit}
    return {"max_tokens": token_limit}


def _decision_token_limit(model: str, configured_limit: int) -> int:
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return max(configured_limit, 128)
    return configured_limit


def _is_output_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "output limit was reached" in message
        or "try again with higher max_tokens" in message
    )


def _normalize_decision_token(token) -> str:
    """Strip the punctuation the decision prompt wraps its answer in."""
    return str(token).strip().lower().strip("<>.,:;!?")


def _yes_probability_from_logprobs(response) -> float:
    try:
        content = response.choices[0].logprobs.content
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError(
            "The paper probability protocol requires token log probabilities, "
            "but the API response did not contain them."
        ) from error
    # build_decision_prompt asks for "<yes>" or "<no>", which models emit as a
    # bare "<" first, so the answer sits at no fixed offset. Scan the emitted
    # tokens for the one carrying the answer rather than assuming a position:
    # scoring the wrong position measures how reliably the model opens with a
    # bracket, not its causal judgement, and that near-constant value collapses
    # every decision below the forbid threshold.
    for token_logprob in content:
        if _normalize_decision_token(getattr(token_logprob, "token", "")) not in (
            "yes",
            "no",
        ):
            continue
        alternatives = getattr(token_logprob, "top_logprobs", None) or []
        if not alternatives:
            # A backend that honours `logprobs` but ignores `top_logprobs`
            # returns the answer token with no alternatives attached. Summing
            # that empty set yields 0.0 for every pair, which reads as a
            # confident "no" and forbids every edge in the sweep. There is no
            # distribution to score here, so say so instead of returning a
            # number that looks like a measurement.
            raise RuntimeError(
                "The decision response carried token log probabilities but no "
                "top_logprobs alternatives, so P(yes) cannot be computed. The "
                "serving backend is ignoring top_logprobs; pin a backend that "
                "supports it (see command/preflight_qwen_router.py) or run "
                "probability_protocol='sampled_vote'."
            )
        # A populated set that simply lacks "yes" is a real, confident no.
        probability = 0.0
        for alternative in alternatives:
            if _normalize_decision_token(alternative.token) == "yes":
                probability += exp(float(alternative.logprob))
        return min(1.0, max(0.0, probability))
    emitted = [str(getattr(t, "token", "")) for t in content[:12]]
    raise _RetryableYesNoResponse(
        "No yes/no token in the decision response, so there is no position "
        f"whose log probabilities answer the question. Emitted: {emitted!r}"
    )


def _yes_vote_from_response(response) -> float:
    try:
        text = str(response.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError) as error:
        raise _RetryableYesNoResponse(
            "OpenAI sampled-vote response did not contain text."
        ) from error
    tokens = text.strip().split(maxsplit=1)
    if not tokens:
        raise _RetryableYesNoResponse(
            "OpenAI returned an empty yes/no response. Increase the decision "
            "completion-token limit."
        )
    first_token = _normalize_decision_token(tokens[0])
    if first_token == "yes":
        return 1.0
    if first_token == "no":
        return 0.0
    raise _RetryableYesNoResponse(
        f"OpenAI response did not begin with yes or no: {text!r}"
    )
