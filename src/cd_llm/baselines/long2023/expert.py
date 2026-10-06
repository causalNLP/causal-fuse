"""Pairwise language-model expert used by the Long et al. adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Callable

from .algorithm import Expert


VERBS = (
    "provokes",
    "triggers",
    "causes",
    "leads to",
    "induces",
    "results in",
    "brings about",
    "yields",
    "generates",
    "initiates",
    "produces",
    "stimulates",
    "instigates",
    "fosters",
    "engenders",
    "promotes",
    "catalyzes",
    "gives rise to",
    "spurs",
    "sparks",
)


def prompt_for(first: str, second: str, verb: str, swapped: bool) -> str:
    left, right = (second, first) if swapped else (first, second)
    return (
        "Among these two options which one is the most likely true:\n"
        f"(A) {left} {verb} {right}\n(B) {right} {verb} {left}\n"
        "Answer with exactly A or B."
    )


def openrouter_expert(
    model: str,
    base_url: str | None,
    prompt_seed: int,
    cache_path: Path,
    *,
    enable_thinking: bool = True,
    prompt_auditor: Callable[[str], None] | None = None,
) -> Expert:
    """Construct the paper-style A/B expert with cached token probabilities."""
    from openai import OpenAI

    loopback = bool(base_url and base_url.startswith("http://127.0.0.1:"))
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ("EMPTY" if loopback else None)
    )
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY or OPENROUTER_API_KEY is required for Long2023"
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )

    def decide(first: int, second: int, first_name: str, second_name: str):
        digest = hashlib.sha256(
            f"{prompt_seed}|{first_name}|{second_name}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        verb, swapped = rng.choice(VERBS), bool(rng.randrange(2))
        prompt = prompt_for(first_name, second_name, verb, swapped)
        if prompt_auditor is not None:
            prompt_auditor(prompt)
        key = hashlib.sha256(
            json.dumps(
                {
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0,
                    "reasoning_effort": "none",
                    "require_provider_parameters": True,
                    "enable_thinking": enable_thinking,
                    "transport": "loopback" if loopback else "router",
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if key in cache:
            saved = cache[key]
            return tuple(saved["arc"]), float(saved["confidence"]), {
                **saved["record"],
                "cache_hit": True,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if loopback and not enable_thinking
            else {
                "reasoning": {"effort": "none"},
                "provider": {"require_parameters": True},
            }
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body=extra_body,
        )
        token = (response.choices[0].message.content or "").strip().upper()
        if token not in {"A", "B"}:
            raise ValueError(f"invalid Long expert answer: {token!r}")
        logprobs = response.choices[0].logprobs
        if logprobs is None or not logprobs.content:
            raise ValueError(
                "selected provider returned no log-probabilities; model is "
                "incompatible with the native Long expert interface"
            )
        masses = {"A": 0.0, "B": 0.0}
        for item in logprobs.content[0].top_logprobs:
            option = item.token.strip().strip("() ").upper()
            if option in masses:
                masses[option] += math.exp(item.logprob)
        total = masses["A"] + masses["B"]
        if total <= 0 or masses[token] <= 0:
            raise ValueError("provider did not return comparable A/B log-probabilities")
        confidence = max(0.5, masses[token] / total)
        left, right = (second, first) if swapped else (first, second)
        arc = (left, right) if token == "A" else (right, left)
        usage = response.usage
        record = {
            "prompt": prompt,
            "answer": token,
            "verb": verb,
            "swapped": swapped,
            "input_tokens": int(usage.prompt_tokens or 0),
            "output_tokens": int(usage.completion_tokens or 0),
            "cache_hit": False,
        }
        cache[key] = {"arc": list(arc), "confidence": confidence, "record": record}
        _write_cache(cache_path, cache)
        return arc, confidence, record

    return decide


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
