"""OpenAI-compatible client adapter for the BFS baseline."""

from __future__ import annotations

import os
from typing import Sequence

from .algorithm import Complete, Completion


def openai_completer(
    model: str,
    temperature: float,
    base_url: str | None,
    *,
    seed: int | None = None,
    enable_thinking: bool = True,
    api_key: str | None = None,
) -> Complete:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("install the optional 'openai' package to run BFS") from error
    loopback = bool(base_url and base_url.startswith("http://127.0.0.1:"))
    credential = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if loopback else None)
    if not credential:
        raise RuntimeError("OPENAI_API_KEY is required to run BFS")
    client = OpenAI(api_key=credential, base_url=base_url)

    def complete(messages: Sequence[dict[str, str]]) -> Completion:
        kwargs: dict[str, object] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if not enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            **kwargs,
        )
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            input_tokens=int(usage.prompt_tokens if usage else 0),
            output_tokens=int(usage.completion_tokens if usage else 0),
        )

    return complete
