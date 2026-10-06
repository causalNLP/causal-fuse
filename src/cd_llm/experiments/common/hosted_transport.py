"""Shared hosted model request accounting and BFS message compaction."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

BFS_CONTEXT_CHARACTER_LIMIT = 32_000

def _compact_bfs_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Bound BFS context while retaining every input needed for the current step."""

    normalized = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in messages
    ]
    if sum(len(item["content"]) for item in normalized) <= BFS_CONTEXT_CHARACTER_LIMIT:
        return normalized
    if len(normalized) < 3:
        return normalized

    prefix = normalized[:2]
    user_turns = [item for item in normalized[2:] if item["role"] == "user"]
    if not user_turns:
        return prefix
    current = user_turns[-1]
    if current["content"].startswith("Your previous response did not follow") and len(user_turns) > 1:
        current = {
            "role": "user",
            "content": user_turns[-2]["content"] + "\n\n" + current["content"],
        }
    return [*prefix, current]



def load_key(path=None):
    """Read a raw key or one dotenv assignment without executing a shell file."""
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if path:
        text = Path(path).expanduser().read_text().strip()
        if text.startswith('sk-or-') and '\n' not in text:
            key = text
        else:
            entries = [line.strip().removeprefix('export ').strip() for line in text.splitlines()]
            keys = [line.split('=', 1)[1].strip().strip('\"\'') for line in entries
                    if line.startswith('OPENROUTER_API_KEY=')]
            if len(keys) != 1:
                raise ValueError('key file must contain a raw key or one OPENROUTER_API_KEY assignment')
            key = keys[0]
    if not key.startswith('sk-or-') or any(c.isspace() for c in key):
        raise ValueError('OPENROUTER_API_KEY is missing or malformed')
    return key



def reserved_cost(request, prices):
    # UTF-8 byte count is a conservative input-token estimate for byte BPE.
    prompt = sum(len(m['content'].encode('utf-8')) + 64 for m in request['messages']) + 64
    return prompt*prices['prompt'] + request['max_tokens']*prices['completion']



def accounted_cost(record, prices):
    usage = record.get('response', {}).get('usage') or {}
    if not usage:
        return record['reserved_usd']  # Unknown billing remains reserved, including failures.
    estimated = usage.get('prompt_tokens', 0)*prices['prompt'] + usage.get('completion_tokens', 0)*prices['completion']
    return max(float(usage.get('cost') or 0), estimated)
