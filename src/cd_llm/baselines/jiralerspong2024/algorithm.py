"""Algorithm and prompts for the Jiralerspong et al. (2024) BFS baseline."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


Complete = Callable[[Sequence[dict[str, str]]], Completion]


def parse_answer(
    text: str, valid_names: Sequence[str], *, drop_unknown: bool = False
) -> list[str]:
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, flags=re.I | re.S)
    if not matches:
        raise ValueError("response has no <Answer>...</Answer> block")
    body = matches[-1].strip()
    if not body or body.lower() in {"none", "no variables", "n/a"}:
        return []
    candidates = [part.strip().strip("'\"") for part in body.split(",")]
    unknown = [name for name in candidates if name and name not in valid_names]
    if unknown:
        if not drop_unknown:
            raise ValueError(f"response contains unknown variables: {unknown}")
        warnings.warn(
            f"discarding variables outside the locked vocabulary: {unknown}",
            RuntimeWarning,
            stacklevel=2,
        )
        candidates = [name for name in candidates if name in valid_names]
    return list(dict.fromkeys(name for name in candidates if name))


def _format_statistics(node: str, correlations: pd.DataFrame) -> str:
    rows = [
        f"{other}: {correlations.loc[other, node]:.2f}"
        for other in correlations.columns
        if other != node
    ]
    return "\nPearson correlations with this variable:\n" + "\n".join(rows)


def creates_cycle(
    edges: Sequence[tuple[int, int]], source: int, target: int, node_count: int
) -> bool:
    if source == target:
        return True
    children = [[] for _ in range(node_count)]
    for parent, child in edges:
        children[parent].append(child)
    children[source].append(target)
    stack = [target]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(children[node])
    return False


def run_bfs(
    frame: pd.DataFrame,
    complete: Complete,
    *,
    include_statistics: bool,
    descriptions: Mapping[str, str] | None = None,
    domain_context: str = "a causal system",
    opaque_identifiers: bool = False,
    empty_root_policy: Literal["error", "empty_graph"] = "error",
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Run Algorithm 1 and the prompts in Sections 3.1--3.2 of the paper."""
    if empty_root_policy not in ("error", "empty_graph"):
        raise ValueError("unknown empty-root policy")
    names = list(frame.columns)
    description_map = {} if descriptions is None else descriptions
    description_text = "\n".join(
        f"{name}: {description_map.get(name, name)}" for name in names
    )
    if opaque_identifiers:
        answer_format = (
            "Think step by step, then give only the opaque identifiers shown to "
            "the left of each colon inside <Answer>...</Answer>, separated by "
            "commas. Never put a description inside the answer block. Use "
            "<Answer>None</Answer> when the set is empty. Preserve identifiers exactly."
        )
    else:
        answer_format = (
            "Think step by step, then give variable names only inside "
            "<Answer>...</Answer>, separated by commas. Use <Answer>None</Answer> "
            "when the set is empty. Preserve names exactly."
        )
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"You are an expert on {domain_context} and causal graphs.",
        },
        {
            "role": "user",
            "content": (
                f"You are a helpful assistant to an expert on {domain_context}. "
                "The following factors are key variables in this domain and may have "
                "causal effects on each other. Our goal "
                f"is to construct a causal graph between these variables.\n{description_text}\n"
                "Now you are going to use the data to construct a causal graph. You "
                "will start with identifying the variable(s) that are unaffected by "
                f"any other variables. {answer_format}"
            ),
        },
    ]
    usage = {"query_count": 0, "input_tokens": 0, "output_tokens": 0}

    def ask(prompt: str | None = None) -> list[str]:
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        for attempt in range(2):
            result = complete(messages)
            usage["query_count"] += 1
            usage["input_tokens"] += result.input_tokens
            usage["output_tokens"] += result.output_tokens
            messages.append({"role": "assistant", "content": result.text})
            try:
                # The published implementation drops out-of-vocabulary answers.
                # For opaque identifiers, retry instead because a description is
                # not an empty expert answer.
                return parse_answer(
                    result.text,
                    names,
                    drop_unknown=not opaque_identifiers,
                )
            except ValueError as error:
                if attempt or (
                    "no <Answer>" not in str(error)
                    and not opaque_identifiers
                ):
                    raise
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not follow the required "
                            "answer format. Repeat it using only exact identifiers "
                            "from the left of the metadata colons, separated by "
                            "commas inside <Answer>...</Answer>, or use "
                            "<Answer>None</Answer>."
                        ),
                    }
                )
        raise AssertionError("unreachable parse-retry state")

    roots = ask()
    if not roots:
        if empty_root_policy == "empty_graph":
            usage["empty_root_response"] = 1
            return [], usage
        raise ValueError("BFS returned no root variable")
    correlations = frame.corr() if include_statistics else None
    visited: set[str] = set()
    queued: set[str] = set(roots)
    frontier = list(roots)
    edges: list[tuple[int, int]] = []
    while frontier:
        node = frontier.pop(0)
        queued.discard(node)
        if node in visited:
            continue
        visited.add(node)
        known = "\n".join(f"{names[u]} causes {names[v]}" for u, v in edges) or "None yet."
        prompt = (
            f"Given {', '.join(roots)} is(are) not affected by any other variable "
            f"and the following causal relationships:\n{known}\n"
            f"Select the variables that are caused by {node}."
        )
        if correlations is not None:
            prompt += _format_statistics(node, correlations)
        children = ask(prompt + "\n" + answer_format)
        for child in children:
            proposed = (names.index(node), names.index(child))
            if not creates_cycle(edges, *proposed, len(names)):
                edges.append(proposed)
            if child not in visited and child not in queued:
                frontier.append(child)
                queued.add(child)

    return list(dict.fromkeys(edges)), usage
