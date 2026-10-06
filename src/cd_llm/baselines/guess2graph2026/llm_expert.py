from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cd_llm.baselines.scp2025.client import build_chat_client, is_router_model
from cd_llm.experiments.common.artifacts import read_json, write_json


@dataclass(frozen=True)
class LLMExpertSettings:
    model: str = "Qwen/Qwen3.5-9B:together"
    temperature: float = 0.7
    max_tokens: int = 2000
    enable_thinking: bool = False
    request_seed: int | None = None


@dataclass(frozen=True)
class SkeletonGuess:
    skeleton: np.ndarray
    edges: tuple[tuple[str, str], ...]
    prompt: str
    response: str
    shuffled_nodes: tuple[str, ...]
    cache_path: Path


@dataclass(frozen=True)
class DAGGuess:
    dag: np.ndarray
    edges: tuple[tuple[str, str], ...]
    prompt: str
    response: str
    shuffled_nodes: tuple[str, ...]
    cache_path: Path


class CachedOpenAISkeletonExpert:
    """Generate one paper-style undirected skeleton guess per trial."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        settings: LLMExpertSettings | None = None,
        client=None,
        base_url: str | None = None,
        api_key: str | None = None,
        domain_context: str = "a clinical system",
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings or LLMExpertSettings()
        self.domain_context = domain_context
        self.descriptions = dict(descriptions or {})
        if client is None:
            client = build_chat_client(
                model=self.settings.model,
                base_url=base_url,
                api_key=api_key,
            )
        self.client = client

    def guess(self, nodes: Sequence[str], *, trial_seed: int) -> SkeletonGuess:
        canonical_nodes = tuple(nodes)
        if len(canonical_nodes) < 2 or len(set(canonical_nodes)) != len(canonical_nodes):
            raise ValueError("Expert nodes must contain at least two unique names.")

        shuffled_nodes = list(canonical_nodes)
        random.Random(trial_seed).shuffle(shuffled_nodes)
        prompt = build_skeleton_prompt(
            shuffled_nodes,
            descriptions=self.descriptions,
            domain_context=self.domain_context,
        )
        request = {
            "kind": "g2g_skeleton_guess",
            "nodes": canonical_nodes,
            "shuffled_nodes": shuffled_nodes,
            "trial_seed": trial_seed,
            "prompt": prompt,
            "settings": asdict(self.settings),
        }
        cache_path = self._cache_path(request)
        if cache_path.exists():
            payload = read_json(cache_path)
            response_text = str(payload["response"])
            edges = parse_skeleton_response(payload["parsed"], canonical_nodes)
        else:
            request_seed = (
                None
                if self.settings.request_seed is None
                else self.settings.request_seed + int(trial_seed)
            )
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful expert in causal inference and "
                            f"{self.domain_context}."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                **_json_mode_kwargs(self.settings.model),
                **_thinking_kwargs(
                    self.settings.model, self.settings.enable_thinking
                ),
                **_temperature_kwargs(
                    self.settings.model,
                    self.settings.temperature,
                ),
                **_token_limit_kwargs(
                    self.settings.model,
                    self.settings.max_tokens,
                ),
                **({"seed": request_seed} if request_seed is not None else {}),
            )
            response_text = str(response.choices[0].message.content or "")
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{self.settings.model} did not return valid JSON for the "
                    f"G2G guess. Response began: {response_text[:200]!r}"
                ) from error
            edges = parse_skeleton_response(parsed, canonical_nodes)
            usage = getattr(response, "usage", None)
            write_json(
                cache_path,
                {
                    "request": request,
                    "response": response_text,
                    "parsed": {"edges": [list(edge) for edge in edges]},
                    "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                },
            )

        return SkeletonGuess(
            skeleton=edges_to_skeleton(canonical_nodes, edges),
            edges=edges,
            prompt=prompt,
            response=response_text,
            shuffled_nodes=tuple(shuffled_nodes),
            cache_path=cache_path,
        )

    def _cache_path(self, request: Mapping[str, Any]) -> Path:
        serialized = json.dumps(request, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"


def _json_mode_kwargs(model: str) -> dict[str, Any]:
    """Ask for JSON without assuming every router backend implements the flag.

    The prompt already pins the schema, so a backend that rejects
    response_format still produces parseable output; one that supports it just
    gets a stronger guarantee.
    """
    return {"response_format": {"type": "json_object"}}


def _thinking_kwargs(model: str, enable_thinking: bool) -> dict[str, Any]:
    if enable_thinking or not is_router_model(model):
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _temperature_kwargs(model: str, temperature: float) -> dict[str, float]:
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": temperature}


def _token_limit_kwargs(model: str, token_limit: int) -> dict[str, int]:
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": max(token_limit, 4096)}
    return {"max_tokens": token_limit}


def build_skeleton_prompt(
    nodes: Sequence[str],
    *,
    descriptions: Mapping[str, str] | None = None,
    domain_context: str = "a clinical system",
) -> str:
    metadata = descriptions or {}
    variable_names = "\n".join(
        f"- {node}: {metadata[node]}" if node in metadata else f"- {node}"
        for node in nodes
    )
    return (
        f"Analyze variables from {domain_context} step by step. "
        f"The variables are:\n{variable_names}\n\n"
        "Step 1: Consider each variable's clinical meaning.\n"
        "Step 2: Identify which variables can have a direct causal relationship.\n"
        "Step 3: Distinguish direct relationships from associations mediated by another "
        "listed variable.\n"
        "Step 4: Include plausible direct pathological, anatomical, or symptomatic "
        "relationships regardless of direction.\n\n"
        "For each pair, ask whether one variable can directly influence the other. "
        "Return all predicted direct adjacencies as undirected pairs. Be comprehensive; "
        "missing a direct edge is worse than including an uncertain one.\n\n"
        'Return only JSON in this schema: {"edges": [["variable 1", "variable 2"], ...]}. '
        "Use the variable names exactly as provided. List each unordered pair at most once."
    )


def build_dag_prompt(
    nodes: Sequence[str],
    *,
    descriptions: Mapping[str, str] | None = None,
    domain_context: str = "a clinical system",
) -> str:
    """Request a directed acyclic expert guess using unambiguous identifiers."""
    metadata = descriptions or {}
    variable_names = "\n".join(
        f"- {node}: {metadata[node]}" if node in metadata else f"- {node}"
        for node in nodes
    )
    return (
        f"Infer a directed acyclic causal graph for variables from {domain_context}. "
        f"The variables are:\n{variable_names}\n\n"
        "Include an edge only for a plausible direct causal influence relative to the "
        "listed variables: indirect influence through another listed variable is not a "
        "direct edge. In each pair [source, target], source directly causes target. "
        "Before writing the response, construct an internal causal ordering of all "
        "variables and include only edges that point forward in that ordering. Never "
        "include both [A, B] and [B, A]. Check the complete edge list and remove the "
        "least plausible edge from every remaining directed cycle. Isolated variables "
        "need not be listed. The internal ordering is only an acyclicity check and must "
        "not be included in the response.\n\n"
        'Return only JSON in this schema: {"directed_edges": '
        '[["source variable", "target variable"], ...]}. '
        "Use the variable identifiers exactly as provided."
    )


def parse_skeleton_response(
    payload: Any,
    nodes: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("edges"), list):
        raise ValueError("LLM skeleton response must be an object containing an 'edges' list.")

    known = set(nodes)
    canonical_position = {node: index for index, node in enumerate(nodes)}
    normalized: set[tuple[str, str]] = set()
    for raw_edge in payload["edges"]:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f"Invalid LLM edge pair: {raw_edge!r}")
        left, right = (str(raw_edge[0]).strip(), str(raw_edge[1]).strip())
        if left not in known or right not in known:
            raise ValueError(f"LLM edge contains an unknown variable: {(left, right)!r}")
        if left == right:
            raise ValueError(f"LLM edge contains a self-loop: {(left, right)!r}")
        if canonical_position[left] > canonical_position[right]:
            left, right = right, left
        normalized.add((left, right))
    return tuple(
        sorted(
            normalized,
            key=lambda edge: (
                canonical_position[edge[0]],
                canonical_position[edge[1]],
            ),
        )
    )


def edges_to_skeleton(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> np.ndarray:
    position = {node: index for index, node in enumerate(nodes)}
    skeleton = np.zeros((len(nodes), len(nodes)), dtype=int)
    for left, right in edges:
        left_index = position[left]
        right_index = position[right]
        skeleton[left_index, right_index] = 1
        skeleton[right_index, left_index] = 1
    return skeleton


def parse_dag_response(
    payload: Any,
    nodes: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Parse a directed edge list and require a valid DAG over ``nodes``."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("directed_edges"), list):
        raise ValueError(
            "LLM DAG response must be an object containing a 'directed_edges' list."
        )
    known = set(nodes)
    position = {node: index for index, node in enumerate(nodes)}
    normalized: set[tuple[str, str]] = set()
    for raw_edge in payload["directed_edges"]:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f"Invalid directed edge: {raw_edge!r}")
        source, target = (str(raw_edge[0]).strip(), str(raw_edge[1]).strip())
        if source not in known or target not in known:
            raise ValueError(
                f"LLM directed edge contains an unknown variable: {(source, target)!r}"
            )
        if source == target:
            raise ValueError(f"LLM directed edge contains a self-loop: {(source, target)!r}")
        normalized.add((source, target))
    edges = tuple(sorted(normalized, key=lambda edge: (position[edge[0]], position[edge[1]])))
    edges_to_dag(nodes, edges)
    return edges


def edges_to_dag(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> np.ndarray:
    """Build and validate a binary DAG adjacency matrix."""
    position = {node: index for index, node in enumerate(nodes)}
    dag = np.zeros((len(nodes), len(nodes)), dtype=int)
    for source, target in edges:
        if source not in position or target not in position:
            raise ValueError(f"Unknown DAG endpoint: {(source, target)!r}")
        if source == target:
            raise ValueError(f"DAG contains a self-loop: {(source, target)!r}")
        dag[position[source], position[target]] = 1
    indegree = dag.sum(axis=0).astype(int)
    queue = [index for index, value in enumerate(indegree) if value == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(dag[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    if visited != len(nodes):
        raise ValueError("LLM directed graph contains a cycle")
    return dag
