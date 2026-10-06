"""Matched neuropathic DAG-gPC evaluation with a PC-oriented CPDAG output."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cd_llm.baselines.guess2graph2026.dag_cpdag import (
    run_native_gpc_guess_dag_cpdag,
)
from cd_llm.baselines.guess2graph2026.llm_expert import (
    build_dag_prompt,
    edges_to_dag,
    parse_dag_response,
)
from cd_llm.core.finite_support import (
    Endpoint,
    PartialGraph,
    graph_from_edges,
    point_prediction_metrics,
    stable_seed,
)
from cd_llm.experiments.common.artifacts import (
    read_json,
    sha256,
    write_csv,
    write_json,
    write_text,
)
from cd_llm.experiments.neuropathic_decontamination import audit_model_text, build_spec
from cd_llm.paths import WORKSPACE_ROOT


STUDY = "neuropathic_gpc_dag_cpdag"
SOURCE = (
    WORKSPACE_ROOT
    / "results/20260903T011254Z/neuropathic_decontamination/benchmark"
)
QWEN_RUNTIME = (
    WORKSPACE_ROOT
    / "generated/runtime/20260825T082523Z/qwen35_neuropathic20_manifest.json"
)
OLD_TERRA = WORKSPACE_ROOT / "results/20260903T011254Z/neuropathic_decontamination"
OLD_QWEN = WORKSPACE_ROOT / "results/20260903T022032Z/neuropathic_decontamination_qwen"
NAMES = tuple(f"V{index:02d}" for index in range(1, 21))
CONDITIONS = tuple((seed, size) for seed in range(10) for size in (200, 1000))
MODEL_SPECS = {
    "qwen": {
        "model": "Qwen/Qwen3.5-9B",
        "display": "Qwen3.5-9B",
        "max_tokens": 2000,
        "seed_base": stable_seed("neuropathic_decontamination_qwen", "guess2graph"),
    },
    "terra": {
        "model": "openai/gpt-5.6-terra",
        "display": "GPT-5.6 Terra",
        "max_tokens": 4096,
        "seed_base": 0,
    },
}
MODEL_KEYS = tuple(MODEL_SPECS)
MAX_PARSE_ATTEMPTS = 3


def _partial_payload(graph: PartialGraph) -> dict[str, Any]:
    return {
        "kind": graph.kind,
        "node_count": graph.node_count,
        "marks": list(graph.marks),
    }


def _partial(payload: Mapping[str, Any]) -> PartialGraph:
    return PartialGraph(
        int(payload["node_count"]),
        tuple(int(value) for value in payload["marks"]),
        str(payload["kind"]),
    )


def _metadata(root: Path) -> tuple[dict[str, str], str]:
    payload = read_json(root / "inputs/model_metadata.json")
    names = tuple(str(item["id"]) for item in payload["nodes"])
    if names != NAMES:
        raise ValueError("model metadata node order differs from Table 1")
    return (
        {str(item["id"]): str(item["description"]) for item in payload["nodes"]},
        str(payload["domain_context"]),
    )


def _implementation_paths() -> tuple[Path, ...]:
    relative = (
        "code/src/cd_llm/baselines/guess2graph2026/dag_cpdag.py",
        "code/src/cd_llm/baselines/guess2graph2026/llm_expert.py",
        "code/src/cd_llm/experiments/neuropathic_gpc_dag_cpdag.py",
        "sbatch/neuropathic_gpc_dag_cpdag_qwen.sbatch",
        "sbatch/neuropathic_gpc_dag_cpdag_terra.sbatch",
        "sbatch/neuropathic_gpc_dag_cpdag_cpu.sbatch",
    )
    return tuple(WORKSPACE_ROOT / name for name in relative)


def prepare(root: Path) -> dict[str, Any]:
    if (root / "manifest.json").exists():
        return verify(root)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("inputs/data", "prompts", "responses", "experts", "predictions"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    copies = {
        SOURCE / "model_metadata.json": root / "inputs/model_metadata.json",
        SOURCE / "truth_graph.json": root / "inputs/truth_graph.json",
    }
    for seed, size in CONDITIONS:
        name = f"seed_{seed:02d}_n{size}.csv"
        copies[SOURCE / "data" / name] = root / "inputs/data" / name
    for source, destination in copies.items():
        shutil.copyfile(source, destination)

    descriptions, context = _metadata(root)
    prompt_hashes: dict[str, str] = {}
    for model_key in MODEL_KEYS:
        for seed in range(10):
            shuffled = list(NAMES)
            random.Random(seed).shuffle(shuffled)
            prompt = build_dag_prompt(
                shuffled, descriptions=descriptions, domain_context=context
            )
            audit_model_text(prompt, build_spec())
            payload = {
                "model_key": model_key,
                "seed": seed,
                "shuffled_nodes": shuffled,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful expert in causal inference and " + context + "."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            path = root / "prompts" / model_key / f"seed_{seed:02d}.json"
            write_json(path, payload)
            prompt_hashes[str(path.relative_to(root))] = sha256(path)

    runtime = read_json(QWEN_RUNTIME)
    manifest = {
        "study": STUDY,
        "source": str(SOURCE),
        "source_manifest_sha256": sha256(SOURCE / "evaluation_manifest.json"),
        "conditions": [list(value) for value in CONDITIONS],
        "models": MODEL_SPECS,
        "qwen_runtime": {
            "path": str(QWEN_RUNTIME),
            "sha256": sha256(QWEN_RUNTIME),
            "model_revision": runtime["model_revision"],
            "image_sha256": runtime["image_sha256"],
        },
        "settings": {
            "expert_dags": "one per model and data seed, shared across sample sizes",
            "temperature": 0.7,
            "max_parse_attempts": MAX_PARSE_ATTEMPTS,
            "alpha": 0.05,
            "ci_test": "chisq",
            "edge_order": "expert nonedges then expert edges; blocks shuffled",
            "conditioning_order": "upstream topological two-block order",
            "orientation": "UC-sepset priority 2, Meek closure, certified CPDAG",
        },
        "inputs": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted((root / "inputs").rglob("*"))
            if path.is_file()
        },
        "prompts": prompt_hashes,
        "implementation": {
            str(path.relative_to(WORKSPACE_ROOT)): sha256(path)
            for path in _implementation_paths()
        },
    }
    write_json(root / "manifest.json", manifest)
    _status(root, "prepared")
    return verify(root)


def verify(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if manifest.get("study") != STUDY:
        raise ValueError("result root belongs to another study")
    if manifest.get("conditions") != [list(value) for value in CONDITIONS]:
        raise ValueError("condition grid changed")
    for section in ("inputs", "prompts"):
        for name, expected in manifest[section].items():
            if sha256(root / name) != expected:
                raise ValueError(f"frozen {section[:-1]} changed: {name}")
    for name, expected in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != expected:
            raise ValueError(f"frozen implementation changed: {name}")
    return manifest


def _status(root: Path, stage: str, **extra: Any) -> None:
    write_json(
        root / "status.json",
        {
            "stage": stage,
            "expert_dags": sum(1 for _ in (root / "experts").rglob("seed_*.json")),
            "expert_dags_total": 20,
            "predictions": sum(1 for _ in (root / "predictions").rglob("*.json")),
            "predictions_total": 40,
            "updated": time.time(),
            **extra,
        },
    )


def _request(model_key: str, messages: list[dict[str, str]], seed: int) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    request: dict[str, Any] = {
        "model": spec["model"],
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
        "seed": int(spec["seed_base"]) + seed,
    }
    if model_key == "qwen":
        request["max_tokens"] = spec["max_tokens"]
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        request["max_completion_tokens"] = spec["max_tokens"]
        request["reasoning_effort"] = "none"
    return request


def _response_usage(response: Mapping[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _query(client: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    last: Exception | None = None
    for retry in range(3):
        try:
            return client.chat.completions.create(**request).model_dump()
        except Exception as error:  # provider and transport exceptions vary
            last = error
            if retry == 2:
                break
            time.sleep(5 * (retry + 1))
    assert last is not None
    raise last


def collect(root: Path, model_key: str, base_url: str) -> None:
    verify(root)
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model key: {model_key}")
    if model_key == "terra":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
    else:
        key = "EMPTY"
        if not base_url.startswith("http://127.0.0.1:") or not base_url.endswith("/v1"):
            raise ValueError("Qwen collection requires a local loopback /v1 endpoint")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=300.0)
    _status(root, f"collecting-{model_key}")
    for seed in range(10):
        expert_path = root / "experts" / model_key / f"seed_{seed:02d}.json"
        if expert_path.exists():
            continue
        prompt_path = root / "prompts" / model_key / f"seed_{seed:02d}.json"
        prompt = read_json(prompt_path)
        total_input = total_output = 0
        failures: list[dict[str, Any]] = []
        for attempt in range(MAX_PARSE_ATTEMPTS):
            response_path = (
                root / "responses" / model_key / f"seed_{seed:02d}_attempt_{attempt}.json"
            )
            request = _request(model_key, prompt["messages"], seed + 10 * attempt)
            if response_path.exists():
                saved = read_json(response_path)
                if saved["request"] != request:
                    raise ValueError(f"cached request changed: {response_path}")
                response = saved["response"]
            else:
                started = time.perf_counter()
                response = _query(client, request)
                write_json(
                    response_path,
                    {
                        "model_key": model_key,
                        "seed": seed,
                        "attempt": attempt,
                        "request": request,
                        "response": response,
                        "latency_seconds": time.perf_counter() - started,
                    },
                )
            input_tokens, output_tokens = _response_usage(response)
            total_input += input_tokens
            total_output += output_tokens
            choice = response["choices"][0]
            raw = str(choice["message"].get("content") or "")
            try:
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"response finish reason is {choice.get('finish_reason')!r}")
                edges = parse_dag_response(json.loads(raw), NAMES)
                dag = edges_to_dag(NAMES, edges)
            except (ValueError, json.JSONDecodeError) as error:
                failures.append({"attempt": attempt, "error": repr(error)})
                continue
            write_json(
                expert_path,
                {
                    "status": "complete",
                    "model_key": model_key,
                    "model": MODEL_SPECS[model_key]["model"],
                    "seed": seed,
                    "successful_attempt": attempt,
                    "directed_edges": [list(edge) for edge in edges],
                    "dag": dag.tolist(),
                    "edge_count": len(edges),
                    "shuffled_nodes": prompt["shuffled_nodes"],
                    "prompt_sha256": sha256(prompt_path),
                    "response_path": str(response_path.relative_to(root)),
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_input + total_output,
                    "logical_requests": attempt + 1,
                    "failures": failures,
                },
            )
            break
        else:
            raise RuntimeError(
                f"{model_key} seed {seed} produced no valid DAG in "
                f"{MAX_PARSE_ATTEMPTS} attempts: {failures}"
            )
    _status(root, f"collected-{model_key}")


def audit(root: Path) -> dict[str, Any]:
    manifest = verify(root)
    spec = build_spec()
    attempts = 0
    for model_key in MODEL_KEYS:
        for seed in range(10):
            expert_path = root / "experts" / model_key / f"seed_{seed:02d}.json"
            if not expert_path.exists():
                raise ValueError(f"missing expert DAG: {model_key} seed {seed}")
            expert = read_json(expert_path)
            dag = edges_to_dag(NAMES, [tuple(edge) for edge in expert["directed_edges"]])
            if dag.tolist() != expert["dag"]:
                raise ValueError(f"expert DAG reconstruction differs: {expert_path}")
            prompt_path = root / "prompts" / model_key / f"seed_{seed:02d}.json"
            prompt = read_json(prompt_path)
            audit_model_text(json.dumps(prompt["messages"], ensure_ascii=False), spec)
            if expert["prompt_sha256"] != sha256(prompt_path):
                raise ValueError(f"expert prompt hash differs: {expert_path}")
            for attempt in range(int(expert["successful_attempt"]) + 1):
                response_path = (
                    root / "responses" / model_key / f"seed_{seed:02d}_attempt_{attempt}.json"
                )
                saved = read_json(response_path)
                expected = _request(model_key, prompt["messages"], seed + 10 * attempt)
                if saved["request"] != expected:
                    raise ValueError(f"request identity differs: {response_path}")
                response = saved["response"]
                if response.get("model") not in (MODEL_SPECS[model_key]["model"], None):
                    raise ValueError(f"response model differs: {response_path}")
                input_tokens, output_tokens = _response_usage(response)
                if input_tokens < 1 or output_tokens < 1:
                    raise ValueError(f"missing usage accounting: {response_path}")
                if model_key == "qwen":
                    details = (response.get("usage") or {}).get("completion_tokens_details") or {}
                    if int(details.get("reasoning_tokens") or 0):
                        raise ValueError(f"Qwen reasoning tokens present: {response_path}")
                attempts += 1
    result = {
        "status": "passed",
        "expert_dags": 20,
        "response_attempts": attempts,
        "manifest_sha256": sha256(root / "manifest.json"),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
    }
    write_json(root / "audit.json", result)
    _status(root, "audited")
    return result


def _condition(index: int) -> tuple[str, int, int]:
    if not 0 <= index < len(MODEL_KEYS) * len(CONDITIONS):
        raise ValueError("condition index must be in 0..39")
    model_key = MODEL_KEYS[index // len(CONDITIONS)]
    seed, size = CONDITIONS[index % len(CONDITIONS)]
    return model_key, seed, size


def _rng(model_key: str, seed: int, size: int) -> np.random.Generator:
    if model_key == "terra":
        return np.random.default_rng(seed)
    return np.random.default_rng(
        stable_seed("neuropathic_decontamination_qwen", "gpc", seed, size)
    )


def infer(root: Path, index: int) -> None:
    verify(root)
    if not (root / "audit.json").exists():
        raise ValueError("inference requires a complete expert-DAG audit")
    model_key, seed, size = _condition(index)
    output = root / "predictions" / model_key / f"seed_{seed:02d}_n{size}.json"
    if output.exists():
        return
    data_path = root / "inputs/data" / f"seed_{seed:02d}_n{size}.csv"
    frame = pd.read_csv(data_path)
    if tuple(frame.columns) != NAMES or frame.shape != (size, 20):
        raise ValueError(f"dataset layout differs: {data_path}")
    expert_path = root / "experts" / model_key / f"seed_{seed:02d}.json"
    expert = read_json(expert_path)
    started = time.perf_counter()
    result = run_native_gpc_guess_dag_cpdag(
        frame.to_numpy(dtype=np.int64),
        np.asarray(expert["dag"], dtype=int),
        alpha=0.05,
        ci_test="chisq",
        rng=_rng(model_key, seed, size),
        node_names=NAMES,
    )
    truth_payload = read_json(root / "inputs/truth_graph.json")["graph"]
    truth = graph_from_edges(20, directed=truth_payload["directed"])
    metrics = point_prediction_metrics(result.cpdag, truth, normalized=False)
    skeleton_marks = np.where(
        result.skeleton != 0, int(Endpoint.TAIL), int(Endpoint.NULL)
    )
    skeleton_metrics = point_prediction_metrics(
        PartialGraph.from_matrix(skeleton_marks, "cpdag"), truth, normalized=False
    )
    write_json(
        output,
        {
            "status": "complete",
            "method": "DAG-gPC-Guess + PC orientation",
            "model_key": model_key,
            "model": MODEL_SPECS[model_key]["model"],
            "seed": seed,
            "sample_size": size,
            "dataset_sha256": sha256(data_path),
            "expert_sha256": sha256(expert_path),
            "skeleton": result.skeleton.tolist(),
            "separating_sets": [
                {"left": left, "right": right, "conditioning": list(conditioning)}
                for (left, right), conditioning in sorted(result.separating_sets.items())
                if left < right
            ],
            "raw_cpdag": _partial_payload(result.raw_cpdag),
            "cpdag": _partial_payload(result.cpdag),
            "topological_order": list(result.topological_order),
            "ci_tests": result.ci_tests,
            "certification_changed": result.certification_changed,
            "metrics": metrics,
            "skeleton_metrics": skeleton_metrics,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    _status(root, "inference", last_completed=index)


def _edge_counts(graph: PartialGraph) -> tuple[int, int]:
    directed = undirected = 0
    for left in range(graph.node_count):
        for right in range(left + 1, graph.node_count):
            marks = graph.mark(left, right), graph.mark(right, left)
            directed += int(marks in ((Endpoint.TAIL, Endpoint.ARROW), (Endpoint.ARROW, Endpoint.TAIL)))
            undirected += int(marks == (Endpoint.TAIL, Endpoint.TAIL))
    return directed, undirected


def _old_metrics(model_key: str, seed: int, size: int, truth: Any) -> dict[str, float]:
    if model_key == "qwen":
        path = OLD_QWEN / "records/gpc-guess_plus_qwen3.5-9b" / f"seed_{seed:02d}_n{size}.json"
    else:
        path = OLD_TERRA / "records/gpc-guess_plus_gpt-5.6_terra" / f"seed_{seed:02d}_n{size}.json"
    record = read_json(path)
    matrix = np.zeros((20, 20), dtype=int)
    for left, right in record["graph"]["edges"]:
        matrix[int(left), int(right)] = matrix[int(right), int(left)] = int(Endpoint.TAIL)
    return point_prediction_metrics(PartialGraph.from_matrix(matrix, "cpdag"), truth, normalized=False)


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def report(root: Path) -> None:
    verify(root)
    audit(root)
    paths = sorted((root / "predictions").rglob("seed_*_n*.json"))
    if len(paths) != 40:
        raise ValueError(f"report requires 40 predictions; found {len(paths)}")
    records = [read_json(path) for path in paths]
    identities = {
        (record["model_key"], int(record["seed"]), int(record["sample_size"]))
        for record in records
    }
    expected = {(model, seed, size) for model in MODEL_KEYS for seed, size in CONDITIONS}
    if identities != expected:
        raise ValueError("prediction identity set is incomplete or unexpected")
    truth_payload = read_json(root / "inputs/truth_graph.json")["graph"]
    truth = graph_from_edges(20, directed=truth_payload["directed"])
    per_seed: list[dict[str, Any]] = []
    for record in records:
        graph = _partial(record["cpdag"])
        recomputed = point_prediction_metrics(graph, truth, normalized=False)
        for name, value in recomputed.items():
            if not math.isclose(float(value), float(record["metrics"][name]), abs_tol=1e-12):
                raise ValueError(f"metric differs for {record['model_key']} {record['seed']} n={record['sample_size']}: {name}")
        directed, undirected = _edge_counts(graph)
        expert = read_json(
            root / "experts" / record["model_key"] / f"seed_{int(record['seed']):02d}.json"
        )
        per_seed.append(
            {
                "method": record["method"],
                "model": MODEL_SPECS[record["model_key"]]["display"],
                "model_key": record["model_key"],
                "seed": record["seed"],
                "sample_size": record["sample_size"],
                **recomputed,
                "directed_edges": directed,
                "undirected_edges": undirected,
                "ci_tests": record["ci_tests"],
                "certification_changed": record["certification_changed"],
                "runtime_seconds": record["runtime_seconds"],
                "input_tokens": expert["input_tokens"],
                "output_tokens": expert["output_tokens"],
                "total_tokens": expert["total_tokens"],
                "logical_requests": expert["logical_requests"],
            }
        )
    write_csv(root / "per_seed.csv", per_seed)

    metrics = (
        "expected_distance",
        "energy_score",
        "adjacency_f1",
        "orientation_f1",
        "directed_edges",
        "undirected_edges",
        "ci_tests",
        "runtime_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "logical_requests",
    )
    summary: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        display = MODEL_SPECS[model_key]["display"]
        for size in (200, 1000):
            group = [
                row for row in per_seed
                if row["model_key"] == model_key and row["sample_size"] == size
            ]
            row: dict[str, Any] = {
                "method": "DAG-gPC-Guess + PC orientation",
                "model": display,
                "sample_size": size,
                "replicate_count": len(group),
                "certification_changed_count": sum(bool(item["certification_changed"]) for item in group),
            }
            for name in metrics:
                mean, se = _mean_se([float(item[name]) for item in group])
                row[name + "_mean"] = mean
                row[name + "_se"] = se
            summary.append(row)
            old = [_old_metrics(model_key, seed, size, truth) for seed in range(10)]
            for method, values in (("Edge-version gPC-Guess", old), (row["method"], group)):
                comparison_row: dict[str, Any] = {
                    "method": method,
                    "model": display,
                    "sample_size": size,
                }
                for name in ("expected_distance", "adjacency_f1", "orientation_f1"):
                    mean, se = _mean_se([float(item[name]) for item in values])
                    comparison_row[name + "_mean"] = mean
                    comparison_row[name + "_se"] = se
                comparison.append(comparison_row)
    write_csv(root / "summary.csv", summary)
    write_csv(root / "comparison.csv", comparison)

    lines = [
        "# Neuropathic DAG-gPC-Guess CPDAG extension",
        "",
        "This is an explicit extension of upstream gPC-Guess: the native DAG-guided search returns a skeleton, after which data-derived separating sets, UC-sepset priority 2, Meek closure, and CPDAG certification supply orientations.",
        "",
        "| Model | n | Method | ED | Adj. F1 | Orient. F1 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in comparison:
        cell = lambda name: f"{row[name + '_mean']:.2f} ({row[name + '_se']:.2f})"
        lines.append(
            f"| {row['model']} | {row['sample_size']} | {row['method']} | "
            f"{cell('expected_distance')} | {cell('adjacency_f1')} | {cell('orientation_f1')} |"
        )
    unique_usage = {}
    for model_key in MODEL_KEYS:
        experts = [read_json(path) for path in sorted((root / "experts" / model_key).glob("*.json"))]
        unique_usage[model_key] = {
            "requests": sum(int(item["logical_requests"]) for item in experts),
            "input_tokens": sum(int(item["input_tokens"]) for item in experts),
            "output_tokens": sum(int(item["output_tokens"]) for item in experts),
        }
    lines += ["", "Unique elicitation usage is recorded separately from the per-dataset accounting that charges each reused expert DAG in full.", ""]
    write_text(root / "report.md", "\n".join(lines))
    write_json(
        root / "completion.json",
        {
            "complete": True,
            "expert_dags": 20,
            "predictions": 40,
            "unique_usage": unique_usage,
            "per_seed_sha256": sha256(root / "per_seed.csv"),
            "summary_sha256": sha256(root / "summary.csv"),
            "comparison_sha256": sha256(root / "comparison.csv"),
            "report_sha256": sha256(root / "report.md"),
            "manifest_sha256": sha256(root / "manifest.json"),
            "audit_sha256": sha256(root / "audit.json"),
        },
    )
    _status(root, "complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect-qwen", "collect-terra", "audit", "infer", "report")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "prepare":
        prepare(args.root)
    elif args.stage == "collect-qwen":
        collect(args.root, "qwen", args.base_url)
    elif args.stage == "collect-terra":
        collect(args.root, "terra", args.base_url)
    elif args.stage == "audit":
        print(json.dumps(audit(args.root), indent=2, sort_keys=True))
    elif args.stage == "infer":
        if args.index is None:
            raise ValueError("infer requires --index")
        infer(args.root, args.index)
    else:
        report(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
