"""Faithful condition-specific DAG-guided gPC-Guess evaluation on ALARM."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from cd_llm.baselines.guess2graph2026 import (
    build_dag_prompt,
    edges_to_dag,
    parse_dag_response,
)
from cd_llm.core.finite_support import PartialGraph, graph_from_edges, point_prediction_metrics
from cd_llm.experiments.alarm_matched_disk import DiskCheckpoint
from cd_llm.experiments.alarm_matched_gpc import search_dag
from cd_llm.experiments.alarm_matched_store import (
    Checkpoint,
    MODEL,
    Store,
    Transport,
    digest,
    validate_response,
)
from cd_llm.experiments.common.artifacts import read_json, sha256, write_json
from cd_llm.paths import WORKSPACE_ROOT


SOURCE = WORKSPACE_ROOT / "results/20260921T013135Z/alarm_matched_baselines"
CONDITIONS = [(seed, n) for seed in range(1000, 1010) for n in (200, 1000)]
TEMPERATURE = 0.7
MAX_TOKENS = 2000
MAX_PARSE_ATTEMPTS = 30


def cid(index: int) -> str:
    seed, n = CONDITIONS[index]
    return f"alarm_s{seed}_n{n}"


def _metadata(root: Path) -> tuple[list[str], dict[str, str], str]:
    payload = read_json(root / "inputs/model_metadata.json")
    return (
        [row["id"] for row in payload["nodes"]],
        {row["id"]: row["description"] for row in payload["nodes"]},
        payload["domain_context"],
    )


def _implementation_paths() -> list[Path]:
    relative = [
        "code/src/cd_llm/baselines/guess2graph2026/llm_expert.py",
        "code/src/cd_llm/baselines/guess2graph2026/paper_methods.py",
        "code/src/cd_llm/experiments/alarm_gpc_dag.py",
        "code/src/cd_llm/experiments/alarm_matched_gpc.py",
        "code/src/cd_llm/experiments/alarm_matched_disk.py",
        "code/src/cd_llm/experiments/alarm_matched_store.py",
        "sbatch/alarm_gpc_dag_gpu.sbatch",
        "sbatch/alarm_gpc_dag_cpu.sbatch",
    ]
    return [WORKSPACE_ROOT / name for name in relative]


def prepare(root: Path) -> dict:
    if (root / "manifest.json").exists():
        return verify(root)
    source_manifest = read_json(SOURCE / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("inputs", "evaluation", "status", "predictions"):
        (root / folder).mkdir(exist_ok=True)
    names_to_copy = [
        "inputs/model_metadata.json",
        "evaluation/truth_graph.json",
        *[f"inputs/{cid(index)}.npy" for index in range(len(CONDITIONS))],
    ]
    for name in names_to_copy:
        source_path = SOURCE / name
        if sha256(source_path) != source_manifest["inputs"][name]:
            raise ValueError(f"source input hash mismatch: {name}")
        shutil.copyfile(source_path, root / name)
    names, descriptions, context = _metadata(root)
    prompts = []
    for index, (seed, n) in enumerate(CONDITIONS):
        shuffled = list(names)
        random.Random(index).shuffle(shuffled)
        user = build_dag_prompt(shuffled, descriptions=descriptions, domain_context=context)
        prompts.append({
            "condition": index,
            "seed": seed,
            "sample_size": n,
            "shuffled_nodes": shuffled,
            "messages": [
                {"role": "system", "content": f"You are a careful expert in causal inference and {context}."},
                {"role": "user", "content": user},
            ],
        })
    write_json(root / "inputs/prompts.json", prompts)
    runtime = read_json(SOURCE / "inputs/runtime.json")
    shutil.copyfile(SOURCE / "inputs/runtime.json", root / "inputs/runtime.json")
    implementation = {str(path.relative_to(WORKSPACE_ROOT)): sha256(path) for path in _implementation_paths()}
    manifest = {
        "study": "alarm_condition_specific_dag_gpc_guess",
        "source": str(SOURCE),
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "conditions": [list(value) for value in CONDITIONS],
        "model": MODEL,
        "runtime": runtime,
        "runtime_fingerprint": source_manifest["runtime_fingerprint"],
        "settings": {
            "expert_outputs": "one independent directed acyclic graph per observation condition",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "reasoning": False,
            "alpha": 0.05,
            "ci_test": "categorical G-squared",
            "conditioning_depth": None,
            "edge_order": "expert nonedges then expert edges; each block shuffled",
            "conditioning_order": "expert-DAG d-separators then all other sets; each block shuffled",
            "adjacency_pool": "smaller current endpoint adjacency set",
            "parse_attempt_limit": MAX_PARSE_ATTEMPTS,
        },
        "inputs": {
            str(path.relative_to(root)): sha256(path)
            for folder in ("inputs", "evaluation")
            for path in (root / folder).iterdir()
            if path.is_file()
        },
        "implementation": implementation,
    }
    write_json(root / "manifest.json", manifest)
    Store(root)
    _status(root, "prepared")
    return verify(root)


def verify(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest["conditions"] != [list(value) for value in CONDITIONS]:
        raise ValueError("condition grid changed")
    for name, expected in manifest["inputs"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    for name, expected in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != expected:
            raise ValueError(f"frozen implementation changed: {name}")
    return manifest


def _status(root: Path, stage: str, **extra) -> None:
    store = Store(root)
    dags = len(store.items("expert_dag"))
    predictions = len(store.items("prediction"))
    with store.db() as db:
        responses = db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        tokens = db.execute("SELECT COALESCE(SUM(input),0),COALESCE(SUM(output),0) FROM responses").fetchone()
    write_json(root / "status.json", {
        "stage": stage,
        "updated": time.time(),
        "expert_dags": dags,
        "expert_dags_total": 20,
        "predictions": predictions,
        "predictions_total": 20,
        "unique_responses": responses,
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        **extra,
    })


def _request(messages: list[dict], seed: int) -> dict:
    return {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def collect_one(root: Path, index: int, deadline: float) -> dict:
    store = Store(root)
    existing = store.get("expert_dag", str(index))
    if existing is not None:
        return existing
    manifest = read_json(root / "manifest.json")
    prompt = read_json(root / "inputs/prompts.json")[index]
    names, _, _ = _metadata(root)
    failures = []
    for attempt in range(MAX_PARSE_ATTEMPTS):
        unit = f"gpc_dag:{index}:attempt:{attempt}"
        transport = Transport(store, unit, manifest["runtime_fingerprint"], online=True, deadline=deadline)
        response, key = transport.query(_request(prompt["messages"], index + 20 * attempt))
        choice = response["choices"][0]
        raw = str(choice["message"].get("content") or "")
        try:
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"response did not finish: {choice.get('finish_reason')}")
            payload = json.loads(raw)
            edges = parse_dag_response(payload, names)
            dag = edges_to_dag(names, edges)
        except (ValueError, json.JSONDecodeError) as error:
            record = {"status": "invalid", "request_key": key, "error": repr(error), "raw_response": raw}
            store.put("dag_attempt", f"{index}:{attempt}", record)
            failures.append(record)
            continue
        record = {
            "status": "complete",
            "condition": index,
            "seed": CONDITIONS[index][0],
            "sample_size": CONDITIONS[index][1],
            "request_key": key,
            "attempt": attempt,
            "directed_edges": [list(edge) for edge in edges],
            "dag": dag.tolist(),
            "edge_count": len(edges),
            "shuffled_nodes": prompt["shuffled_nodes"],
            "base_prompt_sha256": digest(prompt["messages"]),
            "request_messages_sha256": digest(prompt["messages"]),
            "raw_response": raw,
        }
        store.put("dag_attempt", f"{index}:{attempt}", record)
        store.put("expert_dag", str(index), record)
        return record
    raise ValueError(f"condition {index} produced no valid DAG in {MAX_PARSE_ATTEMPTS} attempts: {failures}")


def collect(root: Path, deadline: float) -> None:
    verify(root)
    missing = [index for index in range(20) if Store(root).get("expert_dag", str(index)) is None]
    _status(root, "collecting", missing=missing)
    if missing:
        with ThreadPoolExecutor(max_workers=min(4, len(missing))) as pool:
            list(pool.map(lambda index: collect_one(root, index, deadline), missing))
    audit(root)


def audit(root: Path) -> None:
    manifest = verify(root)
    store = Store(root)
    prompts = read_json(root / "inputs/prompts.json")
    names, _, _ = _metadata(root)
    for index in range(20):
        record = store.get("expert_dag", str(index))
        if record is None or record.get("status") != "complete":
            raise ValueError(f"missing expert DAG {index}")
        dag = edges_to_dag(names, [tuple(edge) for edge in record["directed_edges"]])
        if dag.tolist() != record["dag"]:
            raise ValueError(f"expert DAG reconstruction mismatch {index}")
        if record["base_prompt_sha256"] != digest(prompts[index]["messages"]):
            raise ValueError(f"prompt identity mismatch {index}")
    with store.db() as db:
        rows = db.execute("SELECT key,request,response FROM responses").fetchall()
    requests_by_key = {key: json.loads(request_text) for key, request_text, _ in rows}
    for index in range(20):
        record = store.get("expert_dag", str(index))
        request = requests_by_key.get(record["request_key"])
        if request is None or digest(request["messages"]) != record["request_messages_sha256"]:
            raise ValueError(f"successful request identity mismatch {index}")
    for key, request_text, response_text in rows:
        request = json.loads(request_text)
        if key != digest({"request": request, "runtime": manifest["runtime_fingerprint"]}):
            raise ValueError("cache identity mismatch")
        if request["temperature"] != TEMPERATURE or request["max_tokens"] != MAX_TOKENS:
            raise ValueError("generation setting mismatch")
        if request["chat_template_kwargs"] != {"enable_thinking": False}:
            raise ValueError("reasoning setting mismatch")
        validate_response(json.loads(response_text))
    gate = {"passed": True, "expert_dags": 20, "responses": len(rows), "temperature": TEMPERATURE}
    store.put("gate", "audit", gate)
    write_json(root / "cache_audit.json", gate)
    _status(root, "collected")


def infer(root: Path, index: int, deadline: float) -> None:
    verify(root)
    store = Store(root)
    if not store.get("gate", "audit"):
        raise ValueError("inference requires a complete audited expert collection")
    if store.get("prediction", str(index)) is not None:
        return
    seed, n = CONDITIONS[index]
    data = np.load(root / f"inputs/{cid(index)}.npy", allow_pickle=False)
    if data.shape != (n, 37):
        raise ValueError("observation shape mismatch")
    dag = np.asarray(store.get("expert_dag", str(index))["dag"], dtype=int)
    rng_seed = int(digest(["alarm-gpc", seed])[:8], 16)
    started = time.monotonic()
    with DiskCheckpoint(root, index) as checkpoint:
        matrix, diagnostics = search_dag(
            data,
            dag,
            np.random.default_rng(rng_seed),
            store,
            str(index),
            deadline,
            checkpoint=checkpoint,
            return_diagnostics=True,
        )
    record = {
        "status": "complete",
        "condition": index,
        "seed": seed,
        "sample_size": n,
        "matrix": matrix.tolist(),
        "seconds": time.monotonic() - started,
        **diagnostics,
    }
    store.put("prediction", str(index), record)
    write_json(root / f"predictions/{cid(index)}.json", record)
    _status(root, "inference", last_completed=index)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path) -> None:
    verify(root)
    store = Store(root)
    predictions = store.items("prediction")
    if len(predictions) != 20:
        raise ValueError(f"report requires all 20 predictions; found {len(predictions)}")
    truth_record = read_json(root / "evaluation/truth_graph.json")
    truth = graph_from_edges(37, directed=truth_record["graph"]["directed"])
    rows = []
    for index, (seed, n) in enumerate(CONDITIONS):
        record = predictions[str(index)]
        metrics = point_prediction_metrics(
            PartialGraph.from_matrix(np.asarray(record["matrix"], dtype=int), "cpdag"),
            truth,
            normalized=False,
        )
        usage = store.usage(f"gpc_dag:{index}:attempt:")
        rows.append({
            "method": "gpc",
            "seed": seed,
            "sample_size": n,
            "status": "complete",
            **metrics,
            **usage,
            "seconds": record["seconds"],
            "ci_tests": record["ci_tests"],
            "dsep_checks": record["dsep_checks"],
            "expert_edges": store.get("expert_dag", str(index))["edge_count"],
        })
    summary = []
    metric_names = ["expected_distance", "energy_score", "adjacency_f1", "orientation_f1", "total_tokens", "input_tokens", "output_tokens", "logical_requests", "seconds", "ci_tests", "dsep_checks", "expert_edges"]
    for n in (200, 1000):
        group = [row for row in rows if row["sample_size"] == n]
        result = {"method": "gpc", "label": "gPC-Guess", "sample_size": n, "complete": len(group), "planned": 10}
        for name in metric_names:
            values = np.asarray([row[name] for row in group], dtype=float)
            result[name + "_mean"] = float(values.mean())
            result[name + "_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
        summary.append(result)
    _csv(root / "per_seed.csv", rows)
    _csv(root / "summary.csv", summary)
    completion = {
        "complete": True,
        "expert_dags": 20,
        "predictions": 20,
        "per_seed_sha256": sha256(root / "per_seed.csv"),
        "summary_sha256": sha256(root / "summary.csv"),
        "manifest_sha256": sha256(root / "manifest.json"),
    }
    write_json(root / "completion.json", completion)
    lines = [
        "Condition-specific DAG-guided gPC-Guess on ALARM",
        "",
        "| n | ED | ES | Adj. F1 | Orient. F1 | Tokens | CI tests |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        cell = lambda name: f"{row[name + '_mean']:.2f} ± {row[name + '_se']:.2f}"
        lines.append(f"| {row['sample_size']} | {cell('expected_distance')} | {cell('energy_score')} | {cell('adjacency_f1')} | {cell('orientation_f1')} | {cell('total_tokens')} | {cell('ci_tests')} |")
    lines += ["", "Each condition uses an independently generated Qwen3.5-9B expert DAG at temperature 0.7. The DAG skeleton orders edges; exact d-separation in that DAG orders conditioning sets. The reported output is the undirected skeleton returned by gPC-Guess."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    _status(root, "complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "audit", "infer", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    deadline = float(os.environ.get("ALARM_GPC_DEADLINE", "inf"))
    if args.stage == "prepare":
        prepare(args.root)
    elif args.stage == "collect":
        collect(args.root, deadline)
    elif args.stage == "audit":
        audit(args.root)
    elif args.stage == "infer":
        if args.index is None or not 0 <= args.index < 20:
            parser.error("infer requires --index in 0..19")
        infer(args.root, args.index, deadline)
    elif args.stage == "report":
        report(args.root)


if __name__ == "__main__":
    main()
