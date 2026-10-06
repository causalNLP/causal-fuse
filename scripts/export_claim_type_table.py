"""Rebuild the Section 4 claim-type accuracy/proportion table from fresh runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from cd_llm.core.finite_support import graph_from_edges  # noqa: E402
from cd_llm.methods.active_query import ClaimQuery, query_values  # noqa: E402


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def truth(root: Path):
    payload = read_json(root / "evaluation/truth_graph.json")
    graph = payload.get("graph", payload)
    node_count = payload.get("node_count") or graph.get("node_count")
    if node_count is None:
        node_count = 1 + max(node for edge in graph["directed"] for node in edge)
    return graph_from_edges(
        int(node_count),
        directed=graph["directed"],
    )


def rows(root: Path, dataset: str, model: str) -> list[dict]:
    manifest = read_json(root / "manifest.json")
    target = truth(root)
    grouped = defaultdict(list)
    selected = [task for task in manifest["tasks"] if task["group"] == "all"]
    strategies = {task["strategy"] for task in selected}
    if len(selected) != 20 * len(strategies):
        raise ValueError(f"{root}: incomplete all-claim trajectory grid")
    for task in selected:
        record = read_json(root / "records" / f"{task['id']}.json")
        trace = record["trace"][:100]
        if len(trace) != 100:
            raise ValueError(f"{root}: incomplete 100-attempt trajectory {task['id']}")
        for step in trace:
            query = ClaimQuery(**step["query"])
            family = "ancestral" if query.kind == "ancestry" else "local"
            correct = None
            if step["valid_response"]:
                correct = int(step["answer"] == int(query_values((target,), query)[0]))
            grouped[task["sample_size"], task["strategy"], family].append(correct)
    result = []
    for (sample_size, strategy, family), values in sorted(grouped.items()):
        valid = [value for value in values if value is not None]
        total = sum(len(grouped[sample_size, strategy, current]) for current in ("local", "ancestral"))
        if not valid or total != 1000:
            raise ValueError(f"{root}: incomplete claim-type cell")
        result.append(dict(
            dataset=dataset, model=model, sample_size=sample_size,
            confidence=strategy, claim_type=family,
            accuracy=sum(valid) / len(valid), proportion=len(values) / total,
            attempted=len(values), valid=len(valid), seeds=10,
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuropathic-qwen", type=Path, required=True)
    parser.add_argument("--neuropathic-terra", type=Path, required=True)
    parser.add_argument("--alarm-stated", type=Path, required=True)
    parser.add_argument("--alarm-consistency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (
        rows(args.neuropathic_qwen, "neuropathic_pain", "Qwen3.5-9B")
        + rows(args.neuropathic_terra, "neuropathic_pain", "GPT-5.6 Terra")
        + rows(args.alarm_stated, "alarm", "Qwen3.5-9B")
        + rows(args.alarm_consistency, "alarm", "Qwen3.5-9B")
    )
    # Each ALARM source contributes one strategy; reject duplicates and omissions.
    unique = {(r["dataset"], r["model"], r["sample_size"], r["confidence"], r["claim_type"]): r for r in output}
    if len(unique) != 24:
        raise ValueError(f"claim-type table requires 24 unique cells; found {len(unique)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(unique.values()))))
        writer.writeheader()
        writer.writerows(unique[key] for key in sorted(unique))
    print(args.output)


if __name__ == "__main__":
    main()
