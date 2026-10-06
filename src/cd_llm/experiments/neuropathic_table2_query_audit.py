"""Indirect-pair and directness-error audit for the main neuropathic table."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from cd_llm.experiments.common.query_classification import reachability, classify

WORKSPACE = Path(__file__).resolve().parents[4]
SOURCES = {
    "GPT-5.6 Terra": WORKSPACE / "results/20260903T011254Z/neuropathic_decontamination",
    "Qwen3.5-9B": WORKSPACE / "results/20260903T022032Z/neuropathic_decontamination_qwen",
}


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(destination):
    truth_path = SOURCES["GPT-5.6 Terra"] / "benchmark/truth_graph.json"
    edges = set(map(tuple, load(truth_path)["graph"]["directed"]))
    closure = reachability(20, edges)
    indirect_pairs = [(a,b) for a in range(20) for b in closure[a] if (a,b) not in edges]
    assert len(edges) == 30 and len(indirect_pairs) == 7
    groups, events, provenance, checks = defaultdict(list), [], [], []
    for model, root in SOURCES.items():
        summary = load(root / "summary.json")
        provenance.append(dict(path=str(root / "summary.json"),sha256=sha(root / "summary.json")))
        for directory in sorted((root / "records").iterdir()):
            if not directory.is_dir() or "claim-fuse-smc-active" not in directory.name or "grounded" in directory.name:
                continue
            for path in sorted(directory.glob("seed_*.json")):
                record = load(path)
                strategy = record.get("strategy") or "stated"
                if strategy not in ("stated", "consistency"):
                    continue
                n, seed = record["sample_size"], record["seed"]
                assert n in (200,1000) and seed in range(10)
                trace = record["trace"]
                assert len(trace) == 20 and len({s["query"]["query_id"] for s in trace}) == 20
                groups[(model,n,strategy)].append(record)
                provenance.append(dict(path=str(path),sha256=sha(path)))
                for slot, step in enumerate(trace,1):
                    query = step["query"]
                    category = classify(query, edges, closure)
                    valid = step["valid_response"]
                    answer = int(step["answer"]) if valid else None
                    assert answer in (0,1,None)
                    correct = int(answer == int(category in ("direct", "adjacent"))) if valid else None
                    events.append(dict(model=model,sample_size=n,strategy=strategy,seed=seed,slot=slot,
                        **query, category=category, valid=valid, answer=answer, correct=correct,
                        confidence=step["reported_confidence"] if valid else None))
        # Reproduce the source table metrics and its confidence summaries as a
        # check that the trace selection and reference orientation are correct.
        for (m,n,strategy), records in groups.items():
            if m != model:
                continue
            assert len(records) == 10 and {r["seed"] for r in records} == set(range(10))
            target = next(r for r in summary["rows"] if r["sample_size"]==n and r["method"]==records[0]["method"])
            for metric in ("expected_distance","orientation_f1"):
                assert abs(statistics.mean(r["metrics"][metric] for r in records)-target[metric+"_mean"]) < 1e-12
            confidence_rows = summary.get("calibration_rows", summary.get("confidence_rows"))
            target_confidence = next(r for r in confidence_rows if r["sample_size"]==n and r["strategy"]==strategy)
            selected = [r for r in events if (r["model"],r["sample_size"],r["strategy"])==(model,n,strategy)]
            valid_events = [r for r in selected if r["valid"]]
            accuracy = statistics.mean(r["correct"] for r in valid_events)
            brier = statistics.mean((r["confidence"]-r["correct"])**2 for r in valid_events)
            assert abs(accuracy-target_confidence["accuracy"]) < 1e-12
            assert abs(brier-target_confidence["brier"]) < 1e-12
            assert abs(len(valid_events)/200-target_confidence["coverage"]) < 1e-12
            checks.append(dict(model=model,sample_size=n,strategy=strategy,passed=True))
    assert len(groups) == 8 and len(events) == 1600
    summaries = []
    for key in sorted(groups):
        selected = [r for r in events if (r["model"],r["sample_size"],r["strategy"])==key]
        indirect = [r for r in selected if r["category"]=="indirect"]
        directed = [r for r in selected if r["kind"]=="directed"]
        adjacency_indirect = [r for r in selected if r["category"]=="indirect_either_direction"]
        relation_counts = {}
        for category in ("direct","indirect","no_forward_path","adjacent","indirect_either_direction","no_path_either_direction"):
            entries = [r for r in selected if r["category"]==category]
            counts = Counter("abstain" if not r["valid"] else "yes" if r["answer"] else "no" for r in entries)
            relation_counts[category] = dict(queries=len(entries),unique_pairs=len({(r["source"],r["target"]) for r in entries}),
                yes=counts["yes"],no=counts["no"],abstain=counts["abstain"],
                yes_rate_valid=counts["yes"]/(counts["yes"]+counts["no"]) if counts["yes"]+counts["no"] else None)
        summaries.append(dict(model=key[0],sample_size=key[1],strategy=key[2],total_queries=len(selected),
            directed_queries=len(directed),adjacency_queries=len(selected)-len(directed),
            indirect_queries=len(indirect),indirect_among_all=len(indirect)/len(selected),
            indirect_among_directed=len(indirect)/len(directed),
            indirect_unique_pairs=sorted({(r["source"],r["target"]) for r in indirect}),
            indirect_either_kind_queries=len(indirect)+len(adjacency_indirect),relations=relation_counts))
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(source_table="Main Table 2, neuropathic recovery and confidence quality",
        sources=provenance,truth_path=str(truth_path),truth_sha256=sha(truth_path),
        exclusively_indirect_pairs=sorted(indirect_pairs),verification=checks,summary=summaries)
    (destination / "audit.json").write_text(json.dumps(result,indent=2)+"\n")
    with (destination / "query_events.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    lines=["Indirect queries and directness confusion in main Table 2", "",
        "Each setting has ten seeds and twenty selected queries per seed. Indirect counts refer to directed questions with a forward path and no direct edge. Unknown responses are counted as abstentions. Consistency denotes the final aggregated answer, not its individual paraphrases.", "",
        "| Model | n | Confidence | Directed questions | Exclusively indirect questions | Fraction of directed questions | Unique indirect pairs | Incorrect yes | Correct no | Abstain |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        c=r["relations"]["indirect"]
        lines.append(f"| {r['model']} | {r['sample_size']} | {r['strategy']} | {r['directed_queries']} | {r['indirect_queries']}/200 | {r['indirect_among_directed']:.1%} | {len(r['indirect_unique_pairs'])} | {c['yes']} | {c['no']} | {c['abstain']} |")
    lines += ["", "All eight settings reproduce the saved ED, orientation F1, coverage, accuracy, and Brier summaries."]
    text="\n".join(lines)+"\n"
    (destination / "audit.md").write_text(text)
    print(text)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    audit(args.destination)
