"""Evaluation for the Qwen grid initialization interpretation comparison."""
from collections import defaultdict
import fcntl
import math

import numpy as np

from cd_llm.core.finite_support import has_edge, stable_seed, transitive_closure
from cd_llm.experiments.common.artifacts import read_json, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_active_interpretation_report import diagnose_trace, interval, summarize_distribution
from cd_llm.experiments.neuropathic_grid_interpretation import ARMS, BUDGETS, STRATEGIES, verify

METRICS = ("expected_distance","adjacency_f1","orientation_f1","indirect_path_mass",
           "ancestry_mass","shortcut_mass","map_indirect_paths")
COUNTS = ("indirect_queries","directed_queries","direct_yes","direct_yes_edge_correct",
          "direct_yes_ancestry_correct","direct_yes_applied_correct","direct_yes_indirect",
          "direct_yes_no_path","abstentions")


def report(root):
    manifest=verify(root,initialized=True)
    with (root / "report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        truth=graph(read_json(root / "evaluation/truth_graph.json")["graph"])
        closure=transitive_closure(truth)
        indirect=[(a,b) for a in range(20) for b in range(20) if (closure[a]>>b)&1 and not has_edge(truth,a,b)]
        if len(indirect)!=7:
            raise ValueError("unexpected neuropathic reference graph")
        rows,events,paths=[],[],[]
        paired=0
        for task in manifest["tasks"]:
            records={arm:read_json(root / f"records/{arm}/{task['id']}.json") for arm in ARMS
                     if (root / f"records/{arm}/{task['id']}.json").exists()}
            if len(records)==2:
                paired+=1
                if records["literal"]["initialization_sha256"] != records["ancestry"]["initialization_sha256"]:
                    raise ValueError("interpretations used different initializations")
            for arm,record in records.items():
                if record["task"] != task or len(record["trace"]) != task["budget"]:
                    raise ValueError("reported trajectory has the wrong task or budget")
                identity=dict(budget=task["budget"],strategy=task["strategy"],sample_size=task["sample_size"],
                              seed=task["seed"],interpretation=arm)
                metrics,path=summarize_distribution(record["distribution"],truth,indirect)
                diagnostic,ev=diagnose_trace(record,truth)
                if record["answered_queries"] != task["budget"]-diagnostic["abstentions"]:
                    raise ValueError("usable claim accounting differs from the trace")
                rows.append(dict(**identity,**{k:metrics[k] for k in METRICS},**diagnostic,
                    answered_queries=record["answered_queries"],paired=len(records)==2))
                events.extend(dict(**identity,**e) for e in ev)
                asked={(e["source"],e["target"]) for e in ev if e["kind"]=="directed"}
                paths.extend(dict(**identity,**p,asked=(p["source"],p["target"]) in asked) for p in path)
        groups=defaultdict(list)
        for row in rows:
            groups[row["budget"],row["strategy"],row["sample_size"],row["interpretation"]].append(row)
        summaries=[]
        for (budget,strategy,n,arm),items in sorted(groups.items()):
            summary=dict(budget=budget,strategy=strategy,sample_size=n,interpretation=arm,seeds=len(items))
            for metric in METRICS:
                values=np.asarray([i[metric] for i in items])
                summary[metric+"_mean"]=float(values.mean())
                summary[metric+"_se"]=float(values.std(ddof=1)/math.sqrt(len(values))) if len(values)>1 else None
            for key in (*COUNTS,"answered_queries"):
                summary[key]=sum(i[key] for i in items)
            summary["coverage"]=summary["answered_queries"]/(budget*len(items))
            summary["direct_yes_accuracy"]=(summary["direct_yes_applied_correct"]/summary["direct_yes"]
                if summary["direct_yes"] else None)
            summaries.append(summary)
        differences=[]
        for budget in BUDGETS:
            for strategy in STRATEGIES:
                for n in (200,1000):
                    left={r["seed"]:r for r in groups.get((budget,strategy,n,"literal"),())}
                    right={r["seed"]:r for r in groups.get((budget,strategy,n,"ancestry"),())}
                    seeds=sorted(left.keys() & right.keys())
                    if not seeds:
                        continue
                    for metric in METRICS:
                        effect=interval([right[s][metric]-left[s][metric] for s in seeds],
                            stable_seed("qwen-grid-interpretation-bootstrap",budget,strategy,n,metric))
                        differences.append(dict(budget=budget,strategy=strategy,sample_size=n,metric=metric,
                            paired_seeds=len(seeds),complete=len(seeds)==10,**effect))
        accounting=read_json(root / "status/inference.json") if (root / "status/inference.json").exists() else {}
        complete=len(rows)==240 and paired==120
        summary=dict(complete=complete,trajectories=len(rows),expected_trajectories=240,
            paired_comparisons=paired,expected_comparisons=120,summaries=summaries,differences=differences,
            accounting=accounting,indirect_pairs=indirect)
        output=root / "report"
        write_json(output / "summary.json",summary)
        if rows:
            write_csv(output / "per_run.csv",rows)
            write_csv(output / "summary.csv",summaries)
            write_csv(output / "query_events.csv",events)
            write_csv(output / "path_recovery.csv",paths)
        if differences:
            write_csv(output / "paired_differences.csv",differences)
        text=["Qwen active interpretation with PC and GES grid initialization","",
            f"Completed {len(rows)}/240 trajectories and {paired}/120 matched comparisons.","",
            "Budgets count attempted questions. Each budget is run independently to preserve terminal mutation. Both interpretations use the same frozen grid population, observations and sampler seed. The original Qwen prompts, confidence strategies and EIG working rule are retained.","",
            "| Questions | Confidence | n | Interpretation | Seeds | ED | Adjacency F1 | Orientation F1 | Usable claims | Indirect questions | Positive claim accuracy |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in summaries:
            accuracy=f"{r['direct_yes_accuracy']:.1%}" if r["direct_yes_accuracy"] is not None else "undefined"
            text.append(f"| {r['budget']} | {r['strategy']} | {r['sample_size']} | {r['interpretation']} | {r['seeds']} | {r['expected_distance_mean']:.6f} | {r['adjacency_f1_mean']:.4f} | {r['orientation_f1_mean']:.4f} | {r['answered_queries']}/{r['budget']*r['seeds']} | {r['indirect_queries']} | {accuracy} |")
        text += ["","Paired changes are broader minus literal. Negative ED changes and positive F1 changes favor broader interpretation. Intervals are nominal percentile 95 percent intervals over the ten data seeds, conditional on the cached Qwen responses.","",
            "| Questions | Confidence | n | Metric | Paired seeds | Change | 95% interval |",
            "|---:|---|---:|---|---:|---:|---|"]
        for r in differences:
            if r["metric"] not in ("expected_distance","orientation_f1","indirect_path_mass"):
                continue
            ci=f"[{r['low']:+.6f}, {r['high']:+.6f}]" if r["low"] is not None else "unavailable"
            text.append(f"| {r['budget']} | {r['strategy']} | {r['sample_size']} | {r['metric']} | {r['paired_seeds']} | {r['mean']:+.6f} | {ci} |")
        text += ["","Positive claim accuracy evaluates direct affirmative responses using their applied interpretation. Path recovery distinguishes indirect paths from shortcut edges over all seven exclusively indirect reference pairs, which share one target. The experiment changes claim meaning without recalibrating confidence."]
        if not complete:
            text += ["","This is a partial report. Complete comparisons require ten paired seeds per setting."]
        write_text(output / "report.md","\n".join(text)+"\n")
        print(f"Report saved with {paired}/120 matched comparisons",flush=True)
        return summary
