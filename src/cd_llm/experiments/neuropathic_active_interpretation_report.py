"""Evaluation-only reporting for the matched active interpretation experiment."""
from collections import defaultdict
import fcntl
import math

import numpy as np

from cd_llm.core.finite_support import (
    Claim, distribution_metrics, evaluate_claim, has_edge, map_graph, stable_seed, transitive_closure,
)
from cd_llm.experiments.common.artifacts import read_json, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_active_interpretation import MODELS, graph, verify


def interval(values, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if len(values) < 2:
        return dict(mean=float(values.mean()),low=None,high=None)
    indices=np.random.default_rng(seed).integers(len(values),size=(10000,len(values)))
    lo,hi=np.quantile(values[indices].mean(axis=1),(0.025,0.975))
    return dict(mean=float(values.mean()),low=float(lo),high=float(hi))


def summarize_distribution(entries, truth, indirect):
    distribution={graph(e["graph"]):e["weight"] for e in entries}
    graphs=tuple(sorted(distribution))
    weights=np.asarray([distribution[g] for g in graphs])
    metrics=distribution_metrics(graphs,weights,truth,"dag")
    mode=map_graph(graphs,weights)
    closures=[transitive_closure(g) for g in graphs]
    mode_closure=transitive_closure(mode)
    paths=[]
    for a,b in indirect:
        # A predicted shortcut is reported separately from a recovered indirect path.
        mass=sum(w for g,w,c in zip(graphs,weights,closures) if (c[a] >> b) & 1)
        proper=sum(w for g,w,c in zip(graphs,weights,closures) if (c[a] >> b) & 1 and not has_edge(g,a,b))
        shortcut=sum(w for g,w in zip(graphs,weights) if has_edge(g,a,b))
        paths.append(dict(source=a,target=b,ancestry_mass=float(mass),indirect_path_mass=float(proper),
            shortcut_mass=float(shortcut),map_ancestry=int(bool((mode_closure[a] >> b)&1)),
            map_indirect_path=int(bool((mode_closure[a] >> b)&1) and not has_edge(mode,a,b)),
            map_shortcut=int(has_edge(mode,a,b))))
    metrics.update(indirect_path_mass=float(np.mean([p["indirect_path_mass"] for p in paths])),
        ancestry_mass=float(np.mean([p["ancestry_mass"] for p in paths])),
        shortcut_mass=float(np.mean([p["shortcut_mass"] for p in paths])),
        map_indirect_paths=sum(p["map_indirect_path"] for p in paths))
    return metrics,paths


def diagnose_trace(record, truth):
    closure=transitive_closure(truth)
    events=[]
    for step in record["trace"]:
        q=step["query"]
        a,b=q["source"],q["target"]
        direct=has_edge(truth,a,b)
        ancestral=bool((closure[a]>>b)&1)
        if q["kind"] == "directed":
            category="direct" if direct else "indirect" if ancestral else "no_forward_path"
        else:
            category="adjacent" if direct or has_edge(truth,b,a) else "indirect_either_direction" if ancestral or (closure[b]>>a)&1 else "no_path_either_direction"
        claim=Claim(**step["applied_claim"]) if step.get("applied_claim") else None
        events.append(dict(slot=step["budget"],**q,category=category,answer=step["answer"],
            confidence=step["reported_confidence"],applied_kind=claim.kind if claim else None,
            applied_correct=int(evaluate_claim(truth,claim)==claim.truth) if claim else None,
            direct_truth=int(direct),ancestry_truth=int(ancestral)))
    positive=[e for e in events if e["kind"]=="directed" and e["answer"]==1]
    metrics=dict(indirect_queries=sum(e["category"]=="indirect" for e in events),
        directed_queries=sum(e["kind"]=="directed" for e in events),
        direct_yes=len(positive), direct_yes_edge_correct=sum(e["direct_truth"] for e in positive),
        direct_yes_ancestry_correct=sum(e["ancestry_truth"] for e in positive),
        direct_yes_applied_correct=sum(e["applied_correct"] for e in positive),
        direct_yes_indirect=sum(e["category"]=="indirect" for e in positive),
        direct_yes_no_path=sum(e["category"]=="no_forward_path" for e in positive),
        abstentions=sum(e["answer"] is None for e in events))
    return metrics,events


def report(root):
    manifest=verify(root)
    with (root / "report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        truth=graph(read_json(root / "inputs/truth_graph.json")["graph"])
        closure=transitive_closure(truth)
        indirect=[(a,b) for a in range(20) for b in range(20) if (closure[a]>>b)&1 and not has_edge(truth,a,b)]
        if len(indirect)!=7:
            raise ValueError("frozen neuropathic indirect pairs differ")
        per_run,events,path_rows=[],[],[]
        for task in manifest["tasks"]:
            records={}
            for arm in ("literal","ancestry"):
                path=root / f"records/{arm}/{task['id']}.json"
                if path.exists():
                    records[arm]=read_json(path)
            for arm,record in records.items():
                identity=dict(model=task["model"],strategy=task["strategy"],sample_size=task["sample_size"],
                              seed=task["seed"],interpretation=arm)
                metrics,paths=summarize_distribution(record["distribution"],truth,indirect)
                if arm=="literal":
                    expected=read_json(root / task["source"])["metrics"]
                    for key in ("expected_distance","adjacency_f1","orientation_f1"):
                        if not math.isclose(metrics[key],expected[key],rel_tol=0,abs_tol=1e-11):
                            raise ValueError(f"Table 2 metric reproduction failed {task['id']} {key}")
                diagnostic,ev=diagnose_trace(record,truth)
                per_run.append(dict(**identity,**metrics,**diagnostic,paired=len(records)==2))
                events.extend(dict(**identity,**e) for e in ev)
                asked={(e["source"],e["target"]) for e in ev if e["kind"]=="directed"}
                path_rows.extend(dict(**identity,**p,asked=(p["source"],p["target"]) in asked) for p in paths)
        groups=defaultdict(list)
        for r in per_run:
            groups[r["model"],r["strategy"],r["sample_size"],r["interpretation"]].append(r)
        summaries=[]
        keys=("expected_distance","adjacency_f1","orientation_f1","indirect_path_mass","ancestry_mass","shortcut_mass","map_indirect_paths")
        counts=("indirect_queries","directed_queries","direct_yes","direct_yes_edge_correct","direct_yes_ancestry_correct",
                "direct_yes_applied_correct","direct_yes_indirect","direct_yes_no_path","abstentions")
        for (model,strategy,n,arm),rows in sorted(groups.items()):
            summary=dict(model=model,strategy=strategy,sample_size=n,interpretation=arm,seeds=len(rows))
            for key in keys:
                values=np.array([r[key] for r in rows])
                summary[key+"_mean"]=float(values.mean())
                summary[key+"_se"]=float(values.std(ddof=1)/math.sqrt(len(rows))) if len(rows)>1 else None
            for key in counts:
                summary[key]=sum(r[key] for r in rows)
            summary["direct_yes_accuracy"]=summary["direct_yes_applied_correct"]/summary["direct_yes"] if summary["direct_yes"] else None
            summaries.append(summary)
        differences=[]
        for model in MODELS:
            for strategy in ("stated","consistency"):
                for n in (200,1000):
                    literal={r["seed"]:r for r in groups.get((model,strategy,n,"literal"),[])}
                    broader={r["seed"]:r for r in groups.get((model,strategy,n,"ancestry"),[])}
                    seeds=sorted(literal.keys() & broader.keys())
                    if not seeds:
                        continue
                    for metric in keys:
                        paired=interval([broader[s][metric]-literal[s][metric] for s in seeds],
                            stable_seed(STUDY_SEED,model,strategy,n,metric))
                        differences.append(dict(model=model,strategy=strategy,sample_size=n,metric=metric,
                            paired_seeds=len(seeds),complete=len(seeds)==10,**paired))
        complete=len(per_run)==160
        accounting={}
        for model in MODELS:
            status=root / f"status/{model}.json"
            if status.exists():
                accounting[model]=read_json(status)
        result=dict(complete=complete,completed_trajectories=len(per_run),expected_trajectories=160,
            matched_conditions=sum(r["paired"] and r["interpretation"]=="ancestry" for r in per_run),
            summaries=summaries,differences=differences,accounting=accounting,indirect_pairs=indirect)
        destination=root / "report"
        write_json(destination / "summary.json",result)
        if per_run:
            write_csv(destination / "per_run.csv",per_run)
            write_csv(destination / "summary.csv",summaries)
            write_csv(destination / "query_events.csv",events)
            write_csv(destination / "path_recovery.csv",path_rows)
        if differences:
            write_csv(destination / "paired_differences.csv",differences)
        lines=["Active interpretation on the neuropathic network","",
            f"Completed {len(per_run)} of 160 trajectories and {result['matched_conditions']} of 80 paired comparisons.","",
            "Each arm adaptively selects twenty Group I questions. Only direct affirmative interpretation differs. Confidence and the original acquisition rule are unchanged. Intervals resample data seeds and are conditional on the frozen model responses. The mapping does not fit or establish confidence calibration.","",
            "| Model | Confidence | n | Interpretation | Seeds | ED | Adjacency F1 | Orientation F1 | Indirect questions | Direct yes accuracy |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in summaries:
            acc=f"{r['direct_yes_accuracy']:.1%}" if r["direct_yes_accuracy"] is not None else "undefined"
            lines.append(f"| {MODELS[r['model']]} | {r['strategy']} | {r['sample_size']} | {r['interpretation']} | {r['seeds']} | {r['expected_distance_mean']:.6f} | {r['adjacency_f1_mean']:.4f} | {r['orientation_f1_mean']:.4f} | {r['indirect_queries']}/{20*r['seeds']} | {acc} |")
        lines += ["", "Paired changes are broader minus literal. Negative ED changes and positive F1 changes favor broader interpretation.","",
            "| Model | Confidence | n | Metric | Paired seeds | Change | 95% interval |",
            "|---|---|---:|---|---:|---:|---|"]
        for r in differences:
            if r["metric"] not in ("expected_distance","orientation_f1","indirect_path_mass"):
                continue
            ci=f"[{r['low']:+.6f}, {r['high']:+.6f}]" if r["low"] is not None else "unavailable"
            lines.append(f"| {MODELS[r['model']]} | {r['strategy']} | {r['sample_size']} | {r['metric']} | {r['paired_seeds']} | {r['mean']:+.6f} | {ci} |")
        lines += ["", "Path recovery evaluates all seven exclusively indirect reference pairs. A recovered indirect path requires predicted ancestry without a direct shortcut. The detailed CSV also records ancestry mass, shortcut mass and whether each pair was queried. All seven reference pairs share one target."]
        if not complete:
            lines += ["", "This report is partial. No completed-experiment conclusion is drawn."]
        write_text(destination / "report.md","\n".join(lines)+"\n")
        print(f"Report saved with {result['matched_conditions']}/80 paired conditions",flush=True)
        return result


STUDY_SEED = "neuropathic-active-interpretation-bootstrap-v1"
