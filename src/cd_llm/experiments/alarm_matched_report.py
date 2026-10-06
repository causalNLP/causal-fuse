"""Complete-case matched tables, paired inference, and measured expenditure."""
import csv
from itertools import product
from pathlib import Path
import time
import numpy as np
from cd_llm.experiments.common.artifacts import read_json,write_json
from cd_llm.experiments.alarm_matched_store import Store
from cd_llm.experiments.alarm_matched_baselines import METHODS,CONDITIONS,verify
from cd_llm.core.finite_support import PartialGraph,graph_from_edges,distribution_metrics,point_prediction_metrics

METRICS=("expected_distance","energy_score","adjacency_f1","orientation_f1","total_tokens","input_tokens","output_tokens","logical_requests")
LABELS=dict(pc="PC",ges="GES",notears="NOTEARS",gpc="gPC-Guess",bfs="BFS+Pearson",long="PC+Long-Srisk",order="Causal Order+PC",scp="SCP Pattern 2",smc_group_i="SMC local",smc_group_ii="SMC ancestry",smc_all="SMC mixed")


def csv_write(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize(rows):
    result=[]
    for method,n,budget in sorted({(r["method"],r["sample_size"],r["budget"]) for r in rows}):
        group=[r for r in rows if (r["method"],r["sample_size"],r["budget"])==(method,n,budget)]
        complete=[r for r in group if r["status"]=="complete"]
        row=dict(method=method,label=LABELS[method],sample_size=n,budget=budget,accounted=len(group),complete=len(complete),planned=10)
        # A failed seed must not turn a ten-seed comparison into a survivor mean.
        for metric in METRICS:
            values=np.array([r[metric] for r in complete]) if len(complete)==10 else None
            row[metric+"_mean"]=float(values.mean()) if values is not None else None
            row[metric+"_se"]=float(values.std(ddof=1)/np.sqrt(10)) if values is not None else None
        result.append(row)
    return result


def paired(rows):
    indexed={(r["method"],r["sample_size"],r["seed"],r["budget"]):r for r in rows}
    rng=np.random.default_rng(20260920)
    resamples=rng.integers(0,10,(10000,10))
    signs=np.array(list(product((-1,1),repeat=10)))
    output=[]
    for group in ("group_i","group_ii","all"):
        for method in METHODS:
            for n in (200,1000):
                a=[indexed.get(("smc_"+group,n,s,100)) for s in range(1000,1010)]
                b=[indexed.get((method,n,s,0)) for s in range(1000,1010)]
                row=dict(smc_pool=group,baseline=method,sample_size=n,family_size=48)
                if any(r is None or r["status"]!="complete" for r in a+b):
                    row.update(status="unavailable",reason="requires all ten paired feasible conditions")
                else:
                    delta=np.array([x["expected_distance"]-y["expected_distance"] for x,y in zip(a,b)])
                    low,high=np.quantile(delta[resamples].mean(axis=1),[.025,.975])
                    p=float(np.mean(np.abs((signs*delta).mean(axis=1))>=abs(delta.mean())-1e-12))
                    row.update(status="complete",ed_difference=float(delta.mean()),ci95_low=float(low),ci95_high=float(high),p=p)
                output.append(row)
    # Missing contrasts conservatively occupy the registered 48-test family.
    ordered=sorted(range(48),key=lambda i:output[i].get("p",1.))
    previous=0.
    for rank,i in enumerate(ordered):
        adjusted=max(previous,min(1.,(48-rank)*output[i].get("p",1.)))
        if "p" in output[i]: output[i]["p_holm"]=adjusted
        previous=adjusted
    return output


def report(root):
    manifest=verify(root)
    store=Store(root)
    truth_record=read_json(root/"evaluation/truth_graph.json")
    truth=graph_from_edges(37,directed=truth_record["graph"]["directed"])
    predictions=store.items("prediction")
    rows=[]
    for i,(seed,n) in enumerate(CONDITIONS):
        for method in METHODS:
            record=predictions.get(f"{method}:{i}",dict(status="pending"))
            row=dict(method=method,seed=seed,sample_size=n,budget=0,status=record["status"],reason=record.get("reason",""))
            usage=store.usage("gpc:shared" if method=="gpc" else "order:" if method=="order" else f"{method}:{i}:" if method=="scp" else f"{method}:{i}")
            # For BFS/Long exact unit names avoid prefix collisions (1 vs 10).
            if method in ("bfs","long"):
                with store.db() as db:
                    values=db.execute("SELECT COUNT(*),COALESCE(SUM(r.input),0),COALESCE(SUM(r.output),0) FROM uses u JOIN responses r ON u.key=r.key WHERE u.unit=?",(f"{method}:{i}",)).fetchone()
                usage=dict(logical_requests=values[0],input_tokens=values[1],output_tokens=values[2],total_tokens=values[1]+values[2])
            row.update(usage,seconds=record.get("seconds"))
            if method=="order":
                # The one diagnostic pair is charged only if actually required.
                required={tuple(p) for p in store.get("order_ties","shared") or []}
                required.update(tuple(p) for p in store.get("order_fallbacks","shared") or [])
                usages=[store.usage("order:triplet:")]+[store.usage(f"order:pair:{a}:{b}",exact=True) for a,b in required]
                row.update({k:sum(v[k] for v in usages) for k in ("logical_requests","input_tokens","output_tokens","total_tokens")})
            if record["status"]=="complete":
                if "distribution" in record:
                    graphs=[graph_from_edges(37,directed=r["edges"]) for r in record["distribution"]]
                    weights=np.array([r["weight"] for r in record["distribution"]])
                    metrics=distribution_metrics(graphs,weights,truth,"dag",normalized=False)
                else:
                    metrics=point_prediction_metrics(PartialGraph.from_matrix(np.array(record["matrix"]),"cpdag"),truth,normalized=False)
                row.update({k:metrics[k] for k in METRICS[:4]})
            rows.append(row)
    with (root/"smc/per_run.csv").open() as f:
        for r in csv.DictReader(f):
            budget=int(r["budget"])
            if budget not in (10,20,50,100): continue
            row=dict(method="smc_"+r["group"],seed=int(r["seed"]),sample_size=int(r["sample_size"]),budget=budget,status="complete")
            row.update({k:float(r[k]) for k in METRICS})
            rows.append(row)
    summary=summarize(rows)
    contrasts=paired(rows)
    csv_write(root/"per_seed.csv",rows)
    csv_write(root/"summary.csv",summary)
    csv_write(root/"paired_ed.csv",contrasts)
    main=[r for r in summary if r["budget"] in (0,100)]
    csv_write(root/"full_table.csv",main)
    with store.db() as db:
        attempts=dict(db.execute("SELECT state,COUNT(*) FROM attempts GROUP BY state"))
        actual=db.execute("SELECT COUNT(*),COALESCE(SUM(input),0),COALESCE(SUM(output),0) FROM responses").fetchone()
    runtime=store.items("gpu_runtime")
    recovery=manifest.get("recovery",{})
    accounting=dict(unique_completed_requests=actual[0],unique_input_tokens=actual[1],unique_output_tokens=actual[2],attempts=attempts,
        unresolved_attempts=attempts.get("dispatched",0),actual_api_spending_usd=0,gpu_allocations=len(runtime),
        actual_gpu_seconds=sum(r["seconds"] for r in runtime.values()),shared_expert_charge="full required collection charged per dataset",gpu_runtime=runtime)
    accounting["unknown_historical_token_usage"]=recovery.get("unknown_historical_token_usage",False)
    accounting["interrupted_historical_attempts"]=recovery.get("interrupted_attempts",[])
    with store.db() as db:
        superseded=db.execute("SELECT COUNT(*),COALESCE(SUM(input),0),COALESCE(SUM(output),0) FROM responses "
            "WHERE key IN (SELECT key FROM uses WHERE unit LIKE 'archived:scp_temperature_0p7:%')").fetchone()
    accounting["superseded_scp_generations"]=dict(unique_requests=superseded[0],input_tokens=superseded[1],output_tokens=superseded[2],
        included_in_actual_expenditure=True,included_in_corrected_scp_logical_cost=False)
    accounting["scp_temperature"]=manifest["settings"]["scp"].get("temperature",manifest["settings"]["temperature"])
    accounting["order_cycle_removal"]=store.get("audit","order_cycle_removal")
    accounting["order_cycle_guard"]=store.get("amendment","order_cycle_guard")
    write_json(root/"accounting.json",accounting)
    accounted=sum(store.terminal("prediction",f"{method}:{i}") for i in range(20) for method in METHODS)
    write_json(root/"report.json",dict(accounted=accounted,planned=160,summary=summary,paired_ed=contrasts,accounting=accounting))
    lines=["Matched ALARM baseline comparison","",f"{accounted}/160 baseline evaluations accounted for. SMC uses the completed 60 trajectories (240 checkpoints).",
        "Baselines have no confidence cap; the completed SMC runs retain their 0.8 cap. All means require ten feasible seeds.",
        f"SCP explanation generation and five probability trials use temperature {accounting['scp_temperature']}; other baselines use temperature {manifest['settings']['temperature']}. SCP retains 100 observational bootstrap fits and the existing conflicting-direction filter.","",
        "| n | Method | Feasible | ED ↓ | ES ↓ | Adj. F1 ↑ | Orient. F1 ↑ | Tokens |","|---|---|---|---|---|---|---|---|"]
    for r in main:
        def cell(k):
            v=r[k+"_mean"]
            return "unavailable" if v is None else f"{v:.2f} ± {r[k+'_se']:.2f}"
        lines.append("| "+" | ".join([str(r["sample_size"]),r["label"],f"{r['complete']}/10",*[cell(k) for k in METRICS[:5]]])+" |")
    lines += ["","ED and ES are unnormalized structural-error counts. Partial predictions retain unresolved endpoints. Long uses its uniform surviving MEC; SMC uses its saved particle distribution. Both use the common deterministic modal-graph tie rule for F1.",
        "Uncertainty across data seeds is conditional on shared expert collections. Paired differences are SMC minus baseline, with 10,000 bootstrap resamples and exact paired sign-randomization tests adjusted across 48 contrasts by Holm's procedure.",
        "Logical expenditure includes repairs and every probability trial. Superseded SCP generations are retained in actual expenditure but excluded from corrected SCP logical cost. Unique generated responses and actual GPU runtime are reported separately. Zero-token observational baselines are listed in the table and omitted from logarithmic token axes."]
    correction=accounting["order_cycle_removal"] or {}
    if correction.get("correction_applied"):
        lines += ["",f"Causal Order uses an explicitly recorded threshold-search correction: the upstream cutoff retained a cycle among {correction['original_edges']} edges; the lowest validated edge-weight cutoff retains {correction['retained_edges']} edges. Triplet votes, tie-breaks, and entropy weights are unchanged. The original failure and correction diagnostics are preserved."]
    repair=accounting["order_cycle_guard"] or {}
    if repair.get("all_order_predictions_complete"):
        lines += ["","Causal Order processes undirected PC edges in lexicographic order. It uses the estimated order only when the proposed direction remains acyclic; otherwise Qwen3.5-9B supplies the pair direction. An unusable or cyclic expert answer is replaced only by the unique acyclic direction, and every such override is recorded."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    import importlib.util
    script=Path(__file__).resolve().parents[3]/"figures/alarm_matched_performance_cost.py"
    spec=importlib.util.spec_from_file_location("alarm_matched_figure",script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.draw(root)
    from cd_llm.experiments.alarm_matched_recovery import validated_completion
    complete=validated_completion(store) and bool(store.get("gate","audit")) and len(rows[160:])==240
    write_json(root/"completion.json",dict(accounted=accounted,planned=160,complete=complete,feasible=sum(r["status"]=="complete" for r in rows[:160]),time=time.time()))
    if not complete: raise RuntimeError("Evaluation has pending or technical failures; report is incomplete")
