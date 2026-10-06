"""Matched ALARM baselines with native confidence and exhaustive acquisition."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict
import argparse
import io
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from unittest.mock import patch
import numpy as np

from cd_llm.experiments.common.artifacts import read_json,write_json,sha256
from cd_llm.experiments.alarm_matched_store import Store,Transport,Checkpoint,MODEL,digest
from cd_llm.experiments import alarm_matched_methods as methods
from cd_llm.experiments import alarm_observational as observational
from cd_llm.core.finite_support import PartialGraph,graph_edges
from cd_llm.paths import WORKSPACE_ROOT

ROOT=WORKSPACE_ROOT
SOURCE=ROOT/"results/20260920T164354Z/alarm_claim_pools_local"
GRID=ROOT/"results/20260920T150549Z/alarm_claim_pools"
METHODS=("pc","ges","notears","gpc","bfs","long","order","scp")
CONDITIONS=[(s,n) for s in range(1000,1010) for n in (200,1000)]


class MissingOrderExpertPair(RuntimeError):
    def __init__(self, first, second):
        self.pair = tuple(sorted((first, second)))
        super().__init__(f"missing cached order expert response for pair {self.pair}")


def cid(index):
    seed,n=CONDITIONS[index]
    return f"alarm_s{seed}_n{n}"


def metadata(root):
    m=read_json(root/"inputs/model_metadata.json")
    return [r["id"] for r in m["nodes"]],{r["id"]:r["description"] for r in m["nodes"]},m["domain_context"]


def verify(root):
    m=read_json(root/"manifest.json")
    for name,value in m["inputs"].items():
        if sha256(root/name)!=value: raise ValueError(f"frozen input changed: {name}")
    for name,value in m["implementation"].items():
        if sha256(ROOT/name)!=value: raise ValueError(f"frozen code changed: {name}")
    return m


def prepare(root):
    if (root/"manifest.json").exists(): return verify(root)
    from cd_llm.experiments.alarm_claim_pools_local import verify as verify_source
    source=verify_source(SOURCE)
    completion=read_json(SOURCE/"completion.json")
    if not completion["complete"] or completion["trajectories"]!=60 or completion["checkpoints"]!=240:
        raise ValueError("source SMC experiment incomplete")
    if completion["manifest_sha256"]!=sha256(SOURCE/"manifest.json"): raise ValueError("source manifest mismatch")
    for name,value in completion["artifacts"].items():
        if sha256(SOURCE/name)!=value: raise ValueError(f"source result changed: {name}")
    root.mkdir(parents=True,exist_ok=True)
    for folder in ("inputs","evaluation","smc","status"):
        (root/folder).mkdir(exist_ok=True)
    files=[c["data_path"] for c in source["conditions"]]+["inputs/model_metadata.json","inputs/encoding.json","inputs/presentation.json","inputs/runtime.json","evaluation/truth_graph.json"]
    for name in files: shutil.copyfile(SOURCE/name,root/name)
    for name in ("manifest.json","completion.json","per_run.csv","summary.csv"):
        shutil.copyfile(SOURCE/name,root/"smc"/name)
    fits={}
    grid_manifest=read_json(GRID/"manifest.json")
    for i,(seed,n) in enumerate(CONDITIONS):
        data=np.load(root/f"inputs/{cid(i)}.npy",allow_pickle=False)
        if data.shape!=(n,37): raise ValueError("invalid observation shape")
        if n==200 and not np.array_equal(data,np.load(root/f"inputs/alarm_s{seed}_n1000.npy")[:200]):
            raise ValueError("observations are not paired/nested")
        if sha256(GRID/f"inputs/{cid(i)}.npy")!=sha256(root/f"inputs/{cid(i)}.npy"):
            raise ValueError("statistical fit data mismatch")
        fits[cid(i)]={}
        for method,config in (("pc","pc_a0.05"),("ges","ges_e1_p1")):
            path=GRID/f"fits/{cid(i)}_{config}.json"
            fit=read_json(path)
            if fit["manifest_sha256"]!=sha256(GRID/"manifest.json"): raise ValueError("fit manifest mismatch")
            expected=next(x for x in grid_manifest["configuration"]["grid"] if x["id"]==config)
            if fit["config"]!=expected: raise ValueError("fit configuration mismatch")
            locked={"pc":dict(alpha=.05,family="pc",id="pc_a0.05"),
                "ges":dict(equivalent_sample_size=1.,structure_prior=1.,family="ges",id="ges_e1_p1")}[method]
            if fit["config"]!=locked: raise ValueError(f"unexpected fit configuration: {fit['config']}")
            fits[cid(i)][method]=dict(partial=fit["partial"],config=fit["config"],source=str(path),sha256=sha256(path))
    write_json(root/"inputs/statistics.json",fits)
    names,descriptions,context=metadata(root)
    prompts=methods.order_prompts(names,descriptions,context)
    prompts["gpc"]=methods.gpc_request(names,descriptions,context)
    # Fixed by variable indices alone, including every node, no truth access.
    chosen=[]
    for a in range(0,37,2):
        nodes=sorted({a,(a+1)%37,(a+2)%37})
        chosen.append(next(i for i,r in enumerate(prompts["triplets"]) if r["nodes"]==[names[j] for j in nodes]))
    chosen.append(next(i for i in range(7770) if i not in chosen))
    prompts["pilot_triplet_indices"]=chosen
    write_json(root/"inputs/prompts.json",prompts)
    sources=set(Path(__file__).parent.glob("alarm_matched_*.py"))
    sources.update(Path(__file__).parent/p for p in ("alarm_local_support.py","alarm_llm_recovery.py","alarm_llm_repair.py","alarm_observational.py","alarm_bfs_openrouter.py","causal_order2025_table42.py"))
    for folder in ("code/src/cd_llm/baselines","code/src/cd_llm/core","code/src/cd_llm/methods","code/vendor/Causal_Order_Imperfect_Experts/causal_discovery"):
        sources.update((ROOT/folder).rglob("*.py"))
    sources.update((ROOT/"code/figures").glob("alarm_matched_*.py"))
    sources.update((ROOT/"sbatch").glob("alarm_matched_*.sbatch"))
    sources.add(ROOT/"code/tests/unit/experiments/test_alarm_matched_baselines.py")
    if len(list((SOURCE/"records").glob("*.json")))!=60: raise ValueError("expected sixty frozen SMC trajectories")
    write_json(root/"manifest.json",dict(source=str(SOURCE),source_manifest_sha256=sha256(SOURCE/"manifest.json"),
        smc_trajectories={str(p.relative_to(SOURCE)):sha256(p) for p in (SOURCE/"records").glob("*.json")},
        model=MODEL,runtime=source["runtime"],runtime_fingerprint=source["runtime_fingerprint"],
        methods=METHODS,conditions=CONDITIONS,baseline_confidence_cap=None,smc_confidence_cap=.8,
        settings=dict(observational=observational.SETTINGS,gpc=dict(alpha=.05,trial_seed=0,max_depth=None),
            long=dict(strategy="srisk",tolerance=.1,max_extensions=50000),order=dict(triplets=7770,max_tokens=600,parse_gate=.95,cycle_removal_policy="validated_threshold"),
            scp=dict(bootstraps=100,probability_trials=5,forbid=.05,require=.95,temperature=.7),temperature=0,seed=0,thinking=False),
        input_column_order="already presented; no additional permutation",baseline_evaluations=160,
        inputs={str(p.relative_to(root)):sha256(p) for folder in ("inputs","evaluation","smc") for p in (root/folder).glob("*")},
        implementation={str(p.resolve().relative_to(ROOT.resolve())):sha256(p) for p in sources},actual_api_spending_usd=0))
    Store(root)
    write_json(root/"status.json",dict(stage="prepared",baseline_evaluations=160,scp_pairs=26640,order_triplets=7770))
    return verify(root)


def data(root,i): return np.load(root/f"inputs/{cid(i)}.npy",allow_pickle=False)


def pc_partial(root,i):
    p=read_json(root/"inputs/statistics.json")[cid(i)]["pc"]["partial"]
    return PartialGraph(node_count=p["node_count"],marks=tuple(p["marks"]),kind=p["kind"])


def bootstrap(root,i,deadline):
    store=Store(root)
    names,_,_=metadata(root)
    values=data(root,i)
    for method in ("pc","ges","notears"):
        key=f"{method}:{i}"
        if store.get("prediction",key): continue
        started=time.monotonic()
        if method in ("pc","ges"):
            partial=read_json(root/"inputs/statistics.json")[cid(i)][method]["partial"]
            matrix=np.array(partial["marks"]).reshape(37,37)
            diagnostics=dict(reused_verified_fit=True)
            if i in (0,1):
                refit,diagnostics=observational.fit(method,values,names)
                if not np.array_equal(refit,matrix): raise ValueError(f"representative {method} refit differs")
        else:
            matrix,diagnostics=observational.fit(method,values,names)
        store.put("prediction",key,dict(status="complete",matrix=matrix.tolist(),diagnostics=diagnostics,seconds=time.monotonic()-started))
    existing=store.get("bootstrap",str(i)) or dict(matrices=[],indices_hashes=[])
    rng=np.random.default_rng(int(digest(["alarm-matched-scp-bootstrap",*CONDITIONS[i]])[:8],16))
    for b in range(100):
        indices=rng.integers(0,len(values),len(values))
        h=digest(indices.tolist())
        if b<len(existing["matrices"]):
            if existing["indices_hashes"][b]!=h: raise ValueError("bootstrap resume changed")
            continue
        if time.time()>=deadline: raise Checkpoint("bootstrap deadline")
        matrix,_=observational.fit("pc",values[indices],names)
        existing["matrices"].append(matrix.tolist())
        existing["indices_hashes"].append(h)
        store.put("bootstrap",str(i),existing)
    store.put("bootstrap_ready",str(i),dict(sha256=digest(existing)))


def transport(root,store,unit,online,deadline):
    return Transport(store,unit,read_json(root/"manifest.json")["runtime_fingerprint"],online=online,deadline=deadline)


def pair_job(root,store,i,a,b,online,deadline,stats):
    key=f"{i}:{a}:{b}"
    if online and store.get("scp",key): return
    names,descriptions,context=metadata(root)
    t=transport(root,store,"scp:"+key,online,deadline)
    prompt=methods.scp_prompt(names,descriptions,context,CONDITIONS[i][1],stats,a,b)
    settings=read_json(root/"manifest.json")["settings"]
    row=methods.scp_pair(t,prompt,names[a],names[b],temperature=settings["scp"].get("temperature",settings["temperature"]))
    if online: store.put("scp",key,row)
    elif row!=store.get("scp",key): raise ValueError("SCP offline replay differs")


def order_query(root,store,key,messages,nodes,online,deadline):
    if online and store.get("order_response",key) is not None: return store.get("order_response",key)
    t=transport(root,store,"order:"+key,online,deadline)
    response,request=t.query(dict(model=MODEL,messages=messages,max_tokens=600,temperature=0,seed=0))
    result=dict(parsed=methods.parse_order(response,nodes),request=request)
    if online: store.put("order_response",key,result)
    return result


def order_expert(root,store,online,deadline,*,prepare_only=False):
    from cd_llm.baselines.causal_order2025.upstream_backend import query_triplet_order
    from cd_llm.baselines.causal_order2025.types import TripletConfig
    from cd_llm.baselines.causal_order2025.local_vllm_backend import _pair_prompt
    from cd_llm.experiments.causal_order2025_table42 import required_fallback_pairs
    from cd_llm.experiments.neuropathic_causal_order_subnetwork import _expert_record
    names,descriptions,context=metadata(root)
    prompts=read_json(root/"inputs/prompts.json")
    rows=[store.get("order_response",f"triplet:{i}") for i in range(7770)]
    if any(r is None for r in rows): raise ValueError("incomplete triplet collection")
    valid=sum(r["parsed"] is not None for r in rows)
    store.put("audit","order_parse",dict(parsed=valid,total=7770,passed=valid*100>=95*7770))
    if valid*100<95*7770: raise ValueError("95% triplet parsing gate failed")
    mod=methods.order_modules()
    ties=[]
    def tie(a,b,*args,**kwargs): ties.append((a,b)); return "No_Conn"
    with patch.object(mod.triplet,"_tiebreaker_query",tie),redirect_stdout(io.StringIO()):
        mod.triplet.merge_triplet_votes([r["nodes"] for r in prompts["triplets"]],
            [[tuple(e) for e in r["parsed"]["entries"]] if r["parsed"] else [] for r in rows],names,context,expert_model=MODEL,delay=0)
    for a,b in ties:
        ai,bi=names.index(a),names.index(b)
        if ai>bi: raise ValueError("unexpected upstream pair ordering")
        row=order_query(root,store,f"pair:{ai}:{bi}",prompts["pairs"][f"{ai}:{bi}"],None,online,deadline)
        if row["parsed"] is None: raise ValueError("invalid required vote tie-break")
    store.put("order_ties","shared",[[names.index(a),names.index(b)] for a,b in ties])
    indexed={digest(r["messages"]):rows[i] for i,r in enumerate(prompts["triplets"])}
    indexed.update({digest(messages):store.get("order_response","pair:"+key) for key,messages in prompts["pairs"].items()})
    original_pair=mod.triplet.cot_tiebreaker_prompt
    def pair_prompt(a,b,all_names,_context): return _pair_prompt(original_pair,a,b,all_names,context,descriptions)
    def answer(messages,model,max_completion_tokens):
        messages=methods.order_messages(messages,pair=max_completion_tokens!=300)
        row=indexed.get(digest(messages))
        if row is None: raise ValueError("uncached upstream request")
        result=row["parsed"]
        return "<Answer>"+((result["tag"] if result else "[]") if max_completion_tokens==300 else result)+"</Answer>"
    diagnostic={}
    policy=read_json(root/"manifest.json")["settings"]["order"].get("cycle_removal_policy","upstream")
    with patch.object(mod.triplet,"query_llm",answer),patch.object(mod.triplet,"cot_tiebreaker_prompt",pair_prompt),patch.object(
            mod.triplet,"parse_answer_tag",lambda text:text.split("<Answer>")[-1].split("</Answer>")[0]),redirect_stdout(io.StringIO()):
        expert=query_triplet_order(names,context,descriptions=descriptions,config=TripletConfig(subgraph_model=MODEL,expert_model=MODEL,delay=0),
            cycle_removal_policy=policy,cycle_removal_diagnostics=diagnostic)
    store.put("audit","order_cycle_removal",diagnostic)
    encoded=json.loads(json.dumps(_expert_record(expert)))
    if online or prepare_only:
        store.put("order_expert","shared",encoded)
    else:
        previous=store.get("order_expert","shared")
        if {k:v for k,v in previous.items() if k!="runtime_seconds"}!={k:v for k,v in encoded.items() if k!="runtime_seconds"}:
            raise ValueError("offline shared-order replay differs")
    fallbacks=required_fallback_pairs(expert,[pc_partial(root,i) for i in range(20)])
    if prepare_only:
        store.put("order_fallbacks_required","shared",[list(p) for p in fallbacks])
        return expert
    for a,b in fallbacks:
        order_query(root,store,f"pair:{a}:{b}",prompts["pairs"][f"{a}:{b}"],None,online,deadline)
    store.put("order_fallbacks","shared",[list(p) for p in fallbacks])
    return expert


def cached_order_fallback(store, first, second):
    left, right = sorted((first, second))
    row = store.get("order_response", f"pair:{left}:{right}")
    if row is None:
        raise MissingOrderExpertPair(left, right)
    answer = row.get("parsed")
    if answer == "A":
        return left, right
    if answer == "B":
        return right, left
    return None


def collect_order_cycle_fallbacks(root, deadline):
    """Collect only edgewise cycle-guard expert calls, then restore order fits."""
    from cd_llm.baselines.causal_order2025 import integrate_order_into_cpdag
    from cd_llm.experiments.neuropathic_decontamination_qwen import _expert_from_record
    from cd_llm.experiments.alarm_matched_recovery import finalize_order_cycle_guard

    store = Store(root)
    expert = _expert_from_record(store.get("order_expert", "shared"))
    prompts = read_json(root / "inputs/prompts.json")
    started = time.time()
    collected = []
    while True:
        missing = set()
        used = set()
        for index in range(20):
            try:
                integrated = integrate_order_into_cpdag(
                    pc_partial(root, index), expert,
                    fallback_orienter=lambda a, b: cached_order_fallback(store, a, b),
                )
                used.update(integrated.expert_fallback_pairs)
            except MissingOrderExpertPair as error:
                missing.add(error.pair)
        if not missing:
            break
        if time.time() >= deadline:
            raise Checkpoint("cycle-guard expert collection deadline")

        def query(pair):
            a, b = pair
            order_query(root, store, f"pair:{a}:{b}", prompts["pairs"][f"{a}:{b}"], None, True, deadline)
            return pair

        with ThreadPoolExecutor(min(4, len(missing))) as pool:
            collected.extend(pool.map(query, sorted(missing)))
    pending = read_json(root / "amendments/order_cycle_guard/pending.json")
    before_units = {(row[0], row[1]) for row in pending["causal_order_ledger_before"]["rows"]}
    newly_collected = [pair for pair in sorted(used)
                       if (f"order:pair:{pair[0]}:{pair[1]}", 0) not in before_units]
    store.put("order_fallbacks", "shared", [list(pair) for pair in sorted(used)])
    store.put("audit", "order_cycle_guard", dict(
        passed=True, expert_model=MODEL, edge_order="lexicographic",
        fallback_pairs=[list(pair) for pair in sorted(used)],
        newly_collected_pairs=[list(pair) for pair in newly_collected],
        collected_this_job=[list(pair) for pair in collected],
        seconds=time.time() - started,
    ))
    for index in (3, 10, 16):
        fit(root, index, float("inf"), only="order")
    finalize_order_cycle_guard(root)


def pilot(root,deadline):
    store=Store(root)
    if store.get("gate","pilot"): return
    write_json(root/"status.json",dict(stage="pilot",updated=time.time(),confidence_cap=None,model=MODEL))
    names,descriptions,context=metadata(root)
    prompts=read_json(root/"inputs/prompts.json")
    guess=methods.collect_gpc(transport(root,store,"gpc:shared",True,deadline),prompts["gpc"],names)
    store.put("gpc_guess","shared",guess)
    from cd_llm.baselines.long2023.algorithm import ambiguous_pairs
    a,b=ambiguous_pairs(pc_partial(root,0))[0]
    methods.long_expert(transport(root,store,"pilot:long",True,deadline),context)(a,b,f"{names[a]} ({descriptions[names[a]]})",f"{names[b]} ({descriptions[names[b]]})")
    # The full first BFS traversal doubles as the protocol pilot, then replays offline.
    record=methods.bfs_fit(transport(root,store,"bfs:0",True,deadline),data(root,0),names,descriptions,context)
    store.put("collected","bfs:0",dict(status="complete",**record))
    stats=methods.bootstrap_probabilities(store.get("bootstrap","0")["matrices"],names)
    pair_job(root,store,0,0,1,True,deadline,stats)
    def one(i): return order_query(root,store,f"triplet:{i}",prompts["triplets"][i]["messages"],prompts["triplets"][i]["nodes"],True,deadline)
    with ThreadPoolExecutor(4) as pool: rows=list(pool.map(one,prompts["pilot_triplet_indices"]))
    if sum(r["parsed"] is not None for r in rows)<19: raise ValueError("pilot triplet parsing below 95%")
    pair=order_query(root,store,"pair:0:1",prompts["pairs"]["0:1"],None,True,deadline)
    if pair["parsed"] is None: raise ValueError("pilot pair failed")
    store.put("gate","pilot",dict(passed=True,triplets=20,parsed=sum(r["parsed"] is not None for r in rows)))


def progress(root,store,started,initial):
    with store.db() as db:
        total=db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        attempts=dict(db.execute("SELECT state,COUNT(*) FROM attempts GROUP BY state"))
        scp=db.execute("SELECT COUNT(*) FROM items WHERE kind='scp'").fetchone()[0]
        triplets=db.execute("SELECT COUNT(*) FROM items WHERE kind='order_response' AND key LIKE 'triplet:%'").fetchone()[0]
        tokens=db.execute("SELECT COALESCE(SUM(input),0),COALESCE(SUM(output),0) FROM responses").fetchone()
    elapsed=time.time()-started
    rate=(total-initial)/elapsed if elapsed>0 else 0
    remaining=max(0,26640-scp)*6+max(0,7770-triplets)
    write_json(root/"status.json",dict(stage="collection",updated=time.time(),unique_responses=total,attempts=attempts,
        scp_pairs=scp,scp_pairs_total=26640,triplets=triplets,triplets_total=7770,input_tokens=tokens[0],output_tokens=tokens[1],
        requests_per_second=rate,nominal_remaining_seconds=remaining/rate if rate>0 else None,actual_api_spending_usd=0,
        scp_temperature=read_json(root/"manifest.json")["settings"]["scp"].get("temperature",0)))


def collect(root,deadline,chain=False,method=None,condition=None,shard=0,shards=1):
    store=Store(root)
    pilot(root,deadline)
    if chain:
        missing=[i for i in range(20) if not store.terminal("prediction",f"gpc:{i}")]
        if missing: submit(root,"gpc",array=",".join(map(str,missing))+"%4")
    names,descriptions,context=metadata(root)
    prompts=read_json(root/"inputs/prompts.json")
    started=time.time()
    with store.db() as db: initial=db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    tasks=[]
    if method in (None,"order"):
        tasks += [("order",i) for i in range(7770) if i%shards==shard and store.get("order_response",f"triplet:{i}") is None]
    def run(task):
        if time.time()>=deadline: raise Checkpoint("collection deadline")
        kind,i=task
        return order_query(root,store,f"triplet:{i}",prompts["triplets"][i]["messages"],prompts["triplets"][i]["nodes"],True,deadline)
    with ThreadPoolExecutor(4) as pool:
        # Bounded map batches allow deadline handling without 160k queued futures.
        for first in range(0,len(tasks),32):
            list(pool.map(run,tasks[first:first+32]))
            progress(root,store,started,initial)
    if method in (None,"order") and shards==1:
        order_expert(root,store,True,deadline)
    indices=range(20) if condition is None else [condition]
    for i in indices:
        for name,fit in (("bfs",methods.bfs_fit),("long",methods.long_fit)):
            if method not in (None,name) or i%shards!=shard or store.terminal("collected",f"{name}:{i}"): continue
            t=transport(root,store,f"{name}:{i}",True,deadline)
            try:
                record=fit(t,data(root,i) if name=="bfs" else pc_partial(root,i),names,descriptions,context)
                store.put("collected",f"{name}:{i}",dict(status="complete",**record))
            except Checkpoint: raise
            except Exception as error:
                from cd_llm.experiments.alarm_matched_recovery import scientific_failure
                if not scientific_failure(name,error): raise
                store.put("collected",f"{name}:{i}",dict(status="infeasible",scientific_infeasibility=True,reason=repr(error)))
        if method not in (None,"scp"): continue
        stats=methods.bootstrap_probabilities(store.get("bootstrap",str(i))["matrices"],names)
        pairs=[(a,b) for a in range(37) for b in range(37) if a!=b and (i*1369+a*37+b)%shards==shard and store.get("scp",f"{i}:{a}:{b}") is None]
        with ThreadPoolExecutor(4) as pool:
            for first in range(0,len(pairs),16):
                list(pool.map(lambda p:pair_job(root,store,i,*p,True,deadline,stats),pairs[first:first+16]))
                progress(root,store,started,initial)
    if method is None and shards==1:
        audit(root)
        if chain: submit(root,"fit",array="0-19%4",dependency="afterok:"+os.environ["SLURM_JOB_ID"])


def audit(root):
    m=verify(root)
    store=Store(root)
    from cd_llm.experiments.alarm_matched_store import validate_response,validate_request_temperature
    with store.db() as db:
        for key,req,response in db.execute("SELECT key,request,response FROM responses"):
            req=json.loads(req)
            if key!=digest(dict(request=req,runtime=m["runtime_fingerprint"])): raise ValueError("cache identity mismatch")
            allowed={m["settings"]["temperature"],m["settings"]["scp"].get("temperature",m["settings"]["temperature"])}
            if req["temperature"] not in allowed or req["chat_template_kwargs"]!={"enable_thinking":False}: raise ValueError("generation mismatch")
            validate_response(json.loads(response))
        for unit,req in db.execute("SELECT u.unit,r.request FROM uses u JOIN responses r ON r.key=u.key WHERE u.unit NOT LIKE 'archived:%'"):
            validate_request_temperature(json.loads(req),unit,m["settings"])
        scp=db.execute("SELECT COUNT(*) FROM items WHERE kind='scp'").fetchone()[0]
        triplets=db.execute("SELECT COUNT(*) FROM items WHERE kind='order_response' AND key LIKE 'triplet:%'").fetchone()[0]
    if scp!=26640 or triplets!=7770 or len(store.items("collected"))!=40 or not store.get("order_expert","shared"):
        raise ValueError("baseline collection incomplete")
    for i in range(20):
        if digest(store.get("bootstrap",str(i)))!=store.get("bootstrap_ready",str(i))["sha256"]:
            raise ValueError("bootstrap bundle changed after completion")
    from cd_llm.experiments.alarm_claim_pools import offline_guard
    with offline_guard():
        order_expert(root,store,False,float("inf"))
        # Reconstruct native explanation + five probability trials from raw cache.
        names,_,_=metadata(root)
        for i in (0,1):
            stats=methods.bootstrap_probabilities(store.get("bootstrap",str(i))["matrices"],names)
            pair_job(root,store,i,0,1,False,float("inf"),stats)
    record=dict(passed=True,scp_pairs=scp,triplets=triplets,scp_temperature=m["settings"]["scp"].get("temperature",m["settings"]["temperature"]),actual_api_spending_usd=0)
    store.put("gate","audit",record)
    write_json(root/"cache_audit.json",record)


def fit(root,i,deadline,only=None):
    from cd_llm.experiments.alarm_claim_pools import offline_guard
    store=Store(root)
    if only!="gpc" and not store.get("gate","audit"):
        gate = store.get("method_audit",f"{only}:{i}") if only else None
        if not gate or not gate.get("passed"):
            raise ValueError("offline fitting requires collection or method audit")
    names,descriptions,context=metadata(root)
    with offline_guard():
        for name in ([only] if only else ["bfs","long","order","scp"]):
            key=f"{name}:{i}"
            if store.terminal("prediction",key): continue
            started=time.monotonic()
            try:
                if name=="gpc":
                    from cd_llm.experiments.alarm_matched_gpc import search
                    guess=store.get("gpc_guess","shared")
                    matrix=search(data(root,i),np.array(guess["skeleton"]),np.random.default_rng(int(digest(["alarm-gpc",CONDITIONS[i][0]])[:8],16)),store,str(i),deadline)
                    row=dict(matrix=matrix.tolist())
                elif name in ("bfs","long"):
                    collected=store.get("collected",key)
                    if collected["status"]!="complete":
                        store.put("prediction",key,collected); continue
                    t=transport(root,store,key,False,deadline)
                    row=(methods.bfs_fit if name=="bfs" else methods.long_fit)(t,data(root,i) if name=="bfs" else pc_partial(root,i),names,descriptions,context)
                    if digest(dict(status="complete",**row))!=digest(collected): raise ValueError("offline replay differs")
                elif name=="scp":
                    rows=[store.get("scp",f"{i}:{a}:{b}") for a in range(37) for b in range(37) if a!=b]
                    row=methods.scp_fit(data(root,i),names,rows)
                elif name=="order":
                    from cd_llm.baselines.causal_order2025 import integrate_order_into_cpdag
                    from cd_llm.experiments.neuropathic_decontamination_qwen import _expert_from_record
                    expert=_expert_from_record(store.get("order_expert","shared"))
                    result=integrate_order_into_cpdag(
                        pc_partial(root,i), expert,
                        fallback_orienter=lambda a,b: cached_order_fallback(store,a,b))
                    row=dict(matrix=observational.edge_matrix(37,graph_edges(result.graph)["directed"]).tolist())
                    if result.expert_fallback_pairs:
                        answers={f"{a}:{b}":store.get("order_response",f"pair:{a}:{b}").get("parsed")
                                 for a,b in result.expert_fallback_pairs}
                        row["integration"]=dict(
                            expert_model=MODEL,
                            expert_fallback_pairs=[list(pair) for pair in result.expert_fallback_pairs],
                            cycle_guard_pairs=[list(pair) for pair in result.cycle_guard_pairs],
                            fallback_orientations=[list(edge) for edge in result.fallback_orientations],
                            forced_safe_orientations=[list(edge) for edge in result.forced_safe_orientations],
                            expert_answers=answers,
                            edge_order="lexicographic")
                else: raise ValueError(name)
                store.put("prediction",key,dict(status="complete",seconds=time.monotonic()-started,**row))
            except Checkpoint: raise
            except Exception as error:
                from cd_llm.experiments.alarm_matched_recovery import scientific_failure
                scientific=scientific_failure(name,error)
                store.put("prediction",key,dict(status="infeasible" if scientific else "failed",scientific_infeasibility=scientific,reason=repr(error),traceback=traceback.format_exc()))
                if not scientific: raise


def submit(root,stage,array=None,dependency=None,condition=None):
    from cd_llm.experiments.alarm_matched_recovery import submit_unique
    return submit_unique(root,stage,array,dependency,condition,
        lambda selected,token:_submit(root,stage,selected,dependency,condition,token))


def _submit(root,stage,array=None,dependency=None,condition=None,token=None):
    log=ROOT/"out"/root.parent.name/"alarm_matched_baselines"
    log.mkdir(parents=True,exist_ok=True)
    cmd=["sbatch","--parsable",f"--output={log}/{stage}_%A_%a.out",f"--error={log}/{stage}_%A_%a.err"]
    if array: cmd += ["--array="+array]
    if dependency: cmd += ["--dependency="+dependency]
    if token: cmd += ["--comment="+token]
    if stage=="gpc": cmd += ["--time=04:00:00"]
    if stage in ("prepare","recover","resume","report","amend-scp-temperature","amend-order-threshold","amend-order-cycle-guard"): cmd += ["--time=00:20:00"]
    if stage=="collect-order-fallbacks": cmd += ["--time=00:30:00"]
    env=dict(os.environ,ALARM_MATCHED_ROOT=str(root),ALARM_MATCHED_STAGE=stage)
    if condition is not None: env["ALARM_MATCHED_CONDITION"]=str(condition)
    else: env.pop("ALARM_MATCHED_CONDITION",None)
    for key in ("TMPDIR","SLURM_TMPDIR"): env.pop(key,None)
    cmd += [str(ROOT/"sbatch"/("alarm_matched_gpu.sbatch" if stage in ("collect","collect-order-fallbacks") else "alarm_matched_cpu.sbatch"))]
    job=subprocess.run(cmd,check=True,capture_output=True,text=True,env=env).stdout.strip().split(";")[0]
    print(f"Submitted {stage}: {job}",flush=True)
    if (root/"cache.sqlite3").exists(): Store(root).put("job",job,dict(stage=stage,array=array,condition=condition,submitted=time.time(),token=token))
    return job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","recover","resume","bootstrap","pilot","collect","collect-order-fallbacks","audit","fit","gpc","report","submit","amend-scp-temperature","amend-order-threshold","amend-order-cycle-guard"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--temperature",type=float,choices=(0.7,))
    p.add_argument("--condition",type=int,choices=range(20))
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    p.add_argument("--chain",action="store_true")
    args=p.parse_args()
    root=args.root
    deadline=float(os.environ.get("ALARM_MATCHED_DEADLINE","inf"))
    try:
        if args.stage=="amend-order-cycle-guard":
            from cd_llm.experiments.alarm_matched_recovery import amend_order_cycle_guard
            amend_order_cycle_guard(root)
            if args.chain:
                submit(root,"collect-order-fallbacks",dependency="afterok:"+os.environ["SLURM_JOB_ID"])
            return
        if args.stage=="amend-order-threshold":
            from cd_llm.experiments.alarm_matched_recovery import amend_order_threshold,resume
            amend_order_threshold(root)
            if args.chain: resume(root)
            return
        if args.stage=="amend-scp-temperature":
            if args.temperature is None: p.error("amend-scp-temperature requires --temperature 0.7")
            from cd_llm.experiments.alarm_matched_recovery import amend_scp_temperature,resume
            amend_scp_temperature(root,args.temperature)
            if args.chain: resume(root)
            return
        if args.stage=="submit": submit(root,"prepare"); return
        if args.stage in ("recover","resume"):
            from cd_llm.experiments.alarm_matched_recovery import recover,resume
            if args.stage=="recover":
                if args.source_root is None: p.error("recover requires --source-root")
                recover(args.source_root,root)
                if args.chain: resume(root)
            else: resume(root)
            return
        if args.stage=="prepare":
            prepare(root)
            if args.chain:
                job=submit(root,"bootstrap",array="0-19%4",dependency="afterok:"+os.environ["SLURM_JOB_ID"])
                submit(root,"collect",dependency="afterok:"+job)
            return
        verify(root)
        if args.stage=="bootstrap": bootstrap(root,args.condition,deadline)
        elif args.stage=="pilot": pilot(root,deadline)
        elif args.stage=="collect": collect(root,deadline,args.chain,args.method,args.condition,args.shard,args.shards)
        elif args.stage=="collect-order-fallbacks": collect_order_cycle_fallbacks(root,deadline)
        elif args.stage=="audit": audit(root)
        elif args.stage in ("fit","gpc"):
            fit(root,args.condition,deadline,"gpc" if args.stage=="gpc" else args.method)
            if args.chain:
                store=Store(root)
                from cd_llm.experiments.alarm_matched_recovery import validated_completion
                if validated_completion(store): submit(root,"report",dependency="afterok:"+os.environ["SLURM_JOB_ID"])
        elif args.stage=="report":
            from cd_llm.experiments.alarm_matched_report import report
            report(root)
    except Checkpoint as error:
        write_json(root/"status"/f"checkpoint_{args.stage}_{args.condition}.json",dict(reason=str(error),time=time.time()))
        if args.chain and args.stage in ("collect","gpc","bootstrap","fit"):
            submit(root,args.stage,condition=args.condition,dependency="afterok:"+os.environ["SLURM_JOB_ID"])
        else: raise
    except Exception as error:
        write_json(root/"status"/f"failure_{args.stage}_{args.condition}.json",dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        import sqlite3
        from urllib.error import URLError
        if args.chain and args.stage in ("collect","gpc","bootstrap","fit") and isinstance(error,(OSError,sqlite3.OperationalError,URLError)):
            store=Store(root)
            key=f"{args.stage}:{args.condition}"
            history=store.get("infrastructure_retries",key) or []
            history.append(dict(time=time.time(),error=repr(error)))
            store.put("infrastructure_retries",key,history)
            if len(history)<=3:
                submit(root,args.stage,condition=args.condition,dependency="afterany:"+os.environ["SLURM_JOB_ID"])
        raise


if __name__=="__main__": main()
