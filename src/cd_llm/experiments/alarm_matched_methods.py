"""Native baseline adapters; confidence is never capped by the SMC setting."""
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
import math
import random
import re
import numpy as np
from cd_llm.experiments.alarm_matched_store import MODEL, client
from cd_llm.core.finite_support import PartialGraph, graph_edges

CONCISE = "Give your reasoning in at most two short sentences, then immediately output the graph inside <Answer></Answer> on a single line. Do not repeat the question, descriptions, or examples."
PAIR_CONCISE = "Give your reasoning in at most two short sentences, then immediately output your answer inside <Answer></Answer> on a single line using the requested A/B/C format. Do not repeat the question, descriptions, or examples."


def gpc_request(names, descriptions, context):
    from cd_llm.baselines.guess2graph2026.llm_expert import build_skeleton_prompt
    shuffled = list(names)
    random.Random(0).shuffle(shuffled)
    return dict(model=MODEL,temperature=0,seed=0,max_tokens=2000,response_format={"type":"json_object"},
        messages=[dict(role="system",content=f"You are a careful expert in causal inference and {context}."),
            dict(role="user",content=build_skeleton_prompt(shuffled,descriptions=descriptions,domain_context=context))])


def collect_gpc(t, request, names):
    from cd_llm.experiments.alarm_llm_recovery import IdentifierRepair
    from cd_llm.baselines.guess2graph2026.llm_expert import parse_skeleton_response, edges_to_skeleton
    class Budgeted:
        root = t.root
        def query(self, req):
            if req["messages"][-1]["content"].startswith("Repair only the JSON format"):
                req = dict(req,max_tokens=12288)
            return t.query(req)
    response,_ = IdentifierRepair(Budgeted(),names).query(request)
    edges = parse_skeleton_response(json.loads(response["choices"][0]["message"]["content"]),names)
    return dict(edges=edges,skeleton=edges_to_skeleton(names,edges).tolist())


def long_expert(t, context):
    from cd_llm.baselines.long2023.expert import VERBS,prompt_for
    def decide(a,b,first,second):
        seed = hashlib.sha256(f"0|{first}|{second}".encode()).digest()
        rng = random.Random(int.from_bytes(seed[:8],"big"))
        verb,swapped = rng.choice(VERBS), bool(rng.randrange(2))
        prompt = prompt_for(first,second,verb,swapped)
        response,key = t.query(dict(model=MODEL,messages=[dict(role="system",content=context),dict(role="user",content=prompt)],
            temperature=0,seed=0,max_tokens=1,logprobs=True,top_logprobs=20))
        choice = response["choices"][0]
        answer = choice["message"]["content"].strip().upper()
        if answer not in ("A","B"):
            raise ValueError("invalid native Long A/B answer")
        masses = dict(A=0.,B=0.)
        for item in choice["logprobs"]["content"][0]["top_logprobs"]:
            token = item["token"].strip().strip("() ").upper()
            if token in masses:
                masses[token] += math.exp(item["logprob"])
        total = sum(masses.values())
        if total<=0 or masses[answer]<=0:
            raise ValueError("missing comparable A/B log-probabilities")
        confidence = max(.5,masses[answer]/total)
        left,right = (b,a) if swapped else (a,b)
        return ((left,right) if answer=="A" else (right,left)),confidence,dict(request=key,prompt=prompt,answer=answer,
            verb=verb,swapped=swapped,input_tokens=response["usage"]["prompt_tokens"],output_tokens=response["usage"]["completion_tokens"])
    return decide


def long_fit(t, cpdag, names, descriptions, context):
    from cd_llm.baselines.long2023 import run_long
    result = run_long(cpdag,tuple(f"{n} ({descriptions[n]})" for n in names),long_expert(t,context),
        strategy="srisk",tolerance=.1,max_extensions=50000)
    return dict(distribution=[dict(edges=graph_edges(g)["directed"],weight=1/len(result.final_mec)) for g in result.final_mec],
        initial_mec_size=len(result.initial_mec),retained_probability=result.retained_probability,
        decisions=result.decisions,expert_records=result.expert_records)


def bfs_fit(t, values, names, descriptions, context):
    import pandas as pd
    from cd_llm.baselines.jiralerspong2024 import Completion,run_bfs
    from cd_llm.experiments import alarm_bfs_openrouter as bfs
    from cd_llm.experiments.alarm_observational import edge_matrix
    def complete(messages):
        response,_ = t.query(dict(model=MODEL,messages=bfs.bfs_messages(messages),max_tokens=4096,temperature=0,seed=0))
        return Completion(bfs.completion_text(response["choices"][0]),response["usage"]["prompt_tokens"],response["usage"]["completion_tokens"])
    edges,usage = run_bfs(pd.DataFrame(values,columns=names),complete,include_statistics=True,
        descriptions=descriptions,domain_context=context,opaque_identifiers=False,empty_root_policy="empty_graph")
    return dict(matrix=edge_matrix(len(names),edges).tolist(),native_usage=usage)


def bootstrap_probabilities(matrices,names):
    from cd_llm.baselines.scp2025.evaluation import PCBootstrapProbabilities
    counts = [{},{},{}]
    for matrix in np.asarray(matrices):
        for a,b in combinations(range(len(names)),2):
            marks = tuple(matrix[a,b] for a,b in ((a,b),(b,a)))
            if marks==(0,0):
                continue
            if marks==(1,2): category,pair=0,(names[a],names[b])
            elif marks==(2,1): category,pair=0,(names[b],names[a])
            elif marks==(1,1): category,pair=1,tuple(sorted((names[a],names[b])))
            elif marks==(2,2): category,pair=2,tuple(sorted((names[a],names[b])))
            else: raise ValueError("invalid bootstrap endpoints")
            counts[category][pair] = counts[category].get(pair,0)+1
    return PCBootstrapProbabilities(*[{k:v/len(matrices) for k,v in c.items()} for c in counts])


def scp_prompt(names, descriptions, context, n, stats, a, b):
    from cd_llm.baselines.scp2025.prompts import build_knowledge_prompt
    return build_knowledge_prompt(pattern=2,source=names[a],target=names[b],nodes=names,method="PC",
        dataset_description=f"{n} categorical observations. {context}",theme="a clinical system",
        variable_descriptions=descriptions,pc_bootstrap_probabilities=stats)


def scp_pair(t, prompt, source, target, *, temperature=0.7):
    from cd_llm.baselines.scp2025.client import CachedOpenAIJudge,LLMSettings
    from cd_llm.baselines.scp2025.prompts import build_decision_prompt
    class Judge(CachedOpenAIJudge):
        # The raw response ledger is the only cache. Replaying it reconstructs
        # logical usage, including all five trials and native parsing repairs.
        def _read_cache(self,*args): return None
        def _write_cache(self,*args): pass
    judge = Judge(cache_dir=t.root,settings=LLMSettings(model=MODEL,temperature=temperature,request_seed=0),client=client(t))
    explanation = judge.explanation(prompt)
    probability,trials = judge.yes_probability(build_decision_prompt(knowledge_prompt=prompt,
        explanation=explanation,source=source,target=target))
    return dict(source=source,target=target,mean_yes_probability=probability,trials=trials)


def scp_fit(data,names,rows):
    from causallearn.search.ConstraintBased.PC import pc
    from cd_llm.baselines.scp2025.constraints import build_pc_background_knowledge
    from cd_llm.experiments.alarm_observational import SETTINGS,_causal_learn_partial
    probabilities = {(r["source"],r["target"]):r["mean_yes_probability"] for r in rows}
    if len(probabilities) != len(names)*(len(names)-1):
        raise ValueError("incomplete SCP pair collection")
    knowledge,applications = build_pc_background_knowledge(names,probabilities,forbid_threshold=.05,require_threshold=.95)
    result = pc(data,**SETTINGS["pc"],background_knowledge=knowledge,verbose=False,show_progress=False,node_names=names)
    partial = _causal_learn_partial(result.G,"cpdag")
    return dict(matrix=np.asarray(partial.marks).reshape(len(names),-1).tolist(),applications=[asdict(a) for a in applications])


def order_modules():
    from cd_llm.baselines.causal_order2025.upstream_backend import _load_upstream_modules,verified_upstream_root
    return _load_upstream_modules(verified_upstream_root())


def order_messages(messages, pair=False):
    from cd_llm.baselines.causal_order2025.local_vllm_backend import _qwen_chat_messages
    messages = _qwen_chat_messages(messages)
    messages[-1]["content"] += "\n\n"+(PAIR_CONCISE if pair else CONCISE)
    return messages


def order_prompts(names,descriptions,context):
    from cd_llm.baselines.causal_order2025.local_vllm_backend import _triplet_prompt,_pair_prompt
    mod = order_modules()
    triplets = [dict(nodes=list(group),messages=order_messages(_triplet_prompt(mod.triplet,group,context,descriptions)))
        for group in combinations(names,3)]
    pairs = {f"{a}:{b}":order_messages(_pair_prompt(mod.triplet.cot_tiebreaker_prompt,names[a],names[b],names,context,descriptions),True)
        for a,b in combinations(range(len(names)),2)}
    return dict(triplets=triplets,pairs=pairs)


def parse_order(response,nodes=None):
    import ast
    import networkx as nx
    from cd_llm.experiments.neuropathic_causal_order_terra import pair_answer
    choice = response["choices"][0]
    if choice["finish_reason"]!="stop": return None
    text = re.sub(r"(\])>(</Answer>)",r"\1\2",choice["message"]["content"])
    if nodes is None: return pair_answer(text)
    tags = re.findall(r"<Answer>(.*?)</Answer>",text,re.S)
    for tag in reversed(tags):
        try:
            entries=ast.literal_eval(tag.strip())
            if not isinstance(entries,list): continue
            entries=[x if isinstance(x,tuple) else (x,) for x in entries]
            if any(len(x) not in (1,2) or any(v not in nodes for v in x) for x in entries): continue
            edges=[e for e in entries if len(e)==2]
            g=nx.DiGraph(edges)
            if not nx.is_directed_acyclic_graph(g): continue
            return dict(tag=tag.strip(),entries=entries)
        except (ValueError,SyntaxError,TypeError):
            continue
    return None
