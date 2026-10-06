"""Qwen active interpretation with a fixed PC/GES initialization grid."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.core.finite_support import graph_edges, stable_seed, transitive_closure
from cd_llm.experiments.common import categorical_grid as grid
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_json
from cd_llm.experiments.neuropathic_active_interpretation import graph, provider_for, snapshot_cache
from cd_llm.methods.active_query import active_query_smc
from cd_llm.methods.graph_scores import BDeuScore, parent_masks
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_grid_interpretation"
GRID_SOURCE = WORKSPACE_ROOT / "results/20260911T161400Z/initialization_grid"
ANSWER_SOURCE = WORKSPACE_ROOT / "results/20260911T205557Z/neuropathic_active_interpretation"
BUDGETS = (20, 50, 100)
STRATEGIES = ("stated", "consistency")
ARMS = ("literal", "ancestry")


def root_for(timestamp):
    return WORKSPACE_ROOT / "results" / run_id(timestamp) / STUDY


def make_tasks():
    return [dict(id=f"qwen_{strategy}_s{seed:02d}_n{n}_b{budget:03d}", strategy=strategy,
        seed=seed, sample_size=n, budget=budget, condition=f"neuropathic_s{seed:02d}_n{n}")
        for budget in BUDGETS for strategy in STRATEGIES for n in (200,1000) for seed in range(10)]


def validate_population(population):
    particles = tuple(graph(dict(directed=edges)) for edges in population["particles"])
    weights = np.asarray(population["weights"], dtype=float)
    if len(particles) != 384 or weights.shape != (384,) or not np.isfinite(weights).all():
        raise ValueError("grid initialization must have 384 finite weights")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("grid initialization weights are not probabilities")
    for g in particles:
        transitive_closure(g)
        if max(p.bit_count() for p in parent_masks(g)) > 12:
            raise ValueError("grid initialization violates the parent bound")
    return particles, weights


def register(timestamp):
    root = root_for(timestamp)
    if (root / "manifest.json").exists():
        verify(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    old = read_json(GRID_SOURCE / "manifest.json")
    old_digest = sha256(GRID_SOURCE / "manifest.json")
    sources, inputs, conditions = {}, {}, []
    def copy(source, relative):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[relative] = sources[str(source)] = sha256(target)
    for name in ("model_metadata.json", "runtime.json"):
        copy(ANSWER_SOURCE / "inputs" / name, "inputs/" + name)
    copy(ANSWER_SOURCE / "inputs/truth_graph.json", "evaluation/truth_graph.json")
    for condition in old["conditions"]:
        if condition["panel"] != "neuropathic":
            continue
        cid = grid.condition_id(condition)
        data_path = GRID_SOURCE / condition["data_path"]
        if sha256(data_path) != old["input_sha256"][condition["data_path"]]:
            raise ValueError("source grid observations changed")
        data = np.load(data_path, allow_pickle=False)
        reference = pd.read_csv(ANSWER_SOURCE / "inputs/data" /
            f"seed_{condition['data_seed']:02d}_n{condition['sample_size']}.csv")
        names = [n["id"] for n in read_json(root / "inputs/model_metadata.json")["nodes"]]
        if list(reference.columns) != names or not np.array_equal(data, reference.to_numpy()):
            raise ValueError("grid observations or node order differ from Table 2")
        copy(data_path, condition["data_path"])
        conditions.append(dict(condition, id=cid))
        population = GRID_SOURCE / "initializations" / f"{cid}_grid_union.json"
        if population.exists():
            payload = read_json(population)
            if payload["manifest_sha256"] != old_digest or payload["condition"] != condition:
                raise ValueError("saved grid initialization provenance differs")
            validate_population(payload)
            copy(population, f"inputs/grid_initializations/{cid}.json")
    if len(conditions) != 20:
        raise ValueError("all twenty neuropathic observation conditions are required")
    for strategy in STRATEGIES:
        source = ANSWER_SOURCE / f"inputs/caches/qwen_{strategy}.json"
        historical = read_json(source)
        additions = snapshot_cache(ANSWER_SOURCE / "caches/qwen_incremental.sqlite3", strategy)
        for key, value in additions.items():
            if key in historical and historical[key]["record"]["content"] != value["record"]["content"]:
                raise ValueError("conflicting historical model responses")
            historical.setdefault(key, value)
        dest = root / f"inputs/caches/qwen_{strategy}.json"
        write_json(dest, historical)
        inputs[str(dest.relative_to(root))] = sha256(dest)
        sources[str(source)] = sha256(source)
    names = (
        "experiments/neuropathic_grid_interpretation.py", "experiments/neuropathic_grid_interpretation_report.py",
        "experiments/common/categorical_grid.py", "experiments/neuropathic_active_interpretation.py",
        "experiments/neuropathic_active_interpretation_report.py", "methods/active_query.py",
        "methods/evidence_guided.py", "methods/graph_scores.py", "methods/fast_bdeu.py",
        "methods/llm_claim_confidence.py", "core/finite_support.py",
        "baselines/causal_order2025/local_vllm_backend.py", "baselines/causal_order2025/openrouter_backend.py")
    implementations = {str((WORKSPACE_ROOT / "code/src/cd_llm" / n).resolve()):
                       sha256(WORKSPACE_ROOT / "code/src/cd_llm" / n) for n in names}
    config = {k: old["config"][k] for k in (
        "grid", "completion_limit", "max_parents", "particle_count", "initialization_updates")}
    write_json(root / "manifest.json", dict(version=1,study=STUDY,conditions=conditions,tasks=make_tasks(),
        inputs=inputs,sources=sources,implementation=implementations,initialization_config=config,
        source_grid_manifest_sha256=old_digest,source_answer_manifest_sha256=sha256(ANSWER_SOURCE / "manifest.json"),
        inference=dict(budgets=BUDGETS,budget_unit="attempted questions",strategies=STRATEGIES,
            arms=ARMS,selection_confidence=0.9,soft_confidence_cap=0.99,rejuvenation_steps=2,
            rejuvenate_after_every_query=False,ess_threshold_fraction=0.5,guided_mixture_probability=0.5,
            proposal_cache_size=256,ordinary_kernel="uniform_pair",independent_budget_runs=True),
        total_trajectories=240,matched_comparisons=120))
    print(json.dumps(dict(registered=str(root),tasks=120,trajectories=240,
        saved_initializations=sum((root / f"inputs/grid_initializations/{c['id']}.json").exists() for c in conditions))),flush=True)


def verify(root, initialized=False):
    manifest = read_json(root / "manifest.json")
    for name, digest in manifest["implementation"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"registered implementation changed {name}")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed {name}")
    if initialized:
        ready = read_json(root / "initializations_ready.json")
        if ready["manifest_sha256"] != sha256(root / "manifest.json") or len(ready["populations"]) != 20:
            raise ValueError("initialization registration is incomplete")
        for name, digest in ready["populations"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"shared initialization changed {name}")
    return manifest


def initialize(root):
    manifest = verify(root)
    digest = sha256(root / "manifest.json")
    with (root / "initialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        populations = {}
        for index, condition in enumerate(manifest["conditions"]):
            cid = condition["id"]
            dest = root / f"initializations/{cid}_grid_union.json"
            saved = root / f"inputs/grid_initializations/{cid}.json"
            if dest.exists():
                payload = read_json(dest)
                if payload["manifest_sha256"] != digest:
                    raise ValueError("initialization belongs to another manifest")
            elif saved.exists():
                payload = read_json(saved)
                payload.update(source_manifest_sha256=payload["manifest_sha256"],manifest_sha256=digest,
                    source_initialization_sha256=sha256(saved),reused=True)
                write_json(dest, payload)
            else:
                data, fits = grid.candidates(root,condition,manifest["initialization_config"],digest)
                payload = grid.initialize(root,condition,manifest["initialization_config"],digest,data,fits,"grid_union")
            validate_population(payload)
            populations[str(dest.relative_to(root))] = sha256(dest)
            print(json.dumps(dict(stage="initialization",completed=index+1,total=20,condition=cid,
                                  reused=saved.exists())),flush=True)
        write_json(root / "initializations_ready.json",dict(manifest_sha256=digest,populations=populations))


def inference_arguments(task, particles, weights, arm):
    """One terminal budget per call preserves the original end-of-run move."""
    if task["budget"] not in BUDGETS or arm not in ARMS:
        raise ValueError("unregistered budget or interpretation")
    return dict(budgets=(task["budget"],),confidence=0.9,treatment="soft_evidence",soft_confidence_cap=0.99,
        policy="eig",seed=stable_seed("neuropathic_decontamination","active",task["seed"],task["sample_size"]),
        ess_threshold_fraction=0.5,rejuvenation_steps=2,rejuvenate_after_every_query=False,
        guided_mixture_probability=0.5,proposal_cache_size=256,ordinary_kernel="uniform_pair",
        initial_particles=particles,initial_weights=weights,direct_yes_interpretation=arm)


def execute(root, task, provider, arm):
    data = np.load(root / f"inputs/{task['condition']}.npy",allow_pickle=False)
    pop_path = root / f"initializations/{task['condition']}_grid_union.json"
    particles, weights = validate_population(read_json(pop_path))
    score = BDeuScore(data,equivalent_sample_size=1.0,structure_prior=1.0,cardinalities=(2,)*20,max_parents=12)
    result = active_query_smc(None,score,provider,**inference_arguments(task,particles,weights,arm))
    if len(result.steps) != task["budget"] or len({s.query.query_id for s in result.steps}) != task["budget"]:
        raise ValueError("wrong number of distinct attempted questions")
    return dict(task=task,interpretation=arm,initialization_sha256=sha256(pop_path),
        trace=[asdict(s) for s in result.steps],
        distribution=[dict(graph=graph_edges(g),weight=w) for g,w in sorted(result.distributions[task["budget"]].items())],
        answered_queries=result.answered_query_count,elapsed_seconds=result.cumulative_seconds[task["budget"]])


def run(root, base_url):
    manifest = verify(root,initialized=True)
    manifest_digest = sha256(root / "manifest.json")
    with (root / "inference.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = ResponseCache(root / "caches/qwen_incremental.sqlite3")
        providers = {s:provider_for(root,"qwen",s,cache,base_url,online=True)[0] for s in STRATEGIES}
        completed = 0
        try:
            for task in manifest["tasks"]:
                for arm in ARMS:
                    destination = root / f"records/{arm}/{task['id']}.json"
                    if destination.exists():
                        record = read_json(destination)
                        if record["manifest_sha256"] != manifest_digest or record["task"] != task or record["interpretation"] != arm:
                            raise ValueError("completed trajectory does not match the frozen task")
                        completed += 1
                        continue
                    slot = 0
                    def provider(query):
                        nonlocal slot
                        slot += 1
                        write_json(root / "status/inference.json",dict(stage="answering",task=task["id"],
                            interpretation=arm,slot=slot,budget=task["budget"],completed=completed,total=240))
                        return providers[task["strategy"]](query)
                    result = execute(root,task,provider,arm)
                    result["manifest_sha256"] = manifest_digest
                    write_json(destination,result)
                    completed += 1
                    print(json.dumps(dict(completed=completed,total=240,task=task["id"],interpretation=arm,
                        seconds=result["elapsed_seconds"],answered=result["answered_queries"],requests=cache.summary()["request_count"])),flush=True)
                if completed % 80 == 0:
                    from cd_llm.experiments.neuropathic_grid_interpretation_report import report
                    report(root)
            write_json(root / "status/inference.json",dict(stage="complete",completed=240,total=240,accounting=cache.summary()))
        except BaseException as exc:
            write_json(root / "status/inference.json",dict(stage="failed",completed=completed,total=240,
                error=f"{type(exc).__name__}: {exc}",accounting=cache.summary()))
            raise
        finally:
            cache.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("register","initialize","run","report"))
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--base-url",default="http://127.0.0.1:18000/v1")
    args=parser.parse_args()
    root=root_for(args.run_id)
    if args.stage=="register":
        register(args.run_id)
    elif args.stage=="initialize":
        initialize(root)
    elif args.stage=="run":
        run(root,args.base_url)
    else:
        from cd_llm.experiments.neuropathic_grid_interpretation_report import report
        report(root)


if __name__=="__main__":
    main()
