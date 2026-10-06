"""Matched soft-expert acquisition study from archived data-only PC grids."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import combinations, product
import math
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from cd_llm.core.finite_support import stable_seed
from cd_llm.experiments.claim_fuse_smc_eig import (
    _graph, _initial_particle_population, _metrics, _serialize,
)
from cd_llm.experiments.common.artifacts import (
    read_json, run_id, sha256, write_csv, write_json, write_text,
)
from cd_llm.methods.active_query import QueryResponse, active_query_smc, dag_query_pool, query_values
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "synthetic_pc_grid_ablation"
SOURCE_RUN = "20260816T145618Z"
SCMS = (1, 2, 7, 12, 15, 34, 36, 40, 45, 47)
SIZES = (200, 1000)
ACCURACIES = (0.6, 0.8, 1.0)
BUDGETS = (5, 20, 50)
GROUPS = {
    "group_i": ("local_edge",),
    "group_ii": ("ancestry",),
    "all": ("local_edge", "ancestry"),
}
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
DIAGNOSTICS = ("entropy", "mode_distance", "runtime_seconds", "realized_accuracy",
               "mutation_attempts", "mutation_acceptance", "graph_mass_ess")
SETTINGS = {
    "particle_count": 384, "initialization_updates": 0, "pc_alphas": [0.01, 0.05, 0.1],
    "max_parents": 12, "ess_threshold_fraction": 0.5, "rejuvenation_steps": 2,
    "rejuvenate_after_every_query": False, "guided_mixture_probability": 0.5,
    "proposal_cache_size": 256, "ordinary_kernel": "uniform_pair",
    "soft_confidence_cap": 0.99, "treatment": "soft_evidence", "policy": "eig",
}


def tasks():
    return [dict(task_id=i, graph_index=g, sample_size=n, accuracy=a, group=group)
            for i, (g, n, a, group) in enumerate(product(SCMS, SIZES, ACCURACIES, GROUPS))]


def code_hashes():
    package = Path(__file__).resolve().parents[1]
    names = (
        "experiments/synthetic_pc_grid_ablation.py", "experiments/claim_fuse_smc_eig.py",
        "experiments/common/artifacts.py", "core/finite_support.py",
        "methods/active_query.py", "methods/evidence_guided.py",
        "methods/adaptive_pruning.py", "methods/graph_scores.py", "methods/causal_elicitation.py",
    )
    return {name: sha256(package / name) for name in names}


def pc_grid_distribution(payload, n):
    """Read *pre-evidence* probabilities, never the archived fused posterior."""
    protocol = payload["protocol"]
    if protocol["node_count"] != 20 or protocol["pc_alphas"] != SETTINGS["pc_alphas"]:
        raise ValueError("source is not the locked 20-node, three-alpha PC grid")
    artifact = payload["samples"][str(n)]["methods"]["CEG-Fuse (PC grid)"]
    if not artifact["feasible"]:
        raise ValueError("PC grid unavailable; SCM substitution is forbidden")
    graphs = artifact["graphs"]
    weights = artifact["diagnostics"]["data_probabilities"]
    if not graphs or len(graphs) != len(weights):
        raise ValueError("missing or mismatched data-only weights")
    distribution = {}
    for item, weight in zip(graphs, weights):
        graph = _graph(item, 20)
        if graph.regime != "dag" or not math.isfinite(weight) or weight <= 0:
            raise ValueError("invalid PC grid graph/weight")
        distribution[graph] = distribution.get(graph, 0.0) + weight
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9):
        raise ValueError("data-only probabilities are not normalized")
    return distribution


def check_manifest(root):
    manifest = read_json(root / "manifest.json")
    if manifest["settings"] != SETTINGS or manifest["tasks"] != tasks():
        raise ValueError("run configuration differs; use a new timestamp")
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("inference source changed since prepare; use a new timestamp")
    return manifest


def prepare(root, results_root):
    if (root / "manifest.json").exists():
        return check_manifest(root)
    root.mkdir(parents=True, exist_ok=True)
    inputs, sources = {}, {}
    for g in SCMS:
        source = results_root / SOURCE_RUN / "partials" / f"task_{g:03d}.json"
        payload = read_json(source)
        if payload["graph_index"] != g:
            raise ValueError("source SCM mismatch")
        sources[str(g)] = dict(path=str(source.resolve()), sha256=sha256(source), seeds=payload["seeds"])
        for n in SIZES:
            data = np.asarray(payload["nested_data"][str(n)], dtype=float)
            if data.shape != (n, 20) or not np.isfinite(data).all():
                raise ValueError("source data shape or values differ")
            distribution = pc_grid_distribution(payload, n)
            score = GaussianBICScore(data, standardize_data=True, max_parents=12)
            for graph in distribution:
                score.parent_masks(graph)
            init_seed = stable_seed(STUDY, "initialization", g, n)
            particles, weights, empirical = _initial_particle_population(distribution, init_seed)
            truth = _graph(payload["target"], 20)
            key = f"s{g:02d}_n{n}"
            target = root / "inputs" / f"{key}.json"
            write_json(target, dict(
                graph_index=g, sample_size=n, data_seed=payload["seeds"]["data_seed"],
                initialization_seed=init_seed, data=data.tolist(), truth=payload["target"],
                source_sha256=sources[str(g)]["sha256"],
                raw_data_distribution=_serialize(distribution),
                particles=[_serialize({graph: 1.0})[0]["graph"] for graph in particles],
                weights=weights.tolist(), initialization=_serialize(empirical),
                baseline={k: v for k, v in _metrics(empirical, truth).items() if math.isfinite(v)},
            ))
            inputs[key] = sha256(target)
    code_root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(study=STUDY, version=1, source_run=SOURCE_RUN, settings=SETTINGS,
                    tasks=tasks(), budgets=BUDGETS, sources=sources, inputs=inputs,
                    code_hashes=code_hashes(), code_commit=commit,
                    interpretation="Accuracy and confidence vary together; ten existing common-complete SCM/data pairs.")
    write_json(root / "manifest.json", manifest, exclusive=True)
    return manifest


class Expert:
    def __init__(self, truth, graph_index, sample_size, accuracy):
        self.truth, self.graph_index, self.sample_size, self.accuracy = truth, graph_index, sample_size, accuracy
        self.records = []

    def __call__(self, query):
        seed = stable_seed(STUDY, "expert", self.graph_index, self.sample_size, query.query_id)
        uniform = float(np.random.default_rng(seed).random())
        true_answer = int(query_values((self.truth,), query)[0])
        correct = uniform < self.accuracy
        answer = true_answer if correct else 1 - true_answer
        self.records.append(dict(query_id=query.query_id, true_answer=true_answer, answer=answer,
                                 correct=correct, uniform=uniform))
        return QueryResponse(answer, self.accuracy, treatment="soft_evidence")


def input_for(root, manifest, task):
    key = f"s{task['graph_index']:02d}_n{task['sample_size']}"
    path = root / "inputs" / f"{key}.json"
    if sha256(path) != manifest["inputs"][key]:
        raise ValueError("prepared input changed")
    return read_json(path), manifest["inputs"][key]


def validate_result(payload, task, input_hash):
    if payload["task"] != task or payload["input_sha256"] != input_hash:
        raise ValueError("result identity or initialization mismatch")
    if len(payload["trace"]) != 50 or [row["budget"] for row in payload["rows"]] != list(BUDGETS):
        raise ValueError("incomplete trajectory/checkpoints")
    ids = [step["query"]["query_id"] for step in payload["trace"]]
    if len(set(ids)) != 50:
        raise ValueError("duplicate queried predicate")
    for step in payload["trace"]:
        if (step["treatment"] != "soft_evidence" or step["support_recovered"]
                or step["rejuvenation_hard_constraint_rejections"] != 0
                or not math.isclose(step["effective_confidence"], min(task["accuracy"], 0.99))):
            raise ValueError("response did not retain the required soft semantics")
    for budget in BUDGETS:
        distribution = payload["distributions"][str(budget)]
        masses = [item["probability"] for item in distribution]
        if not masses or not all(math.isfinite(x) and x > 0 for x in masses) or not math.isclose(sum(masses), 1.0, abs_tol=1e-9):
            raise ValueError("invalid checkpoint distribution")
    for row in payload["rows"]:
        if any(not math.isfinite(row[k]) for k in (*METRICS, *DIAGNOSTICS)):
            raise ValueError("nonfinite result metric")


def compute(root, task_id):
    started = time.perf_counter()
    manifest = check_manifest(root)
    if not 0 <= task_id < len(tasks()):
        raise ValueError("task index outside the fixed 180-trajectory grid")
    task = tasks()[task_id]
    source, input_hash = input_for(root, manifest, task)
    output = root / "partials" / f"task_{task_id:03d}.json"
    if output.exists():
        validate_result(read_json(output), task, input_hash)
        return
    truth = _graph(source["truth"], 20)
    expert = Expert(truth, task["graph_index"], task["sample_size"], task["accuracy"])
    particles = tuple(_graph(item, 20) for item in source["particles"])
    pool = dag_query_pool(20, GROUPS[task["group"]])
    seed = stable_seed(STUDY, "inference", task["graph_index"], task["sample_size"])
    score = GaussianBICScore(np.asarray(source["data"]), standardize_data=True, max_parents=12)
    result = active_query_smc(
        None, score, expert, query_pool=pool, budgets=BUDGETS,
        confidence=task["accuracy"], treatment="soft_evidence", soft_confidence_cap=0.99,
        policy="eig", seed=seed, initial_particles=particles, initial_weights=source["weights"],
        ess_threshold_fraction=0.5, rejuvenation_steps=2, rejuvenate_after_every_query=False,
        guided_mixture_probability=0.5, proposal_cache_size=256, ordinary_kernel="uniform_pair",
    )
    trace = [dict(asdict(step), effective_confidence=step.effective_confidence,
                  treatment=step.treatment) for step in result.steps]
    rows = []
    for budget in BUDGETS:
        steps = trace[:budget]
        counts = Counter(step["query"]["kind"] for step in steps)
        attempts = sum(step["rejuvenation_attempted_moves"] for step in steps)
        accepted = sum(step["rejuvenation_accepted_moves"] for step in steps)
        metrics = _metrics(result.distributions[budget], truth)
        rows.append(dict(task, budget=budget, data_seed=source["data_seed"],
                         reported_confidence=task["accuracy"], effective_confidence=min(task["accuracy"], 0.99),
                         **{key: metrics[key] for key in (*METRICS, "entropy", "mode_distance")},
                         runtime_seconds=steps[-1]["cumulative_seconds"],
                         realized_accuracy=statistics.fmean(r["correct"] for r in expert.records[:budget]),
                         mutation_attempts=attempts, mutation_acceptance=accepted / attempts if attempts else 0.0,
                         graph_mass_ess=steps[-1]["graph_mass_ess_after"],
                         directed_queries=counts["directed"], adjacency_queries=counts["adjacency"],
                         ancestry_queries=counts["ancestry"], resampling_count=sum(s["resampled"] for s in steps)))
    payload = dict(task=task, input_sha256=input_hash, code_hashes=manifest["code_hashes"],
                   inference_seed=seed, query_pool_size=len(pool), trace=trace, responses=expert.records,
                   rows=rows, distributions={str(b): _serialize(result.distributions[b]) for b in BUDGETS},
                   wall_seconds=time.perf_counter() - started)
    validate_result(payload, task, input_hash)
    write_json(output, payload, exclusive=True)
    print(f"completed task {task_id}: {payload['wall_seconds']:.2f}s", flush=True)


def summarize(rows, baseline):
    index = {(r["sample_size"], r["accuracy"], r["group"], r["budget"], r["graph_index"]): r for r in rows}
    expected = set(product(SIZES, ACCURACIES, GROUPS, BUDGETS, SCMS))
    if set(index) != expected or len(rows) != len(expected):
        raise ValueError("aggregation requires exactly ten paired SCMs in every cell")
    controls = {(r["sample_size"], r["graph_index"]): r for r in baseline}
    if set(controls) != set(product(SIZES, SCMS)) or len(baseline) != 20:
        raise ValueError("aggregation requires all 20 unique controls")
    summary, contrasts = [], []
    for n, a, group, budget in product(SIZES, ACCURACIES, GROUPS, BUDGETS):
        cell = [index[(n, a, group, budget, g)] for g in SCMS]
        record = dict(sample_size=n, accuracy=a, effective_confidence=min(a, 0.99), group=group,
                      budget=budget, scm_count=len(SCMS))
        for key in (*METRICS, *DIAGNOSTICS):
            values = [r[key] for r in cell]
            record[key + "_mean"] = statistics.fmean(values)
            record[key + "_se"] = statistics.stdev(values) / math.sqrt(len(SCMS))
        summary.append(record)
    pairs = list(combinations(GROUPS, 2)) + [(g, "initialization") for g in GROUPS]
    for n, a, budget, (first, second), metric in product(SIZES, ACCURACIES, BUDGETS, pairs, METRICS):
        delta = np.asarray([index[(n, a, first, budget, g)][metric] -
                            (controls[(n, g)][metric] if second == "initialization" else
                             index[(n, a, second, budget, g)][metric]) for g in SCMS])
        rng = np.random.default_rng(stable_seed(STUDY, "bootstrap", n, a, budget, first, second, metric))
        bootstrap = delta[rng.integers(0, len(SCMS), size=(10000, len(SCMS)))].mean(axis=1)
        lo, hi = np.quantile(bootstrap, (0.025, 0.975))
        contrasts.append(dict(sample_size=n, accuracy=a, budget=budget, first=first, second=second,
                              metric=metric, mean_difference=float(delta.mean()), ci_lower=float(lo),
                              ci_upper=float(hi), scm_count=10, bootstrap_resamples=10000))
    return summary, contrasts


def aggregate(root):
    from cd_llm.metrics.units import metric_metadata
    manifest = check_manifest(root)
    missing = [t["task_id"] for t in tasks() if not (root / "partials" / f"task_{t['task_id']:03d}.json").exists()]
    if missing:
        write_json(root / "incomplete.json", dict(missing_tasks=missing))
        raise ValueError(f"cannot aggregate: {len(missing)} trajectories missing")
    baseline, rows = [], []
    for g, n in product(SCMS, SIZES):
        source, _ = input_for(root, manifest, dict(graph_index=g, sample_size=n))
        baseline.append(dict(graph_index=g, sample_size=n, **source["baseline"]))
    for task in tasks():
        key = f"s{task['graph_index']:02d}_n{task['sample_size']}"
        payload = read_json(root / "partials" / f"task_{task['task_id']:03d}.json")
        validate_result(payload, task, manifest["inputs"][key])
        if payload["code_hashes"] != manifest["code_hashes"]:
            raise ValueError("mixed inference implementations in one run")
        rows.extend(payload["rows"])
    summary, contrasts = summarize(rows, baseline)
    write_csv(root / "per_scm.csv", rows)
    write_csv(root / "baseline.csv", baseline)
    write_csv(root / "summary.csv", summary)
    write_csv(root / "paired_contrasts.csv", contrasts)
    for name in ("per_scm", "baseline", "summary", "paired_contrasts"):
        write_json(root / f"{name}.csv.metrics.json", metric_metadata(20))
    lines = ["# Synthetic PC grid active ablation", "", "180/180 trajectories complete; 540 checkpoints and 20 shared initializations.", "",
             "Ten existing common-complete 20-node SCM/data pairs; nested n=200/1000. PC alpha grid: 0.01, 0.05, 0.10. No previous passive evidence is loaded.", "",
             "EIG uses canonical Group I, Group II, or combined pools. All answers are soft. Reported confidence matches actual accuracy; confidence 1.0 is capped to 0.99. Accuracy and confidence vary together.", "",
             "Budgets 5, 20, 50 are checkpoints of one trajectory. Mean (SE) is over ten SCMs. Mutation is triggered by resampling and the final query, so mutation effort can differ across conditions.", "",
             "Paired intervals use 10,000 SCM-level bootstrap resamples. They are descriptive, without multiplicity correction. Lower ED/ES and higher F1 are better.", "",
             "| n | Accuracy | Group | Budget | ED | ES | Adjacency F1 | Orientation F1 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        values = " | ".join(f"{row[m + '_mean']:.4f} ({row[m + '_se']:.4f})" for m in METRICS)
        lines.append(f"| {row['sample_size']} | {row['accuracy']:.1f} | {row['group']} | {row['budget']} | {values} |")
    lines += ["", "## Paired ED comparisons at 50 queries", "", "Differences below are All minus Group I; negative values favor All.", ""]
    for n, a in product(SIZES, ACCURACIES):
        c = next(c for c in contrasts if c["sample_size"] == n and c["accuracy"] == a and c["budget"] == 50 and c["first"] == "group_i" and c["second"] == "all" and c["metric"] == "expected_distance")
        lines.append(f"- n={n}, accuracy={a:.1f}: {-c['mean_difference']:.5f}, 95% CI [{-c['ci_upper']:.5f}, {-c['ci_lower']:.5f}].")
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "completion.json", dict(trajectories=180, checkpoints=540, controls=20))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "compute", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=WORKSPACE_ROOT / "results")
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args(argv)
    root = args.results_root / run_id(args.run_id) / STUDY
    if args.stage == "prepare":
        prepare(root, args.results_root)
    elif args.stage == "aggregate":
        aggregate(root)
    else:
        if args.task_index is None:
            parser.error("compute requires --task-index")
        compute(root, args.task_index)


if __name__ == "__main__":
    main()
