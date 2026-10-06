"""Shared categorical pairwise facts with native external-knowledge interfaces."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, asdict
from itertools import combinations
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np

from cd_llm.baselines.amirkhani2017.accuracy import AccuracyPrior
from cd_llm.baselines.amirkhani2017.api import fit_amirkhani_from_local_score
from cd_llm.baselines.amirkhani2017.opinions import ExpertOpinion, EdgeState
from cd_llm.baselines.amirkhani2017.search import AmirkhaniConfig
from cd_llm.baselines.camml2006.api import fit_camml
from cd_llm.baselines.camml2006.priors import local_pair_prior
from cd_llm.baselines.camml2006.types import CaMMLConfig, CaMMLPrior
from cd_llm.baselines.eca2025.api import fit_eca
from cd_llm.baselines.eca2025.types import ECAConfig, ECAPrior, EdgeConstraint
from cd_llm.core.finite_support import stable_seed, standardize, resolve_evidence_confidence
from cd_llm.experiments.claim_fuse_smc_eig import _graph, _serialize
from cd_llm.experiments.common.artifacts import read_json, write_json, write_csv, sha256, run_id, json_value
from cd_llm.experiments import synthetic_passive_claim_budgets as archive
from cd_llm.methods.active_query import ClaimQuery, QueryResponse, active_query_smc, query_values
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.metrics.units import metric_metadata
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "synthetic_pairwise_expert_benchmark"
INITIALIZATION_RUN = "20260920T215421Z"
SCMS, SIZES = archive.SCMS, archive.SIZES
METHODS = ("PC", "GES", "NOTEARS", "CaMML", "Marg", "ECA", "FS", "SMC-Passive")
NEW_METHODS = METHODS[3:]
METRICS = ("expected_distance", "adjacency_f1", "orientation_f1")
MARG_PRIOR = AccuracyPrior(6.1, 2.05, 1.85, 7.75, 2.25)
SMC_SETTINGS = dict(ess_threshold_fraction=0.5, rejuvenation_steps=2,
    rejuvenate_after_every_query=False, guided_mixture_probability=0.5,
    proposal_cache_size=256, ordinary_kernel="uniform_pair", soft_confidence_cap=0.99)
SETTINGS = dict(node_count=20, facts=20, draws_per_scm=1, reported_confidence=1.0,
    pair_selection="uniform unordered without replacement, before truth lookup",
    initialization="archived PC alpha=0.05; 384 ordered particle slots",
    smc=dict(**SMC_SETTINGS, budgets=[20], policy="fixed", treatment="soft_evidence", max_parents=12),
    camml=asdict(CaMMLConfig()), marg=asdict(AmirkhaniConfig()),
    marg_prior=asdict(MARG_PRIOR), marg_score_max_parents=19,
    eca=asdict(ECAConfig(backend="paper", standardize_data=False)),
    data_score="gaussian-bic", standardization="once, separately for each sample size")


def plain(value):
    """Serialize diagnostics without deepcopying immutable mapping proxies."""
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return json_value(value)


def root_for(identifier, results_root=WORKSPACE_ROOT / "results"):
    return Path(results_root) / run_id(identifier) / STUDY


def selected_pairs(node_count, seed, count=20):
    pool = tuple(combinations(range(node_count), 2))
    return [pool[int(i)] for i in np.random.default_rng(seed).choice(len(pool), count, replace=False)]


def answer_pairs(truth, pairs):
    records = []
    for i, (left, right) in enumerate(pairs):
        state = ("left_to_right" if truth.directed[left] & (1 << right) else
                 "right_to_left" if truth.directed[right] & (1 << left) else "absent")
        records.append(dict(fact_id=f"pair-{i:02d}", left=left, right=right, state=state,
                            expert_id="synthetic-oracle", reported_confidence=1.0))
    return records


def native_inputs(records, node_count=20):
    """Only saved answers enter these builders; no target graph is available."""
    seen = set()
    queries, local, constraints, opinions = [], [], [], []
    for r in records:
        left, right, state = r["left"], r["right"], r["state"]
        if not 0 <= left < right < node_count or (left, right) in seen:
            raise ValueError("invalid or repeated pair")
        if r["reported_confidence"] != 1.0 or r["expert_id"] != "synthetic-oracle":
            raise ValueError("raw expert identity or confidence differs")
        seen.add((left, right))
        EdgeState(state)
        forward, reverse = float(state == "left_to_right"), float(state == "right_to_left")
        source, target = (right, left) if reverse else (left, right)
        queries.append(dict(kind="adjacency" if state == "absent" else "directed",
            source=source, target=target, query_id=r["fact_id"], answer=int(state != "absent"),
            reported_confidence=1.0, effective_confidence=0.99, treatment="soft_evidence"))
        local.append(local_pair_prior(left, right, left_to_right=forward, right_to_left=reverse))
        constraints.extend((EdgeConstraint(left, right, forward), EdgeConstraint(right, left, reverse)))
        opinions.append(ExpertOpinion(r["expert_id"], left, right, EdgeState(state)))
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("duplicate fact identifier")
    return dict(FS=queries, **{"SMC-Passive": queries}, CaMML=CaMMLPrior(node_count, local=tuple(local)),
                ECA=ECAPrior(node_count, tuple(constraints)), Marg=tuple(opinions))


def query(record):
    return ClaimQuery(**{k: record[k] for k in ("kind", "source", "target", "query_id")})


def finite_posterior(initial, records):
    graphs = tuple(initial)
    log_weights = np.log([initial[g] for g in graphs])
    for r in records:
        confidence = resolve_evidence_confidence(r["reported_confidence"], treatment=r["treatment"],
                                                 soft_confidence_cap=0.99).effective_confidence
        agrees = query_values(graphs, query(r)) == r["answer"]
        log_weights += np.where(agrees, math.log(confidence), math.log1p(-confidence))
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    return dict(zip(graphs, map(float, weights)))


def passive(score, particles, weights, records, seed):
    by_id = {r["query_id"]: r for r in records}
    def answer(q):
        r = by_id[q.query_id]
        return QueryResponse(r["answer"], r["reported_confidence"], treatment="soft_evidence", audit=r)
    result = active_query_smc(None, score, answer, query_pool=tuple(map(query, records)),
        budgets=(20,), policy="fixed", confidence=1.0, treatment="soft_evidence", seed=seed,
        initial_particles=particles, initial_weights=weights, **SMC_SETTINGS)
    if [step.query.query_id for step in result.steps] != list(by_id):
        raise ValueError("SMC did not consume the frozen order")
    return result


def implementation_hashes():
    package = Path(__file__).resolve().parents[1]
    paths = set(Path(p) for p in archive.code_hashes()) | {Path(__file__).resolve()}
    for family in ("camml2006", "amirkhani2017", "eca2025"):
        paths.update((package / "baselines" / family).glob("*.py"))
    return {str(p): sha256(p) for p in sorted(paths)}


def prepare(identifier, results_root):
    root = root_for(identifier, results_root)
    old, reuse = archive.archived_inputs(archive.root_for(INITIALIZATION_RUN, results_root))
    if reuse["initialization"] != "pc-only":
        raise ValueError("PC-only archive required")
    facts, seeds = {}, {}
    for g in SCMS:
        source = read_json(Path(old["sources"][str(g)]["path"]))
        data = {n: np.asarray(source["nested_data"][str(n)], dtype=float) for n in SIZES}
        if any(data[n].shape != (n, 20) or not np.isfinite(data[n]).all() for n in SIZES):
            raise ValueError("invalid observations")
        if not np.array_equal(data[200], data[1000][:200]):
            raise ValueError("observations are not nested")
        seed = stable_seed(STUDY, "pairs", g)
        pairs = selected_pairs(20, seed)
        records = answer_pairs(_graph(source["target"], 20), pairs)
        inputs = native_inputs(records)
        saved = read_json(Path(reuse["partials"][str(g)]["path"]))
        seeds[str(g)] = dict(pairs=seed, samples={})
        for n in SIZES:
            a = saved["artifacts"][str(n)]
            fits, _, _, _, _ = archive.restored_initialization(a, "pc-only")
            if fits[0]["endpoint_matrix"] != source["samples"][str(n)]["methods"]["PC"]["diagnostics"]["output"]["endpoint_matrix"]:
                raise ValueError("initialization differs from archived PC")
            seeds[str(g)]["samples"][str(n)] = dict(**a["seeds"], camml=stable_seed(STUDY, "CaMML", g, n))
        dest = root / "facts" / f"scm_{g:03d}.json"
        write_json(dest, dict(records=records, native_inputs=plain(inputs)), exclusive=True)
        facts[str(g)] = dict(path=str(dest), sha256=sha256(dest))
    write_json(root / "manifest.json", dict(study=STUDY, version=1, run_id=identifier,
        graph_indices=SCMS, sample_sizes=SIZES, settings=SETTINGS, seeds=seeds,
        sources=old["sources"], facts=facts, reuse=reuse, implementation=implementation_hashes()), exclusive=True)
    print(root, flush=True)


def verify(root):
    m = read_json(root / "manifest.json")
    if m["study"] != STUDY or m["settings"] != plain(SETTINGS) or m["graph_indices"] != list(SCMS) or m["sample_sizes"] != list(SIZES):
        raise ValueError("registered protocol differs")
    for p, digest in m["implementation"].items():
        if sha256(Path(p)) != digest:
            raise ValueError(f"implementation changed: {p}")
    reuse = m["reuse"]
    for item in (*m["sources"].values(), *m["facts"].values(), reuse["manifest"], reuse["completion"], *reuse["partials"].values()):
        archive.check_hash(item)
    return m


def result_path(root, g, n, method):
    return root / "evaluations" / f"scm_{g:03d}_n{n}_{method}.json"


def fit_method(method, data, inputs, initial, particles, weights, seeds):
    if method == "FS":
        return finite_posterior(initial, inputs[method]), dict(effective_confidence=0.99)
    if method == "SMC-Passive":
        result = passive(GaussianBICScore(data, standardize_data=False, max_parents=12),
                         particles, weights, inputs[method], seeds["inference"])
        return result.distributions[20], dict(trace=plain(result.steps), timing=plain(result.timing),
            resampling_count=result.resampling_count, final_rejuvenation_budget=20)
    if method == "CaMML":
        result = fit_camml(data, inputs[method], data_score="gaussian-bic", standardize_data=False,
                           config=CaMMLConfig(seed=seeds["camml"]))
        return result.support, dict(diagnostics=plain(result.diagnostics), config=plain(result.config),
                                    total_log_score=result.total_log_score)
    if method == "Marg":
        result = fit_amirkhani_from_local_score(
            GaussianBICScore(data, standardize_data=False, max_parents=data.shape[1] - 1),
            inputs[method], config=AmirkhaniConfig(), priors=MARG_PRIOR, data_score_name="gaussian-bic")
        if not math.isfinite(result.total_log_score) or not result.diagnostics.local_optimum_reached:
            raise ValueError("Marg did not reach a finite local optimum")
        return {result.graph: 1.0}, dict(diagnostics=plain(result.diagnostics), config=plain(result.config),
            accuracy_prior=plain(MARG_PRIOR), total_log_score=result.total_log_score, moves=plain(result.moves))
    if method == "ECA":
        result = fit_eca(data, inputs[method], config=ECAConfig(**SETTINGS["eca"]))
        return result.support, dict(diagnostics=plain(result.diagnostics), config=plain(result.config),
            weighted_adjacency=plain(result.weighted_adjacency), objective=result.objective)
    raise ValueError(method)


def validate_prediction(method, distribution, details, truth):
    if method == "ECA" and (not math.isfinite(details["objective"]) or
            not details["diagnostics"]["converged"] or details["diagnostics"]["acyclicity"] > 1e-8):
        raise ValueError("ECA did not reach the registered acyclicity tolerance")
    values = archive.metrics(distribution, truth)
    return {m: values[m] for m in METRICS}


def compute(identifier, graph_index, results_root):
    if graph_index not in SCMS:
        raise ValueError("SCM outside fixed set")
    root = root_for(identifier, results_root)
    manifest = verify(root)
    digest = sha256(root / "manifest.json")
    g = graph_index
    source = read_json(Path(manifest["sources"][str(g)]["path"]))
    facts = read_json(Path(manifest["facts"][str(g)]["path"]))
    inputs = native_inputs(facts["records"])
    if plain(inputs) != facts["native_inputs"] or len(facts["records"]) != 20:
        raise ValueError("frozen method inputs differ")
    saved = read_json(Path(manifest["reuse"]["partials"][str(g)]["path"]))
    truth = _graph(source["target"], 20)
    for n in SIZES:
        data = standardize(np.asarray(source["nested_data"][str(n)], dtype=float))
        _, _, particles, weights, initial = archive.restored_initialization(saved["artifacts"][str(n)], "pc-only")
        seeds = manifest["seeds"][str(g)]["samples"][str(n)]
        for method in NEW_METHODS:
            path = result_path(root, g, n, method)
            if path.exists():
                previous = read_json(path)
                if previous["manifest_sha256"] != digest or previous["native_input"] != plain(inputs[method]):
                    raise ValueError("existing evaluation has different inputs")
                print(f"reuse {g} {n} {method}: {previous['status']}", flush=True)
                continue
            row = dict(graph_index=g, sample_size=n, method=method, budget=20,
                       manifest_sha256=digest, native_input=plain(inputs[method]), seeds=seeds)
            started = time.perf_counter()
            try:
                dist, details = fit_method(method, data, inputs, initial, particles, weights, seeds)
                row.update(distribution=_serialize(dist), details=details)
                row.update(metrics=validate_prediction(method, dist, details, truth), status="ok", error=None)
            except Exception as error:
                row.update(status="failed", metrics=None, error=f"{type(error).__name__}: {error}",
                           traceback=traceback.format_exc())
            row["seconds"] = time.perf_counter() - started
            write_json(path, row, exclusive=True)
            print(f"{g} {n} {method}: {row['status']} {row['seconds']:.3f}s {row['error'] or ''}", flush=True)


def summarize(rows):
    expected = {(g, n, method) for g in SCMS for n in SIZES for method in METHODS}
    keys = [(r["graph_index"], r["sample_size"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing or duplicate evaluation rows")
    summary = []
    for method in METHODS:
        for n in SIZES:
            selected = [r for r in rows if r["method"] == method and r["sample_size"] == n]
            count = sum(r["status"] == "ok" for r in selected)
            row = dict(method=method, sample_size=n, budget=0 if method in METHODS[:3] else 20,
                       success_count=count, total_count=len(SCMS))
            for metric in METRICS:
                values = [r[metric] for r in selected] if count == len(SCMS) else []
                if values and not all(math.isfinite(v) for v in values):
                    raise ValueError("nonfinite successful metric")
                row[metric + "_mean"] = statistics.fmean(values) if values else None
                row[metric + "_se"] = statistics.stdev(values) / math.sqrt(len(values)) if values else None
            summary.append(row)
    return summary


def aggregate(identifier, results_root):
    root = root_for(identifier, results_root)
    manifest = verify(root)
    digest = sha256(root / "manifest.json")
    rows, evaluations, failures = [], [], []
    for g in SCMS:
        source = read_json(Path(manifest["sources"][str(g)]["path"]))
        facts = read_json(Path(manifest["facts"][str(g)]["path"]))
        truth = _graph(source["target"], 20)
        for n in SIZES:
            for r in archive.baseline_rows(source, n):
                rows.append({k: v for k, v in r.items() if k != "energy_score"} | dict(status="ok", error=None))
            for method in NEW_METHODS:
                path = result_path(root, g, n, method)
                if not path.exists():
                    raise ValueError(f"missing evaluation: {path}")
                r = read_json(path)
                if r["manifest_sha256"] != digest or (r["graph_index"], r["sample_size"], r["method"]) != (g, n, method) or r["native_input"] != facts["native_inputs"][method]:
                    raise ValueError("evaluation provenance differs")
                if r["status"] == "ok":
                    dist = {_graph(v["graph"], 20): v["probability"] for v in r["distribution"]}
                    values = validate_prediction(method, dist, r["details"], truth)
                    if values != r["metrics"]:
                        raise ValueError("saved metrics differ from predictions")
                elif r["status"] == "failed" and r["error"]:
                    values = {m: None for m in METRICS}
                    failures.append(dict(graph_index=g, sample_size=n, method=method, error=r["error"]))
                else:
                    raise ValueError("invalid evaluation status")
                rows.append(dict(graph_index=g, sample_size=n, method=method, budget=20,
                    status=r["status"], error=r["error"], **values))
                evaluations.append(dict(path=str(path), sha256=sha256(path)))
    summary = summarize(rows)
    write_csv(root / "metrics.csv", rows)
    write_csv(root / "summary.csv", summary)
    write_json(root / "completion.json", dict(manifest_sha256=digest, evaluations=evaluations,
        result_rows=len(rows), new_evaluations=len(evaluations), baseline_evaluations=60,
        summary=summary, failures=failures, metric_metadata=metric_metadata(20),
        metrics_sha256=sha256(root / "metrics.csv"), summary_sha256=sha256(root / "summary.csv")))
    print(root / "completion.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "compute", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--graph-index", type=int)
    parser.add_argument("--results-root", type=Path, default=WORKSPACE_ROOT / "results")
    args = parser.parse_args(argv)
    if args.stage == "compute":
        compute(args.run_id, args.graph_index, args.results_root)
    else:
        globals()[args.stage](args.run_id, args.results_root)


if __name__ == "__main__":
    main()
