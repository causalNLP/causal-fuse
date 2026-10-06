"""Matched fixed-order FS and SMC trajectories from a shared PC/GES grid."""
from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cd_llm.core.finite_support import (
    CompletionError, _causal_learn_partial, distribution_metrics, graph_edges, is_acyclic,
    learn_ges, point_prediction_metrics, sample_completions_with_diagnostics, stable_seed, standardize,
)
from cd_llm.experiments.claim_fuse_smc_eig import (
    TABLE41_COMMON_TASK_INDICES, _graph, _initial_particle_population, _serialize,
)
from cd_llm.experiments.common.artifacts import (
    json_value, read_json, run_id, sha256, write_csv, write_json, write_text,
)
from cd_llm.experiments.synthetic_table41 import _artifact_distribution, _partial
from cd_llm.methods.active_query import (
    ClaimQuery, QueryResponse, active_query_smc, group_i_query_pool, query_values,
)
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.methods.graph_scores import parent_masks
from cd_llm.metrics.units import metric_metadata
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "synthetic_passive_claim_budgets"
SOURCE_RUN = "20260816T145618Z"
SCMS = TABLE41_COMMON_TASK_INDICES
SIZES = (200, 1000)
BUDGETS = (10, 20, 50)
CONFIDENCE = 0.99
BASELINES = ("PC", "GES", "NOTEARS")
TABLE_GROUPS = ((0, BASELINES), *((b, ("FS", "SMC-Passive")) for b in BUDGETS))
BASELINE_SETTINGS = dict(pc_alpha=0.05, ges_penalty=0.5,
                         notears_l1=0.05, notears_threshold=0.30)
METRICS = ("expected_distance", "energy_score", "adjacency_f1", "orientation_f1")
GRID = tuple(
    [dict(id=f"pc_a{a:g}", family="PC", alpha=a) for a in (0.01, 0.05, 0.10)]
    + [dict(id=f"ges_l{v:g}", family="GES", penalty=v) for v in (0.25, 0.5, 1.0)]
)
INITIALIZATIONS = ("pc-ges-grid", "pc-only")
PC_CONFIG = next(config for config in GRID if config["id"] == "pc_a0.05")
SETTINGS = dict(
    node_count=20, particle_count=384, initialization_updates=0,
    completion_limit=200, max_parents=12, confidence=CONFIDENCE,
    treatment="soft_evidence", policy="fixed", ess_threshold_fraction=0.5,
    rejuvenation_steps=2, rejuvenate_after_every_query=False,
    guided_mixture_probability=0.5, proposal_cache_size=256,
    ordinary_kernel="uniform_pair", soft_confidence_cap=0.99,
    graph_prior="uniform", score="gaussian-bic", grid=GRID,
    budgets=BUDGETS, final_rejuvenation_budget=50,
)


def settings_for(initialization):
    if initialization not in INITIALIZATIONS:
        raise ValueError("unknown initialization")
    return {**SETTINGS, "grid": (PC_CONFIG,) if initialization == "pc-only" else GRID}


def root_for(identifier, results_root=WORKSPACE_ROOT / "results"):
    return results_root / run_id(identifier) / STUDY


def code_hashes():
    package = Path(__file__).resolve().parents[1]
    names = ("experiments/synthetic_passive_claim_budgets.py",
             "experiments/claim_fuse_smc_eig.py", "methods/active_query.py",
             "methods/evidence_guided.py", "methods/adaptive_pruning.py",
             "methods/graph_scores.py", "core/finite_support.py",
             "experiments/synthetic_table41.py")
    return {str(package / name): sha256(package / name) for name in names}


def claim_sequence(truth, seed, count=50):
    pool = group_i_query_pool(truth.node_count)
    indices = np.random.default_rng(seed).choice(len(pool), count, replace=False)
    return [dict(
        **asdict(pool[int(index)]), answer=int(query_values((truth,), pool[int(index)])[0]),
        reported_confidence=CONFIDENCE, effective_confidence=CONFIDENCE,
        treatment="soft_evidence", provenance_family=f"aq-source-{position}",
        context_id="target", context_compatible=True,
    ) for position, index in enumerate(indices)]


def validate_claims(records, node_count, confidence=CONFIDENCE):
    predicates = set()
    ids = set()
    for record in records:
        q = ClaimQuery(**{key: record[key] for key in ("kind", "source", "target", "query_id")})
        semantic = q.kind, q.source, q.target
        if q.kind not in ("directed", "adjacency") or not 0 <= q.source < node_count or not 0 <= q.target < node_count or q.source == q.target:
            raise ValueError("invalid local predicate")
        if q.kind == "adjacency" and q.source > q.target:
            raise ValueError("noncanonical adjacency predicate")
        if semantic in predicates or q.query_id in ids:
            raise ValueError("duplicate predicate or identifier")
        if record["answer"] not in (0, 1) or record["treatment"] != "soft_evidence":
            raise ValueError("invalid passive response")
        if record["reported_confidence"] != confidence or record["effective_confidence"] != confidence:
            raise ValueError("passive confidence differs")
        predicates.add(semantic)
        ids.add(q.query_id)
    if len({r["provenance_family"] for r in records}) != len(records):
        raise ValueError("passive provenance is not distinct")


def check_hash(item):
    if sha256(Path(item["path"])) != item["sha256"]:
        raise ValueError(f"registered input changed: {item['path']}")


def archived_inputs(root):
    """Validate saved inputs, not the historical implementation against today's code."""
    manifest_path, completion_path = root / "manifest.json", root / "completion.json"
    manifest, completion = read_json(manifest_path), read_json(completion_path)
    if completion["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("archived manifest hash differs")
    expected = json_value(settings_for(manifest.get("initialization", "pc-ges-grid")))
    expected["confidence"] = manifest["settings"]["confidence"]
    if (manifest["study"] != STUDY or manifest["settings"] != expected
            or manifest["graph_indices"] != list(SCMS)
            or manifest["sample_sizes"] != list(SIZES)
            or manifest["source_run"] != SOURCE_RUN
            or completion["trajectories"] != 2 * len(SCMS)
            or completion["paired_checkpoints"] != 6 * len(SCMS)):
        raise ValueError("archived protocol differs")
    partials = {str(r["graph_index"]): r for r in completion["partials"]}
    if (len(completion["partials"]) != len(SCMS)
            or set(partials) != {str(g) for g in SCMS}):
        raise ValueError("archived SCM slices differ")
    for item in (*manifest["sources"].values(), *manifest["claims"].values(), *partials.values()):
        check_hash(item)
    for g in SCMS:
        partial = read_json(Path(partials[str(g)]["path"]))
        if (partial["graph_index"] != g
                or partial["manifest_sha256"] != completion["manifest_sha256"]
                or partial["reference_sha256"] != manifest["sources"][str(g)]["sha256"]
                or partial["claims_sha256"] != manifest["claims"][str(g)]["sha256"]):
            raise ValueError("archived partial inputs differ")
    return manifest, dict(run_id=manifest["run_id"], partials=partials,
        initialization=manifest.get("initialization", "pc-ges-grid"),
        manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
        completion=dict(path=str(completion_path), sha256=sha256(completion_path)))


def prepare(identifier, results_root, reuse_run_id=None, initialization="pc-ges-grid"):
    root = root_for(identifier, results_root)
    settings = settings_for(initialization)
    archived, reuse = (archived_inputs(root_for(reuse_run_id, results_root))
                       if reuse_run_id else (None, None))
    if reuse and reuse["initialization"] == "pc-only" and initialization == "pc-ges-grid":
        raise ValueError("PC-only archive cannot supply the PC+GES grid")
    sources, claim_files, seeds = {}, {}, {}
    for g in SCMS:
        path = results_root / SOURCE_RUN / "partials" / f"task_{g:03d}.json"
        payload = read_json(path)
        if payload["graph_index"] != g or payload["protocol"]["node_count"] != 20:
            raise ValueError("source SCM differs")
        data = {n: np.asarray(payload["nested_data"][str(n)], dtype=float) for n in SIZES}
        if any(data[n].shape != (n, 20) or not np.isfinite(data[n]).all() for n in SIZES):
            raise ValueError("invalid observations")
        if not np.array_equal(data[200], data[1000][:200]):
            raise ValueError("observations are not nested")
        sources[str(g)] = dict(path=str(path), sha256=sha256(path))
        seed = stable_seed(STUDY, "claims", g)
        seeds[str(g)] = dict(claims=seed, samples={str(n): {
            name: stable_seed(STUDY, name, g, n) for name in ("initialization", "inference")
        } for n in SIZES})
        records = claim_sequence(_graph(payload["target"], 20), seed)
        if archived is not None:
            if (sources[str(g)]["sha256"] != archived["sources"][str(g)]["sha256"]
                    or seeds[str(g)] != archived["seeds"][str(g)]):
                raise ValueError("reused observations or seeds differ")
            saved = read_json(Path(archived["claims"][str(g)]["path"]))
            validate_claims(saved, 20, archived["settings"]["confidence"])
            reused_records = [dict(r, reported_confidence=CONFIDENCE,
                                  effective_confidence=CONFIDENCE) for r in saved]
            if reused_records != records:
                raise ValueError("reused claim sequence differs")
            records = reused_records
        validate_claims(records, 20)
        dest = root / "claims" / f"scm_{g:03d}.json"
        write_json(dest, records, exclusive=True)
        claim_files[str(g)] = dict(path=str(dest), sha256=sha256(dest))
    write_json(root / "manifest.json", dict(
        study=STUDY, version=3, run_id=identifier, source_run=SOURCE_RUN,
        graph_indices=SCMS, sample_sizes=SIZES, settings=settings, seeds=seeds,
        initialization=initialization,
        sources=sources, claims=claim_files, implementation=code_hashes(),
        trajectories=20, paired_checkpoints=60, reuse=reuse,
        baseline_settings=BASELINE_SETTINGS,
    ), exclusive=True)
    print(root, flush=True)


def verify(root):
    manifest = read_json(root / "manifest.json")
    if (manifest["settings"] != json_value(settings_for(manifest.get("initialization", "pc-ges-grid")))
            or manifest["graph_indices"] != list(SCMS)):
        raise ValueError("registered protocol differs")
    if manifest["version"] not in (2, 3) or manifest["baseline_settings"] != BASELINE_SETTINGS:
        raise ValueError("registered baseline protocol differs")
    for path, digest in manifest["implementation"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"registered implementation changed: {path}")
    for item in (*manifest["sources"].values(), *manifest["claims"].values()):
        check_hash(item)
    if manifest["reuse"]:
        reuse = manifest["reuse"]
        for item in (reuse["manifest"], reuse["completion"], *reuse["partials"].values()):
            check_hash(item)
    return manifest


def fit_grid(data, graph_index, sample_size, configurations=GRID):
    from causallearn.search.ConstraintBased.PC import pc
    fits = []
    for config in configurations:
        started = time.perf_counter()
        if config["family"] == "PC":
            raw = pc(data, alpha=config["alpha"], indep_test="fisherz", stable=True,
                     uc_rule=0, uc_priority=2, mvpc=False, verbose=False, show_progress=False)
            partial = _causal_learn_partial(raw.G, "cpdag")
        else:
            partial = learn_ges(data, lambda_value=config["penalty"])
        completion_seed = stable_seed(STUDY, "completion", graph_index, sample_size, config["id"])
        failure = None
        try:
            graphs, diagnostics = sample_completions_with_diagnostics(
                partial, "dag", SETTINGS["completion_limit"], completion_seed)
            completion_diagnostics = asdict(diagnostics)
        except CompletionError as error:
            graphs, completion_diagnostics = (), {}
            failure = str(error)
        retained = tuple(g for g in graphs if max(p.bit_count() for p in parent_masks(g)) <= 12)
        fits.append(dict(config=config, completion_seed=completion_seed,
                         endpoint_matrix=np.asarray(partial.marks).reshape(data.shape[1], data.shape[1]).tolist(),
                         graphs=[graph_edges(g) for g in retained],
                         excluded_parent_bound=len(graphs) - len(retained),
                         diagnostics=completion_diagnostics, unavailable_reason=failure,
                         seconds=time.perf_counter() - started))
    return fits


def grid_mixture(fits, node_count=20):
    families = {name: [f for f in fits if f["config"]["family"] == name and f["graphs"]]
                for name in ("PC", "GES")}
    if any(not entries for entries in families.values()):
        raise ValueError("both PC and GES must contribute valid completions")
    masses = defaultdict(float)
    for entries in families.values():
        for fit in entries:
            graphs = tuple(dict.fromkeys(_graph(r, node_count) for r in fit["graphs"]))
            for g in graphs:
                if g.regime != "dag" or not is_acyclic(g):
                    raise ValueError("grid contains an invalid DAG")
                masses[g] += 0.5 / len(entries) / len(graphs)
    if not math.isclose(sum(masses.values()), 1.0, abs_tol=1e-12):
        raise ValueError("grid mixture is not normalized")
    return dict(sorted(masses.items()))


def finite_prefixes(initial, records, budgets=BUDGETS):
    validate_claims(records, next(iter(initial)).node_count)
    graphs = tuple(initial)
    log_weights = np.log([initial[g] for g in graphs])
    snapshots = {}
    for index, record in enumerate(records, 1):
        q = ClaimQuery(**{key: record[key] for key in ("kind", "source", "target", "query_id")})
        agrees = query_values(graphs, q) == record["answer"]
        log_weights += np.where(agrees, math.log(CONFIDENCE), math.log1p(-CONFIDENCE))
        if index in budgets:
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= weights.sum()
            snapshots[index] = dict(zip(graphs, map(float, weights)))
    if set(snapshots) != set(budgets):
        raise ValueError("missing finite-support prefix")
    return snapshots


def passive_trajectory(score, particles, weights, records, seed, budgets=BUDGETS):
    validate_claims(records, score.node_count)
    queries = tuple(ClaimQuery(**{k: r[k] for k in ("kind", "source", "target", "query_id")}) for r in records)
    by_id = {q.query_id: r for q, r in zip(queries, records)}
    def answer(q):
        r = by_id[q.query_id]
        return QueryResponse(r["answer"], CONFIDENCE, treatment="soft_evidence", audit=r)
    return active_query_smc(
        None, score, answer, query_pool=queries, budgets=budgets, policy="fixed",
        confidence=CONFIDENCE, treatment="soft_evidence", seed=seed,
        initial_particles=particles, initial_weights=weights,
        ess_threshold_fraction=0.5, rejuvenation_steps=2,
        rejuvenate_after_every_query=False, guided_mixture_probability=0.5,
        proposal_cache_size=256, ordinary_kernel="uniform_pair", soft_confidence_cap=0.99,
    )


def metrics(distribution, truth):
    graphs = tuple(sorted(distribution))
    weights = np.asarray([distribution[g] for g in graphs])
    if any(g.regime != "dag" or not is_acyclic(g) for g in graphs) or not np.isfinite(weights).all() or np.any(weights <= 0) or not math.isclose(weights.sum(), 1, abs_tol=1e-10):
        raise ValueError("invalid posterior distribution")
    values = distribution_metrics(graphs, weights, truth, "dag")
    if not all(math.isfinite(values[m]) for m in METRICS):
        raise ValueError("nonfinite metric")
    return {m: values[m] for m in METRICS}


def baseline_rows(payload, sample_size):
    protocol = payload["protocol"]
    if (protocol["ges_lambda"] != BASELINE_SETTINGS["ges_penalty"]
            or protocol["notears_l1"] != BASELINE_SETTINGS["notears_l1"]
            or protocol["notears_threshold"] != BASELINE_SETTINGS["notears_threshold"]
            or protocol["pc_alphas"] != [0.01, BASELINE_SETTINGS["pc_alpha"], 0.1]):
        raise ValueError("archived baseline settings differ")
    truth = _graph(payload["target"], 20)
    rows = []
    for method in BASELINES:
        artifact = payload["samples"][str(sample_size)]["methods"][method]
        if not artifact["feasible"]:
            raise ValueError("archived baseline is unavailable")
        if method in ("PC", "GES"):
            values = point_prediction_metrics(_partial(artifact["diagnostics"]["output"]), truth)
        else:
            graphs, probabilities = _artifact_distribution(artifact, 20)
            if len(graphs) != 1:
                raise ValueError("NOTEARS must be a point DAG")
            values = metrics(dict(zip(graphs, probabilities)), truth)
        rows.append(dict(graph_index=payload["graph_index"], sample_size=sample_size,
                         budget=0, method=method, **{m: values[m] for m in METRICS}))
    return rows


def initialization_mixture(fits, initialization):
    if [f["config"] for f in fits] != list(settings_for(initialization)["grid"]):
        raise ValueError("initialization configurations differ")
    if initialization == "pc-ges-grid":
        return grid_mixture(fits)
    graphs = tuple(sorted({_graph(r, 20) for r in fits[0]["graphs"]}))
    if not graphs or any(g.regime != "dag" or not is_acyclic(g)
                         or max(p.bit_count() for p in parent_masks(g)) > 12 for g in graphs):
        raise ValueError("PC initialization has no valid DAG support")
    if len(graphs) > SETTINGS["completion_limit"]:
        raise ValueError("PC completion limit exceeded")
    return {g: 1 / len(graphs) for g in graphs}


def reused_initialization(artifact, initialization, source_initialization, seed):
    if initialization == source_initialization:
        return restored_initialization(artifact, initialization)
    if initialization != "pc-only" or source_initialization != "pc-ges-grid":
        raise ValueError("archive cannot supply requested initialization")
    fits = deepcopy([f for f in artifact["grid_fits"] if f["config"] == PC_CONFIG])
    mixture = initialization_mixture(fits, initialization)
    particles, weights, initial = _initial_particle_population(mixture, seed)
    return fits, mixture, particles, weights, initial


def restored_initialization(artifact, initialization="pc-ges-grid"):
    """Restore slots in their saved order without refitting or reallocating."""
    if not artifact["shared_initialization"] or artifact["initialization_updates"] != 0:
        raise ValueError("archived initialization differs")
    fits = deepcopy(artifact["grid_fits"])
    mixture = initialization_mixture(fits, initialization)
    if _serialize(mixture) != artifact["grid_distribution"]:
        raise ValueError("archived grid mixture differs")
    particles = tuple(_graph(r, 20) for r in artifact["initial_particles"])
    weights = np.asarray(artifact["initial_weights"], dtype=float)
    if len(particles) != 384 or weights.shape != (384,) or np.any(weights != 1 / 384):
        raise ValueError("archived particle slots differ")
    initial = defaultdict(float)
    for graph, weight in zip(particles, weights):
        if graph not in mixture:
            raise ValueError("archived particle is outside grid support")
        initial[graph] += float(weight)
    if _serialize(initial) != artifact["initial_distribution"]:
        raise ValueError("archived empirical distribution differs")
    # Preserve the registered graph order as well as particle order.
    initial = {_graph(r["graph"], 20): r["probability"] for r in artifact["initial_distribution"]}
    return fits, mixture, particles, weights, initial


def compute(identifier, graph_index, results_root):
    if graph_index not in SCMS:
        raise ValueError("SCM is outside the locked set")
    root = root_for(identifier, results_root)
    manifest = verify(root)
    initialization = manifest.get("initialization", "pc-ges-grid")
    payload = read_json(Path(manifest["sources"][str(graph_index)]["path"]))
    records = read_json(Path(manifest["claims"][str(graph_index)]["path"]))
    truth = _graph(payload["target"], 20)
    reused = (read_json(Path(manifest["reuse"]["partials"][str(graph_index)]["path"]))
              if manifest["reuse"] else None)
    artifacts, rows = {}, []
    for n in SIZES:
        data = standardize(np.asarray(payload["nested_data"][str(n)], dtype=float))
        seeds = manifest["seeds"][str(graph_index)]["samples"][str(n)]
        if reused is None:
            fits = fit_grid(data, graph_index, n, settings_for(initialization)["grid"])
            mixture = initialization_mixture(fits, initialization)
            particles, weights, initial = _initial_particle_population(mixture, seeds["initialization"])
        else:
            saved = reused["artifacts"][str(n)]
            if saved["seeds"] != seeds:
                raise ValueError("reused initialization seeds differ")
            fits, mixture, particles, weights, initial = reused_initialization(
                saved, initialization, manifest["reuse"].get("initialization", "pc-ges-grid"), seeds["initialization"])
        if initialization == "pc-only" and fits[0]["endpoint_matrix"] != payload["samples"][str(n)]["methods"]["PC"]["diagnostics"]["output"]["endpoint_matrix"]:
            raise ValueError("PC initialization differs from baseline CPDAG")
        if len(particles) != 384:
            raise ValueError("particle allocation differs")
        rows.extend(baseline_rows(payload, n))
        fs = finite_prefixes(initial, records)
        score = GaussianBICScore(data, standardize_data=False, max_parents=12)
        smc = passive_trajectory(score, particles, weights, records, seeds["inference"])
        if [s.query.query_id for s in smc.steps] != [r["query_id"] for r in records]:
            raise ValueError("SMC did not replay the fixed sequence")
        snapshots = {}
        for budget in BUDGETS:
            snapshots[str(budget)] = {}
            for method, dist in (("FS", fs[budget]), ("SMC-Passive", smc.distributions[budget])):
                values = metrics(dist, truth)
                rows.append(dict(graph_index=graph_index, sample_size=n, budget=budget, method=method, **values))
                snapshots[str(budget)][method] = dict(metrics=values, distribution=_serialize(dist))
        artifacts[str(n)] = dict(
            seeds=seeds, grid_fits=fits, grid_distribution=_serialize(mixture),
            initial_particles=[graph_edges(g) for g in particles], initial_weights=weights.tolist(),
            initial_distribution=_serialize(initial), initial_metrics=metrics(initial, truth),
            shared_initialization=True, initialization_updates=0,
            initialization_reused=bool(reused) and initialization == manifest["reuse"].get("initialization", "pc-ges-grid"),
            initialization_fits_reused=bool(reused),
            checkpoints=snapshots, trace=[asdict(step) for step in smc.steps],
            resampling_count=smc.resampling_count, timing=asdict(smc.timing),
            score=score.metadata(), cumulative_seconds=smc.cumulative_seconds,
        )
    path = root / "partials" / f"scm_{graph_index:03d}.json"
    write_json(path, dict(study=STUDY, run_id=identifier, graph_index=graph_index,
        manifest_sha256=sha256(root / "manifest.json"),
        reference_sha256=manifest["sources"][str(graph_index)]["sha256"],
        claims_sha256=manifest["claims"][str(graph_index)]["sha256"],
        artifacts=artifacts, rows=rows), exclusive=True)
    print(path, flush=True)


def aggregate(identifier, results_root):
    root = root_for(identifier, results_root)
    manifest = verify(root)
    initialization = manifest.get("initialization", "pc-ges-grid")
    rows, provenance, diagnostics = [], [], []
    for g in SCMS:
        path = root / "partials" / f"scm_{g:03d}.json"
        partial = read_json(path)
        if partial["graph_index"] != g or partial["manifest_sha256"] != sha256(root / "manifest.json"):
            raise ValueError("partial provenance differs")
        if partial["claims_sha256"] != manifest["claims"][str(g)]["sha256"] or partial["reference_sha256"] != manifest["sources"][str(g)]["sha256"]:
            raise ValueError("partial inputs differ")
        if set(partial["artifacts"]) != {str(n) for n in SIZES} or len(partial["rows"]) != 18:
            raise ValueError("incomplete SCM slice")
        records = read_json(Path(manifest["claims"][str(g)]["path"]))
        payload = read_json(Path(manifest["sources"][str(g)]["path"]))
        truth = _graph(payload["target"], 20)
        for n in SIZES:
            a = partial["artifacts"][str(n)]
            restored_initialization(a, initialization)
            for expected in baseline_rows(payload, n):
                matching = [r for r in partial["rows"] if (r["sample_size"], r["budget"], r["method"]) == (n, 0, expected["method"])]
                if matching != [expected]:
                    raise ValueError("baseline metrics differ from archived predictions")
            if manifest["reuse"]:
                saved = read_json(Path(manifest["reuse"]["partials"][str(g)]["path"]))["artifacts"][str(n)]
                fits, mixture, particles, weights, empirical = reused_initialization(
                    saved, initialization, manifest["reuse"].get("initialization", "pc-ges-grid"), a["seeds"]["initialization"])
                expected = dict(seeds=saved["seeds"], grid_fits=fits, grid_distribution=_serialize(mixture),
                                initial_particles=[graph_edges(graph) for graph in particles],
                                initial_weights=weights.tolist(), initial_distribution=_serialize(empirical),
                                initial_metrics=metrics(empirical, truth))
                if not a["initialization_fits_reused"] or any(a[key] != value for key, value in expected.items()):
                    raise ValueError("reused initialization changed")
            if initialization == "pc-only" and a["grid_fits"][0]["endpoint_matrix"] != payload["samples"][str(n)]["methods"]["PC"]["diagnostics"]["output"]["endpoint_matrix"]:
                raise ValueError("PC initialization differs from baseline CPDAG")
            if not a["shared_initialization"] or a["initialization_updates"] != 0 or len(a["initial_particles"]) != 384:
                raise ValueError("initialization differs")
            population = defaultdict(float)
            for record, weight in zip(a["initial_particles"], a["initial_weights"], strict=True):
                if weight != 1 / 384:
                    raise ValueError("initial particle weight differs")
                population[_graph(record, 20)] += weight
            initial = {_graph(r["graph"], 20): r["probability"] for r in a["initial_distribution"]}
            if dict(population) != initial:
                raise ValueError("initial particle and graph distributions differ")
            expected_fs = finite_prefixes(initial, records)
            if set(a["checkpoints"]) != {str(b) for b in BUDGETS}:
                raise ValueError("missing checkpoint")
            for b in BUDGETS:
                for method in ("FS", "SMC-Passive"):
                    saved = a["checkpoints"][str(b)][method]
                    dist = {_graph(r["graph"], 20): r["probability"] for r in saved["distribution"]}
                    values = metrics(dist, truth)
                    if method == "FS" and dist != expected_fs[b]:
                        raise ValueError("FS prefix differs from shared initialization and claims")
                    if method == "FS" and initialization == "pc-only":
                        pc = next(r for r in baseline_rows(payload, n) if r["method"] == "PC")
                        if not math.isclose(values["adjacency_f1"], pc["adjacency_f1"], rel_tol=0, abs_tol=1e-12):
                            raise ValueError("FS changed the PC skeleton")
                    matching = [r for r in partial["rows"] if (r["sample_size"], r["budget"], r["method"]) == (n, b, method)]
                    if len(matching) != 1 or any(not math.isclose(values[m], matching[0][m], abs_tol=1e-12) for m in METRICS):
                        raise ValueError("checkpoint metrics differ")
            trace = a["trace"]
            if len(trace) != 50 or [s["query"]["query_id"] for s in trace] != [r["query_id"] for r in records]:
                raise ValueError("claim trajectory differs")
            for step, record in zip(trace, records):
                if step["answer"] != record["answer"] or step["effective_confidence"] != CONFIDENCE or step["selection_rule"] != "fixed" or step["treatment"] != "soft_evidence":
                    raise ValueError("response trajectory differs")
                if step["budget"] < 50 and not step["resampled"] and step["rejuvenation_attempted_moves"]:
                    raise ValueError("unexpected intermediate mutation")
            for b in BUDGETS:
                s = trace[b-1]
                diagnostics.append(dict(graph_index=g, sample_size=n, budget=b,
                    positive_answers=sum(r["answer"] for r in records[:b]),
                    graph_mass_ess=s["graph_mass_ess_after"],
                    support_size=s["unique_positive_graphs_after"],
                    mutation_attempts=sum(x["rejuvenation_attempted_moves"] for x in trace[:b]),
                    resamplings=sum(x["resampled"] for x in trace[:b])))
        rows.extend(partial["rows"])
        provenance.append(dict(graph_index=g, path=str(path), sha256=sha256(path)))
    summary = []
    for n in SIZES:
        for b, methods in TABLE_GROUPS:
            for method in methods:
                selected = [r for r in rows if (r["sample_size"], r["budget"], r["method"]) == (n, b, method)]
                if sorted(r["graph_index"] for r in selected) != list(SCMS):
                    raise ValueError("unmatched result cell")
                record = dict(sample_size=n, budget=b, method=method, replicate_count=len(selected))
                for m in METRICS:
                    values = [r[m] for r in selected]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError("nonfinite result")
                    record[m + "_mean"] = statistics.fmean(values)
                    record[m + "_se"] = statistics.stdev(values) / math.sqrt(len(values))
                summary.append(record)
    write_csv(root / "raw_results.csv", rows)
    write_csv(root / "summary.csv", summary)
    write_csv(root / "diagnostics.csv", diagnostics)
    for filename in ("raw_results.csv", "summary.csv"):
        write_json(root / f"{filename}.metrics.json", metric_metadata(20))
    write_json(root / "completion.json", dict(trajectories=20, paired_checkpoints=60,
        baseline_evaluations=sum(r["budget"] == 0 for r in rows), summary_rows=len(summary),
        result_rows=len(rows), metric_scale="unnormalized", partials=provenance,
        manifest_sha256=sha256(root / "manifest.json")))
    print(root / "summary.csv", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "compute", "aggregate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--graph-index", type=int)
    parser.add_argument("--results-root", type=Path, default=WORKSPACE_ROOT / "results")
    parser.add_argument("--reuse-run-id", help="Preparation only: reuse archived claims and grid particles")
    parser.add_argument("--initialization", choices=INITIALIZATIONS, default="pc-ges-grid",
                        help="Preparation only: select the initial graph distribution")
    args = parser.parse_args(argv)
    identifier = run_id(args.run_id)
    if args.stage == "compute":
        compute(identifier, args.graph_index, args.results_root)
    elif args.stage == "prepare":
        prepare(identifier, args.results_root, args.reuse_run_id, args.initialization)
    else:
        aggregate(identifier, args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
