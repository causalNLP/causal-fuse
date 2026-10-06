"""Categorical PC/GES candidate grid shared by current paper experiments."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import time
import warnings

import numpy as np

from cd_llm.core.finite_support import (
    CompletionError, _causal_learn_partial, graph_edges, graph_from_edges,
    sample_completions_with_diagnostics, stable_seed,
)
from cd_llm.experiments.common.artifacts import read_json, write_json
from cd_llm.methods.evidence_guided import initialize_score_particles
from cd_llm.methods.fast_bdeu import patched_causal_learn_bdeu
from cd_llm.methods.graph_scores import BDeuScore, parent_masks

# Preserve the historical seed namespace and candidate settings.
STUDY = "initialization_grid"
GRID = tuple([dict(id=f"pc_a{a:g}", family="pc", alpha=a)
              for a in (.001, .01, .05, .1, .2, .5)] +
             [dict(id=f"ges_e{e:g}_p{p:g}", family="ges", equivalent_sample_size=e, structure_prior=p)
              for e in (.1, 1., 10.) for p in (.5, 1., 2., 4.)])
DEFAULTS = ("pc_a0.05", "ges_e1_p1")


def condition_id(task):
    return f"{task['panel']}_s{task['data_seed']:02d}_n{task['sample_size']}"



def fit_partial(data, config):
    """Fit only observations. No reference graph enters this interface."""
    names = [f"V{i:02d}" for i in range(data.shape[1])]
    if config["family"] == "pc":
        from causallearn.search.ConstraintBased.PC import pc
        result = pc(data, alpha=config["alpha"], indep_test="gsq", stable=True,
                    uc_rule=0, uc_priority=2, mvpc=False, verbose=False,
                    show_progress=False, node_names=names)
        return _causal_learn_partial(result.G, "cpdag")
    if config["family"] == "ges":
        from causallearn.search.ScoreBased.GES import ges
        with patched_causal_learn_bdeu(data,
                equivalent_sample_size=config["equivalent_sample_size"],
                structure_prior=config["structure_prior"]):
            result = ges(data, score_func="local_score_BDeu", maxP=None, node_names=names)
        return _causal_learn_partial(result["G"], "cpdag")
    raise ValueError("unknown candidate family")



def mixture(fits, selected, count):
    """Equal family mass, then equal valid configuration and completion mass."""
    families = defaultdict(list)
    excluded = []
    for name in selected:
        fit = fits[name]
        if not fit["completions"]:
            excluded.append(name)
        else:
            families[fit["config"]["family"]].append(fit)
    if not families:
        raise ValueError("no valid candidates in this arm")
    masses = defaultdict(float)
    configuration_mass = {}
    for entries in families.values():
        config_mass = 1 / len(families) / len(entries)
        for fit in entries:
            unique = tuple(dict.fromkeys(graph_from_edges(count, directed=e) for e in fit["completions"]))
            configuration_mass[fit["config"]["id"]] = config_mass
            for graph in unique:
                masses[graph] += config_mass / len(unique)
    graphs = tuple(sorted(masses))
    probabilities = tuple(masses[g] for g in graphs)
    if not np.isclose(sum(probabilities), 1):
        raise ValueError("candidate mixture mass does not sum to one")
    return graphs, probabilities, dict(excluded_configurations=excluded,
        configuration_mass=configuration_mass, valid_families=sorted(families))



def common_score(data, condition):
    return BDeuScore(data, equivalent_sample_size=1., structure_prior=1.,
                    cardinalities=condition["cardinalities"], max_parents=12)



def candidates(directory, condition, config, digest):
    cid = condition_id(condition)
    data = np.load(directory / condition["data_path"], allow_pickle=False)
    fits = {}
    for setting in config["grid"]:
        name = setting["id"]
        output = directory / "fits" / f"{cid}_{name}.json"
        if output.exists():
            fit = read_json(output)
            if fit["manifest_sha256"] != digest or fit["config"] != setting:
                raise ValueError("cached fit provenance mismatch")
            fits[name] = fit
            continue
        write_json(directory / "status" / f"{cid}.json", dict(stage="fit", configuration=name))
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            partial = fit_partial(data, setting)
            completion_seed = stable_seed(STUDY, cid, name, "completions")
            error, diagnostics = None, None
            try:
                graphs, diagnostics = sample_completions_with_diagnostics(
                    partial, "dag", config["completion_limit"], completion_seed)
            except CompletionError as exc:
                graphs, error = (), f"{type(exc).__name__}: {exc}"
        retained = tuple(g for g in graphs if max(m.bit_count() for m in parent_masks(g)) <= config["max_parents"])
        fit = dict(config=setting, partial=asdict(partial),
            completions=[graph_edges(g)["directed"] for g in retained],
            completion_seed=completion_seed,
            completion_diagnostics=asdict(diagnostics) if diagnostics else None,
            excluded_parent_cap=len(graphs)-len(retained), completion_error=error,
            warnings=sorted(set(str(w.message) for w in caught)),
            wall_seconds=time.perf_counter()-started, manifest_sha256=digest)
        write_json(output, fit, exclusive=True)
        fits[name] = fit
        print(f"{cid} {name} candidates={len(retained)} seconds={fit['wall_seconds']:.2f}", flush=True)
    return data, fits



def initialize(directory, condition, config, digest, data, fits, arm):
    cid = condition_id(condition)
    path = directory / "initializations" / f"{cid}_{arm}.json"
    if path.exists():
        saved = read_json(path)
        if saved["manifest_sha256"] != digest:
            raise ValueError("cached initialization provenance mismatch")
        return saved
    count = data.shape[1]
    started = time.perf_counter()
    if arm == "frozen_ges":
        saved = read_json(directory / condition["initial_path"])
        output = dict(particles=saved["particles"], weights=saved["weights"],
            raw_distribution=None, starting_particles=None,
            unavailable_stages="The archived population does not retain its sampled starts or raw completions.",
            source=saved["source"], source_sha256=saved["source_sha256"])
    else:
        selected = DEFAULTS if arm == "default_union" else [c["id"] for c in config["grid"]]
        graphs, probabilities, audit = mixture(fits, selected, count)
        # Both new arms share the initialization seed; candidates alone differ.
        seed = stable_seed(STUDY, cid, "initialization")
        result = initialize_score_particles(common_score(data, condition), graphs, seed,
            particle_count=config["particle_count"], updates_per_particle=config["initialization_updates"],
            initial_graph_probabilities=probabilities, warm_start_score_mass=0., ordinary_kernel="uniform_pair")
        output = dict(particles=[graph_edges(g)["directed"] for g in result.particles],
            weights=result.weights, starting_particles=[graph_edges(g)["directed"] for g in result.starting_particles],
            raw_distribution=encode_distribution(dict(zip(graphs, probabilities))),
            mixture_audit=audit, seed=seed, accepted_moves=result.accepted_moves,
            attempted_moves=result.attempted_moves)
    output.update(arm=arm, condition=condition, manifest_sha256=digest,
                  wall_seconds=time.perf_counter()-started)
    write_json(path, output, exclusive=True)
    return output



def encode_distribution(distribution):
    rows = [dict(directed=graph_edges(g)["directed"], weight=float(w))
            for g, w in distribution.items()]
    return sorted(rows, key=lambda row: row["directed"])
