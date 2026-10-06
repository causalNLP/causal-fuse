"""Validate passive finite-support fusion with latent common causes.

The manifest freezes the protocol and implementation fingerprint. Each SCM
shard writes its model once and runs both nested observational sample sizes.
No reference graph is inserted into the candidate support.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import io
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any, Sequence

import numpy as np

import cd_llm.core.finite_support as fs
from cd_llm.experiments.common.paths import WORKSPACE_ROOT


VERSION = "latent-mag-validation-v1"
METRICS = ("expected_endpoint_distance", "endpoint_f1", "bidirected_f1")


@dataclass(frozen=True)
class Config:
    master_seed: int = 20260810
    graph_count: int = 20
    node_count: int = 10
    hidden_count: int = 2
    expected_degree: float = 2.0
    sample_sizes: tuple[int, ...] = (200, 1000)
    alpha: float = 0.05
    completion_limit: int = 200
    claims_per_group: int = 10
    confidence: float = 0.8
    response_replicates: int = 5
    interval_bootstraps: int = 2000


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_value, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fingerprint(config: Config) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True).encode())
    digest.update(VERSION.encode())
    for source in (Path(__file__), Path(fs.__file__)):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def graph_record(graph: fs.Graph) -> dict[str, Any]:
    return {"node_count": graph.node_count, **fs.graph_edges(graph)}


def graph_from_record(record: dict[str, Any]) -> fs.Graph:
    return fs.graph_from_edges(record["node_count"], record["directed"],
                               record["bidirected"], record["undirected"])


def uniform_ensemble(graphs: Sequence[fs.Graph]) -> fs.DataEnsemble:
    unique = tuple(sorted(set(graphs)))
    if not unique:
        raise ValueError("cannot construct a distribution without candidates")
    weights = np.full((1, len(unique)), 1.0 / len(unique))
    return fs.DataEnsemble("mag", unique, weights, ((len(unique),),))


def independent_evidence(
    claims: Sequence[fs.Claim], answers: Sequence[int], confidence: float,
) -> fs.EvidenceGraph:
    if len(answers) != len(claims):
        raise ValueError("one answer is required per claim")
    return fs.EvidenceGraph("latent-validation", tuple(
        fs.EvidenceNode(
            evidence_id=f"answer-{index:02d}", claim_index=index,
            stance=1 if answer else -1, reported_confidence=confidence,
            provenance_family=f"independent-{index:02d}", factor_id=f"claim-{index:02d}",
            context_id="latent-validation", context_compatible=True,
            correct=bool(answer == claim.truth), treatment="soft_evidence",
        ) for index, (claim, answer) in enumerate(zip(claims, answers))
    ))


def worked_example() -> dict[str, Any]:
    graphs = (
        fs.graph_from_edges(3, [(1, 2)], [(0, 1)]),
        fs.graph_from_edges(3, [(0, 1), (1, 2)]),
        fs.graph_from_edges(3, [(1, 0), (1, 2)]),
    )
    # Preserve the display order instead of the canonical ordering used in fits.
    ensemble = fs.DataEnsemble("mag", graphs, np.full((1, 3), 1 / 3), ((3,),))
    claims = (
        fs.Claim("partial", "III", "endpoint", 1, 0, 1,
                 (int(fs.Endpoint.CIRCLE), int(fs.Endpoint.ARROW))),
        fs.Claim("confounding", "III", "bidirected", 1, 0, 1),
    )
    values = fs.claim_value_matrix(graphs, claims)
    first = fs.preference_fusion(ensemble, values[:, :1],
                                independent_evidence(claims[:1], [1], 0.8))
    both = fs.preference_fusion(ensemble, values, independent_evidence(claims, [1, 1], 0.8))
    expected_first = np.array([4 / 9, 4 / 9, 1 / 9])
    expected_both = np.array([16 / 21, 4 / 21, 1 / 21])
    np.testing.assert_allclose(first, expected_first)
    np.testing.assert_allclose(both, expected_both)
    return {
        "graphs": [graph_record(graph) for graph in graphs],
        "claims": [asdict(claim) for claim in claims],
        "confidence": 0.8, "prior": ensemble.q_data,
        "claim_values": values, "after_endpoint_claim": first,
        "after_both_claims": both,
        "exact_fractions": [["4/9", "4/9", "1/9"], ["16/21", "4/21", "1/21"]],
    }


def model_and_observations(config: Config, graph_index: int) -> tuple[
    fs.Graph, fs.Graph, np.ndarray, np.ndarray, tuple[fs.Claim, ...], dict[str, int]
]:
    seeds = {name: fs.stable_seed(config.master_seed, VERSION, graph_index, name)
             for name in ("graph", "data", "claims")}
    target, full, coefficients = fs.generate_latent_model(
        config.node_count, config.expected_degree, seeds["graph"])
    if full.node_count != config.node_count + config.hidden_count:
        raise RuntimeError("generator hidden-variable count differs from the protocol")
    if target != fs.latent_projection(full, config.node_count):
        raise RuntimeError("generator target disagrees with its latent projection")
    if not fs.is_maximal_ancestral(target) or any(target.undirected):
        raise RuntimeError("reference is not a no-selection maximal ancestral graph")
    observations = fs.simulate_linear_sem(
        full, coefficients, max(config.sample_sizes), seeds["data"]
    )[:, :config.node_count]
    claims = tuple(claim for claim in fs.sample_resolved_claims(
        target, "mag", config.claims_per_group, seeds["claims"]
    ) if claim.group in ("I", "II", "III"))
    return target, full, coefficients, observations, claims, seeds


def bidirected_f1(estimated: fs.Graph, truth: fs.Graph) -> float:
    predicted = actual = correct = 0
    for first, second in itertools.combinations(range(truth.node_count), 2):
        estimate = fs.has_bidirected(estimated, first, second)
        target = fs.has_bidirected(truth, first, second)
        predicted += int(estimate)
        actual += int(target)
        correct += int(estimate and target)
    return 1.0 if predicted + actual == 0 else 2.0 * correct / (predicted + actual)


def metrics(graphs: Sequence[fs.Graph], probabilities: np.ndarray, target: fs.Graph) -> dict[str, float]:
    distances = np.array([fs.normalized_structural_distance(graph, target, "mag") for graph in graphs]) * (target.node_count * (target.node_count - 1))
    mode = fs.map_graph(graphs, probabilities)
    return {
        "expected_endpoint_distance": float(probabilities @ distances),
        "endpoint_f1": fs.endpoint_f1(mode, target),
        "bidirected_f1": bidirected_f1(mode, target),
    }


def load_manifest(output: Path) -> tuple[Config, dict[str, Any]]:
    manifest = read_json(output / "manifest.json")
    values = dict(manifest["config"])
    values["sample_sizes"] = tuple(values["sample_sizes"])
    config = Config(**values)
    if manifest["fingerprint"] != fingerprint(config):
        raise RuntimeError("implementation or configuration changed; initialize a fresh result directory")
    return config, manifest


def initialize(output: Path, config: Config = Config()) -> dict[str, Any]:
    path = output / "manifest.json"
    if path.exists():
        saved_config, manifest = load_manifest(output)
        if saved_config != config:
            raise RuntimeError("requested configuration differs from the existing manifest")
        return manifest
    manifest = {
        "version": VERSION, "fingerprint": fingerprint(config), "config": asdict(config),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "discovery": "one FCI fit per observed dataset, Fisher Z, no bootstrap",
            "support": "at most 200 distinct certified MAGs, no undirected edges",
            "initial_weights": "uniform over retained distinct candidates",
            "sample_sizes": "prefixes of the same observed-only 1000-row data array",
            "knowledge": "10 balanced claims per group I, II and III; independent binary channel",
            "response_matching": "claim identities and flips shared across sample sizes",
            "f1": "endpoint and bidirected F1 of the MAP graph; deterministic graph-order tie break",
            "intervals": "paired SCM bootstrap after averaging five response replicates",
            "support_failures": "reported explicitly; no reference graph insertion or replacement",
        },
    }
    atomic_json(path, manifest)
    atomic_json(output / "worked_example.json", worked_example())
    return manifest


def run_dataset(
    output: Path, config: Config, manifest: dict[str, Any], graph_index: int,
    sample_size: int, target: fs.Graph, observations: np.ndarray, claims: Sequence[fs.Claim],
) -> dict[str, Any]:
    folder = output / "scms" / f"scm_{graph_index:02d}" / f"n{sample_size}"
    result_path = folder / "result.json"
    if result_path.exists():
        saved = read_json(result_path)
        if saved["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("saved dataset has a different fingerprint")
        if saved["status"] == "complete":
            return saved
    started = time.perf_counter()
    result: dict[str, Any] = {
        "fingerprint": manifest["fingerprint"], "graph_index": graph_index,
        "sample_size": sample_size, "status": "running", "timings": {},
    }
    atomic_json(result_path, result)
    try:
        observation_path = folder / "observations.npz"
        if not observation_path.exists():
            atomic_npz(observation_path, observed=observations[:sample_size])
        pag_path = folder / "pag.json"
        if pag_path.exists():
            pag_record = read_json(pag_path)
            partial = fs.PartialGraph.from_matrix(np.asarray(pag_record["marks"]), "pag")
            result["timings"]["fci_seconds"] = pag_record["fci_seconds"]
        else:
            discovery_started = time.perf_counter()
            # This is the only information passed to observational discovery.
            with redirect_stdout(io.StringIO()):
                partial = fs.learn_fci(observations[:sample_size], alpha=config.alpha)
            result["timings"]["fci_seconds"] = time.perf_counter() - discovery_started
            atomic_json(pag_path, {
                "marks": np.asarray(partial.marks).reshape(partial.node_count, partial.node_count),
                "fci_seconds": result["timings"]["fci_seconds"],
            })
        support_path = folder / "support.json"
        if support_path.exists():
            support = read_json(support_path)
            ensemble = uniform_ensemble([graph_from_record(record) for record in support["graphs"]])
        else:
            completion_started = time.perf_counter()
            candidates, diagnostics = fs.sample_completions_with_diagnostics(
                partial, "mag", config.completion_limit,
                fs.stable_seed(config.master_seed, VERSION, graph_index, sample_size, "completion"),
                exclude_selection_bias=True,
            )
            ensemble = uniform_ensemble(candidates)
            support = {
                "graphs": [graph_record(graph) for graph in ensemble.graphs],
                "q0": ensemble.q_data, "diagnostics": asdict(diagnostics),
                "completion_seconds": time.perf_counter() - completion_started,
            }
            atomic_json(support_path, support)
        result["timings"]["completion_seconds"] = support["completion_seconds"]
        result["completion_diagnostics"] = support["diagnostics"]
        result["support_size"] = len(ensemble.graphs)
        result["truth_in_support"] = target in ensemble.graphs
        result["support_min_endpoint_distance"] = min(
            fs.normalized_structural_distance(graph, target, "mag") for graph in ensemble.graphs)
        result["data_only"] = metrics(ensemble.graphs, ensemble.q_data, target)
        fusion_started = time.perf_counter()
        values = fs.claim_value_matrix(ensemble.graphs, claims)
        result["informative_claims"] = {
            group: sum(int(np.any(values[:, index] != values[0, index]))
                       for index, claim in enumerate(claims) if claim.group == group)
            for group in ("I", "II", "III")
        }
        responses = []
        for replicate in range(config.response_replicates):
            seed = fs.stable_seed(config.master_seed, VERSION, graph_index, replicate, "answers")
            rng = np.random.default_rng(seed)
            flips = rng.random(len(claims)) >= config.confidence
            answers = np.array([claim.truth for claim in claims], dtype=np.int8) ^ flips
            evidence = independent_evidence(claims, answers, config.confidence)
            posterior = fs.preference_fusion(ensemble, values, evidence)
            responses.append({
                "replicate": replicate, "seed": seed, "answers": answers.astype(int),
                "correct_answers": int(np.count_nonzero(~flips)),
                "posterior": posterior, "claim_fuse": metrics(ensemble.graphs, posterior, target),
            })
        result["responses"] = responses
        result["claim_fuse_mean"] = {
            metric: float(np.mean([response["claim_fuse"][metric] for response in responses]))
            for metric in METRICS
        }
        result["timings"]["fusion_seconds"] = time.perf_counter() - fusion_started
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    result["timings"]["wall_seconds_this_attempt"] = time.perf_counter() - started
    atomic_json(result_path, result)
    return result


def run_shard(output: Path, graph_index: int, sample_size: int | None = None) -> list[dict[str, Any]]:
    config, manifest = load_manifest(output)
    if not 0 <= graph_index < config.graph_count:
        raise ValueError("SCM index is outside the preregistered range")
    sizes = (sample_size,) if sample_size is not None else config.sample_sizes
    if any(size not in config.sample_sizes for size in sizes):
        raise ValueError("sample size is outside the preregistered design")
    folder = output / "scms" / f"scm_{graph_index:02d}"
    model_path = folder / "model.json"
    if model_path.exists():
        model = read_json(model_path)
        if model["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("saved SCM fingerprint differs from the manifest")
        target = graph_from_record(model["target"])
        with np.load(folder / "model_arrays.npz") as arrays:
            observations = arrays["observed"].copy()
        claims = tuple(fs.Claim(**{**record, "expected_marks": (
            tuple(record["expected_marks"]) if record["expected_marks"] is not None else None
        )}) for record in model["claims"])
    else:
        try:
            target, full, coefficients, observations, claims, seeds = model_and_observations(config, graph_index)
        except Exception as error:
            failed = []
            for size in sizes:
                row = {
                    "fingerprint": manifest["fingerprint"], "graph_index": graph_index,
                    "sample_size": size, "status": "failed", "stage": "model_preparation",
                    "error": {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()},
                }
                atomic_json(folder / f"n{size}" / "result.json", row)
                failed.append(row)
            return failed
        atomic_npz(folder / "model_arrays.npz", coefficients=coefficients, observed=observations)
        atomic_json(model_path, {
            "fingerprint": manifest["fingerprint"], "graph_index": graph_index,
            "seeds": seeds, "target": graph_record(target), "full_dag": graph_record(full),
            "projection_verified": True, "hidden_indices": list(range(config.node_count, full.node_count)),
            "claims": [asdict(claim) for claim in claims],
        })
    return [run_dataset(output, config, manifest, graph_index, int(size), target, observations, claims)
            for size in sizes]


def bootstrap_paired(data: Sequence[float], fused: Sequence[float], repetitions: int, seed: int) -> dict[str, Any]:
    differences = np.asarray(fused, dtype=float) - np.asarray(data, dtype=float)
    if not len(differences):
        raise ValueError("paired intervals require at least one SCM")
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(len(differences), size=(repetitions, len(differences)))].mean(axis=1)
    return {"data_only_mean": float(np.mean(data)), "claim_fuse_mean": float(np.mean(fused)),
            "paired_difference": float(np.mean(differences)),
            "paired_95_interval": np.quantile(sampled, [0.025, 0.975]).tolist()}


def aggregate(output: Path) -> dict[str, Any]:
    config, manifest = load_manifest(output)
    summary: dict[str, Any] = {"fingerprint": manifest["fingerprint"], "panels": [],
                               "complete": True, "all_successful": True}
    for size in config.sample_sizes:
        rows, failures, missing = [], [], []
        for graph_index in range(config.graph_count):
            path = output / "scms" / f"scm_{graph_index:02d}" / f"n{size}" / "result.json"
            if not path.exists():
                missing.append(graph_index)
                continue
            row = read_json(path)
            if row["fingerprint"] != manifest["fingerprint"]:
                raise RuntimeError(f"incompatible dataset result {path}")
            if row["status"] == "complete":
                rows.append(row)
            elif row["status"] == "failed":
                failures.append({"graph_index": graph_index, "error": row["error"]})
            else:
                missing.append(graph_index)
        summary["complete"] &= not missing
        summary["all_successful"] &= not missing and not failures
        panel: dict[str, Any] = {
            "sample_size": size, "successful_scms": len(rows), "failed_scms": failures,
            "missing_scms": missing, "metrics": {},
        }
        if rows:
            panel["metrics"] = {metric: bootstrap_paired(
                [row["data_only"][metric] for row in rows],
                [row["claim_fuse_mean"][metric] for row in rows], config.interval_bootstraps,
                fs.stable_seed(config.master_seed, VERSION, size, metric, "interval"),
            ) for metric in METRICS}
            panel.update({
                "truth_in_support_count": sum(row["truth_in_support"] for row in rows),
                "mean_support_size": float(np.mean([row["support_size"] for row in rows])),
                "mean_support_min_endpoint_distance": float(np.mean([
                    row["support_min_endpoint_distance"] for row in rows])),
                "completion_diagnostics": {key: sum(row["completion_diagnostics"].get(key, 0) for row in rows)
                    for key in ("returned_count", "random_attempt_count", "unique_random_count",
                                "duplicate_random_count", "rejected_random_count")},
                "mean_informative_claims": {group: float(np.mean([
                    row["informative_claims"][group] for row in rows])) for group in ("I", "II", "III")},
                "mean_seconds": {key: float(np.mean([row["timings"][key] for row in rows]))
                                 for key in ("fci_seconds", "completion_seconds", "fusion_seconds")},
            })
        summary["panels"].append(panel)
    atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(render_report(summary))
    (output / "latent_mag_table.tex").write_text(render_table(summary))
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = ["# Finite-support latent MAG validation", "", "Paired differences are CLAIM-FUSE minus data-only. F1 uses the MAP graph. Intervals resample SCMs after averaging response replicates.", ""]
    for panel in summary["panels"]:
        lines.extend([f"## n = {panel['sample_size']}", "", f"Successful SCMs {panel['successful_scms']}, failed {len(panel['failed_scms'])}, pending {len(panel['missing_scms'])}.", "", "| Metric | Data-only | CLAIM-FUSE | Paired difference | 95% interval |", "| --- | ---: | ---: | ---: | --- |"])
        for metric, values in panel["metrics"].items():
            low, high = values["paired_95_interval"]
            lines.append(f"| {metric} | {values['data_only_mean']:.4f} | {values['claim_fuse_mean']:.4f} | {values['paired_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        if panel["metrics"]:
            lines.extend(["", f"Reference graph in support for {panel['truth_in_support_count']} SCMs. Mean support size {panel['mean_support_size']:.1f}. Mean attainable endpoint distance {panel['mean_support_min_endpoint_distance']:.4f}."])
        lines.append("")
        for failure in panel["failed_scms"]:
            lines.append(f"SCM {failure['graph_index']} failed with {failure['error']['type']}. {failure['error']['message']}")
    return "\n".join(lines) + "\n"


def render_table(summary: dict[str, Any]) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Finite-support fusion with two latent common causes and ten observed variables. CLAIM-FUSE incorporates thirty uncertain claims while both conditions use identical candidate support. ED is expected endpoint error count. Endpoint and bidirected F1 evaluate the MAP graph. Differences are CLAIM-FUSE minus data-only with paired 95\% bootstrap intervals over SCM seeds after averaging five response realizations.}", r"\label{tab:latent-mag-validation}", r"\scriptsize", r"\begin{tabular}{llrrl}", r"\toprule", r"$n$ & Metric & Data-only & CLAIM-FUSE & Difference [95\% interval] \\", r"\midrule"]
    names = {"expected_endpoint_distance": "Endpoint ED", "endpoint_f1": "Endpoint F1", "bidirected_f1": "Bidirected F1"}
    for panel in summary["panels"]:
        for metric, values in panel["metrics"].items():
            low, high = values["paired_95_interval"]
            lines.append(f"{panel['sample_size']} & {names[metric]} & {values['data_only_mean']:.4f} & {values['claim_fuse_mean']:.4f} & {values['paired_difference']:+.4f} [{low:+.4f}, {high:+.4f}] " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "worker", "aggregate"))
    locations = parser.add_mutually_exclusive_group(required=True)
    locations.add_argument("--output", type=Path)
    locations.add_argument("--run-id")
    parser.add_argument("--graph-index", "--scm-index", dest="graph_index", type=int)
    parser.add_argument("--worker", type=int, choices=range(4))
    parser.add_argument("--sample-size", type=int, choices=(200, 1000))
    args = parser.parse_args(argv)
    output = args.output
    if args.run_id is not None:
        # A run identifier names one timestamped workspace artifact directory.
        datetime.strptime(args.run_id, "%Y%m%dT%H%M%SZ")
        output = WORKSPACE_ROOT / "results" / args.run_id / "latent_mag_validation"
    if args.action == "init":
        result = initialize(output)
    elif args.action == "run":
        if args.graph_index is None:
            parser.error("run requires --graph-index")
        result = run_shard(output, args.graph_index, args.sample_size)
    elif args.action == "worker":
        if args.worker is None:
            parser.error("worker requires --worker")
        result = []
        for graph_index in range(args.worker * 5, (args.worker + 1) * 5):
            rows = run_shard(output, graph_index, args.sample_size)
            result.extend(rows)
            print(json.dumps({"graph_index": graph_index, "statuses": [row["status"] for row in rows]}), flush=True)
    else:
        result = aggregate(output)
    if isinstance(result, list):
        result = [{key: row.get(key) for key in ("graph_index", "sample_size", "status", "support_size", "data_only", "claim_fuse_mean", "error")} for row in result]
    print(json.dumps(result, default=_json_value, allow_nan=False))


if __name__ == "__main__":
    main()
