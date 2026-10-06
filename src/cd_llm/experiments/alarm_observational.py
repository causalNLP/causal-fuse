"""Frozen observational ALARM comparison with the completed BFS experiment."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import signal
import shutil
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from cd_llm.core.finite_support import (
    PartialGraph, _causal_learn_partial, graph_edges, graph_from_edges,
    learn_notears, point_prediction_metrics,
)
from cd_llm.experiments.common.artifacts import read_json, write_json, write_text, sha256, run_id
from cd_llm.methods.fast_bdeu import patched_causal_learn_bdeu
from cd_llm.paths import WORKSPACE_ROOT

SETTINGS = {
    "pc": dict(alpha=0.05, indep_test="gsq", stable=True, uc_rule=0, uc_priority=2),
    "ges": dict(equivalent_sample_size=1.0, structure_prior=1.0, maxP=None),
    "notears": dict(l1_penalty=0.05, threshold=0.30, max_outer_iterations=12),
}
TASKS = [(method, seed) for method in SETTINGS for seed in range(5)]
METRICS = ("adjacency_f1", "directed_f1", "distance", "edges", "directed_edges", "unresolved_edges", "wall_seconds")


def root(identifier: str) -> Path:
    return WORKSPACE_ROOT / "results" / run_id(identifier) / "alarm_observational"


def standardize_constants(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected a finite observation matrix")
    scale = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1.0)


def canonical_matrix(matrix: np.ndarray, order: list[int]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (len(order), len(order)) or sorted(order) != list(range(len(order))):
        raise ValueError("invalid presentation permutation")
    canonical = np.zeros_like(matrix)
    canonical[np.ix_(order, order)] = matrix
    return canonical


def edge_matrix(node_count: int, edges: list) -> np.ndarray:
    result = np.zeros((node_count, node_count), dtype=int)
    for source, target in edges:
        if source == target or result[source, target] != 0:
            raise ValueError("invalid DAG edge list")
        result[source, target], result[target, source] = 1, 2
    return result


def evaluate(matrix: np.ndarray, reference: dict) -> dict:
    estimated = PartialGraph.from_matrix(matrix, "cpdag")
    truth = graph_from_edges(len(reference["names"]), directed=reference["edges"])
    metrics = point_prediction_metrics(estimated, truth, normalized=True)
    count = int(np.count_nonzero(np.triu(matrix, 1)))
    directed = int(np.count_nonzero((matrix == 1) & (matrix.T == 2)))
    return dict(adjacency_f1=metrics["adjacency_f1"], directed_f1=metrics["orientation_f1"],
                distance=metrics["expected_distance"], edges=count,
                directed_edges=directed, unresolved_edges=count - directed)


def verify(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    for relative, expected in manifest["input_sha256"].items():
        if sha256(directory / relative) != expected:
            raise ValueError(f"frozen input changed {relative}")
    for path, expected in manifest["implementation_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"implementation changed {path}")
    if manifest["settings"] != SETTINGS:
        raise ValueError("settings changed after preparation")
    return manifest


def prepare(identifier: str, source_id: str) -> None:
    directory = root(identifier)
    if directory.exists():
        raise FileExistsError(directory)
    source = WORKSPACE_ROOT / "results" / run_id(source_id) / "alarm_bfs_openrouter"
    original_manifest = read_json(source / "manifest.json")
    manifest_hash = sha256(source / "manifest.json")
    relative_files = [f"inputs/observations_{seed}.npy" for seed in range(5)]
    relative_files += ["inputs/encoding.json", "inputs/presentation.json", "evaluation_only/reference.json"]
    frozen = {str(Path(path).resolve()): value for path, value in original_manifest["frozen_sha256"].items()}
    for relative in relative_files:
        if sha256(source / relative) != frozen[str((source / relative).resolve())]:
            raise ValueError(f"BFS input does not match its frozen manifest {relative}")
    for seed in range(5):
        data = np.load(source / f"inputs/observations_{seed}.npy", allow_pickle=False)
        if data.shape != (200, 37) or not np.isfinite(data).all():
            raise ValueError("unexpected ALARM dataset")
        prediction = read_json(source / f"predictions/{seed}.json")
        if prediction["manifest_sha256"] != manifest_hash or prediction["seed"] != seed:
            raise ValueError("BFS prediction provenance mismatch")
    for relative in relative_files:
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    for seed in range(5):
        relative = f"evaluation_only/bfs_{seed}.json"
        shutil.copyfile(source / f"predictions/{seed}.json", directory / relative)
        relative_files.append(relative)
    encoding = read_json(directory / "inputs/encoding.json")
    presentation = read_json(directory / "inputs/presentation.json")
    canonical_matrix(np.zeros((37, 37)), presentation["order"])
    if len(set(presentation["mapping"].values())) != 37 or set(presentation["mapping"]) != set(encoding["names"]):
        raise ValueError("invalid masked identifiers")
    paths = [Path(__file__), WORKSPACE_ROOT / "code/src/cd_llm/core/finite_support.py",
             WORKSPACE_ROOT / "code/src/cd_llm/methods/fast_bdeu.py"]
    write_json(directory / "manifest.json", dict(
        run_id=identifier, source_run=source_id, source_manifest_sha256=manifest_hash,
        settings=SETTINGS, tasks=TASKS, sample_size=200, node_count=37,
        input_sha256={relative: sha256(directory / relative) for relative in relative_files},
        implementation_sha256={str(path): sha256(path) for path in paths},
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "causal-learn")},
        note="PC and GES retain raw partial endpoints. NOTEARS uses standardized category codes with constant columns zeroed. No reference-based tuning.",
    ), exclusive=True)
    report(identifier)


def fit(method: str, data: np.ndarray, names: list[str]) -> tuple[np.ndarray, dict]:
    if method == "pc":
        from causallearn.search.ConstraintBased.PC import pc
        result = pc(data, **SETTINGS[method], mvpc=False, correction_name="MV_Crtn_Fisher_Z",
                    background_knowledge=None, verbose=False, show_progress=False, node_names=names)
        raw = result.G
        diagnostics = dict(raw_causal_learn_endpoints=np.asarray(raw.graph).tolist())
        partial = _causal_learn_partial(raw, "cpdag")
        return np.asarray(partial.marks).reshape(data.shape[1], -1), diagnostics
    if method == "ges":
        from causallearn.search.ScoreBased.GES import ges
        with patched_causal_learn_bdeu(data, equivalent_sample_size=1.0, structure_prior=1.0) as scorer:
            result = ges(data, score_func="local_score_BDeu", maxP=None, node_names=names)
            diagnostics = dict(score=float(result["score"]), score_cache=scorer.cache_info(),
                               raw_causal_learn_endpoints=np.asarray(result["G"].graph).tolist())
        partial = _causal_learn_partial(result["G"], "cpdag")
        return np.asarray(partial.marks).reshape(data.shape[1], -1), diagnostics
    if method == "notears":
        graph = learn_notears(standardize_constants(data), **SETTINGS[method])
        return edge_matrix(data.shape[1], graph_edges(graph)["directed"]), dict(
            constant_columns=np.flatnonzero(data.std(axis=0) == 0).tolist(),
            convergence="not exposed by the existing implementation", acyclic_threshold_pruning=True)
    raise ValueError(method)


def run(identifier: str, task: int) -> None:
    directory = root(identifier)
    method, seed = TASKS[task]
    key = f"{method}_{seed}"
    status_path = directory / "status" / f"{key}.json"
    prediction_path = directory / "predictions" / f"{key}.json"
    manifest = verify(directory)
    if prediction_path.exists():
        report(identifier)
        return
    def timed_out(signum, frame):
        raise TimeoutError("fit exceeded 570 seconds")
    signal.signal(signal.SIGALRM, timed_out)
    signal.signal(signal.SIGTERM, timed_out)
    write_json(status_path, dict(state="running", method=method, seed=seed))
    started = time.perf_counter()
    signal.alarm(570)
    try:
        presentation = read_json(directory / "inputs/presentation.json")
        names = read_json(directory / "inputs/encoding.json")["names"]
        order = presentation["order"]
        data = np.load(directory / f"inputs/observations_{seed}.npy", allow_pickle=False)[:, order]
        masked_names = [presentation["mapping"][names[i]] for i in order]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matrix, diagnostics = fit(method, data, masked_names)
        prediction = dict(method=method, seed=seed, settings=manifest["settings"][method],
                          manifest_sha256=sha256(directory / "manifest.json"),
                          canonical_endpoints=canonical_matrix(matrix, order).tolist(),
                          presentation_endpoints=matrix.tolist(), masked_names=masked_names,
                          wall_seconds=time.perf_counter() - started, diagnostics=diagnostics,
                          warnings=sorted(set(str(w.message) for w in caught)), saved_before_reference_evaluation=True)
        write_json(prediction_path, prediction)
        write_json(status_path, dict(state="complete", method=method, seed=seed))
    except BaseException as error:
        write_json(status_path, dict(state="timeout" if isinstance(error, TimeoutError) else "failed",
                                    method=method, seed=seed, error=str(error), traceback=traceback.format_exc(),
                                    wall_seconds=time.perf_counter() - started))
        raise
    finally:
        signal.alarm(0)
        report(identifier)


def summarize(records: list[dict]) -> dict:
    result = {}
    for method in [*SETTINGS, "bfs_pearson"]:
        rows = [r for r in records if r["method"] == method]
        result[method] = dict(completed=len(rows))
        for metric in METRICS:
            values = [r[metric] for r in rows]
            result[method][metric] = dict(mean=float(np.mean(values)) if values else None,
                                        se=float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)
    return result


def report(identifier: str) -> None:
    directory = root(identifier)
    with (directory / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(directory)
        reference = read_json(directory / "evaluation_only/reference.json")
        records, states = [], []
        for method, seed in TASKS:
            path = directory / "predictions" / f"{method}_{seed}.json"
            state = directory / "status" / f"{method}_{seed}.json"
            states.append(read_json(state) if state.exists() else dict(method=method, seed=seed, state="pending"))
            if path.exists():
                prediction = read_json(path)
                if prediction["manifest_sha256"] != sha256(directory / "manifest.json"):
                    raise ValueError("prediction manifest mismatch")
                metrics = evaluate(np.asarray(prediction["canonical_endpoints"]), reference)
                records.append(dict(method=method, seed=seed, wall_seconds=prediction["wall_seconds"], **metrics))
        for seed in range(5):
            prediction = read_json(directory / f"evaluation_only/bfs_{seed}.json")
            metrics = evaluate(edge_matrix(37, prediction["edges"]), reference)
            records.append(dict(method="bfs_pearson", seed=seed, wall_seconds=prediction["wall_seconds"], **metrics))
        summary = summarize(records)
        completed = sum(r["method"] != "bfs_pearson" for r in records)
        write_json(directory / "report.json", dict(completed=completed, total=15, states=states, summary=summary, records=records))
        lines = [f"ALARM observational comparison\n\n{completed}/15 fits complete. Five matched datasets, 200 observations and 37 variables each.",
                 "Values are mean (standard error) across available seeds. Partial results are explicitly counted. Lower endpoint distance is better.",
                 "| Method | Fits | Adjacency F1 | Directed F1 | Endpoint distance | Edges | Directed | Unresolved | Seconds |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        def cell(stat):
            return "pending" if stat["mean"] is None else f'{stat["mean"]:.4f} ({stat["se"]:.4f})' if stat["se"] is not None else f'{stat["mean"]:.4f}'
        for method, row in summary.items():
            lines.append("| " + " | ".join([method, f'{row["completed"]}/5', *[cell(row[m]) for m in METRICS]]) + " |")
        lines += ["", "PC uses stable G² tests at 0.05. GES uses BDeu with equivalent sample size 1, structure prior 1 and no parent cap. Both retain their raw endpoint predictions.",
                  "NOTEARS uses linear least squares on standardized category codes, penalty 0.05, threshold 0.30 and at most 12 outer iterations. Constant columns become zero. Its existing implementation prunes thresholded edges to maintain acyclicity and does not expose convergence diagnostics. Nominal category coding, particularly intubation, is arbitrary, so this is an encoded categorical linear baseline.",
                  "BFS+Pearson also receives masked biological descriptions and LLM answers. Its runtime includes API collection and cached responses, so it is not directly comparable to local fit time. Its five predictions and all input hashes were verified against the frozen source run.",
                  "", "Per seed results", "", "| Method | Seed | Adjacency F1 | Directed F1 | Endpoint distance | Edges | Directed | Unresolved | Seconds |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in records:
            lines.append("| " + " | ".join([r["method"], str(r["seed"]), *[f'{r[m]:.4f}' for m in METRICS]]) + " |")
        failed = [s for s in states if s["state"] in ("failed", "timeout")]
        if failed:
            lines += ["", "Failed fits", "", *[f'{s["method"]} seed {s["seed"]} {s["state"]} {s["error"]}' for s in failed]]
        write_text(directory / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="20260909T215919Z")
    parser.add_argument("--task", type=int, choices=range(15))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_id, args.source_run)
    elif args.command == "run":
        if args.task is None:
            parser.error("run requires --task")
        run(args.run_id, args.task)
    else:
        report(args.run_id)


if __name__ == "__main__":
    main()
