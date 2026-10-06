"""Report-only evaluation of the authors' unmodified ECA on Table 1 inputs."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import math
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import warnings

import numpy as np

from cd_llm.baselines.eca2025.upstream_backend import UPSTREAM_REVISION, _load_upstream_linear
from cd_llm.core.finite_support import graph_from_edges, graph_edges, standardize
from cd_llm.experiments import synthetic_pairwise_expert_benchmark as matched
from cd_llm.experiments.common.artifacts import read_json, write_json, write_csv, write_text, sha256, run_id
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "synthetic_eca_original"
REFERENCE_RUN = "20260921T010550Z"
UPSTREAM_ROOT = WORKSPACE_ROOT / "code/vendor/ECA"
SETTINGS = dict(lambda1=0.05, sigma=1.0, loss_type="pdf", max_iter=100,
                h_tol=1e-8, rho_max=1e16, w_threshold=0.30)
METHOD = "ECA (authors' original code)"


def root_for(identifier):
    return WORKSPACE_ROOT / "results" / run_id(identifier) / STUDY


def encoded(value):
    """Keep nonfinite diagnostics explicit without nonstandard JSON numbers."""
    if isinstance(value, np.ndarray):
        return encoded(value.tolist())
    if isinstance(value, dict):
        return {k: encoded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(v) for v in value]
    if isinstance(value, np.generic):
        return encoded(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def native_prior(records, node_count=20):
    # Reuse validation of the three-state facts, without an evidence adapter.
    matched.native_inputs(records, node_count)
    signs = np.zeros((node_count, node_count), dtype=float)
    for r in records:
        a, b = r["left"], r["right"]
        signs[a, b] = 1.0 if r["state"] == "left_to_right" else -1.0
        signs[b, a] = 1.0 if r["state"] == "right_to_left" else -1.0
    return signs, 1.0


def polynomial_h(weights):
    dimension = weights.shape[0]
    matrix = np.eye(dimension) + weights * weights / dimension
    return float((np.linalg.matrix_power(matrix, dimension - 1).T * matrix).sum() - dimension)


class RecordingOptimizer:
    """Observe calls; preserve the original optimizer's options and outputs."""

    def __init__(self, original):
        self.original = original
        self.calls = []
        self.last_x = None
        self.last_fun = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def minimize(self, fun, *args, **kwargs):
        record = dict(evaluations=0, nonfinite_evaluations=0, options=kwargs.get("options"),
                      first_evaluation=None, status="running")
        self.calls.append(record)

        def observed(*values, **named):
            result = fun(*values, **named)
            objective, gradient = result
            record["evaluations"] += 1
            finite = bool(np.isfinite(objective) and np.isfinite(gradient).all())
            if not finite:
                record["nonfinite_evaluations"] += 1
            if record["first_evaluation"] is None:
                record["first_evaluation"] = dict(objective=encoded(objective),
                    gradient_finite=bool(np.isfinite(gradient).all()))
            return result

        try:
            solution = self.original.minimize(observed, *args, **kwargs)
        except Exception as error:
            record.update(status="exception", error=f"{type(error).__name__}: {error}")
            raise
        self.last_x = np.array(solution.x, copy=True)
        self.last_fun = float(solution.fun)
        record.update(status="returned", success=bool(solution.success),
            optimizer_status=int(solution.status), message=str(solution.message),
            iterations=int(solution.nit), objective=encoded(solution.fun),
            weights_finite=bool(np.isfinite(solution.x).all()))
        return solution


def checked_prediction(weights, optimizer, node_count):
    if optimizer.last_x is None or optimizer.last_fun is None:
        raise ValueError("original optimizer returned no recorded solution")
    raw = optimizer.last_x
    if raw.shape != (2 * node_count * node_count,) or not np.isfinite(raw).all() or not math.isfinite(optimizer.last_fun):
        raise ValueError("nonfinite final original optimizer objective or weights")
    unthresholded = (raw[:node_count * node_count] - raw[node_count * node_count:]).reshape(node_count, node_count)
    h = polynomial_h(unthresholded)
    if not math.isfinite(h) or h > SETTINGS["h_tol"]:
        raise ValueError(f"original polynomial acyclicity {h} exceeds {SETTINGS['h_tol']}")
    weights = np.asarray(weights)
    if weights.shape != (node_count, node_count) or not np.isfinite(weights).all() or np.any(np.diag(weights)):
        raise ValueError("invalid returned original weights")
    # Match the authors' learn() reporting convention after native thresholding.
    edges = list(zip(*np.nonzero(np.abs(weights) > SETTINGS["w_threshold"])))
    graph = graph_from_edges(node_count, directed=[(int(a), int(b)) for a, b in edges])
    if not matched.archive.is_acyclic(graph):
        raise ValueError("original reported graph is cyclic")
    return graph, h


def source_hashes():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(UPSTREAM_ROOT), *args])
    if git("rev-parse", "HEAD").decode().strip() != UPSTREAM_REVISION:
        raise ValueError("wrong original ECA revision")
    paths = git("ls-files", "-z", "--", "*.py").decode().split("\0")
    result = {}
    for relative in filter(None, paths):
        path = UPSTREAM_ROOT / relative
        expected = hashlib.sha256(git("show", f"{UPSTREAM_REVISION}:{relative}")).hexdigest()
        if sha256(path) != expected:
            raise ValueError(f"authors' Python source modified: {relative}")
        result[str(path)] = expected
    return result


def dependencies():
    return {name: metadata.version(name) for name in ("numpy", "scipy", "torch", "gcastle")}


def prepare(identifier):
    root = root_for(identifier)
    reference_root = matched.root_for(REFERENCE_RUN)
    reference = matched.verify(reference_root)
    completion = read_json(reference_root / "completion.json")
    if completion["manifest_sha256"] != sha256(reference_root / "manifest.json"):
        raise ValueError("reference completion differs")
    for item in completion["evaluations"]:
        matched.archive.check_hash(item)
    inputs = {}
    for g in matched.SCMS:
        facts = read_json(Path(reference["facts"][str(g)]["path"]))
        signs, confidence = native_prior(facts["records"])
        if len(facts["records"]) != 20 or np.count_nonzero(signs) != 40:
            raise ValueError("wrong knowledge budget")
        source = read_json(Path(reference["sources"][str(g)]["path"]))
        data = {n: np.asarray(source["nested_data"][str(n)], dtype=float) for n in matched.SIZES}
        if not np.array_equal(data[200], data[1000][:200]):
            raise ValueError("reference data no longer nested")
        path = root / "inputs" / f"scm_{g:03d}.json"
        write_json(path, dict(facts=facts["records"], prior_signs=signs.tolist(), confidence=confidence,
            source=reference["sources"][str(g)],
            standardized_sha256={str(n): hashlib.sha256(standardize(data[n]).tobytes()).hexdigest() for n in matched.SIZES}), exclusive=True)
        inputs[str(g)] = dict(path=str(path), sha256=sha256(path))
    protected = read_json(WORKSPACE_ROOT / "out" / REFERENCE_RUN / "preserved_hashes.json")
    protected = {str(WORKSPACE_ROOT / p): digest for p, digest in protected.items()}
    for path in reference_root.rglob("*"):
        if path.is_file():
            protected[str(path)] = sha256(path)
    for path in (WORKSPACE_ROOT / "paper").rglob("*"):
        if path.is_file() and path.suffix in (".tex", ".pdf", ".bib", ".sty"):
            protected[str(path)] = sha256(path)
    implementation = matched.implementation_hashes()
    implementation[str(Path(__file__).resolve())] = sha256(Path(__file__))
    write_json(root / "manifest.json", dict(study=STUDY, version=1, run_id=identifier,
        settings=SETTINGS, confidence=1.0, reference_run=REFERENCE_RUN,
        graph_indices=matched.SCMS, sample_sizes=matched.SIZES, inputs=inputs,
        upstream_revision=UPSTREAM_REVISION, upstream_sources=source_hashes(),
        implementation=implementation, dependencies=dependencies(), protected=protected,
        reference_summary=completion["summary"]), exclusive=True)
    print(root, flush=True)


def verify(root):
    m = read_json(root / "manifest.json")
    if m["study"] != STUDY or m["settings"] != SETTINGS or m["confidence"] != 1.0 or m["graph_indices"] != list(matched.SCMS) or m["sample_sizes"] != list(matched.SIZES):
        raise ValueError("registered original-code protocol changed")
    for collection in (m["implementation"], m["upstream_sources"], m["protected"]):
        for path, digest in collection.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"protected input or implementation changed: {path}")
    for item in m["inputs"].values():
        matched.archive.check_hash(item)
        matched.archive.check_hash(read_json(Path(item["path"]))["source"])
    if m["dependencies"] != dependencies():
        raise ValueError("dependency versions changed")
    return m


def fit(data, signs, confidence):
    module = _load_upstream_linear(UPSTREAM_ROOT)
    recorder = RecordingOptimizer(module.sopt)
    module.sopt = recorder  # Isolated module binding; SciPy itself is untouched.
    result = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = module.myECA_prior(**SETTINGS)
            model.load_prior(signs, confidence)
            returned = model.notears_linear(data, **SETTINGS)
            result["returned_weights"] = encoded(returned)
            graph, h = checked_prediction(returned, recorder, data.shape[1])
            result.update(status="ok", graph=graph_edges(graph), acyclicity=h, error=None)
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result.update(optimizer_calls=recorder.calls, final_split_weights=encoded(recorder.last_x),
        final_optimizer_objective=encoded(recorder.last_fun),
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    return result


def compute(identifier, graph_index):
    root = root_for(identifier)
    m = verify(root)
    if graph_index not in matched.SCMS:
        raise ValueError("SCM outside fixed cohort")
    item = m["inputs"][str(graph_index)]
    inputs = read_json(Path(item["path"]))
    source = read_json(Path(inputs["source"]["path"]))
    signs, confidence = native_prior(inputs["facts"])
    if signs.tolist() != inputs["prior_signs"] or confidence != inputs["confidence"]:
        raise ValueError("frozen native prior differs")
    manifest_hash = sha256(root / "manifest.json")
    for n in matched.SIZES:
        path = root / "evaluations" / f"scm_{graph_index:03d}_n{n}.json"
        if path.exists():
            previous = read_json(path)
            if previous["manifest_sha256"] != manifest_hash or previous["input_sha256"] != item["sha256"]:
                raise ValueError("existing evaluation provenance differs")
            print(f"reuse {graph_index} {n}: {previous['status']}", flush=True)
            continue
        data = standardize(np.asarray(source["nested_data"][str(n)], dtype=float))
        if hashlib.sha256(data.tobytes()).hexdigest() != inputs["standardized_sha256"][str(n)]:
            raise ValueError("standardized observations differ")
        started = time.perf_counter()
        row = fit(data, signs.copy(), confidence)
        row.update(graph_index=graph_index, sample_size=n, method=METHOD, budget=20,
            manifest_sha256=manifest_hash, input_sha256=item["sha256"], seconds=time.perf_counter()-started,
            metrics=None)
        if row["status"] == "ok":
            graph = matched._graph(row["graph"], 20)
            values = matched.archive.metrics({graph: 1.0}, matched._graph(source["target"], 20))
            row["metrics"] = {key: values[key] for key in matched.METRICS}
        write_json(path, row, exclusive=True)
        print(f"{graph_index} n={n}: {row['status']}, {row['seconds']:.2f}s, {row['error'] or ''}", flush=True)


def summarize(rows):
    keys = [(r["graph_index"], r["sample_size"]) for r in rows]
    expected = {(g, n) for g in matched.SCMS for n in matched.SIZES}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing or duplicate original-code evaluations")
    summary = []
    for n in matched.SIZES:
        selected = [r for r in rows if r["sample_size"] == n]
        valid = [r for r in selected if r["status"] == "ok"]
        row = dict(method=METHOD, sample_size=n, budget=20, success_count=len(valid), total_count=10)
        for metric in matched.METRICS:
            values = [r["metrics"][metric] for r in valid]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite successful metric")
            row[metric+"_mean"] = statistics.fmean(values) if len(valid)==10 else None
            row[metric+"_se"] = statistics.stdev(values)/math.sqrt(10) if len(valid)==10 else None
        summary.append(row)
    return summary


def aggregate(identifier):
    root = root_for(identifier)
    m = verify(root)
    rows, provenance = [], []
    for g in matched.SCMS:
        source = read_json(Path(read_json(Path(m["inputs"][str(g)]["path"]))["source"]["path"]))
        for n in matched.SIZES:
            path = root / "evaluations" / f"scm_{g:03d}_n{n}.json"
            row = read_json(path)
            if row["manifest_sha256"] != sha256(root / "manifest.json") or row["input_sha256"] != m["inputs"][str(g)]["sha256"] or (row["graph_index"], row["sample_size"]) != (g,n):
                raise ValueError("evaluation provenance differs")
            if row["status"] == "ok":
                recorder = RecordingOptimizer(None)
                recorder.last_x = np.asarray(row["final_split_weights"],dtype=float)
                recorder.last_fun = float(row["final_optimizer_objective"])
                graph, h = checked_prediction(np.asarray(row["returned_weights"],dtype=float),recorder,20)
                if graph_edges(graph) != row["graph"] or h != row["acyclicity"]:
                    raise ValueError("saved original prediction differs")
                values = matched.archive.metrics({graph:1.0},matched._graph(source["target"],20))
                if row["metrics"] != {key:values[key] for key in matched.METRICS}:
                    raise ValueError("saved original metrics differ")
            elif row["status"] != "failed" or not row["error"]:
                raise ValueError("invalid evaluation status")
            rows.append(row)
            provenance.append(dict(path=str(path),sha256=sha256(path)))
    summary = summarize(rows)
    reference = [dict(r, method="ECA (local paper backend)") if r["method"]=="ECA" else r for r in m["reference_summary"]]
    write_csv(root / "summary.csv", summary)
    write_csv(root / "comparison.csv", reference+summary)
    write_csv(root / "diagnostics.csv", [dict(graph_index=r["graph_index"], sample_size=r["sample_size"],
        status=r["status"], error=r["error"], seconds=r["seconds"], optimizer_calls=len(r["optimizer_calls"]),
        nonfinite_evaluations=sum(c["nonfinite_evaluations"] for c in r["optimizer_calls"]),
        final_objective=r["final_optimizer_objective"], acyclicity=r.get("acyclicity")) for r in rows])
    lines = ["# Authors' original ECA on matched Table 1 inputs", "",
        "All twenty datasets and all twenty pairwise facts per SCM are reused exactly. Forty signed entries encode those facts with confidence 1.0. The authors' Python source, zero initialization, prior multiplier, gradient damping, polynomial acyclicity and inner solver defaults are unchanged. The paper and its existing results are unchanged.", "",
        f"Original revision: `{UPSTREAM_REVISION}`. Runtime: `{m['dependencies']}`. Settings: `{SETTINGS}`.", "",
        "| Method | n | Valid fits | ED | Adj. F1 | Orient. F1 |", "|---|---|---|---|---|---|"]
    for r in reference+summary:
        cells = ["—" if r[key+"_mean"] is None else f"{r[key+'_mean']:.6f} ({r[key+'_se']:.6f})" for key in matched.METRICS]
        lines.append(f"| {r['method']} ({r['budget']}) | {r['sample_size']} | {r['success_count']}/10 | " + " | ".join(cells)+" |")
    lines += ["", "Means and sample SEs require ten valid fits at a sample size. No successful-subset means are reported. Optimizer termination is recorded separately from finite-output and acyclicity validity. Nonfinite optimization is never accepted solely because the returned graph is empty.", "", "## Original-code evaluation details", ""]
    for r in rows:
        calls = r["optimizer_calls"]
        first = calls[0]["first_evaluation"] if calls else None
        last = calls[-1].get("message") if calls else None
        lines.append(f"- SCM {r['graph_index']}, n={r['sample_size']}: **{r['status']}**. {r['error'] or ''} Initial evaluation: `{first}`. Final optimizer message: `{last}`.")
    write_text(root / "report.md", "\n".join(lines)+"\n")
    write_json(root / "completion.json",dict(manifest_sha256=sha256(root / "manifest.json"),
        evaluations=provenance, summary=summary, evaluated_datasets=len(rows),
        protected_files_checked=len(m["protected"]), outputs={name:sha256(root/name) for name in ("summary.csv","comparison.csv","diagnostics.csv","report.md")}))
    print(root / "report.md",flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id",required=True)
    p.add_argument("--stage",choices=("prepare","compute","aggregate"),required=True)
    p.add_argument("--graph-index",type=int)
    a=p.parse_args(argv)
    if a.stage=="compute":
        compute(a.run_id,a.graph_index)
    else:
        globals()[a.stage](a.run_id)


if __name__=="__main__":
    main()
