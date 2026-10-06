"""Matched numerical comparison of original ECA at confidence 1.0 and 0.99."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from cd_llm.experiments import synthetic_eca_original as original
from cd_llm.experiments.common.artifacts import read_json, write_json, write_csv, write_text, sha256, run_id

STUDY = "synthetic_eca_confidence_cap"
REFERENCE_RUN = "20260921T014251Z"
CAP = 0.99


def root_for(identifier):
    return original.WORKSPACE_ROOT / "results" / run_id(identifier) / STUDY


def prepare(identifier):
    root = root_for(identifier)
    old_root = original.root_for(REFERENCE_RUN)
    old = original.verify(old_root)
    completion = read_json(old_root / "completion.json")
    if completion["manifest_sha256"] != sha256(old_root / "manifest.json") or completion["evaluated_datasets"] != 20:
        raise ValueError("incomplete confidence-1 reference")
    for item in completion["evaluations"]:
        original.matched.archive.check_hash(item)
    protected = dict(old["protected"])
    for path in old_root.rglob("*"):
        if path.is_file():
            protected[str(path)] = sha256(path)
    implementation = dict(old["implementation"])
    implementation[str(Path(__file__).resolve())] = sha256(Path(__file__))
    write_json(root / "manifest.json", dict(study=STUDY, run_id=identifier,
        reference_run=REFERENCE_RUN, raw_confidence=1.0, effective_confidence=CAP,
        settings=original.SETTINGS, inputs=old["inputs"], implementation=implementation,
        upstream_sources=old["upstream_sources"], dependencies=old["dependencies"],
        graph_indices=original.matched.SCMS, sample_sizes=original.matched.SIZES,
        protected=protected), exclusive=True)
    print(root, flush=True)


def verify(root):
    m = read_json(root / "manifest.json")
    if (m["study"] != STUDY or m["effective_confidence"] != CAP or m["raw_confidence"] != 1.0
            or m["settings"] != original.SETTINGS or m["graph_indices"] != list(original.matched.SCMS)
            or m["sample_sizes"] != list(original.matched.SIZES)):
        raise ValueError("confidence-cap protocol changed")
    for collection in (m["implementation"], m["upstream_sources"], m["protected"]):
        for path, digest in collection.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"protected file changed: {path}")
    if original.dependencies() != m["dependencies"]:
        raise ValueError("runtime changed")
    for item in m["inputs"].values():
        original.matched.archive.check_hash(item)
        original.matched.archive.check_hash(read_json(Path(item["path"]))["source"])
    return m


def compute(identifier, graph_index):
    root = root_for(identifier)
    m = verify(root)
    if graph_index not in original.matched.SCMS:
        raise ValueError("SCM outside fixed cohort")
    item = m["inputs"][str(graph_index)]
    inputs = read_json(Path(item["path"]))
    source = read_json(Path(inputs["source"]["path"]))
    signs, raw_confidence = original.native_prior(inputs["facts"])
    if signs.tolist() != inputs["prior_signs"] or raw_confidence != inputs["confidence"]:
        raise ValueError("native knowledge differs from reference")
    effective_confidence = min(raw_confidence, CAP)
    for n in original.matched.SIZES:
        path = root / "evaluations" / f"scm_{graph_index:03d}_n{n}.json"
        if path.exists():
            saved = read_json(path)
            if saved["manifest_sha256"] != sha256(root / "manifest.json") or saved["effective_confidence"] != CAP:
                raise ValueError("existing fit has different protocol")
            print(f"reuse {graph_index} {n}: {saved['status']}", flush=True)
            continue
        data = original.standardize(np.asarray(source["nested_data"][str(n)], dtype=float))
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()
        if data_hash != inputs["standardized_sha256"][str(n)]:
            raise ValueError("observations differ from confidence-1 run")
        started = time.perf_counter()
        row = original.fit(data, signs.copy(), effective_confidence)
        row.update(graph_index=graph_index, sample_size=n, raw_confidence=raw_confidence,
            effective_confidence=effective_confidence, input_sha256=item["sha256"],
            standardized_sha256=data_hash, manifest_sha256=sha256(root / "manifest.json"),
            seconds=time.perf_counter()-started, metrics=None)
        if row["status"] == "ok":
            graph = original.matched._graph(row["graph"], 20)
            values = original.matched.archive.metrics({graph: 1.0}, original.matched._graph(source["target"], 20))
            row["metrics"] = {key: values[key] for key in original.matched.METRICS}
        write_json(path, row, exclusive=True)
        print(f"{graph_index} n={n}: {row['status']}, {row['seconds']:.2f}s, {row['error'] or ''}", flush=True)


def numerical_details(row, confidence):
    calls = row["optimizer_calls"]
    if any(call["options"] is not None for call in calls):
        raise ValueError("original optimizer options changed")
    first = calls[0]["first_evaluation"] if calls else None
    last = calls[-1] if calls else {}
    initial_finite = bool(first and math.isfinite(float(first["objective"])) and first["gradient_finite"])
    objective = row["final_optimizer_objective"]
    final_finite = objective is not None and math.isfinite(float(objective))
    raw = np.asarray(row["final_split_weights"], dtype=float)
    weights_finite = raw.shape == (800,) and bool(np.isfinite(raw).all())
    h = None
    if weights_finite:
        with np.errstate(over="ignore", invalid="ignore"):
            h = original.polynomial_h((raw[:400]-raw[400:]).reshape(20,20))
    return dict(graph_index=row["graph_index"], sample_size=row["sample_size"], confidence=confidence,
        status=row["status"], error=row["error"], initial_objective=first["objective"] if first else None,
        initial_objective_and_gradient_finite=initial_finite, final_objective_finite=final_finite,
        final_weights_finite=weights_finite, acyclicity=original.encoded(h),
        final_optimizer_success=last.get("success",False), final_optimizer_message=last.get("message"),
        optimizer_calls=len(calls), unsuccessful_inner_calls=sum(not c.get("success",False) for c in calls),
        nonfinite_evaluations=sum(c["nonfinite_evaluations"] for c in calls), seconds=row["seconds"])


def aggregate(identifier):
    root = root_for(identifier)
    m = verify(root)
    rows, old_rows, provenance, numerical = [], [], [], []
    for g in original.matched.SCMS:
        inputs = read_json(Path(m["inputs"][str(g)]["path"]))
        source = read_json(Path(inputs["source"]["path"]))
        for n in original.matched.SIZES:
            name = f"scm_{g:03d}_n{n}.json"
            path = root / "evaluations" / name
            r = read_json(path)
            if ((r["graph_index"],r["sample_size"]) != (g,n) or r["manifest_sha256"] != sha256(root / "manifest.json")
                    or r["input_sha256"] != m["inputs"][str(g)]["sha256"] or r["raw_confidence"] != 1.0
                    or r["effective_confidence"] != CAP or r["standardized_sha256"] != inputs["standardized_sha256"][str(n)]):
                raise ValueError("evaluation provenance differs")
            if r["status"] == "ok":
                recorder = original.RecordingOptimizer(None)
                recorder.last_x = np.asarray(r["final_split_weights"], dtype=float)
                recorder.last_fun = float(r["final_optimizer_objective"])
                graph,h = original.checked_prediction(np.asarray(r["returned_weights"],dtype=float),recorder,20)
                if original.graph_edges(graph) != r["graph"] or h != r["acyclicity"]:
                    raise ValueError("prediction differs")
                values = original.matched.archive.metrics({graph:1.0},original.matched._graph(source["target"],20))
                if r["metrics"] != {key:values[key] for key in original.matched.METRICS}:
                    raise ValueError("metrics differ")
            elif r["status"] != "failed" or not r["error"]:
                raise ValueError("invalid status")
            old = read_json(original.root_for(REFERENCE_RUN) / "evaluations" / name)
            if old["input_sha256"] != r["input_sha256"]:
                raise ValueError("paired knowledge differs")
            rows.append(r)
            old_rows.append(old)
            numerical.extend((numerical_details(old,1.0),numerical_details(r,CAP)))
            provenance.append(dict(path=str(path),sha256=sha256(path)))
    summary = [dict(r, confidence=c) for c, cohort in ((1.0,old_rows),(CAP,rows)) for r in original.summarize(cohort)]
    write_csv(root / "summary.csv", summary)
    write_csv(root / "numerical_comparison.csv", numerical)
    counts = []
    for c in (1.0,CAP):
        for n in original.matched.SIZES:
            selected = [r for r in numerical if r["confidence"]==c and r["sample_size"]==n]
            counts.append(dict(confidence=c,sample_size=n,valid_fits=sum(r["status"]=="ok" for r in selected),
                finite_initial_evaluations=sum(r["initial_objective_and_gradient_finite"] for r in selected),
                finite_final_objectives=sum(r["final_objective_finite"] for r in selected),
                final_optimizer_successes=sum(r["final_optimizer_success"] for r in selected),
                nonfinite_evaluations=sum(r["nonfinite_evaluations"] for r in selected),
                unsuccessful_inner_calls=sum(r["unsuccessful_inner_calls"] for r in selected)))
    lines = ["# Original ECA: confidence 1.0 versus the authorized 0.99 cap", "",
        "The same twenty datasets, twenty pairwise facts per SCM, forty signed prior entries, original source, zero initialization, solver options and runtime are used. Raw expert confidence remains 1.0; only the value passed to ECA is capped to 0.99. No other method is rerun and the paper is unchanged.", "",
        "| Confidence | n | Valid fits | Finite initial objective/gradient | Finite final objective | Final optimizer success | Nonfinite evaluations | Unsuccessful inner calls |",
        "|---|---|---|---|---|---|---|---|"]
    for r in counts:
        lines.append(f"| {r['confidence']} | {r['sample_size']} | {r['valid_fits']}/10 | {r['finite_initial_evaluations']}/10 | {r['finite_final_objectives']}/10 | {r['final_optimizer_successes']}/10 | {r['nonfinite_evaluations']} | {r['unsuccessful_inner_calls']} |")
    lines += ["", "Valid fits require finite final objective and weights, original polynomial acyclicity at most 1e-8, and a reported DAG. Final optimizer success and unsuccessful intermediate solves are recorded separately; graph validity alone does not establish numerical optimality.", "",
        "| Confidence | n | ED | Adj. F1 | Orient. F1 |", "|---|---|---|---|---|"]
    for r in summary:
        cells=["—" if r[key+"_mean"] is None else f"{r[key+'_mean']:.6f} ({r[key+'_se']:.6f})" for key in original.matched.METRICS]
        lines.append(f"| {r['confidence']} | {r['sample_size']} | "+" | ".join(cells)+" |")
    lines += ["", "Means and sample SEs require all ten valid SCMs at a sample size. No successful-subset average is computed. Every dataset's warnings and optimizer calls are saved in its evaluation record; numerical_comparison.csv gives paired diagnostics.", "", "## Capped-run fit details", ""]
    for r in numerical:
        if r["confidence"]==CAP:
            lines.append(f"- SCM {r['graph_index']}, n={r['sample_size']}: {r['status']}; h={r['acyclicity']}; final optimizer success={r['final_optimizer_success']}; unsuccessful inner calls={r['unsuccessful_inner_calls']}; nonfinite evaluations={r['nonfinite_evaluations']}. {r['error'] or ''}")
    write_text(root / "report.md", "\n".join(lines)+"\n")
    write_json(root / "completion.json",dict(manifest_sha256=sha256(root / "manifest.json"),evaluations=provenance,
        summary=summary,numerical_summary=counts,protected_files_checked=len(m["protected"]),
        outputs={name:sha256(root/name) for name in ("summary.csv","numerical_comparison.csv","report.md")}))
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
