"""Render the complete-cohort native pairwise-knowledge comparison."""
import argparse
import math
from pathlib import Path

from cd_llm.experiments.common.artifacts import read_json, sha256, write_text, write_json, write_csv, run_id
from cd_llm.experiments.synthetic_pairwise_expert_benchmark import (
    root_for, verify, METHODS, METRICS, SIZES, SCMS,
)
from cd_llm.experiments.synthetic_passive_claim_budgets import check_hash
from cd_llm.paths import WORKSPACE_ROOT


def render(rows, *, original_eca=False):
    lookup = {(r["sample_size"], r["method"]): r for r in rows}
    best = {}
    for n in SIZES:
        for metric in METRICS:
            values = [r[metric + "_mean"] for r in rows
                      if r["sample_size"] == n and r["success_count"] == 10]
            best[n, metric] = (min if metric == "expected_distance" else max)(values)
    confidence_text = (r"FS/SMC cap their internal soft confidence at $0.99$, and the authors' original ECA implementation receives assertion confidence capped at $0.99$."
                       if original_eca else r"only FS/SMC cap their internal soft confidence at $0.99$.")
    caption = (r"\caption{Graph recovery on ten 20-node SCMs, mean (SE). Parentheses after method names give the number of shared pairwise facts. PC, GES, and NOTEARS use the same observations without external knowledge. Every other method receives the same twenty uniformly sampled pairwise answers (one draw per SCM), with raw confidence $1.0$; "
        + confidence_text + r" FS and SMC-Passive share 384 PC-initialized particles; SMC finishes at twenty facts, including final rejuvenation. SEs are across SCMs conditional on fixed seeds. Bold marks the best complete-cohort mean; a dash denotes an incomplete ten-SCM result. Native encodings and completion counts are in Appendix~\ref{app:synthetic-pairwise}.}")
    if original_eca:
        caption = r"\caption{Graph recovery on ten 20-node SCMs, mean (SE). All methods share observations; parentheses give the number of shared pairwise facts. Bold marks the best mean. FS/SMC use PC initialization; native knowledge encodings and the full protocol are in Appendix~\ref{app:synthetic-pairwise}.}"
    lines = [r"\begin{table}[!b]", caption,
        r"\label{tab:synthetic-main}", r"\centering", r"\begingroup", r"\small",
        r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"& \multicolumn{3}{c}{$n=200$} & \multicolumn{3}{c}{$n=1000$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Method & ED $\downarrow$ & Adj. F1 $\uparrow$ & Orient. F1 $\uparrow$ & ED $\downarrow$ & Adj. F1 $\uparrow$ & Orient. F1 $\uparrow$ \\", r"\midrule"]
    for method in METHODS:
        cells = []
        for n in SIZES:
            row = lookup[n, method]
            for metric in METRICS:
                if row["success_count"] != 10:
                    cells.append("---")
                    continue
                mean, se = row[metric + "_mean"], row[metric + "_se"]
                digits = 2 if metric == "expected_distance" else 3
                cell = f"{mean:.{digits}f} ({se:.{digits}f})"
                if math.isclose(mean, best[n, metric], rel_tol=0, abs_tol=1e-12):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        lines.append(f"{method} ({lookup[200, method]['budget']}) & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table}"))
    return "\n".join(lines) + "\n"


def merge_eca_rows(base_rows, eca_summary):
    expected = {(n, method) for n in SIZES for method in METHODS}
    if len(base_rows) != len(expected) or {(r["sample_size"], r["method"]) for r in base_rows} != expected:
        raise ValueError("base summary has missing or duplicate rows")
    selected = [r for r in eca_summary if r["confidence"] == 0.99]
    if len(selected) != 2 or {r["sample_size"] for r in selected} != set(SIZES):
        raise ValueError("missing or duplicate confidence-0.99 summary")
    replacements = {}
    for r in selected:
        if r["success_count"] != 10 or r["total_count"] != 10 or r["budget"] != 20:
            raise ValueError("ECA requires ten valid fits and twenty facts")
        for metric in METRICS:
            for suffix in ("_mean", "_se"):
                value = r[metric + suffix]
                if value is None or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid ECA aggregate")
        replacements[r["sample_size"]] = {k: v for k, v in r.items() if k != "confidence"} | {"method": "ECA"}
    return [replacements[r["sample_size"]] if r["method"] == "ECA" else dict(r) for r in base_rows]


def validate_matched_eca_input(base_manifest, graph_index, inputs, row):
    facts_item = base_manifest["facts"][str(graph_index)]
    check_hash(facts_item)
    base_facts = read_json(Path(facts_item["path"]))["records"]
    if (inputs["facts"] != base_facts or len(base_facts) != 20
            or inputs["source"] != base_manifest["sources"][str(graph_index)]
            or inputs["confidence"] != 1.0):
        raise ValueError("ECA observations or external knowledge differ from Table 1")
    if (row["raw_confidence"] != 1.0 or row["effective_confidence"] != 0.99
            or row["standardized_sha256"] != inputs["standardized_sha256"][str(row["sample_size"])]):
        raise ValueError("ECA confidence or standardized observations differ")


def archived_eca(identifier, base_manifest):
    """Verify experimental artifacts; historical paper hashes are snapshots."""
    from cd_llm.experiments import synthetic_eca_original as original
    from cd_llm.experiments import synthetic_eca_confidence_cap as capped
    root = capped.root_for(identifier)
    m, c = read_json(root / "manifest.json"), read_json(root / "completion.json")
    if (m["study"] != capped.STUDY or m["raw_confidence"] != 1.0 or m["effective_confidence"] != .99
            or m["settings"] != original.SETTINGS or m["graph_indices"] != list(SCMS)
            or m["sample_sizes"] != list(SIZES) or c["manifest_sha256"] != sha256(root / "manifest.json")):
        raise ValueError("ECA protocol differs")
    for collection in (m["implementation"], m["upstream_sources"]):
        for p, digest in collection.items():
            check_hash(dict(path=p, sha256=digest))
    paper_root = (WORKSPACE_ROOT / "paper").resolve()
    snapshots = {}
    for p, digest in m["protected"].items():
        if Path(p).resolve().is_relative_to(paper_root):
            snapshots[p] = digest
        else:
            check_hash(dict(path=p, sha256=digest))
    for name, digest in c["outputs"].items():
        check_hash(dict(path=str(root / name), sha256=digest))
    rows, keys = [], set()
    for item in c["evaluations"]:
        check_hash(item)
        r = read_json(Path(item["path"]))
        key = r["graph_index"], r["sample_size"]
        if key in keys or key not in {(g, n) for g in SCMS for n in SIZES}:
            raise ValueError("duplicate or unexpected ECA evaluation")
        keys.add(key)
        inp = m["inputs"][str(key[0])]
        check_hash(inp)
        inputs = read_json(Path(inp["path"]))
        check_hash(inputs["source"])
        validate_matched_eca_input(base_manifest, key[0], inputs, r)
        signs, _ = original.native_prior(inputs["facts"])
        if signs.tolist() != inputs["prior_signs"]:
            raise ValueError("ECA native signs differ")
        if (r["manifest_sha256"] != c["manifest_sha256"] or r["input_sha256"] != inp["sha256"]
                or r["status"] != "ok" or r["acyclicity"] > 1e-8
                or not math.isfinite(float(r["final_optimizer_objective"]))):
            raise ValueError("invalid or incomplete ECA evaluation")
        rows.append(r)
    if len(keys) != 20:
        raise ValueError("missing ECA evaluations")
    rows.sort(key=lambda r: (r["graph_index"], r["sample_size"]))
    recomputed = [dict(r, confidence=.99) for r in original.summarize(rows)]
    if recomputed != [r for r in c["summary"] if r["confidence"] == .99]:
        raise ValueError("ECA summary differs from evaluation metrics")
    diagnostics = dict(valid_fits=len(rows),
        finite_initial_evaluations=sum(math.isfinite(float(r["optimizer_calls"][0]["first_evaluation"]["objective"])) and r["optimizer_calls"][0]["first_evaluation"]["gradient_finite"] for r in rows),
        nonfinite_evaluations=sum(call["nonfinite_evaluations"] for r in rows for call in r["optimizer_calls"]),
        final_optimizer_successes=sum(r["optimizer_calls"][-1]["success"] for r in rows),
        unsuccessful_inner_calls=sum(not call["success"] for r in rows for call in r["optimizer_calls"]),
        maximum_acyclicity=max(r["acyclicity"] for r in rows))
    return c, dict(diagnostics=diagnostics,manifest=dict(path=str(root / "manifest.json"),sha256=sha256(root / "manifest.json")),
        completion=dict(path=str(root / "completion.json"),sha256=sha256(root / "completion.json")),
        historical_paper_snapshots=snapshots)


def report(identifier, *, eca_run_id=None, publication_run_id=None):
    if bool(eca_run_id) != bool(publication_run_id):
        raise ValueError("--eca-run-id and --publication-run-id must be supplied together")
    source_root = root_for(identifier)
    manifest = verify(source_root)
    completion = read_json(source_root / "completion.json")
    if completion["manifest_sha256"] != sha256(source_root / "manifest.json") or completion["result_rows"] != 160 or completion["new_evaluations"] != 100:
        raise ValueError("completion differs from registered experiment")
    for item in completion["evaluations"]:
        check_hash(item)
    for name in ("metrics", "summary"):
        if completion[name + "_sha256"] != sha256(source_root / (name + ".csv")):
            raise ValueError("aggregate changed")
    rows = completion["summary"]
    root = source_root
    provenance = None
    if eca_run_id:
        if publication_run_id in (identifier, eca_run_id):
            raise ValueError("publication requires a new timestamped directory")
        root = WORKSPACE_ROOT / "results" / run_id(publication_run_id) / "synthetic_table1_publication"
        eca_completion, eca_provenance = archived_eca(eca_run_id, manifest)
        rows = merge_eca_rows(rows, eca_completion["summary"])
        provenance = dict(base_run=identifier, eca_run=eca_run_id, eca=eca_provenance,
            base_manifest=dict(path=str(source_root / "manifest.json"),sha256=sha256(source_root / "manifest.json")),
            base_completion=dict(path=str(source_root / "completion.json"),sha256=sha256(source_root / "completion.json")))
    table = render(rows, original_eca=bool(eca_run_id))
    write_text(root / "table1.tex", table)
    lines = ["# Shared pairwise expert benchmark", "",
        "One fixed draw of 20 unordered pairs per SCM, reused at both sample sizes and by every knowledge-consuming method. Raw answers have confidence 1.0. FS/SMC use an internal 0.99 soft cap. " + ("The authors' original ECA implementation receives native assertion confidence capped at 0.99. " if eca_run_id else "") + "Counts refer to pairwise facts, not encoded directed entries.", "",
        "| Method | n | Completed | ED | Adj. F1 | Orient. F1 |", "|---|---|---|---|---|---|"]
    for row in rows:
        cells = ["—" if row[m + "_mean"] is None else f"{row[m+'_mean']:.6f} ({row[m+'_se']:.6f})" for m in METRICS]
        lines.append(f"| {row['method']} ({row['budget']}) | {row['sample_size']} | {row['success_count']}/10 | " + " | ".join(cells) + " |")
    lines.extend(["", "No averages are computed over successful subsets. SEs are conditional on the fixed pair and inference seeds. PC/GES/NOTEARS predictions are archived and unchanged.", ""])
    for failure in completion["failures"]:
        if eca_run_id and failure["method"] == "ECA":
            continue
        lines.append(f"- {failure['method']}, SCM {failure['graph_index']}, n={failure['sample_size']}: {failure['error']}")
    if eca_run_id:
        d = eca_provenance["diagnostics"]
        lines.extend(["", f"ECA has {d['valid_fits']}/20 valid fits, {d['finite_initial_evaluations']}/20 finite initial objective/gradient evaluations, {d['nonfinite_evaluations']} nonfinite evaluations, and {d['final_optimizer_successes']}/20 successful final optimizer calls. Maximum polynomial acyclicity is {d['maximum_acyclicity']:.3g}. The {d['unsuccessful_inner_calls']} unsuccessful intermediate optimizer calls remain in its archived diagnostics. Historical local-backend and confidence-1 failures remain in their original reports; they do not supply the current ECA row."])
    write_text(root / "report.md", "\n".join(lines) + "\n")
    if provenance:
        write_csv(root / "summary.csv", rows)
        write_json(root / "publication.json", dict(**provenance, summary=rows,
            outputs={name:sha256(root/name) for name in ("table1.tex","report.md","summary.csv")}))
    print(root / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eca-run-id")
    parser.add_argument("--publication-run-id")
    args = parser.parse_args()
    report(args.run_id, eca_run_id=args.eca_run_id, publication_run_id=args.publication_run_id)
