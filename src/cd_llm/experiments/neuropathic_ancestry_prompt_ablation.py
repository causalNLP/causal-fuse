"""Paired, fully cached evaluation of explicit causal criteria for ancestry."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import math
from pathlib import Path
import shutil
import socket
import time
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.local_vllm_backend import LocalVLLMConfig
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_csv, write_json, write_text
from cd_llm.experiments.neuropathic_active_interpretation import HistoricalChat, LayeredCache
from cd_llm.experiments.neuropathic_claim_cache import (
    parsed_response, provider, request_settings, sqlite_snapshot, verify_bundle,
)
from cd_llm.methods.active_query import dag_query_pool
from cd_llm.methods.llm_claim_confidence import CachedLocalChatClient, _answer, _last_json_object
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_ancestry_prompt_ablation"
SOURCE = WORKSPACE_ROOT / "results/20260917T174601Z/neuropathic_claim_group_checkpoints"
STRATEGIES = ("stated", "consistency")
PROTOCOL = "ancestry_causal_criteria_v1"


def questions():
    return dag_query_pool(20, ("ancestry",))


def prompt_templates():
    return dict(
        protocol=PROTOCOL,
        system=dict(
            stated="You are a careful medical causal expert. Do not infer from variable ordering. Answer the causal question and state the probability that your selected yes/no answer is correct. Abstain when you cannot make a defensible judgment.",
            consistency="You are a careful medical causal expert. Do not infer from variable ordering. Make one independent binary judgment and abstain if uncertain.",
        ),
        question="Assess whether {X} causally influences {Y}, directly or through a directed causal pathway, in the stated clinical context.",
        wrappers=["{question}", "Assess whether this causal relation is true: {question}",
                  "Judge this causal-path proposition: {question}",
                  "In {domain_context}, decide whether the following relation holds: {question}"],
        user="""Context: {domain_context}.
Glossary:
{X} = {description_X}
{Y} = {description_Y}

{question}

Respect the exact variable definitions, anatomical level, and laterality.
Association, shared symptoms, a common cause, or reverse causation
does not by itself establish {X} → {Y}.

Answer YES when your knowledge supports a specific causal mechanism
in the stated direction. Do not invent intermediate mechanisms.
Answer NO when your knowledge supports the absence of this relation.
Answer ABSTAIN when you cannot make either judgment defensibly.
Failure to recall a mechanism is not sufficient evidence for NO.

{response_instructions}""",
        response_instructions=dict(
            stated='''Return JSON only with these fields:
{"answer":"yes|no|abstain","confidence":0.5,"justification":"..."}

For a yes/no answer, confidence is the probability that your selected
answer is correct and must be in [0.5,1].
For abstain, set confidence to null.
Give a one-sentence justification.''',
            consistency='''Allowed labels in this order: {first_label}, {second_label}, abstain.
Return JSON only with these fields:
{"answer":"yes|no|abstain","justification":"..."}

Give a one-sentence justification.''',
        ),
        malformed_retry=" Your previous response was malformed; return JSON only.",
    )


def messages(query, strategy, metadata, templates, *, variant=0, retry=False):
    if strategy not in STRATEGIES or query.kind != "ancestry":
        raise ValueError("unsupported strategy or query")
    if variant not in range(8) or (strategy == "stated" and variant != 0):
        raise ValueError("invalid variant")
    x, y = (metadata["nodes"][i] for i in (query.source, query.target))
    values = dict(X=x["id"], Y=y["id"], description_X=x["description"],
                  description_Y=y["description"], domain_context=metadata["domain_context"])
    question = templates["question"].format(**values)
    instructions = templates["response_instructions"][strategy]
    if strategy == "consistency":
        question = templates["wrappers"][variant // 2].format(question=question, **values)
        labels = ("yes", "no") if variant % 2 == 0 else ("no", "yes")
        instructions = instructions.replace("{first_label}", labels[0]).replace("{second_label}", labels[1])
    text = templates["user"].format(question=question, response_instructions=instructions, **values)
    if retry:
        text += templates["malformed_retry"]
    return ({"role": "system", "content": templates["system"][strategy]},
            {"role": "user", "content": text})


def parse_record(record, strategy):
    choices = record.get("response", {}).get("choices") or []
    truncated = bool(choices and choices[0].get("finish_reason") == "length")
    result = dict(request_key=record["request_key"], answer=None, reported_confidence=None,
                  justification=None, status="truncated" if truncated else "malformed",
                  finish_reason=choices[0].get("finish_reason") if choices else None)
    parsed = _last_json_object(str(record["content"]))
    if parsed is not None and isinstance(parsed.get("justification"), str):
        result["justification"] = parsed["justification"]
    if truncated:
        return result
    try:
        if parsed is None:
            raise ValueError("missing JSON")
        answer = _answer(parsed["answer"])
        if answer is None:
            return dict(result, status="abstain")
        confidence = None
        if strategy == "stated":
            if isinstance(parsed["confidence"], bool):
                raise ValueError("boolean confidence")
            confidence = float(parsed["confidence"])
            if not math.isfinite(confidence) or not 0.5 <= confidence <= 1:
                raise ValueError("invalid confidence")
        return dict(result, answer=answer, reported_confidence=confidence, status="valid")
    except (KeyError, TypeError, ValueError):
        return result


def pool_votes(decisions):
    votes = [r["answer"] for r in decisions if r["status"] == "valid"]
    if len(votes) < 6:
        return None, None
    positives = sum(votes)
    if positives * 2 == len(votes):
        return 0, 0.5
    return int(positives * 2 > len(votes)), max(0.5, (max(positives, len(votes) - positives) + 0.5) / 9)


def elicit(query, strategy, chat, metadata, templates):
    records, decisions = [], []
    for index in range(2 if strategy == "stated" else 8):
        prompt = messages(query, strategy, metadata, templates,
                          variant=index if strategy == "consistency" else 0,
                          retry=strategy == "stated" and index == 1)
        record = chat.request(prompt, purpose=f"{PROTOCOL}:{strategy}:{query.query_id}:{index}", max_tokens=256)
        records.append(record)
        decisions.append(parse_record(record, strategy))
        if strategy == "stated" and decisions[-1]["status"] in ("valid", "abstain"):
            break
    if strategy == "stated":
        answer, confidence = decisions[-1]["answer"], decisions[-1]["reported_confidence"]
        reason = None if answer is not None else decisions[-1]["status"]
    else:
        answer, confidence = pool_votes(decisions)
        reason = None if answer is not None else "fewer-than-six-valid-variants"
    return dict(query=asdict(query), strategy=strategy, protocol=PROTOCOL, answer=answer,
                reported_confidence=confidence, valid=answer is not None, abstention_reason=reason,
                request_keys=[r["request_key"] for r in records], decisions=decisions,
                valid_response_count=sum(d["status"] == "valid" for d in decisions),
                prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in records),
                completion_tokens=sum(r.get("completion_tokens") or 0 for r in records))


def baseline_records(metadata, raw, exported):
    baseline = []
    experts = {s: provider(metadata, raw[s], s, minimum_valid=6)[0] for s in STRATEGIES}
    expected = {(r["strategy"], r["query"]["query_id"]): r for r in exported}
    for query in questions():
        for strategy in STRATEGIES:
            row = parsed_response(query, strategy, experts[strategy](query))
            if row != expected[strategy, query.query_id]:
                raise ValueError("historical baseline replay differs")
            baseline.append(dict(row, arm=f"historical_{strategy}"))
            if strategy == "stated":
                first = parse_record(raw[strategy][row["request_keys"][0]]["record"], strategy)
                corrected = dict(row, arm="corrected_stated")
                if first["status"] == "abstain":
                    corrected.update(answer=None, reported_confidence=None, valid=False,
                        abstention_reason="initial-explicit-abstention", request_keys=row["request_keys"][:1])
                baseline.append(corrected)
    return baseline


def truth_labels(payload):
    reachable = [[False] * 20 for _ in range(20)]
    for source, target in payload["graph"]["directed"]:
        reachable[source][target] = True
    for k in range(20):
        for i in range(20):
            for j in range(20):
                reachable[i][j] |= reachable[i][k] and reachable[k][j]
    if any(reachable[i][i] for i in range(20)):
        raise ValueError("evaluation graph is cyclic")
    return {q.query_id: int(reachable[q.source][q.target]) for q in questions()}


def category(truth, answer):
    if answer is None:
        return "positive_abstain" if truth else "negative_abstain"
    return ("TP" if answer else "FN") if truth else ("FP" if answer else "TN")


def metrics(rows, labels):
    counts = Counter(category(labels[r["query"]["query_id"]], r["answer"]) for r in rows)
    result = {key: counts[key] for key in ("TP", "FN", "FP", "TN", "positive_abstain", "negative_abstain")}
    n, valid = len(rows), sum(r["answer"] is not None for r in rows)
    positive = sum(labels[r["query"]["query_id"]] for r in rows)
    negative = n - positive
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    result.update(total=n, answered=valid, abstain=n - valid, coverage=ratio(valid, n),
        accuracy=ratio(counts["TP"] + counts["TN"], valid),
        precision=ratio(counts["TP"], counts["TP"] + counts["FP"]),
        recall=ratio(counts["TP"], positive), fpr_all_negatives=ratio(counts["FP"], negative),
        fpr_answered_negatives=ratio(counts["FP"], counts["FP"] + counts["TN"]),
        brier=ratio(sum((r["reported_confidence"] - int(r["answer"] == labels[r["query"]["query_id"]])) ** 2
                        for r in rows if r["answer"] is not None), valid))
    return result


def check_baselines(baseline, labels):
    expected = {"historical_stated": (31, 6, 56, 287, 0, 0),
                "corrected_stated": (31, 6, 55, 284, 0, 4),
                "historical_consistency": (32, 5, 42, 270, 0, 31)}
    for arm, values in expected.items():
        selected = [r for r in baseline if r["arm"] == arm]
        m = metrics(selected, labels)
        if (len(selected) != 380 or len({r["query"]["query_id"] for r in selected}) != 380
            or tuple(m[k] for k in ("TP", "FN", "FP", "TN", "positive_abstain", "negative_abstain")) != values):
            raise ValueError(f"baseline matrix differs: {arm}")


def verify(root):
    manifest = read_json(root / "manifest.json")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    if read_json(root / "questions.json") != [asdict(q) for q in questions()]:
        raise ValueError("question pool differs")
    if read_json(root / "prompts.json") != prompt_templates():
        raise ValueError("prompt templates differ")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    completion, source_manifest = read_json(SOURCE / "completion.json"), read_json(SOURCE / "manifest.json")
    if not completion["complete"] or not completion["audit_passed"] or completion["manifest_sha256"] != sha256(SOURCE / "manifest.json"):
        raise ValueError("source is not complete and audited")
    bundle = verify_bundle(SOURCE / "cache")
    inputs = {}

    def copy(source, name, expected):
        if sha256(source) != expected:
            raise ValueError(f"source input changed: {source}")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        inputs[name] = sha256(destination)
        if inputs[name] != expected:
            raise ValueError("copied source differs")

    for name in ("inputs/model_metadata.json", "inputs/runtime.json", "responses.json", "responses_min6.json",
                 "inputs/caches/qwen_stated.json", "inputs/caches/qwen_consistency.json"):
        copy(SOURCE / "cache" / name, "reference/" + name, bundle["files"][name])
    copy(SOURCE / "evaluation/truth_graph.json", "evaluation/truth_graph.json",
         source_manifest["inputs"]["evaluation/truth_graph.json"])
    metadata = read_json(root / "reference/inputs/model_metadata.json")
    raw = {s: read_json(root / f"reference/inputs/caches/qwen_{s}.json") for s in STRATEGIES}
    exported = [r for r in read_json(root / "reference/responses.json") if r["strategy"] == "stated"]
    exported += read_json(root / "reference/responses_min6.json")
    with patch.object(socket, "socket", side_effect=AssertionError("baseline attempted network")):
        baseline = baseline_records(metadata, raw, exported)
    labels = truth_labels(read_json(root / "evaluation/truth_graph.json"))
    check_baselines(baseline, labels)
    for name, value in (("baseline.json", baseline), ("questions.json", [asdict(q) for q in questions()]),
                        ("prompts.json", prompt_templates())):
        write_json(root / name, value)
        inputs[name] = sha256(root / name)
    names = ["code/src/cd_llm/experiments/neuropathic_ancestry_prompt_ablation.py",
             "code/scripts/submit_neuropathic_ancestry_prompt_ablation.py",
             "sbatch/neuropathic_ancestry_prompt_ablation.sbatch",
             "code/src/cd_llm/experiments/neuropathic_claim_cache.py",
             "code/src/cd_llm/experiments/neuropathic_active_interpretation.py",
             "code/src/cd_llm/experiments/common/artifacts.py",
             "code/src/cd_llm/methods/llm_claim_confidence.py",
             "code/src/cd_llm/methods/active_query.py",
             "code/src/cd_llm/baselines/causal_order2025/local_vllm_backend.py",
             "code/src/cd_llm/baselines/causal_order2025/openrouter_backend.py"]
    manifest = dict(study=STUDY, version=1, source=str(SOURCE),
        source_manifest_sha256=sha256(SOURCE / "manifest.json"), source_completion_sha256=sha256(SOURCE / "completion.json"),
        inputs=inputs, implementation={n: sha256(WORKSPACE_ROOT / n) for n in names},
        request_settings={s: request_settings(raw[s]) for s in STRATEGIES},
        question_count=380, expected_outcomes=760, nominal_new_requests=3420,
        consistency_minimum_valid=6, valid_abstentions_are_complete=True,
        primary_comparators=dict(stated="corrected_stated", consistency="historical_consistency"))
    write_json(root / "manifest.json", manifest)
    print(f"Prepared {root}: baseline matrices verified; 380 questions, 760 outcomes.", flush=True)
    return manifest


def config(manifest, strategy, base_url="http://127.0.0.1:18000/v1"):
    params = manifest["request_settings"][strategy]
    return LocalVLLMConfig(model=params["model"], seed=params["seed"], temperature=params["temperature"],
                           base_url=base_url, concurrency=1, pair_max_tokens=256, triplet_max_tokens=256)


def fill(root, base_url):
    manifest = verify(root)
    metadata, templates = read_json(root / "reference/inputs/model_metadata.json"), read_json(root / "prompts.json")
    with (root / "fill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = ResponseCache(root / "cache/responses.sqlite3")
        clients, outcomes = {}, []
        started = time.monotonic()
        try:
            clients = {s: CachedLocalChatClient(cache, config(manifest, s, base_url)) for s in STRATEGIES}
            for index, query in enumerate(questions(), 1):
                for strategy in STRATEGIES:
                    outcomes.append(elicit(query, strategy, clients[strategy], metadata, templates))
                if index == 1 or index % 10 == 0:
                    state = dict(stage="filling", completed_questions=index, outcomes=len(outcomes),
                                 elapsed_seconds=time.monotonic() - started, requests=cache.summary())
                    write_json(root / "status.json", state)
                    write_json(root / "responses.partial.json", outcomes)
                    print(f"{index}/380 questions; {state['requests']['request_count']} cached raw responses", flush=True)
            write_json(root / "responses.json", outcomes)
            write_json(root / "status.json", dict(stage="filled", completed_questions=380,
                outcomes=len(outcomes), elapsed_seconds=time.monotonic() - started, requests=cache.summary()))
        except BaseException as exc:
            write_json(root / "responses.partial.json", outcomes)
            write_json(root / "status.json", dict(stage="incomplete", outcomes=len(outcomes),
                error=f"{type(exc).__name__}: {exc}", requests=cache.summary()))
            raise
        finally:
            for client in clients.values():
                client.client.close()
            cache.close()


def replay(root, manifest, raw):
    metadata, templates = read_json(root / "reference/inputs/model_metadata.json"), read_json(root / "prompts.json")
    with patch.object(socket, "socket", side_effect=AssertionError("offline replay attempted network")):
        clients = {s: HistoricalChat("qwen", config(manifest, s), LayeredCache(raw, None)) for s in STRATEGIES}
        return [elicit(q, s, clients[s], metadata, templates) for q in questions() for s in STRATEGIES]


def report(root):
    manifest = verify(root)
    saved = read_json(root / "responses.json")
    raw = sqlite_snapshot(root / "cache/responses.sqlite3")
    outcomes = replay(root, manifest, raw)
    if outcomes != saved or len(outcomes) != 760:
        raise ValueError("offline replay differs or is incomplete")
    if {k for row in outcomes for k in row["request_keys"]} != set(raw):
        raise ValueError("unexpected or missing raw requests")
    baseline = read_json(root / "baseline.json")
    labels = truth_labels(read_json(root / "evaluation/truth_graph.json"))
    check_baselines(baseline, labels)
    groups = {arm: [r for r in baseline if r["arm"] == arm]
              for arm in ("historical_stated", "corrected_stated", "historical_consistency")}
    groups.update({"new_" + s: [r for r in outcomes if r["strategy"] == s] for s in STRATEGIES})
    summaries = [dict(arm=arm, **metrics(rows, labels)) for arm, rows in groups.items()]
    paired, comparisons = [], []
    for strategy in STRATEGIES:
        arm = manifest["primary_comparators"][strategy]
        old = {r["query"]["query_id"]: r for r in groups[arm]}
        historical = {r["query"]["query_id"]: r for r in groups["historical_" + strategy]}
        transitions = Counter()
        for row in groups["new_" + strategy]:
            qid = row["query"]["query_id"]
            before, truth = old[qid], labels[qid]
            previous, current = category(truth, before["answer"]), category(truth, row["answer"])
            transitions[f"{previous} -> {current}"] += 1
            paired.append(dict(strategy=strategy, query_id=qid, source=row["query"]["source"], target=row["query"]["target"],
                truth=truth, comparator=arm, baseline_answer=before["answer"], baseline_confidence=before["reported_confidence"],
                historical_answer=historical[qid]["answer"], historical_confidence=historical[qid]["reported_confidence"],
                new_answer=row["answer"], new_confidence=row["reported_confidence"], before=previous, after=current,
                justifications=[d["justification"] for d in row["decisions"]],
                response_statuses=[d["status"] for d in row["decisions"]]))
        a, b = metrics(groups[arm], labels), metrics(groups["new_" + strategy], labels)
        delta = {k: b[k] - a[k] if a[k] is not None and b[k] is not None else None for k in a}
        comparisons.append(dict(strategy=strategy, comparator=arm, transitions=dict(sorted(transitions.items())),
            delta=delta, fp_decreased=b["FP"] < a["FP"],
            eliminated_fp=sum(v for k, v in transitions.items() if k.startswith("FP -> ") and k != "FP -> FP"),
            introduced_fp=sum(v for k, v in transitions.items() if k.endswith(" -> FP") and k != "FP -> FP"),
            lost_tp=sum(v for k, v in transitions.items() if k.startswith("TP -> ") and k != "TP -> TP"),
            recovered_tp=sum(v for k, v in transitions.items() if k.endswith(" -> TP") and k != "TP -> TP")))
    statuses = {s: dict(Counter(d["status"] for r in outcomes if r["strategy"] == s for d in r["decisions"])) for s in STRATEGIES}
    summary = dict(study=STUDY, questions=380, positive_questions=sum(labels.values()),
        negative_questions=380 - sum(labels.values()), metrics=summaries, comparisons=comparisons,
        raw_response_statuses=statuses, raw_requests=len(raw),
        prompt_tokens=sum(r["prompt_tokens"] for r in outcomes), completion_tokens=sum(r["completion_tokens"] for r in outcomes),
        interpretation="Descriptive paired comparison on a fixed question pool; no downstream structural evaluation.")
    write_json(root / "cache/raw_responses.json", raw)
    write_json(root / "summary.json", summary)
    write_csv(root / "summary.csv", summaries)
    write_json(root / "per_question.json", paired)
    write_csv(root / "per_question.csv", paired)
    lines = ["# Qwen ancestry prompt comparison", "", summary["interpretation"], "",
        "All 380 ordered pairs were evaluated once per confidence method. The target graph was used only for scoring. The primary stated baseline accepts initial abstentions; the historical baseline is retained to separate the parser change from the prompt effect.", "",
        "| Arm | TP | FN | FP | TN | Abstain (+/−) | Precision | Recall | Coverage | Accuracy | Brier | FPR/all negatives | FPR/answered negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "NA" if x is None else f"{x:.4f}"
    for row in summaries:
        lines.append(f"| {row['arm']} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} | {row['positive_abstain']}/{row['negative_abstain']} | " +
            " | ".join(fmt(row[k]) for k in ("precision", "recall", "coverage", "accuracy", "brier", "fpr_all_negatives", "fpr_answered_negatives")) + " |")
    for comparison in comparisons:
        delta = comparison["delta"]
        lines += ["", f"{comparison['strategy']}: false positives {'decreased' if comparison['fp_decreased'] else 'did not decrease'} relative to {comparison['comparator']} (change {delta['FP']:+d}). Eliminated/new FP: {comparison['eliminated_fp']}/{comparison['introduced_fp']}; lost/recovered TP: {comparison['lost_tp']}/{comparison['recovered_tp']}. Recall change {delta['recall']:+.4f}; coverage change {delta['coverage']:+.4f}."]
    lines += ["", "Accuracy and Brier score use answered questions. Recall uses all 37 positives, including abstentions. Both negative-class denominators are reported. Malformed or truncated responses are tracked separately from explicit abstentions.", "",
        f"Raw responses: {len(raw)}; prompt/completion tokens: {summary['prompt_tokens']}/{summary['completion_tokens']}. All 760 outcomes reproduced with network access disabled.", "",
        f"Raw response status counts: {statuses}", "", "Exact templates are in `prompts.json`; full messages and raw outputs are in `cache/raw_responses.json`. Paired transitions and justifications are in `per_question.json`."]
    write_text(root / "report.md", "\n".join(lines) + "\n")
    audit = dict(passed=True, offline_outcomes=760, question_count=380, raw_requests=len(raw),
                 baseline_matrices_verified=True, manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "audit.json", audit)
    artifacts = ("responses.json", "cache/raw_responses.json", "summary.json", "summary.csv", "per_question.json", "per_question.csv", "report.md", "audit.json")
    write_json(root / "completion.json", dict(complete=True, audit_passed=True,
        manifest_sha256=sha256(root / "manifest.json"), artifacts={name: sha256(root / name) for name in artifacts}))
    print(f"Completed paired ancestry prompt report: {root / 'report.md'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fill", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    args = parser.parse_args(argv)
    root = WORKSPACE_ROOT / "results" / run_id(args.run_id) / STUDY
    if args.stage == "prepare":
        prepare(root)
    elif args.stage == "fill":
        fill(root, args.base_url)
    else:
        report(root)


if __name__ == "__main__":
    main()
