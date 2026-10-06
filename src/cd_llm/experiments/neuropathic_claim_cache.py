"""Complete and export the fixed neuropathic Group I--II Qwen response cache."""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import socket
import sqlite3
import tarfile
import tempfile
import time
from unittest.mock import patch

from cd_llm.baselines.causal_order2025.local_vllm_backend import LocalVLLMConfig
from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache, _request_key
from cd_llm.experiments.common.artifacts import read_json, run_id, sha256, write_json, write_text
from cd_llm.experiments.neuropathic_active_interpretation import HistoricalChat, LayeredCache
from cd_llm.methods.active_query import ClaimQuery, dag_query_pool
from cd_llm.methods.llm_claim_confidence import CachedLocalChatClient, QwenClaimConfidenceProvider
from cd_llm.paths import WORKSPACE_ROOT

STUDY = "neuropathic_claim_cache"
SOURCE = WORKSPACE_ROOT / "results/20260917T023943Z/neuropathic_valid_answer_budget"
STRATEGIES = ("stated", "consistency")


def questions():
    return (*dag_query_pool(20, ("local_edge",)), *dag_query_pool(20, ("ancestry",)))


def sqlite_snapshot(path):
    """Read a consistent SQLite snapshot, including committed WAL records."""
    if not path.exists():
        return {}
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        rows = connection.execute("SELECT * FROM responses ORDER BY request_key").fetchall()
    return {r["request_key"]: dict(request=json.loads(r["request_json"]), purpose=r["purpose"],
        record=dict(request_key=r["request_key"], content=r["content"], response=json.loads(r["response_json"]),
            provider_request_id=r["provider_request_id"], response_model=r["response_model"],
            prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
            total_tokens=r["total_tokens"], cost_usd=r["cost_usd"], latency_seconds=r["latency_seconds"],
            attempts=r["attempts"], cached=True)) for r in rows}


def merge_records(*sources):
    merged = {}
    for source in sources:
        for key, value in source.items():
            if key != _request_key(value["request"]) or value["record"]["request_key"] != key:
                raise ValueError("raw response request key differs")
            if key in merged:
                previous = merged[key]
                if (previous["request"] != value["request"] or previous["purpose"] != value["purpose"]
                    or previous["record"]["content"] != value["record"]["content"]
                    or previous["record"]["response"] != value["record"]["response"]):
                    raise ValueError(f"conflicting cached response: {key}")
            else:
                merged[key] = value
    return merged


def request_settings(records):
    settings = [{k: v for k, v in entry["request"].items() if k != "messages"}
                for entry in records.values()]
    if not settings or any(value != settings[0] for value in settings):
        raise ValueError("inconsistent or absent request settings")
    params = settings[0]
    if (params["model"] != "Qwen/Qwen3.5-9B" or params["provider"] != "local-vllm"
        or params["temperature"] != 0 or params["thinking"] is not False or params["max_tokens"] != 256):
        raise ValueError("request settings differ from the source protocol")
    return params


def provider(metadata, records, strategy, *, incremental=None, base_url=None, minimum_valid=5):
    params = request_settings(records)
    config = LocalVLLMConfig(model=params["model"], seed=params["seed"], temperature=params["temperature"],
        base_url=base_url or "http://127.0.0.1:18000/v1", concurrency=1,
        triplet_max_tokens=256, pair_max_tokens=256)
    cache = LayeredCache(records, incremental)
    delegate = CachedLocalChatClient(cache, config) if base_url is not None else None
    expert = QwenClaimConfidenceProvider(strategy, [n["id"] for n in metadata["nodes"]],
        HistoricalChat("qwen", config, cache, delegate),
        descriptions={n["id"]: n["description"] for n in metadata["nodes"]},
        domain_context=metadata["domain_context"], protocol="legacy_v1",
        minimum_valid_consistency_responses=minimum_valid)
    return expert, delegate


def parsed_response(query, strategy, response):
    return dict(query=asdict(query), strategy=strategy, answer=response.answer,
        reported_confidence=response.reported_confidence, valid=response.valid,
        treatment=response.treatment, abstention_reason=response.abstention_reason,
        votes=response.audit.get("votes"), valid_response_count=response.audit.get("valid_response_count"),
        minimum_valid_response_count=response.audit.get("minimum_valid_response_count"),
        request_keys=response.audit["request_keys"], prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens)


def verify(root):
    manifest = read_json(root / "manifest.json")
    if read_json(root / "questions.json") != [asdict(q) for q in questions()]:
        raise ValueError("question pool changed")
    for name, digest in manifest["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    for name, digest in manifest["implementation"].items():
        if sha256(WORKSPACE_ROOT / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    return manifest


def prepare(root):
    if (root / "manifest.json").exists():
        return verify(root)
    completion = read_json(SOURCE / "completion.json")
    if not completion["complete"] or not completion["audit_passed"]:
        raise ValueError("source experiment is not complete and audited")
    source_manifest = read_json(SOURCE / "manifest.json")
    if completion["manifest_sha256"] != sha256(SOURCE / "manifest.json"):
        raise ValueError("source manifest differs from completed experiment")
    inputs = {}
    for name in ("model_metadata.json", "runtime.json"):
        relative = f"inputs/{name}"
        if sha256(SOURCE / relative) != source_manifest["inputs"][relative]:
            raise ValueError("source metadata changed")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, destination)
        inputs[relative] = sha256(destination)
    write_json(root / "questions.json", [asdict(q) for q in questions()])
    inputs["questions.json"] = sha256(root / "questions.json")
    additions = sqlite_snapshot(SOURCE / "caches/qwen_incremental.sqlite3")
    metadata = read_json(root / "inputs/model_metadata.json")
    settings, counts = {}, {}
    for strategy in STRATEGIES:
        relative = f"inputs/caches/qwen_{strategy}.json"
        if sha256(SOURCE / relative) != source_manifest["inputs"][relative]:
            raise ValueError("source historical cache changed")
        merged = merge_records(read_json(SOURCE / relative),
            {k: v for k, v in additions.items() if v["purpose"].startswith(strategy + ":")})
        write_json(root / relative, merged)
        inputs[relative] = sha256(root / relative)
        settings[strategy] = request_settings(merged)
        expert, _ = provider(metadata, merged, strategy)
        available = 0
        for query in questions():
            try:
                expert(query)
                available += 1
            except RuntimeError as exc:
                if not str(exc).startswith("uncached historical prompt"):
                    raise
        counts[strategy] = dict(raw_requests=len(merged), complete_questions=available,
                                remaining_questions=950 - available)
    names = ("experiments/neuropathic_claim_cache.py", "experiments/neuropathic_active_interpretation.py",
             "experiments/common/artifacts.py", "methods/llm_claim_confidence.py", "methods/active_query.py",
             "baselines/causal_order2025/local_vllm_backend.py", "baselines/causal_order2025/openrouter_backend.py")
    implementation = {"code/src/cd_llm/" + n: sha256(WORKSPACE_ROOT / "code/src/cd_llm" / n) for n in names}
    manifest = dict(study=STUDY, version=1, inputs=inputs, implementation=implementation,
        source=str(SOURCE), source_manifest_sha256=sha256(SOURCE / "manifest.json"),
        source_completion_sha256=sha256(SOURCE / "completion.json"),
        request_settings=settings, imported=counts, question_count=950,
        group_counts={"I": 570, "II": 380}, strategies=list(STRATEGIES),
        consistency_minimum_valid=5, confidence_rule="max(0.5, (majority_count + 0.5)/9); ties=0.5",
        valid_abstentions_are_complete=True)
    write_json(root / "manifest.json", manifest)
    print(json.dumps(dict(prepared=str(root), imported=counts)), flush=True)
    return manifest


def raw_records(root):
    additions = sqlite_snapshot(root / "caches/qwen_incremental.sqlite3")
    return {s: merge_records(read_json(root / f"inputs/caches/qwen_{s}.json"),
                            {k: v for k, v in additions.items() if v["purpose"].startswith(s + ":")})
            for s in STRATEGIES}


def fill(root, base_url):
    manifest = verify(root)
    started = time.monotonic()
    with (root / "fill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        incremental = ResponseCache(root / "caches/qwen_incremental.sqlite3")
        experts, delegates, outcomes = {}, [], []
        try:
            metadata = read_json(root / "inputs/model_metadata.json")
            for strategy in STRATEGIES:
                expert, delegate = provider(metadata, read_json(root / f"inputs/caches/qwen_{strategy}.json"),
                    strategy, incremental=incremental, base_url=base_url)
                experts[strategy] = expert
                delegates.append(delegate)
            for index, query in enumerate(questions(), 1):
                for strategy in STRATEGIES:
                    outcomes.append(parsed_response(query, strategy, experts[strategy](query)))
                if index % 10 == 0 or index == 1:
                    state = dict(stage="filling", completed_questions=index, expected_questions=950,
                                 last_query=asdict(query), elapsed_seconds=time.monotonic() - started,
                                 new_requests=incremental.summary())
                    write_json(root / "status.json", state)
                    print(f"{index}/950 questions; {state['new_requests']['request_count']} new raw requests", flush=True)
            write_json(root / "responses.json", outcomes)
            write_json(root / "status.json", dict(stage="filled", completed_questions=950,
                outcomes=len(outcomes), elapsed_seconds=time.monotonic() - started,
                new_requests=incremental.summary(), manifest_sha256=sha256(root / "manifest.json")))
        except BaseException as exc:
            write_json(root / "responses.partial.json", outcomes)
            write_json(root / "status.json", dict(stage="incomplete", outcomes=len(outcomes),
                error=f"{type(exc).__name__}: {exc}", new_requests=incremental.summary()))
            raise
        finally:
            for delegate in delegates:
                if delegate is not None:
                    delegate.client.close()
            incremental.close()


def verify_bundle(root):
    manifest = read_json(root / "bundle_manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"bundle checksum differs: {name}")
    if read_json(root / "questions.json") != [asdict(q) for q in questions()]:
        raise ValueError("bundle question pool differs")
    return manifest


def load_offline_provider(root: Path, strategy: str, *, minimum_valid: int = 5):
    """Load a complete exported cache; no network client is constructed."""
    if strategy not in STRATEGIES:
        raise ValueError("unknown confidence strategy")
    verify_bundle(root)
    records = read_json(root / f"inputs/caches/qwen_{strategy}.json")
    expert, _ = provider(read_json(root / "inputs/model_metadata.json"), records,
                         strategy, minimum_valid=minimum_valid)
    allowed = frozenset(questions())

    def answer(query):
        if query not in allowed:
            raise ValueError("question is outside the frozen cache pool")
        return expert(query)

    return answer


def replay(root, records, *, expected=None):
    metadata = read_json(root / "inputs/model_metadata.json")
    outcomes = []
    # A socket cannot be opened even if a regression accidentally constructs a client.
    with patch.object(socket, "socket", side_effect=AssertionError("offline audit attempted network access")):
        experts = {s: provider(metadata, records[s], s)[0] for s in STRATEGIES}
        for query in questions():
            for strategy in STRATEGIES:
                outcomes.append(parsed_response(query, strategy, experts[strategy](query)))
        legacy = provider(metadata, records["consistency"], "consistency", minimum_valid=6)[0]
        six = [parsed_response(q, "consistency", legacy(q)) for q in questions()]
    if expected is not None and outcomes != expected:
        raise ValueError("offline replay differs from saved outcomes")
    return outcomes, six


def audit(root):
    if (root / "bundle_manifest.json").exists():
        verify_bundle(root)
        records = {s: read_json(root / f"inputs/caches/qwen_{s}.json") for s in STRATEGIES}
        outcomes, six = replay(root, records, expected=read_json(root / "responses.json"))
        if six != read_json(root / "responses_min6.json"):
            raise ValueError("six-response replay differs")
        with patch.object(socket, "socket", side_effect=AssertionError("offline loader attempted network access")):
            loaders = {s: load_offline_provider(root, s) for s in STRATEGIES}
            loaded = [parsed_response(q, s, loaders[s](q)) for q in questions() for s in STRATEGIES]
        if loaded != outcomes:
            raise ValueError("public offline loader differs from replay")
        print("Portable cache passed 1,900 offline outcomes and 950 six-response checks.", flush=True)
        return dict(passed=True, offline_outcomes=len(outcomes), six_response_outcomes=len(six))
    manifest = verify(root)
    records = raw_records(root)
    outcomes, six = replay(root, records, expected=read_json(root / "responses.json"))
    expected_keys = set()
    for strategy in STRATEGIES:
        imported = read_json(root / f"inputs/caches/qwen_{strategy}.json")
        if any(records[strategy].get(key) != value for key, value in imported.items()):
            raise ValueError("an imported response changed")
        if request_settings(records[strategy]) != manifest["request_settings"][strategy]:
            raise ValueError("new responses use different settings")
    for outcome in outcomes:
        expected_keys.update(outcome["request_keys"])
        if outcome["strategy"] == "consistency" and (len(outcome["votes"]) != 8 or len(outcome["request_keys"]) != 8):
            raise ValueError("consistency lacks eight variants")
    all_keys = set(records["stated"]) | set(records["consistency"])
    if expected_keys != all_keys:
        raise ValueError("raw requests are missing or outside the frozen protocol")
    write_json(root / "responses_min6.json", six)
    summary = dict(passed=True, question_count=950, offline_outcomes=len(outcomes),
        six_response_outcomes=len(six), imported_responses_unchanged=True, raw_requests=len(all_keys),
        manifest_sha256=sha256(root / "manifest.json"), coverage=[])
    for strategy in STRATEGIES:
        for group in ("I", "II"):
            selected = [r for r in outcomes if r["strategy"] == strategy and
                        ("II" if r["query"]["kind"] == "ancestry" else "I") == group]
            summary["coverage"].append(dict(strategy=strategy, group=group, cached=len(selected),
                valid=sum(r["valid"] for r in selected), abstentions=sum(not r["valid"] for r in selected),
                neutral=sum(r["reported_confidence"] == 0.5 for r in selected)))
    write_json(root / "audit.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def export(root):
    summary = audit(root)
    records = raw_records(root)
    bundle = root / "portable_cache"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("questions.json", "responses.json", "responses_min6.json", "audit.json",
                 "inputs/model_metadata.json", "inputs/runtime.json"):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copyfile(root / "manifest.json", bundle / "provenance.json")
    for strategy in STRATEGIES:
        write_json(bundle / f"inputs/caches/qwen_{strategy}.json", records[strategy])
    readme = """# Offline neuropathic response cache

This bundle covers 570 Group I and 380 ancestry questions, with stated and eight-variant consistency confidence. An abstention is a cached outcome. Model prompts, source provenance, checkpoint metadata, and raw responses are included. Data seeds and sample sizes do not change these prompts.

Use the matching cd-llm code and its Python environment. This is a response-data bundle, not a Python environment or a model checkpoint.

    cd-llm experiment neuropathic-claim-cache audit --cache-root /path/to/neuropathic_claim_cache

In Python, call `load_offline_provider(Path(bundle_directory), "stated")` or `load_offline_provider(Path(bundle_directory), "consistency", minimum_valid=5)` from `cd_llm.experiments.neuropathic_claim_cache`. The returned function accepts the canonical ClaimQuery objects returned by that module's `questions()` function. `minimum_valid=6` reproduces the historical consistency rule using exactly the same raw variants. No API keys, network service, model weights, observations, or reference graph are required. SMC should request only its adaptively selected question's response.

The five-response rule uses max(0.5, (majority count + 0.5)/9); ties receive 0.5. Reported confidence remains unmodified in this cache; the inference procedure applies its own soft-confidence cap. Separate ordering wordings and changed clinical descriptions, prompts, models, or decoding settings are outside this cache's coverage.
"""
    write_text(bundle / "README.md", readme)
    files = {str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob("*"))
             if p.is_file() and p.name != "bundle_manifest.json"}
    write_json(bundle / "bundle_manifest.json", dict(version=1, study=STUDY,
        question_count=950, outcomes=1900, files=files, source_manifest_sha256=sha256(root / "manifest.json")))
    archive = root / "neuropathic_claim_cache.tar.gz"
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        handle.add(bundle, arcname=STUDY)
    temporary.replace(archive)
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="relocated_", dir=validation) as temporary_directory:
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(temporary_directory, filter="data")
        relocated = audit(Path(temporary_directory) / STUDY)
    status = read_json(root / "status.json")
    manifest = read_json(root / "manifest.json")
    lines = ["# Complete neuropathic Group I–II response cache", "",
        "All 950 questions are cached for both stated and consistency confidence (1,900 outcomes), including genuine abstentions. Group I has 380 directed-edge and 190 adjacency questions; Group II has 380 canonical ancestry questions. Separate ordering wordings are excluded.", "",
        "The cache preserves the completed Table 2 ablation's Qwen3.5-9B checkpoint, masked node descriptions, exact prompts, per-strategy seeds, greedy decoding, disabled thinking, and 256-token limit. Consistency retains all eight variants and exports both five-response and historical six-response interpretations. Its confidence is max(0.5, (majority count + 0.5)/9), with ties at 0.5. No target graph or observations were used for elicitation.", "",
        "| Confidence | Group | Cached | Valid | Abstentions | Confidence 0.5 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in summary["coverage"]:
        lines.append(f"| {row['strategy']} | {row['group']} | {row['cached']} | {row['valid']} | {row['abstentions']} | {row['neutral']} |")
    lines += ["", f"Imported {sum(v['raw_requests'] for v in manifest['imported'].values())} raw responses unchanged. Generated {status['new_requests']['request_count']} new responses, yielding {summary['raw_requests']} raw records including conditional stated retries.",
        f"New requests consumed {status['new_requests']['prompt_tokens']} prompt tokens and {status['new_requests']['completion_tokens']} completion tokens. The fill stage took {status['elapsed_seconds']:.1f} seconds, excluding model startup and final validation.", "",
        "All 1,900 outcomes replayed with socket creation disabled; all 950 six-response consistency outcomes also replayed. The public offline loader passed, and the compressed bundle was extracted to a different directory and passed the same offline audit. Input and raw-response checksums are included.", "",
        "Download `neuropathic_claim_cache.tar.gz` or copy `portable_cache/`. See the included README for offline usage. These responses are reusable across observational seeds and sample sizes with the same node descriptions and prompt protocol; future experiments must expose only adaptively selected answers to their inference procedure."]
    write_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "completion.json", dict(complete=True, audit_passed=True, question_count=950,
        outcomes=1900, relocated_audit=relocated, manifest_sha256=sha256(root / "manifest.json"),
        archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size))
    print(f"Complete portable cache: {archive}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fill", "audit", "export"))
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--run-id")
    location.add_argument("--cache-root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    args = parser.parse_args(argv)
    root = args.cache_root if args.cache_root is not None else WORKSPACE_ROOT / "results" / run_id(args.run_id) / STUDY
    if args.stage in ("prepare", "fill") and args.run_id is None:
        parser.error("prepare/fill require --run-id")
    if args.stage == "prepare":
        prepare(root)
    elif args.stage == "fill":
        fill(root, args.base_url)
    elif args.stage == "audit":
        audit(root)
    else:
        export(root)


if __name__ == "__main__":
    main()
