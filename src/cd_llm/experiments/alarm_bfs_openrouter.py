"""Frozen masked ALARM evaluation using the existing BFS+Pearson baseline."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import itertools
from pathlib import Path
import json
import re
import statistics
import threading
import time
import urllib.request

import networkx as nx
import numpy as np
import pandas as pd

from cd_llm.baselines.jiralerspong2024 import Completion, parse_answer, run_bfs
from cd_llm.core.finite_support import graph_from_edges, point_prediction_metrics
from cd_llm.paths import WORKSPACE_ROOT
from .common.artifacts import read_json, write_json, write_text, run_id, sha256
from .common.hosted_transport import _compact_bfs_messages, load_key, reserved_cost, accounted_cost

MODEL = "qwen/qwen3.5-9b"
PROVIDER = "deepinfra/bf16"
SOURCE = "https://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz"
ENDPOINTS = "https://openrouter.ai/api/v1/models/qwen/qwen3.5-9b/endpoints"
PRICES = dict(prompt=0.10e-6, completion=0.15e-6)
REPORT_LOCK = threading.RLock()

# Definitions describe individual quantities, never reference parents or children.
DESCRIPTIONS = {
    "HISTORY": "Whether the patient's recorded history indicates left ventricular failure.",
    "CVP": "Pressure measured in the central veins.",
    "PCWP": "Pressure obtained by wedging a catheter in a small pulmonary artery.",
    "HYPOVOLEMIA": "Whether the circulating blood volume is abnormally low.",
    "LVEDVOLUME": "Blood volume in the left ventricle at the end of filling.",
    "LVFAILURE": "Whether the left ventricle has impaired pumping function.",
    "STROKEVOLUME": "Blood volume ejected by the heart in one beat.",
    "ERRLOWOUTPUT": "Whether a low cardiac output measurement error is present.",
    "HRBP": "Heart rate reported by the blood pressure monitoring channel.",
    "HREKG": "Heart rate reported by an electrocardiographic monitor.",
    "ERRCAUTER": "Whether electrical cautery interference is present.",
    "HRSAT": "Heart rate reported by the oxygen saturation monitoring channel.",
    "INSUFFANESTH": "Whether anesthesia or pain relief is insufficient.",
    "ANAPHYLAXIS": "Whether a severe systemic allergic reaction is present.",
    "TPR": "Overall resistance of the peripheral blood vessels.",
    "EXPCO2": "Carbon dioxide level measured in exhaled gas.",
    "KINKEDTUBE": "Whether the breathing tube is kinked.",
    "MINVOL": "Measured volume of gas breathed per minute.",
    "FIO2": "Oxygen fraction of the gas supplied for inhalation.",
    "PVSAT": "Oxygen saturation of blood in the pulmonary veins.",
    "SAO2": "Oxygen saturation measured in arterial blood.",
    "PAP": "Pressure measured in the pulmonary artery.",
    "PULMEMBOLUS": "Whether a clot obstructs the pulmonary circulation.",
    "SHUNT": "Extent of blood flow bypassing effective gas exchange in the lungs.",
    "INTUBATION": "Breathing tube placement. Numeric categories 0, 1 and 2 mean normal placement, placement in the esophagus and placement into one lung, respectively. These categories have no natural numeric order.",
    "PRESS": "Pressure measured in the breathing circuit.",
    "DISCONNECT": "Whether the breathing circuit is disconnected.",
    "MINVOLSET": "Prescribed ventilator volume per minute.",
    "VENTMACH": "Ventilation delivered at the mechanical ventilator.",
    "VENTTUBE": "Ventilation passing through the breathing tube.",
    "VENTLUNG": "Ventilation reaching the lungs.",
    "VENTALV": "Ventilation reaching the gas exchanging air spaces.",
    "ARTCO2": "Carbon dioxide level in arterial blood.",
    "CATECHOL": "Level of circulating catecholamines.",
    "HR": "Actual heart rate.",
    "CO": "Volume of blood pumped by the heart per minute.",
    "BP": "Systemic arterial blood pressure.",
}
CONTEXT = (
    "clinical physiology and monitoring of a patient receiving anesthesia and mechanical ventilation. "
    "The variables describe physiological states, equipment conditions, settings and recorded measurements. "
    "Identifiers are arbitrary. Binary variables are coded absent 0 and present 1. "
    "Ordered levels are coded consecutively from 0 in increasing order. "
    "The symbol nan in a correlation means that it is undefined because an observed variable is constant"
)


def root_for(timestamp):
    return WORKSPACE_ROOT / "results" / run_id(timestamp) / "alarm_bfs_openrouter"


def parse_bif(text):
    """Read and validate the finite table subset used by the official ALARM BIF."""
    variables = re.findall(
        r"variable\s+(\w+)\s*\{\s*type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{([^}]+)\};\s*\}", text)
    names = [name for name, _, _ in variables]
    levels = {name: [v.strip() for v in body.split(",")] for name, _, body in variables}
    if not names or len(set(names)) != len(names):
        raise ValueError("missing or duplicate variables")
    for name, count, _ in variables:
        if len(levels[name]) != int(count) or len(set(levels[name])) != int(count):
            raise ValueError("invalid state declarations")
    parents, tables = {}, {}
    for header, body in re.findall(r"probability\s*\(([^)]+)\)\s*\{([^}]+)\}", text):
        pieces = header.split("|")
        name = pieces[0].strip()
        if name not in levels or name in parents:
            raise ValueError("invalid or duplicate probability block")
        ps = [] if len(pieces) == 1 else [p.strip() for p in pieces[1].split(",")]
        if any(p not in levels for p in ps) or len(set(ps)) != len(ps):
            raise ValueError("invalid parents")
        parents[name] = ps
        table = {}
        root_row = re.fullmatch(r"\s*table\s+([^;]+);\s*", body) if not ps else None
        if not ps and root_row is None:
            raise ValueError("invalid root probability table")
        rows = re.findall(r"\(([^)]+)\)\s*([^;]+);", body) if ps else [("", root_row.group(1))]
        for states, values in rows:
            key = tuple(levels[p].index(v.strip()) for p, v in zip(ps, states.split(","), strict=True)) if ps else ()
            probabilities = np.array([float(v.strip()) for v in values.split(",")])
            if key in table or len(probabilities) != len(levels[name]):
                raise ValueError("invalid conditional table shape")
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-6, rtol=0):
                raise ValueError("invalid probability row")
            table[key] = probabilities / probabilities.sum()
        expected = set(itertools.product(*(range(len(levels[p])) for p in ps)))
        if set(table) != expected:
            raise ValueError("incomplete conditional table")
        tables[name] = table
    if set(parents) != set(names):
        raise ValueError("missing probability block")
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from((p, c) for c, ps in parents.items() for p in ps)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("source must be a DAG")
    return names, levels, parents, tables, graph


def sample_states(parsed, count, seed):
    names, _, parents, tables, graph = parsed
    indices = {name: i for i, name in enumerate(names)}
    result = np.empty((count, len(names)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for name in nx.topological_sort(graph):
        uniforms = rng.random(count)
        for configuration, probabilities in tables[name].items():
            mask = np.ones(count, dtype=bool)
            for parent, value in zip(parents[name], configuration):
                mask &= result[:, indices[parent]] == value
            cumulative = np.cumsum(probabilities)
            cumulative[-1] = 1.
            result[mask, indices[name]] = np.searchsorted(cumulative, uniforms[mask], side="right")
    return result


def encoding(levels):
    result = {}
    ordinal = ("ZERO", "LOW", "NORMAL", "HIGH")
    for name, states in levels.items():
        if set(states) == {"TRUE", "FALSE"}:
            result[name] = {state: int(state == "TRUE") for state in states}
        elif set(states) <= set(ordinal):
            ordered = [s for s in ordinal if s in states]
            result[name] = {state: ordered.index(state) for state in states}
        elif name == "INTUBATION" and states == ["NORMAL", "ESOPHAGEAL", "ONESIDED"]:
            result[name] = dict(zip(states, range(3)))
        else:
            raise ValueError(f"unregistered categorical encoding for {name}")
    return result


def encode_states(states, names, levels, codes):
    return np.column_stack([
        np.array([codes[name][state] for state in levels[name]])[states[:, i]]
        for i, name in enumerate(names)])


def presentation(names):
    labels = np.random.default_rng(20270911).permutation(len(names))
    return dict(mapping={name: f"V{int(labels[i])+1:02d}" for i, name in enumerate(names)},
                order=np.random.default_rng(20270912).permutation(len(names)).tolist(),
                identifier_seed=20270911, presentation_seed=20270912)


def transform(values, names, spec):
    return pd.DataFrame(values[:, spec["order"]],
                        columns=[spec["mapping"][names[i]] for i in spec["order"]])


def audit_messages(messages, names, original_names):
    supplied = "\n".join(m["content"] for m in messages if m["role"] != "assistant")
    if re.search(r"\b(?:ALARM|bnlearn|Beinlich)\b", supplied, re.I):
        raise ValueError("benchmark identity in prompt")
    if any(re.search(r"\b" + re.escape(name) + r"\b", supplied) for name in original_names):
        raise ValueError("original identifier in supplied prompt")
    if set(re.findall(r"\bV\d+\b", supplied)) - set(names):
        raise ValueError("unregistered masked identifier")
    labels = re.findall(r"^([A-Za-z_]\w*):", supplied, re.M)
    if set(labels) - set(names):
        raise ValueError("nonmasked metadata or correlation label")


def prepare(timestamp):
    root = root_for(timestamp)
    if (root / "manifest.json").exists():
        return validate(root)
    raw = urllib.request.urlopen(SOURCE, timeout=60).read()
    text = gzip.decompress(raw).decode()
    parsed = parse_bif(text)
    names, levels, _, _, graph = parsed
    if set(names) != set(DESCRIPTIONS) or len(graph.edges()) != 46 or nx.dag_longest_path_length(graph) != 10:
        raise ValueError("source does not match the registered ALARM graph")
    endpoint_info = json.load(urllib.request.urlopen(ENDPOINTS, timeout=60))
    endpoint = next(e for e in endpoint_info["data"]["endpoints"] if e["tag"] == PROVIDER)
    required = {"reasoning", "seed", "temperature", "max_tokens"}
    if endpoint["quantization"] != "bf16" or not required <= set(endpoint["supported_parameters"]):
        raise ValueError("provider does not support the frozen configuration")
    if any(float(endpoint["pricing"][k]) > PRICES[k] + 1e-15 for k in PRICES):
        raise ValueError("provider prices exceed the planned rates")
    codes, spec = encoding(levels), presentation(names)
    write_text(root / "source/alarm.bif", text)
    write_json(root / "source/provider_endpoints.json", endpoint_info)
    write_json(root / "evaluation_only/reference.json", dict(
        names=names, edges=[[names.index(a), names.index(b)] for a, b in graph.edges()],
        nodes=37, edge_count=46, longest_path_edges=10))
    definitions = []
    for i in spec["order"]:
        name = names[i]
        description = DESCRIPTIONS[name]
        if set(levels[name]) <= {"ZERO", "LOW", "NORMAL", "HIGH"}:
            description += " Ordered levels are " + ", ".join(
                state.lower() for state in sorted(levels[name], key=codes[name].get)) + "."
        definitions.append(dict(id=spec["mapping"][name], description=description))
    metadata = dict(context=CONTEXT, nodes=definitions)
    write_json(root / "inputs/metadata.json", metadata)
    write_json(root / "inputs/presentation.json", spec)
    write_json(root / "inputs/encoding.json", dict(names=names, levels=levels, codes=codes))
    write_json(root / "inputs/description_review.json", [
        dict(original=name, states=levels[name], paraphrase=DESCRIPTIONS[name]) for name in names])
    audit = []
    for seed in range(5):
        states = sample_states(parsed, 200, seed)
        values = encode_states(states, names, levels, codes)
        np.save(root / f"inputs/observations_{seed}.npy", values, allow_pickle=False)
        original = pd.DataFrame(values, columns=names).corr()
        nominal = names.index("INTUBATION")
        variants = []
        for perm in itertools.permutations(range(3)):
            alternative = values.copy()
            alternative[:, nominal] = np.array(perm)[states[:, nominal]]
            correlations = pd.DataFrame(alternative, columns=names).corr()
            differences = np.abs(correlations["INTUBATION"] - original["INTUBATION"]).drop("INTUBATION")
            variants.append(dict(encoding=list(perm), max_absolute_correlation_change=float(differences.max())))
        audit.append(dict(seed=seed, constant_variables=[names[i] for i in range(37) if np.ptp(values[:, i]) == 0],
                          nominal_encoding_sensitivity=variants))
    write_json(root / "inputs/encoding_audit.json", audit)
    from cd_llm.baselines.jiralerspong2024 import algorithm
    from .common import hosted_transport
    from cd_llm.core import finite_support
    sources = [Path(__file__), Path(algorithm.__file__), Path(hosted_transport.__file__),
               Path(finite_support.__file__)]
    frozen = sources + list((root / "inputs").glob("*")) + list((root / "source").glob("*")) + [root / "evaluation_only/reference.json"]
    manifest = dict(model=MODEL, provider=PROVIDER, quantization="bf16", revision=None,
                    source_url=SOURCE, variable_documentation="https://www.bnlearn.com/documentation/man/alarm.html",
                    seeds=list(range(5)), observations=200, temperature=.7, max_tokens=4096,
                    reasoning_enabled=False, context_character_limit=32000, cost_cap_usd=1.,
                    prices=PRICES, concurrency=2, request_timeout_seconds=600,
                    frozen_sha256={str(path): sha256(path) for path in frozen})
    write_json(root / "manifest.json", manifest, exclusive=True)
    def example(messages):
        audit_messages(messages, [v["id"] for v in definitions], names)
        write_json(root / "example_root_prompt.json", messages)
        return Completion("<Answer>None</Answer>", 0, 0)
    run_bfs(transform(np.load(root / "inputs/observations_0.npy"), names, spec), example,
            include_statistics=True, descriptions={v["id"]: v["description"] for v in definitions},
            domain_context=CONTEXT, empty_root_policy="empty_graph")
    print(json.dumps(dict(prepared=str(root), runs=5, observations=200)), flush=True)
    return manifest


def validate(root):
    manifest = read_json(root / "manifest.json")
    for filename, expected in manifest["frozen_sha256"].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"frozen input or implementation changed at {filename}")
    return manifest


def request_for(messages, seed):
    return dict(model=MODEL, messages=messages, temperature=.7, seed=seed, max_tokens=4096,
                extra_body=dict(reasoning=dict(enabled=False), provider=dict(
                    only=[PROVIDER], allow_fallbacks=False, require_parameters=True,
                    max_price=dict(prompt=.10, completion=.15))))


def request_digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completion_text(choice):
    """Keep unfinished text for the repair prompt without accepting its XML."""
    text = choice["message"]["content"] or ""
    if choice["finish_reason"] != "stop":
        text = re.sub(r"<(/?)Answer>", r"<\1UnfinishedAnswer>", text, flags=re.I)
    return text


def bfs_messages(messages):
    """Preserve the failed answer when the existing format retry is compacted."""
    result = _compact_bfs_messages(messages)
    if not messages[-1]["content"].startswith("Your previous response did not follow"):
        return result
    # A repair refers to the preceding assistant response. Dropping that
    # response during compaction turns the repair into another generation.
    if len(result) != len(messages):
        previous_user = next(m for m in reversed(messages[:-2]) if m["role"] == "user")
        result = [dict(m) for m in messages[:2]]
        if previous_user is not messages[1]:
            result.append(dict(previous_user))
        result.extend([dict(messages[-2]), dict(messages[-1])])
    else:
        result = [dict(m) for m in result]
    result[-1]["content"] += (
        "\nThis turn is only an answer-format repair. Return exactly one <Answer>...</Answer> block "
        "and no explanation or additional reasoning. Use the preceding assessment and the same question."
    )
    return result


class Transport:
    def __init__(self, root, key):
        self.root, self.key = root, key
        self.lock = threading.Lock()
        self.cache = {}
        self.spent = 0.
        for path in sorted((root / "requests").glob("*.json")):
            record = read_json(path)
            self.spent += accounted_cost(record, PRICES)
            key = request_digest(record["request"])
            self.cache.setdefault(key, []).append((path, record))

    async def send(self, request):
        from openai import AsyncOpenAI
        async with AsyncOpenAI(api_key=self.key, base_url="https://openrouter.ai/api/v1",
                               max_retries=0, timeout=600) as client:
            response = await asyncio.wait_for(client.chat.completions.create(**request), timeout=600)
            return response.model_dump()

    def query(self, seed, index, messages):
        request = request_for(messages, seed)
        key = request_digest(request)
        cached = self.cache.get(key, [])
        completed = [(p, r) for p, r in cached if "response" in r]
        if completed:
            path, saved = completed[0]
            write_json(self.root / f"cache_uses/r{seed}_q{index:03d}.json",
                       dict(request_sha256=key, source_request=str(path), recorded=time.time()))
            return self.checked_response(saved["response"])
        if any(r.get("status_code") not in (429, 500, 502, 503, 504) for _, r in cached):
            raise RuntimeError("API request failed or has uncertain completion; see saved request")
        for attempt in range(3):
            path = self.root / f"requests/r{seed}_q{index:03d}_{key[:16]}_a{attempt}.json"
            if attempt < len(cached):
                path, saved = cached[attempt]
            else:
                reserve = reserved_cost(request, PRICES)
                with self.lock:
                    if self.spent + reserve > 1.:
                        raise RuntimeError("API spending ceiling reached")
                    saved = dict(request=request, reserved_usd=reserve, started=time.time())
                    write_json(path, saved, exclusive=True)
                    self.spent += reserve
                try:
                    saved["response"] = asyncio.run(self.send(request))
                except Exception as error:
                    saved.update(error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
                saved["seconds"] = time.time() - saved["started"]
                with self.lock:
                    write_json(path, saved)
                    self.spent += accounted_cost(saved, PRICES) - reserve
                    self.cache.setdefault(key, []).append((path, saved))
                print(json.dumps(dict(seed=seed, query=index, attempt=attempt,
                                      status="response" if "response" in saved else saved.get("error_type"),
                                      accounted_usd=round(self.spent, 6))), flush=True)
            if "response" in saved:
                return self.checked_response(saved["response"])
            if saved.get("status_code") not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError("API request failed or has uncertain completion; see saved request")
            time.sleep(15 * (attempt + 1))
        raise AssertionError("unreachable")

    @staticmethod
    def checked_response(response):
        if response.get("model") not in (MODEL, MODEL + "-20260310"):
            raise ValueError("unexpected response model")
        if response.get("provider", "").lower() != "deepinfra":
            raise ValueError("unexpected or missing provider")
        details = (response.get("usage") or {}).get("completion_tokens_details") or {}
        if details.get("reasoning_tokens", 0):
            raise ValueError("reasoning enabled despite the frozen request")
        return response


def run_one(root, seed, transport):
    destination = root / f"predictions/{seed}.json"
    if destination.exists():
        return True
    validate(root)
    spec = read_json(root / "inputs/presentation.json")
    original_names = read_json(root / "inputs/encoding.json")["names"]
    metadata = read_json(root / "inputs/metadata.json")
    frame = transform(np.load(root / f"inputs/observations_{seed}.npy"), original_names, spec)
    names = list(frame.columns)
    diagnostics, roots, visited = [], [], set()
    started = time.time()
    write_json(root / f"status/{seed}.json", dict(state="running", started=started))
    try:
        def complete(messages):
            nonlocal roots
            request_messages = bfs_messages(messages)
            audit_messages(request_messages, names, original_names)
            queried = next((re.search(r"Select the variables that are caused by (V\d+)\.", m["content"])
                            for m in reversed(messages) if re.search(r"Select the variables that are caused by (V\d+)\.", m["content"])), None)
            source = queried.group(1) if queried else None
            response = transport.query(seed, len(diagnostics), request_messages)
            choice = response["choices"][0]
            text = completion_text(choice)
            valid = True
            try:
                parsed = parse_answer(text, names, drop_unknown=True)
            except ValueError:
                valid, parsed = False, []
            body = re.findall(r"<Answer>(.*?)</Answer>", text, re.I | re.S)
            unknown = [v.strip().strip("'\"") for v in body[-1].split(",")
                       if v.strip().strip("'\"") not in names and v.strip().lower() not in ("none", "", "n/a", "no variables")] if body else []
            diagnostics.append(dict(source=source, valid_answer_block=valid, unknown_names=unknown,
                                    finish_reason=choice["finish_reason"], parsed_identifiers=parsed,
                                    format_retry=messages[-1]["content"].startswith("Your previous response did not follow"),
                                    request_sha256=request_digest(request_for(request_messages, seed))))
            if source is None and valid:
                roots = parsed
            if source is not None:
                visited.add(source)
            write_json(root / f"progress/{seed}.json", dict(seed=seed, queries=len(diagnostics),
                       roots=roots, visited=sorted(visited), diagnostics=diagnostics, updated=time.time()))
            usage = response.get("usage") or {}
            return Completion(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        edges, native_usage = run_bfs(frame, complete, include_statistics=True,
            descriptions={v["id"]: v["description"] for v in metadata["nodes"]},
            domain_context=metadata["context"], opaque_identifiers=False, empty_root_policy="empty_graph")
        if any(d["finish_reason"] != "stop" and d["valid_answer_block"] for d in diagnostics):
            raise ValueError("a parsed answer came from a truncated response")
        canonical = [(spec["order"][a], spec["order"][b]) for a, b in edges]
        write_json(destination, dict(seed=seed, n=200, edges=canonical,
                   masked_edges=[[names[a], names[b]] for a, b in edges],
                   roots=roots, visited=sorted(visited), diagnostics=diagnostics, usage=native_usage,
                   wall_seconds=time.time()-started, manifest_sha256=sha256(root / "manifest.json"),
                   saved_before_reference_evaluation=True), exclusive=True)
        write_json(root / f"status/{seed}.json", dict(state="complete", seconds=time.time()-started))
        return True
    except Exception as error:
        write_json(root / f"status/{seed}.json", dict(state="incomplete", error_type=type(error).__name__,
                   error=str(error), seconds=time.time()-started, diagnostics=diagnostics))
        print(json.dumps(dict(seed=seed, state="incomplete", error=str(error))), flush=True)
        return False


def precision_recall(predicted, actual):
    tp = len(predicted & actual)
    precision = tp / len(predicted) if predicted else 0.
    recall = tp / len(actual) if actual else 0.
    return dict(precision=precision, recall=recall, f1=2*tp/(len(predicted)+len(actual)) if predicted or actual else 1.)


def score_edges(edges, reference):
    count = reference["nodes"]
    actual, predicted = set(map(tuple, reference["edges"])), set(map(tuple, edges))
    truth = graph_from_edges(count, directed=actual)
    point = graph_from_edges(count, directed=predicted)
    values = point_prediction_metrics(point, truth, normalized=True)
    tg, pg = nx.DiGraph(), nx.DiGraph()
    tg.add_nodes_from(range(count))
    pg.add_nodes_from(range(count))
    tg.add_edges_from(actual)
    pg.add_edges_from(predicted)
    ta, pa = set(nx.transitive_closure_dag(tg).edges()), set(nx.transitive_closure_dag(pg).edges())
    indirect = ta - actual
    return dict(adjacency_f1=values["adjacency_f1"], directed_edge_f1=values["orientation_f1"],
                graph_distance=values["expected_distance"], edge_count=len(predicted),
                directed=precision_recall(predicted, actual), ancestry=precision_recall(pa, ta),
                indirect_shortcut_edges=len(predicted & indirect),
                indirect_shortcut_fraction=len(predicted & indirect)/len(predicted) if predicted else 0.,
                correct_direct_edges=len(predicted & actual))


def report(timestamp, quiet=False):
    with REPORT_LOCK:
        return _report(timestamp, quiet)


def _report(timestamp, quiet=False):
    root = root_for(timestamp)
    validate(root)
    predictions = [read_json(p) for p in sorted((root / "predictions").glob("*.json"))]
    reference = read_json(root / "evaluation_only/reference.json")
    records = [dict(p, metrics=score_edges(p["edges"], reference)) for p in predictions]
    requests = [read_json(p) for p in (root / "requests").glob("*.json")]
    tokens = dict(input=0, output=0)
    reported_cost = 0.
    for request in requests:
        usage = request.get("response", {}).get("usage") or {}
        tokens["input"] += usage.get("prompt_tokens", 0)
        tokens["output"] += usage.get("completion_tokens", 0)
        reported_cost += float(usage.get("cost") or 0)
    summary = {}
    for metric in ("adjacency_f1", "directed_edge_f1", "graph_distance", "edge_count",
                   "indirect_shortcut_edges", "indirect_shortcut_fraction"):
        values = [p["metrics"][metric] for p in records]
        summary[metric] = dict(mean=statistics.mean(values) if values else None,
                              se=statistics.stdev(values)/len(values)**.5 if len(values)>1 else None)
    for metric in ("precision", "recall", "f1"):
        values = [p["metrics"]["ancestry"][metric] for p in records]
        summary["ancestry_"+metric] = dict(mean=statistics.mean(values) if values else None,
                              se=statistics.stdev(values)/len(values)**.5 if len(values)>1 else None)
    result = dict(completed=len(records), planned=5, records=records, summary=summary,
                  updated=time.time(),
                  statuses={p.stem: read_json(p) for p in (root / "status").glob("*.json")},
                  progress={p.stem: read_json(p) for p in (root / "progress").glob("*.json")},
                  api=dict(dispatches=len(requests), tokens=tokens, reported_usd=reported_cost,
                           accounted_usd=sum(accounted_cost(r, PRICES) for r in requests), cap_usd=1.))
    write_json(root / "report.json", result)
    lines = ["# Masked ALARM BFS+Pearson", "", f"Completed {len(records)} of 5 runs with 200 observations each.",
             "", "| Seed | State | Responses received | Variables visited | Format retries | Truncated responses |",
             "| --- | --- | --- | --- | --- | --- |"]
    for seed in range(5):
        state = result["statuses"].get(str(seed), {}).get("state", "waiting")
        progress = result["progress"].get(str(seed), {})
        diagnostics = progress.get("diagnostics", [])
        retries = sum(d.get("format_retry", False) for d in diagnostics)
        truncated = sum(d["finish_reason"] == "length" for d in diagnostics)
        lines.append(f'| {seed} | {state} | {progress.get("queries", 0)} | {len(progress.get("visited", []))} | {retries} | {truncated} |')
    lines += [
             "", "| Seed | Adjacency F1 | Directed F1 | Graph distance | Edges | Ancestry F1 | Indirect shortcuts |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for p in records:
        m = p["metrics"]
        lines.append(f'| {p["seed"]} | {m["adjacency_f1"]:.4f} | {m["directed_edge_f1"]:.4f} | {m["graph_distance"]:.4f} | {m["edge_count"]} | {m["ancestry"]["f1"]:.4f} | {m["indirect_shortcut_edges"]} |')
    lines += ["", "| Metric | Mean | Standard error |", "| --- | --- | --- |"]
    for name, value in summary.items():
        if value["mean"] is not None:
            se = "undefined" if value["se"] is None else f'{value["se"]:.4f}'
            lines.append(f'| {name} | {value["mean"]:.4f} | {se} |')
    lines += ["", f'API dispatches {len(requests)}. Input tokens {tokens["input"]}. Output tokens {tokens["output"]}.',
              f'Reported API cost USD {reported_cost:.6f}. Cost including uncertain reservations USD {result["api"]["accounted_usd"]:.6f}.',
              "", "Masking retains clinical meanings and cannot rule out benchmark recognition.",
              "The intubation category coding is arbitrary. Its correlation sensitivity is recorded with the frozen inputs.",
              "These are samples from a clinical Bayesian network, not patient cohort observations.",
              "Incomplete runs remain visible and are excluded from numerical averages. No seeds are replaced."]
    write_text(root / "report.md", "\n".join(lines)+"\n")
    if not quiet:
        print(json.dumps(dict(completed=len(records), planned=5, api=result["api"], summary=summary)), flush=True)
    return result


def run(timestamp):
    root = root_for(timestamp)
    validate(root)
    key = load_key()
    with (root / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport = Transport(root, key)
        stop = threading.Event()
        def monitor():
            while not stop.wait(30):
                try:
                    report(timestamp, quiet=True)
                except Exception as error:
                    write_json(root / "report_error.json", dict(error=str(error), at=time.time()))
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            if not run_one(root, 0, transport):
                raise RuntimeError("first complete test failed; remaining runs were not started")
            report(timestamp)
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(lambda seed: run_one(root, seed, transport), range(1, 5)))
            if not all(outcomes):
                raise RuntimeError("some runs are incomplete; inspect statuses")
        finally:
            stop.set()
            watcher.join()
            report(timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.command](args.run_id)


if __name__ == "__main__":
    main()
