"""Evidence reports and calibrated, joint likelihoods for a DAG variable pair.

Source accuracy is not a posterior probability of a causal claim. Calibration
fits P(report | pair state), and one report contributes one factor. Abstention
is deliberately ignored rather than interpreted as a negative causal answer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cd_llm.core.finite_support import Graph, has_edge, transitive_closure


PAIR_STATES = ("forward_direct", "forward_indirect", "reverse_direct", "reverse_indirect", "unrelated")
# Each report denotes one event, not a product of its logical consequences.
REPORT_STATES = {
    "forward_direct": (0,), "reverse_direct": (2,),
    "forward_ancestry": (0, 1), "reverse_ancestry": (2, 3),
    "no_forward_direct": (1, 2, 3, 4), "no_reverse_direct": (0, 1, 3, 4),
    "no_forward_ancestry": (2, 3, 4), "no_reverse_ancestry": (0, 1, 4),
    "unrelated": (4,), "unknown": (0, 1, 2, 3, 4),
}
REPORTS = tuple(REPORT_STATES)
TRUTH_MATRIX = np.asarray(((1, 0, 1, 0), (0, 0, 1, 0), (0, 1, 0, 1),
                           (0, 0, 0, 1), (0, 0, 0, 0)), dtype=float)
TARGETS = ("forward_direct", "reverse_direct", "forward_ancestry", "reverse_ancestry")


def pair_state(graph: Graph, source: int, target: int) -> int:
    if graph.regime != "dag" or source == target or not (0 <= source < graph.node_count and 0 <= target < graph.node_count):
        raise ValueError("pair states require a DAG and distinct valid endpoints")
    if has_edge(graph, source, target):
        return 0
    if has_edge(graph, target, source):
        return 2
    closure = transitive_closure(graph)
    if closure[source] & (1 << target):
        return 1
    if closure[target] & (1 << source):
        return 3
    return 4


@dataclass(frozen=True)
class CausalReport:
    relation: str
    confidence: float | None
    source_id: str
    quote: str = ""
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.relation not in REPORT_STATES or not self.source_id:
            raise ValueError("invalid report category or missing source identity")
        if self.relation == "unknown":
            if self.confidence is not None:
                raise ValueError("abstention cannot carry confidence")
        elif self.confidence is None or not math.isfinite(self.confidence) or not 0.5 <= self.confidence <= 1:
            raise ValueError("answered reports need finite confidence in [0.5, 1]")

    @property
    def usable(self) -> bool:
        return self.valid and self.relation != "unknown"

    @property
    def symbol(self) -> str:
        if not self.usable:
            return "unknown"
        return self.relation + ("/high" if self.confidence >= 0.8 else "/ordinary")


SYMBOLS = ("unknown",) + tuple(f"{r}/{b}" for r in REPORTS if r != "unknown" for b in ("ordinary", "high"))


@dataclass(frozen=True)
class PairLikelihood:
    source: int
    target: int
    log_values: tuple[float, ...]
    source_family: str

    def __post_init__(self) -> None:
        if type(self.source) is not int or type(self.target) is not int or min(self.source, self.target) < 0 or self.source == self.target:
            raise ValueError("factor endpoints must be distinct nonnegative integers")
        if len(self.log_values) != 5 or not all(math.isfinite(x) for x in self.log_values) or not self.source_family:
            raise ValueError("factor needs five finite log likelihoods and provenance")

    def log_value(self, graph: Graph) -> float:
        return self.log_values[pair_state(graph, self.source, self.target)]

    @property
    def informative(self) -> bool:
        return max(self.log_values) - min(self.log_values) > 1e-12


def validate_factors(factors: Sequence[PairLikelihood], node_count: int | None = None) -> tuple[PairLikelihood, ...]:
    values = tuple(factors)
    if len({f.source_family for f in values}) != len(values):
        raise ValueError("dependent source family would be counted more than once")
    if node_count is not None and any(max(f.source, f.target) >= node_count for f in values):
        raise ValueError("factor endpoint is outside the graph")
    return values


def report_factor(report: CausalReport, source: int, target: int, *, confidence: float | None = None) -> PairLikelihood:
    """Legacy symmetric weighting of one report's event, including ancestry."""
    logs = np.zeros(5)
    if report.usable:
        raw = report.confidence if confidence is None else confidence
        if raw is None or not math.isfinite(raw) or not 0 <= raw <= 1:
            raise ValueError("invalid effective confidence")
        c = float(np.clip(raw, 0.01, 0.99))
        logs[:] = math.log1p(-c)
        logs[list(REPORT_STATES[report.relation])] = math.log(c)
        logs -= logs.max()
    return PairLikelihood(source, target, tuple(float(x) for x in logs), report.source_id)


@dataclass(frozen=True)
class ResponseChannel:
    probabilities: tuple[tuple[float, ...], ...]
    counts: tuple[int, ...]
    shrinkage: float
    training_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.probabilities)
        if array.shape != (5, len(SYMBOLS)) or not np.all(np.isfinite(array)) or np.any(array <= 0) or not np.allclose(array.sum(axis=1), 1):
            raise ValueError("invalid response channel")

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]], *, shrinkage: float = 10.0) -> ResponseChannel:
        """Shrink toward the pooled response law, not an informative prior.

        The same response distribution in every state gives a neutral factor,
        including when a source always says no and classes are imbalanced.
        Only calibration records may train a channel.
        """
        if not math.isfinite(shrinkage) or shrinkage <= 0 or not records:
            raise ValueError("calibration data and positive shrinkage are required")
        counts = np.zeros((5, len(SYMBOLS)))
        groups = []
        for row in records:
            if row["split"] != "calibration" or row.get("truth_kind") != "generating_graph":
                raise ValueError("channel fitting requires calibration graph truth, never passage annotations")
            state = row["state"]
            if type(state) is not int or state not in range(5):
                raise ValueError("invalid calibration state")
            groups.append(str(row["group_id"]))
            counts[state, SYMBOLS.index(CausalReport(**row["report"]).symbol)] += 1
        if len(groups) != len(set(groups)):
            raise ValueError("duplicate calibration system or dependent report")
        totals = counts.sum(axis=1)
        if np.any(totals == 0):
            raise ValueError("every pair state requires calibration examples")
        pooled = (counts.sum(axis=0) + 0.5) / (counts.sum() + 0.5 * len(SYMBOLS))
        # Share the shrinkage fraction across states. Otherwise unequal class
        # counts make a constant responder spuriously informative.
        strength = float(totals.min())
        probabilities = (strength * counts / totals[:, None] + shrinkage * pooled) / (strength + shrinkage)
        return cls(tuple(tuple(float(x) for x in row) for row in probabilities), tuple(int(x) for x in totals), shrinkage, tuple(sorted(groups)))

    def factor(self, report: CausalReport, source: int, target: int) -> PairLikelihood:
        if not report.usable:
            return report_factor(report, source, target)
        logs = np.log(np.asarray(self.probabilities)[:, SYMBOLS.index(report.symbol)])
        logs -= logs.max()
        return PairLikelihood(source, target, tuple(float(x) for x in logs), report.source_id)


def state_posterior(prior: Sequence[float], factor: PairLikelihood) -> np.ndarray:
    p = np.asarray(prior, dtype=float)
    if p.shape != (5,) or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("five nonnegative prior masses are required")
    weights = np.zeros(5)
    positive = p > 0
    joint = np.log(p[positive]) + np.asarray(factor.log_values)[positive]
    weights[positive] = np.exp(joint-joint.max())
    return weights / weights.sum()


def fuse_distribution(distribution: Mapping[Graph, float], factors: Sequence[PairLikelihood]) -> dict[Graph, float]:
    if not distribution:
        raise ValueError("empty graph distribution")
    graphs = list(distribution)
    factors = validate_factors(factors, graphs[0].node_count)
    weights = np.asarray([distribution[g] for g in graphs], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid graph weights")
    logs = np.asarray([sum(f.log_value(g) for f in factors) for g in graphs])
    # Center on positive support so a high-scoring zero-mass graph cannot cause underflow.
    logs -= max(logs[weights > 0])
    updated = np.zeros_like(weights)
    positive = weights > 0
    joint = np.log(weights[positive]) + logs[positive]
    updated[positive] = np.exp(joint - joint.max())
    updated /= updated.sum()
    return dict(zip(graphs, map(float, updated), strict=True))


def parse_report(content: str, *, source_id: str, evidence: str, allowed: Sequence[str]) -> CausalReport:
    """Malformed output remains distinguishable from a valid abstention."""
    try:
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"relation", "confidence", "quote"}:
            raise ValueError("expected exactly relation, confidence and quote")
        relation, confidence, quote = value["relation"], value["confidence"], value["quote"]
        if relation not in allowed or not isinstance(quote, str):
            raise ValueError("invalid relation or quote")
        if relation == "unknown":
            if confidence is not None:
                raise ValueError("unknown confidence must be null")
        elif type(confidence) not in (float, int) or not quote.strip() or quote not in evidence:
            raise ValueError("a retained report requires confidence and an exact evidence excerpt")
        return CausalReport(relation, confidence, source_id, quote)
    except (ValueError, TypeError, KeyError) as error:
        return CausalReport("unknown", None, source_id, valid=False, reason=str(error))


def aggregate_reports(reports: Sequence[CausalReport]) -> CausalReport:
    """Two label orders are dependent checks, never independent likelihoods."""
    if not reports or len({r.source_id for r in reports}) != 1:
        raise ValueError("matched reports must have one common source identity")
    source_id = reports[0].source_id
    if not all(r.valid for r in reports):
        return CausalReport("unknown", None, source_id, valid=False, reason="invalid_variant")
    if len({r.relation for r in reports}) != 1:
        return CausalReport("unknown", None, source_id, reason="order_disagreement")
    if not reports[0].usable:
        return reports[0]
    return CausalReport(reports[0].relation, min(r.confidence for r in reports), source_id, reports[0].quote)


def elicitation_messages(context: str, evidence: str, *, protocol: str, reversed_order: bool = False) -> tuple[dict[str, str], ...]:
    if protocol not in ("fixed", "typed"):
        raise ValueError("unknown elicitation protocol")
    allowed = ("forward_direct", "no_forward_direct", "unknown") if protocol == "fixed" else REPORTS
    meanings = {
        "forward_direct": "A directly causes B", "reverse_direct": "B directly causes A",
        "forward_ancestry": "A is an ancestor of B, whether direct or mediated",
        "reverse_ancestry": "B is an ancestor of A, whether direct or mediated",
        "no_forward_direct": "an A to B direct edge is explicitly excluded",
        "no_reverse_direct": "a B to A direct edge is explicitly excluded",
        "no_forward_ancestry": "any A to B directed path is explicitly excluded",
        "no_reverse_ancestry": "any B to A directed path is explicitly excluded",
        "unrelated": "directed paths in BOTH directions are explicitly excluded",
        "unknown": "insufficient applicable evidence",
    }
    order = tuple(reversed(allowed)) if reversed_order else allowed
    instructions = (
        "Assess only the supplied evidence about the named expression variables in the target population. "
        "Forward always means A to B. Directness is relative to the stated graph variables. "
        "A mediated effect does not exclude an additional direct edge. Protein binding, shared effects on a third "
        "variable, correlation, and missing evidence do not establish a causal expression relation or its absence. "
        "Evidence from another population without established transportability is insufficient. "
        "Do not use background knowledge. A false statement and an unsupported statement are different. "
        + ("Assess whether a direct A to B edge is supported, explicitly contradicted, or unresolved. " if protocol == "fixed" else
           "Extract the most specific supported statement. Prefer a supported direct effect to its implied ancestry; "
           "otherwise preserve supported ancestry. Prefer a supported positive direction to an implied reverse exclusion. ")
        + "Return only a JSON object with relation, confidence and quote. Confidence is a number from 0.5 to 1 "
        "for a retained statement, and null for unknown. Quote one exact excerpt from the evidence; use an empty "
        "quote for unknown. Do not put the answer inside a Markdown code block. Allowed relation values follow.\n"
        + "\n".join(f"{key} means {meanings[key]}." for key in order)
    )
    return ({"role": "system", "content": instructions},
            {"role": "user", "content": f"Target context\n{context}\nEvidence\n{evidence}"})
