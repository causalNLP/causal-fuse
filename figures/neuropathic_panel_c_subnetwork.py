#!/usr/bin/env python3
"""Draw the canonical 20-node neuropathic Panel C target DAG."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from cd_llm.core.finite_support import graph_edges
from cd_llm.experiments.neuropathic_causal_order_subnetwork import (
    SUBNETWORK_EDGE_COUNT,
    SUBNETWORK_NODE_COUNT,
    build_subnetwork_spec,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "paper" / "figs" / "neuropathic_panel_c_subnetwork.pdf"
)


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _node_label(name: str) -> str:
    if name.startswith("DLS "):
        return name.replace("DLS ", "DLS\\n").replace("-", "–")
    if name == "Kraniocervikal ledskada":
        return "Kraniocervikal\\nledskada"
    if name.endswith(" Radikulopati"):
        return name.removesuffix(" Radikulopati").replace(" ", "\\n")
    return name


def _dot_source() -> str:
    specification = build_subnetwork_spec()
    edges = [tuple(edge) for edge in graph_edges(specification.target)["directed"]]
    if specification.node_count != SUBNETWORK_NODE_COUNT:
        raise ValueError(
            f"expected {SUBNETWORK_NODE_COUNT} nodes, found {specification.node_count}"
        )
    if len(edges) != SUBNETWORK_EDGE_COUNT:
        raise ValueError(f"expected {SUBNETWORK_EDGE_COUNT} edges, found {len(edges)}")

    causes = [index for index, original in enumerate(specification.original_indices) if original < 27]
    radiculopathies = [
        index
        for index, original in enumerate(specification.original_indices)
        if 27 <= original < 79
    ]
    symptoms = [
        index for index, original in enumerate(specification.original_indices) if original >= 79
    ]
    if (len(causes), len(radiculopathies), len(symptoms)) != (7, 12, 1):
        raise ValueError("unexpected Panel C layer sizes")

    lines = [
        "digraph neuropathic_panel_c {",
        "  graph [rankdir=TB, bgcolor=\"transparent\", margin=0, pad=0.04, nodesep=0.12, ranksep=0.62, outputorder=edgesfirst, splines=spline];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Times-Roman\", fontsize=8.6, margin=\"0.06,0.035\", height=0.38, penwidth=0.75];",
        "  edge [color=\"#66737C\", penwidth=0.78, arrowsize=0.52, arrowhead=normal];",
        "  cause_label [label=\"Etiology\", shape=plaintext, fontname=\"Times-Bold\", fontsize=9.0, fontcolor=\"#30353A\"] ;",
        "  radic_label [label=\"Radiculopathy\", shape=plaintext, fontname=\"Times-Bold\", fontsize=9.0, fontcolor=\"#30353A\"] ;",
        "  symptom_label [label=\"Symptom\", shape=plaintext, fontname=\"Times-Bold\", fontsize=9.0, fontcolor=\"#30353A\"] ;",
    ]
    for index, name in enumerate(specification.node_names):
        original = specification.original_indices[index]
        if original < 26:
            attributes = 'fillcolor="#D8E8F2", color="#527A95", fontcolor="#24343E"'
        elif original == 26:
            attributes = 'fillcolor="#E3E7EA", color="#67727A", fontcolor="#30363A"'
        elif original < 79:
            attributes = 'fillcolor="#F1F5F7", color="#7792A3", fontcolor="#29343B", width=0.38'
        else:
            attributes = 'fillcolor="#2E6F9E", color="#234F6D", fontcolor="white", fontname="Times-Bold", penwidth=0.95'
        lines.append(f"  n{index} [label={_quoted(_node_label(name))}, {attributes}];")

    lines.extend(
        [
            "  { rank=same; cause_label; " + "; ".join(f"n{index}" for index in causes) + "; }",
            "  { rank=same; radic_label; " + "; ".join(f"n{index}" for index in radiculopathies) + "; }",
            "  { rank=same; symptom_label; " + "; ".join(f"n{index}" for index in symptoms) + "; }",
            "  cause_label -> radic_label -> symptom_label [style=invis, weight=100];",
            "  " + " -> ".join(f"n{index}" for index in causes) + " [style=invis, weight=100];",
            "  " + " -> ".join(f"n{index}" for index in radiculopathies) + " [style=invis, weight=100];",
        ]
    )
    symptom = symptoms[0]
    for source, target in edges:
        if target == symptom:
            style = 'color="#2E6F9E", penwidth=0.95'
        else:
            style = 'color="#66737C", penwidth=0.78'
        lines.append(f"  n{source} -> n{target} [{style}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        raw_output = Path(temporary_directory) / "subnetwork.pdf"
        subprocess.run(
            ["dot", "-Tpdf", "-o", str(raw_output)],
            input=_dot_source(),
            text=True,
            check=True,
        )
        subprocess.run(
            ["qpdf", "--force-version=1.5", str(raw_output), str(output)],
            check=True,
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Graphviz did not produce a nonempty PDF: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    print(generate(arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
