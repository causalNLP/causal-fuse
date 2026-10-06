#!/usr/bin/env python3
"""Generate ``claim_robustness.pdf`` from an aggregated robustness run."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Sequence

if "MPLCONFIGDIR" not in os.environ:
    _MATPLOTLIB_CONFIG = Path(tempfile.gettempdir()) / "cd_llm_matplotlib"
    _MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MATPLOTLIB_CONFIG)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from cd_llm.metrics.units import read_metric_csv


CONDITIONS = (
    ("Overconfident", "40%", "iid_overconfident_40_as_70"),
    ("Calibrated", "70%", "iid_calibrated_70"),
    ("Underconfident", "100%", "iid_underconfident_100_as_70"),
)
METRICS = (
    ("expected_distance", "ED $\\downarrow$"),
    ("energy_score", "ES $\\downarrow$"),
    ("adjacency_f1", "Adjacency F1 $\\uparrow$"),
    ("orientation_f1", "Orientation F1 $\\uparrow$"),
)
COLORS = {
    "iid_overconfident_40_as_70": "#B8C1CA",
    "iid_calibrated_70": "#6F9FC2",
    "iid_underconfident_100_as_70": "#2E6F9E",
}
HATCHES = {
    "iid_overconfident_40_as_70": "",
    "iid_calibrated_70": "...",
    "iid_underconfident_100_as_70": "///",
}
DISPLAY_BUDGET = 10


def _read(path: Path) -> list[dict[str, str]]:
    return read_metric_csv(path, node_count=20, legacy_normalized=True)


def _row(
    rows: Sequence[dict[str, str]], sample_size: int, condition: str, budget: int
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if int(row["sample_size"]) == sample_size
        and row["condition_id"] == condition
        and int(row["budget"]) == budget
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one summary row for n={sample_size}/{condition}/budget={budget}"
        )
    return matches[0]


def generate(summary_path: Path, output_path: Path) -> Path:
    rows = _read(summary_path)
    plt.rcParams.update(
        {
            "axes.edgecolor": "#40454A",
            "axes.labelcolor": "#25282B",
            "axes.linewidth": 0.65,
            "axes.titleweight": "bold",
            "font.family": "serif",
            "font.serif": ("Times New Roman", "Times", "DejaVu Serif"),
            "font.size": 7.6,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "text.color": "#25282B",
            "xtick.color": "#353A3F",
            "ytick.color": "#353A3F",
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(7.15, 3.45), sharex=True, sharey="col")
    positions = np.arange(len(CONDITIONS), dtype=float)

    for row_index, sample_size in enumerate((200, 1_000)):
        baseline = _row(rows, sample_size, "data_only_pc_ges", 0)
        for column_index, (metric, title) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            selected = [
                _row(rows, sample_size, condition_id, DISPLAY_BUDGET)
                for _, _, condition_id in CONDITIONS
            ]
            means = [float(row[f"{metric}_mean"]) for row in selected]
            errors = [float(row[f"{metric}_se"]) for row in selected]
            bars = axis.bar(
                positions,
                means,
                yerr=errors,
                color=[COLORS[condition_id] for _, _, condition_id in CONDITIONS],
                edgecolor="#26333D",
                linewidth=0.45,
                width=0.48,
                error_kw={
                    "ecolor": "#25282B",
                    "elinewidth": 0.7,
                    "capsize": 1.8,
                    "capthick": 0.7,
                },
                zorder=3,
            )
            for bar, (_, _, condition_id) in zip(bars, CONDITIONS):
                bar.set_hatch(HATCHES[condition_id])
            axis.axhline(
                float(baseline[f"{metric}_mean"]),
                color="#4F555A",
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )
            axis.grid(axis="y", color="#D9DEE2", linewidth=0.55, alpha=0.9, zorder=0)
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.tick_params(axis="both", labelsize=6.8, length=2.2, width=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                axis.set_title(title, fontsize=8.2, pad=5.0)
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xticks(
                    positions,
                    [accuracy for _, accuracy, _ in CONDITIONS],
                )
                axis.tick_params(axis="x", labelsize=6.8, pad=2.5)
            if metric in {"adjacency_f1", "orientation_f1"}:
                axis.set_ylim(0.0, 1.0)

    figure.text(0.012, 0.695, r"$n=200$", rotation=90, ha="center", va="center", fontsize=8.2, weight="bold")
    figure.text(0.012, 0.285, r"$n=1000$", rotation=90, ha="center", va="center", fontsize=8.2, weight="bold")

    legend_handles = [
        Patch(
            facecolor=COLORS[condition_id],
            edgecolor="#26333D",
            linewidth=0.45,
            hatch=HATCHES[condition_id],
            label=label,
        )
        for label, _, condition_id in CONDITIONS
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#4F555A",
            linestyle="--",
            linewidth=0.8,
            label="Data-only PC+GES",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.005),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.35,
    )
    figure.text(0.525, 0.085, "Actual accuracy", ha="center", va="center", fontsize=7.8)
    figure.subplots_adjust(left=0.076, right=0.995, bottom=0.22, top=0.88, wspace=0.34, hspace=0.24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Creator": "code/figures/claim_robustness.py"},
    )
    plt.close(figure)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.summary.with_name("claim_robustness.pdf")
    print(generate(arguments.summary, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
