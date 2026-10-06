"""Plot all four recovery metrics for the PC grid soft-expert ablation."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cd_llm.metrics.units import read_metric_csv


def generate(root: Path):
    rows = read_metric_csv(root / "summary.csv", node_count=20, legacy_normalized=True)
    controls = read_metric_csv(root / "baseline.csv", node_count=20, legacy_normalized=True)
    if len(rows) != 54 or len(controls) != 20:
        raise ValueError("plot requires the complete matched study")
    output = root / "synthetic_pc_grid_ablation.pdf"
    metrics = (("expected_distance", "Expected distance (lower is better)"),
               ("energy_score", "Energy score (lower is better)"),
               ("adjacency_f1", "Adjacency F1 (higher is better)"),
               ("orientation_f1", "Orientation F1 (higher is better)"))
    with PdfPages(output) as pdf:
        for metric, title in metrics:
            fig, axes = plt.subplots(2, 3, figsize=(10.8, 6), sharex=True, sharey="row")
            for row_index, n in enumerate((200, 1000)):
                baseline = [float(r[metric]) for r in controls if int(r["sample_size"]) == n]
                for col_index, accuracy in enumerate((0.6, 0.8, 1.0)):
                    ax = axes[row_index, col_index]
                    for group, label, color in (("group_i", "Group I", "#2878b5"),
                                                 ("group_ii", "Group II", "#e07b39"),
                                                 ("all", "All", "#3c995a")):
                        selected = sorted((r for r in rows if int(r["sample_size"]) == n
                                           and float(r["accuracy"]) == accuracy and r["group"] == group),
                                          key=lambda r: int(r["budget"]))
                        if [int(r["budget"]) for r in selected] != [5, 20, 50]:
                            raise ValueError("missing or duplicate plotting cell")
                        ax.errorbar([5, 20, 50], [float(r[metric + "_mean"]) for r in selected],
                                    yerr=[float(r[metric + "_se"]) for r in selected],
                                    marker="o", capsize=3, label=label, color=color)
                    ax.axhline(sum(baseline) / len(baseline), ls="--", color="0.45", label="Initialization")
                    ax.set_title(f"n={n}; accuracy={accuracy:.1f}; confidence={min(accuracy, .99):.2f}", fontsize=10)
                    ax.set_xticks([5, 20, 50])
                    ax.grid(alpha=.2)
                    if row_index == 1:
                        ax.set_xlabel("Queries")
            fig.suptitle(title + "\nMean ± one SE over ten SCM/data pairs", fontsize=12)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4)
            fig.tight_layout(rect=(0, .055, 1, .93))
            pdf.savefig(fig)
            plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    print(generate(parser.parse_args().result_root))
