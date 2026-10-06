"""Export matched RQ2 claim-pool trajectories for local figure construction."""

from __future__ import annotations

import csv
import itertools
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from cd_llm.metrics.units import read_metric_csv  # noqa: E402


RESULTS = ROOT / "results"
GROUPS = {"group_i": "local", "group_ii": "ancestral", "all": "all"}
FIELDS = (
    "dataset",
    "sample_size",
    "model",
    "confidence_strategy",
    "expert_accuracy",
    "claim_group",
    "attempted_queries",
    "ed_mean",
    "ed_se",
    "replicate_count",
)


def make_row(source: dict, *, dataset: str, model: str,
             confidence_strategy: str, expert_accuracy: str,
             replicate_field: str) -> dict:
    mean = float(source["expected_distance_mean"])
    se = float(source["expected_distance_se"])
    if not math.isfinite(mean) or not math.isfinite(se) or mean < 0 or se < 0:
        raise ValueError(f"invalid ED result: {source}")
    count = int(source[replicate_field])
    if count != 10:
        raise ValueError(f"expected ten replicates, got {count}: {source}")
    return dict(
        dataset=dataset,
        sample_size=int(source["sample_size"]),
        model=model,
        confidence_strategy=confidence_strategy,
        expert_accuracy=expert_accuracy,
        claim_group=GROUPS[source["group"]],
        attempted_queries=int(source["budget"]),
        ed_mean=mean,
        ed_se=se,
        replicate_count=count,
    )


def validate(rows: list[dict], expected: set[tuple], key_fields: tuple[str, ...]) -> None:
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate RQ2 conditions")
    if set(keys) != expected:
        raise ValueError(f"RQ2 conditions differ: missing={expected - set(keys)}, extra={set(keys) - expected}")


def synthetic_rows() -> list[dict]:
    path = RESULTS / "20260917T004634Z" / "synthetic_pc_grid_ablation" / "summary.csv"
    source = read_metric_csv(path, node_count=20, legacy_normalized=True)
    rows = [make_row(row, dataset="synthetic", model="simulated expert",
                     confidence_strategy="simulated", expert_accuracy=row["accuracy"],
                     replicate_field="scm_count") for row in source]
    key = ("sample_size", "expert_accuracy", "claim_group", "attempted_queries")
    expected = set(itertools.product((200, 1000), ("0.6", "0.8", "1.0"),
                                     GROUPS.values(), (5, 20, 50)))
    validate(rows, expected, key)
    return sorted(rows, key=lambda row: tuple(row[field] for field in key))


def clinical_rows(path: Path, *, dataset: str, node_count: int,
                  method_filter: str | None = None) -> list[dict]:
    source = read_metric_csv(path, node_count=node_count)
    rows = [make_row(row, dataset=dataset, model="Qwen3.5-9B",
                     confidence_strategy=row["strategy"], expert_accuracy="",
                     replicate_field="seeds") for row in source
            if (method_filter is None or row["method"] == method_filter)
            and int(row["sample_size"]) in (200, 1000)
            and row["strategy"] in ("stated", "consistency")
            and row["group"] in GROUPS
            and int(row["budget"]) in (10, 20, 50, 100)]
    key = ("sample_size", "confidence_strategy", "claim_group", "attempted_queries")
    expected = set(itertools.product((200, 1000), ("stated", "consistency"),
                                     GROUPS.values(), (10, 20, 50, 100)))
    validate(rows, expected, key)
    return sorted(rows, key=lambda row: tuple(row[field] for field in key))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    datasets = {
        "synthetic.csv": synthetic_rows(),
        "neuropathic_pain.csv": clinical_rows(
            RESULTS / "20260919T005552Z" / "neuropathic_performance_cost" / "summary.csv",
            dataset="neuropathic_pain", node_count=20, method_filter="SMC-Active"),
        "alarm.csv": clinical_rows(
            RESULTS / "20260922T142041Z" / "alarm_claim_pools_consistency" / "combined_summary.csv",
            dataset="alarm", node_count=37),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS / stamp / "rq2_figure_data"
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in datasets.items():
        write_csv(output / filename, rows)
        print(f"{output / filename}: {len(rows)} rows")


if __name__ == "__main__":
    main()
