"""Combine saved figure summaries into one all-method plotting CSV per dataset."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NEUROPATHIC_SOURCE = (
    ROOT / "results/20260921T220659Z/neuropathic_performance_cost/summary.csv"
)
ALARM_SOURCE = (
    ROOT / "results/20260923T143202Z/alarm_performance_cost/plotted_points.csv"
)
ALARM_SCP_SOURCE = (
    ROOT / "results/20260924T014634Z/scp_plot_data/alarm_scp.csv"
)

FIELDS = (
    "dataset",
    "method",
    "source_method",
    "sample_size",
    "strategy",
    "group",
    "budget",
    "seeds",
    "expected_distance_mean",
    "expected_distance_se",
    "approx_runtime_seconds_mean",
    "approx_runtime_seconds_se",
    "total_tokens_mean",
    "total_tokens_se",
    "input_tokens_mean",
    "input_tokens_se",
    "output_tokens_mean",
    "output_tokens_se",
    "response_events_mean",
    "response_events_se",
    "source_file",
)
METRICS = FIELDS[8:20]
ALARM_METHODS = {
    "Claim-Fuse": "Claim-Fuse",
    "bfs": "BFS+Pearson",
    "long": "PC+Long-Srisk",
    "order": "Causal Order+PC",
    "gpc": "gPC-Guess",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_row(source, dataset, method, source_method, source_file):
    assert all(source.get(metric) not in (None, "") for metric in METRICS)
    return {
        "dataset": dataset,
        "method": method,
        "source_method": source_method,
        "sample_size": source["sample_size"],
        "strategy": source["strategy"],
        "group": source["group"],
        "budget": source["budget"],
        "seeds": source["seeds"],
        **{metric: source[metric] for metric in METRICS},
        "source_file": str(source_file.relative_to(ROOT)),
    }


def neuropathic_rows():
    source = read_csv(NEUROPATHIC_SOURCE)
    assert len(source) == 58
    rows = [make_row(row, "neuropathic_pain",
                     "Claim-Fuse" if row["method"] == "SMC-Active" else row["method"],
                     row["method"], NEUROPATHIC_SOURCE) for row in source]
    assert {row["method"] for row in rows} == {
        "Claim-Fuse", "gPC-Guess", "PC+Long-Srisk", "BFS+Pearson",
        "Causal Order+PC", "SCP Pattern 2",
    }
    return rows


def alarm_rows():
    source = read_csv(ALARM_SOURCE)
    scp = read_csv(ALARM_SCP_SOURCE)
    assert len(source) == 56 and len(scp) == 2
    rows = [make_row(row, "alarm", ALARM_METHODS[row["method"]],
                     row["method"], ALARM_SOURCE) for row in source]
    for row in scp:
        native = dict(row, strategy="native", group="", budget="0")
        rows.append(make_row(native, "alarm", "SCP Pattern 2", "scp",
                             ALARM_SCP_SOURCE))
    assert len(rows) == 58
    assert {row["method"] for row in rows} == {
        "Claim-Fuse", "gPC-Guess", "PC+Long-Srisk", "BFS+Pearson",
        "Causal Order+PC", "SCP Pattern 2",
    }
    return rows


def validate(rows):
    assert len(rows) == 58
    keys = [(row["sample_size"], row["strategy"], row["group"],
             row["method"], row["budget"]) for row in rows]
    assert len(keys) == len(set(keys))
    for sample_size in ("200", "1000"):
        native = [row for row in rows if row["sample_size"] == sample_size
                  and row["strategy"] == "native"]
        assert len(native) == 5
        for strategy in ("stated", "consistency"):
            for group in ("group_i", "group_ii", "all"):
                claim = [row for row in rows if row["sample_size"] == sample_size
                         and row["strategy"] == strategy and row["group"] == group]
                assert sorted(int(row["budget"]) for row in claim) == [10, 20, 50, 100]
                assert all(int(row["seeds"]) == 10 for row in claim)
    for row in rows:
        assert abs(float(row["input_tokens_mean"])
                   + float(row["output_tokens_mean"])
                   - float(row["total_tokens_mean"])) < 1e-6


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    neuropathic = neuropathic_rows()
    alarm = alarm_rows()
    validate(neuropathic)
    validate(alarm)
    write_csv(args.output_dir / "neuropathic_pain_all_methods.csv", neuropathic)
    write_csv(args.output_dir / "alarm_all_methods.csv", alarm)
    print(f"Wrote {len(neuropathic)} neuropathic and {len(alarm)} ALARM rows")


if __name__ == "__main__":
    main()
