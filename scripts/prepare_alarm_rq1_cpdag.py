"""Archive the ALARM RQ1 table with CPDAG-based gPC recovery metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from cd_llm.experiments.common.artifacts import sha256


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(source: Path, gpc: Path, output: Path) -> None:
    old_seed = read_csv(source / "per_seed.csv")
    old_summary = read_csv(source / "summary.csv")
    gpc_seed = read_csv(gpc / "per_seed.csv")
    gpc_summary = read_csv(gpc / "summary.csv")
    provenance = json.loads((gpc / "provenance.json").read_text())
    assert provenance["per_seed_sha256"] == sha256(gpc / "per_seed.csv")
    assert provenance["summary_sha256"] == sha256(gpc / "summary.csv")
    counts = {int(row["sample_size"]): int(row["complete_seeds"]) for row in gpc_summary}
    assert counts in ({200: 10, 1000: 3}, {200: 10, 1000: 6})
    assert len(gpc_seed) == 10 + counts[1000] and len(gpc_summary) == 2
    assert {int(row["sample_size"]): {int(x["seed"]) for x in gpc_seed if x["sample_size"] == row["sample_size"]}
            for row in gpc_summary} == {200: set(range(1000, 1010)),
                                        1000: {1001, 1003, 1004} if counts[1000] == 3
                                        else {1001, 1003, 1004, 1005, 1006, 1007}}

    seed_rows = [row for row in old_seed if row["method"] != "gpc"]
    summary_rows = [row for row in old_summary if row["method"] != "gpc"]
    for row in gpc_seed:
        seed_rows.append({"method": "gpc", "strategy": "native", "sample_size": row["sample_size"],
                          "seed": row["seed"], "expected_distance": row["expected_distance"],
                          "orientation_f1": row["orientation_f1"]})
    for row in gpc_summary:
        n = int(row["sample_size"])
        group = [item for item in gpc_seed if int(item["sample_size"]) == n]
        result = {"method": "gpc", "strategy": "native", "sample_size": n,
                  "complete_seeds": len(group), "paired_ten": n == 200}
        for metric in ("expected_distance", "orientation_f1"):
            values = [float(item[metric]) for item in group]
            mean = statistics.mean(values)
            se = statistics.stdev(values) / len(values) ** 0.5
            assert abs(mean - float(row[metric + "_mean"])) < 1e-10
            assert abs(se - float(row[metric + "_se"])) < 1e-10
            result[metric + "_mean"] = mean
            result[metric + "_se"] = se
        summary_rows.append(result)
    assert len(seed_rows) == len(old_seed) - (6 - counts[1000])
    assert len(summary_rows) == len(old_summary)
    seed_rows.sort(key=lambda row: (row["method"], row["strategy"], int(row["sample_size"]), int(row["seed"])))
    summary_rows.sort(key=lambda row: (row["method"], row["strategy"], int(row["sample_size"])))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_seed.csv", seed_rows)
    write_csv(output / "summary.csv", summary_rows)
    (output / "provenance.json").write_text(json.dumps({
        "source_table": str(source), "source_table_summary_sha256": sha256(source / "summary.csv"),
        "source_table_per_seed_sha256": sha256(source / "per_seed.csv"),
        "gpc_cpdag": str(gpc), "gpc_provenance_sha256": sha256(gpc / "provenance.json"),
        "gpc_complete_seeds": {"n200": 10, "n1000": counts[1000]},
        "metric_normalized": False,
        "summary_sha256": sha256(output / "summary.csv"),
        "per_seed_sha256": sha256(output / "per_seed.csv"),
    }, indent=2) + "\n")
    print(output / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--gpc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.gpc.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
