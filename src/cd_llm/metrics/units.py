"""Explicit conversion of archived ED/ES reports to structural-error counts.

This module has no inference dependencies. Unknown artifact units must be
provided by the caller; numerical magnitudes are never used to infer units.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

METRICS = {"expected_distance", "energy_score", "expected_endpoint_distance",
           "posterior_expected_distance", "ed", "es", "negative_transfer_ed", "negative_transfer_es"}
STATS = {"mean", "se", "sd", "std", "low", "high", "lower", "upper", "delta",
         "difference", "ci_low", "ci_high", "ci_lower", "ci_upper",
         "seed_differences", "values", "estimate", "median", "min", "max",
         "delta_mean", "delta_se", "delta_sd", "claim_fuse_mean", "data_only_mean",
         "paired_difference", "paired_95_interval", "mean_difference", "interval95"}


def distance_factor(node_count: int, regime: str = "dag") -> int:
    if node_count < 0 or regime not in ("dag", "mag"):
        raise ValueError("invalid node count or regime")
    return math.comb(node_count, 2) * (2 if regime == "mag" else 1)


def is_distance_field(key: str) -> bool:
    return key in METRICS or any(
        key.startswith(m + "_") and key[len(m) + 1:] in STATS
        for m in METRICS
    )


def convert_metrics(value, factor: float, *, source_scale: str, _context=False):
    """Copy a report, scaling only identified ED/ES values and statistics.

    The explicit source scale makes conversion idempotent. Metadata, counts,
    confidence, PD, normalized support diagnostics, and F1 are untouched.
    """
    if source_scale not in ("normalized", "unnormalized"):
        raise ValueError("source_scale must be normalized or unnormalized")
    if isinstance(value, dict):
        declared = value.get("metric_scale")
        if declared is not None and declared != source_scale:
            raise ValueError("declared metric scale disagrees with source_scale")
        metric_row = value.get("metric") in METRICS
        return {k: ("unnormalized" if k == "metric_scale" else
                    convert_metrics(v, factor, source_scale=source_scale,
                                    _context=is_distance_field(k) or
                                    ((_context or metric_row) and k in STATS)))
                for k, v in value.items()}
    if isinstance(value, list):
        return [convert_metrics(v, factor, source_scale=source_scale,
                                _context=_context) for v in value]
    if not _context or value is None or isinstance(value, bool) or source_scale == "unnormalized":
        return value
    if isinstance(value, str) and value in ("", "NA", "N/A", "nan"):
        return value
    result = float(value) * factor
    return str(result) if isinstance(value, str) else result


def read_metric_csv(path: Path, *, node_count: int, regime: str = "dag",
                    legacy_normalized: bool = False) -> list[dict]:
    """Read raw units, accepting unmarked archives only by explicit opt-in."""
    metadata_path = path.with_suffix(path.suffix + ".metrics.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["node_count"] != node_count or metadata["regime"] != regime:
            raise ValueError("metric metadata disagrees with requested graph")
        scale = metadata["metric_scale"]
    elif legacy_normalized:
        scale = "normalized"
    else:
        raise ValueError(f"missing metric units for {path}")
    with path.open(newline="") as handle:
        return convert_metrics(list(csv.DictReader(handle)), distance_factor(node_count, regime),
                               source_scale=scale)


def metric_metadata(node_count: int, regime: str = "dag") -> dict:
    return dict(metric_scale="unnormalized", node_count=node_count, regime=regime,
                legacy_normalization_factor=distance_factor(node_count, regime))
