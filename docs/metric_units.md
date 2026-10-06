# ED and ES units

ED and ES use unnormalized structural error counts. A DAG edge addition,
deletion, or reversal contributes one; an unresolved direction in a partial
DAG contributes one half. MAG errors count endpoint mismatches, so deleting
an edge contributes two. Both terms of ES use the same distance units.

`distribution_metrics` and `point_prediction_metrics` default to these units.
Pass `normalized=True` explicitly to reproduce legacy ED/ES. Normalized
distance helpers, neighborhood radii, support diagnostics, PD and mode distance
retain their existing semantics. F1 and confidence metrics are unchanged.

For a fixed graph size, legacy DAG ED/ES and their standard deviations,
standard errors, absolute differences and interval endpoints are multiplied
by `choose(p, 2)`. MAG values are multiplied by `2 * choose(p, 2)`. Convert
individual records before aggregating datasets with different graph sizes.
Relative changes and within-dataset rankings are unaffected.

CSV metadata uses a `.csv.metrics.json` sidecar; JSON reports carry
`metric_scale` and provenance. `metrics.units.read_metric_csv` checks these
units and requires explicit opt-in for unmarked normalized archives.
