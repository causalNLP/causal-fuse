# Table 1 baseline comparison status

Table 1 uses the approved [shared pairwise expert protocol](synthetic_pairwise_expert_benchmark.md). Its current publication is `20260921T020934Z/synthetic_table1_publication`: seven methods retain their results from `20260921T010550Z`, and ECA uses the authors' original implementation from the authorized confidence-cap run `20260921T015754Z`. Every method has complete ten-SCM results at both sample sizes. No other method's numbers were changed.

The user approved twenty uniformly selected unordered pairs per SCM, one categorical answer per pair, reused across all methods and both sample sizes. CaMML receives native exact 0/1 directed priors. ECA receives forty signed directed assertions encoding those twenty answers, with native assertion confidence capped at 0.99. Marg receives native categorical opinions from one expert with the approved good-population accuracy prior. No coarsened-opinion extension is used. Only complete ten-SCM aggregates are published.

Raw answers have confidence 1.0. FS and SMC cap their internal soft confidence to 0.99. The separate ECA cap was explicitly authorized; CaMML and Marg inputs remain unchanged. ECA retains the authors' original solver, zero initialization, objective, and validation settings. See the [confidence-cap diagnostics](synthetic_eca_confidence_cap.md).

Reproduce the current table without rerunning experiments:

```bash
PYTHONPATH=src ../.venv/bin/python -B scripts/report_synthetic_pairwise_expert_benchmark.py --run-id 20260921T010550Z --eca-run-id 20260921T015754Z --publication-run-id 20260921T020934Z
```

The reporter verifies matched data and facts, source and result hashes, complete cohorts, and ECA aggregates. The publication manifest records both source runs and hashes of the generated outputs.

The unapproved experiment run `20260920T222103Z/synthetic_matched_expert_baselines` and publication `20260920T235159Z/synthetic_expert_table` are withdrawn from paper use. The earlier preparation attempts `20260920T221421Z` and `20260920T221704Z` use the same unapproved protocol. Saved inputs, predictions, manifests, reports, and logs remain intact for provenance, with adjacent withdrawal notices. The newly added experiment, categorical-likelihood extension, submission scripts, and conditional-cohort reporting and publication code have been removed. Their performance claims are not retained in the paper.

Correction artifacts are recorded under `20260921T001231Z` in the workspace results and log directories. The replacement run is recorded under `20260921T010550Z`; it preserves the accepted cohort and implements the explicitly agreed native-input protocol.
