# Shared pairwise expert benchmark

This documents the base experiment `20260921T010550Z`. The current Table 1
retains seven methods from this run and replaces only ECA with the authors'
original implementation at the explicitly authorized confidence cap of 0.99
from `20260921T015754Z`. The exact-endpoint local ECA settings and failures
below describe the historical base run, not the current ECA row. See
[current publication and reproduction instructions](synthetic_matched_expert_baselines.md).

The shared protocol uses ten fixed 20-node SCMs
`1,2,7,12,15,34,36,40,45,47`, with the archived nested 200/1000 observations
from `20260816T145618Z`. Each dataset is standardized once, separately.

One uniformly sampled sequence of 20 unique unordered pairs is selected per
SCM before looking up truth, without balancing answer classes. Its categorical
answers are reused across methods and sample sizes. One answer is one external
fact, regardless of how many native entries encode it. Answers have raw
confidence 1.0 and state either left-to-right, right-to-left, or no direct edge
in either direction. Absence does not imply absence of a causal path.

| Raw answer | FS/SMC | CaMML/ECA directed probabilities (forward, reverse) | Marg |
|---|---|---|---|
| Forward edge | Positive directed claim | (1, 0) | Forward categorical opinion |
| Reverse edge | Positive reversed directed claim | (0, 1) | Reverse categorical opinion |
| No direct edge | One negative adjacency claim | (0, 0) | Absent categorical opinion |

Only FS/SMC cap confidence internally to 0.99, contributing likelihood factors
0.99/0.01. Native CaMML/ECA priors retain exact endpoints; evidence adapters are
not used. Unqueried pairs retain each method's defaults. Marg receives one
synthetic expert identity and the approved population accuracy prior, not a
conversion of the raw confidence.

FS and SMC reuse every ordered particle slot, weight, duplicate, and inference
seed from the PC-only initialization in
`20260920T215421Z/synthetic_passive_claim_budgets`. PC is stable Fisher-Z at
alpha 0.05, with 384 particle slots, no grid, and no initialization moves.
FS reweights its fixed empirical support. SMC uses fixed-order passive updates,
Gaussian BIC, maximum 12 parents, ESS threshold 0.5, two rejuvenation proposals
per particle after resampling and at the final twentieth fact, guided mixture
0.5, a 256-state cache, and uniform-pair ordinary proposals.

CaMML uses its native local pair priors with four chains, 25,000 iterations,
5,000 burn-in, thinning 10, seven-parent limit, temperature 1, default arc
probability 0.5, and 10,000 initialization attempts. Its seed is fixed per dataset.
The continuous observational score is the existing Gaussian-BIC adaptation.

Marg uses native marginal categorical opinion scoring and empty-DAG greedy
search, improvement tolerance 1e-12, without a parent limit. The continuous
Gaussian-BIC local score explicitly allows 19 parents rather than inheriting
the shared score's 12-parent default. The ten good-population rows of Table 3
in [Amirkhani et al.](https://www.cs.ru.nl/P.Lucas/BN/pami-hossein2017.pdf)
average to (0.61, 0.205, 0.775). Equation (18) with concentration 10 yields
Dirichlet(6.1, 2.05, 1.85) and Beta(7.75, 2.25). This models expert accuracy;
it is not a confidence-1 likelihood or an estimate from the benchmark truth.

ECA uses its existing paper backend, sigma 1, L1 penalty 0.05, threshold 0.30,
100 iterations, acyclicity tolerance 1e-8, maximum rho 1e16, multiplier 10,
and no additional standardization. Its native warm start and numerical
machine-tiny likelihood floor are retained, with no smoothing of input priors.
Success requires finite output, a DAG, and acyclicity within tolerance;
inner optimizer status is recorded separately. No cycle repair is applied.

PC/GES/NOTEARS predictions are reused unchanged. Metrics are ED in structural
error counts, adjacency F1, and orientation F1. ED averages over the FS/SMC/CaMML
distributions; F1 uses the canonical MAP graph. Marg/ECA/NOTEARS are point DAGs;
PC/GES retain the existing raw partial-graph scoring convention.

Use `python -B -m cd_llm experiment synthetic-pairwise-expert-benchmark`:

1. `--stage prepare --run-id TIMESTAMP` validates the archive and freezes facts,
   native inputs, seeds, configurations, and implementation hashes.
2. `--stage compute --run-id TIMESTAMP --graph-index INDEX` computes both sample
   sizes for one SCM. Each method/sample result is saved separately. Verified
   existing records are reused without rerunning completed fits.
3. `--stage aggregate --run-id TIMESTAMP` requires all 100 new evaluations,
   validates predictions and metrics, and includes the 60 archived baseline
   evaluations. Failed fits remain explicit records.
4. `python -B scripts/report_synthetic_pairwise_expert_benchmark.py
   --run-id TIMESTAMP` renders the verified report and table inside the result
   directory.

Run a full-configuration SCM 1 pilot, inspect its inputs and diagnostics, then
retain it while computing the other nine SCMs. Infrastructure retries use the
same inputs, settings, and seeds. Algorithmic failures do not authorize tuning.

Means and sample SEs require all ten SCMs for each method/sample size. Otherwise
the three cells are dashes, with completion counts and failure reasons exposed.
No successful-subset averages or common-success cohort are used. SEs are
conditional on this single pair draw and the fixed inference seeds. Historical
binary-query trajectories and withdrawn experiments remain separate artifacts.

The completed run is `20260921T010550Z/synthetic_pairwise_expert_benchmark`.
All 100 new evaluations are recorded, together with 60 unchanged data-only
evaluations. CaMML, Marg, FS, and SMC each succeed on all ten SCMs at both sizes.
ECA succeeds on 5/10 at n=200 and 7/10 at n=1000; seven fits miss the acyclicity
tolerance and one overflows in the matrix exponential. Its aggregate cells are
dashes. `audit.json` verifies exact FS replay, all final SMC updates, unchanged
baseline means/SEs, and preservation of 318 historical artifacts. The report
lists every failed SCM. The pilot is retained in the completed run.
