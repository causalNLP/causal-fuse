# Original-code ECA on the Table 1 inputs

This report-only evaluation calls the authors' linear `myECA_prior` class at
revision `c98002720dca8c170ba0f3d25ac3681e5def0bdc`. Every tracked Python file is
verified against that commit. Missing tracked bytecode caches are irrelevant
to source verification. The original Python source is not edited.

The reference is `20260921T010550Z/synthetic_pairwise_expert_benchmark`: ten
20-node SCMs, the same nested 200/1000 observations, and the same twenty saved
pairwise facts per SCM. The separately standardized matrices are hashed before
fitting; the authors' internal centering is retained.

Each directed answer becomes one required edge and a forbidden reverse.
Absence becomes two forbidden edges. Unqueried entries remain zero. These
forty signed entries represent twenty facts, passed through the original
`load_prior` API with one assertion confidence of exactly 1.0.

The experiment bypasses the existing upstream wrapper: that wrapper rejects
endpoint probabilities and evaluates a reconstructed objective with local
numerical handling. Instead, this runner calls the original class directly
with Table 1's sigma 1, L1 penalty 0.05, Gaussian `pdf` loss, threshold 0.30,
100 outer iterations, native polynomial acyclicity tolerance 1e-8, and rho
ceiling 1e16. It preserves the original zero initialization, 0.4 prior
multiplier, gradient damping, and SciPy inner solver defaults. There is no
confidence smoothing, added likelihood floor, warm start, replacement solver,
graph repair, or automatic adjustment of the sparsity penalty.

An isolated optimizer proxy records the original evaluations and solutions
without changing input arguments, solver options, objective/gradient values,
or returned solution objects. It does not patch global SciPy state. The final
unthresholded weights and objective are retained because the authors' routine
returns thresholded weights without exposing convergence diagnostics.

A valid evaluation requires a finite final optimizer objective, finite
unthresholded weights, native polynomial acyclicity at most 1e-8, and a valid
reported DAG. Optimizer termination remains a separate diagnostic. In
particular, an empty thresholded graph is not accepted if the optimizer's
objective is nonfinite. Nonfinite values are written as explicit strings in
diagnostic JSON, with warnings and exceptions retained.

Run `python -B -m cd_llm experiment synthetic-eca-original` with a timestamped
`--run-id` and `--stage prepare`, `--stage compute --graph-index INDEX`, or
`--stage aggregate`. Preparation freezes observations, facts, original source,
dependencies, experiment implementation, reference artifacts, and paper
hashes. Each method/dataset result is resumable. Pilot SCMs 1 and 15 cover
required-edge and absence-only facts; retain those evaluations when computing
the other eight SCMs. Infrastructure retries preserve algorithm settings.

Aggregation verifies predictions and metrics, requires all twenty recorded
evaluations, and writes `summary.csv`, `comparison.csv`, `diagnostics.csv`,
`completion.json`, and `report.md`. The comparison includes unchanged reference
rows and identifies the previous ECA implementation as the local paper backend.
ED, adjacency F1, and orientation F1 use the existing point-DAG conventions.
Means and sample SEs require all ten valid SCMs for a sample size; otherwise
cells are dashes with counts and individual failure reasons. No successful
subset is averaged. There is no publication option and the paper is unchanged.

The completed evaluation is recorded under
`20260921T014251Z/synthetic_eca_original`. It has 3/10 valid SCMs at each sample
size: SCMs 15, 34, and 40, whose twenty facts contain only absent adjacencies.
All fourteen evaluations containing required edges have an infinite initial
objective, nonfinite initial gradient, and nonfinite final optimizer output.
They are retained as failures, even when the original routine returns an empty
thresholded matrix. No full-cohort ECA mean or SE is reported. This establishes
a confidence-1/zero-initialization numerical failure in this version of the
authors' code, not a performance ranking for ECA under other input settings.
