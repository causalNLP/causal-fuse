# Original ECA with the authorized confidence cap

This numerical follow-up repeats all twenty datasets from the original-code
ECA run `20260921T014251Z`. The user explicitly authorized capping confidence
to 0.99. Raw expert answers remain at confidence 1.0 in the saved facts; the
shared assertion confidence passed to the authors' `load_prior` is 0.99.

The observational matrices, twenty pairwise facts per SCM, forty signed
entries, source revision, runtime, zero initialization, objective, gradient
damping, sparsity penalty, acyclicity constraint, solver defaults, thresholds,
and validation rules are unchanged. The capped run imports and uses the
original-code runner without modifying it or invalidating its frozen hashes.
There is no new warm start, numerical floor, solver substitution, or repair.

Use `python -B -m cd_llm experiment synthetic-eca-confidence-cap --run-id
TIMESTAMP` with stages `prepare`, `compute --graph-index INDEX`, and
`aggregate`. Preparation verifies the confidence-1 reference and preserves all
its artifacts and the paper. Computation verifies the exact standardized data
hashes and saves raw confidence and effective confidence separately.

The paired report includes initial objective/gradient finiteness, final
objective finiteness, native continuous acyclicity, final optimizer status,
nonfinite evaluations, and unsuccessful inner solves. These diagnostics are
separate from graph validity. Metrics use the existing ten-SCM completeness
rule; no successful-subset average is reported. The experiment produces a
separate report; publication is a separate step.

Run `20260921T015754Z/synthetic_eca_confidence_cap` produced twenty valid fits,
compared with six in the exact-confidence reference. Every initial evaluation
was finite, no objective/gradient evaluation was nonfinite, and all twenty
final optimizer calls reported success. The maximum continuous acyclicity was
approximately 7.22e-9, below the unchanged 1e-8 tolerance. There were 26
unsuccessful intermediate optimizer calls; these are retained in the paired
diagnostics rather than being hidden by the final success status.

These results now supply the ECA row in Table 1 through publication
`20260921T020934Z/synthetic_table1_publication`. The seven other methods retain
their base-run results. The original exact-confidence and capped experiment
artifacts remain intact. See the [publication instructions](synthetic_matched_expert_baselines.md)
for the command and provenance checks.
