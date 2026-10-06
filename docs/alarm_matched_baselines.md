# Matched ALARM baseline comparison

The experiment pairs PC, GES, NOTEARS, gPC-Guess, BFS+Pearson,
PC+Long-Srisk, Causal Order+PC, and SCP Pattern 2 with the completed local
Qwen3.5-9B SMC claim-pool study. It uses the same observations, variable
presentation, definitions, model revision, and runtime. There are 20
conditions: sample sizes 200 and 1,000, each with seeds 1000–1009.
The 200-row dataset is the prefix of its paired 1,000-row dataset.

Each baseline uses its registered query acquisition and confidence procedure.
There is no baseline confidence cap. Long uses native A/B token
probabilities and Srisk tolerance 0.1. SCP uses five native probability
trials per ordered pair, with forbid/require thresholds 0.05/0.95. The saved
SMC trajectories retain their 0.8 answer-confidence cap and are not rerun.
SCP explanation generation, five probability trials, and parsing repairs use
temperature 0.7. Other baselines use temperature zero. Thinking remains disabled,
and base request seed zero and probability-trial seed offsets are preserved.
SCP retains 100 observational bootstrap fits and the existing filter that skips
contradictory required directions; these remain adaptations of the published method.

The source is `results/20260920T164354Z/alarm_claim_pools_local/`. New
artifacts belong in a separate timestamped workspace result directory.
The workflow freezes source result hashes, all 60 trajectory hashes,
observations, statistical fits, runtime identity, prompts, and implementation
hashes. Statistical defaults are stable PC with G² at 0.05, GES with BDeu
ESS 1 and structure prior 1, and the existing 12-iteration NOTEARS settings.
Representative PC/GES refits must reproduce the saved partial graphs.

`python -m cd_llm.experiments.alarm_matched_baselines` exposes `prepare`,
`bootstrap`, `pilot`, `collect`, `audit`, `fit`, `gpc`, `report`, and `submit`.
All stages require `--root`; per-condition stages also require
`--condition 0..19`. Collection accepts optional method, condition, shard,
and shard-count selectors. The complete workflow uses all 7,770 triplets
and all 26,640 SCP ordered pairs. It does not replace an unfinished
exhaustive run with a smaller sample.

The SQLite cache includes complete effective requests and runtime identity.
An attempt ledger distinguishes completed, failed, and unresolved dispatches.
Logical uses survive replay without double counting and charge shared
answers in full to each dataset. Native parsing repairs count toward token
expenditure. A pilot checks gPC, BFS, Long probabilities, SCP probabilities,
20 predetermined triplets covering every node, and one pair response.
Pilot answers are reused whenever the full method requires them.

Final fitting uses cached responses with network access disabled. Causal
Order graph answers and A/B/C answers use separate parsers; valid C answers
retain the native tie-break behavior. For a PC orientation fallback, Qwen's
A/B direction is used when acyclic; an invalid, C, or cyclic answer is replaced
only when the opposite direction is uniquely acyclic. gPC
checkpoints preserve the edge, conditioning depth, subset permutation,
cursor, and RNG state, without restricting conditioning depth. Long retains
the exact-MEC safeguard of 50,000 extensions and reports violations.

The reviewed Causal Order integration processes undirected PC edges in
lexicographic order. An expert-order direction is accepted only when it keeps
the working graph acyclic; otherwise the frozen Qwen3.5-9B pair prompt supplies
the direction. If that response is unusable or cyclic, the implementation uses
the unique acyclic direction and records the override. It uses no reference
graph.

`per_seed.csv`, `summary.csv`, and `full_table.csv` report unnormalized ED
and ES, adjacency and orientation F1, and logical token expenditure. Point
and partial predictions retain unresolved endpoints and have ED = ES.
Long uses its uniform surviving MEC; SMC uses its actual saved distribution.
Means and standard errors require all ten feasible seeds; failures are never
silently removed from an aggregate. Paired endpoint comparisons use 10,000
bootstrap resamples and exact paired sign-randomization tests with Holm
adjustment across 48 comparisons. Uncertainty is conditional on the shared
expert collections.

`accounting.json` separates logical expenditure from unique generations,
actual accelerator time, and actual API spending (zero). The separate
performance–cost PDF uses logarithmic token axes, checkpoint markers,
standard-error bars, side legends, and no connecting lines. The manuscript
and its figures are not modified by this workflow.

## Recovering interrupted evaluations

`recover --source-root OLD --root NEW --chain` validates the saved experiment,
backs up its database, records a manifest amendment, and resumes missing work.
Recovery requires the original implementation archive and preparation record;
unreviewed input or implementation changes stop execution. `resume --root NEW`
reconciles saved outcomes against active jobs instead of trusting old submission
markers. Submission intents have unique identifiers so an accepted job can be
recovered even if recording its returned identifier was interrupted.

The database uses persistent rollback journals and full synchronization.
Interrupted request attempts remain visible; exact cached requests are reused,
and missing responses may be collected again with unknown historical token
usage disclosed separately. Infrastructure failures remain retryable and never
count as scientific infeasibility or successful completion. Only documented
native method limitations may produce terminal infeasible outcomes.

Completion requires all 160 baseline outcomes, the cache audit, and the 240
unchanged SMC checkpoints. SCP retains 100 PC bootstrap fits per condition;
the paired confidence intervals use 10,000 resamples of data seeds.

## SCP temperature amendment

`amend-scp-temperature --root ROOT --temperature 0.7 --chain` migrates the existing
experiment after its collector has stopped. It requires the saved pre-amendment
manifest and matching implementation archive under
`amendments/scp_temperature_0p7/`. It creates a consistent database backup,
archives SCP pair records and logical uses, invalidates dependent gates and
reports, and records the reviewed configuration and implementation changes.
Other methods' cached answers, predictions, and search checkpoints are retained.
The amendment is idempotent and resumes collection after the pilot passes again.

## Causal Order cycle-guard amendment

`amend-order-cycle-guard --root ROOT --chain` restores the original 20 Causal
Order records, preserves the 17 already-acyclic predictions byte-for-byte, and
collects only the missing Qwen pair decisions needed by the three cyclic
conditions. Completion records all expert calls, answers, forced safe
directions, and token expenditure.

Effective request identities include temperature. The cache audit checks each
active logical request against its method's registered temperature. Superseded
SCP responses and attempts remain available for auditing and actual expenditure;
only the corrected collection contributes to SCP's logical token cost.

## Causal Order threshold correction

`amend-order-threshold --root ROOT --chain` validates a registered correction
using the existing triplet and tie-break cache with network access disabled.
The pinned upstream code remains unchanged. The adapter defaults to upstream
behavior; this experiment explicitly selects `validated_threshold`.
Already acyclic upstream outputs are retained. If its last binary-search cutoff
still leaves a cycle, the adapter searches distinct entropy-weight cutoffs and
returns the lowest cutoff whose retained graph is acyclic, including a cutoff
above the maximum weight. Equal-weight edges are treated together.

The amendment requires a saved manifest and matching implementation archive
under `amendments/order_validated_threshold/`. It preserves the original failure,
backs up the cache, records both thresholds and removed edges, and verifies
that replay does not change request or logical-use counts. It then identifies
missing PC fallback questions and resumes remaining collection. Existing votes,
weights, raw responses, and other baseline results are preserved; the report
identifies the numerical implementation correction explicitly.

## Completing the matched evaluation

The execution amendment preserves scientific settings and migrates the five
unfinished gPC searches to a disk-backed shuffle mapping. Its random draws,
conditioning sets, and stopping rule are unchanged. Atomic checkpoints contain
the mapping, cursor, graph, and RNG state together. The original checkpoints
and response database are retained in the amendment archive.

`python -m cd_llm.experiments.alarm_matched_execution` exposes `migrate`,
`resume`, `baselines`, `prepare_shards`, `scp`, `gpc_pilot`, `gpc`, `early_fit`,
`merge`, `fit`, and `report`. Stages take `--root`; individual workers and
conditions also take `--index`. Continuations retain their stage and index.
Execution changes require the archived pre-amendment manifest and code,
regression checks, and verified input hashes.

BFS and Long collection finishes before four independent SCP workers begin.
Equal knowledge-prompt families are assigned to the same worker, preserving
shared-response reuse. Each worker reads a frozen existing response cache and
writes a separate durable ledger. The merge rejects conflicting responses,
logical uses, or pair outcomes and imports each attempt only once.

Method-specific audit gates permit early offline BFS, Long, and Causal Order
evaluation. The final complete-cache audit still gates SCP fitting and the
complete report. Initial worker validation uses the first 64 required pairs
and reuses their responses. Unexpected infrastructure failures remain visible
and stop the affected work rather than triggering unchanged retries.

Execution progress and failures are recorded beneath `execution/`; gPC
progress records omit its large mapping. The completion requirements and all
statistical reporting rules above continue to apply. The paper is unchanged.
