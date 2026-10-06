# Matched ALARM claim-pool experiment

The registered comparison uses Qwen3.5-9B with the DeepInfra BF16 provider, stated confidence capped at 0.8, and local, ancestry, and mixed claims. Ten observation seeds each provide nested samples of 200 and 1,000 observations. All pools share PC+GES populations and inference seeds. Each trajectory runs to 100 attempted queries; checkpoints at 10, 20, and 50 do not trigger extra mutation. Abstentions consume budget and recorded tokens without changing the posterior. EIG uses working confidence 0.9, separately from the response cap.

Artifacts belong under workspace `results/<timestamp>/alarm_claim_pools/`, and command logs belong under workspace `out/<timestamp>/alarm_claim_pools/`. Inspect the available protocol stages with:

```bash
../.venv/bin/cd-llm experiment alarm-claim-pools --help
```

Preparation freezes observations, prompts, provider metadata, configuration, and implementation hashes. Response collection uses an atomic response cache, persistent spending reservations, a fixed cost cap, and a parsing pilot. Malformed responses permit one format repair. Invalid repaired answers, accounting failures, and spending exhaustion stop dependent inference.

An offline audit gates the inference pilot and its exact replay. All 60 trajectories and 240 nonzero checkpoints must validate before reporting. Primary paired ED contrasts compare the mixed pool with each restricted pool at 100 queries, separately by sample size. Secondary contrasts compare mixed at 50 against each restricted pool at 100. Reports contain paired bootstrap intervals, Bonferroni intervals, and Holm-adjusted sign-flip tests. Uncertainty across observation seeds is conditional on the network and shared answer collection.

Scoring infers observed categorical levels, including constant columns and gaps in canonical clinical codes. It does not interpret numeric category codes as continuous values.
