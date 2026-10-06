# Experiments

Use either the installed `cd-llm` command or `python -m cd_llm`. Both route to the same entry points.

List the available commands:

```bash
../.venv/bin/cd-llm --help
../.venv/bin/cd-llm experiment --help
```

The publication interface includes:

- Claim-Fuse passive and active SMC drivers.
- Synthetic pairwise-expert, ECA, claim-robustness, PC-grid, and latent-MAG studies.
- Neuropathic Claim-Fuse, decontamination, Causal Order, GPC, BFS, Guess2Graph, Long, and SCP studies.
- ALARM claim-pool, matched-baseline, GPC DAG, and CPDAG-replay studies.

Inspect a protocol without running it:

```bash
../.venv/bin/cd-llm experiment claim-fuse-smc --help
../.venv/bin/cd-llm experiment synthetic-pairwise-expert-benchmark --help
../.venv/bin/cd-llm experiment latent-mag-validation --help
../.venv/bin/cd-llm experiment neuropathic-new-prompt-smc --help
../.venv/bin/cd-llm experiment alarm-matched-baselines --help
```

Experiment modules preserve their protocol-specific stage, run identifier, cache, and cost-control options. Store new artifacts in matching timestamped directories under the workspace-level `results/` and `out/` roots.
