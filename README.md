# Claim-Fuse

This repository contains the Python implementation, comparison baselines, and reproducible experiment protocols for Claim-Fuse.

First-party code is installed from `src/cd_llm/`. Canonical inputs live in `resources/`. Generated datasets, numerical results, and command logs belong in the workspace-level `generated/`, `results/`, and `out/` directories.

## Install

Python 3.12 or newer and Java 21 or newer are required for the complete experiment suite.

```bash
python3.12 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install -e '.[full]'
```

Some comparison methods require pinned upstream checkouts. Their setup instructions are in the corresponding package README under `src/cd_llm/baselines/`.

## Run

The commands below rebuild the active experiments in Section 4. They make fresh model calls and write responses, checkpoints, and numerical results to `../results/<timestamp>/`; command logs go to `../out/<timestamp>/`. These generated files are ignored by Git and are not part of the release.

Each workflow stops if a child command fails. Add `--dry-run` to print its complete command sequence without running it.

Start in the repository root and choose one UTC run identifier for the full reproduction:

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export CLI="../.venv/bin/cd-llm"
export RESULT_ROOT="../results/$RUN_ID"
export LOG_ROOT="../out/$RUN_ID"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
```

The fixed input archives and their SHA-256 values are registered in [`resources/reproduction/protocol.json`](resources/reproduction/protocol.json). They contain observations, masked metadata, evaluation graphs, prompts, and data-only initial particle populations. They contain no LLM responses or reported results.

### Model endpoints

The ALARM experiments and the local Qwen neuropathic baselines use Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16, and vLLM `0.26.1rc1.dev1133+gf94666b60`. Start an OpenAI-compatible endpoint with the pinned settings:

```bash
export QWEN_CHECKPOINT="/path/to/Qwen3.5-9B-c202236235762e1c871ad0ccb60c8ee5ba337b9a"
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"

../.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_CHECKPOINT" \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name Qwen/Qwen3.5-9B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85
```

The hosted neuropathic runs require `OPENROUTER_API_KEY`. They pin the model slug, provider route, prompt, request seed, temperature, reasoning setting, and output limit. A hosted provider can update the weights behind an unchanged slug, so those calls cannot pin weights as strongly as the local Qwen run.

### RQ2 synthetic claim pools

This is the 180-trajectory grid for ten SCMs, two sample sizes, three simulated-expert accuracies, three claim pools, and query checkpoints 5, 20, and 50.

```bash
../.venv/bin/python scripts/reproduce.py synthetic-claim-pools --run-id "$RUN_ID"
```

### Neuropathic pain: Qwen ClaimFuse

The first workflow collects every local and ancestral answer afresh through the pinned DeepInfra BF16 route. The second applies the paper's 0.8 answer-confidence cap and reruns inference without new model calls.

```bash
../.venv/bin/python scripts/reproduce.py neuropathic-claim-fuse-qwen --run-id "$RUN_ID"
```

### Neuropathic pain: GPT-5.6 Terra ClaimFuse

This run uses the same fixed observations and initial populations as the Qwen run, calls `openai/gpt-5.6-terra` through the OpenAI provider route, and retains the same 0.8 confidence cap.

```bash
../.venv/bin/python scripts/reproduce.py neuropathic-claim-fuse-terra --run-id "$RUN_ID"
```

### Neuropathic pain baselines

Run the shared gPC study once; it collects independent Qwen and Terra expert DAGs and evaluates both sample sizes for ten seeds.

```bash
../.venv/bin/python scripts/reproduce.py neuropathic-gpc --run-id "$RUN_ID"
```

The data-only PC/GES rows and Qwen Causal Order+PC share one workflow. The 20 baseline tasks are ten seeds by two sample sizes. Qwen elicitation is shared across sample sizes.

```bash
../.venv/bin/python scripts/reproduce.py neuropathic-causal-order-qwen --run-id "$RUN_ID"
```

BFS+Pearson and PC+Long use the checksummed benchmark archive directly. The workflow runs the configured Qwen and Terra variants. SCP Pattern 2 requires first-token log probabilities and therefore applies only to Qwen in the main table.

```bash
../.venv/bin/python scripts/reproduce.py neuropathic-baselines --run-id "$RUN_ID"
```

### ALARM ClaimFuse

The stated and consistency workflows use the same fixed datasets and data-only initial populations. The local server makes all responses afresh; no saved response file is required.

```bash
../.venv/bin/python scripts/reproduce.py alarm-claim-fuse --run-id "$RUN_ID"
```

### ALARM baselines

The matched baseline workflow produces PC, GES, NOTEARS, BFS+Pearson, PC+Long, Causal Order+PC, SCP Pattern 2, and its gPC collection from the fresh ALARM observations and local Qwen endpoint.

```bash
../.venv/bin/python scripts/reproduce.py alarm-baselines --run-id "$RUN_ID"
```

### Section 4 exports

These exporters accept only explicit fresh paths. They reject missing or duplicate experimental cells instead of silently filling them from older runs.

```bash
../.venv/bin/python scripts/reproduce.py export-main-results --run-id "$RUN_ID"
```

Build the RQ3 performance, token, and estimated serial-runtime tables directly from the fresh per-seed records. The exporter fits one nonnegative request/input-token/output-token latency model to each dataset's fresh Qwen response ledger, applies it consistently to all methods, and rejects missing ClaimFuse checkpoints or baseline cells.

```bash
../.venv/bin/python scripts/reproduce.py export-cost-results --run-id "$RUN_ID"
```

Fresh LLM calls can produce small numerical differences even with fixed prompts, request seeds, model names, and configurations. Exact reruns are most reliable for the locally pinned Qwen checkpoint; hosted model slugs and provider infrastructure can change over time. The release intentionally excludes saved responses and large result archives.

The complete command interface remains available through:

```bash
$CLI --help
$CLI experiment neuropathic-new-prompt-smc --help
$CLI experiment alarm-claim-pools-local --help
```

## Test

```bash
../.venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

See [`docs/index.md`](docs/index.md) for installation, architecture, data, and experiment documentation.
