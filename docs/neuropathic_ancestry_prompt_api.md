# Ancestry prompt comparison through an API

This study evaluates all 380 ordered ancestry questions from the masked 20-node clinical network. Both the archived prompt and the explicit causal-criteria prompt run through OpenRouter's `qwen/qwen3.5-9b`, pinned to `deepinfra/bf16`, with provider fallbacks disabled. The matched API comparison is primary; historical local answers remain secondary references.

Each prompt uses stated confidence and eight-variant consistency. Stated abstentions terminate immediately, and malformed stated answers receive one retry. Consistency requires six valid responses and preserves the original majority/tie confidence rule. Requests use temperature zero, the archived per-strategy seeds, disabled reasoning, and a 256-token output limit. There are 6,840 nominal requests and 1,520 question outcomes. The default concurrency is four and the spending limit is US$1. The estimated token charge is US$0.20–$0.30 at the frozen endpoint prices.

From the workspace root, prepare the inputs without making paid requests:

```bash
.venv/bin/python code/scripts/run_neuropathic_ancestry_prompt_api.py --stage prepare --run-id YYYYMMDDTHHMMSSZ
```

Make `OPENROUTER_API_KEY` available through your credential environment, then run or resume:

```bash
.venv/bin/python code/scripts/run_neuropathic_ancestry_prompt_api.py --run-id YYYYMMDDTHHMMSSZ
```

The run first checks two questions under both prompts and confidence methods. These responses are retained in the full study. A provider/model mismatch or returned reasoning stops the run. Malformed answers, truncations, and explicit abstentions remain distinct in the audit. API failures preserve completed requests for resumption; uncertain charges retain conservative reservations against the limit.

Results live under `results/<timestamp>/neuropathic_ancestry_prompt_api/` and command logs under the corresponding `out/` directory. The manifest freezes prompts, input hashes, endpoint metadata, prices, and request settings. Credentials never enter the cache or artifacts. Final reports contain confusion matrices, both false-positive-rate denominators, precision, recall, coverage, accuracy, Brier score, paired transitions, costs, and offline replay verification. Target-graph labels are used only during scoring.
