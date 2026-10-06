# Guess2Graph and SCP Neuropathic Baselines

The Guess2Graph and SCP/LLM-SCD adapters were contributed on `origin/luke/neuropathic-baselines` and integrated from commit `e388190`. They were ported to the canonical `cd_llm` package without restoring the former top-level compatibility packages.

Guess2Graph exposes the PC-Guess and gPC-Guess skeleton procedures. Both take an explicit NumPy random generator, return undirected skeletons, and receive no evaluation target. The upstream Guess2Graph snapshot is retained under `vendor/guess2graph` with its original license, notice, and revision file. Project code does not modify or import through the vendor tree.

The SCP adapter provides cached model judgments, paper Patterns 0 and 2, bootstrap evidence summaries, PC background constraints, and the neuropathic PC/FCI evaluation driver. The neuropathic experiment is an adaptation because it uses binary G-squared tests and applies endpoint constraints to PAGs at latent levels. This distinction must be retained when reporting results.

Both model-facing interfaces accept generic node identifiers and separate descriptions. Runtime artifacts default to the workspace-level `results/<UTC timestamp>/` directory. Model responses are keyed by the complete request settings and prompt, and evaluation targets are not arguments to the inference functions.
