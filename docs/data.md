# Data and Resources

Dataset generation is part of the installed package; generated datasets are never stored under `code/`.

Generate mixed synthetic SCM data:

```bash
../.venv/bin/cd-llm data generate-synthetic \
  --sample-sizes 200 500 1000 \
  --num-nodes 20 \
  --hidden-ratio 0.2 \
  --seeds 0 1 2
```

Generate neuropathic simulator data:

```bash
../.venv/bin/cd-llm data generate-neuropathic \
  --sample-sizes 200 500 1000 \
  --levels 0 1 2 3 \
  --seeds 0 1 2
```

Defaults write to the workspace-level `generated/` directory. `resources/neuropathic/` contains the canonical model and reference inputs. The external simulator source is retained under `vendor/neuropathic-pain-simulator/` for provenance and is not imported by the main package.
