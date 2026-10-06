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

The supported command interface is available as both `cd-llm` and `python -m cd_llm`.

```bash
../.venv/bin/cd-llm --help
../.venv/bin/cd-llm experiment claim-fuse-smc --help
../.venv/bin/cd-llm experiment synthetic-pairwise-expert-benchmark --help
../.venv/bin/cd-llm experiment neuropathic-terra-smc --help
../.venv/bin/cd-llm experiment alarm-claim-pools --help
```

## Test

```bash
../.venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

See [`docs/index.md`](docs/index.md) for installation, architecture, data, and experiment documentation.
