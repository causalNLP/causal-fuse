# Installation

Python 3.12 or newer and Java 21 or newer are required for the complete release. The pinned Tetrad dependency supplies Conservative PC and latent-MAG completion support.

From `code/`:

```bash
python3.12 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install -e '.[full]'
```

For the core installation without optional analysis packages:

```bash
../.venv/bin/python -m pip install -e .
```

The `full` extra adds optional analysis packages. The `llm-baselines` extra adds the OpenAI-compatible client used by model-backed experiments. Neuropathic dataset generation uses the compatibility loader included in `cd_llm`.

The ECA upstream comparison and Causal Order baseline require pinned external checkouts under `vendor/`. Follow the README in each baseline package before selecting its upstream backend.

Verify the installation:

```bash
../.venv/bin/python -c "import cd_llm; print(cd_llm.__version__)"
../.venv/bin/python -c "import importlib.metadata as md; import jpype, pytetrad; print(md.version('py-tetrad'))"
../.venv/bin/cd-llm --help
```
