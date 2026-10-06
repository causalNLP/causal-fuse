# Tetrad backend

The project installs a pinned `py-tetrad` revision as a core dependency. The retained experiments use it for Conservative PC and for certified PAG/MAG construction and completion in the latent-variable validation.

## Install

Install Java 21 or newer, then run from `code/`:

```bash
../.venv/bin/python -m pip install -e .
```

The exact revision is declared in `pyproject.toml`.

## Verify

```bash
java -version
../.venv/bin/python -c "import importlib.metadata as md; import jpype, pytetrad; print(md.version('py-tetrad'))"
../.venv/bin/python -c "from cd_llm.core.finite_support import py_tetrad_jar; print(py_tetrad_jar())"
../.venv/bin/cd-llm experiment latent-mag-validation --help
```

If JPype cannot find Java, verify that `java`, `javac`, and `jar` resolve to the same Java 21-or-newer installation before recreating the environment.
