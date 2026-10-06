# Architecture

First-party Python code is installed from `src/cd_llm/`. Dependencies flow from shared primitives toward executable protocols:

```text
core/data -> methods/baselines -> metrics -> experiments -> CLI
```

## Packages

- `cd_llm.core` contains immutable graph representations, finite-support inference helpers, discovery adapters, simulation, completion, and structural metrics.
- `cd_llm.methods` contains Claim-Fuse SMC, active query selection, proposal kernels, and graph scores.
- `cd_llm.baselines` contains the comparison methods used by the reported experiments.
- `cd_llm.metrics` contains metric unit conversion and artifact metadata helpers.
- `cd_llm.data` contains metadata validation, loading, generation, and visualization.
- `cd_llm.experiments` contains the reported synthetic, neuropathic, ALARM, and latent-variable protocols.
- `cd_llm.cli` exposes the installed command interface.

The PAG graph representation in `cd_llm.methods.fci.graph` is retained for the SCP baseline. The removed local FCI and EG-FCI solvers are outside the current paper scope.

## Repository boundaries

- `resources/` contains canonical immutable project inputs.
- `vendor/` contains tracked external sources and locations for optional pinned checkouts.
- `tests/` contains first-party unit and integration tests for the retained release.
- Generated data, numerical results, and logs are written outside `code/` to the workspace-level `generated/`, `results/`, and `out/` directories.

New runs should use a UTC timestamp as their top-level result and log directory.
