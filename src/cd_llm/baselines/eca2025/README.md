# ECA (Chen et al., 2025)

This project-owned package adapts Chen et al. (2025), *Continuous Structure Constraint Integration for Robust Causal Discovery*, to the project's continuous, causally sufficient DAG setting. The separate Apache-2.0 upstream checkout is expected at `vendor/ECA` and pinned at `c98002720dca8c170ba0f3d25ac3681e5def0bdc`; its tracked files are left unchanged.

## Implementations

- `backend="paper"` (default) implements the normalized linear objective from Equation 18, per-edge existence probabilities, exact matrix-exponential NOTEARS acyclicity, and a split-variable L-BFGS-B augmented-Lagrangian optimizer.
- `backend="upstream"` invokes the archived linear class directly. The wrapper repairs only the missing `load_prior` call and threshold forwarding. It otherwise retains the commit's scalar assertion confidence, `0.4` prior multiplier, near-zero gradient damping, polynomial acyclicity, and PDF-scaled data term.

The upstream demo is not used: it generates a prior without loading it, writes global files, refers to missing default data, and terminates with a nonzero exit status. Optional upstream dependencies are isolated in `requirements-upstream.txt`:

```bash
git clone https://github.com/baosws/Continuous-Structure-Constraint-Integration.git vendor/ECA
git -C vendor/ECA checkout c98002720dca8c170ba0f3d25ac3681e5def0bdc
python -m pip install -r src/cd_llm/baselines/eca2025/requirements-upstream.txt
```

The paper backend needs only the project's NumPy and SciPy dependencies. Nonlinear ECA and MAG/PAG output are intentionally unsupported.

## API

```python
from cd_llm.baselines.eca2025 import (
    ECAConfig,
    ECAPrior,
    EdgeConstraint,
    fit_eca,
    fit_eca_with_evidence,
)

prior = ECAPrior(3, (EdgeConstraint(0, 1, 0.8),))
result = fit_eca(
    observations,
    prior,
    config=ECAConfig(backend="paper", standardize_data=True),
)

# Project-native Claim/EvidenceGraph input; target truth is never consulted.
evidence_result = fit_eca_with_evidence(
    observations,
    claims,
    evidence_graph,
    config=ECAConfig(backend="paper", sigma=1.0),
)
```

`ECAResult` contains the canonical `cd_llm.core.finite_support.Graph`, read-only weighted and binary adjacency matrices, an immutable point-mass support, the objective decomposition, convergence diagnostics, the effective prior, and (for the evidence entry point) a complete evidence-adaptation audit.

## Evidence conversion

Only signed Group-I predicates are accepted. Directed inclusion/exclusion maps to one ordered edge. Adjacency and non-adjacency map to two equally weighted ordered-edge factors. In particular, a positive adjacency becomes two directed-present factors: over DAGs, either orientation has likelihood ratio `c/(1-c)` relative to no edge, although the continuous relaxation may temporarily encourage both directions before acyclicity suppresses one.

For conditional response confidence `p`, the adapter uses

```text
logit(c) = logit(p) / sigma^2
```

so the paper objective has the same agreement-to-contradiction odds as the project's evidence factor. Independent evidence on the same ordered edge is combined by adding signed log odds; exact cancellation removes that edge from the mask. The adapter uses context-compatible provenance/factor representatives and deliberately ignores evaluation-only `truth` and `correct` fields.

The five fixed same-SCM claims map without target refinement to eight ordered entries: four forbidden entries for the two non-adjacencies, required `3 -> 5`, forbidden `3 -> 6`, and both directed-present factors for adjacency `0 ~ 5`.

## Numerical behavior

The defaults are `lambda1=0.05`, `sigma=1`, threshold `0.30`, `h_tolerance=1e-8`, and no internal standardization. Because the paper's absolute-value edge gate is nonsmooth at zero, existence-constrained entries receive a deterministic `1e-3` warm start whose coefficient sign follows the data gradient (positive on an exact tie); all other entries retain the NOTEARS zero initialization. This prevents the zero-subgradient convention from falsely declaring the all-zero matrix stationary. Callers using the paper's mixed existence/absence evidence keep the fixed sparsity penalty because the paper does not define an automatic mixed-prior correction. Thresholded cyclic output raises an error; the adapter never silently drops edges.
