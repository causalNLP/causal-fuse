# Amirkhani et al. (2017) baseline

This package is a clean-room Python implementation of H. Amirkhani, M.
Rahmati, P. J. F. Lucas, and A. Hommersom, “Exploiting Experts' Knowledge for
Structure Learning of Bayesian Networks,” *IEEE TPAMI* 39(11), 2154–2170,
2017, DOI: [10.1109/TPAMI.2016.2636828](https://doi.org/10.1109/TPAMI.2016.2636828).
It was written from the published algorithm and equations; it does not copy an
official implementation.

## Implemented method

The package combines a decomposable observational-data log score with either
of the paper's expert-opinion log scores:

- `explicit` uses one shared or per-expert supplied accuracy triple
  `(gamma1, gamma2, gamma3)`.
- `em` estimates those accuracies from opinions alone and then uses the
  explicit score. The default edge-state prior is `(0.1, 0.1, 0.8)`, the
  default initialization is `(1/3, 1/3, 1/2)`, and the convergence tolerance
  is `0.001`. As in the paper, this estimator uses the tractable
  independent-pair approximation. The implementation conservatively checks
  changes in both the edge-state prior and accuracies, while the experiment
  description states its stopping rule only in terms of accuracies.
- `marginal` integrates each expert's accuracy out in closed form. It defaults
  to uniform `Dirichlet(1,1,1)` and `Beta(1,1)` priors.

The neutral accuracy and marginal-prior defaults are conservative API
defaults, not the paper's simulation settings. Those experiments initialized
EM from population-mean accuracies and formed the marginal priors from the
same means using equation (18) with concentration `c=10`. The latter is
available as `AccuracyPrior.from_mean(mean_accuracy, concentration=10)`.

Search starts from the empty DAG. At every iteration it scores all acyclic
single-edge additions, deletions, and reversals, accepts the best strict
improvement, and stops at a local optimum. Ties are deterministic. As in the
paper, the default does not cap the number of parents. The graph prior is
uniform, so its constant log term is omitted.

Exact boundary accuracies can assign zero likelihood to the empty DAG. Search
can leave that state when one edge operation reaches finite likelihood, but it
does not cross a plateau of zero-likelihood graphs. For example, two compatible
perfect directed opinions may require two edge additions before their joint
likelihood becomes nonzero. The implementation raises a diagnostic error in
that case; use non-boundary accuracies to retain the paper's strict-improvement
search without adding an unreported plateau-traversal rule.

The native data model is complete discrete data with BDeu and equivalent
sample size 1. The optional `gaussian-bic` backend is a project adaptation for
complete continuous data. It replaces only the observational-data term; the
three-state expert model and greedy DAG search are unchanged. Results from the
adaptation should be labeled as such rather than presented as a native data
type supported by the paper. For BDeu, category cardinalities are inferred
from observed values unless they are supplied explicitly.

## API sketch

```python
from cd_llm.baselines.amirkhani2017 import (
    AmirkhaniConfig,
    ExpertOpinion,
    fit_amirkhani,
)

opinions = [
    ExpertOpinion.directed("expert-1", 0, 2),
    ExpertOpinion.absent("expert-2", 0, 1),
]
result = fit_amirkhani(
    discrete_data,
    opinions,
    config=AmirkhaniConfig(expert_score="marginal"),
)
dag = result.graph
```

`fit_amirkhani_from_local_score` accepts any higher-is-better decomposable
score object exposing `node_count` and `local_score(node, parent_mask)`.
`result` records the data and expert score terms, accepted moves, evaluation
counters, and EM diagnostics when applicable.

## Opinion semantics and limitations

Each response categorizes one canonical pair as `left_to_right`,
`right_to_left`, or `absent`; an omitted response is missing rather than an
absent-edge assertion. Each expert may respond at most once per pair. The
publication experiments construct these native categorical opinions directly
from their shared pairwise answers. Missing observational values and partial
or mixed graphs are not supported.
