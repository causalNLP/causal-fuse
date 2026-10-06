# Causal Order (Vashishtha et al., 2025)

This package provides the two downstream causal-discovery baselines from Appendix D of *Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference*: **Causal Order + PC (PC+LLM)** and **Causal Order + CaMML (CaMML+LLM)**.

## Upstream boundary

The authors' repository implements triplet prompting, per-pair vote aggregation, strong-model tie breaking, entropy-based cycle removal, and the OpenAI client. It does not implement Appendix-D Algorithm 1 or Algorithm 2 and contains no PC or CaMML dependency. This package therefore invokes those upstream components unchanged and implements only the missing project integration.

The adapter is pinned to commit `ad609dbd3d93e50d890c9967eb746945546515d1`. Because that revision has no `LICENSE` file, its checkout remains ignored and no upstream source is copied into this repository. The wrapper rejects a checkout at another revision or with modified tracked files.

Set up the optional checkout and dependencies from the `code/` repository root:

```bash
git clone https://github.com/AniketVashishtha/Causal_Order_Imperfect_Experts.git vendor/Causal_Order_Imperfect_Experts
git -C vendor/Causal_Order_Imperfect_Experts checkout ad609dbd3d93e50d890c9967eb746945546515d1
python -m pip install -r src/cd_llm/baselines/causal_order2025/requirements-upstream.txt
export OPENAI_API_KEY="your-key-here"
```

The current upstream defaults are `gpt-4o-mini` for triplet subgraphs, `gpt-4o` for tie breaking, and a 12-second delay per call. These are current repository defaults, not the GPT-3.5/GPT-4 configurations reported in the paper.

For exhaustive OpenRouter benchmarks, `elicit_triplet_order_openrouter` preserves the pinned prompts, parsing, voting, and cycle-removal code while replacing only the missing transport layer. It uses `OPENROUTER_API_KEY`, an atomic SQLite response cache, bounded retries, seeded requests, concurrent calls, and a hard cost cap. The cache records request IDs, raw responses, usage, cost, latency, and attempts, but never the credential.

## API

Query the LLM once and reuse its result across observational sample sizes and data seeds:

```python
from cd_llm.baselines.causal_order2025 import (
    fit_camml_with_causal_order,
    fit_pc_with_causal_order,
    query_triplet_order,
)

expert = query_triplet_order(
    node_names,
    context="Domain description supplied to the expert.",
    descriptions=node_descriptions,
)

pc_result = fit_pc_with_causal_order(
    observations,
    node_names,
    expert,
)

camml_result = fit_camml_with_causal_order(
    observations,
    node_names,
    expert,
    prior_probability=0.9,
)
```

`fit_pc_with_llm` and `fit_camml_with_llm` are end-to-end convenience functions. The cached-result entry points are preferable for benchmarks because the upstream triplet stage uses all `n choose 3` subgroups and is independent of the sample rows.

## Algorithm 1: Causal Order + PC

PC is run with configurable causal-learn settings. Existing directed CPDAG edges are preserved. Undirected edges are processed in lexicographic order. When both endpoints occur in the expert order, the earlier-to-later direction is used only if it keeps the working graph acyclic; otherwise the supplied pair expert is consulted. A valid acyclic expert direction is retained. If the expert answer is unusable or cyclic, the unique acyclic direction is used and recorded; without a unique safe direction the integration is infeasible.

The implementation follows the symmetric main-text rule. Appendix-D Algorithm 1 literally sends the reverse-order case to the expert, which contradicts the surrounding description that the edge should be oriented according to the order.

## Algorithm 2: Causal Order + CaMML

The raw upstream expert graph is converted to a level order. Strongly connected components are assigned one level before levels are computed on the condensation DAG, providing a deterministic interpretation of the paper's overlapping-cycle pseudocode. Nodes isolated in the expert graph contribute no ordering constraint, allowing the observational learner to place them.

For nodes on different levels, prior probability `p` is encoded as a CaMML TOM-order prior without asserting that an edge exists. The project default is `p=0.9`; the paper does not provide a numeric value. Nodes on the same level and isolated nodes receive no causal-order prior. The existing CaMML implementation supplies the statistical score, sampler, DAG support, and diagnostics.

## Outputs and limitations

Both APIs return the learned project-native DAG and final full-node topological order required by Appendix D. PC+LLM is a deterministic point mass; CaMML+LLM preserves the sampled DAG support. Results also retain the upstream expert output, configuration, and timings.

The upstream strategy is exhaustive: it makes one subgroup call for every triplet plus strong-model tie calls. The paper mentions a constant-auxiliary approximation but neither specifies nor implements its sampling rule, so this package does not add one.

## References

- Aniket Vashishtha et al., *Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference*, ICLR 2025, Appendix D: https://arxiv.org/abs/2310.15117
- Authors' implementation: https://github.com/AniketVashishtha/Causal_Order_Imperfect_Experts
