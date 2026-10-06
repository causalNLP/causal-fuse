from __future__ import annotations

from typing import Mapping, Sequence, Tuple

from .evaluation import PCBootstrapProbabilities


def format_bootstrap_frequencies(
    frequencies: Mapping[Tuple[str, str], Mapping[str, float]],
) -> str:
    lines = [
        "Bootstrap frequencies for pairwise PAG/CPDAG endpoint states are shown below.",
        "Endpoint states are written from the first named variable to the second.",
    ]
    for (left, right), states in sorted(frequencies.items()):
        nonzero = [
            f"{state}={probability:.3f}"
            for state, probability in sorted(states.items())
            if probability > 0.0
        ]
        lines.append(f"- {left} | {right}: {', '.join(nonzero)}")
    return "\n".join(lines)


def format_pc_pattern2_probabilities(
    probabilities: PCBootstrapProbabilities,
) -> str:
    lines = [
        "All of the directed edges with non-zero bootstrap probabilities "
        "suggested by the statistical causal discovery are below:"
    ]
    directed = [
        f"- {source} → {target} (bootstrap probability = {probability})"
        for (source, target), probability in sorted(probabilities.directed.items())
        if probability > 0.0
    ]
    lines.extend(directed or ["- None"])
    lines.append(
        "All of the undirected edges with non-zero bootstrap probabilities "
        "suggested by the statistical causal discovery are below:"
    )
    undirected = [
        f"- {left} — {right} (bootstrap probability = {probability})"
        for (left, right), probability in sorted(probabilities.undirected.items())
        if probability > 0.0
    ]
    lines.extend(undirected or ["- None"])
    return "\n".join(lines)


def pc_pattern2_pair_statement(
    *,
    source: str,
    target: str,
    probabilities: PCBootstrapProbabilities,
) -> str:
    directed = probabilities.directed.get((source, target))
    pair = tuple(sorted((source, target)))
    undirected = probabilities.undirected.get(pair)
    if directed is not None and undirected is not None:
        return (
            f"there may be a direct impact of a change in {source} on {target} "
            f"with a bootstrap probability of {directed}. In addition, there "
            f"may be a direct causal relationship between {source} and {target} "
            "whose direction has not been determined, with a bootstrap "
            f"probability of {undirected}."
        )
    if directed is not None:
        return (
            f"there may be a direct impact of a change in {source} on {target} "
            f"with a bootstrap probability of {directed}."
        )
    if undirected is not None:
        return (
            f"there may be a direct causal relationship between {source} and "
            f"{target} whose direction has not been determined, with a bootstrap "
            f"probability of {undirected}."
        )
    return (
        f"there may be no direct impact of a change in {source} on {target} "
        "according to the bootstrap statistical causal discovery."
    )


def build_knowledge_prompt(
    *,
    pattern: int,
    source: str,
    target: str,
    nodes: Sequence[str],
    method: str,
    dataset_description: str,
    bootstrap_frequencies: Mapping[Tuple[str, str], Mapping[str, float]] | None = None,
    pc_bootstrap_probabilities: PCBootstrapProbabilities | None = None,
    theme: str = "a clinical system",
    variable_descriptions: Mapping[str, str] | None = None,
) -> str:
    if pattern not in {0, 2}:
        raise ValueError("Only paper Patterns 0 and 2 are supported.")
    variables = "\n".join(
        (
            f"- {name}: {variable_descriptions[name]}"
            if variable_descriptions is not None and name in variable_descriptions
            else f"- {name}"
        )
        for name in nodes
    )
    opening = (
        f"We want to carry out causal inference on {theme}, "
        "considering the following as variables:\n"
        f"{variables}"
    )
    if pattern == 0:
        return (
            f"{opening}\n\n"
            f"If {source} is modified, will it have a direct impact on {target}?\n"
            "Please provide an explanation using your domain knowledge about the "
            f"causal relationship between {source} and {target}."
        )

    if pc_bootstrap_probabilities is not None:
        statistical_context = format_pc_pattern2_probabilities(
            pc_bootstrap_probabilities
        )
        pair_statement = pc_pattern2_pair_statement(
            source=source,
            target=target,
            probabilities=pc_bootstrap_probabilities,
        )
    elif bootstrap_frequencies is not None:
        statistical_context = (
            "This is an explicitly adapted latent-variable representation, not "
            "the paper's PC directed/undirected bootstrap matrix.\n"
            f"{format_bootstrap_frequencies(bootstrap_frequencies)}"
        )
        pair_statement = (
            f"the endpoint-state bootstrap evidence for {source} and {target} "
            "is included in the global list above."
        )
    else:
        raise ValueError("Pattern 2 requires bootstrap probabilities.")

    return (
        f"{opening}\n\n"
        f"First, we have conducted statistical causal discovery with {method}, "
        f"using {dataset_description}.\n\n"
        f"{statistical_context}\n\n"
        "According to the results shown above, it has been determined that "
        f"{pair_statement}\n\n"
        "Your task is to interpret this result from a domain-knowledge perspective "
        "and determine whether the statistically suggested hypothesis is plausible. "
        f"Please explain the causal relationship between {source} and {target} "
        "and assess the naturalness of this causal discovery result."
    )


def build_decision_prompt(
    *,
    knowledge_prompt: str,
    explanation: str,
    source: str,
    target: str,
) -> str:
    return (
        "An expert was asked the question below:\n"
        f"{knowledge_prompt}\n\n"
        "Then, the expert replied with its domain knowledge:\n"
        f"{explanation}\n\n"
        "Considering objectively this discussion above, "
        f"if {source} is modified, will it have a direct or indirect impact on "
        f"{target}? Please answer this question with <yes> or <no>. "
        "No answers except these two responses are needed."
    )
