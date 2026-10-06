"""Independently check saved likelihood updates and artifact consistency."""

import argparse
import json
from pathlib import Path

import numpy as np

from cd_llm.core.finite_support import Claim, claim_value_matrix, evaluate_claim
from cd_llm.experiments.latent_mag_validation import (
    atomic_json, graph_from_record, load_manifest,
)
from cd_llm.paths import WORKSPACE_ROOT


def audit(run_id: str) -> dict:
    root = WORKSPACE_ROOT / "results" / run_id / "latent_mag_validation"
    config, manifest = load_manifest(root)
    complete = failed = checked_updates = 0
    worst_error = 0.0
    failures = []
    for index in range(config.graph_count):
        folder = root / "scms" / f"scm_{index:02d}"
        model = json.loads((folder / "model.json").read_text())
        claims = tuple(Claim(**record) for record in model["claims"])
        target = graph_from_record(model["target"])
        assert len(claims) == 30
        for group in ("I", "II", "III"):
            selected = [claim for claim in claims if claim.group == group]
            assert len(selected) == 10 and sum(claim.truth for claim in selected) == 5
        assert all(evaluate_claim(target, claim) == claim.truth for claim in claims)
        with np.load(folder / "model_arrays.npz") as arrays:
            observations = arrays["observed"].copy()
        previous_answers = None
        for size in config.sample_sizes:
            dataset = folder / f"n{size}"
            record = json.loads((dataset / "result.json").read_text())
            assert record["fingerprint"] == manifest["fingerprint"]
            assert record["status"] in ("complete", "failed"), (index, size)
            with np.load(dataset / "observations.npz") as arrays:
                np.testing.assert_array_equal(arrays["observed"], observations[:size])
            if record["status"] == "failed":
                failed += 1
                failures.append({"scm": index, "n": size, "reason": record["error"]["message"]})
                continue
            complete += 1
            support = json.loads((dataset / "support.json").read_text())
            graphs = tuple(graph_from_record(item) for item in support["graphs"])
            assert graphs == tuple(sorted(set(graphs)))
            assert 0 < len(graphs) <= config.completion_limit
            assert all(not any(graph.undirected) for graph in graphs)
            assert record["truth_in_support"] == (target in graphs)
            prior = np.asarray(support["q0"])
            np.testing.assert_allclose(prior, np.full(len(graphs), 1 / len(graphs)))
            values = claim_value_matrix(graphs, claims)
            answers = [response["answers"] for response in record["responses"]]
            if previous_answers is not None:
                np.testing.assert_array_equal(answers, previous_answers)
            previous_answers = answers
            for response in record["responses"]:
                factors = np.where(values == np.asarray(response["answers"])[None, :],
                                   config.confidence, 1 - config.confidence)
                unnormalized = prior * np.prod(factors, axis=1)
                expected = unnormalized / unnormalized.sum()
                actual = np.asarray(response["posterior"])
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
                worst_error = max(worst_error, float(np.max(np.abs(actual - expected))))
                checked_updates += 1
    report = {
        "run_id": run_id, "fingerprint": manifest["fingerprint"],
        "completed_datasets": complete, "failed_datasets": failed,
        "independently_verified_updates": checked_updates,
        "maximum_absolute_probability_error": worst_error,
        "failures": failures,
    }
    atomic_json(root / "artifact_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_id), indent=2))
