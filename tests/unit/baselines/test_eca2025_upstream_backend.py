from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

from cd_llm.baselines.eca2025 import (
    ECAConfig,
    ECAPrior,
    EdgeConstraint,
    UPSTREAM_REVISION,
    fit_eca,
)
from cd_llm.baselines.eca2025.upstream_backend import (
    _default_upstream_root,
    _upstream_prior,
    _validated_inputs,
)
from cd_llm.core.finite_support import graph_from_edges


UPSTREAM_DEPENDENCIES_INSTALLED = all(
    importlib.util.find_spec(name) is not None for name in ("castle", "torch")
)
RUN_UPSTREAM_SMOKE = os.environ.get("RUN_ECA_UPSTREAM_SMOKE") == "1"


class UpstreamValidationTests(unittest.TestCase):
    def test_default_checkout_is_the_ignored_sibling_clone(self) -> None:
        expected = Path(__file__).resolve().parents[3] / "vendor" / "ECA"
        self.assertEqual(_default_upstream_root(), expected)

    def test_upstream_prior_requires_one_confidence_but_preserves_signs(self) -> None:
        mask = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
        probabilities = np.asarray(
            [[0.5, 0.8, 0.5], [0.2, 0.5, 0.5], [0.5, 0.5, 0.5]]
        )
        signs, confidence = _upstream_prior(mask, probabilities)
        self.assertEqual(confidence, 0.8)
        np.testing.assert_array_equal(
            signs, np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
        probabilities[1, 0] = 0.3
        with self.assertRaisesRegex(ValueError, "shared assertion confidence"):
            _upstream_prior(mask, probabilities)

    def test_upstream_native_validation_rejects_diagonal_and_boundary_priors(self) -> None:
        data = np.ones((4, 2))
        probabilities = np.full((2, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "self-edge"):
            _validated_inputs(data, np.eye(2, dtype=int), probabilities)
        mask = np.asarray([[0, 1], [0, 0]])
        probabilities[0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "strictly between"):
            _validated_inputs(data, mask, probabilities)


@unittest.skipUnless(
    UPSTREAM_DEPENDENCIES_INSTALLED and RUN_UPSTREAM_SMOKE,
    "set RUN_ECA_UPSTREAM_SMOKE=1 with optional gCastle and Torch dependencies",
)
class ActualUpstreamSmokeTests(unittest.TestCase):
    def test_archived_backend_executes_with_loaded_prior_and_returns_point_mass(self) -> None:
        rng = np.random.default_rng(28)
        source = rng.normal(size=24)
        target = 1.2 * source + 0.3 * rng.normal(size=24)
        data = np.column_stack((source, target))
        prior = ECAPrior(2, (EdgeConstraint(0, 1, 0.75),))
        # A deliberately high reporting threshold makes this one-iteration
        # integration smoke independent of incomplete acyclicity convergence.
        result = fit_eca(
            data,
            prior,
            config=ECAConfig(
                backend="upstream",
                max_iterations=1,
                threshold=10.0,
            ),
        )
        empty = graph_from_edges(2)
        self.assertEqual(result.graph, empty)
        self.assertEqual(result.support, {empty: 1.0})
        np.testing.assert_array_equal(result.binary_adjacency, np.zeros((2, 2)))
        self.assertTrue(np.all(np.isfinite(result.weighted_adjacency)))
        self.assertTrue(np.isfinite(result.objective))
        self.assertEqual(result.config.backend, "upstream")
        self.assertEqual(result.prior, prior)
        self.assertIn("returned normally", result.diagnostics.optimizer_message)
        self.assertEqual(UPSTREAM_REVISION, "c98002720dca8c170ba0f3d25ac3681e5def0bdc")


if __name__ == "__main__":
    unittest.main()
