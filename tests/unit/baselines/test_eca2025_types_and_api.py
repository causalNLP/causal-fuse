from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cd_llm.baselines.eca2025.api as api_module
from cd_llm.baselines.eca2025 import (
    ECAConfig,
    ECADiagnostics,
    ECAPrior,
    ECAResult,
    EdgeConstraint,
)
from cd_llm.core.finite_support import Claim, EvidenceGraph, EvidenceNode, graph_from_edges


def _diagnostics() -> ECADiagnostics:
    return ECADiagnostics(
        iterations=2,
        inner_optimizations=3,
        objective_evaluations=12,
        gradient_evaluations=12,
        converged=True,
        acyclicity=1e-10,
        rho=10.0,
        alpha=0.2,
        runtime_seconds=0.01,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="fixture",
    )


class PublicTypeTests(unittest.TestCase):
    def test_prior_is_sorted_immutable_and_uses_source_target_orientation(self) -> None:
        prior = ECAPrior(
            3,
            (
                EdgeConstraint(2, 0, 0.2),
                EdgeConstraint(0, 1, 0.8),
            ),
        )
        self.assertEqual(
            prior.constraints,
            (EdgeConstraint(0, 1, 0.8), EdgeConstraint(2, 0, 0.2)),
        )
        mask, probabilities = prior.as_matrices()
        expected_mask = np.zeros((3, 3), dtype=np.int8)
        expected_mask[0, 1] = 1
        expected_mask[2, 0] = 1
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertEqual(probabilities[0, 1], 0.8)
        self.assertEqual(probabilities[2, 0], 0.2)
        self.assertEqual(probabilities[1, 2], 0.5)
        with self.assertRaises(ValueError):
            mask[0, 1] = 0
        with self.assertRaises(ValueError):
            probabilities[0, 1] = 0.1
        with self.assertRaises(FrozenInstanceError):
            prior.node_count = 4  # type: ignore[misc]

    def test_prior_matrix_round_trip_and_validation(self) -> None:
        mask = np.zeros((3, 3), dtype=int)
        mask[1, 2] = 1
        probabilities = np.full((3, 3), 0.5)
        probabilities[1, 2] = 0.73
        prior = ECAPrior.from_matrices(mask, probabilities)
        self.assertEqual(prior, ECAPrior(3, (EdgeConstraint(1, 2, 0.73),)))
        mask[1, 2] = 0
        probabilities[1, 2] = 0.1
        self.assertEqual(prior.constraints[0].probability, 0.73)

        with self.assertRaisesRegex(ValueError, "square"):
            ECAPrior.from_matrices(np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "same shape"):
            ECAPrior.from_matrices(np.zeros((2, 2)), np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "binary"):
            ECAPrior.from_matrices(np.asarray([[0, 2], [0, 0]]), np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "diagonal"):
            ECAPrior.from_matrices(np.eye(2), np.full((2, 2), 0.5))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            ECAPrior.from_matrices(
                np.asarray([[0, 1], [0, 0]]),
                np.asarray([[0.5, 1.1], [0.5, 0.5]]),
            )

    def test_constraint_and_prior_reject_ambiguous_edges(self) -> None:
        with self.assertRaises(TypeError):
            EdgeConstraint(True, 1, 0.8)
        with self.assertRaisesRegex(ValueError, "self-edge"):
            EdgeConstraint(1, 1, 0.8)
        with self.assertRaisesRegex(ValueError, "at most 1"):
            EdgeConstraint(0, 1, 1.01)
        with self.assertRaisesRegex(ValueError, "outside"):
            ECAPrior(2, (EdgeConstraint(0, 2, 0.8),))
        with self.assertRaisesRegex(ValueError, "multiple probabilities"):
            ECAPrior(
                2,
                (EdgeConstraint(0, 1, 0.2), EdgeConstraint(0, 1, 0.8)),
            )

    def test_config_defaults_and_domain_validation(self) -> None:
        self.assertEqual(
            ECAConfig(),
            ECAConfig(
                backend="paper",
                lambda1=0.05,
                sigma=1.0,
                threshold=0.3,
                max_iterations=100,
                h_tolerance=1e-8,
                rho_max=1e16,
                rho_multiplier=10.0,
                standardize_data=False,
            ),
        )
        for kwargs, pattern in (
            ({"backend": "unknown"}, "paper.*upstream"),
            ({"lambda1": -0.1}, "at least"),
            ({"sigma": 0.0}, "positive"),
            ({"threshold": 0.0}, "positive"),
            ({"max_iterations": 0}, "positive"),
            ({"rho_max": 1.0}, "exceed"),
            ({"rho_multiplier": 1.0}, "exceed"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    ECAConfig(**kwargs)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ECAConfig(standardize_data=1)  # type: ignore[arg-type]

    def test_result_is_a_read_only_point_mass_with_consistent_scores(self) -> None:
        graph = graph_from_edges(2, [(0, 1)])
        weighted = np.asarray([[0.0, 0.75], [0.0, 0.0]])
        binary = np.asarray([[0, 1], [0, 0]])
        result = ECAResult(
            graph=graph,
            weighted_adjacency=weighted,
            binary_adjacency=binary,
            objective=2.5,
            data_loss=1.5,
            prior_log_likelihood=-0.4,
            prior_penalty=0.4,
            l1_penalty=0.6,
            diagnostics=_diagnostics(),
            config=ECAConfig(),
            prior=ECAPrior.empty(2),
        )
        weighted[0, 1] = 9.0
        binary[0, 1] = 0
        self.assertEqual(result.weighted_adjacency[0, 1], 0.75)
        self.assertEqual(result.binary_adjacency[0, 1], 1)
        self.assertEqual(dict(result.support), {graph: 1.0})
        self.assertTrue(result.converged)
        with self.assertRaises(TypeError):
            result.support[graph] = 0.5  # type: ignore[index]
        with self.assertRaises(ValueError):
            result.weighted_adjacency[0, 1] = 0.0
        with self.assertRaises(ValueError):
            result.binary_adjacency[0, 1] = 0

        with self.assertRaisesRegex(ValueError, "objective"):
            ECAResult(
                graph=graph,
                weighted_adjacency=np.asarray([[0.0, 0.75], [0.0, 0.0]]),
                binary_adjacency=np.asarray([[0, 1], [0, 0]]),
                objective=99.0,
                data_loss=1.5,
                prior_log_likelihood=-0.4,
                prior_penalty=0.4,
                l1_penalty=0.6,
                diagnostics=_diagnostics(),
                config=ECAConfig(),
                prior=ECAPrior.empty(2),
            )
        for invalid_binary in (
            np.asarray([[0.0, 0.5], [0.0, 0.0]]),
            np.asarray([[0, 256], [0, 0]]),
        ):
            with self.subTest(invalid_binary=invalid_binary):
                with self.assertRaisesRegex(ValueError, "zero or one"):
                    ECAResult(
                        graph=graph,
                        weighted_adjacency=np.asarray(
                            [[0.0, 0.75], [0.0, 0.0]]
                        ),
                        binary_adjacency=invalid_binary,
                        objective=2.5,
                        data_loss=1.5,
                        prior_log_likelihood=-0.4,
                        prior_penalty=0.4,
                        l1_penalty=0.6,
                        diagnostics=_diagnostics(),
                        config=ECAConfig(),
                        prior=ECAPrior.empty(2),
                    )


class APIDispatchAndValidationTests(unittest.TestCase):
    @staticmethod
    def _paper_backend_fixture() -> SimpleNamespace:
        graph = graph_from_edges(2, [(0, 1)])
        return SimpleNamespace(
            graph=graph,
            weighted_adjacency=np.asarray([[0.0, 0.7], [0.0, 0.0]]),
            binary_adjacency=np.asarray([[0, 1], [0, 0]]),
            terms=SimpleNamespace(
                objective=2.0,
                data_loss=1.8,
                prior_log_score=0.0,
                l1_penalty=0.2,
            ),
            diagnostics=SimpleNamespace(
                outer_iterations=4,
                inner_optimizations=6,
                objective_evaluations=20,
                gradient_evaluations=20,
                converged=True,
                acyclicity=0.0,
                rho=100.0,
                alpha=0.3,
                runtime_seconds=0.1,
                optimizer_success=True,
                optimizer_status=0,
                optimizer_message="paper fixture",
            ),
        )

    @staticmethod
    def _upstream_backend_fixture() -> SimpleNamespace:
        graph = graph_from_edges(2)
        return SimpleNamespace(
            graph=graph,
            weighted_adjacency=np.zeros((2, 2)),
            binary_adjacency=np.zeros((2, 2), dtype=np.int8),
            terms=SimpleNamespace(
                objective=2.0,
                data_loss=1.5,
                prior_log_score=-0.5,
                prior_penalty=0.2,
                l1_penalty=0.3,
            ),
            diagnostics=SimpleNamespace(
                acyclicity=0.0,
                fit_seconds=0.2,
            ),
        )

    def test_default_dispatches_every_config_value_to_paper_backend(self) -> None:
        data = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
        prior = ECAPrior(2, (EdgeConstraint(0, 1, 0.8),))
        config = ECAConfig(
            lambda1=0.07,
            sigma=1.2,
            threshold=0.25,
            max_iterations=17,
            h_tolerance=1e-7,
            rho_max=1e9,
            rho_multiplier=5.0,
            standardize_data=True,
        )
        fixture = self._paper_backend_fixture()
        with patch.object(
            api_module, "fit_paper_eca", return_value=fixture
        ) as paper_fit, patch.object(api_module, "fit_upstream_eca") as upstream_fit:
            result = api_module.fit_eca(data, prior, config=config)
        upstream_fit.assert_not_called()
        args, kwargs = paper_fit.call_args
        self.assertIs(args[0], data)
        np.testing.assert_array_equal(args[1], prior.mask)
        np.testing.assert_array_equal(args[2], prior.probabilities)
        self.assertEqual(
            kwargs,
            {
                "lambda1": 0.07,
                "sigma": 1.2,
                "threshold": 0.25,
                "standardize_data": True,
                "max_outer_iterations": 17,
                "max_inner_iterations": 17,
                "h_tolerance": 1e-7,
                "rho_max": 1e9,
                "rho_multiplier": 5.0,
            },
        )
        self.assertEqual(result.graph, fixture.graph)
        self.assertEqual(result.prior, prior)
        self.assertEqual(result.prior_penalty, 0.0)
        self.assertEqual(result.diagnostics.iterations, 4)

    def test_upstream_dispatch_is_explicit_and_forwards_checkout(self) -> None:
        data = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        config = ECAConfig(
            backend="upstream",
            lambda1=0.08,
            sigma=0.9,
            threshold=0.4,
            max_iterations=9,
            h_tolerance=1e-6,
            rho_max=1e8,
            standardize_data=True,
        )
        fixture = self._upstream_backend_fixture()
        checkout = Path("/tmp/eca-upstream-fixture")
        with patch.object(api_module, "fit_paper_eca") as paper_fit, patch.object(
            api_module, "fit_upstream_eca", return_value=fixture
        ) as upstream_fit:
            result = api_module.fit_eca(
                data, config=config, upstream_root=checkout
            )
        paper_fit.assert_not_called()
        args, kwargs = upstream_fit.call_args
        self.assertIs(args[0], data)
        np.testing.assert_array_equal(args[1], np.zeros((2, 2), dtype=np.int8))
        np.testing.assert_array_equal(args[2], np.full((2, 2), 0.5))
        self.assertEqual(kwargs["upstream_root"], checkout)
        self.assertEqual(kwargs["max_outer_iterations"], 9)
        self.assertEqual(kwargs["standardize_data"], True)
        self.assertEqual(result.objective, 2.0)
        self.assertEqual(result.prior_penalty, 0.2)
        self.assertIn("not exposed", result.diagnostics.optimizer_message)

    def test_evidence_entry_point_attaches_the_complete_adaptation_audit(self) -> None:
        data = np.asarray([[0.0, 0.0], [1.0, 1.2], [-1.0, -1.1]])
        claims = (Claim("edge", "I", "directed", 0, 0, 1),)
        evidence = EvidenceGraph(
            "target",
            (
                EvidenceNode(
                    "ev",
                    0,
                    1,
                    3.7 / 4.7,
                    "source",
                    "factor",
                    "target",
                    True,
                    False,
                ),
            ),
        )
        fixture = self._paper_backend_fixture()
        with patch.object(api_module, "fit_paper_eca", return_value=fixture) as paper_fit:
            result = api_module.fit_eca_with_evidence(data, claims, evidence)
        self.assertIsNotNone(result.evidence_adaptation)
        assert result.evidence_adaptation is not None
        self.assertEqual(result.evidence_adaptation.used_evidence_ids, ("ev",))
        self.assertEqual(result.evidence_adaptation.prior, result.prior)
        self.assertEqual(result.prior.constraints[0].source, 0)
        self.assertEqual(result.prior.constraints[0].target, 1)
        _, mask, probabilities = paper_fit.call_args.args
        self.assertEqual(mask[0, 1], 1)
        self.assertGreater(probabilities[0, 1], 0.5)

    def test_api_rejects_invalid_data_prior_and_backend_specific_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            api_module.fit_eca(np.empty((0, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            api_module.fit_eca(np.asarray([[0.0, math.nan]]))
        with self.assertRaisesRegex(ValueError, "node count"):
            api_module.fit_eca(np.ones((4, 2)), ECAPrior.empty(3))
        with self.assertRaises(TypeError):
            api_module.fit_eca(np.ones((4, 2)), config="paper")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "upstream_root"):
            api_module.fit_eca(
                np.ones((4, 2)),
                config=ECAConfig(),
                upstream_root=Path("/tmp/not-used"),
            )
        with self.assertRaisesRegex(ValueError, "hard-codes"):
            api_module.fit_eca(
                np.ones((4, 2)),
                config=ECAConfig(backend="upstream", rho_multiplier=2.0),
            )


if __name__ == "__main__":
    unittest.main()
