from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.cpc import (
    CPCBackendProvenance,
    CPCRawResult,
    CPCSettings,
    certify_cpdag,
    fit_cpc,
    learn_cpc,
)
from cd_llm.core.finite_support import Endpoint, PartialGraph


def partial(matrix: list[list[int]]) -> PartialGraph:
    return PartialGraph.from_matrix(np.asarray(matrix, dtype=int), "cpdag")


def skeleton(graph: PartialGraph) -> set[tuple[int, int]]:
    return {
        (first, second)
        for first in range(graph.node_count)
        for second in range(first + 1, graph.node_count)
        if graph.mark(first, second) != Endpoint.NULL
    }


class CPCValidityTests(unittest.TestCase):
    def test_canonical_collider_is_returned_unchanged(self) -> None:
        collider = partial(
            [
                [0, 0, Endpoint.TAIL],
                [0, 0, Endpoint.TAIL],
                [Endpoint.ARROW, Endpoint.ARROW, 0],
            ]
        )
        fit = fit_cpc(
            np.ones((8, 3)),
            backend_runner=lambda *_: CPCRawResult(
                collider,
                ((1, 2, 0),),
                CPCBackendProvenance(java_version="21"),
            ),
        )
        self.assertEqual(fit.graph, collider)
        self.assertEqual(fit.diagnostics.resolution, "identity")
        self.assertEqual(fit.diagnostics.ambiguous_triples, ((0, 2, 1),))
        self.assertTrue(fit.diagnostics.certified)
        self.assertTrue(fit.diagnostics.canonical_fixed_point)

    def test_extendible_noncanonical_pdag_is_canonicalized(self) -> None:
        directed_chain = partial(
            [
                [0, Endpoint.TAIL, 0],
                [Endpoint.ARROW, 0, Endpoint.TAIL],
                [0, Endpoint.ARROW, 0],
            ]
        )
        fit = fit_cpc(
            np.ones((8, 3)), backend_runner=lambda *_: directed_chain
        )
        self.assertEqual(fit.diagnostics.resolution, "canonicalized")
        self.assertTrue(fit.diagnostics.normalization_applied)
        self.assertFalse(fit.diagnostics.projection_applied)
        self.assertEqual(skeleton(fit.graph), {(0, 1), (1, 2)})
        self.assertTrue(
            all(
                fit.graph.mark(first, second) == Endpoint.TAIL
                and fit.graph.mark(second, first) == Endpoint.TAIL
                for first, second in skeleton(fit.graph)
            )
        )

    def test_directed_cycle_is_projected_deterministically(self) -> None:
        cycle = partial(
            [
                [0, Endpoint.TAIL, Endpoint.ARROW],
                [Endpoint.ARROW, 0, Endpoint.TAIL],
                [Endpoint.TAIL, Endpoint.ARROW, 0],
            ]
        )
        first = fit_cpc(np.ones((8, 3)), backend_runner=lambda *_: cycle)
        second = fit_cpc(np.ones((8, 3)), backend_runner=lambda *_: cycle)
        self.assertEqual(first.graph, second.graph)
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertEqual(first.diagnostics.resolution, "projected")
        self.assertEqual(first.diagnostics.cyclic_sccs, ((0, 1, 2),))
        self.assertTrue(first.diagnostics.reversed_preferences)
        self.assertEqual(skeleton(first.graph), skeleton(cycle))
        self.assertEqual(certify_cpdag(first.graph), first.graph)

    def test_acyclic_nonextendible_pdag_is_projected(self) -> None:
        nonextendible = partial(
            [
                [0, Endpoint.TAIL, 0, Endpoint.TAIL],
                [Endpoint.ARROW, 0, Endpoint.TAIL, 0],
                [0, Endpoint.ARROW, 0, Endpoint.TAIL],
                [Endpoint.TAIL, 0, Endpoint.ARROW, 0],
            ]
        )
        fit = fit_cpc(
            np.ones((8, 4)), backend_runner=lambda *_: nonextendible
        )
        self.assertEqual(fit.diagnostics.resolution, "projected")
        self.assertEqual(fit.diagnostics.cyclic_sccs, ())
        self.assertEqual(skeleton(fit.graph), skeleton(nonextendible))
        self.assertTrue(fit.diagnostics.certified)

    def test_illegal_endpoint_pair_is_projected_without_losing_skeleton(self) -> None:
        bidirected = partial(
            [
                [0, Endpoint.ARROW, 0],
                [Endpoint.ARROW, 0, Endpoint.TAIL],
                [0, Endpoint.TAIL, 0],
            ]
        )
        fit = fit_cpc(np.ones((8, 3)), backend_runner=lambda *_: bidirected)
        self.assertEqual(fit.diagnostics.resolution, "projected")
        self.assertEqual(skeleton(fit.graph), skeleton(bidirected))
        self.assertTrue(fit.diagnostics.legal_endpoint_pairs)


class CPCAdapterContractTests(unittest.TestCase):
    def test_continuous_settings_are_locked(self) -> None:
        seen: list[tuple[np.ndarray, CPCSettings, tuple[str, ...]]] = []
        empty = partial([[0, 0], [0, 0]])

        def runner(data, settings, names):
            seen.append((data, settings, names))
            return empty

        fit_cpc(
            np.asarray([[1, 2], [3, 4]], dtype=int),
            alpha=0.01,
            node_names=("A", "B"),
            backend_runner=runner,
        )
        data, settings, names = seen[0]
        self.assertEqual(data.dtype, np.float64)
        self.assertEqual(names, ("A", "B"))
        self.assertEqual(settings.independence_test, "fisher_z")
        self.assertEqual(settings.singularity_lambda, 0.0)
        self.assertEqual(settings.depth, -1)
        self.assertTrue(settings.stable_fas)
        self.assertFalse(settings.allow_bidirected)

    def test_discrete_settings_are_locked(self) -> None:
        seen: list[CPCSettings] = []
        empty = partial([[0, 0], [0, 0]])

        def runner(data, settings, _names):
            self.assertEqual(data.dtype, np.int64)
            seen.append(settings)
            return empty

        fit_cpc(
            np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            variable_type="discrete",
            backend_runner=runner,
        )
        settings = seen[0]
        self.assertEqual(settings.independence_test, "g_square")
        self.assertEqual(settings.min_count, 1)
        self.assertEqual(settings.cell_table_type, 1)
        self.assertIsNone(settings.singularity_lambda)

    def test_discrete_data_rejects_fractional_categories(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            fit_cpc(
                np.asarray([[0.5], [1.0]]),
                variable_type="discrete",
                backend_runner=lambda *_: partial([[0]]),
            )

    def test_finite_support_learn_pc_delegates_to_cpc(self) -> None:
        expected = partial([[0, Endpoint.TAIL], [Endpoint.TAIL, 0]])
        with patch("cd_llm.core.cpc.learn_cpc", return_value=expected) as delegated:
            from cd_llm.core.finite_support import learn_pc

            observed = learn_pc(np.ones((4, 2)), alpha=0.02)
        self.assertEqual(observed, expected)
        delegated.assert_called_once()
        self.assertEqual(delegated.call_args.args[1], 0.02)
        self.assertEqual(delegated.call_args.kwargs["variable_type"], "continuous")


@unittest.skipUnless(
    os.environ.get("RUN_TETRAD_CPC_SMOKE") == "1", "Tetrad CPC smoke is opt-in"
)
class TetradCPCSmokeTests(unittest.TestCase):
    def test_fisher_z_and_g_square_cpc(self) -> None:
        rng = np.random.default_rng(17)
        parents = rng.normal(size=(5_000, 2))
        child = parents[:, 0] + parents[:, 1] + 0.15 * rng.normal(size=5_000)
        continuous = np.column_stack((parents, child))
        discrete = (continuous > 0).astype(int)
        continuous_graph = learn_cpc(continuous, alpha=0.01)
        self.assertEqual(continuous_graph.mark(0, 2), Endpoint.TAIL)
        self.assertEqual(continuous_graph.mark(2, 0), Endpoint.ARROW)
        self.assertEqual(continuous_graph.mark(1, 2), Endpoint.TAIL)
        self.assertEqual(continuous_graph.mark(2, 1), Endpoint.ARROW)
        for graph in (
            continuous_graph,
            learn_cpc(discrete, alpha=0.05, variable_type="discrete"),
        ):
            self.assertEqual(graph.kind, "cpdag")
            self.assertEqual(certify_cpdag(graph), graph)


if __name__ == "__main__":
    unittest.main()
