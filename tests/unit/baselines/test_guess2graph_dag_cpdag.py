from __future__ import annotations

import numpy as np

from cd_llm.baselines.guess2graph2026.dag_cpdag import (
    deterministic_topological_order,
    order_conditioning_sets,
    orient_skeleton_as_cpdag,
    run_native_gpc_guess_dag_cpdag,
)
from cd_llm.core.finite_support import Endpoint


def test_topological_order_matches_upstream_fifo_rule() -> None:
    dag = np.zeros((5, 5), dtype=int)
    dag[0, 2] = 1
    dag[1, 2] = 1
    dag[2, 4] = 1
    assert deterministic_topological_order(dag) == (0, 1, 3, 2, 4)


def test_conditioning_sets_with_no_late_node_form_first_block() -> None:
    order = (0, 1, 2, 3, 4)
    candidates = [(0,), (1,), (3,), (4,)]
    result = order_conditioning_sets(
        candidates, 1, 2, order, np.random.default_rng(7)
    )
    assert set(result[:2]) == {(0,), (1,)}
    assert set(result[2:]) == {(3,), (4,)}


def test_separating_set_orients_collider_and_certifies_cpdag() -> None:
    skeleton = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    raw, cpdag = orient_skeleton_as_cpdag(
        skeleton,
        {(0, 2): (), (2, 0): ()},
        node_names=("A", "B", "C"),
    )
    for graph in (raw, cpdag):
        assert graph.mark(0, 1) == Endpoint.TAIL
        assert graph.mark(1, 0) == Endpoint.ARROW
        assert graph.mark(2, 1) == Endpoint.TAIL
        assert graph.mark(1, 2) == Endpoint.ARROW


def test_middle_in_separating_set_keeps_chain_unoriented() -> None:
    skeleton = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    raw, cpdag = orient_skeleton_as_cpdag(
        skeleton,
        {(0, 2): (1,), (2, 0): (1,)},
    )
    for graph in (raw, cpdag):
        assert graph.mark(0, 1) == Endpoint.TAIL
        assert graph.mark(1, 0) == Endpoint.TAIL
        assert graph.mark(1, 2) == Endpoint.TAIL
        assert graph.mark(2, 1) == Endpoint.TAIL


def test_native_search_returns_symmetric_skeleton_and_cpdag() -> None:
    generator = np.random.default_rng(9)
    left = generator.binomial(1, 0.5, 400)
    middle = np.bitwise_xor(left, generator.binomial(1, 0.1, 400))
    right = np.bitwise_xor(middle, generator.binomial(1, 0.1, 400))
    data = np.column_stack((left, middle, right))
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    result = run_native_gpc_guess_dag_cpdag(
        data,
        dag,
        ci_test="chisq",
        rng=np.random.default_rng(3),
        node_names=("A", "B", "C"),
    )
    assert np.array_equal(result.skeleton, result.skeleton.T)
    assert not np.diag(result.skeleton).any()
    assert result.cpdag.kind == "cpdag"
    assert result.ci_tests > 0
