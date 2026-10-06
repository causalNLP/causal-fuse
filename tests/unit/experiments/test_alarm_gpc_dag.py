from unittest.mock import patch

import numpy as np
import pytest

from cd_llm.experiments.alarm_matched_gpc import search_dag
from cd_llm.experiments.alarm_matched_store import Checkpoint, Store


def _chain_with_isolate():
    dag = np.zeros((4, 4), dtype=int)
    dag[0, 1] = 1
    dag[1, 2] = 1
    return dag


def test_dag_guidance_tests_d_separators_before_other_sets(tmp_path):
    calls = []

    def dependent(left, right, conditioning):
        calls.append((min(left, right), max(left, right), tuple(conditioning)))
        return 0.01

    search_dag(
        np.zeros((3, 4)),
        _chain_with_isolate(),
        np.random.default_rng(91),
        Store(tmp_path),
        "condition",
        tester=dependent,
    )
    pair_calls = [conditioning for left, right, conditioning in calls if (left, right) == (0, 2)]
    assert pair_calls[0] == ()
    assert [value for value in pair_calls if len(value) == 1] == [(1,), (3,)]


def test_dag_guided_search_restart_replays_exact_query_order(tmp_path):
    data = np.zeros((3, 4))
    dag = _chain_with_isolate()
    expected_calls = []
    actual_calls = []

    def tester(log):
        def call(left, right, conditioning):
            log.append((left, right, tuple(conditioning)))
            return 0.01
        return call

    (tmp_path / "expected").mkdir()
    expected = search_dag(
        data,
        dag,
        np.random.default_rng(44),
        Store(tmp_path / "expected"),
        "condition",
        tester=tester(expected_calls),
    )
    clock = [0]

    def interrupted(left, right, conditioning):
        clock[0] += 1
        return tester(actual_calls)(left, right, conditioning)

    (tmp_path / "actual").mkdir()
    actual_store = Store(tmp_path / "actual")
    with patch("cd_llm.experiments.alarm_matched_gpc.time.time", lambda: clock[0]):
        with pytest.raises(Checkpoint):
            search_dag(
                data,
                dag,
                np.random.default_rng(44),
                actual_store,
                "condition",
                deadline=4,
                tester=interrupted,
            )
    actual = search_dag(
        data,
        dag,
        np.random.default_rng(44),
        actual_store,
        "condition",
        tester=tester(actual_calls),
    )
    assert np.array_equal(actual, expected)
    assert actual_calls == expected_calls


def test_dag_checkpoint_rejects_a_different_expert_graph(tmp_path):
    data = np.zeros((3, 4))
    store = Store(tmp_path)
    search_dag(data, _chain_with_isolate(), np.random.default_rng(2), store, "condition", tester=lambda *_: 0.01)
    changed = _chain_with_isolate()
    changed[2, 3] = 1
    with pytest.raises(ValueError, match="different expert-DAG protocol"):
        search_dag(data, changed, np.random.default_rng(2), store, "condition", tester=lambda *_: 0.01)
