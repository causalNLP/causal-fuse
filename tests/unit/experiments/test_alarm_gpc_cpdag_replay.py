"""The CPDAG replay must preserve the historical shared-skeleton search."""

import numpy as np
import pytest

from cd_llm.baselines.guess2graph2026.dag_cpdag import orient_skeleton_as_cpdag
from cd_llm.core.finite_support import Endpoint
from cd_llm.experiments.alarm_matched_disk import DiskCheckpoint
from cd_llm.experiments.alarm_matched_gpc import search
from cd_llm.experiments.alarm_matched_store import Checkpoint, Store


def test_separating_set_capture_preserves_search_and_orients_collider(tmp_path):
    guess = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    (tmp_path / "original").mkdir()
    (tmp_path / "replay").mkdir()
    calls = []

    def tester(left, right, conditioning):
        calls.append((left, right, tuple(conditioning)))
        return 0.8 if {left, right} == {0, 2} else 0.01

    original = search(np.zeros((4, 3)), guess, np.random.default_rng(7),
                      Store(tmp_path / "original"), "trial", tester=tester)
    original_calls = list(calls)
    calls.clear()
    replayed, diagnostics = search(
        np.zeros((4, 3)), guess, np.random.default_rng(7),
        Store(tmp_path / "replay"), "trial", tester=tester,
        return_diagnostics=True)
    assert np.array_equal(replayed, original)
    assert calls == original_calls
    assert diagnostics["ci_tests"] == len(calls)
    assert diagnostics["separating_sets"] == {"0:2": []}
    _, cpdag = orient_skeleton_as_cpdag(replayed, {(0, 2): ()})
    assert cpdag.mark(0, 1) == Endpoint.TAIL
    assert cpdag.mark(1, 0) == Endpoint.ARROW
    assert cpdag.mark(2, 1) == Endpoint.TAIL
    assert cpdag.mark(1, 2) == Endpoint.ARROW


def test_separating_sets_survive_a_checkpoint_resume(tmp_path):
    guess = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    tester = lambda left, right, conditioning: 0.8 if {left, right} == {0, 2} else 0.01
    with DiskCheckpoint(tmp_path, "trial") as checkpoint:
        with pytest.raises(Checkpoint):
            search(np.zeros((4, 3)), guess, np.random.default_rng(7), None, "trial",
                   deadline=0, tester=tester, checkpoint=checkpoint,
                   return_diagnostics=True)
    with DiskCheckpoint(tmp_path, "trial") as checkpoint:
        skeleton, diagnostics = search(
            np.zeros((4, 3)), guess, np.random.default_rng(7), None, "trial",
            tester=tester, checkpoint=checkpoint, return_diagnostics=True)
    assert skeleton[0, 2] == skeleton[2, 0] == 0
    assert diagnostics["separating_sets"]["0:2"] == []
