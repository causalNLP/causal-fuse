import json
from itertools import combinations
from pathlib import Path
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from cd_llm.experiments.alarm_matched_disk import DiskCheckpoint
from cd_llm.experiments.alarm_matched_gpc import search
from cd_llm.experiments.alarm_matched_store import Store, Transport, Checkpoint, MODEL
from cd_llm.experiments.alarm_matched_workers import WorkerStore, owner, merge_worker, audit_responses


def response():
    return dict(model=MODEL, choices=[dict(finish_reason="stop", message=dict(content="yes"))],
                usage=dict(prompt_tokens=17, completion_tokens=3))


def test_disk_mapping_matches_lazy_shuffle_across_restart(tmp_path):
    rng = np.random.default_rng(81)
    reference = {}
    checkpoint = DiskCheckpoint(tmp_path, 1)
    for cursor in range(20000):
        chosen = int(rng.integers(cursor, 200003))
        expected = reference.get(str(chosen), chosen)
        actual = checkpoint.mapping.get(str(chosen), chosen)
        assert actual == expected
        replacement = reference.pop(str(cursor), cursor)
        assert checkpoint.mapping.pop(str(cursor), cursor) == replacement
        if chosen != cursor:
            reference[str(chosen)] = replacement
            checkpoint.mapping[str(chosen)] = replacement
        if cursor == 10000:
            checkpoint.save(dict(cursor=cursor+1, swaps=checkpoint.mapping, rng=rng.bit_generator.state))
            checkpoint.close()
            checkpoint = DiskCheckpoint(tmp_path, 1)
            assert checkpoint.load()["cursor"] == 10001
    assert dict(checkpoint.mapping) == reference
    checkpoint.mapping.clear()
    assert len(checkpoint.mapping) == 0
    checkpoint.close()


def test_migrated_large_combination_search_is_exact(tmp_path):
    from cd_llm.baselines.guess2graph2026.paper_methods import expert_edge_order
    guess = np.ones((37, 37), dtype=int)-np.eye(37, dtype=int)
    seed = 147
    rng = np.random.default_rng(seed)
    edges = expert_edge_order(list(combinations(range(37), 2)), guess, rng)
    state = dict(matrix=guess.tolist(), edges=[list(e) for e in edges], edge=665, size=9,
                 cursor=0, tests=0, pool=None, permutation=None, swaps={}, rng=rng.bit_generator.state)
    a = tmp_path/"native"; a.mkdir()
    b = tmp_path/"disk"; b.mkdir()
    native, disk = Store(a), Store(b)
    native.put("gpc_progress", "1", state)
    expected_queries, actual_queries = [], []
    def tester(log):
        def call(a, b, z):
            log.append((a, b, tuple(z)))
            return float(len(log) == 37)
        return call
    data = np.zeros((2,37))
    expected = search(data, guess, np.random.default_rng(seed), native, "1", tester=tester(expected_queries))
    with DiskCheckpoint(b, 1) as checkpoint:
        checkpoint.import_state(state)
        clock = [0]
        real_test = tester(actual_queries)
        def interrupted(a,b,z):
            clock[0] += 1
            return real_test(a,b,z)
        with patch("cd_llm.experiments.alarm_matched_gpc.time.time", lambda: clock[0]):
            with pytest.raises(Checkpoint):
                search(data, guess, np.random.default_rng(seed), disk, "1", deadline=13, tester=interrupted, checkpoint=checkpoint)
    with DiskCheckpoint(b, 1) as checkpoint:
        assert checkpoint.load()["tests"] == 13
        actual = search(data, guess, np.random.default_rng(seed), disk, "1", tester=tester(actual_queries), checkpoint=checkpoint)
    assert np.array_equal(actual, expected)
    assert actual_queries == expected_queries


def test_uncommitted_local_changes_do_not_replace_durable_state(tmp_path):
    with DiskCheckpoint(tmp_path, 1) as checkpoint:
        checkpoint.mapping["123"] = 45
        checkpoint.save(dict(tests=12, swaps=checkpoint.mapping))
        checkpoint.mapping["123"] = 99
    with DiskCheckpoint(tmp_path, 1) as checkpoint:
        assert checkpoint.load()["tests"] == 12
        assert checkpoint.mapping["123"] == 45


def test_worker_reuses_base_and_merge_is_idempotent(tmp_path):
    main = tmp_path/"main"; main.mkdir()
    master = Store(main)
    request = dict(messages=[dict(role="user",content="question")], max_tokens=256)
    calls = []
    def sender(req):
        calls.append(req)
        return response()
    Transport(master, "scp:0:1:2", "r", online=True, sender=sender).query(request)
    worker = WorkerStore(tmp_path/"worker", master.path)
    Transport(worker, "scp:1:1:2", "r", online=True, sender=sender).query(request)
    assert len(calls) == 1
    Transport(worker, "scp:1:2:3", "r", online=True, sender=sender).query(dict(request, seed=1))
    worker.put("scp", "1:1:2", dict(trials=[1]*5))
    merge_worker(master, worker.path)
    merge_worker(master, worker.path)
    assert len(calls) == 2
    with master.db() as db:
        assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM uses").fetchone()[0] == 3
    assert master.usage("")["total_tokens"] == 60
    assert master.get("scp", "1:1:2")["trials"] == [1]*5


def test_merge_rejects_conflicting_use_and_rolls_back(tmp_path):
    main = tmp_path/"main"; main.mkdir()
    master = Store(main)
    request = dict(messages=[dict(role="user",content="question")], max_tokens=256)
    Transport(master, "a", "r", online=True, sender=lambda _: response()).query(request)
    worker = WorkerStore(tmp_path/"worker", master.path)
    with worker.db() as db:
        db.execute("UPDATE uses SET key='conflicting' WHERE unit='a'")
    with pytest.raises(ValueError, match="conflicting worker uses"):
        merge_worker(master, worker.path)
    with master.db() as db:
        assert db.execute("SELECT key FROM uses WHERE unit='a'").fetchone()[0] != "conflicting"


def test_identical_prompt_families_have_one_owner():
    keys = [(i,a,b) for i in range(20) for a in range(37) for b in range(37) if a != b]
    assignments = [{key for key in keys if owner(str(key), "runtime") == shard} for shard in range(4)]
    assert sum(map(len, assignments)) == 26640
    assert len(set.union(*assignments)) == 26640
    assert owner("same full prompt", "runtime") == owner("same full prompt", "runtime")
    assert all(assignments)


def test_probability_settings_audit_rejects_wrong_temperature(tmp_path):
    store = Store(tmp_path)
    request = dict(messages=[dict(role="user",content="question")], max_tokens=256, temperature=.7)
    Transport(store, "scp:0:1:2", "r", online=True, sender=lambda _: response()).query(request)
    audit_responses(store, "r", dict(temperature=0, scp=dict(temperature=.7)))
    with pytest.raises(ValueError, match="temperature"):
        audit_responses(store, "r", dict(temperature=0, scp=dict(temperature=0)))


def test_early_fit_requires_its_own_method_audit(tmp_path):
    from cd_llm.experiments.alarm_matched_baselines import fit
    store = Store(tmp_path)
    store.put("method_audit", "bfs:0", dict(passed=True))
    with pytest.raises(ValueError, match="audit"):
        fit(tmp_path, 0, float("inf"), only="long")


def test_scheduler_preserves_shard_and_deduplicates(tmp_path, monkeypatch):
    from cd_llm.experiments import alarm_matched_execution as execution
    monkeypatch.setattr(execution.base, "ROOT", tmp_path)
    root = tmp_path/"results"/"timestamp"/"run"; root.mkdir(parents=True)
    active, commands = {}, []
    monkeypatch.setattr(execution, "live_jobs", lambda: active)
    def run(cmd, **kwargs):
        from types import SimpleNamespace
        job = str(100+len(commands))
        commands.append((cmd, kwargs))
        active[job] = next(x.split("=",1)[1] for x in cmd if x.startswith("--comment="))
        return SimpleNamespace(stdout=job+"\n")
    monkeypatch.setattr(execution.subprocess, "run", run)
    assert execution.submit(root, "scp", 2) == "100"
    assert execution.submit(root, "scp", 2) == "100"
    assert len(commands) == 2  # worker plus its failure-surviving coordinator
    assert commands[0][1]["env"]["ALARM_EXEC_INDEX"] == "2"
    assert "--dependency=afterany:100" in commands[1][0]


def test_oom_is_recorded_and_not_treated_as_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cd_llm.experiments import alarm_matched_execution as execution
    from cd_llm.experiments.common.artifacts import write_json, read_json
    folder = execution.folder(tmp_path)
    write_json(folder/"jobs.json", [dict(stage="gpc", index=5, job="101", token="test")])
    monkeypatch.setattr(execution, "live_jobs", lambda: {})
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="101|OUT_OF_MEMORY\n"))
    execution.reconcile_failures(tmp_path)
    assert read_json(folder/"failure_gpc_5.json")["automatic_retry"] is False
