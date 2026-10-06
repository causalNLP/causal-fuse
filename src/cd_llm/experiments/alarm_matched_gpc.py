"""Exact, resumable DAG-guided gPC-Guess conditioning search."""
from itertools import combinations
from math import comb
import hashlib
import json
import time

import numpy as np

from cd_llm.experiments.alarm_local_support import unrank_combination
from cd_llm.experiments.alarm_matched_store import Checkpoint


def _digest_dag(dag):
    payload = json.dumps(np.asarray(dag, dtype=int).tolist(), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def search_dag(data, guess, rng, store, key, deadline=float("inf"), tester=None, checkpoint=None,
               return_diagnostics=False):
    """Run Algorithm gPC-Guess using exact expert-DAG d-separation order."""
    from causallearn.utils.cit import CIT
    from cd_llm.baselines.guess2graph2026.paper_methods import (
        DAGDSeparationOracle, expert_edge_order, neighbors, validate_guessed_dag,
    )

    dag = validate_guessed_dag(guess, data.shape[1])
    dag_digest = _digest_dag(dag)
    skeleton = ((dag != 0) | (dag.T != 0)).astype(int)
    oracle = DAGDSeparationOracle(dag)
    cit = tester or CIT(data, "gsq")
    edges = expert_edge_order(list(combinations(range(data.shape[1]), 2)), skeleton, rng)
    state = checkpoint.load() if checkpoint is not None else store.get("gpc_progress", key)
    if state is None:
        state = dict(
            version=2, expert_dag_sha256=dag_digest,
            matrix=(np.ones((data.shape[1], data.shape[1]), dtype=int) - np.eye(data.shape[1], dtype=int)).tolist(),
            edges=[list(edge) for edge in edges], edge=0, size=0, phase="d_first",
            cursor=0, tests=0, dsep_checks=0, separating_sets={}, pool=None,
            permutation=None, swaps={},
        )
        if checkpoint is not None:
            state["swaps"] = checkpoint.mapping
    else:
        if state.get("version") != 2 or state.get("expert_dag_sha256") != dag_digest:
            raise ValueError("gPC checkpoint belongs to a different expert-DAG protocol")
        if state["edges"] != [list(edge) for edge in edges]:
            raise ValueError("gPC edge order changed")
        rng.bit_generator.state = state["rng"]
    matrix = np.asarray(state["matrix"], dtype=int)
    last = time.monotonic()

    def save():
        nonlocal last
        state.update(matrix=matrix.tolist(), rng=rng.bit_generator.state)
        if checkpoint is None:
            store.put("gpc_progress", key, state)
        else:
            checkpoint.save(state)
            from cd_llm.experiments.common.artifacts import write_json
            write_json(checkpoint.folder / "progress.json", dict(
                edge=state["edge"], size=state["size"], phase=state["phase"],
                cursor=state["cursor"], tests=state["tests"],
                dsep_checks=state["dsep_checks"], updated=time.time()))
        last = time.monotonic()

    def reset_shuffle():
        state["swaps"].clear()
        state.update(cursor=0, permutation=None)

    while state["edge"] < len(edges):
        left, right = edges[state["edge"]]
        if not matrix[left, right]:
            state.update(edge=state["edge"] + 1, size=0, phase="d_first", pool=None)
            reset_shuffle()
            save()
            continue
        if state["pool"] is None:
            left_pool = neighbors(matrix, left, exclude=right)
            right_pool = neighbors(matrix, right, exclude=left)
            state["pool"] = left_pool if len(left_pool) <= len(right_pool) else right_pool
        pool = state["pool"]
        removed = False
        while state["size"] <= len(pool) and not removed:
            count = comb(len(pool), state["size"])
            if state["permutation"] is None and count <= 100000:
                values = list(range(count))
                rng.shuffle(values)
                state["permutation"] = values
            while state["cursor"] < count:
                if time.time() >= deadline:
                    save()
                    raise Checkpoint("gPC checkpoint within DAG-guided conditioning order")
                index = state["cursor"]
                if state["permutation"] is not None:
                    rank = state["permutation"][index]
                else:
                    chosen = int(rng.integers(index, count))
                    rank = state["swaps"].get(str(chosen), chosen)
                    replacement = state["swaps"].pop(str(index), index)
                    if chosen != index:
                        state["swaps"][str(chosen)] = replacement
                conditioning = unrank_combination(pool, state["size"], rank)
                predicted_separation = oracle(left, right, conditioning)
                state["dsep_checks"] += 1
                state["cursor"] += 1
                selected = predicted_separation if state["phase"] == "d_first" else not predicted_separation
                if selected:
                    value = cit(left, right, list(conditioning))
                    if hasattr(cit, "pvalue_cache") and len(cit.pvalue_cache) > 100000:
                        cit.pvalue_cache.clear()
                    state["tests"] += 1
                    if value > .05:
                        matrix[left, right] = matrix[right, left] = 0
                        state["separating_sets"][f"{left}:{right}"] = list(conditioning)
                        removed = True
                        break
                if time.monotonic() - last > (300 if checkpoint is not None else 30):
                    save()
            if removed:
                break
            if state["phase"] == "d_first":
                state["phase"] = "d_other"
                reset_shuffle()
            else:
                state.update(size=state["size"] + 1, phase="d_first")
                reset_shuffle()
        state.update(edge=state["edge"] + 1, size=0, phase="d_first", pool=None)
        reset_shuffle()
        save()
    diagnostics = dict(
        separating_sets=state["separating_sets"], ci_tests=state["tests"],
        dsep_checks=state["dsep_checks"], expert_dag_sha256=dag_digest)
    return (matrix, diagnostics) if return_diagnostics else matrix


def search(data, guess, rng, store, key, deadline=float("inf"), tester=None, checkpoint=None,
           return_diagnostics=False):
    """Historical skeleton-guided search, optionally retaining separating sets."""
    from causallearn.utils.cit import CIT
    from cd_llm.baselines.guess2graph2026.paper_methods import (
        expert_edge_order, neighbors, validate_guessed_skeleton,
    )
    validate_guessed_skeleton(guess, data.shape[1])
    cit = tester or CIT(data, "gsq")
    edges = expert_edge_order(list(combinations(range(data.shape[1]), 2)), guess, rng)
    state = checkpoint.load() if checkpoint is not None else store.get("gpc_progress", key)
    if state is None:
        state = dict(
            matrix=(np.ones((data.shape[1], data.shape[1]), dtype=int) - np.eye(data.shape[1], dtype=int)).tolist(),
            edges=[list(edge) for edge in edges], edge=0, size=0, cursor=0,
            tests=0, pool=None, permutation=None, swaps={})
        if return_diagnostics:
            state["separating_sets"] = {}
        if checkpoint is not None:
            state["swaps"] = checkpoint.mapping
    else:
        if state["edges"] != [list(edge) for edge in edges]:
            raise ValueError("gPC edge order changed")
        if return_diagnostics and "separating_sets" not in state:
            raise ValueError("gPC checkpoint does not contain separating sets")
        rng.bit_generator.state = state["rng"]
    matrix = np.asarray(state["matrix"], dtype=int)
    last = time.monotonic()

    def save():
        nonlocal last
        state.update(matrix=matrix.tolist(), rng=rng.bit_generator.state)
        if checkpoint is None:
            store.put("gpc_progress", key, state)
        else:
            checkpoint.save(state)
            from cd_llm.experiments.common.artifacts import write_json
            write_json(checkpoint.folder / "progress.json", dict(
                edge=state["edge"], size=state["size"], cursor=state["cursor"],
                tests=state["tests"], updated=time.time()))
        last = time.monotonic()

    while state["edge"] < len(edges):
        left, right = edges[state["edge"]]
        if state["pool"] is None:
            state["pool"] = min(
                [neighbors(matrix, left, exclude=right), neighbors(matrix, right, exclude=left)],
                key=len)
        pool = state["pool"]
        removed = False
        while state["size"] <= len(pool) and not removed:
            count = comb(len(pool), state["size"])
            if state["permutation"] is None and count <= 100000:
                values = list(range(count))
                rng.shuffle(values)
                state["permutation"] = values
            while state["cursor"] < count:
                if time.time() >= deadline:
                    save()
                    raise Checkpoint("gPC checkpoint within edge/conditioning depth")
                index = state["cursor"]
                if state["permutation"] is not None:
                    rank = state["permutation"][index]
                else:
                    chosen = int(rng.integers(index, count))
                    rank = state["swaps"].get(str(chosen), chosen)
                    replacement = state["swaps"].pop(str(index), index)
                    if chosen != index:
                        state["swaps"][str(chosen)] = replacement
                conditioning = unrank_combination(pool, state["size"], rank)
                value = cit(left, right, list(conditioning))
                if hasattr(cit, "pvalue_cache") and len(cit.pvalue_cache) > 100000:
                    cit.pvalue_cache.clear()
                state["tests"] += 1
                state["cursor"] += 1
                if value > .05:
                    matrix[left, right] = matrix[right, left] = 0
                    if return_diagnostics:
                        state["separating_sets"][f"{left}:{right}"] = list(conditioning)
                    removed = True
                    break
                if time.monotonic() - last > (300 if checkpoint is not None else 30):
                    save()
            state["swaps"].clear()
            state.update(size=state["size"] + 1, cursor=0, permutation=None)
        state["swaps"].clear()
        state.update(edge=state["edge"] + 1, size=0, cursor=0, pool=None, permutation=None)
        save()
    if return_diagnostics:
        return matrix, dict(separating_sets=state["separating_sets"], ci_tests=state["tests"])
    return matrix
