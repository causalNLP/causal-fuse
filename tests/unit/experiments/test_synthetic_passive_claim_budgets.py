from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cd_llm.core.finite_support import graph_edges, graph_from_edges
from cd_llm.experiments.synthetic_passive_claim_budgets import (
    BUDGETS, CONFIDENCE, GRID, SIZES, claim_sequence, finite_prefixes,
    grid_mixture, passive_trajectory, validate_claims, fit_grid,
    restored_initialization, baseline_rows, archived_inputs, SETTINGS, SOURCE_RUN,
    STUDY, SCMS, PC_CONFIG, reused_initialization, initialization_mixture, settings_for,
)
from cd_llm.experiments.claim_fuse_smc_eig import _initial_particle_population, _serialize
from cd_llm.experiments.common.artifacts import write_json, sha256, json_value
from cd_llm.methods.active_query import ClaimQuery, query_values
from cd_llm.methods.adaptive_pruning import GaussianBICScore


def problem():
    empty = graph_from_edges(4)
    forward = graph_from_edges(4, [(0, 1), (1, 2)])
    reverse = graph_from_edges(4, [(1, 0), (2, 1)])
    initial = {empty: 0.25, forward: 0.5, reverse: 0.25}
    score = GaussianBICScore(np.random.default_rng(7).normal(size=(30, 4)), max_parents=3)
    records = claim_sequence(forward, 9, count=8)
    return initial, score, records


def test_sequence_is_fixed_unique_and_soft():
    truth = graph_from_edges(20, [(0, 1), (1, 2)])
    records = claim_sequence(truth, 17)
    assert records == claim_sequence(truth, 17)
    assert records != claim_sequence(truth, 18)
    assert len(records) == 50
    validate_claims(records, 20)
    for r in records:
        q = ClaimQuery(**{k: r[k] for k in ("kind", "source", "target", "query_id")})
        assert r["answer"] == query_values((truth,), q)[0]
        assert r["effective_confidence"] == CONFIDENCE < 1
    duplicate = deepcopy(records)
    duplicate[1] = {**duplicate[0], "query_id": "different-name", "provenance_family": "different-source"}
    with pytest.raises(ValueError, match="duplicate"):
        validate_claims(duplicate, 20)


def test_grid_family_mass_and_duplicate_aggregation():
    empty, forward = graph_from_edges(3), graph_from_edges(3, [(0, 1)])
    fits = [
        dict(config=dict(family="PC"), graphs=[graph_edges(empty), graph_edges(forward)]),
        dict(config=dict(family="PC"), graphs=[graph_edges(empty)]),
        dict(config=dict(family="PC"), graphs=[]),
        dict(config=dict(family="GES"), graphs=[graph_edges(forward)]),
    ]
    mixture = grid_mixture(fits, 3)
    assert mixture == {empty: 0.375, forward: 0.625}
    particles, weights, empirical = _initial_particle_population(mixture, 4)
    assert len(particles) == 384
    np.testing.assert_array_equal(weights, np.full(384, 1/384))
    assert sum(empirical.values()) == pytest.approx(1)
    for g in empirical:
        assert empirical[g] == pytest.approx(particles.count(g)/384)
    with pytest.raises(ValueError, match="both PC and GES"):
        grid_mixture(fits[:-1], 3)


def test_finite_prefixes_match_direct_likelihood_products():
    initial, _, records = problem()
    for b, dist in finite_prefixes(initial, records, (2, 5, 8)).items():
        direct = {}
        for graph, weight in initial.items():
            for r in records[:b]:
                q = ClaimQuery(**{k: r[k] for k in ("kind", "source", "target", "query_id")})
                weight *= CONFIDENCE if query_values((graph,), q)[0] == r["answer"] else 1-CONFIDENCE
            direct[graph] = weight
        total = sum(direct.values())
        for g in initial:
            assert dist[g] == pytest.approx(direct[g] / total, abs=1e-14)


def test_fixed_trajectory_checkpoints_do_not_change_inference():
    initial, score, records = problem()
    particles = tuple(initial) * 4
    weights = np.asarray(list(initial.values()) * 4) / 4
    with patch("cd_llm.methods.active_query._select_uniform_candidate_index", side_effect=AssertionError("acquisition used")):
        full = passive_trajectory(score, particles, weights, records, 73, budgets=(2, 5, 8))
        endpoint = passive_trajectory(score, particles, weights, records, 73, budgets=(8,))
    assert full.distributions[8] == endpoint.distributions[8]
    assert full.particles == endpoint.particles
    assert full.weights == endpoint.weights
    assert [s.query.query_id for s in full.steps] == [r["query_id"] for r in records]
    for a, b in zip(full.steps, endpoint.steps):
        left, right = asdict(a), asdict(b)
        left.pop("cumulative_seconds")
        right.pop("cumulative_seconds")
        assert left == right
        if a.budget < 8 and not a.resampled:
            assert a.rejuvenation_attempted_moves == 0
    assert all(s.selection_rule == "fixed" for s in full.steps)
    assert all(s.treatment == "soft_evidence" and s.effective_confidence == 0.99 for s in full.steps)


def test_fixed_smc_without_moves_matches_fs_each_prefix():
    initial, score, records = problem()
    # Leave ample copies to prevent resampling while testing exactly-once factors.
    from cd_llm.methods.active_query import active_query_smc
    queries = tuple(ClaimQuery(**{k: r[k] for k in ("kind", "source", "target", "query_id")}) for r in records)
    answers = {r["query_id"]: r["answer"] for r in records}
    result = active_query_smc(initial, score, lambda q: answers[q.query_id],
        query_pool=queries, budgets=(2, 5, 8), policy="fixed", confidence=CONFIDENCE,
        treatment="soft_evidence", rejuvenation_steps=0, ess_threshold_fraction=1e-8)
    expected = finite_prefixes(initial, records, (2, 5, 8))
    assert result.resampling_count == 0
    for b in expected:
        for g in expected[b]:
            assert result.distributions[b][g] == pytest.approx(expected[b][g], abs=1e-14)


def test_small_real_grid_uses_six_configurations():
    data = np.random.default_rng(2).normal(size=(40, 3))
    fits = fit_grid(data, 1, 40)
    assert len(fits) == 6
    assert [r["config"]["family"] for r in fits] == ["PC"] * 3 + ["GES"] * 3
    mixture = grid_mixture(fits, 3)
    assert sum(mixture.values()) == pytest.approx(1)


def test_soft_confidence_point_99_has_expected_likelihood_odds():
    empty, forward = graph_from_edges(2), graph_from_edges(2, [(0, 1)])
    records = [dict(kind="directed", source=0, target=1, query_id="q", answer=1,
                    reported_confidence=0.99, effective_confidence=0.99,
                    treatment="soft_evidence", provenance_family="source")]
    assert CONFIDENCE == 0.99
    posterior = finite_prefixes({empty: 0.5, forward: 0.5}, records, (1,))[1]
    assert posterior[forward] == pytest.approx(0.99)
    assert posterior[empty] == pytest.approx(0.01)
    assert posterior[forward] / posterior[empty] == pytest.approx(99)
    records[0]["effective_confidence"] = 0.7
    with pytest.raises(ValueError, match="confidence"):
        finite_prefixes({empty: 0.5, forward: 0.5}, records, (1,))


def saved_initialization():
    empty, forward = graph_from_edges(20), graph_from_edges(20, [(0, 1)])
    fits = [dict(config=dict(config), graphs=[graph_edges(empty), graph_edges(forward)])
            for config in GRID]
    mixture = grid_mixture(fits)
    particles, weights, initial = _initial_particle_population(mixture, 7)
    return dict(shared_initialization=True, initialization_updates=0, grid_fits=fits,
                grid_distribution=_serialize(mixture),
                initial_particles=[graph_edges(g) for g in reversed(particles)],
                initial_weights=weights.tolist(), initial_distribution=_serialize(initial))


def test_restore_preserves_particle_order_and_rejects_changed_mass():
    saved = saved_initialization()
    with patch("cd_llm.experiments.synthetic_passive_claim_budgets.fit_grid", side_effect=AssertionError("refit")), \
         patch("cd_llm.experiments.synthetic_passive_claim_budgets._initial_particle_population", side_effect=AssertionError("reallocation")):
        fits, mixture, particles, weights, initial = restored_initialization(saved)
    assert [graph_edges(g) for g in particles] == saved["initial_particles"]
    assert weights.tolist() == saved["initial_weights"]
    assert _serialize(initial) == saved["initial_distribution"]
    saved["initial_weights"][0] = 0.5
    with pytest.raises(ValueError, match="particle slots"):
        restored_initialization(saved)


def test_archived_hash_validation_does_not_require_historical_code(tmp_path):
    # All ten input hashes are registered; the historical implementation is unavailable.
    sources, claims, partials = {}, {}, []
    for g in SCMS:
        for name, destination in (("source", sources), ("claim", claims)):
            path = tmp_path / f"{name}_{g}.json"
            write_json(path, {})
            destination[str(g)] = dict(path=str(path), sha256=sha256(path))
    manifest = dict(study=STUDY, run_id="20260920T205251Z", source_run=SOURCE_RUN,
                    settings={**json_value(SETTINGS), "confidence": 3.7 / 4.7},
                    graph_indices=list(SCMS), sample_sizes=list(SIZES),
                    sources=sources, claims=claims,
                    implementation={"/missing/historical/code.py": "historical-hash"})
    write_json(tmp_path / "manifest.json", manifest)
    digest = sha256(tmp_path / "manifest.json")
    for g in SCMS:
        path = tmp_path / f"partial_{g}.json"
        write_json(path, dict(graph_index=g, manifest_sha256=digest,
                   reference_sha256=sources[str(g)]["sha256"], claims_sha256=claims[str(g)]["sha256"]))
        partials.append(dict(graph_index=g, path=str(path), sha256=sha256(path)))
    write_json(tmp_path / "completion.json", dict(manifest_sha256=digest,
               trajectories=20, paired_checkpoints=60, partials=partials))
    restored, _ = archived_inputs(tmp_path)
    assert restored == manifest
    write_json(Path(partials[0]["path"]), {"tampered": True})
    with pytest.raises(ValueError, match="registered input changed"):
        archived_inputs(tmp_path)


def test_baselines_use_partial_predictions_and_zero_claims():
    truth = graph_from_edges(20, [(0, 1)])
    # An unresolved edge: correct adjacency, no claimed orientation, half ED error.
    matrix = np.zeros((20, 20), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 1
    partial = dict(feasible=True, diagnostics=dict(output=dict(endpoint_matrix=matrix.tolist(), kind="cpdag")))
    artifact = dict(feasible=True, graphs=[graph_edges(truth)], probabilities=[1.0])
    payload = dict(graph_index=1, target=graph_edges(truth),
        protocol=dict(ges_lambda=0.5, notears_l1=0.05, notears_threshold=0.3, pc_alphas=[0.01, 0.05, 0.1]),
        samples={"200": dict(methods={"PC": partial, "GES": partial, "NOTEARS": artifact})})
    rows = baseline_rows(payload, 200)
    assert all(r["budget"] == 0 for r in rows)
    for row in rows[:2]:
        assert row["expected_distance"] == 0.5
        assert row["adjacency_f1"] == 1.0
        assert row["orientation_f1"] == 0.0
    assert rows[2]["expected_distance"] == 0.0
    assert rows[2]["orientation_f1"] == 1.0




def test_pc_only_reuse_selects_single_fit_and_reallocates_shared_slots():
    saved = saved_initialization()
    forward = graph_from_edges(20, [(0, 1)])
    reverse = graph_from_edges(20, [(1, 0)])
    selected = next(f for f in saved["grid_fits"] if f["config"] == PC_CONFIG)
    selected["graphs"] = [graph_edges(forward), graph_edges(reverse), graph_edges(forward)]
    with patch("cd_llm.experiments.synthetic_passive_claim_budgets.fit_grid", side_effect=AssertionError("refit")):
        fits, mixture, particles, weights, initial = reused_initialization(saved, "pc-only", "pc-ges-grid", 7)
    assert [f["config"] for f in fits] == [PC_CONFIG]
    assert mixture == {forward: 0.5, reverse: 0.5}
    assert set(particles) == {forward, reverse} and len(particles) == 384
    assert np.all(weights == 1 / 384)
    assert [graph_edges(g) for g in particles] != saved["initial_particles"]
    assert set(initial) == {forward, reverse}
    from cd_llm.experiments.synthetic_passive_claim_budgets import metrics
    for dist in finite_prefixes(initial, claim_sequence(forward, 9), BUDGETS).values():
        assert metrics(dist, forward)["adjacency_f1"] == 1.0
    with pytest.raises(ValueError, match="configurations"):
        initialization_mixture(saved["grid_fits"], "pc-only")
    assert settings_for("pc-only")["grid"] == (PC_CONFIG,)
    assert settings_for("pc-ges-grid")["grid"] == GRID
