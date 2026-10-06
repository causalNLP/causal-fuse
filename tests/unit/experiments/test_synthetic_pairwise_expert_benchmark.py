from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments import synthetic_pairwise_expert_benchmark as study


def test_selection_does_not_consult_truth():
    pairs = study.selected_pairs(20, 17)
    assert len(pairs) == len(set(pairs)) == 20
    assert pairs == study.selected_pairs(20, 17)
    assert all(a < b for a, b in pairs)
    empty = study.answer_pairs(graph_from_edges(20), pairs)
    other = study.answer_pairs(graph_from_edges(20, pairs), pairs)
    assert [(r['left'], r['right']) for r in empty] == [(r['left'], r['right']) for r in other]
    assert all(r['state'] == 'absent' for r in empty)
    assert all(r['state'] == 'left_to_right' for r in other)


def test_native_mapping_semantics_and_confidence():
    truth = graph_from_edges(3, [(0, 1), (2, 1)])
    raw = study.answer_pairs(truth, [(0, 1), (1, 2), (0, 2)])
    inputs = study.native_inputs(raw, 3)
    assert all(r['reported_confidence'] == 1 for r in raw)
    assert [(r['kind'], r['source'], r['target'], r['answer']) for r in inputs['FS']] == [
        ('directed', 0, 1, 1), ('directed', 2, 1, 1), ('adjacency', 0, 2, 0)]
    assert inputs['FS'] == inputs['SMC-Passive']
    assert all(r['reported_confidence'] == 1 and r['effective_confidence'] == .99 for r in inputs['FS'])
    assert {(p.left, p.right): p.probabilities for p in inputs['CaMML'].local} == {
        (0, 1): (1., 0., 0., 0.), (1, 2): (0., 1., 0., 0.), (0, 2): (0., 0., .5, .5)}
    assert {(c.source, c.target): c.probability for c in inputs['ECA'].constraints} == {
        (0, 1): 1, (1, 0): 0, (1, 2): 0, (2, 1): 1, (0, 2): 0, (2, 0): 0}
    assert [p.state.value for p in inputs['Marg']] == ['left_to_right', 'right_to_left', 'absent']
    assert {p.expert_id for p in inputs['Marg']} == {'synthetic-oracle'}
    assert study.plain(study.MARG_PRIOR) == dict(alpha1=6.1, alpha2=2.05, alpha3=1.85, beta1=7.75, beta2=2.25)
    with pytest.raises(ValueError, match='repeated'):
        study.native_inputs(raw + raw[:1], 3)


def test_fs_absence_contributes_one_factor():
    empty, edge = graph_from_edges(3), graph_from_edges(3, [(0, 1)])
    inputs = study.native_inputs(study.answer_pairs(empty, [(0, 1)]), 3)
    posterior = study.finite_posterior({empty: .5, edge: .5}, inputs['FS'])
    assert posterior[empty] == pytest.approx(.99)
    assert posterior[edge] == pytest.approx(.01)


def test_marg_local_score_does_not_inherit_twelve_parent_limit():
    data = np.random.default_rng(1).normal(size=(50, 20))
    score = study.GaussianBICScore(data, max_parents=19, standardize_data=False)
    assert np.isfinite(score.local_score(19, (1 << 13) - 1))


def test_passive_finishes_at_twenty_with_raw_confidence():
    truth = graph_from_edges(7)
    inputs = study.native_inputs(study.answer_pairs(truth, study.selected_pairs(7, 1)), 7)
    score = study.GaussianBICScore(np.random.default_rng(1).normal(size=(30, 7)), max_parents=6)
    result = study.passive(score, (truth,) * 8, np.full(8, 1/8), inputs['FS'], 2)
    assert set(result.distributions) == {20}
    assert len(result.steps) == 20
    assert all(s.reported_confidence == 1 and s.effective_confidence == .99 for s in result.steps)
    assert result.steps[-1].rejuvenation_attempted_moves > 0


def complete_rows():
    return [dict(graph_index=g, sample_size=n, method=m, status='ok',
                 **{key: float(g) for key in study.METRICS})
            for g in study.SCMS for n in study.SIZES for m in study.METHODS]


def test_failed_cohort_has_no_mean_or_bold():
    rows = complete_rows()
    for row in rows:
        if row['method'] == 'ECA' and row['sample_size'] == 200:
            row.update({m: 0. for m in study.METRICS})
            if row['graph_index'] == study.SCMS[0]:
                row.update(status='failed', **{m: None for m in study.METRICS})
    summary = study.summarize(rows)
    failed = next(r for r in summary if r['method'] == 'ECA' and r['sample_size'] == 200)
    assert failed['success_count'] == 9
    assert all(failed[m + '_mean'] is None and failed[m + '_se'] is None for m in study.METRICS)
    path = Path(__file__).resolve().parents[3] / 'scripts/report_synthetic_pairwise_expert_benchmark.py'
    spec = importlib.util.spec_from_file_location('pairwise_report', path)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    line = next(line for line in reporter.render(summary).splitlines() if line.startswith('ECA (20)'))
    assert line.startswith('ECA (20) & --- & --- & --- &')
    with pytest.raises(ValueError, match='missing or duplicate'):
        study.summarize(rows[:-1])
    with pytest.raises(ValueError, match='missing or duplicate'):
        study.summarize(rows + rows[:1])


def test_invalid_eca_is_not_reported_as_success():
    truth = graph_from_edges(3)
    with pytest.raises(ValueError, match='acyclicity'):
        study.validate_prediction('ECA', {truth: 1.}, dict(objective=1.,
            diagnostics=dict(converged=False, acyclicity=1e-5)), truth)
