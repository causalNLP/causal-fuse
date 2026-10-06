from types import SimpleNamespace

import numpy as np
import pytest

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments import synthetic_eca_original as study


def test_saved_facts_map_to_signed_native_prior():
    truth = graph_from_edges(4, [(0, 1), (2, 1)])
    records = study.matched.answer_pairs(truth, [(0, 1), (1, 2), (0, 2)])
    signs, confidence = study.native_prior(records, 4)
    assert confidence == 1.0
    np.testing.assert_array_equal(signs, [[0,1,-1,0],[-1,0,-1,0],[-1,1,0,0],[0,0,0,0]])
    assert np.count_nonzero(signs) == 2 * len(records)
    assert all(r['reported_confidence'] == 1.0 for r in records)


def test_recording_optimizer_preserves_arguments_values_and_solution_identity():
    initial = np.zeros(8)
    gradient = np.arange(8.)
    returned_evaluation = (3., gradient)
    bounds = [(0., None)] * 8
    sol = SimpleNamespace(x=initial, fun=3., success=True, status=0, message='done', nit=1)
    def real_minimize(fun, *args, **kwargs):
        assert args == (initial,)
        assert kwargs == dict(method='L-BFGS-B', jac=True, bounds=bounds)
        assert fun(initial) is returned_evaluation
        return sol
    recorder = study.RecordingOptimizer(SimpleNamespace(minimize=real_minimize))
    output = recorder.minimize(lambda w: returned_evaluation, initial, method='L-BFGS-B', jac=True, bounds=bounds)
    assert output is sol
    assert recorder.calls[0]['options'] is None
    assert recorder.calls[0]['first_evaluation'] == dict(objective=3., gradient_finite=True)
    assert not np.shares_memory(recorder.last_x, sol.x)


def test_nonfinite_optimization_is_not_a_valid_empty_graph():
    sol = SimpleNamespace(x=np.zeros(8), fun=float('nan'), success=False, status=2, message='ABNORMAL', nit=0)
    def original(fun, *args, **kwargs):
        value = fun(args[0])
        assert np.isinf(value[0]) and np.isnan(value[1]).all()
        return sol
    recorder = study.RecordingOptimizer(SimpleNamespace(minimize=original))
    recorder.minimize(lambda w: (float('inf'), np.full(8,np.nan)), np.zeros(8))
    assert recorder.calls[0]['nonfinite_evaluations'] == 1
    assert recorder.calls[0]['first_evaluation']['objective'] == 'inf'
    with pytest.raises(ValueError,match='nonfinite final'):
        study.checked_prediction(np.zeros((2,2)),recorder,2)


def test_validity_uses_unthresholded_native_constraint_and_separate_solver_status():
    recorder = study.RecordingOptimizer(None)
    recorder.last_fun = 1.
    # A thresholded empty graph cannot hide a continuous cycle.
    recorder.last_x = np.array([0,.1,.1,0,0,0,0,0])
    with pytest.raises(ValueError,match='polynomial acyclicity'):
        study.checked_prediction(np.zeros((2,2)),recorder,2)
    recorder.last_x = np.zeros(8)
    graph,h = study.checked_prediction(np.zeros((2,2)),recorder,2)
    assert graph == graph_from_edges(2) and h == 0


def test_original_source_is_byte_identical_to_pinned_python_files():
    hashes = study.source_hashes()
    assert str(study.UPSTREAM_ROOT/'src/ECA_prior.py') in hashes


def test_complete_cohort_required_and_no_duplicate_rows():
    rows = [dict(graph_index=g,sample_size=n,status='ok',metrics={m:1. for m in study.matched.METRICS})
            for g in study.matched.SCMS for n in study.matched.SIZES]
    rows[0].update(status='failed',metrics=None)
    summary=study.summarize(rows)
    assert summary[0]['success_count']==9 and summary[0]['expected_distance_mean'] is None
    assert summary[1]['success_count']==10 and summary[1]['expected_distance_mean']==1
    with pytest.raises(ValueError,match='missing or duplicate'):
        study.summarize(rows[:-1])
    with pytest.raises(ValueError,match='missing or duplicate'):
        study.summarize(rows+rows[:1])


def test_fit_calls_original_class_with_exact_settings_and_no_local_solver(monkeypatch):
    inputs = []
    sol=SimpleNamespace(x=np.zeros(8),fun=1.,success=False,status=1,message='iteration limit',nit=100)
    original_optimizer=SimpleNamespace(minimize=lambda fun,*args,**kwargs: sol)
    module=SimpleNamespace(sopt=original_optimizer)
    class OriginalClass:
        def __init__(self,**kwargs):
            assert kwargs == study.SETTINGS
        def load_prior(self,signs,confidence):
            assert confidence==1.
            inputs.append(signs.copy())
        def notears_linear(self,data,**kwargs):
            assert kwargs==study.SETTINGS
            module.sopt.minimize(lambda x:(1.,np.zeros(8)),np.zeros(8),method='L-BFGS-B',jac=True,bounds=[])
            return np.zeros((2,2))
    module.myECA_prior=OriginalClass
    monkeypatch.setattr(study,'_load_upstream_linear',lambda root:module)
    signs=np.array([[0.,-1.],[-1.,0.]])
    result=study.fit(np.ones((4,2)),signs,1.)
    assert result['status']=='ok'
    assert result['optimizer_calls'][0]['success'] is False
    assert result['optimizer_calls'][0]['options'] is None
    assert len(inputs)==1
    np.testing.assert_array_equal(inputs[0],signs)
