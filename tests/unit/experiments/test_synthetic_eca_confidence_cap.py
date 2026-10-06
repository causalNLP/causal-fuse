import hashlib
import numpy as np
import pytest

from cd_llm.experiments import synthetic_eca_confidence_cap as study


def test_cap_changes_only_the_native_confidence_and_preserves_raw_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(study.original.matched, 'SIZES', (200,))
    data=np.random.default_rng(4).normal(size=(200,20))
    expected=study.original.standardize(data)
    graph=study.original.graph_from_edges(20)
    facts=study.original.matched.answer_pairs(graph,study.original.matched.selected_pairs(20,5))
    signs,raw=study.original.native_prior(facts)
    source=tmp_path/'source.json'
    study.write_json(source,dict(nested_data={'200':data.tolist()}))
    inputs=tmp_path/'input.json'
    study.write_json(inputs,dict(facts=facts,prior_signs=signs.tolist(),confidence=raw,
        source={'path':str(source)},standardized_sha256={'200':hashlib.sha256(expected.tobytes()).hexdigest()}))
    m=dict(inputs={'1':dict(path=str(inputs),sha256=study.sha256(inputs))})
    study.write_json(tmp_path/'manifest.json',m)
    monkeypatch.setattr(study,'root_for',lambda identifier:tmp_path)
    monkeypatch.setattr(study,'verify',lambda root:m)
    calls=[]
    def fit(observations,prior,confidence):
        np.testing.assert_array_equal(observations,expected)
        np.testing.assert_array_equal(prior,signs)
        calls.append(confidence)
        return dict(status='failed',error='test sentinel')
    monkeypatch.setattr(study.original,'fit',fit)
    study.compute('unused',1)
    assert calls==[.99]
    assert study.read_json(inputs)['confidence']==1.
    assert all(r['reported_confidence']==1 for r in study.read_json(inputs)['facts'])
    result=study.read_json(tmp_path/'evaluations/scm_001_n200.json')
    assert result['raw_confidence']==1. and result['effective_confidence']==.99


def test_numerical_report_separates_graph_validity_solver_status_and_nonfinite_trials():
    call=dict(options=None,first_evaluation=dict(objective=2.,gradient_finite=True),
              success=False,message='ABNORMAL',nonfinite_evaluations=3)
    row=dict(graph_index=1,sample_size=200,status='ok',error=None,seconds=1.,
             optimizer_calls=[call],final_optimizer_objective=1.,final_split_weights=[0.]*800)
    details=study.numerical_details(row,.99)
    assert details['initial_objective_and_gradient_finite']
    assert details['final_objective_finite'] and details['acyclicity']==0.
    assert details['status']=='ok' and not details['final_optimizer_success']
    assert details['nonfinite_evaluations']==3 and details['unsuccessful_inner_calls']==1
    row['final_optimizer_objective']='nan'
    assert not study.numerical_details(row,.99)['final_objective_finite']
    call['options']={'maxiter':100}
    with pytest.raises(ValueError,match='options changed'):
        study.numerical_details(row,.99)
