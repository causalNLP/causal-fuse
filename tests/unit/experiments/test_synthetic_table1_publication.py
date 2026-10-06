from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from cd_llm.experiments.common.artifacts import write_json,sha256

path=Path(__file__).resolve().parents[3]/'scripts/report_synthetic_pairwise_expert_benchmark.py'
spec=importlib.util.spec_from_file_location('table1_publication',path)
reporter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def summaries():
    base=[dict(method=method,sample_size=n,budget=0 if method in ('PC','GES','NOTEARS') else 20,
               success_count=10,total_count=10,**{metric+suffix:1. for metric in reporter.METRICS for suffix in ('_mean','_se')})
          for method in reporter.METHODS for n in reporter.SIZES]
    capped=[dict(r,confidence=confidence,expected_distance_mean=15.3)
            for confidence in (1.,.99) for r in base if r['method']=='ECA']
    return base,capped


def test_only_two_eca_rows_are_replaced_and_inputs_preserved():
    base,capped=summaries()
    before=deepcopy(base)
    merged=reporter.merge_eca_rows(base,capped)
    assert base==before
    for old,new in zip(base,merged):
        if old['method']=='ECA':
            assert new['expected_distance_mean']==15.3 and 'confidence' not in new
        else:
            assert new==old
    text=reporter.render(merged,original_eca=True)
    assert "native knowledge encodings and the full protocol" in text
    assert 'ECA (20) & 15.30 (1.00)' in text
    assert 'only FS/SMC' not in text


def test_incomplete_or_ambiguous_eca_summary_rejected():
    base,capped=summaries()
    capped[-1]['success_count']=9
    with pytest.raises(ValueError,match='ten valid'):
        reporter.merge_eca_rows(base,capped)
    base,capped=summaries()
    with pytest.raises(ValueError,match='duplicate'):
        reporter.merge_eca_rows(base,capped+capped[-1:])
    with pytest.raises(ValueError,match='missing'):
        reporter.merge_eca_rows(base,capped[:-1])
    with pytest.raises(ValueError,match='together'):
        reporter.report('unused',eca_run_id='unused')


def test_mismatched_facts_observations_or_cap_rejected(tmp_path):
    facts=[dict(fact_id=str(i)) for i in range(20)]
    path=tmp_path/'facts.json'
    write_json(path,dict(records=facts))
    source=dict(path='same-observations',sha256='source-hash')
    m=dict(facts={'1':dict(path=str(path),sha256=sha256(path))},sources={'1':source})
    inputs=dict(facts=facts,source=source,confidence=1.,standardized_sha256={'200':'data-hash'})
    row=dict(raw_confidence=1.,effective_confidence=.99,sample_size=200,standardized_sha256='data-hash')
    reporter.validate_matched_eca_input(m,1,inputs,row)
    for replacement in (dict(facts=facts[:-1]),dict(source=dict(path='other',sha256='different'))):
        with pytest.raises(ValueError,match='knowledge differ'):
            reporter.validate_matched_eca_input(m,1,dict(inputs,**replacement),row)
    with pytest.raises(ValueError,match='confidence'):
        reporter.validate_matched_eca_input(m,1,inputs,dict(row,effective_confidence=1.))
