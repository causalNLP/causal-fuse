"""Audit saved 0.8-cap checkpoints and prepare Figure 3, without inference."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls

from cd_llm.metrics.units import read_metric_csv, metric_metadata
from cd_llm.paths import WORKSPACE_ROOT as ROOT

SOURCE=ROOT/'results/20260918T153548Z/neuropathic_confidence_cap'
CACHE=ROOT/'results/20260918T003643Z/neuropathic_new_prompt_smc'
BASE=ROOT/'results/20260919T000434Z/neuropathic_performance_cost'
BUDGETS=(10,20,50,100)


def prepare(out):
    hashes={}
    def read(path,expected=None):
        raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if expected:assert digest==expected,path
        hashes[str(path.relative_to(ROOT))]=digest
        return json.loads(raw)
    completion=read(SOURCE/'completion.json')
    assert completion['complete'] and completion['audit_passed'] and completion['checkpoints']==480
    manifest=read(SOURCE/'manifest.json',completion['manifest_sha256'])
    audit=read(SOURCE/'audit.json',completion['audit_sha256'])
    assert manifest['answer_confidence_cap']==.8 and manifest['acquisition_confidence']==.9
    source_rows=read_metric_csv(SOURCE/'per_run.csv',node_count=20,legacy_normalized=True)
    assert hashlib.sha256((SOURCE/'per_run.csv').read_bytes()).hexdigest()==completion['artifacts']['per_run.csv']
    hashes[str((SOURCE/'per_run.csv').relative_to(ROOT))]=completion['artifacts']['per_run.csv']
    metrics={(int(r['sample_size']),r['strategy'],r['group'],int(r['seed']),int(r['budget'])):r
             for r in source_rows if float(r['evidence_cap'])==.8}
    assert len(metrics)==480
    outcomes=read(SOURCE/'responses.json',manifest['inputs']['responses.json'])
    outcome_index={(r['strategy'],r['query']['query_id']):r for r in outcomes}
    cache_audit=read(SOURCE/'cache_audit.json',manifest['inputs']['cache_audit.json'])
    cache_manifest=read(CACHE/'manifest.json',cache_audit['manifest_sha256'])
    raw=read(CACHE/'cache/raw_responses.json',cache_audit['artifacts']['cache/raw_responses.json'])
    ancestry=read(CACHE/'ancestry/cache/raw_responses.json',cache_manifest['inputs']['ancestry/cache/raw_responses.json'])
    assert not set(raw).intersection(ancestry)
    raw.update(ancestry)
    output,events,used_request_keys=[],[],set()
    initialization_seconds={}
    for task in manifest['tasks']:
        record=read(SOURCE/f"records/{task['id']}.json",audit['records'][task['id']])
        assert record['manifest_sha256']==completion['manifest_sha256'] and len(record['trace'])==100
        condition=task['condition']
        if condition not in initialization_seconds:
            name=f'initializations/{condition}_grid_union.json'
            initial=read(SOURCE/name,manifest['inputs'][name])
            initialization_seconds[condition]=float(initial['wall_seconds'])
        inputs=outputs=calls=abstentions=repairs=abstention_tokens=repair_tokens=0
        for slot,step in enumerate(record['trace'],1):
            outcome=outcome_index[task['strategy'],step['query']['query_id']]
            keys=step['response_audit']['request_keys']
            assert keys==outcome['request_keys'] and step['query']==outcome['query']
            assert step['answer']==outcome['answer'] and step['selection_effective_confidence']==.9
            if step['valid_response']:
                assert step['effective_confidence']==min(outcome['reported_confidence'],.8)
            if task['strategy']=='consistency':
                assert len(keys)==8 and step['response_audit']['minimum_valid_response_count']==6
                assert step['valid_response']==(step['response_audit']['valid_response_count']>=6)
            ip=op=0
            for attempt,key in enumerate(keys):
                cached=raw[key]['record'];usage=cached['response']['usage']
                used_request_keys.add(key)
                pair=(cached['prompt_tokens'],cached['completion_tokens'])
                assert all(isinstance(v,int) and v>=0 for v in pair) and sum(pair)>0
                assert pair==(usage['prompt_tokens'],usage['completion_tokens'])
                ip+=pair[0];op+=pair[1]
                repair=task['strategy']=='stated' and attempt>0
                repairs+=repair;repair_tokens+=sum(pair) if repair else 0
                events.append(dict(sample_size=task['sample_size'],strategy=task['strategy'],group=task['group'],
                    seed=task['seed'],query=slot,request_key=key,input_tokens=pair[0],output_tokens=pair[1],
                    latency_seconds=float(cached['latency_seconds']),abstention=not step['valid_response'],repair=repair))
            assert (ip,op)==(step['prompt_tokens'],step['completion_tokens'])==(outcome['prompt_tokens'],outcome['completion_tokens'])
            inputs+=ip;outputs+=op;calls+=len(keys);abstentions+=not step['valid_response']
            abstention_tokens+=ip+op if not step['valid_response'] else 0
            if slot in BUDGETS:
                metric=metrics[task['sample_size'],task['strategy'],task['group'],task['seed'],slot]
                assert (inputs,outputs,calls,abstentions)==tuple(int(metric[k]) for k in
                    ('prompt_tokens','completion_tokens','logical_requests','abstentions'))
                output.append(dict(method='SMC-Active',sample_size=task['sample_size'],strategy=task['strategy'],
                    group=task['group'],seed=task['seed'],budget=slot,expected_distance=float(metric['expected_distance']),
                    input_tokens=inputs,output_tokens=outputs,total_tokens=inputs+outputs,response_events=calls,
                    numerical_seconds=initialization_seconds[condition]+float(record['checkpoint_seconds'][str(slot)]),
                    abstentions=abstentions,repair_events=repairs,abstention_tokens=abstention_tokens,repair_tokens=repair_tokens))
    prior=read(BASE/'protocol.json')
    read(BASE/'completion.json')
    for row in csv.DictReader((BASE/'per_run.csv').open()):
        if row['strategy']!='native':continue
        output.append(dict(method=row['method'],sample_size=int(row['sample_size']),strategy=row['strategy'],
            group=row['group'],seed=int(row['seed']),budget=int(row['budget']),
            expected_distance=float(row['expected_distance']),input_tokens=int(row['input_tokens']),
            output_tokens=int(row['output_tokens']),total_tokens=int(row['total_tokens']),
            response_events=int(row['response_events']),numerical_seconds=0.,abstentions='',repair_events='',
            abstention_tokens='',repair_tokens=''))
    hashes[str((BASE/'per_run.csv').relative_to(ROOT))]=hashlib.sha256((BASE/'per_run.csv').read_bytes()).hexdigest()
    assert len(output)==580
    latency_rows=[]
    for key in sorted(used_request_keys):
        record=raw[key]['record']
        latency_rows.append((1.,float(record['prompt_tokens']),float(record['completion_tokens']),
                             float(record['latency_seconds'])))
    coefficients,residual_norm=nnls(np.asarray([row[:3] for row in latency_rows]),
                                    np.asarray([row[3] for row in latency_rows]))
    intercept,input_coefficient,output_coefficient=map(float,coefficients)
    for row in output:
        row['estimated_llm_seconds']=(intercept*row['response_events']+
            input_coefficient*row['input_tokens']+output_coefficient*row['output_tokens'])
        row['approx_runtime_seconds']=row['numerical_seconds']+row['estimated_llm_seconds']
    grouped=defaultdict(list)
    for row in output:grouped[row['method'],row['sample_size'],row['strategy'],row['group'],row['budget']].append(row)
    summary=[]
    for (method,n,strategy,group,budget),values in sorted(grouped.items()):
        assert sorted(v['seed'] for v in values)==list(range(10))
        row=dict(method=method,sample_size=n,strategy=strategy,group=group,budget=budget,seeds=10)
        for key in ('expected_distance','input_tokens','output_tokens','total_tokens','response_events',
                    'numerical_seconds','estimated_llm_seconds','approx_runtime_seconds'):
            numbers=[v[key] for v in values]
            row[key+'_mean']=statistics.mean(numbers);row[key+'_se']=statistics.stdev(numbers)/math.sqrt(10)
        summary.append(row)
    assert len(summary)==58
    archived=read_metric_csv(SOURCE/'summary.csv',node_count=20,legacy_normalized=True)
    for row in summary:
        if row['strategy']=='native':continue
        ref=next(r for r in archived if float(r['evidence_cap'])==.8 and
            all(str(row[k])==r[k] for k in ('sample_size','strategy','group','budget')))
        for key in ('expected_distance_mean','expected_distance_se'):
            assert math.isclose(row[key],float(ref[key]),abs_tol=1e-10)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('per_run',output),('summary',summary),('smc_response_ledger',events)]:
        with (out/f'{name}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (out/f'{name}.csv.metrics.json').write_text(json.dumps(metric_metadata(20),indent=2)+'\n')
    protocol=dict(model=manifest['configuration']['model'],provider='deepinfra/bf16',
        configuration=manifest['configuration'],answer_confidence_cap=.8,claim_pool_protocol_locked=True,
        source=str(SOURCE),source_manifest_sha256=completion['manifest_sha256'],sources=hashes,
        baseline_source=str(BASE),baseline_source_protocol_sha256=hashes[str((BASE/'protocol.json').relative_to(ROOT))],
        token_accounting=prior['token_accounting'],budgets=list(BUDGETS),data_seeds=list(range(10)),
        sample_sizes=[200,1000],smc_checkpoints=480,baseline_records=100,
        response_ledger_events=len(events),new_model_calls=0,new_inference_runs=0,
        runtime_accounting=dict(definition='approximate serial runtime = archived numerical time + fitted LLM time',
            fitted_llm_seconds='alpha * requests + beta_input * input_tokens + beta_output * output_tokens',
            alpha_seconds_per_request=intercept,beta_input_seconds_per_token=input_coefficient,
            beta_output_seconds_per_token=output_coefficient,training_requests=len(latency_rows),
            residual_norm=float(residual_norm),numerical_time_smc='saved initialization wall time plus checkpoint time',
            numerical_time_native_baselines='unavailable per dataset; zero lower-bound contribution',
            excluded='scheduler delay, model-server startup, and parallel execution'),
        sources_metric_scale='normalized',output_metric_scale='unnormalized')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(complete=True,smc_checkpoints=480,
        baseline_records=100,summary_points=58,response_cost_audit_passed=True,
        runtime_points=58,runtime_model_training_requests=len(latency_rows),
        new_model_calls=0,new_inference_runs=0),indent=2)+'\n')
    print('Audited 480 existing capped checkpoints; 100 baseline records retained.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    prepare(ROOT/'results'/parser.parse_args().run_id/'neuropathic_performance_cost')
