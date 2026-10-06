"""Change only the response-confidence cap in the locked Figure 3 protocol."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import math
from pathlib import Path
import statistics

import numpy as np

from cd_llm.core.finite_support import graph_edges
from cd_llm.experiments import neuropathic_claim_group_ablation as original
from cd_llm.experiments.common.artifacts import read_json, write_json, write_csv, sha256, json_value
from cd_llm.experiments.neuropathic_active_interpretation import graph
from cd_llm.experiments.neuropathic_claim_cache import load_offline_provider
from cd_llm.experiments.neuropathic_claim_group_checkpoints import BUDGETS, prefix_record, validate_checkpoints
from cd_llm.experiments.neuropathic_confidence_cap import cap_response
from cd_llm.experiments.neuropathic_grid_interpretation import inference_arguments, validate_population
from cd_llm.methods.active_query import active_query_smc, dag_query_pool
from cd_llm.methods.graph_scores import BDeuScore
from cd_llm.metrics.units import metric_metadata
from cd_llm.paths import WORKSPACE_ROOT as ROOT

SOURCE=ROOT/'results/20260917T174601Z/neuropathic_claim_group_checkpoints'
PREVIOUS=ROOT/'results/20260919T000434Z/neuropathic_performance_cost'
STUDY='neuropathic_performance_cost'
CAP=.8


def prepare(root):
    source=read_json(SOURCE/'manifest.json')
    complete=read_json(SOURCE/'completion.json')
    assert complete['complete'] and complete['audit_passed']
    assert sha256(SOURCE/'manifest.json')==complete['manifest_sha256']
    inputs={str(SOURCE/name):digest for name,digest in source['inputs'].items()
            if name.startswith(('inputs/','initializations/','evaluation/','cache/inputs/'))}
    for path,digest in inputs.items():
        assert sha256(Path(path))==digest,path
    paths=['code/scripts/run_neuropathic_cost_cap.py','code/src/cd_llm/core/finite_support.py',
           'code/src/cd_llm/experiments/neuropathic_claim_group_ablation.py','code/src/cd_llm/methods/active_query.py',
           'code/src/cd_llm/methods/evidence_guided.py','code/src/cd_llm/methods/graph_scores.py',
           'code/src/cd_llm/experiments/neuropathic_confidence_cap.py',
           'code/src/cd_llm/experiments/neuropathic_grid_interpretation.py',
           'code/src/cd_llm/experiments/neuropathic_claim_cache.py']
    manifest=dict(tasks=source['tasks'],inputs=inputs,source=str(SOURCE),
        source_manifest_sha256=sha256(SOURCE/'manifest.json'),
        previous_cost_protocol_sha256=sha256(PREVIOUS/'protocol.json'),
        previous_per_run_sha256=sha256(PREVIOUS/'per_run.csv'),
        implementation={p:sha256(ROOT/p) for p in paths},
        configuration=dict(source['configuration'],answer_confidence_cap=CAP,
                           selection_confidence=.9,engine_soft_confidence_cap=.99),
        budgets=list(BUDGETS),answer_confidence_cap=CAP,new_model_calls=0)
    write_json(root/'manifest.json',manifest)
    print('Prepared 120 offline trajectories; only answer-confidence cap changes.',flush=True)


def verify(root):
    m=read_json(root/'manifest.json')
    for path,digest in m['implementation'].items():assert sha256(ROOT/path)==digest,path
    assert sha256(SOURCE/'manifest.json')==m['source_manifest_sha256']
    return m


def execute(task,provider):
    population=SOURCE/f"initializations/{task['condition']}_grid_union.json"
    particles,weights=validate_population(read_json(population))
    data=np.load(SOURCE/f"inputs/{task['condition']}.npy",allow_pickle=False)
    score=BDeuScore(data,equivalent_sample_size=1,structure_prior=1,cardinalities=(2,)*20,max_parents=12)
    arguments=inference_arguments(task,particles,weights,'literal')
    arguments['budgets']=BUDGETS
    result=active_query_smc(None,score,provider,query_pool=dag_query_pool(20,original.GROUPS[task['group']]),
                            budget_unit='attempted',**arguments)
    distributions={str(b):[dict(graph=graph_edges(g),weight=w) for g,w in sorted(result.distributions[b].items())]
                   for b in BUDGETS}
    return json_value(dict(task=task,initialization_sha256=sha256(population),
        budget_unit=result.budget_unit,trace=[dict(asdict(s),cumulative_query_cost=i) for i,s in enumerate(result.steps,1)],
        distributions=distributions,distribution=distributions['100'],checkpoint_seconds=result.cumulative_seconds,
        attempted_queries=result.query_count,answered_queries=result.answered_query_count,
        unused_query_budget=result.unused_query_budget,early_stop_reason=result.early_stop_reason,
        elapsed_seconds=result.timing.total_seconds))


def run_block(root,block):
    m=verify(root)
    for path,digest in m['inputs'].items():
        if '/cache/inputs/' in path or f'_s{block//2:02d}_n{1000 if block%2 else 200}' in path:
            assert sha256(Path(path))==digest,path
    with original.offline_guard():
        experts={s:load_offline_provider(SOURCE/'cache',s,minimum_valid=6) for s in original.STRATEGIES}
        for task in (t for t in m['tasks'] if t['block']==block):
            expert=experts[task['strategy']]
            # Four uncapped controls establish exact reproduction of the original protocol.
            if block<2 and task['group']=='all':
                control=execute(task,expert)
                reference=read_json(SOURCE/f"records/{task['id']}.json")
                original.assert_same_inference(control,reference)
                assert control['distributions']==reference['distributions']
                write_json(root/f"controls/{task['id']}.json",dict(passed=True,checkpoints=4,
                    reference_sha256=sha256(SOURCE/f"records/{task['id']}.json")))
            record=execute(task,lambda query:cap_response(expert(query),CAP))
            record['manifest_sha256']=sha256(root/'manifest.json')
            validate_checkpoints(record)
            assert len(record['trace'])==100 and record['attempted_queries']==100
            for step in record['trace']:
                assert step['selection_effective_confidence']==.9
                if step['valid_response']:
                    assert step['effective_confidence']==min(step['response_audit']['raw_reported_confidence'],CAP)
                else:assert not step['resampled'] and step['rejuvenation_attempted_moves']==0
            write_json(root/f"records/{task['id']}.json",record)
            print(task['id'],record['elapsed_seconds'],flush=True)


def report(root):
    m=verify(root)
    controls=list((root/'controls').glob('*.json'))
    assert len(controls)==4 and all(read_json(p)['passed'] for p in controls)
    assert sha256(PREVIOUS/'per_run.csv')==m['previous_per_run_sha256']
    assert sha256(PREVIOUS/'protocol.json')==m['previous_cost_protocol_sha256']
    previous=read_json(PREVIOUS/'protocol.json')
    truth=graph(read_json(SOURCE/'evaluation/truth_graph.json')['graph'])
    caches={s:read_json(SOURCE/f'cache/inputs/caches/qwen_{s}.json') for s in original.STRATEGIES}
    rows,events,hashes=[],[],{}
    for task in m['tasks']:
        path=root/f"records/{task['id']}.json"
        record=read_json(path);hashes[task['id']]=sha256(path)
        assert record['manifest_sha256']==sha256(root/'manifest.json')
        assert record['initialization_sha256']==m['inputs'][str(SOURCE/f"initializations/{task['condition']}_grid_union.json")]
        validate_checkpoints(record)
        assert len(record['trace'])==100
        for slot,step in enumerate(record['trace'],1):
            keys=step['response_audit']['request_keys']
            assert len(keys)==(8 if task['strategy']=='consistency' else step['response_attempts'])
            ip=op=0
            for index,key in enumerate(keys):
                cached=caches[task['strategy']][key]['record']
                assert cached['response_model']==previous['model']
                for field in ('prompt_tokens','completion_tokens'):
                    assert cached[field]==cached['response']['usage'][field]
                ip+=cached['prompt_tokens'];op+=cached['completion_tokens']
                events.append(dict(method='SMC-Active',sample_size=task['sample_size'],strategy=task['strategy'],
                    group=task['group'],seed=task['seed'],query=slot,request_key=key,
                    input_tokens=cached['prompt_tokens'],output_tokens=cached['completion_tokens'],
                    abstention=not step['valid_response'],repair=task['strategy']=='stated' and index>0))
            assert (ip,op)==(step['prompt_tokens'],step['completion_tokens'])
            assert step['selection_effective_confidence']==.9 and step['response_audit']['answer_confidence_cap']==CAP
            if task['strategy']=='consistency':
                audit=step['response_audit'];votes=[v for v in audit['votes'] if v is not None]
                assert len(audit['votes'])==8 and audit['minimum_valid_response_count']==6
                assert step['valid_response']==(len(votes)>=6)
            if step['valid_response']:
                assert step['effective_confidence']==min(step['response_audit']['raw_reported_confidence'],CAP)
        for budget in BUDGETS:
            metrics,_=original.metrics_for(prefix_record(record,budget),truth)
            rows.append(dict(method='SMC-Active',sample_size=task['sample_size'],strategy=task['strategy'],
                group=task['group'],seed=task['seed'],budget=budget,
                expected_distance=metrics['expected_distance'],input_tokens=metrics['prompt_tokens'],
                output_tokens=metrics['completion_tokens'],total_tokens=metrics['prompt_tokens']+metrics['completion_tokens'],
                response_events=metrics['logical_requests'],abstentions=metrics['abstentions']))
    for row in csv.DictReader((PREVIOUS/'per_run.csv').open()):
        if row['strategy']!='native':continue
        rows.append({key:float(row[key]) if key=='expected_distance' else
                     int(row[key]) if key in ('sample_size','seed','budget','input_tokens','output_tokens','total_tokens','response_events')
                     else row[key] for key in rows[0]})
    assert len(rows)==580
    grouped=defaultdict(list)
    for row in rows:grouped[row['method'],row['sample_size'],row['strategy'],row['group'],row['budget']].append(row)
    summaries=[]
    for (method,n,strategy,group,budget),members in sorted(grouped.items()):
        assert sorted(r['seed'] for r in members)==list(range(10))
        row=dict(method=method,sample_size=n,strategy=strategy,group=group,budget=budget,seeds=10)
        for field in ('expected_distance','input_tokens','output_tokens','total_tokens','response_events'):
            values=[r[field] for r in members]
            row[field+'_mean']=statistics.mean(values);row[field+'_se']=statistics.stdev(values)/math.sqrt(10)
        summaries.append(row)
    assert len(summaries)==58
    for name,values in [('per_run',rows),('summary',summaries),('smc_response_ledger',events)]:
        write_csv(root/f'{name}.csv',values)
        write_json(root/f'{name}.csv.metrics.json',metric_metadata(20))
    protocol=dict(previous,configuration=m['configuration'],answer_confidence_cap=CAP,
        table2_group_i_endpoints_reproduced=False,matched_protocol=True,
        excluded_protocols=['new ancestry prompt','OpenRouter API'],
        cap_only_change_from_original=True,uncapped_control_reproductions=4,
        sources={str(root/'manifest.json'):sha256(root/'manifest.json'),
                 str(PREVIOUS/'protocol.json'):m['previous_cost_protocol_sha256']},
        baseline_accounting='Unchanged native baseline records from the previously audited original-model comparison.',
        smc_checkpoints=480,baseline_records=100,new_model_calls=0)
    write_json(root/'protocol.json',protocol)
    write_json(root/'audit.json',dict(passed=True,records=hashes,uncapped_controls=4,
        response_events=len(events),cap=.8,acquisition_confidence=.9))
    write_json(root/'completion.json',dict(complete=True,matched_protocol=True,smc_checkpoints=480,
        baseline_records=100,summary_points=58,new_model_calls=0))
    print('Completed 480 capped checkpoints and validated all response costs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run-block','report'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--block',type=int)
    args=parser.parse_args();root=ROOT/'results'/args.run_id/STUDY
    if args.stage=='run-block':run_block(root,args.block)
    else:{'prepare':prepare,'report':report}[args.stage](root)
