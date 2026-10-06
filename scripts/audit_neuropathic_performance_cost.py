"""Freeze a matched, recorded-token comparison without new model calls."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

from cd_llm.metrics.units import read_metric_csv, metric_metadata
from cd_llm.paths import WORKSPACE_ROOT as ROOT

SMC = ROOT / 'results/20260917T174601Z/neuropathic_claim_group_checkpoints'
BASE = ROOT / 'results/20260903T022032Z/neuropathic_decontamination_qwen'
TABLE = ROOT / 'results/20260917T023943Z/neuropathic_valid_answer_budget'
BUDGETS = (10, 20, 50, 100)
MODEL = 'Qwen/Qwen3.5-9B'
METHODS = {
    'gpc-guess_plus_qwen3.5-9b': 'gPC-Guess',
    'bfspluspearson_plus_qwen3.5-9b': 'BFS+Pearson',
    'scp_pattern_2_plus_qwen3.5-9b': 'SCP Pattern 2',
    'causal_orderpluspc_plus_qwen3.5-9b': 'Causal Order+PC',
    'pcpluslong-srisk_plus_qwen3.5-9b': 'PC+Long-Srisk',
}


def audit(out):
    from cd_llm.baselines.long2023 import ambiguous_pairs
    from cd_llm.baselines.jiralerspong2024 import Completion, run_bfs
    from cd_llm.experiments.neuropathic_decontamination_qwen import (
        _partial, _scp_probabilities, build_spec, build_knowledge_prompt,
        build_decision_prompt, DOMAIN_CONTEXT)
    sources = {}

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None:
            assert digest == expected, path
        sources[str(path.relative_to(ROOT))] = digest
        return json.loads(raw)

    def token_pair(record, keys=('prompt_tokens', 'completion_tokens')):
        values = tuple(record[k] for k in keys)
        assert all(isinstance(v, int) and v >= 0 for v in values), values
        assert sum(values) > 0
        return values

    def database_usage(path):
        # Hash logical rows, not a mutable SQLite file/WAL container.
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            records = db.execute('SELECT request_key,request_json,response_json,prompt_tokens,completion_tokens FROM responses ORDER BY request_key').fetchall()
        assert records
        sources[str(path.relative_to(ROOT)) + '#logical_rows'] = hashlib.sha256(json.dumps(records).encode()).hexdigest()
        inputs = outputs = 0
        for _, request, response, inp, output in records:
            assert json.loads(request)['model'] == MODEL
            payload = json.loads(response)
            assert token_pair(payload['usage']) == (inp, output)
            inputs += inp
            outputs += output
        return inputs, outputs, len(records)

    completion = read(SMC / 'completion.json')
    assert completion['complete'] and completion['audit_passed'] and completion['checkpoints'] == 480
    manifest = read(SMC / 'manifest.json', completion['manifest_sha256'])
    prior_audit = read(SMC / 'audit.json', completion['audit_sha256'])
    assert prior_audit['passed'] and prior_audit['exact_100_query_matches'] == 120
    config = manifest['configuration']
    for key, value in dict(model=MODEL, consistency_minimum_valid=6, consistency_variants=8,
                           soft_confidence_cap=.99, selection_confidence=.9,
                           particle_count=384, budget_unit='attempted').items():
        assert config[key] == value, key
    assert manifest['budgets'] == list(BUDGETS)
    caches = {strategy: read(SMC / f'cache/inputs/caches/qwen_{strategy}.json',
              manifest['inputs'][f'cache/inputs/caches/qwen_{strategy}.json'])
              for strategy in ('stated', 'consistency')}
    rows = read_metric_csv(SMC / 'per_run.csv', node_count=20, legacy_normalized=True)
    sources[str((SMC/'per_run.csv').relative_to(ROOT))] = hashlib.sha256((SMC/'per_run.csv').read_bytes()).hexdigest()
    metrics = {(int(r['sample_size']), r['strategy'], r['group'], int(r['seed']), int(r['budget'])): r for r in rows}
    assert len(metrics) == 480
    output, events = [], []
    for task in manifest['tasks']:
        record = read(SMC / f"records/{task['id']}.json", prior_audit['records'][task['id']])
        assert record['manifest_sha256'] == completion['manifest_sha256']
        assert len(record['trace']) == 100
        inp = outp = calls = abstentions = repairs = abstention_tokens = repair_tokens = 0
        strategy = task['strategy']
        for slot, step in enumerate(record['trace'], 1):
            keys = step['response_audit']['request_keys']
            assert len(keys) == (8 if strategy == 'consistency' else step['response_attempts'])
            pairs = []
            for attempt, key in enumerate(keys):
                cached = caches[strategy][key]['record']
                assert cached['response_model'] == MODEL
                pair = token_pair(cached)
                assert pair == token_pair(cached['response']['usage'])
                pairs.append(pair)
                repair = strategy == 'stated' and attempt > 0
                repairs += repair
                repair_tokens += sum(pair) if repair else 0
                events.append(dict(method='SMC-Active', sample_size=task['sample_size'],
                    strategy=strategy, group=task['group'], seed=task['seed'], query=slot,
                    request_key=key, input_tokens=pair[0], output_tokens=pair[1],
                    abstention=not step['valid_response'], repair=repair))
            current = tuple(sum(p[i] for p in pairs) for i in (0, 1))
            assert current == token_pair(step)
            inp += current[0]
            outp += current[1]
            calls += len(keys)
            abstentions += not step['valid_response']
            abstention_tokens += sum(current) if not step['valid_response'] else 0
            if slot in BUDGETS:
                row = metrics[task['sample_size'], strategy, task['group'], task['seed'], slot]
                assert (inp, outp, calls, abstentions) == tuple(int(row[k]) for k in
                    ('prompt_tokens', 'completion_tokens', 'logical_requests', 'abstentions'))
                output.append(dict(method='SMC-Active', sample_size=task['sample_size'], strategy=strategy,
                    group=task['group'], seed=task['seed'], budget=slot,
                    expected_distance=float(row['expected_distance']), input_tokens=inp, output_tokens=outp,
                    total_tokens=inp+outp, response_events=calls, abstentions=abstentions,
                    repair_events=repairs, abstention_tokens=abstention_tokens, repair_tokens=repair_tokens))
    print('Audited all 480 SMC checkpoints, including abstentions and repairs.', flush=True)
    del caches, record, step, cached

    benchmark = read(BASE/'benchmark/evaluation_manifest.json')
    spec = build_spec()
    metadata = read(SMC/'inputs/model_metadata.json', manifest['inputs']['inputs/model_metadata.json'])
    assert metadata['nodes'] == [{'id': name, 'description': spec.descriptions[name]} for name in spec.names]
    # Verify the observational datasets, not merely their seed labels.
    import numpy as np
    for dataset in benchmark['datasets']:
        path = Path(dataset['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == dataset['sha256']
        sources[str(path.relative_to(ROOT))] = dataset['sha256']
        frozen = SMC/f"inputs/neuropathic_s{dataset['seed']:02d}_n{dataset['sample_size']}.npy"
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == manifest['inputs'][str(frozen.relative_to(SMC))]
        assert np.array_equal(np.load(frozen), np.loadtxt(path, delimiter=',', skiprows=1))
    long_cache = read(BASE/'caches/long.json')
    long_by_pair = {tuple(sorted(r['arc'])): r['record'] for r in long_cache.values()}
    assert len(long_by_pair) == len(long_cache)
    order_tokens = database_usage(BASE/'caches/causal_order.sqlite3')
    shared = read(BASE/'shared/causal_order_expert.json')
    assert order_tokens[:2] == token_pair(shared['audit']['cache'])
    baseline_detail = []
    import pandas as pd
    for slug, method in METHODS.items():
        for n in (200, 1000):
            for seed in range(10):
                record = read(BASE/f'records/{slug}/seed_{seed:02d}_n{n}.json')
                assert record['model'] == MODEL
                assert record['runtime_manifest']['sha256'] == manifest['inputs']['inputs/runtime.json']
                if method == 'gPC-Guess':
                    cached = read(record['expert_cache'])
                    inp, outp = token_pair(cached, ('input_tokens', 'output_tokens'))
                    calls = 1
                elif method == 'BFS+Pearson':
                    path=BASE/f'caches/bfs/seed_{seed:02d}_n{n}.sqlite3'
                    database_usage(path)
                    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
                        responses=db.execute('SELECT request_json,content,prompt_tokens,completion_tokens FROM responses').fetchall()
                    cached={json.dumps(json.loads(req)['messages'],sort_keys=True):(content,ipt,opt)
                            for req,content,ipt,opt in responses}
                    used=[]
                    def complete(messages):
                        key=json.dumps(list(messages),sort_keys=True)
                        content,ipt,opt=cached[key]
                        used.append(key)
                        return Completion(content,ipt,opt)
                    dataset=next(e for e in benchmark['datasets'] if e['seed']==seed and e['sample_size']==n)
                    edges,usage=run_bfs(pd.read_csv(dataset['path']),complete,include_statistics=True,
                        descriptions=spec.descriptions,domain_context=DOMAIN_CONTEXT,
                        opaque_identifiers=True,empty_root_policy='empty_graph')
                    assert sorted(map(tuple,edges))==sorted(map(tuple,record['graph']['directed']))
                    assert set(used)==set(cached)
                    inp,outp=token_pair(usage,('input_tokens','output_tokens'))
                    calls=len(used)
                    assert calls==record['usage']['query_count']
                elif method == 'Causal Order+PC':
                    inp, outp, calls = order_tokens
                    assert record['fallback_audit']['fallback_pair_count'] == 0
                else:
                    entry = next(e for e in benchmark['source_statistics'] if e['seed'] == seed and e['sample_size'] == n)
                    stats = read(entry['path'], entry['sha256'])
                    if method == 'PC+Long-Srisk':
                        pairs = ambiguous_pairs(_partial(stats['pc']))
                        assert len(pairs) == record['query_count']
                        usage = [token_pair(long_by_pair[tuple(sorted(pair))], ('input_tokens','output_tokens')) for pair in pairs]
                    else:
                        cache = {}
                        paths=sorted((BASE/f'caches/scp/seed_{seed:02d}_n{n}').glob('*.json'))
                        # Overlap filesystem latency; JSON decoding still uses one CPU.
                        with ThreadPoolExecutor(max_workers=16) as pool:
                            for item in pool.map(read,paths):
                                req = item['request']
                                assert req['settings']['model'] == MODEL
                                key = (req['kind'], req['prompt'], req.get('trial'))
                                assert key not in cache
                                cache[key] = item
                        assert len(cache) == 380*6
                        usage, selected = [], set()
                        for probability in record['probability_records']:
                            knowledge = build_knowledge_prompt(pattern=2, source=probability['source'],
                                target=probability['target'], nodes=spec.names,
                                method='PC with G-squared conditional-independence tests',
                                dataset_description=f'{n} binary clinical observations',
                                pc_bootstrap_probabilities=_scp_probabilities(stats),
                                theme=DOMAIN_CONTEXT, variable_descriptions=spec.descriptions)
                            key = ('explanation', knowledge, None)
                            item = cache[key]
                            assert item['text'] == probability['explanation']
                            selected.add(key)
                            usage.append(token_pair(item, ('input_tokens', 'output_tokens')))
                            decision = build_decision_prompt(knowledge_prompt=knowledge, explanation=item['text'],
                                source=probability['source'], target=probability['target'])
                            for trial, expected in enumerate(probability['trial_probabilities']):
                                key = ('yes_probability', decision, trial)
                                item = cache[key]
                                assert math.isclose(item['probability'], expected, abs_tol=1e-12)
                                selected.add(key)
                                usage.append(token_pair(item, ('input_tokens', 'output_tokens')))
                        assert len(selected) == len(cache)
                    inp, outp = (sum(u[i] for u in usage) for i in (0,1))
                    calls = len(usage)
                if method == 'SCP Pattern 2':
                    print(f'Audited SCP n={n}, seed={seed}: {inp+outp:,} tokens.',flush=True)
                output.append(dict(method=method, sample_size=n, strategy='native', group='none', seed=seed,
                    budget=0, expected_distance=record['metrics']['expected_distance']*190,
                    input_tokens=inp, output_tokens=outp, total_tokens=inp+outp, response_events=calls,
                    abstentions='', repair_events='', abstention_tokens='', repair_tokens=''))
                baseline_detail.append(dict(method=method,sample_size=n,seed=seed,input_tokens=inp,output_tokens=outp,response_events=calls))
        print(f'Audited {method}.', flush=True)

    grouped = defaultdict(list)
    for row in output:
        grouped[row['method'],row['sample_size'],row['strategy'],row['group'],row['budget']].append(row)
    summary = []
    for (method,n,strategy,group,budget), values in sorted(grouped.items()):
        assert sorted(v['seed'] for v in values) == list(range(10))
        row = dict(method=method,sample_size=n,strategy=strategy,group=group,budget=budget,seeds=10)
        for key in ('expected_distance','total_tokens','input_tokens','output_tokens','response_events'):
            numbers = [v[key] for v in values]
            row[key+'_mean'] = statistics.mean(numbers)
            row[key+'_se'] = statistics.stdev(numbers)/math.sqrt(10)
        summary.append(row)
    assert len(output) == 580 and len(summary) == 58
    table = read_metric_csv(TABLE/'summary.csv',node_count=20,legacy_normalized=True)
    for row in summary:
        if row['group'] == 'group_i' and row['budget'] == 100:
            reference = next(r for r in table if r['arm']=='table2' and int(r['sample_size'])==row['sample_size'] and r['strategy']==row['strategy'])
            assert math.isclose(row['expected_distance_mean'],float(reference['expected_distance_mean']),abs_tol=1e-10)
            assert math.isclose(row['total_tokens_mean'],(int(reference['prompt_tokens'])+int(reference['completion_tokens']))/10,abs_tol=1e-10)
    out.mkdir(parents=True,exist_ok=True)
    for name, values in [('per_run',output),('summary',summary),('smc_response_ledger',events)]:
        with (out/f'{name}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        (out/f'{name}.csv.metrics.json').write_text(json.dumps(metric_metadata(20),indent=2)+'\n')
    lock = dict(model=MODEL,model_revision=record['model_revision'],configuration=config,
        budgets=list(BUDGETS),data_seeds=list(range(10)),sample_sizes=[200,1000],
        sources=sources,baseline_native_protocols=True,table2_group_i_endpoints_reproduced=True,
        excluded_protocols=['new ancestry prompt','OpenRouter API','confidence cap 0.8'],
        token_accounting='Recorded input plus output tokens for every response required by a trajectory, including abstentions, eight consistency variants, and format repairs. Cached responses and shared experts are charged in full per dataset, without amortization. No costs are inferred from question counts. Tokens are not monetary fees or numerical inference cost.',
        smc_checkpoints=480,baseline_records=100,response_ledger_events=len(events),
        sources_metric_scale='normalized',output_metric_scale='unnormalized',new_model_calls=0)
    (out/'protocol.json').write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(complete=True,matched_protocol=True,
        smc_checkpoints=480,baseline_records=100,summary_points=58),indent=2)+'\n')
    print(f'Complete: {out}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    audit(ROOT/'results'/args.run_id/'neuropathic_performance_cost')
