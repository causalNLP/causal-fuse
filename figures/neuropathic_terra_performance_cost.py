"""Figure 4: audited Terra performance and token cost in Figure 3's style."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import statistics

import matplotlib
matplotlib.use('Agg')
from neuropathic_performance_cost import render

ROOT = Path(__file__).resolve().parents[2]
SMC = ROOT / 'results/20260920T031654Z/neuropathic_terra_smc'
ORDER = ROOT / 'results/20260919T145904Z/neuropathic_causal_order_terra'
BASE = ROOT / 'results/20260902T234859Z/neuropathic_decontamination'
MODEL = 'openai/gpt-5.6-terra'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, values):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def generate(stamp):
    if not re.fullmatch(r'\d{8}T\d{6}Z', stamp):
        raise ValueError('Use a UTC timestamp for --run-id')
    out = ROOT / 'results' / stamp / 'neuropathic_terra_performance_cost'
    if out.exists():
        raise FileExistsError(out)
    hashes = {}

    def read(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        hashes[str(path.relative_to(ROOT))] = digest
        return json.loads(path.read_text())

    completions = {}
    manifests = {}
    for source in (SMC, ORDER):
        completed = read(source / 'completion.json')
        assert completed['complete'] and completed['audit_passed']
        manifests[source] = read(source / 'manifest.json', completed['manifest_sha256'])
        for name in ('per_run.csv', 'summary.csv', 'per_run.csv.metrics.json'):
            path = source / name
            assert sha(path) == completed['artifacts'][name]
            hashes[str(path.relative_to(ROOT))] = sha(path)
        assert read(source / 'per_run.csv.metrics.json')['metric_scale'] == 'unnormalized'
        completions[source] = completed
    assert completions[SMC]['trajectories'] == 120 and completions[SMC]['checkpoints'] == 480
    assert manifests[SMC]['answer_confidence_cap'] == .8
    assert manifests[SMC]['model'] == MODEL and manifests[ORDER]['model'] == MODEL
    output = []
    for r in rows(SMC / 'per_run.csv'):
        assert int(r['attempted']) == int(r['budget'])
        inp, opt = int(r['prompt_tokens']), int(r['completion_tokens'])
        assert inp + opt == int(r['total_tokens'])
        output.append(dict(method='SMC-Active', sample_size=int(r['sample_size']),
            strategy=r['strategy'], group=r['group'], seed=int(r['seed']), budget=int(r['budget']),
            expected_distance=float(r['expected_distance']), input_tokens=inp,
            output_tokens=opt, total_tokens=inp+opt))
    for r in rows(ORDER / 'per_run.csv'):
        assert r['status'] == 'complete'
        inp, opt = int(r['input_tokens']), int(r['output_tokens'])
        assert inp+opt == int(r['total_tokens'])
        output.append(dict(method='Causal Order+PC', sample_size=int(r['sample_size']),
            strategy='native', group='none', seed=int(r['seed']), budget=0,
            expected_distance=float(r['expected_distance']), input_tokens=inp,
            output_tokens=opt, total_tokens=inp+opt))
    # Audit every required native baseline response, without rerunning the methods.
    cache_path = BASE / 'caches/openrouter_request_cache.sqlite3'
    used = {}
    with sqlite3.connect(f'file:{cache_path}?mode=ro', uri=True) as db:
        for folder, method in (('gpc', 'gPC-Guess'), ('bfs', 'BFS+Pearson')):
            for n in (200, 1000):
                for seed in range(10):
                    relative = f'reference/terra/{folder}/seed_{seed:02d}_n{n}.json'
                    record = read(ORDER / relative, manifests[ORDER]['inputs'][relative])
                    assert record['model'] == MODEL
                    keys = [record['request']['request_key']] if folder == 'gpc' else record['request_keys']
                    usage = record['request'] if folder == 'gpc' else record['cost']
                    costs = []
                    for key in keys:
                        result = db.execute('SELECT request_json,response_json,prompt_tokens,completion_tokens FROM responses WHERE request_key=?', (key,)).fetchone()
                        assert result is not None, key
                        request, response = json.loads(result[0]), json.loads(result[1])
                        assert request['model'] == MODEL
                        assert (response['usage']['prompt_tokens'], response['usage']['completion_tokens']) == result[2:]
                        used[key] = result
                        costs.append(result[2:])
                    inp, opt = (sum(pair[i] for pair in costs) for i in (0, 1))
                    assert (inp, opt) == (usage['input_tokens'], usage['output_tokens'])
                    output.append(dict(method=method, sample_size=n, strategy='native', group='none',
                        seed=seed, budget=0, expected_distance=record['metrics']['expected_distance'] * 190,
                        input_tokens=inp, output_tokens=opt, total_tokens=inp+opt))
    hashes[str(cache_path.relative_to(ROOT))+'#required_rows'] = hashlib.sha256(
        json.dumps(used, sort_keys=True).encode()).hexdigest()
    assert len(output) == 540
    grouped = defaultdict(list)
    for r in output:
        grouped[r['method'],r['sample_size'],r['strategy'],r['group'],r['budget']].append(r)
    summary = []
    source_summary = rows(SMC / 'summary.csv')
    for key, members in sorted(grouped.items()):
        assert sorted(r['seed'] for r in members) == list(range(10)), key
        row = dict(zip(('method','sample_size','strategy','group','budget'), key), seeds=10)
        for metric in ('expected_distance','input_tokens','output_tokens','total_tokens'):
            values = [r[metric] for r in members]
            row[metric+'_mean'] = statistics.mean(values)
            row[metric+'_se'] = statistics.stdev(values) / math.sqrt(10)
        if row['method'] == 'SMC-Active':
            original = next(r for r in source_summary if all(str(row[k]) == r[k] for k in ('sample_size','strategy','group','budget')))
            for metric in ('expected_distance','total_tokens'):
                for suffix in ('_mean','_se'):
                    assert math.isclose(row[metric+suffix], float(original[metric+suffix]), abs_tol=1e-12)
        summary.append(row)
    assert len(summary) == 54
    figure3 = ROOT / 'paper/figs/neuropathic_performance_cost.pdf'
    prior_figure3 = sha(figure3)
    out.mkdir(parents=True)
    write_csv(out/'per_run.csv', output)
    write_csv(out/'summary.csv', summary)
    for name in ('per_run', 'summary'):
        (out/f'{name}.csv.metrics.json').write_text(json.dumps(dict(metric_scale='unnormalized', node_count=20))+'\n')
    protocol = dict(model=MODEL, answer_confidence_cap=.8, claim_pool_protocol_locked=True,
        budgets=[10,20,50,100], seeds=list(range(10)), sample_sizes=[200,1000],
        source_hashes=hashes, source_manifest_sha256=completions[SMC]['manifest_sha256'],
        figure3_pdf_sha256=prior_figure3, baseline_methods=['gPC-Guess','BFS+Pearson','Causal Order+PC'],
        baseline_protocol='Native procedures; shared expert responses charged in full to each dataset.',
        token_accounting='Input plus output tokens, including abstentions, consistency variants and repairs; no amortization.',
        metric_scale='unnormalized', error_bars='One standard error across ten paired data seeds.',
        new_model_calls=0, new_inference_runs=0,
        script_sha256=sha(Path(__file__)), renderer_sha256=sha(Path(__file__).with_name('neuropathic_performance_cost.py')))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    pdf = out/'neuropathic_terra_performance_cost.pdf'
    render(summary, out, .8, pdf, model='GPT-5.6 Terra', xlim=(550,2e6), ylim=(9,32))
    shutil.copy2(pdf, ROOT/'paper/figs'/pdf.name)
    assert sha(figure3) == prior_figure3
    (out/'completion.json').write_text(json.dumps(dict(complete=True, checkpoints=480, baseline_conditions=60,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    generate(parser.parse_args().run_id)
