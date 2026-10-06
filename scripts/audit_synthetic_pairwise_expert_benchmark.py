"""Audit frozen shared facts, posterior replay, unchanged baselines and history."""
import argparse
import csv
from pathlib import Path

import numpy as np

from cd_llm.experiments import synthetic_pairwise_expert_benchmark as study
from cd_llm.experiments.common.artifacts import read_json, sha256, write_json
from cd_llm.paths import WORKSPACE_ROOT


def audit(identifier, graph_indices=None):
    root = study.root_for(identifier)
    manifest = study.verify(root)
    summaries = []
    for g in (study.SCMS if graph_indices is None else graph_indices):
        facts = read_json(Path(manifest['facts'][str(g)]['path']))
        source = read_json(Path(manifest['sources'][str(g)]['path']))
        saved = read_json(Path(manifest['reuse']['partials'][str(g)]['path']))
        pairs = study.selected_pairs(20, manifest['seeds'][str(g)]['pairs'])
        assert facts['records'] == study.answer_pairs(study._graph(source['target'], 20), pairs)
        inputs = study.native_inputs(facts['records'])
        assert study.plain(inputs) == facts['native_inputs']
        for n in study.SIZES:
            _, _, particles, weights, initial = study.archive.restored_initialization(saved['artifacts'][str(n)], 'pc-only')
            assert len(particles) == 384 and np.array_equal(weights, np.full(384, 1 / 384))
            fs = read_json(study.result_path(root, g, n, 'FS'))
            assert fs['status'] == 'ok'
            assert study._serialize(study.finite_posterior(initial, inputs['FS'])) == fs['distribution']
            smc = read_json(study.result_path(root, g, n, 'SMC-Passive'))
            assert smc['status'] == 'ok'
            trace = smc['details']['trace']
            assert len(trace) == 20
            assert [s['query']['query_id'] for s in trace] == [r['fact_id'] for r in facts['records']]
            assert all(s['reported_confidence'] == 1 and s['effective_confidence'] == .99 for s in trace)
            assert trace[-1]['rejuvenation_attempted_moves'] == 768
            assert smc['seeds']['inference'] == saved['artifacts'][str(n)]['seeds']['inference']
            statuses = {}
            for method in study.NEW_METHODS:
                r = read_json(study.result_path(root, g, n, method))
                assert r['native_input'] == facts['native_inputs'][method]
                assert r['manifest_sha256'] == sha256(root / 'manifest.json')
                statuses[method] = r['status']
            summaries.append(dict(graph_index=g, sample_size=n, statuses=statuses,
                positive_pairs=sum(r['state'] != 'absent' for r in facts['records']),
                smc_final_proposals=trace[-1]['rejuvenation_attempted_moves']))
    preserved = read_json(WORKSPACE_ROOT / 'out' / identifier / 'preserved_hashes.json')
    for path, digest in preserved.items():
        assert sha256(WORKSPACE_ROOT / path) == digest, path
    completion_path = root / 'completion.json'
    baseline_checked = 0
    if completion_path.exists():
        completion = read_json(completion_path)
        with (study.archive.root_for(study.INITIALIZATION_RUN) / 'summary.csv').open() as handle:
            old = {(int(r['sample_size']), r['method']): r for r in csv.DictReader(handle) if int(r['budget']) == 0}
        for row in completion['summary']:
            if row['method'] in study.METHODS[:3]:
                for metric in study.METRICS:
                    for suffix in ('_mean', '_se'):
                        assert row[metric + suffix] == float(old[row['sample_size'], row['method']][metric + suffix])
                baseline_checked += 1
    report = dict(slices=summaries, preserved_artifacts=len(preserved),
                  unchanged_baseline_summary_rows=baseline_checked)
    name = 'audit.json' if graph_indices is None else 'pilot_audit.json'
    write_json(root / name, report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--graph-index', type=int)
    args = parser.parse_args()
    audit(args.run_id, None if args.graph_index is None else [args.graph_index])
