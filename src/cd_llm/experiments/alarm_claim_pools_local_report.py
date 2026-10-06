"""Offline scoring, prespecified contrasts, and publication artifacts for ALARM."""
from collections import Counter, defaultdict
from itertools import product
import math
import shutil
import statistics
import subprocess

import numpy as np

from cd_llm.core.finite_support import distribution_metrics, graph_edges, stable_seed
from cd_llm.experiments.common.artifacts import read_json, sha256, write_csv, write_json, write_text
from cd_llm.experiments import alarm_claim_pools_local as study
from cd_llm.methods.active_query import ClaimQuery, query_values
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.metrics.units import metric_metadata

from cd_llm.experiments.alarm_claim_pools_report import score_record, aggregate, contrasts


def report(root):
    m = study.verify(root)
    gate = study.prompts.require_cache(root)
    study.require_pilot(root)
    rows, events, ledger, hashes = [], [], [], {}
    with study.offline_guard():
        answer = study.provider(root,m)
        truth = study.graph(read_json(root/'evaluation/truth_graph.json')['graph'])
        raw = read_json(root/'cache/raw_responses.json')
        for task in study.tasks():
            path = root / f'records/{task["id"]}.json'
            record = read_json(path)
            study.validate_record(root,record,task,answer)
            hashes[task['id']] = sha256(path)
            for budget in (0,*study.BUDGETS):
                metrics, diagnostic = score_record(record,truth,budget)
                rows.append(dict(sample_size=task['sample_size'],group=task['group'],strategy='stated',
                    seed=task['seed'],budget=budget,**metrics))
                if budget == 100:
                    events.extend(dict(task=task['id'],sample_size=task['sample_size'],group=task['group'],seed=task['seed'],**e) for e in diagnostic)
            for slot,step in enumerate(record['trace'],1):
                usage = [raw[key]['record'] for key in step['response_audit']['request_keys']]
                if (sum(r['prompt_tokens'] for r in usage) != step['prompt_tokens']
                    or sum(r['completion_tokens'] for r in usage) != step['completion_tokens']):
                    raise ValueError('trace expenditure differs from response usage')
                for attempt,r in enumerate(usage):
                    ledger.append(dict(task=task['id'],slot=slot,request_key=r['request_key'],
                        prompt_tokens=r['prompt_tokens'],completion_tokens=r['completion_tokens'],
                        repair=attempt>0,abstention=not step['valid_response']))
    if len(rows) != 300 or len(events) != 6000:
        raise ValueError('60 trajectories, 240 checkpoints, and 60 initialization rows required')
    summary, paired = aggregate(rows), contrasts(rows)
    diagnostics = []
    for n in (200,1000):
        for group in study.GROUPS:
            for lo, hi in ((1,10),(11,20),(21,50),(51,100)):
                for kind in ('directed','adjacency','ancestry'):
                    chosen = [e for e in events if e['sample_size']==n and e['group']==group and e['kind']==kind and lo<=e['slot']<=hi]
                    valid = [e for e in chosen if e['valid']]
                    negatives = [e for e in valid if e['truth']==0]
                    positives = [e for e in valid if e['truth']==1]
                    diagnostics.append(dict(sample_size=n,group=group,first_attempt=lo,last_attempt=hi,kind=kind,
                        attempted=len(chosen),valid=len(valid),coverage=len(valid)/len(chosen) if chosen else None,
                        accuracy=sum(e['correct'] for e in valid)/len(valid) if valid else None,
                        true_positive=sum(e['answer']==1 for e in positives),false_negative=sum(e['answer']==0 for e in positives),
                        false_positive=sum(e['answer']==1 for e in negatives),true_negative=sum(e['answer']==0 for e in negatives),
                        false_positive_rate=sum(e['answer']==1 for e in negatives)/len(negatives) if negatives else None,
                        indirect_ancestry=sum(e['indirect_ancestry'] for e in chosen)))
    outputs = dict(per_run=rows,summary=summary,paired_ed_differences=paired,
                   selected_claims=events,claim_diagnostics=diagnostics,response_ledger=ledger)
    for name, value in outputs.items():
        write_csv(root/f'{name}.csv',value)
        write_json(root/f'{name}.csv.metrics.json',metric_metadata(37))
    write_json(root/'summary.json',dict(summary=summary,paired=paired,diagnostics=diagnostics))
    lines = ['# Local Qwen ALARM heterogeneous claim study','',
        'Qwen3.5-9B, stated confidence capped at 0.8. All 60 trajectories and 240 nonzero checkpoints are complete. '
        'ED and ES are unnormalized. Intermediate results are prefixes without additional terminal mutation. '
        'Uncertainty is conditional on this network and shared expert collection.','',
        '| n | Pool | Attempts | ED (SE) | Tokens (SE) |','|---:|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['sample_size']} | {r['group']} | {r['budget']} | {r['expected_distance_mean']:.3f} ({r['expected_distance_se']:.3f}) | {r['total_tokens_mean']:.1f} ({r['total_tokens_se']:.1f}) |")
    lines += ['', '## Prespecified contrasts','',
        'Negative differences favor mixed claims. The 95% percentile bootstrap intervals use 10,000 paired resamples. '
        'family_low/high give 98.75% intervals for each family of four comparisons (Bonferroni). '
        'Holm-adjusted exact sign-flip p-values assume symmetric paired differences. An interval containing zero does not establish equivalence.','',
        '| Comparison | n | Restricted pool | Difference [95% interval] | Holm p |',
        '|---|---:|---|---:|---:|']
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['sample_size']} | {r['right_group']} | {r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] | {r['holm_p_value']:.4f} |")
    runtime_rows = [read_json(p) for p in sorted((root/'gpu_runtime').glob('*.json'))]
    write_json(root/'gpu_runtime.json', dict(jobs=runtime_rows, allocated_gpu_wall_seconds=sum(r['elapsed_seconds'] for r in runtime_rows), actual_api_spending_usd=0.))
    lines += ['', f"GPU allocation runtime: {sum(r['elapsed_seconds'] for r in runtime_rows):.1f} seconds across {len(runtime_rows)} collection jobs."]
    lines += ['',f"Local collection accounting (API expenditure is zero): {gate['accounting']}",
        f"Parsing policy: {gate['parsing_policy']}; recovered queries: {gate['syntactically_recovered_queries']}. "
        'This local recovery adds only a missing final object brace after the normal format-repair attempt; '
        'raw responses, stated confidence, generation, and token expenditure are preserved.',
        'Per-dataset tokens include all recorded required responses, abstentions, and format repairs without amortizing cache reuse. '
        'Transport attempts without returned usage are reported as unresolved local attempts; their unobserved tokens are not imputed.',
        'Earlier Qwen ALARM baselines use different observations and protocols and are not a matched ablation with this study.']
    write_text(root/'report.md','\n'.join(lines)+'\n')
    write_json(root/'audit.json',dict(passed=True,records=hashes,trajectories=60,checkpoints=240,
        manifest_sha256=sha256(root/'manifest.json'),network_disabled_during_scoring=True))
    artifacts = [f'{n}.csv' for n in outputs]+[f'{n}.csv.metrics.json' for n in outputs]
    artifacts += ['summary.json','report.md','audit.json','cache_audit.json','accounting.json','gpu_runtime.json']
    write_json(root/'completion.json',dict(complete=True,audit_passed=True,trajectories=60,checkpoints=240,
        manifest_sha256=sha256(root/'manifest.json'),artifacts={n:sha256(root/n) for n in artifacts}))
    subprocess.run([str(study.ROOT/'.venv/bin/python'),str(study.ROOT/'code/figures/alarm_claim_pool_efficiency_local.py'),
                    '--run-id',root.parent.name],check=True)
    write_json(root/'status.json',dict(stage='complete',trajectories=60,checkpoints=240,accounting=gate['accounting']))
