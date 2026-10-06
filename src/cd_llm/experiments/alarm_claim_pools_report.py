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
from cd_llm.experiments import alarm_claim_pools as study
from cd_llm.methods.active_query import ClaimQuery, query_values
from cd_llm.methods.llm_claim_confidence import confidence_metrics
from cd_llm.metrics.units import metric_metadata

METRICS = ('expected_distance','energy_score','adjacency_f1','orientation_f1','total_tokens')


def bootstrap(values, identity):
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError('ten paired finite differences required')
    rng = np.random.default_rng(stable_seed(study.STUDY, 'bootstrap', identity))
    means = values[rng.integers(0,10,size=(10000,10))].mean(axis=1)
    low, high = np.quantile(means,[.025,.975])
    family_low, family_high = np.quantile(means,[.05/8,1-.05/8])
    # Exact two-sided sign-flip test, conditional on symmetric paired differences.
    signs = np.asarray(list(product((-1,1), repeat=10)))
    p = float(np.mean(np.abs((signs*values).mean(axis=1)) >= abs(values.mean())-1e-12))
    return dict(mean=float(values.mean()),low=float(low),high=float(high),
                family_low=float(family_low),family_high=float(family_high),p_value=p)


def holm(rows):
    previous = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r:r['p_value'])):
        previous = max(previous,min(1.,(len(rows)-rank)*row['p_value']))
        row['holm_p_value'] = previous
    return rows


def contrasts(rows):
    index = {(r['sample_size'],r['group'],r['seed'],r['budget']):r for r in rows}
    result = []
    for label, budget in (('primary_same_budget',100),('efficiency_50_vs_100',50)):
        family = []
        for n in (200,1000):
            for other in ('group_i','group_ii'):
                values = [index[n,'all',s,budget]['expected_distance']-index[n,other,s,100]['expected_distance'] for s in study.SEEDS]
                family.append(dict(comparison=label,sample_size=n,left_group='all',right_group=other,
                    left_budget=budget,right_budget=100,paired_seeds=10,
                    **bootstrap(values,f'{label}:{n}:{other}')))
        result.extend(holm(family))
    return result


def score_record(record, truth, budget):
    entries, trace = record['distributions'][str(budget)], record['trace'][:budget]
    metrics = distribution_metrics(tuple(study.graph(e['graph']) for e in entries),
        np.asarray([e['weight'] for e in entries]),truth,'dag',normalized=False)
    raw, effective, events = [], [], []
    direct = set(map(tuple,graph_edges(truth)['directed']))
    for slot, step in enumerate(trace,1):
        query = ClaimQuery(**step['query'])
        target = int(query_values((truth,),query)[0])
        valid = step['valid_response']
        correct = int(step['answer'] == target) if valid else None
        confidence = step['response_audit']['raw_reported_confidence']
        if valid:
            raw.append((correct,confidence))
            effective.append((correct,step['effective_confidence']))
        events.append(dict(slot=slot,kind=query.kind,truth=target,answer=step['answer'],valid=valid,
            correct=correct,raw_confidence=confidence,effective_confidence=step['effective_confidence'],
            indirect_ancestry=query.kind=='ancestry' and target==1 and (query.source,query.target) not in direct,
            prompt_tokens=step['prompt_tokens'],completion_tokens=step['completion_tokens']))
    a,b = confidence_metrics(raw),confidence_metrics(effective)
    if not raw:
        a = b = dict(accuracy=None,brier=None,ece=None)
    row = {k:metrics[k] for k in METRICS[:-1]}
    row.update(attempted=budget,answered=len(raw),abstentions=budget-len(raw),
        coverage=len(raw)/budget if budget else None,valid_count=len(raw),
        accuracy=a['accuracy'],raw_brier=a['brier'],raw_ece=a['ece'],effective_brier=b['brier'],effective_ece=b['ece'],
        input_tokens=sum(s['prompt_tokens'] for s in trace),output_tokens=sum(s['completion_tokens'] for s in trace),
        logical_requests=sum(len(s['response_audit']['request_keys']) for s in trace))
    row['total_tokens'] = row['input_tokens']+row['output_tokens']
    return row, events


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sample_size'],row['group'],row['budget']].append(row)
    summary = []
    for (n, pool, budget), members in sorted(groups.items()):
        if sorted(r['seed'] for r in members) != list(study.SEEDS):
            raise ValueError('incomplete condition; do not aggregate successful runs only')
        row = dict(sample_size=n,group=pool,strategy='stated',budget=budget,seeds=10)
        for metric in (*METRICS,'input_tokens','output_tokens'):
            values = [r[metric] for r in members]
            row[metric+'_mean'] = statistics.mean(values)
            row[metric+'_se'] = statistics.stdev(values)/math.sqrt(10)
        valid = sum(r['valid_count'] for r in members)
        attempted = sum(r['attempted'] for r in members)
        row.update(attempted=attempted,valid_count=valid,coverage=valid/attempted if attempted else None)
        for metric in ('accuracy','raw_brier','effective_brier'):
            row[metric] = sum(r[metric]*r['valid_count'] for r in members if r['valid_count'])/valid if valid else None
        summary.append(row)
    if len(summary) != 30:
        raise ValueError('30 aggregate conditions including initialization required')
    return summary


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
    lines = ['# ALARM heterogeneous claim study','',
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
    lines += ['',f"API accounting: {gate['accounting']}",
        'Per-dataset tokens include all recorded required responses, abstentions, and format repairs without amortizing cache reuse. '
        'Transport attempts without returned usage remain conservatively reserved; their unobserved tokens are not imputed.',
        'Earlier Qwen ALARM baselines use different observations and protocols and are not a matched ablation with this study.']
    write_text(root/'report.md','\n'.join(lines)+'\n')
    write_json(root/'audit.json',dict(passed=True,records=hashes,trajectories=60,checkpoints=240,
        manifest_sha256=sha256(root/'manifest.json'),network_disabled_during_scoring=True))
    artifacts = [f'{n}.csv' for n in outputs]+[f'{n}.csv.metrics.json' for n in outputs]
    artifacts += ['summary.json','report.md','audit.json','cache_audit.json','accounting.json']
    write_json(root/'completion.json',dict(complete=True,audit_passed=True,trajectories=60,checkpoints=240,
        manifest_sha256=sha256(root/'manifest.json'),artifacts={n:sha256(root/n) for n in artifacts}))
    subprocess.run([str(study.ROOT/'.venv/bin/python'),str(study.ROOT/'code/figures/alarm_claim_pool_efficiency.py'),
                    '--run-id',root.parent.name],check=True)
    write_json(root/'status.json',dict(stage='complete',trajectories=60,checkpoints=240,accounting=gate['accounting']))


def paper_text(summary, paired):
    index={(r['sample_size'],r['group'],r['budget']):r for r in summary}
    if len(index)!=30 or len(paired)!=8:
        raise ValueError('all prespecified conditions and contrasts required')
    text=[r'\begin{figure}[t]',r'\centering',
          r'\includegraphics[width=\linewidth]{figs/alarm_claim_pool_efficiency.pdf}',
          r'\caption{ALARM recovery with Qwen3.5-9B and stated confidence capped at $0.8$. Colors identify claim pools; columns show attempted queries and recorded token cost. Means and one SE over ten paired seeds; costs include abstentions and repairs.}',
          r'\label{fig:alarm-claim-pool-efficiency}',r'\end{figure}','',
          r'Figure~\ref{fig:alarm-claim-pool-efficiency} compares local, ancestry, and mixed claims with shared PC+GES initialization and inference settings. All nonzero checkpoints are prefixes of the same 100-attempt trajectories.']
    for n in (200,1000):
        values=[index[n,g,100]['expected_distance_mean'] for g in ('group_i','group_ii','all')]
        text[-1]+=f' At $n={n}$ and 100 queries, mean ED is ${values[0]:.2f}$ for local, ${values[1]:.2f}$ for ancestry, and ${values[2]:.2f}$ for mixed claims.'
    primary=[r for r in paired if r['comparison']=='primary_same_budget']
    secondary=[r for r in paired if r['comparison']=='efficiency_50_vs_100']
    better=sum(r['family_high']<0 for r in primary)
    worse=sum(r['family_low']>0 for r in primary)
    text+=['',f'Of the four prespecified same-budget comparisons, {better} have a multiplicity-adjusted bootstrap interval entirely favoring mixed claims and {worse} entirely favoring a restricted pool. '
           'The remaining intervals include zero. These intervals describe variation across observation seeds conditional on this network and shared expert collection.']
    efficiency=sum(r['mean']<0 for r in secondary)
    supported=sum(r['family_high']<0 for r in secondary)
    text+=['',f'Mixed claims at 50 attempts have lower observed mean ED than restricted pools at 100 in {efficiency} of four comparisons; {supported} have an adjusted interval entirely below zero. '
           r'This tests a 50\% reduction in questions; a zero-crossing difference interval does not establish equivalence. The token panels separately show the cost of required responses, including repairs and abstentions. Appendix~\ref{app:alarm-claim-pools} gives the matched protocol.','']
    return '\n'.join(text)


def publish(root):
    study.verify(root)
    completed=read_json(root/'completion.json')
    if not completed['complete'] or not completed['audit_passed'] or completed['trajectories']!=60 or completed['checkpoints']!=240:
        raise ValueError('only complete audited results can enter the paper')
    for name,digest in completed['artifacts'].items():
        if sha256(root/name)!=digest:
            raise ValueError(f'completed artifact changed: {name}')
    figure=read_json(root/'figure_provenance.json')
    if (figure['summary_sha256']!=sha256(root/'summary.csv')
        or figure['figure_sha256']!=sha256(root/'alarm_claim_pool_efficiency.pdf')):
        raise ValueError('figure provenance differs')
    values=read_json(root/'summary.json')
    fragment=paper_text(values['summary'],values['paired'])
    write_text(root/'paper_results.tex',fragment)
    paper=study.ROOT/'paper'
    targets={'sections/alarm_claim_pools_results.tex':root/'paper_results.tex',
             'figs/alarm_claim_pool_efficiency.pdf':root/'alarm_claim_pool_efficiency.pdf'}
    for relative,source in targets.items():
        target=paper/relative
        if target.exists() and sha256(target)!=sha256(source):
            raise ValueError(f'paper artifact already exists with different contents: {relative}')
    for relative,source in targets.items():
        shutil.copyfile(source,paper/relative)
    logs=study.ROOT/'out'/root.parent.name/study.STUDY
    logs.mkdir(parents=True,exist_ok=True)
    with (logs/'paper_build.log').open('w') as log:
        subprocess.run(['latexmk','-pdf','main.tex'],cwd=paper,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(root/'publication.json',dict(complete=True,figure='4',
        artifacts={p:sha256(paper/p) for p in (*targets,'main.pdf')}))
    write_json(root/'status.json',dict(stage='published',trajectories=60,checkpoints=240,
        accounting=read_json(root/'accounting.json')))
