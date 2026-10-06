"""Prespecified ALARM claim-family study: fresh Qwen answers, offline inference."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import time

import numpy as np

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache, _request_key
from cd_llm.core.finite_support import graph_edges, graph_from_edges, stable_seed, transitive_closure
from cd_llm.experiments import alarm_bfs_openrouter as data_source
from cd_llm.experiments.common import categorical_grid as grid
from cd_llm.experiments import neuropathic_ancestry_prompt_api as api
from cd_llm.experiments import neuropathic_new_prompt_cache as prompts
from cd_llm.experiments import neuropathic_qwen38_smc as hosted
from cd_llm.experiments.common.artifacts import json_value, read_json, run_id, sha256, write_json, write_text
from cd_llm.experiments.neuropathic_claim_cache import sqlite_snapshot
from cd_llm.experiments.neuropathic_claim_group_ablation import offline_guard
from cd_llm.experiments.neuropathic_confidence_cap import cap_response
from cd_llm.methods.active_query import ClaimQuery, active_query_smc, dag_query_pool
from cd_llm.methods.graph_scores import BDeuScore, parent_masks
from cd_llm.paths import WORKSPACE_ROOT as ROOT

STUDY = 'alarm_claim_pools'
SOURCE = ROOT / 'results/20260909T215919Z/alarm_bfs_openrouter'
MODEL, PROVIDER, CAP, LIMIT = 'qwen/qwen3.5-9b', 'deepinfra/bf16', .8, 25.
SEEDS = tuple(range(1000, 1010))
BUDGETS = (10, 20, 50, 100)
GROUPS = dict(group_i=('local_edge',), group_ii=('ancestry',), all=('local_edge', 'ancestry'))
STOP = False


class CheckpointRequested(RuntimeError):
    pass


def settings():
    return dict(hosted.settings(MODEL, PROVIDER), seed=0)


def conditions():
    return [dict(panel='alarm', data_seed=s, sample_size=n, block=i,
                 data_path=f'inputs/alarm_s{s:02d}_n{n}.npy',
                 id=f'alarm_s{s:02d}_n{n}')
            for i, (s, n) in enumerate((s, n) for s in SEEDS for n in (200, 1000))]


def tasks():
    return [dict(id=f'qwen_stated_{g}_s{c["data_seed"]}_n{c["sample_size"]}_b100',
                 condition=c['id'], block=c['block'], seed=c['data_seed'],
                 sample_size=c['sample_size'], strategy='stated', group=g, budget=100)
            for c in conditions() for g in GROUPS]


def templates():
    value = prompts.prompt_templates()
    for family in ('ancestry', 'group_i'):
        value[family]['user'] = value[family]['user'].replace(
            'Respect the exact variable definitions, anatomical level, and laterality.',
            'Respect the exact variable definitions and measurement roles.')
    return value


def configuration():
    return dict(grid=list(grid.GRID), completion_limit=200, max_parents=12,
        particle_count=384, initialization_updates=192, selection_confidence=.9,
        answer_confidence_cap=CAP, engine_soft_confidence_cap=.99,
        ess_threshold_fraction=.5, rejuvenation_steps=2,
        guided_mixture_probability=.5, ordinary_kernel='uniform_pair',
        score_equivalent_sample_size=1., score_structure_prior=1.,
        budget_unit='attempted', checkpoint_mutation=False, point_refinement=False,
        primary_comparisons='mixed minus each restricted pool at 100, separately by n',
        efficiency_comparisons='mixed at 50 minus each restricted pool at 100',
        bootstrap_resamples=10000, primary_family_size=4)


def implementation_paths():
    files = [Path(__file__), Path(__file__).with_name('alarm_claim_pools_report.py'),
             ROOT / 'code/figures/alarm_claim_pool_efficiency.py',
             ROOT / 'code/tests/unit/experiments/test_alarm_claim_pools.py',
             ROOT / 'sbatch/alarm_claim_pools.sbatch']
    for module in (data_source, grid, api, prompts, hosted):
        files.append(Path(module.__file__))
    files += list((ROOT / 'code/src/cd_llm/methods').glob('*.py'))
    files += list((ROOT / 'code/src/cd_llm/core').glob('*.py'))
    files += [ROOT / 'code/src/cd_llm/experiments'/name for name in
              ('neuropathic_ancestry_prompt_ablation.py', 'neuropathic_confidence_cap.py',
               'neuropathic_claim_cache.py', 'neuropathic_claim_group_ablation.py',
               'common/artifacts.py')]
    files.append(ROOT / 'code/src/cd_llm/baselines/causal_order2025/openrouter_backend.py')
    return sorted(set(files))


def verify(root):
    m = read_json(root / 'manifest.json')
    if (m['study'] != STUDY or m['tasks'] != tasks() or m['settings'] != settings()
        or m['configuration'] != configuration() or m['cost_limit_usd'] != LIMIT):
        raise ValueError('registered study settings differ')
    for key, base in (('inputs', root), ('implementation', ROOT)):
        for name, digest in m[key].items():
            if sha256(base / name) != digest:
                raise ValueError(f'frozen {key} changed: {name}')
    return m


def prepare(root):
    if (root / 'manifest.json').exists():
        return verify(root)
    root.mkdir(parents=True, exist_ok=True)
    source = read_json(SOURCE / 'manifest.json')
    frozen = {str(Path(p).resolve()): v for p, v in source['frozen_sha256'].items()}
    for relative in ('source/alarm.bif', 'inputs/presentation.json', 'inputs/metadata.json', 'inputs/encoding.json'):
        path = SOURCE / relative
        if sha256(path) != frozen[str(path.resolve())]:
            raise ValueError(f'archived source changed: {relative}')
        dest = root / ('generation/alarm.bif' if relative.endswith('.bif') else relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    parsed = data_source.parse_bif((root / 'generation/alarm.bif').read_text())
    names, levels, parents, _, _ = parsed
    presentation = read_json(root / 'inputs/presentation.json')
    order = presentation['order']
    if len(names) != 37 or sorted(order) != list(range(37)):
        raise ValueError('invalid variable presentation')
    codes = data_source.encoding(levels)
    for seed in SEEDS:
        states = data_source.sample_states(parsed, 1000, seed)
        values = data_source.encode_states(states, names, levels, codes)[:, order]
        for n in (200, 1000):
            np.save(root / f'inputs/alarm_s{seed}_n{n}.npy', values[:n], allow_pickle=False)
    metadata = read_json(root / 'inputs/metadata.json')
    expected_ids = [presentation['mapping'][names[i]] for i in order]
    if [r['id'] for r in metadata['nodes']] != expected_ids:
        raise ValueError('metadata does not match the observed column order')
    write_json(root / 'inputs/model_metadata.json', dict(
        domain_context='clinical physiology and monitoring during anesthesia and mechanical ventilation',
        nodes=metadata['nodes']))
    inverse = {names[original]: masked for masked, original in enumerate(order)}
    write_json(root / 'evaluation/truth_graph.json', dict(node_count=37,
        graph=dict(directed=[[inverse[p], inverse[c]] for c, ps in parents.items() for p in ps],
                   bidirected=[], undirected=[])))
    write_json(root / 'prompts.json', templates())
    inputs = prompts.load_inputs(root)
    for q in dag_query_pool(37, GROUPS['all']):
        messages = prompts.messages(q, 'stated', inputs)
        text = '\n'.join(m['content'] for m in messages)
        if re.search(r'\b(?:ALARM|bnlearn|Beinlich)\b', text, re.I):
            raise ValueError('benchmark identity in prompt')
        if set(re.findall(r'\bV\d+\b', text)) - set(expected_ids):
            raise ValueError('unknown identifier in prompt')
    examples = [prompts.messages(next(q for q in dag_query_pool(37, GROUPS['all']) if q.kind == kind), 'stated', inputs)
                for kind in ('directed', 'adjacency', 'ancestry')]
    write_json(root / 'example_prompts.json', examples)
    payload, endpoint = hosted.endpoints(MODEL, PROVIDER)
    write_json(root / 'provider_endpoints.json', payload)
    # Canonical clinical codes can have gaps when a sample omits a category.
    # None makes BDeu factorize the observed levels, matching the GES scorer,
    # while retaining the canonical observations and their nested row identity.
    cs = [dict(c, cardinalities=None,
               observed_cardinalities=[len(np.unique(np.load(root / c['data_path'])[:, i])) for i in range(37)])
          for c in conditions()]
    # Inference learns observed cardinalities, exactly as the earlier categorical grid.
    files = [p for folder in ('inputs', 'evaluation', 'generation') for p in (root / folder).glob('*')]
    files += [root / name for name in ('prompts.json', 'provider_endpoints.json', 'example_prompts.json')]
    write_json(root / 'manifest.json', dict(study=STUDY, version=1, tasks=tasks(), conditions=cs,
        settings=settings(), seeds=dict(stated=0), observation_seeds=list(SEEDS), response_provider='DeepInfra',
        require_token_usage=True, pricing=endpoint['pricing'], cost_limit_usd=LIMIT, concurrency=4,
        expected_outcomes=3330, trajectories=60, checkpoints=240, configuration=configuration(),
        source_manifest_sha256=sha256(SOURCE / 'manifest.json'),
        inputs={str(p.relative_to(root)):sha256(p) for p in files},
        implementation={str(p.relative_to(ROOT)):sha256(p) for p in implementation_paths()}))
    write_json(root / 'status.json', dict(stage='prepared', expected_outcomes=3330, expected_trajectories=60))
    return verify(root)


def initialize(root, block):
    m = verify(root)
    condition = m['conditions'][block]
    path = root / f'locks/initial_{block}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock, offline_guard():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, fits = grid.candidates(root, condition, m['configuration'], sha256(root / 'manifest.json'))
        initial = grid.initialize(root, condition, m['configuration'], sha256(root / 'manifest.json'), data, fits, 'grid_union')
        validate_population(initial)
        write_json(root / f'status/initial_{block}.json', dict(complete=True,
            sha256=sha256(root / f'initializations/{condition["id"]}_grid_union.json')))


def graph(payload):
    return graph_from_edges(37, directed=payload['directed'])


def validate_population(value):
    particles = tuple(graph_from_edges(37, directed=e) for e in value['particles'])
    weights = np.asarray(value['weights'], dtype=float)
    if len(particles) != 384 or weights.shape != (384,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('invalid initial population')
    for g in particles:
        transitive_closure(g)
        if max(p.bit_count() for p in parent_masks(g)) > 12:
            raise ValueError('initial parent bound exceeded')
    return particles, weights


class Chat(prompts.APIChat):
    def __init__(self, *args, deadline=math.inf, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = deadline

    def reserve(self, request):
        if STOP or time.monotonic() >= self.deadline:
            raise CheckpointRequested('collection checkpoint reached')
        return super().reserve(request)


def valid_outcome(row):
    if row['decisions'][-1]['status'] not in ('valid', 'abstain'):
        raise ValueError('unparseable response after permitted repair; collection incomplete')


async def collect(root):
    m = verify(root)
    if not os.environ.get('OPENROUTER_API_KEY'):
        write_json(root / 'status.json', dict(stage='awaiting-credential'))
        raise RuntimeError('OPENROUTER_API_KEY is required; no paid request made')
    _, endpoint = hosted.endpoints(MODEL, PROVIDER)
    if any(float(endpoint['pricing'][k]) > float(m['pricing'][k]) for k in ('prompt', 'completion')):
        raise ValueError('provider pricing increased')
    db = ResponseCache(root / 'cache/responses.sqlite3')
    chat = Chat(db, m, deadline=time.monotonic()+float(os.environ.get('ALARM_COLLECTION_SECONDS', '13800')))
    done, begun, stop = {}, time.monotonic(), asyncio.Event()
    starting = db.summary()['request_count']
    def progress(stage):
        elapsed = time.monotonic()-begun
        rate = (db.summary()['request_count']-starting)/elapsed if elapsed else 0
        write_json(root / 'status.json', dict(stage=stage, completed_outcomes=len(done), expected_outcomes=3330,
            requests_per_minute=rate*60, eta_seconds=(3330-len(done))/rate if rate > 0 else None,
            accounting=chat.accounting(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        write_json(root / 'responses.partial.json', list(done.values()))
    async def heartbeat():
        while not stop.is_set():
            progress('collecting')
            try:
                await asyncio.wait_for(stop.wait(), 15)
            except asyncio.TimeoutError:
                pass
    async def one(q):
        row = await prompts.elicit(q, 'stated', chat, prompts.load_inputs(root))
        valid_outcome(row)
        done[q.query_id] = row
        return row
    beat = asyncio.create_task(heartbeat())
    failure = None
    try:
        queries = dag_query_pool(37, GROUPS['all'])
        pilot = [q for kind in ('directed', 'adjacency', 'ancestry') for q in [q for q in queries if q.kind == kind][:2]]
        results = await asyncio.gather(*(one(q) for q in pilot), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        write_json(root / 'api_pilot.json', dict(passed=True, outcomes=results, accounting=chat.accounting()))
        queue, errors = iter(q for q in queries if q.query_id not in done), []
        async def worker():
            for q in queue:
                if errors:
                    return
                try:
                    await one(q)
                except BaseException as exc:
                    errors.append(exc)
                    return
                if len(done) % 50 == 0:
                    print(f'Collected {len(done)}/3330; ${chat.accounting()["budget_used_usd"]:.4f}', flush=True)
        await asyncio.gather(*(worker() for _ in range(4)))
        if errors:
            raise errors[0]
        write_json(root / 'responses.json', [done[q.query_id] for q in queries])
    except BaseException as exc:
        failure = exc
        raise
    finally:
        stop.set()
        await beat
        write_json(root / 'accounting.json', chat.accounting())
        progress('cache-filled' if failure is None else 'checkpointed' if isinstance(failure, CheckpointRequested) else 'collection-failed')
        await chat.close()
        db.close()


def audit_cache(root):
    m = verify(root)
    raw = sqlite_snapshot(root / 'cache/responses.sqlite3')
    async def replay():
        with offline_guard():
            chat = api.OfflineChat(raw, m)
            return [await prompts.elicit(q, 'stated', chat, prompts.load_inputs(root)) for q in dag_query_pool(37, GROUPS['all'])]
    rows = asyncio.run(replay())
    if len(rows) != 3330 or rows != read_json(root / 'responses.json'):
        raise ValueError('response replay differs')
    for row in rows:
        valid_outcome(row)
    if {k for r in rows for k in r['request_keys']} != set(raw):
        raise ValueError('unexpected or missing cached response')
    for key, item in raw.items():
        r, req = item['record'], item['request']
        if key != _request_key(req) or req['body'] != dict(settings(), messages=req['body']['messages']):
            raise ValueError('request identity or generation settings differ')
        api.validate_response(r['response'], m)
        if any(r[k] != r['response']['usage'][k] for k in ('prompt_tokens','completion_tokens')):
            raise ValueError('token usage differs')
    with sqlite3.connect(root / 'cache/responses.sqlite3') as db:
        db.row_factory = sqlite3.Row
        attempts = [dict(r) for r in db.execute('SELECT * FROM api_attempts ORDER BY id')]
    charged = sum(r['charged_usd'] or 0 for r in attempts)
    unresolved = sum(r['reserved_usd'] for r in attempts if r['charged_usd'] is None)
    accounting = read_json(root / 'accounting.json')
    if (charged+unresolved > LIMIT or len(attempts) != accounting['attempts']
        or not math.isclose(charged, accounting['accounted_cost_usd'], abs_tol=1e-10)
        or not math.isclose(unresolved, accounting['unresolved_reservations_usd'], abs_tol=1e-10)):
        raise ValueError('spending ledger differs')
    write_json(root / 'cache/raw_responses.json', raw)
    write_json(root / 'cache/attempts.json', attempts)
    names = ('responses.json','accounting.json','cache/raw_responses.json','cache/attempts.json','api_pilot.json')
    write_json(root / 'cache_audit.json', dict(passed=True, outcomes=3330, accounting=accounting,
        manifest_sha256=sha256(root / 'manifest.json'), artifacts={n:sha256(root/n) for n in names}))


def provider(root, m):
    prompts.require_cache(root)
    bank = {r['query']['query_id']:r for r in read_json(root / 'responses.json')}
    def answer(query):
        row = bank[query.query_id]
        if row['query'] != asdict(query):
            raise ValueError('cached query differs')
        return cap_response(prompts.response_for(row, m), CAP)
    return answer


def arguments(seed, n, particles, weights):
    # Response wrapping caps both weighting and MH evidence at .8; the engine's
    # general bound must stay .99 so acquisition retains its working accuracy .9.
    return dict(budgets=BUDGETS, confidence=.9, treatment='soft_evidence', soft_confidence_cap=.99,
        policy='eig', seed=stable_seed(STUDY, 'active', seed, n), budget_unit='attempted',
        ess_threshold_fraction=.5, rejuvenation_steps=2, rejuvenate_after_every_query=False,
        guided_mixture_probability=.5, proposal_cache_size=256, ordinary_kernel='uniform_pair',
        initial_particles=particles, initial_weights=weights, direct_yes_interpretation='literal')


def execute(root, task, answer, m):
    pop = root / f'initializations/{task["condition"]}_grid_union.json'
    population = read_json(pop)
    if population['manifest_sha256'] != sha256(root / 'manifest.json'):
        raise ValueError('initial population belongs to another experiment')
    particles, weights = validate_population(population)
    data = np.load(root / f'inputs/{task["condition"]}.npy', allow_pickle=False)
    condition = next(c for c in m['conditions'] if c['id'] == task['condition'])
    score = grid.common_score(data, condition)
    result = active_query_smc(None, score, answer, query_pool=dag_query_pool(37, GROUPS[task['group']]),
        **arguments(task['seed'], task['sample_size'], particles, weights))
    initial = defaultdict(float)
    for g, w in zip(particles, weights, strict=True):
        initial[g] += float(w)
    distributions = {str(b):[dict(graph=graph_edges(g), weight=w) for g, w in sorted(d.items()) if w > 0]
                     for b, d in {0:initial, **result.distributions}.items()}
    return json_value(dict(task=task, manifest_sha256=sha256(root / 'manifest.json'),
        initialization_sha256=sha256(pop), cache_audit_sha256=sha256(root / 'cache_audit.json'),
        budget_unit=result.budget_unit, trace=[dict(asdict(s), cumulative_query_cost=i) for i,s in enumerate(result.steps,1)],
        distributions=distributions, distribution=distributions['100'],
        checkpoint_seconds=result.cumulative_seconds, elapsed_seconds=result.timing.total_seconds))


def validate_record(root, record, task, answer):
    if (record['task'] != task or record['manifest_sha256'] != sha256(root / 'manifest.json')
        or record['cache_audit_sha256'] != sha256(root / 'cache_audit.json')
        or record['initialization_sha256'] != sha256(root / f'initializations/{task["condition"]}_grid_union.json')
        or record['budget_unit'] != 'attempted'):
        raise ValueError('trajectory provenance differs')
    trace = record['trace']
    allowed = {q.query_id:asdict(q) for q in dag_query_pool(37, GROUPS[task['group']])}
    if len(trace) != 100 or len({s['query']['query_id'] for s in trace}) != 100:
        raise ValueError('100 distinct attempts required')
    for slot, step in enumerate(trace, 1):
        if allowed.get(step['query']['query_id']) != step['query']:
            raise ValueError('query outside registered pool')
        response = answer(ClaimQuery(**step['query']))
        if (step['budget'] != slot or step['cumulative_query_cost'] != slot
            or step['selection_effective_confidence'] != .9 or step['support_recovered']
            or step['valid_response'] != (response.answer is not None) or step['answer'] != response.answer
            or step['reported_confidence'] != response.reported_confidence
            or step['response_audit'] != json_value(response.audit)
            or step['prompt_tokens'] != response.prompt_tokens or step['completion_tokens'] != response.completion_tokens):
            raise ValueError('response, acquisition, or accounting differs')
        if step['valid_response']:
            if step['effective_confidence'] != min(response.reported_confidence, CAP):
                raise ValueError('incorrect effective confidence')
            if step['rejuvenation_attempted_moves'] != (768 if step['resampled'] or slot == 100 else 0):
                raise ValueError('incorrect mutation schedule')
        elif step['resampled'] or step['rejuvenation_attempted_moves'] or step['ess_before'] != step['ess_after']:
            raise ValueError('abstention changed posterior')
    if set(record['distributions']) != set(map(str, (0,*BUDGETS))) or record['distribution'] != record['distributions']['100']:
        raise ValueError('missing checkpoints')
    for entries in record['distributions'].values():
        graphs = [graph(e['graph']) for e in entries]
        if len(set(graphs)) != len(graphs) or any(e['weight'] <= 0 for e in entries) or not math.isclose(sum(e['weight'] for e in entries),1,abs_tol=1e-10):
            raise ValueError('invalid posterior weights')
        for g in graphs:
            transitive_closure(g)
            if max(p.bit_count() for p in parent_masks(g)) > 12:
                raise ValueError('posterior parent bound exceeded')


def require_pilot(root):
    gate = read_json(root / 'validation/pilot.json')
    if not gate['passed'] or gate['manifest_sha256'] != sha256(root / 'manifest.json'):
        raise ValueError('pilot gate failed')
    for name, digest in gate['records'].items():
        if sha256(root / f'records/{name}.json') != digest:
            raise ValueError('pilot record changed')


def run_block(root, block):
    m = verify(root)
    if block >= 2:
        require_pilot(root)
    lock = root / f'locks/block_{block}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle, offline_guard():
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        answer = provider(root, m)
        for task in (t for t in tasks() if t['block'] == block):
            path = root / f'records/{task["id"]}.json'
            record = read_json(path) if path.exists() else execute(root, task, answer, m)
            validate_record(root, record, task, answer)
            if not path.exists():
                write_json(path, record)
            print(f'{task["id"]}: {record["elapsed_seconds"]:.1f}s', flush=True)
        write_json(root / f'status/block_{block}.json', dict(complete=True, trajectories=3, checkpoints=12))


def validate_pilot(root):
    m, hashes = verify(root), {}
    with offline_guard():
        answer = provider(root, m)
        for task in (t for t in tasks() if t['block'] < 2):
            path = root / f'records/{task["id"]}.json'
            saved = read_json(path)
            replay = execute(root, task, answer, m)
            validate_record(root, saved, task, answer)
            validate_record(root, replay, task, answer)
            if replay['distributions'] != saved['distributions']:
                raise ValueError('pilot posterior replay differs')
            for a,b in zip(replay['trace'], saved['trace'], strict=True):
                if {k:v for k,v in a.items() if k != 'cumulative_seconds'} != {k:v for k,v in b.items() if k != 'cumulative_seconds'}:
                    raise ValueError('pilot trace replay differs')
            hashes[task['id']] = sha256(path)
    write_json(root / 'validation/pilot.json', dict(passed=True, exact_replays=6, records=hashes,
        manifest_sha256=sha256(root / 'manifest.json')))


def launch(root, stage, *, dependency=None, array=None):
    logs = ROOT / 'out' / root.parent.name / STUDY
    logs.mkdir(parents=True, exist_ok=True)
    wall = {'prepare':'00:10:00','initialize':'01:00:00','collect':'04:00:00',
            'audit':'00:10:00','run-block':'00:30:00','validate-pilot':'01:00:00',
            'report':'00:20:00','publish':'00:05:00'}[stage]
    command = ['sbatch','--parsable',f'--job-name=alarm-pools-{stage}',f'--time={wall}',
        f'--output={logs}/{stage}_%A_%a.out',f'--error={logs}/{stage}_%A_%a.err',
        f'--export=ALL,ALARM_POOLS_RUN_ID={root.parent.name},ALARM_POOLS_STAGE={stage}']
    if dependency:
        command.append(f'--dependency=afterok:{dependency}')
    if array:
        command.append(f'--array={array}')
    if stage == 'collect':
        command.append('--signal=USR1@600')
    command.append(str(ROOT / 'sbatch/alarm_claim_pools.sbatch'))
    env = dict(os.environ)
    if stage not in ('prepare','collect'):
        env.pop('OPENROUTER_API_KEY',None)
    job = subprocess.check_output(command, text=True, env=env, timeout=60).strip().split(';')[0]
    write_json(root / f'jobs/{stage}_{job}.json',dict(job_id=job,stage=stage,dependency=dependency,array=array))
    print(f'Submitted {stage}: {job}',flush=True)
    return job


def chain(root, stage):
    current = os.environ.get('SLURM_JOB_ID')
    if stage == 'prepare':
        initial = launch(root,'initialize',dependency=current,array='0-19%4')
        write_json(root / 'jobs/initialization.json', dict(job_id=initial))
        if os.environ.get('OPENROUTER_API_KEY'):
            launch(root,'collect',dependency=current)
        else:
            write_json(root / 'status.json',dict(stage='awaiting-credential',initialization_job=initial))
    elif stage == 'collect':
        launch(root,'audit',dependency=current)
    elif stage == 'audit':
        initial = read_json(root / 'jobs/initialization.json')['job_id']
        pilot = launch(root,'run-block',dependency=f'{current}:{initial}',array='0-1%2')
        launch(root,'validate-pilot',dependency=pilot)
    elif stage == 'validate-pilot':
        rest = launch(root,'run-block',dependency=current,array='2-19%4')
        launch(root,'report',dependency=rest)
    elif stage == 'report':
        launch(root,'publish',dependency=current)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('submit','prepare','initialize','submit-collect','collect','audit','run-block','validate-pilot','report','publish'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--block',type=int,choices=range(20))
    parser.add_argument('--chain',action='store_true')
    args = parser.parse_args(argv)
    root = ROOT / 'results' / run_id(args.run_id) / STUDY
    root.mkdir(parents=True,exist_ok=True)
    if args.stage in ('submit','submit-collect'):
        if args.stage == 'submit-collect' and not os.environ.get('OPENROUTER_API_KEY'):
            raise RuntimeError('Export OPENROUTER_API_KEY before submitting collection')
        launch(root,'prepare' if args.stage == 'submit' else 'collect')
        return
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGUSR1,stop)
    if args.stage in ('initialize','run-block'):
        if args.block is None:
            raise ValueError('--block required')
        (initialize if args.stage == 'initialize' else run_block)(root,args.block)
        return
    with (root / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == 'prepare':
                prepare(root)
            elif args.stage == 'collect':
                asyncio.run(collect(root))
            elif args.stage == 'audit':
                audit_cache(root)
            elif args.stage == 'validate-pilot':
                validate_pilot(root)
            elif args.stage == 'report':
                from cd_llm.experiments.alarm_claim_pools_report import report
                report(root)
            elif args.stage == 'publish':
                from cd_llm.experiments.alarm_claim_pools_report import publish
                publish(root)
        except CheckpointRequested:
            if args.chain:
                launch(root,'collect',dependency=os.environ.get('SLURM_JOB_ID'))
            return
        if args.chain:
            chain(root,args.stage)


if __name__ == '__main__':
    main()
