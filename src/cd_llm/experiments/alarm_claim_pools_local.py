"""Local Qwen replication of the frozen ALARM claim-pool experiment."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

import numpy as np

from cd_llm.experiments import alarm_claim_pools as original
from cd_llm.experiments import alarm_claim_pools_local_transport as transport
from cd_llm.experiments import alarm_local_format_recovery as recovery
from cd_llm.experiments.alarm_claim_pools import (
    ROOT, GROUPS, BUDGETS, CAP, SEEDS, tasks, arguments, graph, validate_population,
    validate_record, require_pilot, offline_guard, cap_response, prompts, grid,
    graph_from_edges, graph_edges, active_query_smc, dag_query_pool, ClaimQuery,
    ResponseCache, sqlite_snapshot, _request_key, json_value, read_json, write_json,
    write_text, sha256, run_id, valid_outcome,
)
from cd_llm.experiments.common.reproduction_inputs import materialize, protocol
from cd_llm.paths import CODE_ROOT

STUDY = 'alarm_claim_pools_local'
MODEL = 'Qwen/Qwen3.5-9B'
SCRIPT = ROOT/'sbatch/alarm_claim_pools_local.sbatch'
GPU_SCRIPT = ROOT/'sbatch/alarm_claim_pools_local_gpu.sbatch'
RECOVERY_SOURCE = ROOT/'results/20260920T161338Z/alarm_claim_pools_local'
RECOVERY_ARCHIVE = 'recovery/20260920T164354Z_original_sources.json'


def settings():
    return dict(model=MODEL, temperature=0., seed=0, max_tokens=256,
                extra_body=dict(chat_template_kwargs=dict(enable_thinking=False)))


def verify(root):
    m = read_json(root/'manifest.json')
    if (m['study'] != STUDY or m['tasks'] != tasks() or m['settings'] != settings()
            or m['configuration'] != original.configuration()
            or m.get('parsing_policy') != recovery.POLICY):
        raise ValueError('registered local protocol differs')
    for key, base in (('inputs', root), ('implementation', CODE_ROOT)):
        for name, digest in m[key].items():
            if sha256(base/name) != digest:
                raise ValueError(f'frozen {key} changed: {name}')
    if sha256(root/'inputs/runtime.json') != m['runtime_fingerprint']:
        raise ValueError('runtime identity differs')
    return m


def prepare(root):
    if (root/'manifest.json').exists():
        return verify(root)
    root.mkdir(parents=True, exist_ok=True)
    inputs = materialize('alarm', root)
    source_manifest_hashes = set()
    for condition in original.conditions():
        population = read_json(root/f'initializations/{condition["id"]}_grid_union.json')
        validate_population(population)
        source_manifest_hashes.add(population['manifest_sha256'])
    if len(source_manifest_hashes) != 1:
        raise ValueError('packaged ALARM initializations do not share one source manifest')
    source_manifest_sha256 = source_manifest_hashes.pop()
    if read_json(root/'prompts.json') != original.templates():
        raise ValueError('packaged ALARM prompts differ from the implementation')
    runtime = protocol()['local_model']
    write_json(root/'inputs/runtime.json', runtime)
    inputs['inputs/runtime.json'] = sha256(root/'inputs/runtime.json')
    paths = [Path(__file__), Path(transport.__file__), Path(recovery.__file__), Path(original.__file__),
             Path(__file__).with_name('alarm_claim_pools_local_report.py')]
    implementation = {str(p.resolve().relative_to(CODE_ROOT.resolve())):sha256(p) for p in paths}
    write_json(root/'manifest.json', dict(study=STUDY, version=2,
        tasks=tasks(), conditions=original.conditions(), configuration=original.configuration(),
        settings=settings(), runtime=runtime, runtime_fingerprint=inputs['inputs/runtime.json'],
        backend='local-vllm', input_bundle=protocol()['bundles']['alarm'],
        source_manifest_sha256=source_manifest_sha256,
        expected_outcomes=3330, trajectories=60, checkpoints=240,
        concurrency=4, actual_api_spending_usd=0., inputs=inputs, implementation=implementation,
        shared_initialization_bytes=True, parsing_policy=recovery.POLICY))
    write_json(root/'status.json', dict(stage='prepared', expected_outcomes=3330))
    return verify(root)


def prepare_recovery(root):
    """Fork the stopped run while keeping its manifest and raw cache intact."""
    if (root/'manifest.json').exists():
        verify(root)
        validate_recovery(root)
        return
    import hashlib
    old = read_json(RECOVERY_SOURCE/'manifest.json')
    archived = read_json(RECOVERY_SOURCE/RECOVERY_ARCHIVE)
    for name,digest in old['implementation'].items():
        actual = hashlib.sha256(archived[name].encode()).hexdigest() if name in archived else sha256(ROOT/name)
        if actual != digest:
            raise ValueError(f'previous implementation provenance differs: {name}')
    if list((RECOVERY_SOURCE/'records').glob('*.json')):
        raise ValueError('recovery must precede inference')
    inputs = {}
    def copy(source,name,digest):
        if sha256(source)!=digest:
            raise ValueError(f'previous artifact changed: {source}')
        dest=root/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        if sha256(dest)!=digest:
            raise ValueError('copied artifact differs')
        inputs[name]=digest
    for name,digest in old['inputs'].items():
        copy(RECOVERY_SOURCE/name,name,digest)
    for name in ('manifest.json','responses.partial.json','accounting.json','status.json',RECOVERY_ARCHIVE):
        copy(RECOVERY_SOURCE/name,'recovery/previous_'+Path(name).name,sha256(RECOVERY_SOURCE/name))
    for path in (RECOVERY_SOURCE/'gpu_runtime').glob('*.json'):
        copy(path,'gpu_runtime/'+path.name,sha256(path))
    cache_path=root/'cache/responses.sqlite3'
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(f'file:{RECOVERY_SOURCE}/cache/responses.sqlite3?mode=ro',uri=True) as source, sqlite3.connect(cache_path) as target:
        source.backup(target)
    implementation={name:sha256(ROOT/name) for name in old['implementation']}
    implementation[str(Path(recovery.__file__).relative_to(ROOT))]=sha256(Path(recovery.__file__))
    write_json(root/'manifest.json',dict(old,version=2,inputs=inputs,implementation=implementation,
        parsing_policy=recovery.POLICY,recovery=dict(source=str(RECOVERY_SOURCE),
            previous_manifest_sha256=sha256(RECOVERY_SOURCE/'manifest.json'),
            copied_cache_sha256=sha256(cache_path),generation_and_request_keys_unchanged=True)))
    runtime_check(root)
    validate_recovery(root)
    write_json(root/'status.json',dict(stage='prepared',expected_outcomes=3330,
        recovered_cache_outcomes=read_json(root/'validation/recovery.json')['cached_outcomes']))


def validate_recovery(root):
    m=verify(root)
    raw=sqlite_snapshot(root/'cache/responses.sqlite3')
    prior={r['query']['query_id']:r for r in read_json(root/'recovery/previous_responses.partial.json')}
    async def replay():
        rows=[]
        with offline_guard():
            chat=transport.OfflineChat(raw,m)
            for query in dag_query_pool(37,GROUPS['all']):
                try:
                    row=await recovery.elicit(query,'stated',chat,prompts.load_inputs(root))
                except KeyError:
                    continue
                valid_outcome(row)
                if query.query_id in prior and row!=prior[query.query_id]:
                    raise ValueError('previously valid outcome changed')
                rows.append(row)
        return rows
    rows=asyncio.run(replay())
    if not set(prior).issubset({r['query']['query_id'] for r in rows}):
        raise ValueError('previous valid outcome lost')
    repaired=[r for r in rows if any('syntactic_recovery' in d for d in r['decisions'])]
    if not repaired:
        raise ValueError('the failing response was not recovered')
    write_json(root/'validation/recovery.json',dict(passed=True,unchanged_outcomes=len(prior),
        cached_outcomes=len(rows),recovered_outcomes=repaired,parsing_policy=recovery.POLICY,
        manifest_sha256=sha256(root/'manifest.json'),network_disabled=True))


def runtime_check(root):
    return verify(root)


async def collect(root):
    m = runtime_check(root)
    base_url = os.environ['ALARM_LOCAL_BASE_URL']
    db = ResponseCache(root/'cache/responses.sqlite3')
    deadline = time.monotonic()+max(0.,float(os.environ.get('ALARM_LOCAL_DEADLINE', time.time()+14400))-time.time())
    chat = transport.LocalChat(db, m, base_url, deadline=deadline)
    signal.signal(signal.SIGUSR1, lambda *_: setattr(chat, 'stopped', True))
    done, begun, stop = {}, time.monotonic(), asyncio.Event()
    starting = db.summary()['request_count']
    def progress(stage):
        elapsed = time.monotonic()-begun
        rate = (db.summary()['request_count']-starting)/elapsed if elapsed else 0
        write_json(root/'status.json', dict(stage=stage, completed_outcomes=len(done), expected_outcomes=3330,
            requests_per_minute=rate*60, eta_seconds=(3330-len(done))/rate if rate else None,
            accounting=chat.accounting(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        write_json(root/'responses.partial.json', list(done.values()))
    async def heartbeat():
        while not stop.is_set():
            progress('collecting')
            try:
                await asyncio.wait_for(stop.wait(), 15)
            except asyncio.TimeoutError:
                pass
    inputs = prompts.load_inputs(root)
    async def one(q):
        row = await recovery.elicit(q, 'stated', chat, inputs)
        valid_outcome(row)
        done[q.query_id] = row
        return row
    beat = asyncio.create_task(heartbeat())
    failure = None
    try:
        models = await chat.client.models.list()
        if MODEL not in [r.id for r in models.data]:
            raise ValueError('local server does not expose the frozen model')
        queries = dag_query_pool(37, GROUPS['all'])
        pilot = [q for kind in ('directed','adjacency','ancestry') for q in [q for q in queries if q.kind==kind][:2]]
        results = await asyncio.gather(*(one(q) for q in pilot), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        write_json(root/'local_pilot.json', dict(passed=True, outcomes=results,
            model=MODEL, runtime_fingerprint=m['runtime_fingerprint'], accounting=chat.accounting()))
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
                if len(done)%50==0:
                    print(f'Collected {len(done)}/3330 local outcomes', flush=True)
        await asyncio.gather(*(worker() for _ in range(4)))
        if errors:
            raise errors[0]
        write_json(root/'responses.json', [done[q.query_id] for q in queries])
    except BaseException as exc:
        failure = exc
        if not isinstance(exc, transport.CheckpointRequested):
            raise
    finally:
        stop.set()
        await beat
        write_json(root/'accounting.json', chat.accounting())
        progress('cache-filled' if failure is None else 'checkpointed' if isinstance(failure, transport.CheckpointRequested) else 'collection-failed')
        await chat.close()
        db.close()


def audit_cache(root):
    m = verify(root)
    raw = sqlite_snapshot(root/'cache/responses.sqlite3')
    async def replay():
        with offline_guard():
            chat = transport.OfflineChat(raw, m)
            return [await recovery.elicit(q,'stated',chat,prompts.load_inputs(root)) for q in dag_query_pool(37,GROUPS['all'])]
    rows = asyncio.run(replay())
    if len(rows)!=3330 or rows!=read_json(root/'responses.json'):
        raise ValueError('offline response replay differs')
    for row in rows:
        valid_outcome(row)
    if {k for r in rows for k in r['request_keys']}!=set(raw):
        raise ValueError('extra or missing cached request')
    for key,item in raw.items():
        r, req = item['record'], item['request']
        if key!=_request_key(req) or req!=transport.request_identity(req['body']['messages'],m):
            raise ValueError('cache identity differs')
        transport.validate_response(r['response'],m)
        if r['cost_usd']!=0 or any(r[k]!=r['response']['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')):
            raise ValueError('token accounting differs')
    with sqlite3.connect(root/'cache/responses.sqlite3') as db:
        db.row_factory=sqlite3.Row
        attempts=[dict(r) for r in db.execute('SELECT * FROM local_attempts ORDER BY id')]
    accounting=read_json(root/'accounting.json')
    returned=[json.loads(a['response_json']) for a in attempts if a['response_json'] is not None]
    if (len(attempts)!=accounting['attempts'] or accounting['actual_api_spending_usd']!=0
        or sum(r['usage']['prompt_tokens'] for r in returned)!=accounting['returned_prompt_tokens']
        or sum(r['usage']['completion_tokens'] for r in returned)!=accounting['returned_completion_tokens']):
        raise ValueError('attempt ledger differs')
    for key in raw:
        if not any(a['request_key']==key and a['response_json'] is not None for a in attempts):
            raise ValueError('cached response has no attempt ledger entry')
    write_json(root/'cache/raw_responses.json',raw)
    write_json(root/'cache/attempts.json',attempts)
    names=('responses.json','accounting.json','cache/raw_responses.json','cache/attempts.json','local_pilot.json')
    recovered=[r['query']['query_id'] for r in rows if any('syntactic_recovery' in d for d in r['decisions'])]
    write_json(root/'cache_audit.json',dict(passed=True,outcomes=3330,accounting=accounting,
        parsing_policy=recovery.POLICY,syntactically_recovered_queries=recovered,
        manifest_sha256=sha256(root/'manifest.json'),artifacts={n:sha256(root/n) for n in names}))


def response_for(row, m):
    # Reuse the parser-to-evidence mapping, replacing its transport provenance.
    response = prompts.response_for(row, dict(settings=dict(model=MODEL,provider=dict(only=['local-vllm']))))
    response = replace(response, audit=dict(response.audit, source='local-vllm-cache',
        syntactic_recoveries=[d.get('syntactic_recovery') for d in row['decisions']],
        model_revision=m['runtime']['model_revision'],runtime_fingerprint=m['runtime_fingerprint']))
    return cap_response(response,CAP)


def provider(root,m):
    prompts.require_cache(root)
    bank={r['query']['query_id']:r for r in read_json(root/'responses.json')}
    def answer(q):
        row=bank[q.query_id]
        if row['query']!=asdict(q):
            raise ValueError('query differs')
        return response_for(row,m)
    return answer


def launch(root,stage,dependency=None,array=None):
    logs=ROOT/'out'/root.parent.name/STUDY
    logs.mkdir(parents=True,exist_ok=True)
    wall={'prepare':'00:20:00','prepare-recovery':'00:05:00','collect':'01:00:00','audit':'00:10:00',
          'run-block':'00:20:00','validate-pilot':'00:40:00','report':'00:20:00'}[stage]
    command=['sbatch','--parsable',f'--job-name=alarm-local-{stage}',f'--time={wall}',
        f'--output={logs}/{stage}_%A_%a.out',f'--error={logs}/{stage}_%A_%a.err',
        f'--export=ALL,ALARM_LOCAL_RUN_ID={root.parent.name},ALARM_LOCAL_STAGE={stage}']
    if dependency:
        command.append(f'--dependency=afterok:{dependency}')
    if array:
        command.append(f'--array={array}')
    command.append(str(GPU_SCRIPT if stage=='collect' else SCRIPT))
    env=dict(os.environ)
    env.pop('OPENROUTER_API_KEY',None)
    # A child allocation creates its own temporary directory.
    env.pop('TMPDIR',None)
    env.pop('SLURM_TMPDIR',None)
    job=subprocess.check_output(command,text=True,env=env,timeout=60).strip().split(';')[0]
    write_json(root/f'jobs/{stage}_{job}.json',dict(job_id=job,stage=stage,dependency=dependency,array=array))
    print(f'Submitted {stage}: {job}',flush=True)
    return job


def chain(root,stage):
    current=os.environ['SLURM_JOB_ID']
    if stage in ('prepare','prepare-recovery'):
        launch(root,'collect',current)
    elif stage=='collect':
        state=read_json(root/'status.json')['stage']
        if state not in ('checkpointed','cache-filled'):
            raise ValueError('collection failed')
        launch(root,'collect' if state=='checkpointed' else 'audit',current)
    elif stage=='audit':
        pilot=launch(root,'run-block',current,'0-1%2')
        launch(root,'validate-pilot',pilot)
    elif stage=='validate-pilot':
        remaining=launch(root,'run-block',current,'2-19%4')
        launch(root,'report',remaining)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('submit','submit-recovery','prepare','prepare-recovery','runtime-check','collect','audit','run-block','validate-pilot','report'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--block',type=int,choices=range(20))
    parser.add_argument('--chain',action='store_true')
    args=parser.parse_args(argv)
    root=ROOT/'results'/run_id(args.run_id)/STUDY
    root.mkdir(parents=True,exist_ok=True)
    if args.stage in ('submit','submit-recovery'):
        launch(root,'prepare-recovery' if args.stage=='submit-recovery' else 'prepare')
        return
    if args.stage=='runtime-check':
        runtime_check(root)
        return
    if args.stage=='run-block':
        if args.block is None:
            raise ValueError('--block required')
        run_block(root,args.block)
        return
    with (root/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='prepare':
            prepare(root)
        elif args.stage=='prepare-recovery':
            prepare_recovery(root)
        elif args.stage=='collect':
            asyncio.run(collect(root))
        elif args.stage=='audit':
            audit_cache(root)
        elif args.stage=='validate-pilot':
            validate_pilot(root)
        elif args.stage=='report':
            from cd_llm.experiments.alarm_claim_pools_local_report import report
            report(root)
        if args.chain:
            chain(root,args.stage)


# Inference entry points below retain the original seeds and transition kernel.

def execute(root, task, answer, m):
    pop = root / f'initializations/{task["condition"]}_grid_union.json'
    population = read_json(pop)
    if population['manifest_sha256'] != m['source_manifest_sha256']:
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



if __name__ == '__main__':
    main()
