"""Complete the registered ALARM study with isolated workers and exact recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import tarfile
import time

import numpy as np
from cd_llm.experiments.common.artifacts import read_json, write_json, sha256
from cd_llm.experiments.alarm_matched_store import Store, Checkpoint, digest
from cd_llm.experiments import alarm_matched_baselines as base
from cd_llm.experiments.alarm_matched_workers import WorkerStore, owner, merge_worker, audit_responses


def folder(root):
    result = Path(root) / "execution"
    result.mkdir(exist_ok=True)
    return result


def live_jobs():
    value = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%k"], check=True, capture_output=True, text=True)
    return dict(line.split("|", 1) for line in value.stdout.splitlines() if "|" in line)


def submit(root, stage, index=None, dependency=None):
    """Persist a unique submission intent and reconcile it against the queue."""
    import uuid
    execution = folder(root)
    with (execution / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = execution / "jobs.json"
        records = read_json(path) if path.exists() else []
        active = live_jobs()
        for record in records:
            if (record["stage"], record["index"]) != (stage, index):
                continue
            if stage == "resume" and record.get("dependency") != dependency:
                continue
            for job, comment in active.items():
                if comment == record["token"] and job != os.environ.get("SLURM_JOB_ID"):
                    if record.get("job") != job:
                        record["job"] = job
                        write_json(path, records)
                    return job
        token = "alarm-exec-" + uuid.uuid4().hex
        record = dict(stage=stage, index=index, token=token, submitted=time.time(), dependency=dependency)
        records.append(record)
        write_json(path, records)
        logs = base.ROOT / "out" / Path(root).parent.name / "alarm_matched_baselines"
        logs.mkdir(parents=True, exist_ok=True)
        cmd = ["sbatch", "--parsable", "--comment=" + token,
               f"--output={logs}/exec_{stage}_{index}_%j.out", f"--error={logs}/exec_{stage}_{index}_%j.err"]
        if dependency:
            cmd.append("--dependency=" + dependency)
        if stage == "migrate":
            cmd += ["--mem=8G", "--time=00:30:00"]
        elif stage in ("gpc", "gpc_pilot"):
            cmd += ["--mem=4G", "--time=" + ("00:30:00" if stage == "gpc_pilot" else "04:00:00")]
        elif stage in ("prepare_shards", "merge"):
            cmd += ["--time=00:30:00"]
        gpu = stage in ("baselines", "scp")
        cmd.append(str(base.ROOT / "sbatch" / ("alarm_matched_gpu.sbatch" if gpu else "alarm_matched_execution.sbatch")))
        env = dict(os.environ, ALARM_MATCHED_ROOT=str(root), ALARM_EXEC_STAGE=stage,
                   ALARM_EXEC_INDEX="" if index is None else str(index))
        for name in ("TMPDIR", "SLURM_TMPDIR", "ALARM_MATCHED_CONDITION"):
            env.pop(name, None)
        record["job"] = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        write_json(path, records)
        print(f"Submitted {stage} {index}: {record['job']}", flush=True)
        # A scheduler dependency survives OOM/termination of the Python process.
        # It is deliberately registered before the worker starts doing work.
        if stage not in ("migrate", "resume", "report"):
            watcher = dict(stage="resume", index=None, token="alarm-exec-"+uuid.uuid4().hex,
                           submitted=time.time(), dependency="afterany:"+record["job"])
            records.append(watcher)
            write_json(path, records)
            watcher_cmd = ["sbatch", "--parsable", "--comment="+watcher["token"],
                "--dependency="+watcher["dependency"], f"--output={logs}/exec_resume_%j.out",
                f"--error={logs}/exec_resume_%j.err", str(base.ROOT/"sbatch/alarm_matched_execution.sbatch")]
            watcher_env = dict(env, ALARM_EXEC_STAGE="resume", ALARM_EXEC_INDEX="")
            watcher["job"] = subprocess.run(watcher_cmd, env=watcher_env, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
            write_json(path, records)
        return record["job"]


def current(root, stage):
    path = folder(root) / "jobs.json"
    active = live_jobs()
    return [r for r in (read_json(path) if path.exists() else [])
            if r["stage"] == stage and any(c == r["token"] for c in active.values())]


def reconcile_failures(root):
    execution = folder(root)
    path = execution / "jobs.json"
    with (execution / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = read_json(path) if path.exists() else []
        active = live_jobs()
        ended = [r for r in records if r.get("job") and r["job"] not in active and not r.get("final_state")]
        if not ended:
            return
        result = subprocess.run(["sacct", "-X", "-j", ",".join(r["job"] for r in ended),
                                 "--noheader", "--parsable2", "--format=JobID,State"],
                                check=True, capture_output=True, text=True)
        states = dict(line.split("|", 1) for line in result.stdout.splitlines() if "|" in line)
        for r in ended:
            state = states.get(r["job"], "").split()[0] if states.get(r["job"]) else ""
            if state in ("COMPLETED", "OUT_OF_MEMORY", "TIMEOUT", "FAILED", "CANCELLED", "NODE_FAIL", "PREEMPTED"):
                r["final_state"] = state
                if state != "COMPLETED":
                    failure = execution/f"failure_{r['stage']}_{r['index']}.json"
                    if not failure.exists():
                        write_json(failure, dict(stage=r["stage"], index=r["index"], job=r["job"],
                            error=state, time=time.time(), automatic_retry=False))
        write_json(path, records)


def migrate(root, deadline):
    amendment = root / "amendments" / "execution_parallel"
    if (amendment / "complete.json").exists():
        base.verify(root)
        return
    original = read_json(amendment / "manifest_before.json")
    with tarfile.open(amendment / "implementation_before.tar.gz") as archive:
        hashes = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive if m.isfile()}
    if hashes != original["implementation"]:
        raise ValueError("original implementation archive mismatch")
    for name, expected in original["inputs"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    allowed = {
        "code/src/cd_llm/experiments/alarm_matched_gpc.py",
        "code/src/cd_llm/experiments/alarm_matched_store.py",
        "code/src/cd_llm/experiments/alarm_matched_baselines.py",
        "code/src/cd_llm/experiments/alarm_matched_recovery.py",
        "sbatch/alarm_matched_gpu.sbatch",
    }
    changes = {}
    for name, expected in original["implementation"].items():
        actual = sha256(base.ROOT / name)
        if actual != expected:
            if name not in allowed:
                raise ValueError(f"unreviewed implementation change: {name}")
            changes[name] = dict(before=expected, after=actual)
    # No old collector may write during the amendment or spawn legacy retries.
    store = Store(root)
    active = live_jobs()
    for token in store.items("dispatch"):
        if any(c == token for c in active.values()):
            raise ValueError("legacy experiment job is still active")
    backup = amendment / "cache_before.sqlite3"
    if not backup.exists():
        temporary = backup.with_suffix(".partial.sqlite3")
        with sqlite3.connect(f"file:{store.path}?mode=ro", uri=True) as source, sqlite3.connect(temporary) as target:
            def check(*_):
                if time.time() >= deadline:
                    raise Checkpoint("database backup deadline")
            source.backup(target, pages=4096, progress=check)
            if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("backup integrity check failed")
        temporary.replace(backup)
    from cd_llm.experiments.alarm_matched_disk import DiskCheckpoint
    missing = [i for i in range(20) if not store.terminal("prediction", f"gpc:{i}")]
    for i in missing:
        if time.time() >= deadline:
            raise Checkpoint("gPC migration deadline")
        with DiskCheckpoint(root, i) as checkpoint:
            if checkpoint.load() is None:
                state = store.get("gpc_progress", str(i))
                if state is None:
                    raise ValueError(f"missing original gPC checkpoint {i}")
                checkpoint.import_state(state)
                loaded = checkpoint.load()
                if any(loaded[k] != v for k, v in state.items() if k != "swaps"):
                    raise ValueError("gPC scalar checkpoint migration differs")
                if len(checkpoint.mapping) != len(state["swaps"]) or any(checkpoint.mapping[k] != v for k, v in state["swaps"].items()):
                    raise ValueError("gPC shuffle migration differs")
                del state, loaded
        print(f"Migrated gPC condition {i}", flush=True)
    additions = ["code/src/cd_llm/experiments/alarm_matched_" + n + ".py" for n in ("disk", "workers", "execution")]
    additions += ["sbatch/alarm_matched_execution.sbatch", "code/tests/unit/experiments/test_alarm_matched_execution.py"]
    original["implementation"] = {name: sha256(base.ROOT / name) for name in [*original["implementation"], *additions]}
    record = dict(implementation_changes=changes, missing_gpc=missing, gpu_workers=4,
                  backup_sha256=sha256(backup), scientific_settings_changed=False, time=time.time())
    original.setdefault("amendments", {})["execution_parallel"] = record
    write_json(root / "manifest.json", original)
    write_json(folder(root) / "config.json", dict(workers=4, checkpoint_interval_seconds=300))
    base.verify(root)
    write_json(amendment / "complete.json", record)


def baselines(root, deadline):
    store = Store(root)
    base.order_expert(root, store, False, deadline)
    names, descriptions, context = base.metadata(root)
    for method, fit in (("bfs", base.methods.bfs_fit), ("long", base.methods.long_fit)):
        for i in range(20):
            if store.terminal("collected", f"{method}:{i}"):
                continue
            if time.time() >= deadline:
                raise Checkpoint("baseline collection deadline")
            try:
                t = base.transport(root, store, f"{method}:{i}", True, deadline)
                value = fit(t, base.data(root, i) if method == "bfs" else base.pc_partial(root, i), names, descriptions, context)
                record = dict(status="complete", **value)
            except Checkpoint:
                raise
            except Exception as error:
                from cd_llm.experiments.alarm_matched_recovery import scientific_failure
                if not scientific_failure(method, error):
                    raise
                record = dict(status="infeasible", scientific_infeasibility=True, reason=repr(error))
            store.put("collected", f"{method}:{i}", record)
            write_json(folder(root) / "baseline_progress.json", dict(method=method, condition=i, updated=time.time()))


def prepare_shards(root):
    execution = folder(root)
    if (execution / "shards_ready.json").exists():
        return
    store, manifest = Store(root), base.verify(root)
    audit_responses(store, manifest["runtime_fingerprint"], manifest["settings"])
    if not all(store.terminal("collected", f"{method}:{i}") for method in ("bfs", "long") for i in range(20)):
        raise ValueError("BFS/Long collection incomplete")
    order = store.get("order_expert", "shared")
    if order is None or not store.get("audit", "order_cycle_removal")["acyclic"]:
        raise ValueError("validated shared order missing")
    for method in ("bfs", "long", "order"):
        for i in range(20):
            store.put("method_audit", f"{method}:{i}", dict(passed=True, time=time.time()))
    # Copy only experiment records, never the large legacy shuffle JSON blobs.
    snapshot_folder = execution / "base"
    snapshot_folder.mkdir(exist_ok=True)
    snapshot = Store(snapshot_folder)
    with snapshot.db() as db:
        db.execute("ATTACH DATABASE ? AS original", (f"file:{store.path}?mode=ro",))
        for table in ("responses", "uses"):
            db.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM original.{table}")
        db.execute("INSERT OR IGNORE INTO items SELECT * FROM original.items WHERE kind IN "
                   "('scp','bootstrap','bootstrap_ready','collected','order_expert','order_response')")
    names, descriptions, context = base.metadata(root)
    tasks = [[] for _ in range(4)]
    existing = 0
    for i in range(20):
        bootstrap = store.get("bootstrap", str(i))
        if digest(bootstrap) != store.get("bootstrap_ready", str(i))["sha256"]:
            raise ValueError("bootstrap hash mismatch")
        stats = base.methods.bootstrap_probabilities(bootstrap["matrices"], names)
        with store.db() as db:
            completed = {r[0] for r in db.execute("SELECT key FROM items WHERE kind='scp' AND key LIKE ?", (f"{i}:%",))}
        for a in range(37):
            for b in range(37):
                if a == b:
                    continue
                if f"{i}:{a}:{b}" in completed:
                    existing += 1
                    continue
                prompt = base.methods.scp_prompt(names, descriptions, context, base.CONDITIONS[i][1], stats, a, b)
                shard = owner(prompt, manifest["runtime_fingerprint"])
                tasks[shard].append([i, a, b, digest(prompt)])
    if existing + sum(map(len, tasks)) != 26640:
        raise ValueError("SCP ownership coverage mismatch")
    for shard, rows in enumerate(tasks):
        write_json(execution / f"tasks_{shard}.json", rows)
    write_json(execution / "shards_ready.json", dict(existing=existing, counts=list(map(len, tasks)),
        base_sha256=sha256(snapshot.path), tasks_sha256=[sha256(execution/f"tasks_{i}.json") for i in range(4)]))


def scp(root, shard, deadline):
    execution = folder(root)
    manifest = base.verify(root)
    ready = read_json(execution / "shards_ready.json")
    task_path = execution / f"tasks_{shard}.json"
    if sha256(task_path) != ready["tasks_sha256"][shard]:
        raise ValueError("SCP worklist changed")
    if sha256(execution / "base/cache.sqlite3") != ready["base_sha256"]:
        raise ValueError("frozen response snapshot changed")
    tasks = read_json(task_path)
    worker = WorkerStore(execution / f"worker_{shard}", execution / "base/cache.sqlite3")
    names, descriptions, context = base.metadata(root)
    statistics = {i: base.methods.bootstrap_probabilities(worker.get("bootstrap", str(i))["matrices"], names)
                  for i in {r[0] for r in tasks}}
    with worker.db() as db:
        completed = {r[0] for r in db.execute("SELECT key FROM items WHERE kind='scp'")}
    pending = [r for r in tasks if f"{r[0]}:{r[1]}:{r[2]}" not in completed]
    started = time.time()
    initial = len(tasks) - len(pending)
    gate = execution / f"pilot_scp_{shard}.json"

    def run(row):
        i, a, b, expected = row
        prompt = base.methods.scp_prompt(names, descriptions, context, base.CONDITIONS[i][1], statistics[i], a, b)
        if digest(prompt) != expected or owner(prompt, manifest["runtime_fingerprint"]) != shard:
            raise ValueError("SCP request family ownership changed")
        base.pair_job(root, worker, i, a, b, True, deadline, statistics[i])

    with ThreadPoolExecutor(4) as pool:
        for first in range(0, len(pending), 16):
            if time.time() >= deadline:
                raise Checkpoint("SCP worker deadline")
            batch = pending[first:first+16]
            list(pool.map(run, batch))
            done = initial + first + len(batch)
            elapsed = time.time() - started
            write_json(execution / f"progress_scp_{shard}.json", dict(completed=done, total=len(tasks),
                pairs_per_second=(done-initial)/max(elapsed,1), updated=time.time(), job=os.environ.get("SLURM_JOB_ID")))
            if not gate.exists() and done >= min(64, len(tasks)):
                audit_responses(worker, manifest["runtime_fingerprint"], manifest["settings"])
                from cd_llm.experiments.alarm_claim_pools import offline_guard
                with offline_guard():
                    for row in tasks[:min(64, len(tasks))]:
                        i, a, b, _ = row
                        result = worker.get("scp", f"{i}:{a}:{b}")
                        if result is None or len(result["trials"]) != 5:
                            raise ValueError("SCP pilot lacks five probability trials")
                        base.pair_job(root, worker, i, a, b, False, deadline, statistics[i])
                write_json(gate, dict(passed=True, pairs=min(64, len(tasks)), time=time.time()))
    with worker.db() as db:
        count = db.execute("SELECT COUNT(*) FROM items WHERE kind='scp'").fetchone()[0]
    if count != len(tasks):
        raise ValueError("SCP shard incomplete")
    audit_responses(worker, manifest["runtime_fingerprint"], manifest["settings"])
    write_json(execution / f"done_scp_{shard}.json", dict(passed=True, pairs=len(tasks), time=time.time()))


def gpc(root, i, deadline, pilot=False):
    from cd_llm.experiments.alarm_matched_disk import DiskCheckpoint
    from cd_llm.experiments.alarm_matched_gpc import search
    store = Store(root)
    start = time.time()
    tests_before = 0
    progress = None
    try:
        with DiskCheckpoint(root, i) as checkpoint:
            state = checkpoint.load()
            tests_before = state["tests"]
            guess = np.array(store.get("gpc_guess", "shared")["skeleton"])
            rng = np.random.default_rng(int(digest(["alarm-gpc", base.CONDITIONS[i][0]])[:8], 16))
            try:
                from cd_llm.experiments.alarm_claim_pools import offline_guard
                with offline_guard():
                    matrix = search(base.data(root, i), guess, rng, store, str(i), deadline, checkpoint=checkpoint)
                store.put("prediction", f"gpc:{i}", dict(status="complete", matrix=matrix.tolist(), seconds=time.time()-start))
            finally:
                progress = checkpoint.load()
    finally:
        if pilot and progress is not None and (progress["tests"] > tests_before or progress["edge"] == 666):
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if peak >= 3 * 1024 * 1024:
                raise RuntimeError("gPC pilot exceeded the 3 GB memory acceptance gate")
            write_json(folder(root) / "gpc_pilot.json", dict(passed=True, peak_rss_kib=peak,
                additional_tests=progress["tests"]-tests_before, seconds=time.time()-start))


def early_fit(root, i, deadline):
    for method in ("bfs", "long", "order"):
        base.fit(root, i, deadline, only=method)


def merge(root):
    execution, store = folder(root), Store(root)
    if current(root, "scp"):
        raise ValueError("cannot merge an active worker")
    for i in range(4):
        if not (execution / f"done_scp_{i}.json").exists():
            raise ValueError("SCP worker unfinished")
        merge_worker(store, execution / f"worker_{i}/cache.sqlite3")
    base.audit(root)
    write_json(execution / "merged.json", dict(passed=True, time=time.time()))


def resume(root):
    base.verify(root)
    execution, store = folder(root), Store(root)
    with (execution / "resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reconcile_failures(root)
        failures = list(execution.glob("failure_*.json"))
        blocked = {(r["stage"], r["index"]) for p in failures for r in [read_json(p)]}
        if not (execution / "gpc_pilot.json").exists():
            if ("gpc_pilot", 1) not in blocked:
                submit(root, "gpc_pilot", 1)
        else:
            active = current(root, "gpc")
            active_indices = {r["index"] for r in active}
            for i in range(20):
                if len(active_indices) >= 4:
                    break
                if i not in active_indices and not store.terminal("prediction", f"gpc:{i}") and ("gpc", i) not in blocked:
                    submit(root, "gpc", i)
                    active_indices.add(i)
        if not all(store.terminal("collected", f"{m}:{i}") for m in ("bfs", "long") for i in range(20)):
            if ("baselines", None) not in blocked:
                submit(root, "baselines")
        elif not (execution / "shards_ready.json").exists():
            if ("prepare_shards", None) not in blocked:
                submit(root, "prepare_shards")
        else:
            for i in range(4):
                if not (execution/f"done_scp_{i}.json").exists() and ("scp", i) not in blocked:
                    submit(root, "scp", i)
            active = current(root, "early_fit")
            active_indices = {r["index"] for r in active}
            for i in range(20):
                if len(active_indices) >= 4:
                    break
                if i not in active_indices and any(not store.terminal("prediction", f"{m}:{i}") for m in ("bfs", "long", "order")) and ("early_fit", i) not in blocked:
                    submit(root, "early_fit", i)
                    active_indices.add(i)
            if all((execution/f"done_scp_{i}.json").exists() for i in range(4)) and not (execution/"merged.json").exists() and ("merge", None) not in blocked:
                submit(root, "merge")
        if (execution / "merged.json").exists():
            active_indices = {r["index"] for r in current(root, "fit")}
            for i in range(20):
                if len(active_indices) >= 4:
                    break
                if i not in active_indices and not store.terminal("prediction", f"scp:{i}") and ("fit", i) not in blocked:
                    submit(root, "fit", i)
                    active_indices.add(i)
            from cd_llm.experiments.alarm_matched_recovery import validated_completion
            if validated_completion(store) and ("report", None) not in blocked:
                if not (root / "completion.json").exists() or not read_json(root / "completion.json")["complete"]:
                    submit(root, "report")
        write_json(execution / "status.json", dict(updated=time.time(), failures=[str(p.name) for p in failures],
            predictions={m:sum(store.terminal("prediction", f"{m}:{i}") for i in range(20)) for m in base.METHODS},
            scp_workers=[read_json(execution/f"progress_scp_{i}.json") if (execution/f"progress_scp_{i}.json").exists() else None for i in range(4)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("migrate", "resume", "baselines", "prepare_shards", "scp", "gpc_pilot", "gpc", "early_fit", "merge", "fit", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root, stage, index = args.root.resolve(), args.stage, args.index
    deadline = float(os.environ.get("ALARM_MATCHED_DEADLINE", "inf"))
    try:
        if stage != "migrate":
            base.verify(root)
        if stage == "migrate": migrate(root, deadline)
        elif stage == "resume": resume(root)
        elif stage == "baselines": baselines(root, deadline)
        elif stage == "prepare_shards": prepare_shards(root)
        elif stage == "scp": scp(root, index, deadline)
        elif stage in ("gpc", "gpc_pilot"): gpc(root, index, deadline, pilot=stage=="gpc_pilot")
        elif stage == "early_fit": early_fit(root, index, deadline)
        elif stage == "merge": merge(root)
        elif stage == "fit": base.fit(root, index, deadline, only="scp")
        elif stage == "report":
            from cd_llm.experiments.alarm_matched_report import report
            report(root)
    except Checkpoint as error:
        write_json(folder(root)/f"checkpoint_{stage}_{index}.json", dict(reason=str(error), updated=time.time()))
        if stage == "migrate":
            submit(root, stage, index, "afterok:"+os.environ["SLURM_JOB_ID"])
            return
    except Exception as error:
        import traceback
        write_json(folder(root)/f"failure_{stage}_{index}.json", dict(stage=stage, index=index, error=repr(error), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        # The coordinator runs after the allocation ends, so it can accurately
        # count vacant worker slots and never merge a writer's open database.
        if stage not in ("resume", "report") and (root/"amendments/execution_parallel/complete.json").exists():
            submit(root, "resume", dependency="afterany:"+os.environ["SLURM_JOB_ID"])


if __name__ == "__main__":
    main()
