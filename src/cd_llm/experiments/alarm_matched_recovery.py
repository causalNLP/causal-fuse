"""Verified recovery and idempotent scheduling of the matched ALARM study."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile
import time
import uuid

from cd_llm.experiments.common.artifacts import read_json, write_json, sha256
from cd_llm.experiments.alarm_matched_store import Store, digest, encode, validate_response


@contextmanager
def scheduling_lock(root):
    with (Path(root) / "scheduling.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def array_indices(array):
    result = []
    for part in array.split("%", 1)[0].split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            result.extend(range(a, b + 1))
        else:
            result.append(int(part))
    return result


def queue():
    output = subprocess.run(["squeue", "--me", "-r", "-h", "-o", "%i|%k"],
                            check=True, capture_output=True, text=True).stdout
    return [tuple(line.strip().split("|", 1)) for line in output.splitlines() if "|" in line]


def submit_unique(root, stage, array, dependency, condition, launch):
    """Persist intent before submission; recover accepted jobs by unique comment."""
    root = Path(root)
    with scheduling_lock(root):
        store = Store(root)
        running = queue()  # A queue-query failure must never trigger blind submission.
        ignored = {dependency.split(":", 1)[1]} if dependency and dependency.startswith(("afterok:", "afterany:")) else set()
        if os.environ.get("SLURM_JOB_ID") in ignored and os.environ.get("SLURM_ARRAY_JOB_ID"):
            ignored.add(os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"])
        wanted = array_indices(array) if array else [condition]
        active, existing = set(), []
        for token, record in store.items("dispatch").items():
            if record["stage"] != stage:
                continue
            matches = [job for job, comment in running if comment == token and job not in ignored]
            if not matches:
                continue
            existing.extend(matches)
            for job in matches:
                if "_" in job:
                    active.add(int(job.split("_", 1)[1]))
                else:
                    active.update(record["conditions"])
            if not record.get("job"):
                record.update(job=matches[0].split("_", 1)[0], reconciled=True)
                store.put("dispatch", token, record)
        remaining = [i for i in wanted if i not in active]
        if not remaining:
            return existing[0].split("_", 1)[0]
        if array:
            array = ",".join(map(str, remaining)) + "%4"
        token = "alarm-" + uuid.uuid4().hex
        record = dict(stage=stage, conditions=remaining, dependency=dependency,
                      created=time.time(), state="intent")
        store.put("dispatch", token, record)
        job = launch(array, token)
        record.update(job=job, state="submitted")
        store.put("dispatch", token, record)
        return job


def validated_completion(store):
    from cd_llm.experiments.alarm_matched_baselines import METHODS
    return all(store.terminal("prediction", f"{method}:{i}")
               for i in range(20) for method in METHODS)


def scientific_failure(method, error):
    """Only known native-method limitations count as scientific outcomes."""
    from cd_llm.baselines.causal_order2025 import CausalOrderInfeasibleError
    from cd_llm.baselines.long2023.algorithm import CompletionEnumerationLimitError
    if method == "order" and isinstance(error, CausalOrderInfeasibleError):
        return True
    if method == "long" and isinstance(error, CompletionEnumerationLimitError):
        return True
    if not isinstance(error, ValueError):
        return False
    message = str(error)
    if method == "long":
        return message in {"invalid native Long A/B answer", "missing comparable A/B log-probabilities",
                           "CPDAG has no compatible DAG extension", "expert posterior has zero or invalid normalizer"}
    if method == "bfs":
        return message == "response has no <Answer>...</Answer> block" or message.startswith("response contains unknown variables:")
    return False


def recover(source, root):
    from cd_llm.experiments.alarm_matched_baselines import ROOT, verify
    source, root = Path(source).resolve(), Path(root).resolve()
    assert root != source
    if (root / "recovery.json").exists():
        verify(root)
        return
    original = read_json(source / "manifest.json")
    preparation = read_json(root / "recovery_preparation.json")
    assert preparation["source_manifest_sha256"] == sha256(source / "manifest.json")
    for name, expected in original["inputs"].items():
        assert sha256(source / name) == expected, name
    archived = {}
    with tarfile.open(root / "original_implementation.tar.gz") as bundle:
        for member in bundle:
            assert member.isfile()
            archived[member.name] = hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
    assert archived == original["implementation"]
    allowed = {"code/src/cd_llm/methods/active_query.py",
               "code/tests/unit/experiments/test_alarm_matched_baselines.py",
               "sbatch/alarm_matched_cpu.sbatch", "sbatch/alarm_matched_gpu.sbatch"}
    allowed.update(k for k in original["implementation"] if "/alarm_matched_" in k)
    changes = {}
    for name, expected in original["implementation"].items():
        actual = sha256(ROOT / name)
        if expected != actual:
            assert name in allowed, f"Unreviewed implementation change: {name}"
            changes[name] = dict(before=expected, after=actual)
    for name in original["inputs"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, root / name)
    (root / "status").mkdir(exist_ok=True)
    shutil.copyfile(source / "manifest.json", root / "original_manifest.json")
    with sqlite3.connect(f"file:{source / 'cache.sqlite3'}?mode=ro", uri=True) as old:
        assert old.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        with sqlite3.connect(root / "cache.sqlite3") as new:
            old.backup(new)
    store = Store(root)
    with store.db() as db:
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        for key, request, response in db.execute("SELECT key,request,response FROM responses"):
            assert key == digest(dict(request=json.loads(request), runtime=original["runtime_fingerprint"]))
            validate_response(json.loads(response))
        unresolved = [dict(id=i, key=k, cached=bool(c)) for i,k,c in db.execute(
            "SELECT a.id,a.key,r.key IS NOT NULL FROM attempts a LEFT JOIN responses r ON a.key=r.key WHERE a.state='dispatched'")]
    for i in range(20):
        bundle = store.get("bootstrap", str(i))
        assert len(bundle["matrices"]) == len(bundle["indices_hashes"]) == 100
        assert digest(bundle) == store.get("bootstrap_ready", str(i))["sha256"]
    smc_source = Path(original["source"])
    assert sha256(smc_source / "manifest.json") == original["source_manifest_sha256"]
    for name, expected in original["smc_trajectories"].items():
        assert sha256(smc_source / name) == expected, name
    for name, expected in read_json(smc_source / "completion.json")["artifacts"].items():
        assert sha256(smc_source / name) == expected, name
    amendment = dict(source=str(source), source_manifest_sha256=sha256(source / "manifest.json"),
                     implementation_changes=changes, interrupted_attempts=unresolved,
                     unknown_historical_token_usage=bool(unresolved),
                     scientific_settings_changed=False, confidence_interval_resamples=10000,
                     note="The active_query.py difference adds the unused fixed policy for a separate synthetic experiment.")
    original["implementation"] = {k: sha256(ROOT / k) for k in original["implementation"]}
    for p in Path(__file__).parent.glob("alarm_matched_*.py"):
        original["implementation"][str(p.resolve().relative_to(ROOT.resolve()))] = sha256(p)
    original["recovery"] = amendment
    write_json(root / "manifest.json", original)
    # Preserve the original failed GPU allocation even though cleanup could not write it.
    record = subprocess.run(["sacct", "-X", "-j", "60661411", "-n", "-P", "-o", "JobID,ElapsedRaw,State"],
                            check=True, capture_output=True, text=True).stdout.strip().splitlines()
    fields = next(line.split("|") for line in record if line.startswith("60661411|"))
    store.put("gpu_runtime", "60661411", dict(seconds=int(fields[1]), state=fields[2],
              gpu="H100 40GB MIG", source="scheduler accounting", actual_api_spending_usd=0))
    verify(root)
    write_json(root / "recovery.json", amendment)
    write_json(root / "status.json", dict(stage="recovered", cached_responses=1302,
               completed_predictions=len(store.items("prediction")), interrupted_attempts=len(unresolved)))


def resume(root):
    from cd_llm.experiments.alarm_matched_baselines import verify, submit
    if (Path(root)/"amendments/execution_parallel/complete.json").exists():
        from cd_llm.experiments.alarm_matched_execution import resume as execution_resume
        return execution_resume(root)
    verify(root)
    store = Store(root)
    missing = [i for i in range(20) if not store.terminal("prediction", f"gpc:{i}")]
    if missing:
        submit(root, "gpc", array=",".join(map(str, missing)) + "%4")
    if store.get("gate", "audit"):
        missing = [i for i in range(20) if any(not store.terminal("prediction", f"{m}:{i}") for m in ("bfs","long","order","scp"))]
        if missing:
            submit(root, "fit", array=",".join(map(str, missing)) + "%4")
        elif validated_completion(store):
            submit(root, "report")
    else:
        submit(root, "collect")


SCP_TEMPERATURE_AMENDMENT = "scp_temperature_0p7"


def migrate_scp_temperature_records(store):
    """Retire only SCP-dependent state; keep raw generations and all other work."""
    name = SCP_TEMPERATURE_AMENDMENT
    prefix = "archived:" + name + ":"
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute("SELECT value FROM items WHERE kind='amendment' AND key=?", (name,)).fetchone()
        if previous:
            return json.loads(previous[0])
        rows = list(db.execute("SELECT kind,key,value FROM items WHERE kind='scp' "
            "OR (kind='prediction' AND key LIKE 'scp:%') "
            "OR (kind='gate' AND key IN ('pilot','audit'))"))
        uses = list(db.execute("SELECT unit,ordinal,key FROM uses WHERE unit LIKE 'scp:%'"))
        for unit, ordinal, key in uses:
            request = json.loads(db.execute("SELECT request FROM responses WHERE key=?", (key,)).fetchone()[0])
            if request.get("temperature") != 0:
                raise ValueError("amendment expects zero-temperature source SCP requests")
        for kind, key, value in rows:
            db.execute("INSERT INTO items VALUES(?,?,?)", (prefix + kind, key, value))
            db.execute("DELETE FROM items WHERE kind=? AND key=?", (kind, key))
        db.execute("UPDATE uses SET unit=? || unit WHERE unit LIKE 'scp:%'", (prefix,))
        record = dict(from_temperature=0, to_temperature=.7, time=time.time(),
                      archived_pairs=sum(kind == "scp" for kind, _, _ in rows),
                      archived_predictions=sum(kind == "prediction" for kind, _, _ in rows),
                      archived_logical_requests=len(uses), archive_prefix=prefix)
        db.execute("INSERT INTO items VALUES('amendment',?,?)", (name, encode(record)))
    return record


def amend_scp_temperature(root, temperature):
    """Apply a reviewed, resumable protocol amendment to the existing experiment."""
    from cd_llm.experiments.alarm_matched_baselines import ROOT, verify
    if temperature != .7:
        raise ValueError("only the reviewed SCP temperature 0.7 amendment is supported")
    root = Path(root)
    folder = root / "amendments" / SCP_TEMPERATURE_AMENDMENT
    with scheduling_lock(root):
        store = Store(root)
        if (folder / "complete.json").exists():
            m = verify(root)
            if m["settings"]["scp"]["temperature"] != temperature:
                raise ValueError("completed amendment has a different temperature")
            return read_json(folder / "complete.json")
        running = queue()
        # CPU searches may continue: their checkpoints and predictions are retained.
        for token, dispatch in store.items("dispatch").items():
            if dispatch["stage"] in ("collect", "fit", "report") and any(comment == token for _, comment in running):
                raise ValueError("stop active collection/integration/report jobs before amending SCP")
        original = read_json(folder / "manifest_before.json")
        archived = {}
        with tarfile.open(folder / "implementation_before.tar.gz") as bundle:
            for member in bundle:
                if not member.isfile():
                    raise ValueError("unexpected implementation archive member")
                archived[member.name] = hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
        if archived != original["implementation"]:
            raise ValueError("pre-amendment implementation archive mismatch")
        for name, expected in original["inputs"].items():
            if sha256(root / name) != expected:
                raise ValueError(f"frozen input changed: {name}")
        allowed = {"code/src/cd_llm/experiments/alarm_matched_" + name + ".py"
                   for name in ("baselines", "methods", "store", "recovery", "report")}
        allowed.update({"code/tests/unit/experiments/test_alarm_matched_baselines.py",
                        "sbatch/alarm_matched_cpu.sbatch"})
        changes = {}
        for name, expected in original["implementation"].items():
            actual = sha256(ROOT / name)
            if actual != expected:
                if name not in allowed:
                    raise ValueError(f"unreviewed amendment change: {name}")
                changes[name] = dict(before=expected, after=actual)
        backup = folder / "cache_before.sqlite3"
        if not backup.exists():
            temporary = folder / "cache_before.partial.sqlite3"
            with sqlite3.connect(f"file:{store.path}?mode=ro", uri=True) as source:
                with sqlite3.connect(temporary) as destination:
                    source.backup(destination)
                    if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("database backup failed integrity check")
            temporary.replace(backup)
        record = migrate_scp_temperature_records(store)
        # Archive stale derived artifacts, including a status snapshot. Raw inputs
        # and every other method's records remain in the same experiment.
        for name in ("status.json", "cache_audit.json", "completion.json", "per_seed.csv",
                     "summary.csv", "paired_ed.csv", "full_table.csv", "report.json",
                     "report.md", "accounting.json", "alarm_matched_performance_cost.pdf"):
            path = root / name
            if path.exists():
                target = folder / name
                if not target.exists():
                    path.replace(target)
        amendment = dict(**record, source_manifest_sha256=sha256(folder / "manifest_before.json"),
                         database_backup_sha256=sha256(backup), implementation_changes=changes,
                         scientific_changes={"scp.temperature": {"before": 0, "after": temperature}})
        original["settings"]["scp"]["temperature"] = temperature
        original["implementation"] = {name: sha256(ROOT / name) for name in original["implementation"]}
        original.setdefault("amendments", {})[SCP_TEMPERATURE_AMENDMENT] = amendment
        write_json(root / "manifest.json", original)
        verify(root)
        write_json(root / "status.json", dict(stage="scp_temperature_amended", scp_temperature=temperature,
                   archived_scp_pairs=record["archived_pairs"], scp_pairs=0, scp_pairs_total=26640,
                   updated=time.time(), actual_api_spending_usd=0))
        write_json(folder / "complete.json", amendment)
        return amendment


def amend_order_threshold(root):
    """Validate the cached order with a corrected cutoff, then enable resume."""
    from cd_llm.experiments.alarm_matched_baselines import ROOT, verify, order_expert
    from cd_llm.experiments.alarm_claim_pools import offline_guard
    root = Path(root)
    name = "order_validated_threshold"
    folder = root / "amendments" / name
    with scheduling_lock(root):
        if (folder / "complete.json").exists():
            verify(root)
            return read_json(folder / "complete.json")
        store = Store(root)
        running = queue()
        for token, dispatch in store.items("dispatch").items():
            if dispatch["stage"] in ("collect", "fit", "report") and any(comment == token for _, comment in running):
                raise ValueError("stop active order collection/integration before amending")
        original = read_json(folder / "manifest_before.json")
        archived = {}
        with tarfile.open(folder / "implementation_before.tar.gz") as bundle:
            for member in bundle:
                if not member.isfile():
                    raise ValueError("unexpected archived implementation member")
                archived[member.name] = hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
        if archived != original["implementation"]:
            raise ValueError("original implementation archive mismatch")
        for file, expected in original["inputs"].items():
            if sha256(root / file) != expected:
                raise ValueError(f"frozen input changed: {file}")
        allowed = {"code/src/cd_llm/baselines/causal_order2025/upstream_backend.py",
                   "code/src/cd_llm/experiments/alarm_matched_baselines.py",
                   "code/src/cd_llm/experiments/alarm_matched_recovery.py",
                   "code/src/cd_llm/experiments/alarm_matched_report.py",
                   "code/tests/unit/experiments/test_alarm_matched_baselines.py",
                   "sbatch/alarm_matched_cpu.sbatch"}
        changes = {}
        for file, expected in original["implementation"].items():
            actual = sha256(ROOT / file)
            if expected != actual:
                if file not in allowed:
                    raise ValueError(f"unreviewed implementation change: {file}")
                changes[file] = dict(before=expected, after=actual)
        backup = folder / "cache_before.sqlite3"
        if not backup.exists():
            temporary = folder / "cache_before.partial.sqlite3"
            with sqlite3.connect(f"file:{store.path}?mode=ro", uri=True) as source:
                with sqlite3.connect(temporary) as destination:
                    source.backup(destination)
                    if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("cache backup integrity check failed")
            temporary.replace(backup)
        with store.db() as db:
            if db.execute("SELECT COUNT(*) FROM items WHERE kind='prediction' AND key LIKE 'order:%'").fetchone()[0]:
                raise ValueError("unexpected existing order predictions; explicit migration required")
        original["settings"]["order"]["cycle_removal_policy"] = "validated_threshold"
        original["implementation"] = {file: sha256(ROOT / file) for file in original["implementation"]}
        test_path = "code/tests/unit/baselines/test_causal_order2025.py"
        original["implementation"][test_path] = sha256(ROOT / test_path)
        amendment = dict(source_manifest_sha256=sha256(folder / "manifest_before.json"),
                         database_backup_sha256=sha256(backup), implementation_changes=changes,
                         policy="validated_threshold", scope="ALARM only; upstream default unchanged")
        original.setdefault("amendments", {})[name] = amendment
        write_json(root / "manifest.json", original)
        verify(root)

        def accounting():
            with store.db() as db:
                return {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        for table in ("responses", "attempts", "uses")}

        before = accounting()
        with offline_guard():
            order_expert(root, store, False, float("inf"), prepare_only=True)
        if accounting() != before:
            raise ValueError("offline order replay changed generation or logical request counts")
        diagnostic = store.get("audit", "order_cycle_removal")
        if not (diagnostic["correction_applied"] and diagnostic["original_edges"] == 256
                and diagnostic["retained_edges"] == 255 and diagnostic["acyclic"]
                and diagnostic["removed_edges"] == [["V25", "V23"]]):
            raise ValueError(f"unexpected ALARM threshold correction: {diagnostic}")
        required = store.get("order_fallbacks_required", "shared")
        missing = [pair for pair in required if not store.get("order_response", f"pair:{pair[0]}:{pair[1]}")]
        report = dict(**amendment, diagnostic=diagnostic, required_fallbacks=required,
                      missing_fallbacks=missing, accounting_before=before, accounting_after=accounting(),
                      offline_replay_passed=True, time=time.time())
        write_json(folder / "complete.json", report)
        store.put("amendment", name, report)
        write_json(root / "status.json", dict(stage="order_threshold_validated", updated=time.time(),
                   corrected_order_edges=255, missing_order_fallbacks=len(missing), scp_temperature=.7))
        return report


def _order_ledger_snapshot(store):
    """Return a compact, content-addressed snapshot of causal-order model use."""
    with store.db() as db:
        rows = [list(row[:5]) + [hashlib.sha256(row[5].encode()).hexdigest()]
                for row in db.execute(
                    "SELECT u.unit,u.ordinal,u.key,r.input,r.output,r.response "
                    "FROM uses u JOIN responses r ON u.key=r.key "
                    "WHERE u.unit LIKE 'order:%' ORDER BY u.unit,u.ordinal")]
    return dict(rows=rows, sha256=digest(rows), uses=len(rows),
                input_tokens=sum(row[3] for row in rows), output_tokens=sum(row[4] for row in rows))


def amend_order_cycle_guard(root):
    """Prepare the reviewed edgewise cycle guard and restore native source records."""
    from cd_llm.experiments.alarm_matched_baselines import ROOT, verify

    root = Path(root)
    old_name = "order_compatible_extension"
    old_folder = root / "amendments" / old_name
    name = "order_cycle_guard"
    folder = root / "amendments" / name
    folder.mkdir(parents=True, exist_ok=True)
    before_manifest = read_json(old_folder / "manifest_before.json")
    before_records = read_json(old_folder / "order_records_before.json")
    failed = (3, 10, 16)
    with scheduling_lock(root):
        if (folder / "complete.json").exists():
            verify(root)
            return read_json(folder / "complete.json")
        if (folder / "pending.json").exists():
            verify(root)
            return read_json(folder / "pending.json")
        with tarfile.open(old_folder / "implementation_before.tar.gz") as archive:
            archived = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                        for member in archive if member.isfile()}
        if archived != before_manifest["implementation"]:
            raise ValueError("pre-amendment implementation archive mismatch")
        for file, expected in before_manifest["inputs"].items():
            if sha256(root / file) != expected:
                raise ValueError(f"frozen input changed: {file}")
        observed_failed = {int(key.split(":")[1]) for key, value in before_records["predictions"].items()
                           if value.get("status") != "complete"}
        if observed_failed != set(failed):
            raise ValueError(f"unexpected native Causal Order failures: {sorted(observed_failed)}")

        allowed = {
            "code/src/cd_llm/baselines/causal_order2025/api.py",
            "code/src/cd_llm/baselines/causal_order2025/integration.py",
            "code/src/cd_llm/baselines/causal_order2025/types.py",
            "code/src/cd_llm/experiments/alarm_matched_baselines.py",
            "code/src/cd_llm/experiments/alarm_matched_recovery.py",
            "code/src/cd_llm/experiments/alarm_matched_report.py",
            "code/tests/unit/baselines/test_causal_order2025.py",
            "code/tests/unit/experiments/test_alarm_matched_baselines.py",
            "sbatch/alarm_matched_cpu.sbatch",
            "sbatch/alarm_matched_gpu.sbatch",
        }
        changes = {}
        for file, expected in before_manifest["implementation"].items():
            actual = sha256(ROOT / file)
            if actual != expected:
                if file not in allowed:
                    raise ValueError(f"unreviewed cycle-guard change: {file}")
                changes[file] = dict(before=expected, after=actual)
        amendment = dict(
            source_manifest_sha256=before_records["manifest_sha256"],
            implementation_changes=changes, restored_conditions=list(failed),
            policy="edgewise_cycle_guard", edge_order="lexicographic",
            expert_model="Qwen/Qwen3.5-9B", unusable_expert_policy="unique_acyclic_direction",
            reference_free_selection=True, state="pending_expert_collection",
            abandoned_jobs=[60805248, 60806092],
        )
        manifest = before_manifest
        manifest["settings"]["order"].update(
            integration_cycle_guard=True, edge_order="lexicographic",
            cycle_fallback_expert="Qwen/Qwen3.5-9B",
            unusable_expert_policy="unique_acyclic_direction")
        manifest["implementation"] = {file: sha256(ROOT / file) for file in manifest["implementation"]}
        manifest.setdefault("amendments", {}).pop(old_name, None)
        manifest["amendments"][name] = amendment
        write_json(root / "manifest.json", manifest)
        verify(root)

        store = Store(root)
        ledger_before = _order_ledger_snapshot(store)
        with store.db() as db:
            for index in range(20):
                key = f"order:{index}"
                expected = encode(before_records["predictions"][key])
                current = db.execute("SELECT value FROM items WHERE kind='prediction' AND key=?", (key,)).fetchone()
                if index not in failed and (current is None or current[0] != expected):
                    raise ValueError(f"successful native prediction changed: {key}")
                db.execute("INSERT OR REPLACE INTO items VALUES('prediction',?,?)", (key, expected))
            db.execute("DELETE FROM items WHERE kind='archived_prediction' AND key LIKE ?", (old_name + ":%",))
            db.execute("DELETE FROM items WHERE kind='amendment' AND key=?", (old_name,))
            for index in failed:
                db.execute("DELETE FROM items WHERE kind='prediction' AND key=?", (f"order:{index}",))
        abandoned = dict(
            method="compatible_extension", scientific_result_produced=False,
            jobs=[dict(id=60805248, state="cancelled"), dict(id=60806092, state="cancelled")],
            reason="replaced by the requested edgewise expert fallback",
            retained_archive=str(old_folder.relative_to(root)), time=time.time())
        write_json(folder / "abandoned_compatible_extension.json", abandoned)
        pending = dict(**amendment, causal_order_ledger_before=ledger_before,
                       preserved_native_conditions=[i for i in range(20) if i not in failed],
                       time=time.time())
        write_json(folder / "pending.json", pending)
        store.put("amendment", name, pending)
        write_json(root / "status.json", dict(stage="order_cycle_guard_pending",
                   missing_order_conditions=list(failed), expert_model="Qwen/Qwen3.5-9B", updated=time.time()))
        return pending


def finalize_order_cycle_guard(root):
    """Audit restored Causal Order predictions after Qwen fallback collection."""
    from cd_llm.experiments.alarm_matched_baselines import verify, pc_partial
    from cd_llm.core.finite_support import Endpoint, graph_from_edges, has_edge
    import numpy as np

    root = Path(root)
    folder = root / "amendments" / "order_cycle_guard"
    pending = read_json(folder / "pending.json")
    before_records = read_json(root / "amendments/order_compatible_extension/order_records_before.json")
    store = Store(root)
    verify(root)
    predictions = store.items("prediction")
    failed = (3, 10, 16)
    for index in range(20):
        key = f"order:{index}"
        row = predictions.get(key)
        if row is None or row.get("status") != "complete":
            raise ValueError(f"cycle-guard prediction is incomplete: {key}")
        if index not in failed and row != before_records["predictions"][key]:
            raise ValueError(f"native successful prediction changed: {key}")
        matrix = np.asarray(row["matrix"], dtype=int)
        edges = [(a, b) for a in range(37) for b in range(37)
                 if matrix[a, b] == int(Endpoint.TAIL) and matrix[b, a] == int(Endpoint.ARROW)]
        graph = graph_from_edges(37, directed=edges)
        partial = pc_partial(root, index)
        for a in range(37):
            for b in range(a + 1, 37):
                graph_adjacent = has_edge(graph, a, b) or has_edge(graph, b, a)
                if graph_adjacent != (partial.mark(a, b) != Endpoint.NULL):
                    raise ValueError(f"cycle guard changed the PC skeleton: {key} {(a,b)}")
                if partial.mark(a, b) == Endpoint.TAIL and partial.mark(b, a) == Endpoint.ARROW and not has_edge(graph, a, b):
                    raise ValueError(f"cycle guard changed a directed PC edge: {key} {(a,b)}")
                if partial.mark(a, b) == Endpoint.ARROW and partial.mark(b, a) == Endpoint.TAIL and not has_edge(graph, b, a):
                    raise ValueError(f"cycle guard changed a directed PC edge: {key} {(b,a)}")
    diagnostics = {f"order:{index}": predictions[f"order:{index}"]["integration"] for index in failed}
    if any(not value.get("cycle_guard_pairs") for value in diagnostics.values()):
        raise ValueError("cycle guard did not activate in every restored condition")
    audit = store.get("audit", "order_cycle_guard")
    if not audit or not audit.get("passed"):
        raise ValueError("cycle-guard expert audit is missing")
    before = pending["causal_order_ledger_before"]
    after = _order_ledger_snapshot(store)
    before_rows = {(r[0], r[1]): r for r in before["rows"]}
    after_rows = {(r[0], r[1]): r for r in after["rows"]}
    if any(after_rows.get(key) != value for key, value in before_rows.items()):
        raise ValueError("pre-existing causal-order ledger entries changed")
    new_units = sorted(set(after_rows) - set(before_rows))
    expected_units = sorted((f"order:pair:{a}:{b}", 0) for a, b in audit["newly_collected_pairs"])
    if new_units != expected_units:
        raise ValueError(f"unexpected cycle-guard ledger additions: {new_units}")
    report = {**pending, "state": "complete", "diagnostics": diagnostics,
              "expert_audit": audit, "causal_order_ledger_after": after,
              "new_ledger_units": [list(v) for v in new_units],
              "all_order_predictions_complete": True, "time": time.time()}
    write_json(folder / "complete.json", report)
    store.put("amendment", "order_cycle_guard", report)
    manifest = read_json(root / "manifest.json")
    manifest["amendments"]["order_cycle_guard"] = {k:v for k,v in report.items()
                                                     if k not in ("causal_order_ledger_before", "causal_order_ledger_after", "diagnostics")}
    write_json(root / "manifest.json", manifest)
    verify(root)
    write_json(root / "status.json", dict(stage="order_cycle_guard_complete",
               restored_order_conditions=list(failed), order_predictions=20,
               expert_fallback_pairs=audit["fallback_pairs"], updated=time.time()))
    return report


def amend_order_cycle_guard_finalizer_fix(root):
    """Record the audit-writer serialization fix without changing scientific state."""
    from cd_llm.experiments.alarm_matched_baselines import ROOT, verify

    root = Path(root)
    target = "code/src/cd_llm/experiments/alarm_matched_recovery.py"
    folder = root / "amendments" / "order_cycle_guard"
    manifest = read_json(root / "manifest.json")
    pending = read_json(folder / "pending.json")
    mismatches = {name: (expected, sha256(ROOT / name))
                  for name, expected in manifest["implementation"].items()
                  if sha256(ROOT / name) != expected}
    if set(mismatches) != {target}:
        raise ValueError(f"unexpected implementation changes after collection: {mismatches}")
    before, after = mismatches[target]
    fix = dict(before=before, after=after,
               reason="avoid passing the pending time field twice when writing the completion audit")
    manifest["implementation"][target] = after
    manifest["amendments"]["order_cycle_guard"]["finalizer_serialization_fix"] = fix
    pending["implementation_changes"][target]["after"] = after
    pending["finalizer_serialization_fix"] = fix
    write_json(root / "manifest.json", manifest)
    write_json(folder / "pending.json", pending)
    Store(root).put("amendment", "order_cycle_guard", pending)
    verify(root)
    return fix
