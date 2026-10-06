"""Isolated response ledgers, deterministic SCP ownership, and checked merging."""
import json
from pathlib import Path
import sqlite3

from cd_llm.experiments.alarm_matched_store import Store, digest, encode


class WorkerStore(Store):
    def __init__(self, root, base):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.base = Path(base)
        super().__init__(root)
        with self.db() as db:
            if db.execute("SELECT 1 FROM items WHERE kind='worker' AND key='initialized'").fetchone():
                return
            db.execute("ATTACH DATABASE ? AS frozen", (f"file:{self.base}?mode=ro",))
            db.execute("INSERT OR IGNORE INTO uses SELECT * FROM frozen.uses")
            db.execute("INSERT INTO items VALUES('worker','initialized','true')")

    def get(self, kind, key):
        value = super().get(kind, key)
        if value is not None:
            return value
        with sqlite3.connect(f"file:{self.base}?mode=ro", uri=True) as db:
            row = db.execute("SELECT value FROM items WHERE kind=? AND key=?", (kind, str(key))).fetchone()
        return json.loads(row[0]) if row else None

    def cached_response(self, key):
        row = super().cached_response(key)
        if row is not None:
            return row
        with sqlite3.connect(f"file:{self.base}?mode=ro", uri=True) as db:
            record = db.execute("SELECT * FROM responses WHERE key=?", (key,)).fetchone()
        if record is None:
            return None
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO responses VALUES(?,?,?,?,?)", record)
        return (record[2],)


def owner(prompt, runtime, shards=4):
    # Every trial and repair includes this complete knowledge prompt. Equal
    # request families share a worker even when used by several datasets.
    return int(digest([runtime, prompt]), 16) % shards


def merge_worker(target, worker_path):
    """One transaction, repeatable without duplicate attempts or logical uses."""
    with target.db() as db:
        db.execute("ATTACH DATABASE ? AS worker", (f"file:{worker_path}?mode=ro",))
        for table, keys, columns in (
            ("responses", "m.key=w.key", "m.request<>w.request OR m.response<>w.response OR m.input<>w.input OR m.output<>w.output"),
            ("uses", "m.unit=w.unit AND m.ordinal=w.ordinal", "m.key<>w.key"),
        ):
            if db.execute(f"SELECT 1 FROM {table} m JOIN worker.{table} w ON {keys} WHERE {columns} LIMIT 1").fetchone():
                raise ValueError(f"conflicting worker {table}")
            db.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM worker.{table}")
        if db.execute("SELECT 1 FROM items m JOIN worker.items w ON m.kind=w.kind AND m.key=w.key "
                      "WHERE w.kind='scp' AND m.value<>w.value LIMIT 1").fetchone():
            raise ValueError("conflicting SCP record")
        db.execute("INSERT OR IGNORE INTO items SELECT * FROM worker.items WHERE kind='scp'")
        marker = str(Path(worker_path).resolve())
        old = db.execute("SELECT value FROM items WHERE kind='merged_attempts' AND key=?", (marker,)).fetchone()
        last = int(json.loads(old[0])) if old else 0
        maximum = db.execute("SELECT COALESCE(MAX(id),0) FROM worker.attempts").fetchone()[0]
        db.execute("INSERT INTO attempts(key,started,finished,state,error,job) "
                   "SELECT key,started,finished,state,error,job FROM worker.attempts WHERE id>?", (last,))
        db.execute("INSERT OR REPLACE INTO items VALUES('merged_attempts',?,?)", (marker, encode(maximum)))


def audit_responses(store, runtime, settings):
    from cd_llm.experiments.alarm_matched_store import validate_response, validate_request_temperature
    with store.db() as db:
        for key, request, response in db.execute("SELECT key,request,response FROM responses"):
            request, response = json.loads(request), json.loads(response)
            if digest(dict(request=request, runtime=runtime)) != key:
                raise ValueError("response cache identity mismatch")
            validate_response(response)
            if request.get("chat_template_kwargs") != {"enable_thinking": False}:
                raise ValueError("reasoning settings changed")
        for unit, request in db.execute("SELECT u.unit,r.request FROM uses u JOIN responses r ON u.key=r.key "
                                        "WHERE u.unit NOT LIKE 'archived:%'"):
            validate_request_temperature(json.loads(request), unit, settings)
