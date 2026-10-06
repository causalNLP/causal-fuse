"""Compact, restartable local-response ledger for the matched ALARM study."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
import urllib.request

MODEL = "Qwen/Qwen3.5-9B"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class Checkpoint(Exception):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "cache.sqlite3"
        self.locks = [threading.Lock() for _ in range(64)]
        with self.db() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY, request TEXT NOT NULL,
                response TEXT NOT NULL, input INTEGER, output INTEGER);
              CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, key TEXT, started REAL,
                finished REAL, state TEXT, error TEXT, job TEXT);
              CREATE TABLE IF NOT EXISTS uses(unit TEXT, ordinal INTEGER, key TEXT,
                PRIMARY KEY(unit, ordinal));
              CREATE TABLE IF NOT EXISTS items(kind TEXT, key TEXT, value TEXT,
                PRIMARY KEY(kind,key));
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=120)
        try:
            db.execute("PRAGMA busy_timeout=120000")
            db.execute("PRAGMA journal_mode=PERSIST")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def get(self, kind, key):
        with self.db() as db:
            row = db.execute("SELECT value FROM items WHERE kind=? AND key=?", (kind,str(key))).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO items VALUES(?,?,?)", (kind,str(key),encode(value)))

    def items(self, kind):
        with self.db() as db:
            return {k:json.loads(v) for k,v in db.execute("SELECT key,value FROM items WHERE kind=?", (kind,))}

    def terminal(self, kind, key):
        row = self.get(kind, key)
        return bool(row and (row.get("status") == "complete" or
                    (row.get("status") == "infeasible" and row.get("scientific_infeasibility"))))

    def cached_response(self, key):
        with self.db() as db:
            return db.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()

    def usage(self, prefix, *, exact=False):
        with self.db() as db:
            row = db.execute("SELECT COUNT(*),COALESCE(SUM(r.input),0),COALESCE(SUM(r.output),0) "
                "FROM uses u JOIN responses r ON u.key=r.key WHERE u.unit "+("= ?" if exact else "LIKE ?"), (prefix if exact else prefix+"%",)).fetchone()
        return dict(logical_requests=row[0],input_tokens=row[1],output_tokens=row[2],total_tokens=row[1]+row[2])


def canonical_request(request):
    req = dict(request)
    extra = req.pop("extra_body", {}) or {}
    req.update(extra)
    if req.get("model", MODEL) != MODEL:
        raise ValueError("wrong model")
    req.update(model=MODEL, chat_template_kwargs={"enable_thinking":False})
    req.setdefault("temperature", 0)
    req.setdefault("seed", 0)
    req.pop("reasoning", None)
    req.pop("provider", None)
    if "max_completion_tokens" in req:
        req["max_tokens"] = req.pop("max_completion_tokens")
    return req


def validate_request_temperature(request, unit, settings):
    expected = settings["scp"].get("temperature", settings["temperature"]) if unit.startswith("scp:") else settings["temperature"]
    if request.get("temperature") != expected:
        raise ValueError(f"generation temperature mismatch for {unit}: expected {expected}")


def validate_response(response):
    if response.get("model") != MODEL:
        raise ValueError("response model mismatch")
    usage = response["usage"]
    for k in ("prompt_tokens", "completion_tokens"):
        if not isinstance(usage.get(k), int) or usage[k] < 0:
            raise ValueError("missing token accounting")
    if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens",0):
        raise ValueError("unexpected reasoning tokens")
    choice = response["choices"][0]
    if choice["message"].get("reasoning_content") or choice["message"].get("reasoning"):
        raise ValueError("unexpected reasoning content")
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("invalid completion finish")


class Transport:
    def __init__(self, store, unit, runtime, *, online=False, deadline=float("inf"), sender=None):
        self.store, self.root, self.unit, self.runtime = store, store.root, unit, runtime
        self.online, self.deadline, self.sender = online, deadline, sender
        self.ordinal = 0

    def query(self, request):
        request = canonical_request(request)
        key = digest(dict(request=request, runtime=self.runtime))
        with self.store.locks[int(key[:2],16)%64]:
            row = self.store.cached_response(key)
            if row:
                response = json.loads(row[0])
            else:
                if not self.online:
                    raise RuntimeError(f"offline cache miss: {self.unit} {key}")
                response = self.send(request, key)
            with self.store.db() as db:
                old = db.execute("SELECT key FROM uses WHERE unit=? AND ordinal=?", (self.unit,self.ordinal)).fetchone()
                if old and old[0] != key:
                    raise ValueError("non-deterministic logical request sequence")
                db.execute("INSERT OR IGNORE INTO uses VALUES(?,?,?)", (self.unit,self.ordinal,key))
            self.ordinal += 1
        return response, key

    def send(self, request, key):
        for retry in range(3):
            if time.time() >= self.deadline:
                raise Checkpoint("collection deadline; cached responses are resumable")
            with self.store.db() as db:
                attempt = db.execute("INSERT INTO attempts(key,started,state,job) VALUES(?,?,?,?)",
                    (key,time.time(),"dispatched",os.environ.get("SLURM_JOB_ID"))).lastrowid
            try:
                if self.sender:
                    response = self.sender(request)
                else:
                    base = os.environ["ALARM_MATCHED_BASE_URL"]
                    if not base.startswith("http://127.0.0.1:") or not base.endswith("/v1"):
                        raise ValueError("only the registered local endpoint is allowed")
                    req = urllib.request.Request(base+"/chat/completions", data=encode(request).encode(),
                        headers={"Content-Type":"application/json"})
                    with urllib.request.urlopen(req, timeout=300) as stream:
                        response = json.load(stream)
                validate_response(response)
                with self.store.db() as db:
                    db.execute("INSERT INTO responses VALUES(?,?,?,?,?)", (key,encode(request),encode(response),
                        response["usage"]["prompt_tokens"],response["usage"]["completion_tokens"]))
                    db.execute("UPDATE attempts SET finished=?,state='complete' WHERE id=?", (time.time(),attempt))
                return response
            except Exception as error:
                with self.store.db() as db:
                    db.execute("UPDATE attempts SET finished=?,state='failed',error=? WHERE id=?",
                        (time.time(),repr(error),attempt))
                if getattr(error,"code",None) not in (429,500,502,503,504) or retry == 2:
                    raise
                time.sleep(5*(retry+1))


def client(transport):
    from openai.types.chat import ChatCompletion
    def create(**request):
        response, _ = transport.query(request)
        return ChatCompletion.model_validate(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
