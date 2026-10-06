"""Pinned OpenRouter transport with a shared, durable ALARM spending ledger."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import urllib.error
import urllib.request

MODEL = "qwen/qwen3.5-9b"
PROVIDER = "parasail/bf16"
PRICES = {"prompt": 1e-7, "completion": 2.5e-7}
CAP = 10.0
ENDPOINTS = "https://openrouter.ai/api/v1/models/qwen/qwen3.5-9b/endpoints"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def effective_request(request):
    result = dict(request)
    if result.get("model") not in (MODEL, "Qwen/Qwen3.5-9B") or result.get("stream", False):
        raise ValueError("only the frozen nonstreaming Qwen model is allowed")
    result.pop("chat_template_kwargs", None)
    result.pop("extra_body", None)
    result.update(model=MODEL, reasoning={"enabled": False}, provider={
        "only": [PROVIDER], "allow_fallbacks": False, "require_parameters": True,
        "max_price": {"prompt": .10, "completion": .25}})
    return result


def reservation(request):
    # UTF-8 bytes plus chat overhead conservatively bound ordinary text tokens.
    tokens = sum(len(m["content"].encode()) + 128 for m in request["messages"]) + 128
    return tokens * PRICES["prompt"] + request.get("max_tokens", 4096) * PRICES["completion"]


def checked_response(response):
    if response.get("model") not in (MODEL, MODEL + "-20260310"):
        raise ValueError("unexpected response model")
    if response.get("provider", "").lower() != "parasail":
        raise ValueError("unexpected response provider")
    if not response.get("choices"):
        raise ValueError("response contains no choices")
    usage = response.get("usage") or {}
    if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
        raise ValueError("reasoning enabled despite frozen request")
    return response


class Transport:
    def __init__(self, root, method, sender=None):
        self.root, self.method, self.sender = Path(root), method, sender
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "api.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY, key TEXT, method TEXT, request TEXT,
                    state TEXT, reserved REAL, charged REAL, response TEXT,
                    error TEXT, started REAL, seconds REAL);
                CREATE INDEX IF NOT EXISTS request_key ON requests(key);
                CREATE TABLE IF NOT EXISTS uses (
                    id INTEGER PRIMARY KEY, request_id INTEGER, method TEXT,
                    cached INTEGER, time REAL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=120)
        db.execute("PRAGMA busy_timeout=120000")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def slot(self):
        # Cross-process slots bound the sum of all methods to 16 live requests.
        handles = []
        try:
            for slot in range(16):
                path = self.root / "locks" / f"api-slot-{slot}.lock"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles.append(path.open("a"))
            while True:
                for handle in handles:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    yield
                    return
                time.sleep(.1)
        finally:
            for handle in handles:
                handle.close()

    def send(self, request):
        if self.sender is not None:
            return self.sender(request)
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key.startswith("sk-or-") or any(c.isspace() for c in key):
            raise ValueError("OPENROUTER_API_KEY missing or malformed")
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(request).encode(), headers={"Authorization": "Bearer " + key,
                                                       "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    def query(self, original):
        request = effective_request(original)
        key = digest(request)
        # Hash buckets avoid creating a file for every triplet.
        with file_lock(self.root / "locks" / f"request-{key[:2]}.lock"):
            with self.connect() as db:
                saved = db.execute("SELECT id,response FROM requests WHERE key=? AND state='complete' ORDER BY id LIMIT 1", (key,)).fetchone()
                if saved:
                    db.execute("INSERT INTO uses(request_id,method,cached,time) VALUES(?,?,1,?)", (saved[0], self.method, time.time()))
                    return checked_response(json.loads(saved[1])), f"api.sqlite3#{saved[0]}"
                uncertain = db.execute("SELECT COUNT(*) FROM requests WHERE key=? AND state='dispatched'", (key,)).fetchone()[0]
                if uncertain:
                    raise RuntimeError("uncertain previous request retained; inspect before redispatch")
            for attempt in range(3):
                with self.slot():
                    with self.connect() as db:
                        db.execute("BEGIN IMMEDIATE")
                        spent = db.execute("SELECT COALESCE(SUM(charged),0) FROM requests").fetchone()[0]
                        reserve = reservation(request)
                        if spent + reserve > CAP:
                            raise RuntimeError("ALARM API spending ceiling reached")
                        cursor = db.execute("INSERT INTO requests(key,method,request,state,reserved,charged,started) VALUES(?,?,?,'dispatched',?,?,?)",
                            (key, self.method, json.dumps(request), reserve, reserve, time.time()))
                        request_id = cursor.lastrowid
                    started = time.monotonic()
                    error = None
                    try:
                        response = self.send(request)
                    except Exception as exc:
                        error = exc
                        detail = {"type": type(exc).__name__, "status": getattr(exc, "code", None)}
                        if isinstance(exc, urllib.error.HTTPError):
                            detail["body"] = exc.read().decode(errors="replace")[:4000]
                        with self.connect() as db:
                            db.execute("UPDATE requests SET state='failed',error=?,seconds=? WHERE id=?",
                                (json.dumps(detail), time.monotonic()-started, request_id))
                    else:
                        usage = response.get("usage") or {}
                        cost = max(float(usage.get("cost") or 0), usage.get("prompt_tokens", 0)*PRICES["prompt"] + usage.get("completion_tokens", 0)*PRICES["completion"]) if usage else reserve
                        try:
                            checked_response(response)
                        except Exception as exc:
                            error = exc
                        with self.connect() as db:
                            db.execute("UPDATE requests SET state=?,response=?,charged=?,seconds=?,error=? WHERE id=?",
                                ("invalid" if error else "complete", json.dumps(response), cost, time.monotonic()-started,
                                 json.dumps({"type": type(error).__name__, "message": str(error)}) if error else None, request_id))
                            if not error:
                                db.execute("INSERT INTO uses(request_id,method,cached,time) VALUES(?,?,0,?)", (request_id, self.method, time.time()))
                        if error is None:
                            return response, f"api.sqlite3#{request_id}"
                if getattr(error, "code", None) not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"API request {request_id} failed; inspect the ledger") from error
                time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    def summary(self):
        with self.connect() as db:
            rows = db.execute("SELECT method,state,charged,response FROM requests").fetchall()
            uses = dict(db.execute("SELECT method,SUM(cached) FROM uses GROUP BY method").fetchall())
        groups = {}
        for method, state, charged, raw in rows:
            group = groups.setdefault(method, dict(requests=0, completed=0, accounted_usd=0., prompt_tokens=0, completion_tokens=0))
            group["requests"] += 1
            group["completed"] += int(state == "complete")
            group["accounted_usd"] += charged
            usage = (json.loads(raw).get("usage") or {}) if raw else {}
            for key in ("prompt_tokens", "completion_tokens"):
                group[key] += usage.get(key, 0)
        return dict(by_method=groups, cached_uses=uses, requests=len(rows),
                    accounted_usd=sum(r[2] for r in rows), cap_usd=CAP)


@contextmanager
def proxy(transport):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            try:
                if self.path != "/v1/chat/completions":
                    raise ValueError("unsupported endpoint")
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                response, _ = transport.query(request)
                payload, status = json.dumps(response).encode(), 200
            except Exception as error:
                payload, status = json.dumps({"error": {"message": str(error)}}).encode(), 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
