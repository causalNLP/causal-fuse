"""Bounded-memory storage for the exact, resumable gPC shuffle."""
from collections.abc import MutableMapping
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile


class DiskMap(MutableMapping):
    def __init__(self, db):
        self.db = db

    def __getitem__(self, key):
        row = self.db.execute("SELECT value FROM swaps WHERE key=?", (int(key),)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __setitem__(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO swaps VALUES(?,?)", (int(key), int(value)))

    def __delitem__(self, key):
        if not self.db.execute("DELETE FROM swaps WHERE key=?", (int(key),)).rowcount:
            raise KeyError(key)

    def __iter__(self):
        return (str(row[0]) for row in self.db.execute("SELECT key FROM swaps ORDER BY key"))

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM swaps").fetchone()[0]

    def clear(self):
        self.db.execute("DELETE FROM swaps")


class DiskCheckpoint:
    """A local transactional working copy and an atomic durable checkpoint."""
    def __init__(self, root, key):
        self.folder = Path(root) / "execution" / "gpc" / str(key)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder / "lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.durable = self.folder / "checkpoint.sqlite3"
        self.local_dir = Path(tempfile.mkdtemp(prefix="alarm-gpc-", dir=os.environ.get("SLURM_TMPDIR")))
        self.local = self.local_dir / "state.sqlite3"
        if self.durable.exists():
            shutil.copyfile(self.durable, self.local)
        self.db = sqlite3.connect(self.local)
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("CREATE TABLE IF NOT EXISTS swaps(key INTEGER PRIMARY KEY,value INTEGER NOT NULL);"
                             "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);")
        self.mapping = DiskMap(self.db)

    def load(self):
        row = self.db.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
        if row is None:
            return None
        state = json.loads(row[0])
        state["swaps"] = self.mapping
        return state

    def save(self, state):
        value = {k: v for k, v in state.items() if k != "swaps"}
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES('state',?)", (json.dumps(value),))
        self.db.commit()
        temporary = self.durable.with_suffix(".partial.sqlite3")
        with sqlite3.connect(temporary) as target:
            self.db.backup(target)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, self.durable)
        fd = os.open(self.folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def import_state(self, state):
        if self.load() is not None:
            return
        self.db.executemany("INSERT INTO swaps VALUES(?,?)",
                            ((int(k), int(v)) for k, v in state["swaps"].items()))
        self.save(state)

    def close(self):
        self.db.close()
        shutil.rmtree(self.local_dir)
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
