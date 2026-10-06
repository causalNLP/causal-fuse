from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = 2
LEGACY_ARTIFACT_SCHEMA_VERSION = 1


def run_id(value: str | None = None) -> str:
    result = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"\d{8}T\d{6}Z", result):
        raise ValueError("run ID must be a UTC timestamp formatted YYYYMMDDTHHMMSSZ")
    return result


def json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        return None
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{token}.tmp")
    mode = "x" if exclusive else "w"
    with temporary.open(mode, encoding="utf-8") as handle:
        json.dump(json_value(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    if exclusive and path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields is not None else list(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_schema_version(payload: Mapping[str, Any]) -> int:
    return int(payload.get("artifact_schema_version", LEGACY_ARTIFACT_SCHEMA_VERSION))


def require_current_cache(payload: Mapping[str, Any]) -> None:
    version = artifact_schema_version(payload)
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "cache was produced by the pre-package artifact schema and cannot be resumed; "
            "start a new timestamped run"
        )
