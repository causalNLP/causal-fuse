"""Materialize the tracked, response-free inputs used by the paper experiments."""

from __future__ import annotations

import tarfile
from pathlib import Path

from cd_llm.experiments.common.artifacts import read_json, sha256
from cd_llm.paths import RESOURCE_ROOT


ROOT = RESOURCE_ROOT / "reproduction"


def protocol() -> dict:
    """Return the pinned public reproduction protocol after checking its files."""
    value = read_json(ROOT / "protocol.json")
    for name, expected in value["artifacts"].items():
        path = ROOT / name
        if sha256(path) != expected:
            raise ValueError(f"reproduction input checksum differs: {name}")
    return value


def materialize(name: str, destination: Path) -> dict[str, str]:
    """Extract one registered input bundle and return hashes of extracted files."""
    registered = protocol()["bundles"]
    if name not in registered:
        raise ValueError(f"unknown reproduction input bundle: {name}")
    archive = ROOT / registered[name]
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe path in reproduction bundle: {member.name}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe path in reproduction bundle: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    return {
        member.name: sha256(destination / member.name)
        for member in members
    }


__all__ = ["materialize", "protocol"]
