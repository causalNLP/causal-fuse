from __future__ import annotations

import zipfile
from pathlib import Path

from cd_llm.experiments.neuropathic_bfs2024 import (
    load_dataset,
    load_manifest,
)


def test_locked_kit_uses_original_indices_then_canonical_names() -> None:
    archive_path = Path("20260826T120230Z.zip")
    if not archive_path.exists():
        return
    with zipfile.ZipFile(archive_path) as archive:
        manifest = load_manifest(archive)
        frame, digest = load_dataset(archive, manifest, 0, 200)
    assert frame.shape == (200, 20)
    assert frame.columns[0] == "DLS C1-C2"
    assert frame.columns[-1] == "L Nackbesvär"
    assert len(digest) == 64
