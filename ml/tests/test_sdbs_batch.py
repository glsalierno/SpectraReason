"""Tests for SDBS subfolder + manifest enrichment."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ml.external.import_sdbs import ingest_sdbs
from ml.external.sdbs_batch import enrich_sdbs_metadata, load_sdbs_manifest


@pytest.fixture
def sdbs_tree(tmp_path: Path) -> Path:
    root = tmp_path / "sdbs"
    (root / "nitro_positive").mkdir(parents=True)
    src = Path("examples/spectra/Benzoic acid - 65-85-0-IR.jdx")
    if not src.is_file():
        pytest.skip("example jdx missing")
    shutil.copy(src, root / "nitro_positive" / "nitrobenzene.jdx")
    manifest = root / "sdbs_download_manifest.csv"
    manifest.write_text(
        "batch_folder,compound_name,sdbs_id,sdbs_url,local_filename,cas,notes,download_date,downloaded\n"
        'nitro_positive,nitrobenzene,TEST123,https://sdbs.db.aist.go.jp/test,nitrobenzene.jdx,98-95-3,KBr pellet,2026-05-18,y\n',
        encoding="utf-8",
    )
    return root


def test_manifest_load(sdbs_tree: Path):
    m = load_sdbs_manifest(sdbs_tree / "sdbs_download_manifest.csv")
    assert "nitrobenzene" in m or _norm_key(m)


def _norm_key(m: dict) -> bool:
    return any("nitrobenzene" in k for k in m)


def test_enrich_metadata(sdbs_tree: Path):
    p = sdbs_tree / "nitro_positive" / "nitrobenzene.jdx"
    md = enrich_sdbs_metadata({}, p, sdbs_tree, load_sdbs_manifest(sdbs_tree / "sdbs_download_manifest.csv"))
    assert md.get("batch_folder") == "nitro_positive"
    assert md.get("sdbs_id") == "TEST123"
    assert "nitro" in (md.get("batch_tags") or [])


def test_ingest_sdbs_subfolder(sdbs_tree: Path):
    out = sdbs_tree / "out.sqlite"
    stats, db = ingest_sdbs(sdbs_tree, out)
    assert stats.ingested == 1
    assert db.is_file()
