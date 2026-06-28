"""Tests for external dataset ingestion framework."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ml.dataset_quality import audit_sqlite_index
from ml.external.build_confounder_benchmarks import build_confounder_benchmarks
from ml.external.import_jcamp_folder import ingest_jcamp_folder
from ml.external.ingest_common import (
    attach_provenance,
    basic_spectrum_qa,
    keyword_tags_from_metadata,
    make_reference_id,
    preprocess_for_index,
)
from ml.external.provenance import PREPROCESSING_VERSION
from ml.external.spectrum_index import count_spectra
from lib.ftir_foundation import read_jdx_spectrum, preprocess_spectrum


EXAMPLES = Path("examples/spectra")


@pytest.fixture
def any_jdx() -> Path:
    files = list(EXAMPLES.glob("*.jdx"))
    if not files:
        pytest.skip("no example jdx")
    return files[0]


def test_provenance_fields():
    md = attach_provenance(
        {"title": "test"},
        source_id="test_src",
        source_name="Test",
        source_license="CC0",
        original_identifier="abc",
    )
    assert md["source_id"] == "test_src"
    assert md["preprocessing_version"] == PREPROCESSING_VERSION
    assert md["dataset_tier"] == "experimental"


def test_reference_id_stable():
    a = make_reference_id("src", "id1")
    b = make_reference_id("src", "id1")
    c = make_reference_id("src", "id2")
    assert a == b
    assert a != c


def test_keyword_tags_nitro():
    tags = keyword_tags_from_metadata({"title": "4-nitrotoluene"})
    assert "nitro" in tags


def test_ingest_examples_folder(tmp_path: Path):
    if not EXAMPLES.is_dir():
        pytest.skip("examples missing")
    out_db = tmp_path / "ex.sqlite"
    stats, db = ingest_jcamp_folder(
        EXAMPLES,
        out_db,
        source_id="test_examples",
        source_name="Examples",
        source_license="test",
        redistribution_allowed=False,
    )
    assert stats.ingested >= 5
    assert count_spectra(db) == stats.ingested

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT metadata_json FROM spectra LIMIT 1"
    ).fetchone()
    conn.close()
    md = json.loads(row[0])
    assert md.get("source_id") == "test_examples"
    assert md.get("source_license") == "test"
    assert "preprocessing_version" in md


def test_qa_on_ingested_index(tmp_path: Path):
    if not EXAMPLES.is_dir():
        pytest.skip("examples missing")
    out_db = tmp_path / "ex.sqlite"
    ingest_jcamp_folder(
        EXAMPLES, out_db, source_id="qa_test", source_name="T", source_license="test"
    )
    report = audit_sqlite_index(out_db)
    assert report.spectrum_count >= 5
    assert report.duplicate_spectra == 0


def test_benchmark_build(tmp_path: Path):
    if not EXAMPLES.is_dir():
        pytest.skip("examples missing")
    out_db = tmp_path / "ex.sqlite"
    ingest_jcamp_folder(
        EXAMPLES, out_db, source_id="bm", source_name="T", source_license="test"
    )
    summary = build_confounder_benchmarks(out_db, tmp_path / "benchmarks")
    assert "benchmarks" in summary
    assert (tmp_path / "benchmarks" / "index.json").is_file()


def test_preprocess_matches_foundation(any_jdx: Path):
    wn, y, _ = read_jdx_spectrum(any_jdx)
    wn1, y1, info = preprocess_spectrum(wn, y, intensity_mode="auto")
    md = {"xunits": "1/CM", "title": any_jdx.stem}
    prepped = preprocess_for_index(wn, y, md, intensity_mode="auto")
    assert prepped is not None
    wn2, y2, _ = prepped
    np.testing.assert_allclose(wn1, wn2, rtol=1e-5)
    np.testing.assert_allclose(y1, y2, rtol=1e-5)
    assert info.get("baseline") is not None


def test_flat_spectrum_qa_flag():
    wn = np.linspace(4000, 400, 200)
    y = np.zeros_like(wn)
    flags = basic_spectrum_qa(wn, y)
    assert "flat_spectrum" in flags
