"""Tests for targeted confounder coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.external.confounder_targets import (
    NITRO_POSITIVE,
    classify_spectrum,
    match_target_class,
)
from ml.external.summarize_confounder_coverage import summarize_confounder_coverage


def test_nitro_positive_matches_title():
    md = {"title": "4-nitrotoluene", "dataset_tags": ["nitro"]}
    assert match_target_class(md, NITRO_POSITIVE)


def test_nitroso_excluded_from_nitro_positive():
    md = {"title": "nitrosobenzene", "dataset_tags": ["nitroso"]}
    assert not match_target_class(md, NITRO_POSITIVE)


def test_heteroaromatic_from_examples():
    md = {"title": "Indole", "dataset_tags": ["heteroaromatic"]}
    classes = classify_spectrum(md)
    assert "nitro_hn_heteroaromatic" in classes


def test_summarize_on_examples_index():
    db = Path("data/experimental/examples_index.sqlite")
    if not db.is_file():
        pytest.skip("examples index not built")
    summary = summarize_confounder_coverage(sqlite_paths=[db])
    assert summary["total_spectra"] >= 5
    assert "coverage_gaps" in summary
    assert len(summary["coverage_gaps"]) > 0  # examples lack nitro/siloxane


def test_manifest_files_exist():
    base = Path("data/benchmark_sets")
    for name in (
        "nitro_vs_noxide_manifest.json",
        "amide_vs_enamine_manifest.json",
        "siloxane_vs_CO_manifest.json",
    ):
        assert (base / name).is_file()
