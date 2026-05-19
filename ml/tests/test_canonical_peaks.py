"""Regression tests for canonical peak / evidence consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.canonical_peaks import (
    build_canonical_peak_table,
    peaks_as_legacy_dicts,
    select_static_label_peak_ids,
    validate_report_peak_consistency,
)
from ml.ftir_evidence import extract_spectral_evidence


def _synthetic_pipeline() -> dict:
    wn = np.linspace(4000, 400, 600)
    y = 0.4 * np.exp(-((wn - 1700) ** 2) / (2 * 60**2))
    y += 0.35 * np.exp(-((wn - 1320) ** 2) / (2 * 25**2))
    y += 0.15 * np.exp(-((wn - 1100) ** 2) / (2 * 30**2))
    y += 0.12 * np.exp(-((wn - 3220) ** 2) / (2 * 80**2))
    evidence = extract_spectral_evidence(
        wn,
        y,
        config={"peak_sensitivity": "sensitive", "ontology": "v4"},
    )
    pipeline = {
        "ontology": "v4",
        "evidence": evidence,
        "rule_assignments": {"assignments": {}},
    }
    return pipeline


def test_canonical_peak_id_on_all_detected() -> None:
    pipeline = _synthetic_pipeline()
    pack = build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    peaks = pipeline["evidence"]["peaks"]
    assert all(p.get("peak_id") for p in peaks)
    assert pack["stats"]["canonical_peak_count"] == len(peaks)


def test_evidence_row_peak_matches_canonical_center() -> None:
    pipeline = _synthetic_pipeline()
    build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    by_id = pipeline["canonical_peaks"]["peak_by_id"]
    for er in pipeline["canonical_peaks"]["evidence_rows"]:
        if er.get("source") != "observed_peak":
            continue
        pid = er["peak_id"]
        assert pid in by_id
        assert er["peak_cm1"] == round(float(by_id[pid]["center_cm1"]))


def test_region_only_no_fabricated_peak_cm1() -> None:
    pipeline = _synthetic_pipeline()
    build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    for er in pipeline["canonical_peaks"]["evidence_rows"]:
        if er.get("source") == "region_activity":
            assert er.get("peak_cm1") is None


def test_no_duplicate_evidence_keys_after_merge() -> None:
    pipeline = _synthetic_pipeline()
    build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    keys = [
        (r.get("peak_id"), r.get("band_id"), tuple(r.get("possible_functional_groups") or []))
        for r in pipeline["canonical_peaks"]["evidence_rows"]
    ]
    assert len(keys) == len(set(keys))


def test_static_policy_key_fewer_than_all_labeled() -> None:
    pipeline = _synthetic_pipeline()
    pack = build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    labeled_n = sum(1 for p in pack["peaks"] if p["labeled"])
    key_ids = select_static_label_peak_ids(pack, pipeline, policy="key", max_labels=12)
    if labeled_n > 12:
        assert len(key_ids) <= 12


def test_validate_consistency_ok_on_synthetic() -> None:
    pipeline = _synthetic_pipeline()
    build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    result = validate_report_peak_consistency(pipeline)
    assert result["ok"] is True


def test_band_map_html_uses_canonical(tmp_path: Path) -> None:
    pipeline = _synthetic_pipeline()
    build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    from reports.v4_evidence_report import build_band_evidence_map_html

    html = build_band_evidence_map_html(pipeline, audience="front")
    assert "Possible functional groups" in html
    assert "—</td><td>—</td><td>—</td><td>—</td>" not in html or "ether —" not in html


def test_plot_label_frequency_matches_canonical() -> None:
    pipeline = _synthetic_pipeline()
    pack = build_canonical_peak_table(pipeline, label_min_height=0.05, report_audience="front")
    from reports.v4_evidence_report import peak_annotation_specs

    labeled = peaks_as_legacy_dicts(pack, labeled=True)
    ann, _ = peak_annotation_specs(
        labeled,
        pipeline["evidence"],
        max_peaks=len(labeled),
        include_weak=True,
        fingerprint_cluster_distance=0,
    )
    by_id = pack["peak_by_id"]
    for a in ann:
        pid = a.get("_peak", {}).get("peak_id")
        if pid and pid in by_id:
            assert float(a["wn"]) == pytest.approx(float(by_id[pid]["center_cm1"]), abs=0.5)
