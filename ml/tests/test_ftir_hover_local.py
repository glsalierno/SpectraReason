"""Tests for band-local Plotly hover context (no global probability dump)."""

from __future__ import annotations

from ml.ftir_band_library import load_band_library
from ml.ftir_evidence import build_local_hover_context, format_peak_marker_hover


def _sample_rule_assignments() -> dict:
    return {
        "assignments": {
            "phenol": {"score": 0.83, "status": "pass"},
            "aryl_ether": {"score": 0.87, "status": "pass"},
            "ester": {"score": 0.2, "status": "weak"},
        }
    }


def test_hover_around_1240_includes_phenol_or_aryl_ether_bands():
    lib = load_band_library(prefer_python=True)
    ctx = build_local_hover_context(
        1240.0,
        0.5,
        detected_peaks=[],
        band_library=lib,
        rule_assignments=_sample_rule_assignments(),
        ml_assignments=None,
        evidence=None,
        tolerance_cm1=12.0,
        max_labels=5,
    )
    h = ctx["hover_text"].lower()
    assert "phenol" in h or "aryl" in h or "phenolic" in h
    assert ctx["matching_bands"]
    assert ctx["matching_bands"][0].get("support_status") in (
        "supported",
        "partial",
        "not_supported",
        "unknown",
    )


def test_hover_far_region_no_diagnostic_band():
    lib = load_band_library(prefer_python=True)
    ctx = build_local_hover_context(
        2460.0,
        0.02,
        detected_peaks=[],
        band_library=lib,
        rule_assignments=_sample_rule_assignments(),
        evidence=None,
        tolerance_cm1=12.0,
    )
    assert "No diagnostic FTIR band" in ctx["hover_text"]


def test_unrelated_high_ml_not_in_hover_text():
    lib = load_band_library(prefer_python=True)
    ctx = build_local_hover_context(
        1240.0,
        0.5,
        detected_peaks=[],
        band_library=lib,
        rule_assignments=_sample_rule_assignments(),
        ml_assignments={"nitrile": 0.99, "phenol": 0.12},
        evidence=None,
        tolerance_cm1=12.0,
    )
    h = ctx["hover_text"]
    assert "nitrile" not in h.lower()
    assert "ML=0.99" not in h
    # Local FG may still show ML for phenol when band matches phenol key
    if "ML=" in h:
        assert "0.99" not in h


def test_no_rule_assignments_backward_compatible():
    lib = load_band_library(prefer_python=True)
    ctx = build_local_hover_context(
        1240.0,
        0.4,
        detected_peaks=[{"wn_cm1": 1230.0, "height": 0.3}],
        band_library=lib,
        rule_assignments=None,
        evidence=None,
    )
    assert ctx["nu"] == 1240.0
    assert ctx["near_peak"] is True
    for row in ctx["matching_bands"]:
        assert row["support_status"] == "unknown"


def test_peak_marker_hover_uses_local_context():
    lib = load_band_library(prefer_python=True)
    ctx = build_local_hover_context(
        1280.0,
        0.66,
        detected_peaks=[{"wn_cm1": 1280.0, "height": 0.66}],
        band_library=lib,
        rule_assignments=_sample_rule_assignments(),
        ml_assignments=None,
        evidence=None,
    )
    ph = format_peak_marker_hover(ctx)
    assert "Peak" in ph
    assert "1280.0" in ph or "1280" in ph
