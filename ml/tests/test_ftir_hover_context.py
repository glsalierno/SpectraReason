"""Tests for band-local Plotly hover (no global probability dumping per point)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_band_library import BAND_LIBRARY, load_band_library
from ml.ftir_evidence import build_local_hover_context, format_peak_marker_hover


def test_hover_1240_includes_phenol_aryl_ether_style_bands() -> None:
    # 1240 cm⁻¹ lies in phenolic C–O (1180–1260) and aryl ether (1180–1280) library windows.
    peaks = [{"wn_cm1": 1240.0, "height": 0.8}]
    rules = {
        "assignments": {
            "phenol": {"score": 0.83},
            "aryl_ether": {"score": 0.87},
            "ester": {"score": 0.2},
        }
    }
    ctx = build_local_hover_context(
        1240.0,
        0.66,
        peaks,
        BAND_LIBRARY,
        rule_assignments=rules,
        ml_assignments=None,
        evidence=None,
        tolerance_cm1=12.0,
        max_labels=5,
    )
    labels = {b["label"] for b in ctx["matching_bands"]}
    assert "phenol" in labels
    assert "aryl_ether" in labels
    tail = ctx["plotly_tail"]
    assert "phenol" in tail or "Phenol" in tail
    assert "aryl" in tail.lower() or "aryl_ether" in tail


def test_hover_2460_no_diagnostic_band() -> None:
    peaks: list[dict] = [{"wn_cm1": 500.0, "height": 0.1}]
    ctx = build_local_hover_context(
        2460.0,
        0.02,
        peaks,
        load_band_library(prefer_python=True),
        rule_assignments={"assignments": {"ketone": {"score": 0.99}}},
        ml_assignments={"ketone": 0.99},
        evidence=None,
        tolerance_cm1=12.0,
    )
    assert ctx["matching_bands"] == []
    assert "No diagnostic" in ctx["plotly_tail"] or "no diagnostic" in ctx["hover_text"].lower()
    assert "ketone" not in ctx["plotly_tail"].lower()


def test_high_ml_unrelated_label_not_in_hover() -> None:
    peaks = [{"wn_cm1": 1240.0, "height": 1.0}]
    ctx = build_local_hover_context(
        1240.0,
        0.5,
        peaks,
        BAND_LIBRARY,
        rule_assignments=None,
        ml_assignments={"ketone": 0.99, "phenol": 0.12},
        evidence=None,
        max_labels=5,
    )
    # Ketone C=O bands are not near 1240 cm⁻¹; ML must not be surfaced for ketone.
    assert "ketone" not in ctx["plotly_tail"].lower()
    assert "ML=0.99" not in ctx["plotly_tail"]


def test_peak_marker_hover_smoke() -> None:
    peaks = [{"wn_cm1": 1280.0, "height": 0.9}]
    ctx = build_local_hover_context(
        1280.0,
        0.9,
        peaks,
        BAND_LIBRARY,
        rule_assignments={"assignments": {"phenol": {"score": 0.8}}},
        max_labels=4,
    )
    h = format_peak_marker_hover(ctx)
    assert "Peak 1280" in h
    assert "Matched bands" in h


def test_backward_compat_no_rules_no_crash() -> None:
    ctx = build_local_hover_context(
        1185.0,
        0.4,
        [],
        BAND_LIBRARY,
        rule_assignments=None,
        ml_assignments=None,
        evidence=None,
    )
    assert "nu" in ctx and ctx["nu"] == 1185.0
    assert ctx["matching_bands"]
    assert all(b.get("support_status") == "unknown" for b in ctx["matching_bands"])
