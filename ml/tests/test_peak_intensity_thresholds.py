"""Intensity threshold (--peak-min-height / --peak-min-prominence) for peak picking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_evidence import extract_spectral_evidence
from ml.ftir_peak_picking import (
    intensity_passes,
    peaks_for_display,
    pick_spectral_peaks,
    resolve_peak_thresholds,
)


def _synth_peak_at_rel_height(
    rel_h: float = 0.05,
    peak_wn: float = 2240.0,
    width_cm1: float = 3.5,
    *,
    broad: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    wn = np.linspace(400, 4000, 3600)
    y = 0.02 + 0.01 * np.sin(wn * 0.007)
    y += 0.92 * np.exp(-0.5 * ((wn - 1700.0) / 40.0) ** 2)
    sig = 75.0 if broad else width_cm1
    y += rel_h * np.exp(-0.5 * ((wn - peak_wn) / sig) ** 2)
    y = y - float(np.min(y))
    y = y / (float(np.max(y)) + 1e-9)
    return wn, y


def _has_peak_near(peaks: list[dict], wn: float, tol: float = 15.0) -> bool:
    return any(abs(float(p["wn_cm1"]) - wn) <= tol for p in peaks)


def test_resolve_peak_threshold_defaults() -> None:
    th = resolve_peak_thresholds("sensitive")
    assert th["peak_min_height"] == 0.05
    assert th["peak_min_prominence"] == 0.025
    th2 = resolve_peak_thresholds("sensitive", peak_min_height=0.04)
    assert th2["peak_min_height"] == 0.04
    assert th2["peak_min_height_overridden"] is True


def test_narrow_peak_005_detected_with_explicit_thresholds() -> None:
    wn, y = _synth_peak_at_rel_height(0.05, peak_wn=2240.0, width_cm1=3.0)
    peaks = pick_spectral_peaks(
        wn,
        y,
        sensitivity="sensitive",
        peak_min_height=0.05,
        peak_min_prominence=0.025,
        max_peaks=80,
    )
    assert _has_peak_near(peaks, 2240.0)
    near = [p for p in peaks if abs(float(p["wn_cm1"]) - 2240.0) < 15]
    assert near
    assert float(near[0]["rel_height"]) >= 0.04


def test_same_peak_missed_with_conservative_defaults() -> None:
    # Crowded fingerprint (strict region): 0.05 peak should not pass conservative gates.
    wn, y = _synth_peak_at_rel_height(0.05, peak_wn=1280.0, width_cm1=3.0)
    peaks = pick_spectral_peaks(wn, y, sensitivity="conservative", max_peaks=80)
    assert not _has_peak_near(peaks, 1280.0)


def test_broad_ripple_005_not_diagnostic() -> None:
    wn, y = _synth_peak_at_rel_height(0.05, peak_wn=1100.0, broad=True)
    peaks = pick_spectral_peaks(
        wn,
        y,
        sensitivity="very_sensitive",
        peak_min_height=0.03,
        peak_min_prominence=0.015,
        max_peaks=80,
    )
    near = [p for p in peaks if abs(float(p["wn_cm1"]) - 1100.0) < 45]
    for p in near:
        assert str(p.get("peak_role")) != "diagnostic_peak" or float(
            p.get("rule_support_weight", 0)
        ) < 0.5


def test_weak_peak_hidden_without_show_flag() -> None:
    peaks = [
        {"wn_cm1": 2200.0, "peak_quality": "weak", "peak_role": "weak_peak"},
        {"wn_cm1": 1700.0, "peak_quality": "strong", "peak_role": "diagnostic_peak"},
        {"wn_cm1": 1600.0, "peak_quality": "moderate", "peak_role": "diagnostic_peak"},
    ]
    shown = peaks_for_display(peaks, show_weak_peaks=False, report_density="balanced")
    assert len(shown) == 2
    assert all(str(p.get("peak_quality")) in ("strong", "moderate") for p in shown)


def test_weak_peak_does_not_support_siloxane_ester_nitro() -> None:
    from ml.tests.test_ftir_atr_siloxane import _atr_evidence, _bm, _run

    for band_id, label in (
        ("siloxane_sio", "siloxane"),
        ("ester_co_o", "ester"),
        ("nitro_asym", "nitro"),
    ):
        ev = _atr_evidence(_bm(band_id, support=0.22, wn=1050 if "siloxane" in band_id else 1550))
        ev["peaks"] = [
            {
                "wn_cm1": 1050.0,
                "rel_height": 0.05,
                "peak_role": "weak_peak",
                "peak_quality": "weak",
                "rule_support_weight": 0.0,
            }
        ]
        ev["band_matches"][0]["peak_support"] = 0.04
        ev["band_matches"][0]["support_score"] = 0.22
        r = _run(ev)
        ent = r["assignments"].get(label) or {}
        assert float(ent.get("score", 0)) <= 0.25
        assert ent.get("confidence_class") != "supported"


def test_intensity_gate_requires_height_and_prominence() -> None:
    p = {
        "rel_height": 0.06,
        "local_prominence": 0.002,
        "wn_cm1": 2240.0,
        "_y_range": 1.0,
        "quality_isolation": 1.5,
        "quality_width_cm1": 12.0,
        "quality_sharpness": 0.04,
    }
    assert not intensity_passes(
        p, peak_min_height=0.05, peak_min_prominence=0.025, y_range=1.0
    )
    p["local_prominence"] = 0.04
    assert intensity_passes(p, peak_min_height=0.05, peak_min_prominence=0.025, y_range=1.0)


def test_evidence_config_passes_thresholds() -> None:
    wn, y = _synth_peak_at_rel_height(0.05, peak_wn=2235.0)
    ev = extract_spectral_evidence(
        wn,
        y,
        peaks=None,
        config={
            "peak_sensitivity": "sensitive",
            "peak_min_height": 0.05,
            "peak_min_prominence": 0.025,
            "ontology": "v4",
        },
    )
    assert ev["summary"]["peak_min_height"] == 0.05
    assert ev["summary"]["peak_min_prominence"] == 0.025
    assert _has_peak_near(ev["peaks"], 2235.0)
