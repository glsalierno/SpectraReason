"""Peak sensitivity presets, quality classes, and rule-support separation."""

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
    assign_peak_role,
    classify_peak_quality,
    peaks_for_display,
    pick_spectral_peaks,
)
from ml.ftir_rules import assign_functional_groups_from_evidence


def _synth_narrow_peak(
    peak_wn: float = 2240.0,
    peak_amp: float = 0.10,
    width_cm1: float = 4.0,
    *,
    broad_bump: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    wn = np.linspace(400, 4000, 3600)
    y = 0.04 + 0.015 * np.sin(wn * 0.008)
    y += 0.85 * np.exp(-0.5 * ((wn - 1700.0) / 35.0) ** 2)
    sig = 80.0 if broad_bump else width_cm1
    amp = 0.05 if broad_bump else peak_amp
    y += amp * np.exp(-0.5 * ((wn - peak_wn) / sig) ** 2)
    y = y - float(np.min(y))
    y = y / (float(np.max(y)) + 1e-9)
    return wn, y


def _has_peak_near(peaks: list[dict], wn: float, tol: float = 12.0) -> bool:
    return any(abs(float(p["wn_cm1"]) - wn) <= tol for p in peaks)


def test_sensitive_detects_low_narrow_peak_conservative_may_miss() -> None:
    wn, y = _synth_narrow_peak(peak_wn=2240.0, peak_amp=0.10, width_cm1=3.5)
    sens = pick_spectral_peaks(wn, y, sensitivity="sensitive", max_peaks=80)
    cons = pick_spectral_peaks(wn, y, sensitivity="conservative", max_peaks=80)
    assert _has_peak_near(sens, 2240.0)
    assert len(sens) >= len(cons)


def test_broad_low_bump_not_diagnostic() -> None:
    wn, y = _synth_narrow_peak(peak_wn=1100.0, broad_bump=True)
    peaks = pick_spectral_peaks(wn, y, sensitivity="very_sensitive", max_peaks=80)
    near = [p for p in peaks if abs(float(p["wn_cm1"]) - 1100.0) < 40.0]
    if not near:
        pytest.skip("broad bump not picked even in very_sensitive (acceptable noise rejection)")
    for p in near:
        assert str(p.get("peak_role")) != "diagnostic_peak" or float(
            p.get("rule_support_weight", 0)
        ) < 0.5
        if float(p.get("quality_width_cm1") or 0) > 50:
            assert str(p.get("peak_quality")) in ("noise_like", "weak", "moderate")


def test_weak_peak_display_flag() -> None:
    peaks = [
        {"wn_cm1": 2200.0, "height": 0.1, "peak_quality": "weak", "peak_role": "weak_peak"},
        {"wn_cm1": 1700.0, "height": 0.9, "peak_quality": "strong", "peak_role": "diagnostic_peak"},
    ]
    hidden = peaks_for_display(peaks, show_weak_peaks=False, report_density="balanced")
    shown = peaks_for_display(peaks, show_weak_peaks=True, report_density="balanced")
    assert len(hidden) == 1
    assert len(shown) == 2


def test_weak_peak_low_rule_support_weight() -> None:
    p = {
        "wn_cm1": 2240.0,
        "rel_height": 0.11,
        "quality_isolation": 2.0,
        "quality_width_cm1": 12.0,
        "quality_sharpness": 0.04,
        "quality_snr_proxy": 1.2,
        "local_prominence": 0.05,
        "_y_range": 1.0,
    }
    q = classify_peak_quality(p)
    role = assign_peak_role(p, q)
    from ml.ftir_peak_picking import rule_support_weight

    w = rule_support_weight({**p, "peak_quality": q, "peak_role": role})
    if role == "weak_peak":
        assert w <= 0.35


def test_band_peak_support_ignores_weak_only() -> None:
    wn, y = _synth_narrow_peak()
    peaks_weak = [
        {
            "wn_cm1": 2240.0,
            "height": 0.1,
            "rel_height": 0.10,
            "peak_role": "weak_peak",
            "peak_quality": "weak",
            "rule_support_weight": 0.0,
        }
    ]
    ev = extract_spectral_evidence(wn, y, peaks=peaks_weak, config={"ontology": "v4"})
    nitrile = next(
        (m for m in ev["band_matches"] if "nitrile" in str(m.get("band_id", ""))),
        {"peak_support": 0.0},
    )
    assert float(nitrile.get("peak_support", 0)) < 0.08


def test_atr_weak_sio_peak_does_not_support_siloxane() -> None:
    from ml.tests.test_ftir_atr_siloxane import _atr_evidence, _bm, _run

    ev = _atr_evidence(
        _bm("siloxane_sio", support=0.18, wn=1050),
    )
    ev["peaks"] = [
        {
            "wn_cm1": 1050.0,
            "height": 0.08,
            "rel_height": 0.08,
            "peak_role": "weak_peak",
            "peak_quality": "weak",
            "rule_support_weight": 0.0,
            "quality_sharpness": 0.01,
        }
    ]
    ev["band_matches"][0]["peak_support"] = 0.05
    ev["band_matches"][0]["support_score"] = 0.18
    r = _run(ev)
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) <= 0.20
    assert sil["confidence_class"] != "supported"


def test_nitrile_sharp_weak_can_be_tentative() -> None:
    wn, y = _synth_narrow_peak(peak_wn=2235.0, peak_amp=0.12, width_cm1=3.0)
    ev = extract_spectral_evidence(
        wn, y, peaks=None, config={"peak_sensitivity": "sensitive", "ontology": "v4"}
    )
    peaks = ev["peaks"]
    near = [p for p in peaks if 2210 <= float(p["wn_cm1"]) <= 2260]
    assert near
    r = assign_functional_groups_from_evidence(
        ev, config={"guardrails_mode": "v3", "ontology": "v4"}
    )
    nit = r["assignments"].get("nitrile") or {}
    if float(nit.get("score", 0)) > 0.05:
        assert nit.get("confidence_class") in (
            "local_possible",
            "tentative",
            "supported",
            "v3_guarded",
        )
