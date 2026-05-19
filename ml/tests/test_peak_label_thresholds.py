"""Peak label thresholds (separate from detection) and nitro/N-O guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_guardrails import apply_nitro_noxide_confounder_guardrails, apply_v3_guardrails
from ml.ftir_peak_picking import (
    peak_normalized_absorbance,
    peak_passes_label_threshold,
    peaks_for_label,
    pick_spectral_peaks,
    resolve_peak_label_thresholds,
)
from ml.ftir_region_ruler import FTIR_RULER_REGIONS


def _synth_peak(rel_h: float, wn: float = 1520.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(400, 4000, 3600)
    y = 0.02 + 0.9 * np.exp(-0.5 * ((x - 1700.0) / 40.0) ** 2)
    y += rel_h * np.exp(-0.5 * ((x - wn) / 4.0) ** 2)
    y = (y - y.min()) / (y.max() + 1e-9)
    return x, y


def test_peak_label_height_defaults_to_detection() -> None:
    th = resolve_peak_label_thresholds(
        "front",
        peak_min_height=0.05,
        peak_min_prominence=0.025,
    )
    assert th["peak_label_min_height"] == 0.05
    assert th["peak_label_min_prominence"] == 0.025
    o = resolve_peak_label_thresholds(
        "front",
        peak_min_height=0.05,
        peak_min_prominence=0.025,
        peak_label_min_height=0.12,
    )
    assert o["peak_label_min_height"] == 0.12


def _peak_at_height(h: float) -> dict:
    return {
        "wn_cm1": 1520.0,
        "height": h,
        "rel_height": h,
        "local_prominence": 0.04,
        "_y_range": 1.0,
        "peak_quality": "moderate",
        "peak_role": "detected_peak",
    }


def test_sensitive_preset_labels_peak_at_006() -> None:
    labeled = peaks_for_label(
        [_peak_at_height(0.06)],
        show_weak_peaks=True,
        max_peak_labels=5,
        peak_min_height=0.05,
        peak_min_prominence=0.025,
        peak_label_preset="sensitive",
        report_audience="front",
    )
    assert labeled
    assert labeled[0].get("label_reason") in ("height_prominence", "key_evidence", "all_visible")


def test_conservative_preset_skips_peak_at_006() -> None:
    labeled = peaks_for_label(
        [_peak_at_height(0.06)],
        show_weak_peaks=True,
        max_peak_labels=5,
        peak_min_height=0.05,
        peak_min_prominence=0.025,
        peak_label_preset="conservative",
        report_audience="front",
    )
    assert not labeled


def test_peak_labels_with_low_threshold() -> None:
    wn, y = _synth_peak(0.10, wn=1520.0)
    peaks = pick_spectral_peaks(
        wn, y, sensitivity="sensitive", peak_min_height=0.05, peak_min_prominence=0.025, max_peaks=40
    )
    near = [p for p in peaks if abs(float(p["wn_cm1"]) - 1520.0) < 20]
    assert near
    labeled = peaks_for_label(
        peaks,
        show_weak_peaks=True,
        max_peak_labels=20,
        peak_label_min_height=0.05,
        peak_label_min_prominence=0.025,
        report_audience="front",
    )
    assert any(abs(float(p["wn_cm1"]) - 1520.0) < 20 for p in labeled)


def test_product_default_skips_weak_unless_diagnostic() -> None:
    wn, y = _synth_peak(0.12, wn=1280.0)
    peaks = pick_spectral_peaks(
        wn, y, sensitivity="sensitive", peak_min_height=0.05, peak_min_prominence=0.025, max_peaks=40
    )
    labeled = peaks_for_label(
        peaks,
        show_weak_peaks=True,
        max_peak_labels=20,
        report_audience="front",
    )
    # Strict fingerprint peak may not label under front defaults without key evidence
    assert all(
        p.get("label_reason") in ("height_prominence", "diagnostic", "key_evidence", "forced", "audit")
        for p in labeled
    )


def test_no_hidden_0225_label_cutoff() -> None:
    """Labeling must not hard-reject rel_height 0.10–0.22 when label thresholds allow."""
    p = {
        "wn_cm1": 1520.0,
        "height": 0.18,
        "rel_height": 0.18,
        "local_prominence": 0.08,
        "_y_range": 1.0,
        "peak_role": "weak_peak",
    }
    assert peak_passes_label_threshold(p, peak_label_min_height=0.15, peak_label_min_prominence=0.05)
    labeled = peaks_for_label(
        [p],
        show_weak_peaks=True,
        max_peak_labels=5,
        peak_label_min_height=0.15,
        peak_label_min_prominence=0.05,
        report_audience="front",
    )
    assert labeled and labeled[0].get("label_reason") == "height_prominence"


def test_label_all_above_height_labels_every_qualifying_peak() -> None:
    wn, y = _synth_peak(0.10, wn=1520.0)
    wn2, y2 = _synth_peak(0.11, wn=1280.0)
    # merge two peaks into one trace
    y = np.maximum(y, y2)
    peaks = pick_spectral_peaks(
        wn, y, sensitivity="sensitive", peak_min_height=0.05, peak_min_prominence=0.025, max_peaks=40
    )
    labeled = peaks_for_label(peaks, label_all_above_height=0.1, max_peak_labels=3)
    qualifying = [
        p
        for p in peaks
        if peak_normalized_absorbance(p) >= 0.1
        and str(p.get("peak_quality") or "") != "noise_like"
    ]
    assert len(labeled) == len(qualifying)
    assert len(labeled) >= 2
    assert all(peak_normalized_absorbance(p) >= 0.1 for p in labeled)
    assert all(p.get("label_reason") == "above_height" for p in labeled)


def test_key_evidence_peak_labels_below_height_threshold() -> None:
    p = {
        "wn_cm1": 1510.0,
        "rel_height": 0.08,
        "local_prominence": 0.04,
        "_y_range": 1.0,
        "peak_role": "weak_peak",
    }
    labeled = peaks_for_label(
        [p],
        max_peak_labels=5,
        peak_label_min_height=0.15,
        peak_label_min_prominence=0.05,
        key_evidence_wn={1510.0},
        report_audience="front",
    )
    assert labeled
    assert labeled[0]["label_reason"] == "key_evidence"


def test_ruler_unsat_mid_includes_n_o() -> None:
    spec = next(r for r in FTIR_RULER_REGIONS if r.id == "unsat_mid")
    assert "N–O" in spec.short_label or "N-O" in spec.short_label


def _evidence_nitro_asym_only() -> dict:
    return {
        "band_matches": [
            {
                "band_id": "nitro_asym",
                "matched": True,
                "support_score": 0.55,
                "peaks_near": [{"wn_cm1": 1540.0, "rel_height": 0.3}],
            },
            {"band_id": "nitro_sym", "matched": False, "support_score": 0.0, "peaks_near": []},
        ],
        "regions": {},
        "local_motifs": {},
        "artifacts": {"flags": {}},
    }


def test_single_no2_band_does_not_support_nitro() -> None:
    assignments = {
        "nitro": {"score": 0.62, "ontology_category": "specific_fg", "confidence_class": "supported"},
    }
    apply_nitro_noxide_confounder_guardrails(assignments, _evidence_nitro_asym_only())
    assert float(assignments["nitro"]["score"]) <= 0.28


def test_paired_no2_can_remain_higher() -> None:
    ev = _evidence_nitro_asym_only()
    ev["band_matches"].append(
        {
            "band_id": "nitro_sym",
            "matched": True,
            "support_score": 0.4,
            "peaks_near": [{"wn_cm1": 1350.0}],
        }
    )
    assignments = {"nitro": {"score": 0.55, "ontology_category": "specific_fg"}}
    apply_v3_guardrails(assignments, ev)
    apply_nitro_noxide_confounder_guardrails(assignments, ev)
    assert float(assignments["nitro"]["score"]) >= 0.28


def test_heteroaromatic_suppresses_single_band_nitro() -> None:
    assignments = {
        "nitro": {"score": 0.5},
        "heteroaromatic": {"score": 0.45},
        "pyrrole_like_NH": {"score": 0.3},
    }
    apply_nitro_noxide_confounder_guardrails(assignments, _evidence_nitro_asym_only())
    assert float(assignments["nitro"]["score"]) <= 0.25
    assert any("N–O" in c or "NO₂" in c for c in assignments["nitro"].get("caution_flags", []))
