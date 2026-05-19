"""FTIR region ruler and peak label UX (product_v1)."""

from __future__ import annotations

import numpy as np

from ml.ftir_evidence import build_local_hover_context, extract_spectral_evidence
from ml.ftir_peak_picking import peaks_for_label, pick_spectral_peaks
from ml.ftir_region_ruler import FTIR_RULER_REGIONS, ruler_region_activity


def _gauss(wn: np.ndarray, center: float, amp: float) -> np.ndarray:
    return amp * np.exp(-((wn - center) ** 2) / 4000.0)


def _ruler_by_id(rid: str) -> tuple[float, float]:
    for s in FTIR_RULER_REGIONS:
        if s.id == rid:
            return s.lo, s.hi
    raise KeyError(rid)


def test_ruler_aromatic_aliphatic_aldehydic_ranges() -> None:
    a0, a1 = _ruler_by_id("aromatic_ch")
    l0, l1 = _ruler_by_id("aliphatic_ch")
    d0, d1 = _ruler_by_id("aldehydic_ch")
    assert (a0, a1) == (3000.0, 3100.0)
    assert (l0, l1) == (2850.0, 2965.0)
    assert (d0, d1) == (2720.0, 2820.0)
    assert not (a0 <= 2800 <= a1)


def test_activity_2925_aliphatic_not_aromatic() -> None:
    wn = np.linspace(400.0, 4000.0, 3600)
    y = _gauss(wn, 2925.0, 0.2)
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4", "peak_sensitivity": "sensitive"})
    rel_ali = ruler_region_activity(ev, wn, y, next(s for s in FTIR_RULER_REGIONS if s.id == "aliphatic_ch"))
    rel_aro = ruler_region_activity(ev, wn, y, next(s for s in FTIR_RULER_REGIONS if s.id == "aromatic_ch"))
    assert rel_ali >= 0.05
    assert rel_ali > rel_aro


def test_hover_2800_aldehydic_not_aromatic() -> None:
    wn = np.linspace(400.0, 4000.0, 3600)
    y = _gauss(wn, 2780.0, 0.15) + _gauss(wn, 1700.0, 0.2)
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4"})
    ctx = build_local_hover_context(
        2780.0,
        float(y[np.argmin(np.abs(wn - 2780))]),
        ev.get("peaks") or [],
        evidence=ev,
        ontology="v4",
    )
    html = (ctx.get("hover_text") or ctx.get("plotly_tail") or "").lower()
    assert "aldehydic" in html or "2720" in html
    assert "aromatic/sp" not in html and "3000" not in html


def test_weak_diagnostic_labeled_with_show_weak() -> None:
    wn = np.linspace(400.0, 4000.0, 2000)
    y = _gauss(wn, 2230.0, 0.08)
    peaks = pick_spectral_peaks(
        wn, y, sensitivity="sensitive", peak_min_height=0.05, peak_min_prominence=0.025
    )
    labeled = peaks_for_label(
        peaks, show_weak_peaks=True, max_peak_labels=30, label_all_diagnostic=False
    )
    assert len(labeled) >= 1


def test_build_figure_has_ruler_subplot() -> None:
    from reports.structural_fg_svm_kronecker_report import _build_stacked_interactive_figure

    wn = np.linspace(400.0, 4000.0, 400)
    y = _gauss(wn, 1700.0, 0.3)
    pipeline = {
        "ontology": "v4",
        "evidence": extract_spectral_evidence(wn, y, config={"ontology": "v4"}),
        "rule_assignments": {"assignments": {}},
    }
    peaks = pipeline["evidence"]["peaks"]
    fig, meta = _build_stacked_interactive_figure(
        name="test",
        wn=wn,
        y=y,
        peak_wn=[float(p["wn_cm1"]) for p in peaks[:5]],
        peak_h=[float(p["height"]) for p in peaks[:5]],
        pipeline=pipeline,
        peaks_dicts=peaks,
        show_region_ruler=True,
        report_style="product_v1",
        show_band_shading=False,
    )
    assert meta.get("show_region_ruler") is True
    assert len(fig.layout.annotations or []) >= 1 or len(fig.data) >= 3
