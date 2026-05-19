"""Upper-mid (3200–1800 cm⁻¹) tiered shading, C–H motifs, and conservative FG guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from ml.ftir_evidence import build_local_hover_context, extract_spectral_evidence
from ml.ftir_rules import assign_functional_groups_from_evidence
from ml.ftir_shade_regions import render_tiered_band_shading, resolve_shade_thresholds, SPECTRUM_SHADE_REGION_SPECS
from reports.structural_fg_svm_kronecker_report import _traditional_region_shading_shapes


def _gauss(wn: np.ndarray, center: float, amp: float, width: float = 4000.0) -> np.ndarray:
    return amp * np.exp(-((wn - center) ** 2) / width)


def _base_wn() -> np.ndarray:
    return np.linspace(400.0, 4000.0, 3600)


def _shaded_ids(
    rel_peak: float,
    center: float,
    *,
    shade_sensitive: bool = False,
    faint_min: float = 0.05,
) -> set[str]:
    wn = _base_wn()
    y = _gauss(wn, center, rel_peak)
    evidence = {"regions": {}, "summary": {"y_max": float(y.max())}}
    shapes, _ = render_tiered_band_shading(
        evidence,
        wn,
        y,
        0.0,
        1.0,
        shade_faint_min=faint_min,
        shade_sensitive=shade_sensitive,
    )
    return {s.get("_shade_region_id", "") for s in shapes}


def test_aliphatic_ch_faint_vs_strong_shading() -> None:
    faint_ids = _shaded_ids(0.06, 2925.0)
    assert "aliphatic_ch" in faint_ids
    strong_ids = _shaded_ids(0.14, 2925.0)
    assert "aliphatic_ch" in strong_ids
    shapes, names = render_tiered_band_shading(
        {"regions": {}, "summary": {"y_max": 1.0}},
        _base_wn(),
        _gauss(_base_wn(), 2925.0, 0.14),
        0.0,
        1.0,
    )
    by_id = {s["_shade_region_id"]: s.get("_shade_tier") for s in shapes}
    assert by_id.get("aliphatic_ch") == "strong"


def test_aromatic_ch_hover_at_3050() -> None:
    wn = _base_wn()
    y = _gauss(wn, 3050.0, 0.12)
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4"})
    ctx = build_local_hover_context(
        3050.0,
        float(y[np.argmin(np.abs(wn - 3050))]),
        ev.get("peaks") or [],
        evidence=ev,
        ontology="v4",
    )
    html = (ctx.get("hover_text") or ctx.get("plotly_tail") or "").lower()
    assert "aromatic" in html or "region activity" in html or "local motif" in html


def test_upper_mid_activity_shades_without_supported_fg() -> None:
    wn = _base_wn()
    y = _gauss(wn, 2500.0, 0.11)
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4"})
    ids = _shaded_ids(0.11, 2500.0)
    assert "upper_mid_activity" in ids
    r = assign_functional_groups_from_evidence(
        ev, config={"guardrails_mode": "v3", "ontology": "v4"}
    )
    um = r["assignments"].get("upper_mid_activity_region") or {}
    assert float(um.get("score", 0) or 0) <= 0.35
    assert um.get("confidence_class") != "supported"
    assert um.get("ontology_category") == "local_motif"


def test_shading_independent_of_peak_threshold() -> None:
    wn = _base_wn()
    y = _gauss(wn, 2930.0, 0.08)
    ev_lo = extract_spectral_evidence(
        wn, y, config={"peak_sensitivity": "conservative", "ontology": "v4"}
    )
    ev_hi = extract_spectral_evidence(
        wn, y, config={"peak_sensitivity": "sensitive", "ontology": "v4"}
    )
    s_lo, _ = render_tiered_band_shading(ev_lo, wn, y, 0.0, 1.0)
    s_hi, _ = render_tiered_band_shading(ev_hi, wn, y, 0.0, 1.0)
    ids_lo = {s["_shade_region_id"] for s in s_lo}
    ids_hi = {s["_shade_region_id"] for s in s_hi}
    assert ("aliphatic_ch" in ids_lo) == ("aliphatic_ch" in ids_hi)


def test_shade_sensitive_lowers_faint_threshold() -> None:
    spec = next(s for s in SPECTRUM_SHADE_REGION_SPECS if s.id == "aliphatic_ch")
    _, faint_default = resolve_shade_thresholds(spec, shade_faint_min=0.05, shade_sensitive=False)
    _, faint_sens = resolve_shade_thresholds(spec, shade_faint_min=0.05, shade_sensitive=True)
    assert faint_sens < faint_default
    assert faint_sens <= 0.03

    wn = _base_wn()
    y = _gauss(wn, 2925.0, 0.035)
    evidence = {"regions": {}, "summary": {"y_max": float(y.max())}}
    shapes_off, _ = _traditional_region_shading_shapes(
        evidence=evidence, wn=wn, y=y, y_min=0.0, y_max=1.0, min_rel_max=0.10, shade_sensitive=False
    )
    shapes_on, _ = _traditional_region_shading_shapes(
        evidence=evidence,
        wn=wn,
        y=y,
        y_min=0.0,
        y_max=1.0,
        min_rel_max=0.10,
        shade_sensitive=True,
    )
    ids_off = {s.get("_shade_region_id") for s in shapes_off}
    ids_on = {s.get("_shade_region_id") for s in shapes_on}
    assert "aliphatic_ch" not in ids_off or len(ids_on) >= len(ids_off)


def test_ch_hover_and_key_motif_partition() -> None:
    wn = _base_wn()
    y = _gauss(wn, 2940.0, 0.15) + _gauss(wn, 1700.0, 0.2)
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4", "peak_sensitivity": "balanced"})
    ctx = build_local_hover_context(
        2940.0,
        float(y[np.argmin(np.abs(wn - 2940))]),
        ev.get("peaks") or [],
        evidence=ev,
        ontology="v4",
    )
    html = (ctx.get("hover_text") or ctx.get("plotly_tail") or "").lower()
    assert "aliphatic" in html or "region activity" in html or "local motif" in html
    lm = ev.get("local_motifs") or {}
    assert "aliphatic_CH_region" in lm


def test_ch_alone_does_not_create_supported_fg() -> None:
    wn = _base_wn()
    y = _gauss(wn, 2935.0, 0.25)
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4", "peak_sensitivity": "sensitive"})
    r = assign_functional_groups_from_evidence(
        ev, config={"guardrails_mode": "v3", "ontology": "v4"}
    )
    assigns = r["assignments"]
    for lab in ("aliphatic_CH_region", "aliphatic_CH_present", "CH_stretch_region"):
        ent = assigns.get(lab) or {}
        if float(ent.get("score", 0) or 0) > 0:
            assert ent.get("confidence_class") != "supported"
            assert float(ent.get("score", 0)) <= 0.35
    for lab, ent in assigns.items():
        if not isinstance(ent, dict):
            continue
        if ent.get("ontology_category") == "specific_fg" and float(ent.get("score", 0)) >= 0.5:
            assert lab not in ("aliphatic_CH_region", "aliphatic_CH_present", "CH_stretch_region")
