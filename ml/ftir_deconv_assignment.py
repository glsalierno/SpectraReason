"""
Advisory deconvolution → functional-group candidate mapping for FTIR reports.

Does not replace rules, evidence_v2, or SVM consensus. Outputs are labeled as
*candidate* / *possible* only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ml.ftir_band_library import load_band_library
from ml.ftir_deconvolution import (
    PeakComponent,
    RegionFitResult,
    _baseline_correct,
    _crop_region,
    _fit_region,
    _multi_pseudo_voigt,
    _pseudo_voigt,
    deconv_to_evidence_dict,
    deconvolve_spectrum,
    region_fit_to_dict,
)

DeconvModel = Literal["lorentzian", "gaussian", "pseudo_voigt"]
RegionPreset = Literal["auto", "all", "fingerprint", "carbonyl", "oh_nh", "ch", "custom"]

# region_id, wn_lo, wn_hi, max_components, min_sep, min_fwhm, max_fwhm
REPORT_DECONV_WINDOWS: tuple[tuple[str, float, float, int, float, float, float], ...] = (
    ("oh_nh", 3000.0, 3700.0, 4, 55.0, 60.0, 450.0),
    ("ch_stretch", 2800.0, 3100.0, 4, 25.0, 15.0, 120.0),
    ("triple_bond", 2000.0, 2300.0, 3, 22.0, 5.0, 50.0),
    ("carbonyl", 1600.0, 1850.0, 4, 18.0, 8.0, 90.0),
    ("unsat_mid", 1450.0, 1650.0, 6, 18.0, 10.0, 100.0),
    ("fingerprint_co", 1000.0, 1450.0, 6, 15.0, 8.0, 120.0),
    ("sio_overlap", 850.0, 1100.0, 5, 15.0, 8.0, 120.0),
)

REGION_PRESET_IDS: dict[str, frozenset[str]] = {
    "auto": frozenset(
        {"oh_nh", "carbonyl", "unsat_mid", "fingerprint_co", "sio_overlap", "triple_bond"}
    ),
    "all": frozenset(r[0] for r in REPORT_DECONV_WINDOWS),
    "fingerprint": frozenset({"fingerprint_co", "unsat_mid", "sio_overlap"}),
    "carbonyl": frozenset({"carbonyl"}),
    "oh_nh": frozenset({"oh_nh"}),
    "ch": frozenset({"ch_stretch"}),
    "custom": frozenset(),
}

AMBIGUITY_1500_1600 = (
    "aromatic C=C / amide II / nitro asym / enamine / heterocyclic N–O overlap (candidate only)"
)

SILOXANE_LABELS = frozenset({"siloxane", "silicone_or_silane", "siloxane_sio", "silicone_sic"})


@dataclass
class DeconvReportConfig:
    max_components_per_region: int = 6
    min_component_height: float = 0.03
    model: DeconvModel = "pseudo_voigt"
    region_preset: str = "auto"
    match_tolerance_cm1: float = 28.0
    front_max_table_rows: int = 12


def _model_eta(model: str) -> float:
    m = str(model or "pseudo_voigt").lower()
    if m == "lorentzian":
        return 1.0
    if m == "gaussian":
        return 0.0
    return 0.5


def _regions_for_preset(preset: str, custom_ids: frozenset[str] | None = None) -> list[tuple]:
    p = str(preset or "auto").lower()
    allowed = REGION_PRESET_IDS.get(p, REGION_PRESET_IDS["auto"])
    if p == "custom" and custom_ids:
        allowed = custom_ids
    out: list[tuple] = []
    for spec in REPORT_DECONV_WINDOWS:
        rid = spec[0]
        if rid not in allowed:
            continue
        out.append(spec)
    if not out and p != "custom":
        for spec in REPORT_DECONV_WINDOWS:
            if spec[0] in REGION_PRESET_IDS["auto"]:
                out.append(spec)
    return out


def deconvolve_report_spectrum(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    config: DeconvReportConfig | None = None,
    region_preset: str | None = None,
) -> dict[str, Any]:
    """Region-limited deconvolution for report overlay (advisory)."""
    cfg = config or DeconvReportConfig()
    preset = region_preset or cfg.region_preset
    wn = np.asarray(wn, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]
    y_max = float(np.nanmax(y)) if y.size else 1.0
    min_h = float(cfg.min_component_height)
    max_n_cap = int(cfg.max_components_per_region)
    eta = _model_eta(cfg.model)

    regions_out: dict[str, Any] = {}
    flat_components: list[dict[str, Any]] = []

    for rid, lo, hi, max_n, min_sep, min_fwhm, max_fwhm in _regions_for_preset(preset):
        max_n = min(max_n, max_n_cap)
        x, seg = _crop_region(wn, y, lo, hi)
        if x.size == 0:
            regions_out[rid] = region_fit_to_dict(_empty_fit(rid))
            continue
        seg_bc = _baseline_correct(x, seg)
        if y_max > 1e-9 and float(np.nanmax(seg_bc)) / y_max < min_h * 0.5:
            regions_out[rid] = region_fit_to_dict(_empty_fit(rid))
            continue
        fit = _fit_region(
            x,
            seg_bc,
            region_id=rid,
            max_components=max_n,
            min_sep=min_sep,
            min_fwhm=min_fwhm,
            max_fwhm=max_fwhm,
        )
        kept = [c for c in fit.components if c.amplitude >= min_h]
        if kept:
            fit.components = kept
            fit.success = True
        regions_out[rid] = region_fit_to_dict(fit)
        for c in fit.components:
            flat_components.append(_component_record(c, fit, rid, eta=eta))

    flat_components.sort(key=lambda r: (-float(r.get("rel_area", 0)), -float(r.get("height", 0))))
    curves = _build_curve_overlay(wn, y, regions_out, eta=eta)
    return {
        "regions": regions_out,
        "components": flat_components,
        "curves": curves,
        "profile_type": cfg.model,
        "region_preset": preset,
        "disclaimer": "Advisory deconvolution — candidate peaks only; not ground truth.",
    }


def _empty_fit(rid: str) -> RegionFitResult:
    return RegionFitResult(
        region=rid,
        success=False,
        r2=0.0,
        residual_norm=1.0,
        components=[],
        overlap_index=0.0,
        total_area=0.0,
        max_height=0.0,
        dominant_center=0.0,
        dominant_fwhm=0.0,
        mean_fwhm=0.0,
        min_fwhm=0.0,
    )


def _component_record(
    c: PeakComponent, fit: RegionFitResult, region_id: str, *, eta: float
) -> dict[str, Any]:
    total_a = float(fit.total_area) or 1e-12
    return {
        "region_id": region_id,
        "center": float(c.center),
        "height": float(c.amplitude),
        "fwhm": float(c.fwhm),
        "area": float(c.area),
        "rel_area": float(c.area / total_a),
        "fit_r2": float(fit.r2),
        "residual_norm": float(fit.residual_norm),
        "eta": eta,
    }


def _build_curve_overlay(
    wn: np.ndarray, y: np.ndarray, regions: dict[str, Any], *, eta: float
) -> dict[str, Any]:
    """Evaluate per-region measured segment, total fit, and components on native grids."""
    segments: list[dict[str, Any]] = []
    for rid, reg in regions.items():
        if not reg.get("success") or not reg.get("components"):
            continue
        lo = min(float(c["center"]) for c in reg["components"]) - 80
        hi = max(float(c["center"]) for c in reg["components"]) + 80
        for spec in REPORT_DECONV_WINDOWS:
            if spec[0] == rid:
                lo, hi = max(lo, spec[1]), min(hi, spec[2])
                break
        x, seg = _crop_region(wn, y, lo, hi)
        if x.size < 8:
            continue
        seg_bc = _baseline_correct(x, seg)
        comps = reg["components"]
        params: list[float] = []
        for c in comps:
            sig = float(c.get("fwhm", 20)) / 2.355
            params.extend([float(c["center"]), float(c.get("amplitude", c.get("height", 0))), sig])
        yhat = _multi_pseudo_voigt(x, *params) if params else np.zeros_like(x)
        comp_curves = []
        for c in comps:
            sig = float(c.get("fwhm", 20)) / 2.355
            yc = _pseudo_voigt(x, float(c["center"]), float(c.get("amplitude", 0)), sig, eta)
            comp_curves.append({"center": c["center"], "y": yc.tolist()})
        segments.append(
            {
                "region_id": rid,
                "wn": x.tolist(),
                "measured": seg_bc.tolist(),
                "total_fit": yhat.tolist(),
                "components": comp_curves,
            }
        )
    return {"segments": segments}


def _band_center(band: dict[str, Any]) -> float:
    return 0.5 * (float(band["region_min_cm1"]) + float(band["region_max_cm1"]))


def _specificity_rank(spec: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(spec or "low").lower(), 2)


def rank_candidates_for_component(
    comp: dict[str, Any],
    library: list[dict[str, Any]],
    *,
    tolerance_cm1: float = 28.0,
) -> list[dict[str, Any]]:
    center = float(comp.get("center", 0))
    height = float(comp.get("height", 0))
    scored: list[tuple[float, dict[str, Any]]] = []
    for band in library:
        lo = float(band.get("region_min_cm1", 0))
        hi = float(band.get("region_max_cm1", 0))
        if center < lo - tolerance_cm1 or center > hi + tolerance_cm1:
            continue
        bc = _band_center(band)
        dist = abs(center - bc)
        in_window = lo <= center <= hi
        score = height * (1.0 + (0.35 if in_window else 0.0)) / (1.0 + dist / max(tolerance_cm1, 1.0))
        score *= 1.0 / (1.0 + _specificity_rank(str(band.get("specificity", "low"))))
        scored.append((score, band))
    scored.sort(key=lambda t: -t[0])
    return [
        {
            "band_id": b.get("id", ""),
            "label": b.get("label", b.get("id", "")),
            "mode": b.get("mode", ""),
            "specificity": b.get("specificity", "low"),
            "notes": b.get("notes", ""),
            "score": round(s, 4),
        }
        for s, b in scored[:8]
    ]


def _ambiguity_note(center: float) -> str:
    if 1450 <= center <= 1605:
        return AMBIGUITY_1500_1600
    if 1000 <= center <= 1120:
        return "Si–O / C–O / ATR crystal / siloxane overlap caution (candidate)"
    if 2000 <= center <= 2350:
        return "C≡N / C≡C / CO₂ artifact caution (candidate)"
    return ""


def _evidence_role(candidates: list[dict[str, Any]], comp: dict[str, Any]) -> str:
    if not candidates:
        return "local motif"
    top = candidates[0]
    spec = str(top.get("specificity", "low"))
    h = float(comp.get("height", 0))
    if spec == "high" and h >= 0.08:
        return "possible supporting"
    if spec == "medium":
        return "candidate"
    return "weak candidate"


def _linked_assignment(
    candidates: list[dict[str, Any]],
    assignments: dict[str, Any],
) -> str:
    if not candidates:
        return "—"
    labels = [str(c.get("label", "")) for c in candidates[:3]]
    linked = []
    for lab in assignments:
        if any(lab.lower() in str(c.get("label", "")).lower() for c in candidates[:5]):
            linked.append(lab)
    if linked:
        return ", ".join(linked[:3])
    return " / ".join(labels[:2]) if labels else "—"


def build_component_rows(
    deconv_pack: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    library: list[dict[str, Any]] | None = None,
    config: DeconvReportConfig | None = None,
    audit: bool = False,
) -> list[dict[str, Any]]:
    cfg = config or DeconvReportConfig()
    lib = library if library is not None else load_band_library(prefer_python=True)
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    evidence = pipeline.get("evidence") or {}
    measurement = evidence.get("measurement") or {}
    atr_active = bool(measurement.get("atr_aware") or measurement.get("mode", "").lower() == "atr")
    art = (evidence.get("artifacts") or {}).get("flags") or {}

    rows: list[dict[str, Any]] = []
    for comp in deconv_pack.get("components") or []:
        if float(comp.get("height", 0)) < cfg.min_component_height:
            continue
        cands = comp.get("candidates") or rank_candidates_for_component(
            comp, lib, tolerance_cm1=cfg.match_tolerance_cm1
        )
        amb = _ambiguity_note(float(comp["center"]))
        caut_parts = []
        if amb:
            caut_parts.append(amb)
        top_labels = [str(c.get("label", "")) for c in cands[:4]]
        if any(l in SILOXANE_LABELS for l in top_labels):
            caut_parts.append("Siloxane/Si–O: tentative unless multiple Si-related regions agree")
            if atr_active or art.get("atr_crystal_band"):
                caut_parts.append("ATR/crystal overlap — do not confirm siloxane from one component")
        if audit and float(comp.get("fit_r2", 0)) < 0.35:
            caut_parts.append(f"Poor regional fit R²={float(comp.get('fit_r2', 0)):.2f}")
        role = _evidence_role(cands, comp)
        if "possible supporting" in role:
            role = "candidate"
        rows.append(
            {
                "center": round(float(comp["center"]), 1),
                "width": round(float(comp.get("fwhm", 0)), 1),
                "rel_area": round(float(comp.get("rel_area", 0)), 3),
                "height": round(float(comp.get("height", 0)), 3),
                "top_candidates": ", ".join(top_labels[:4]) or "—",
                "evidence_role": role,
                "caution": "; ".join(caut_parts) if caut_parts else "—",
                "linked_assignment": _linked_assignment(cands, assigns),
                "region_id": comp.get("region_id", ""),
                "fit_r2": comp.get("fit_r2"),
                "candidates": cands,
            }
        )
    rows.sort(key=lambda r: (-float(r.get("rel_area", 0)), -float(r.get("height", 0))))
    if not audit:
        rows = rows[: cfg.front_max_table_rows]
    return rows


def fg_deconv_support_snippet(
    label: str,
    deconv_pack: dict[str, Any],
    *,
    pipeline: dict[str, Any] | None = None,
) -> str:
    """One-line supporting deconv components for an FG card."""
    _ = pipeline
    comps = deconv_pack.get("components") or []
    lab = str(label).lower()
    hits: list[str] = []
    for c in comps:
        center = float(c.get("center", 0))
        cands = c.get("candidates") or []
        if not cands and pipeline:
            cands = rank_candidates_for_component(c, load_band_library(prefer_python=True))
        if any(lab in str(x.get("label", "")).lower() for x in cands[:3]):
            hits.append(f"{center:.0f} cm⁻¹ ({c.get('region_id', '')})")
    if not hits:
        if lab in ("phenol",) and any(1180 <= float(c.get("center", 0)) <= 1260 for c in comps):
            hits = [f"{float(c['center']):.0f} cm⁻¹ (C–O/aryl-O)" for c in comps if 1180 <= float(c["center"]) <= 1260][:2]
        if lab in ("amide", "amide_carbonyl") and any(1630 <= float(c.get("center", 0)) <= 1690 for c in comps):
            hits.append("amide I component ~1650–1690 cm⁻¹")
        if lab in ("nitro",) and any(1500 <= float(c.get("center", 0)) <= 1570 for c in comps):
            hits.append("possible NO₂ asym ~1520–1560 cm⁻¹")
    if not hits:
        return ""
    return "Supporting deconvoluted components (candidate): " + "; ".join(hits[:4])


def enrich_deconv_pack_for_pipeline(
    deconv_pack: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    config: DeconvReportConfig | None = None,
) -> dict[str, Any]:
    """Attach ranked candidates to each flat component."""
    cfg = config or DeconvReportConfig()
    lib = load_band_library(prefer_python=True)
    pack = dict(deconv_pack)
    enriched = []
    for comp in pack.get("components") or []:
        c2 = dict(comp)
        c2["candidates"] = rank_candidates_for_component(
            c2, lib, tolerance_cm1=cfg.match_tolerance_cm1
        )
        enriched.append(c2)
    pack["components"] = enriched
    pack["table_rows"] = build_component_rows(pack, pipeline, library=lib, config=cfg, audit=False)
    pack["table_rows_audit"] = build_component_rows(
        pack, pipeline, library=lib, config=cfg, audit=True
    )
    return pack


def run_report_deconvolution(
    wn: np.ndarray,
    y: np.ndarray,
    pipeline: dict[str, Any],
    *,
    config: DeconvReportConfig | None = None,
) -> dict[str, Any]:
    """Full report deconv: fit, evidence dict for guardrails, enriched display pack."""
    cfg = config or DeconvReportConfig()
    pack = deconvolve_report_spectrum(wn, y, config=cfg)
    legacy = deconv_to_evidence_dict(
        deconvolve_spectrum(wn, y, mode="full" if cfg.region_preset == "all" else "fast")
    )
    pack["legacy_regions"] = legacy.get("regions")
    return enrich_deconv_pack_for_pipeline(pack, pipeline, config=cfg)


def candidate_summary_line(center: float, candidates: list[dict[str, Any]]) -> str:
    """Human-readable one-liner for hover."""
    labels = [str(c.get("label", "")) for c in candidates[:5]]
    amb = _ambiguity_note(center)
    base = f"{center:.0f} cm⁻¹ → possible: {', '.join(labels)}" if labels else f"{center:.0f} cm⁻¹ (unassigned)"
    if amb:
        base += f"<br>{amb}"
    return base
