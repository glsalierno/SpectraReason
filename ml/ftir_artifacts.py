"""
Atmospheric / baseline / noise artifact hints for evidence-first FTIR (v3 guardrails).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_spectral_artifacts(
    wavenumber: np.ndarray,
    absorbance: np.ndarray,
    evidence: dict[str, Any],
    *,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return flags + human-readable cautions. Soft signals only — no hard zeroing here.
    """
    wn = np.asarray(wavenumber, dtype=float).reshape(-1)
    y = np.asarray(absorbance, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]
    y_max = float(np.nanmax(y)) if y.size else 0.0
    regions = evidence.get("regions") or {}
    oh = regions.get("oh_nh_broad", {})
    fp = regions.get("fingerprint", {})
    nit = regions.get("nitrile", {})

    meas = measurement or evidence.get("measurement") or {}
    is_atr = bool(meas.get("is_atr")) or str(meas.get("mode") or "").upper() == "ATR"
    atr_aware = bool(meas.get("atr_aware", is_atr))

    flags: dict[str, bool] = {
        "water_vapor_or_moisture_like": False,
        "co2_region_elevated": False,
        "fingerprint_crowding": False,
        "atr_crystal_fingerprint_overlap": False,
        "possible_baseline_artifact": False,
        "possible_edge_artifact": False,
        "possible_saturation": False,
        "weak_nitrile_region_spike": False,
    }
    cautions: list[str] = []

    broad = float(oh.get("broadness", 0) or 0)
    oh_rel = float(oh.get("rel_max", 0) or 0)
    if broad > 0.55 and oh_rel > 0.35:
        flags["water_vapor_or_moisture_like"] = True
        cautions.append(
            "Broad O–H / H-bond envelope resembles moisture / water vapor interference; "
            "do not assign alcohol/phenol/acid from this region alone."
        )

    # CO2 doublet region ~2300–2400 often shows atmospheric lines in some instruments
    co2_lo, co2_hi = 2280.0, 2400.0
    m_co2 = (wn >= co2_lo) & (wn <= co2_hi)
    if np.any(m_co2):
        seg = y[m_co2]
        co2_rel = float(np.nanmax(seg) / (y_max + 1e-9)) if y_max > 0 else 0.0
        if co2_rel > 0.18:
            flags["co2_region_elevated"] = True
            cautions.append(
                "Elevated signal in CO₂ / atmospheric region — avoid nitrile vs alkyne calls from weak 2200–2260 cm⁻¹ spikes alone."
            )

    fp_rel = float(fp.get("rel_max", 0) or 0)
    c_o = regions.get("c_o_stretch", {})
    c_o_rel = float(c_o.get("rel_max", 0) or 0)
    if fp_rel > 0.55 and c_o_rel > 0.35:
        flags["fingerprint_crowding"] = True
        cautions.append(
            "Dense fingerprint / C–O overlap — siloxane vs ether vs ester requires paired evidence, not one band."
        )

    # ATR crystal / contact / fingerprint overlap (interpretation only — no spectral correction)
    if atr_aware and is_atr:
        flags["atr_crystal_fingerprint_overlap"] = True
        atr_msg = (
            "ATR-sensitive overlap region: crystal/contact/fingerprint effects can mimic Si–O / Si–O–Si "
            "or organic C–O bands; organosilicon requires paired silicon evidence."
        )
        if atr_msg not in cautions:
            cautions.append(atr_msg)

    # Baseline tilt proxy: difference of endpoints vs median
    if wn.size > 40:
        left = float(np.nanmean(y[:20]))
        right = float(np.nanmean(y[-20:]))
        mid = float(np.nanmedian(y))
        tilt = abs(left - right) / (abs(mid) + 1e-6)
        if tilt > 0.35:
            flags["possible_baseline_artifact"] = True
            cautions.append("Possible baseline tilt / offset; peak heights relative to local baseline may be biased.")

    # Edge clipping / saturation
    if y_max >= 0.995 or float(np.nanmax(np.abs(y))) > 1.5:
        flags["possible_saturation"] = True
        cautions.append("Possible saturation or clipped absorbance — peak intensities are unreliable.")

    if wn.size > 10:
        span = float(wn.max() - wn.min())
        if span < 700:
            flags["possible_edge_artifact"] = True
            cautions.append("Narrow spectral span — edge features may be truncated.")

    # Isolated nitrile-region spike without regional context
    nit_rel = float(nit.get("rel_max", 0) or 0)
    peaks = evidence.get("peaks") or []
    in_nit = [p for p in peaks if 2200 <= float(p.get("wn_cm1", 0)) <= 2265]
    if in_nit and nit_rel < 0.12:
        flags["weak_nitrile_region_spike"] = True
        cautions.append(
            "Weak nitrile-region peak with low regional envelope — treat nitrile / terminal alkyne as tentative."
        )

    return {
        "flags": flags,
        "cautions": cautions[:12],
        "summary": ", ".join(k for k, v in flags.items() if v) or "none",
    }
