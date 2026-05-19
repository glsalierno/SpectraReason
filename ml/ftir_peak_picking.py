"""
Configurable FTIR peak picking: sensitivity presets, intensity thresholds, quality classes.

Peaks are classified for **display** vs **rule diagnostic support** separately so more
visible peaks do not automatically inflate functional-group confidence.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks

PeakSensitivity = Literal["conservative", "balanced", "sensitive", "very_sensitive"]
PeakQuality = Literal["strong", "moderate", "weak", "noise_like"]
PeakRole = Literal["diagnostic_peak", "weak_peak", "detected_peak"]

# Wavenumber windows (cm⁻¹): lower multiplier → easier scipy pre-filter in that ν range.
REGION_LENIENT: list[tuple[float, float]] = [
    (3280, 3340),   # alkyne C-H
    (2100, 2260),   # C≡C / C≡N
    (2200, 2265),   # nitrile
    (1500, 1575),   # nitro asymmetric
    (1450, 1605),   # aromatic / amide II
    (1630, 1820),   # carbonyl envelope + shoulders
    (1730, 1775),   # ester C=O
]
REGION_STRICT: list[tuple[float, float]] = [
    (650, 950),     # fingerprint / aromatic oop
    (1000, 1180),   # C-O / Si-O overlap
    (1180, 1450),   # dense fingerprint
    (3600, 4000),   # noisy high-ν edge
    (400, 650),     # low-ν edge
]

PEAK_SENSITIVITY_PRESETS: dict[str, dict[str, float]] = {
    "conservative": {
        "height_frac": 0.018,
        "prominence_frac": 0.018,
        "min_rel_height": 0.028,
        "min_prominence_frac": 0.014,
        "distance_pts": 5,
        "quality_scale": 1.15,
        "peak_min_height": 0.10,
        "peak_min_prominence": 0.06,
    },
    "balanced": {
        "height_frac": 0.012,
        "prominence_frac": 0.012,
        "min_rel_height": 0.020,
        "min_prominence_frac": 0.010,
        "distance_pts": 4,
        "quality_scale": 1.0,
        "peak_min_height": 0.07,
        "peak_min_prominence": 0.04,
    },
    "sensitive": {
        "height_frac": 0.007,
        "prominence_frac": 0.008,
        "min_rel_height": 0.085,
        "min_prominence_frac": 0.005,
        "distance_pts": 3,
        "quality_scale": 0.88,
        "peak_min_height": 0.05,
        "peak_min_prominence": 0.025,
    },
    "very_sensitive": {
        "height_frac": 0.004,
        "prominence_frac": 0.005,
        "min_rel_height": 0.055,
        "min_prominence_frac": 0.003,
        "distance_pts": 2,
        "quality_scale": 0.75,
        "peak_min_height": 0.03,
        "peak_min_prominence": 0.015,
    },
}

QUALITY_VISUAL = {
    "strong": {"size": 8, "opacity": 0.95, "color": "#1e40af", "symbol": "circle"},
    "moderate": {"size": 6, "opacity": 0.75, "color": "#64748b", "symbol": "circle"},
    "weak": {"size": 4, "opacity": 0.45, "color": "#94a3b8", "symbol": "circle-open"},
    "noise_like": {"size": 3, "opacity": 0.2, "color": "#cbd5e1", "symbol": "circle-open"},
}


def normalize_peak_sensitivity(value: str | None) -> PeakSensitivity:
    v = str(value or "balanced").lower().strip().replace("-", "_")
    if v in PEAK_SENSITIVITY_PRESETS:
        return v  # type: ignore[return-value]
    return "balanced"


def resolve_peak_thresholds(
    sensitivity: str | PeakSensitivity = "balanced",
    *,
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
) -> dict[str, Any]:
    """Preset defaults; explicit CLI values override preset."""
    sens = normalize_peak_sensitivity(sensitivity)
    preset = PEAK_SENSITIVITY_PRESETS[sens]
    return {
        "peak_sensitivity": sens,
        "peak_min_height": float(peak_min_height if peak_min_height is not None else preset["peak_min_height"]),
        "peak_min_prominence": float(
            peak_min_prominence if peak_min_prominence is not None else preset["peak_min_prominence"]
        ),
        "peak_min_height_overridden": peak_min_height is not None,
        "peak_min_prominence_overridden": peak_min_prominence is not None,
    }


PeakLabelPreset = Literal["conservative", "balanced", "sensitive", "all_visible"]

PEAK_LABEL_PRESET_HEIGHTS: dict[str, float | None] = {
    "conservative": 0.15,
    "balanced": 0.10,
    "sensitive": 0.05,
    "all_visible": None,
}


def normalize_peak_label_preset(preset: str | None) -> str:
    p = str(preset or "").strip().lower()
    if p in PEAK_LABEL_PRESET_HEIGHTS:
        return p
    return ""


def resolve_peak_label_thresholds(
    report_audience: str = "front",
    *,
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
    peak_label_min_height: float | None = None,
    peak_label_min_prominence: float | None = None,
    peak_label_preset: str | None = None,
) -> dict[str, Any]:
    """
    Label thresholds are separate from detection.

    Unless ``peak_label_min_height`` / ``peak_label_min_prominence`` are set explicitly,
    defaults follow detection (``peak_min_height`` / ``peak_min_prominence``).
    ``peak_label_preset`` applies named height floors (prominence still follows detection
    unless overridden).
    """
    _ = report_audience  # reserved for future audience-specific presets
    det_h = float(peak_min_height if peak_min_height is not None else 0.05)
    det_p = float(peak_min_prominence if peak_min_prominence is not None else 0.025)
    preset = normalize_peak_label_preset(peak_label_preset)
    if preset == "all_visible":
        lh, lp = det_h, det_p
    elif preset and PEAK_LABEL_PRESET_HEIGHTS.get(preset) is not None:
        lh = float(PEAK_LABEL_PRESET_HEIGHTS[preset])  # type: ignore[arg-type]
        lp = det_p
    else:
        lh, lp = det_h, det_p
    if peak_label_min_height is not None:
        lh = float(peak_label_min_height)
    if peak_label_min_prominence is not None:
        lp = float(peak_label_min_prominence)
    return {
        "peak_label_min_height": lh,
        "peak_label_min_prominence": lp,
        "peak_label_preset": preset or None,
        "peak_label_min_height_overridden": peak_label_min_height is not None,
        "peak_label_min_prominence_overridden": peak_label_min_prominence is not None,
    }


def peak_normalized_absorbance(peak: dict[str, Any]) -> float:
    """Y-value at the picked peak on the processed (0–1) absorbance trace."""
    return float(peak.get("height", 0) or 0)


def peak_passes_label_threshold(
    peak: dict[str, Any],
    *,
    peak_label_min_height: float,
    peak_label_min_prominence: float,
) -> bool:
    h = peak_normalized_absorbance(peak)
    if not math.isfinite(h):
        return False
    prom_rel = _prominence_rel(peak, float(peak.get("_y_range", 1.0) or 1.0))
    return h >= peak_label_min_height and prom_rel >= peak_label_min_prominence


def _peak_near_wavenumber(peak: dict[str, Any], wn: float, tol: float = 14.0) -> bool:
    return abs(float(peak.get("wn_cm1", 0)) - float(wn)) <= tol


def key_evidence_peak_wavenumbers(
    pipeline: dict[str, Any],
    *,
    min_score: float = 0.22,
    tol: float = 14.0,
) -> set[float]:
    """Wavenumbers linked to supported assignments (for label priority B)."""
    out: set[float] = set()
    evidence = pipeline.get("evidence") or {}
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    match_map = {str(m.get("band_id")): m for m in (evidence.get("band_matches") or []) if m.get("matched")}
    for lab, ent in assigns.items():
        if not isinstance(ent, dict):
            continue
        if str(ent.get("ontology_category") or "") in ("local_motif", "artifact"):
            continue
        if float(ent.get("score", 0) or 0) < min_score:
            continue
        band_ids = list(ent.get("supporting_band_ids") or [])
        if not band_ids:
            for raw in ent.get("supporting_bands") or []:
                s = str(raw)
                for m in match_map.values():
                    if str(m.get("band_id", "")) in s or str(m.get("label", "")) in s:
                        band_ids.append(str(m.get("band_id")))
                        break
        for bid in dict.fromkeys(band_ids):
            m = match_map.get(str(bid))
            if not m:
                continue
            for npk in m.get("peaks_near") or []:
                if isinstance(npk, dict):
                    out.add(round(float(npk.get("wn_cm1", 0)), 1))
    for m in match_map.values():
        if float(m.get("support_score", 0) or 0) < 0.35:
            continue
        for npk in m.get("peaks_near") or []:
            if isinstance(npk, dict):
                out.add(round(float(npk.get("wn_cm1", 0)), 1))
    return out


def _region_multiplier(wn_cm1: float) -> float:
    """<1 lenient (easier), >1 strict (harder)."""
    for lo, hi in REGION_LENIENT:
        if lo <= wn_cm1 <= hi:
            return 0.72
    for lo, hi in REGION_STRICT:
        if lo <= wn_cm1 <= hi:
            return 1.38
    return 1.0


def _in_diagnostic_low_height_region(wn_cm1: float) -> bool:
    return _region_multiplier(wn_cm1) < 1.0


def _local_prominence(y: np.ndarray, idx: int) -> float:
    i0 = max(0, idx - 8)
    i1 = min(y.size, idx + 9)
    local = y[i0:i1]
    base = float(np.nanmin(local))
    return max(float(y[idx]) - base, 0.0)


def _prominence_rel(peak: dict[str, Any], y_range: float) -> float:
    prom = float(peak.get("local_prominence", 0) or 0)
    return prom / (y_range + 1e-9)


def intensity_passes(
    peak: dict[str, Any],
    *,
    peak_min_height: float,
    peak_min_prominence: float,
    y_range: float,
) -> bool:
    """
    Retain peak when normalized height AND prominence meet thresholds.
    Diagnostic regions: allow lower height if prominence + isolation are strong.
    """
    rel = float(peak.get("rel_height", 0) or 0)
    prom_rel = _prominence_rel(peak, y_range)
    wn = float(peak.get("wn_cm1", 0))
    isolation = float(peak.get("quality_isolation", 0) or 0)
    width = float(peak.get("quality_width_cm1") or 999)
    sharp = float(peak.get("quality_sharpness", 0) or 0)

    if rel >= peak_min_height and prom_rel >= peak_min_prominence:
        return True

    if not _in_diagnostic_low_height_region(wn):
        return False

    h_floor = max(peak_min_height * 0.55, 0.03)
    p_floor = peak_min_prominence * 0.85
    if prom_rel < p_floor:
        return False
    if isolation < 1.1 and width > 70:
        return False
    if rel >= h_floor and (isolation >= 1.2 or sharp >= 0.03):
        return True
    if rel >= peak_min_height * 0.45 and isolation >= 1.35 and width < 50 and sharp >= 0.025:
        return True
    return False


def pick_spectral_peaks(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    *,
    sensitivity: str | PeakSensitivity = "balanced",
    max_peaks: int = 80,
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
) -> list[dict[str, Any]]:
    """Pick peaks with prominence/height/isolation; assign quality and rule-support role."""
    sens = normalize_peak_sensitivity(sensitivity)
    preset = PEAK_SENSITIVITY_PRESETS[sens]
    thresholds = resolve_peak_thresholds(
        sens, peak_min_height=peak_min_height, peak_min_prominence=peak_min_prominence
    )
    min_h = float(thresholds["peak_min_height"])
    min_prom = float(thresholds["peak_min_prominence"])

    x = np.asarray(wavenumber_cm, dtype=float).reshape(-1)
    y = np.asarray(absorbance, dtype=float).reshape(-1)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if y.size < 8:
        return []

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, 1e-9)

    prom_base = float(preset["prominence_frac"]) * y_range
    height_base = y_min + float(preset["height_frac"]) * y_range
    dist = int(preset["distance_pts"])

    cand_idx, props = find_peaks(
        y,
        prominence=max(prom_base * 0.45, min_prom * y_range * 0.5, 1e-9),
        height=height_base * 0.75,
        distance=max(dist, 1),
    )
    if cand_idx.size == 0:
        return []

    prom_arr = props.get("prominences")
    prom_by_idx: dict[int, float] = {}
    if prom_arr is not None:
        for j, i in enumerate(cand_idx):
            if j < len(prom_arr):
                prom_by_idx[int(i)] = float(prom_arr[j])

    kept: list[int] = []
    for i in cand_idx:
        wn_i = float(x[i])
        mult = _region_multiplier(wn_i)
        prom_req = prom_base * mult
        height_req = y_min + float(preset["height_frac"]) * y_range * mult
        prom_loc = prom_by_idx.get(int(i), _local_prominence(y, int(i)))
        if prom_loc >= prom_req and float(y[i]) >= height_req:
            kept.append(int(i))

    if not kept:
        return []

    scored: list[tuple[float, int]] = []
    for i in kept:
        prom = prom_by_idx.get(i, _local_prominence(y, i))
        scored.append((prom * float(y[i]), i))
    scored.sort(key=lambda t: -t[0])
    kept = [i for _, i in scored[: max(int(max_peaks) * 2, int(max_peaks))]]

    peak_list = [
        {
            "wn_cm1": float(x[i]),
            "height": float(y[i]),
            "rel_height": float(y[i] / (y_max + 1e-9)),
            "local_prominence": prom_by_idx.get(i, _local_prominence(y, i)),
            "_y_range": y_range,
        }
        for i in kept
    ]
    from ml.ftir_evidence import _enrich_peaks_with_quality

    peak_list = _enrich_peaks_with_quality(x, y, peak_list, y_range=y_range)

    qscale = float(preset["quality_scale"])
    filtered: list[dict[str, Any]] = []
    for p in peak_list:
        if not intensity_passes(p, peak_min_height=min_h, peak_min_prominence=min_prom, y_range=y_range):
            continue
        p["peak_sensitivity"] = sens
        p["peak_min_height"] = min_h
        p["peak_min_prominence"] = min_prom
        pq = classify_peak_quality(p, quality_scale=qscale)
        p["peak_quality"] = pq
        p["peak_role"] = assign_peak_role(p, pq)
        p["rule_support_weight"] = rule_support_weight(p)
        filtered.append(p)

    peak_list = filtered
    if sens != "very_sensitive":
        peak_list = [p for p in peak_list if p.get("peak_quality") != "noise_like"]
    peak_list.sort(
        key=lambda p: (
            0 if p.get("peak_role") == "diagnostic_peak" else (1 if p.get("peak_role") == "weak_peak" else 2),
            -float(p.get("rel_height", 0)),
        ),
    )
    return peak_list[: int(max_peaks)]


def classify_peak_quality(
    peak: dict[str, Any],
    *,
    quality_scale: float = 1.0,
) -> PeakQuality:
    """strong | moderate | weak | noise_like (normalized height + shape)."""
    rel = float(peak.get("rel_height", 0) or 0)
    snr = float(peak.get("quality_snr_proxy", 0) or 0)
    sharp = float(peak.get("quality_sharpness", 0) or 0)
    width = float(peak.get("quality_width_cm1") or 999)
    isolation = float(peak.get("quality_isolation", 0) or 0)
    prom_rel = _prominence_rel(peak, float(peak.get("_y_range", 1.0) or 1.0))

    if width > 55 and rel < 0.12:
        return "weak" if rel >= 0.04 else "noise_like"
    if width > 70 and rel < 0.22:
        return "weak" if rel >= 0.05 else "noise_like"
    if width > 95 and sharp < 0.04 and rel < 0.12:
        return "noise_like"
    if width > 120 and rel < 0.08:
        return "noise_like"
    if snr < 0.65 * quality_scale and rel < 0.05 and isolation < 1.1:
        return "noise_like"
    if prom_rel < 0.003 * quality_scale and rel < 0.04:
        return "noise_like"
    if rel < 0.05 and isolation < 1.05:
        return "noise_like"

    if rel >= 0.20 or (prom_rel >= 0.08 and isolation >= 1.35 and rel >= 0.12):
        return "strong"
    if rel >= 0.10 or (snr >= 1.35 and sharp >= 0.03 and rel >= 0.10):
        return "moderate"
    if rel >= 0.05:
        return "weak"
    return "noise_like"


def assign_peak_role(peak: dict[str, Any], quality: PeakQuality | None = None) -> PeakRole:
    q = quality or str(peak.get("peak_quality") or "weak")
    if q == "noise_like":
        return "detected_peak"
    width = float(peak.get("quality_width_cm1") or 999)
    rel = float(peak.get("rel_height", 0) or 0)
    if width > 55 and rel < 0.12:
        return "weak_peak"
    if width > 70 and rel < 0.20:
        return "weak_peak"
    if q == "strong":
        return "diagnostic_peak"
    if q == "moderate":
        return "diagnostic_peak"
    wn = float(peak.get("wn_cm1", 0))
    isolation = float(peak.get("quality_isolation", 0) or 0)
    sharp = float(peak.get("quality_sharpness", 0) or 0)
    in_diag = _in_diagnostic_low_height_region(wn)
    if width > 45 or rel < 0.08:
        return "weak_peak"
    if isolation >= 1.35 and width < 40 and sharp >= 0.03 and in_diag:
        return "diagnostic_peak"
    if in_diag and isolation >= 1.2 and width < 50 and sharp >= 0.025:
        return "diagnostic_peak"
    return "weak_peak"


def rule_support_weight(peak: dict[str, Any]) -> float:
    role = str(peak.get("peak_role") or "detected_peak")
    q = str(peak.get("peak_quality") or "weak")
    rel = float(peak.get("rel_height", 0) or 0)
    if role != "diagnostic_peak":
        return 0.0
    if q == "strong":
        return 1.0
    if q == "moderate":
        return 0.75
    if q == "weak":
        return min(0.35, 0.15 + 0.4 * rel)
    return 0.0


def diagnostic_peaks_only(peaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in peaks if str(p.get("peak_role")) == "diagnostic_peak"]


def peaks_for_display(
    peaks: list[dict[str, Any]],
    *,
    show_weak_peaks: bool = False,
    report_density: str = "balanced",
    include_noise: bool = False,
) -> list[dict[str, Any]]:
    """product_v1: strong/moderate default; weak optional; noise only in audit."""
    dens = str(report_density or "balanced").lower()
    if dens == "audit" or include_noise:
        return list(peaks)
    out: list[dict[str, Any]] = []
    for p in peaks:
        q = str(p.get("peak_quality") or "moderate")
        if q == "noise_like":
            continue
        if q == "weak" and not show_weak_peaks:
            continue
        if q not in ("strong", "moderate", "weak"):
            continue
        out.append(p)
    return out


def _peak_prominence_score(p: dict[str, Any]) -> float:
    prom = float(p.get("local_prominence", 0) or 0)
    yr = float(p.get("_y_range", 1.0) or 1.0)
    rel_prom = prom / (yr + 1e-9)
    rh = float(p.get("rel_height", p.get("height", 0)) or 0)
    return max(rel_prom, rh * 0.5)


def compute_peak_labeling(
    peaks: list[dict[str, Any]],
    *,
    show_weak_peaks: bool = False,
    max_peak_labels: int = 24,
    label_all_diagnostic: bool = False,
    report_density: str = "balanced",
    report_audience: str = "front",
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
    peak_label_min_height: float | None = None,
    peak_label_min_prominence: float | None = None,
    peak_label_preset: str | None = None,
    key_evidence_wn: set[float] | None = None,
    label_all_above_height: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Select peaks for numeric labels and return diagnostics (unlabeled reason counts).
    """
    meta: dict[str, Any] = {
        "detected_peaks_count": len(peaks),
        "displayed_peaks_count": 0,
        "labeled_peaks_count": 0,
        "unlabeled_reason_counts": {},
    }
    unlabeled: dict[str, int] = {}

    def _bump(reason: str, n: int = 1) -> None:
        unlabeled[reason] = unlabeled.get(reason, 0) + n

    if label_all_above_height is not None:
        threshold = float(label_all_above_height)
        pool = [
            p
            for p in peaks
            if peak_normalized_absorbance(p) >= threshold
            and str(p.get("peak_quality") or "") != "noise_like"
        ]
        not_displayed = len(peaks) - len(pool)
        if not_displayed:
            _bump("not_displayed", not_displayed)
        for p in peaks:
            if str(p.get("peak_quality") or "") == "noise_like":
                _bump("noise_like")
            elif peak_normalized_absorbance(p) < threshold:
                _bump("below_label_height")
        pool.sort(key=lambda p: (-peak_normalized_absorbance(p), -float(p.get("wn_cm1", 0))))
        labeled = []
        for p in pool:
            p2 = dict(p)
            p2["label_reason"] = "above_height"
            labeled.append(p2)
        meta["displayed_peaks_count"] = len(pool)
        meta["labeled_peaks_count"] = len(labeled)
        meta["unlabeled_reason_counts"] = unlabeled
        return labeled, meta

    label_th = resolve_peak_label_thresholds(
        report_audience,
        peak_min_height=peak_min_height,
        peak_min_prominence=peak_min_prominence,
        peak_label_min_height=peak_label_min_height,
        peak_label_min_prominence=peak_label_min_prominence,
        peak_label_preset=peak_label_preset,
    )
    lh = float(label_th["peak_label_min_height"])
    lp = float(label_th["peak_label_min_prominence"])
    preset = normalize_peak_label_preset(peak_label_preset)
    plotted = peaks_for_display(
        peaks,
        show_weak_peaks=show_weak_peaks,
        report_density=report_density,
        include_noise=str(report_density or "").lower() == "audit",
    )
    meta["displayed_peaks_count"] = len(plotted)
    if not plotted:
        for p in peaks:
            if str(p.get("peak_quality") or "") == "noise_like":
                _bump("noise_like")
            else:
                _bump("not_displayed")
        meta["unlabeled_reason_counts"] = unlabeled
        return [], meta

    plotted_ids = {id(p) for p in plotted}
    for p in peaks:
        if id(p) in plotted_ids:
            continue
        if str(p.get("peak_quality") or "") == "noise_like":
            _bump("noise_like")
        else:
            _bump("not_displayed")

    dens = str(report_density or "balanced").lower()
    aud = str(report_audience or "debug").lower()
    is_audit = dens == "audit" or aud == "debug"
    key_wn = key_evidence_wn or set()
    cap = max(1, int(max_peak_labels))
    labeled: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _add(p: dict[str, Any], reason: str) -> bool:
        if id(p) in seen_ids:
            return True
        if len(labeled) >= cap:
            return False
        p2 = dict(p)
        p2["label_reason"] = reason
        labeled.append(p2)
        seen_ids.add(id(p))
        return True

    if is_audit and show_weak_peaks:
        for p in sorted(plotted, key=_peak_prominence_score, reverse=True):
            if str(p.get("peak_role")) in ("diagnostic_peak", "weak_peak"):
                _add(p, "audit")
        if label_all_diagnostic:
            for p in plotted:
                _add(p, "forced")
        meta["labeled_peaks_count"] = len(labeled)
        for p in plotted:
            if id(p) not in seen_ids:
                _bump("max_labels")
        meta["unlabeled_reason_counts"] = unlabeled
        return labeled[:cap], meta

    if label_all_diagnostic or preset == "all_visible":
        for p in sorted(plotted, key=_peak_prominence_score, reverse=True):
            if not _add(p, "all_visible" if preset == "all_visible" else "forced"):
                _bump("max_labels")
        meta["labeled_peaks_count"] = len(labeled)
        meta["unlabeled_reason_counts"] = unlabeled
        return labeled[:cap], meta

    for p in plotted:
        wn = float(p.get("wn_cm1", 0))
        if not math.isfinite(wn) or not math.isfinite(peak_normalized_absorbance(p)):
            _bump("nonfinite")
            continue
        if any(abs(wn - kw) <= 14.0 for kw in key_wn):
            _add(p, "key_evidence")

    for p in plotted:
        if id(p) in seen_ids:
            continue
        if peak_passes_label_threshold(p, peak_label_min_height=lh, peak_label_min_prominence=lp):
            if not _add(p, "height_prominence"):
                _bump("max_labels")
            continue
        role = str(p.get("peak_role") or "")
        if role == "diagnostic_peak" and _in_diagnostic_low_height_region(float(p.get("wn_cm1", 0))):
            prom_rel = _prominence_rel(p, float(p.get("_y_range", 1.0) or 1.0))
            rel = peak_normalized_absorbance(p)
            if prom_rel >= lp * 0.85 and rel >= lh * 0.55:
                if not _add(p, "diagnostic"):
                    _bump("max_labels")
                continue
        h = peak_normalized_absorbance(p)
        prom_rel = _prominence_rel(p, float(p.get("_y_range", 1.0) or 1.0))
        if h < lh:
            _bump("below_label_height")
        elif prom_rel < lp:
            _bump("below_label_prominence")

    pool = [p for p in plotted if id(p) not in seen_ids]
    pool.sort(key=_peak_prominence_score, reverse=True)
    for p in pool:
        if len(labeled) >= cap:
            _bump("max_labels", len(pool) - sum(1 for q in pool if id(q) in seen_ids))
            break
        if is_audit or peak_passes_label_threshold(
            p, peak_label_min_height=lh, peak_label_min_prominence=lp
        ):
            if not _add(p, "height_prominence"):
                _bump("max_labels")
        else:
            h = peak_normalized_absorbance(p)
            prom_rel = _prominence_rel(p, float(p.get("_y_range", 1.0) or 1.0))
            if h < lh:
                _bump("below_label_height")
            elif prom_rel < lp:
                _bump("below_label_prominence")

    meta["labeled_peaks_count"] = len(labeled)
    meta["unlabeled_reason_counts"] = unlabeled
    meta["label_thresholds"] = label_th
    return labeled[:cap], meta


def peaks_for_label(
    peaks: list[dict[str, Any]],
    *,
    show_weak_peaks: bool = False,
    max_peak_labels: int = 24,
    label_all_diagnostic: bool = False,
    report_density: str = "balanced",
    report_audience: str = "front",
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
    peak_label_min_height: float | None = None,
    peak_label_min_prominence: float | None = None,
    peak_label_preset: str | None = None,
    key_evidence_wn: set[float] | None = None,
    label_all_above_height: float | None = None,
) -> list[dict[str, Any]]:
    """Peaks that receive numeric cm⁻¹ labels on the spectrum."""
    labeled, _meta = compute_peak_labeling(
        peaks,
        show_weak_peaks=show_weak_peaks,
        max_peak_labels=max_peak_labels,
        label_all_diagnostic=label_all_diagnostic,
        report_density=report_density,
        report_audience=report_audience,
        peak_min_height=peak_min_height,
        peak_min_prominence=peak_min_prominence,
        peak_label_min_height=peak_label_min_height,
        peak_label_min_prominence=peak_label_min_prominence,
        peak_label_preset=peak_label_preset,
        key_evidence_wn=key_evidence_wn,
        label_all_above_height=label_all_above_height,
    )
    return labeled


def summarize_peak_display(
    peaks: list[dict[str, Any]],
    *,
    show_weak_peaks: bool = False,
    max_peak_labels: int = 24,
    label_all_diagnostic: bool = False,
    report_density: str = "balanced",
    report_audience: str = "front",
    peak_label_min_height: float | None = None,
    peak_label_min_prominence: float | None = None,
    key_evidence_wn: set[float] | None = None,
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
    peak_sensitivity: str = "balanced",
    label_all_above_height: float | None = None,
    peak_label_preset: str | None = None,
) -> dict[str, Any]:
    """Counts and thresholds for report metadata: detected vs plotted vs labeled."""
    label_th = resolve_peak_label_thresholds(
        report_audience,
        peak_min_height=peak_min_height,
        peak_min_prominence=peak_min_prominence,
        peak_label_min_height=peak_label_min_height,
        peak_label_min_prominence=peak_label_min_prominence,
        peak_label_preset=peak_label_preset,
    )
    labeled, label_meta = compute_peak_labeling(
        peaks,
        show_weak_peaks=show_weak_peaks,
        max_peak_labels=max_peak_labels,
        label_all_diagnostic=label_all_diagnostic,
        report_density=report_density,
        report_audience=report_audience,
        peak_min_height=peak_min_height,
        peak_min_prominence=peak_min_prominence,
        peak_label_min_height=label_th["peak_label_min_height"],
        peak_label_min_prominence=label_th["peak_label_min_prominence"],
        peak_label_preset=peak_label_preset,
        key_evidence_wn=key_evidence_wn,
        label_all_above_height=label_all_above_height,
    )
    out: dict[str, Any] = {
        "n_detected_peaks": label_meta.get("detected_peaks_count", len(peaks)),
        "n_plotted_peaks": label_meta.get("displayed_peaks_count", 0),
        "n_labeled_peaks": label_meta.get("labeled_peaks_count", len(labeled)),
        "detected_peaks_count": label_meta.get("detected_peaks_count", len(peaks)),
        "displayed_peaks_count": label_meta.get("displayed_peaks_count", 0),
        "labeled_peaks_count": label_meta.get("labeled_peaks_count", len(labeled)),
        **label_th,
    }
    if label_meta.get("unlabeled_reason_counts"):
        out["unlabeled_reason_counts"] = label_meta["unlabeled_reason_counts"]
    if peak_min_height is not None:
        out["peak_min_height"] = float(peak_min_height)
    if peak_min_prominence is not None:
        out["peak_min_prominence"] = float(peak_min_prominence)
    if peak_sensitivity:
        out["peak_sensitivity"] = peak_sensitivity
    reasons: dict[str, int] = {}
    for p in labeled:
        r = str(p.get("label_reason") or "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    if reasons:
        out["label_reason_counts"] = reasons
    if label_all_above_height is not None:
        out["label_all_above_height"] = float(label_all_above_height)
    return out


def peaks_for_kronecker(
    peaks: list[dict[str, Any]],
    *,
    show_weak_peaks: bool = False,
    report_density: str = "balanced",
) -> list[dict[str, Any]]:
    displayed = peaks_for_display(
        peaks, show_weak_peaks=show_weak_peaks, report_density=report_density
    )
    if str(report_density or "").lower() == "audit":
        return displayed
    return [p for p in displayed if str(p.get("peak_role")) in ("diagnostic_peak", "weak_peak")]


def pick_peaks_simple_compat(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    max_peaks: int = 40,
    *,
    sensitivity: str = "balanced",
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
) -> tuple[list[float], list[float]]:
    peaks = pick_spectral_peaks(
        wavenumber_cm,
        absorbance,
        sensitivity=sensitivity,
        max_peaks=max_peaks,
        peak_min_height=peak_min_height,
        peak_min_prominence=peak_min_prominence,
    )
    return [float(p["wn_cm1"]) for p in peaks], [float(p["height"]) for p in peaks]
