"""
Two-stage paper peak picking: generous detection on normalized absorbance, then
region-quota label selection with scoring and optional overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences, peak_widths

from reports.paper_peak_overrides import PaperPeakOverrides

PAPER_LABEL_REGIONS: list[dict[str, Any]] = [
    {
        "id": "oh_nh",
        "lo": 3000.0,
        "hi": 3700.0,
        "title": "3700–3000 cm⁻¹ (O–H / N–H stretch)",
        "max_labels": 2,
        "min_spacing": 80.0,
    },
    {
        "id": "ch",
        "lo": 2800.0,
        "hi": 3000.0,
        "title": "3000–2800 cm⁻¹ (C–H stretch)",
        "max_labels": 2,
        "min_spacing": 80.0,
    },
    {
        "id": "co_aromatic",
        "lo": 1500.0,
        "hi": 1800.0,
        "title": "1800–1500 cm⁻¹ (C=O / aromatic / N–H bend)",
        "max_labels": 3,
        "min_spacing": 60.0,
    },
    {
        "id": "ring_cn",
        "lo": 1200.0,
        "hi": 1500.0,
        "title": "1500–1200 cm⁻¹ (ring / C–N)",
        "max_labels": 3,
        "min_spacing": 60.0,
    },
    {
        "id": "co_fp",
        "lo": 900.0,
        "hi": 1200.0,
        "title": "1200–900 cm⁻¹ (C–O / fingerprint)",
        "max_labels": 3,
        "min_spacing": 55.0,
    },
]

DEFAULT_IGNORE_LABEL_RANGES: list[tuple[float, float]] = [(400.0, 900.0)]


@dataclass
class PaperPeakSelectionConfig:
    min_prominence: float = 0.04
    min_height: float = 0.05
    min_distance_cm1: float = 20.0
    max_labels: int = 10
    ignore_label_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_IGNORE_LABEL_RANGES)
    )
    transmittance_match_cm1: float = 25.0
    use_shoulder_detection: bool = False


@dataclass
class PaperPeakSelectionResult:
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    suppressed: list[dict[str, Any]]
    selected_for_plot: list[dict[str, Any]]


def classify_label_region(wn: float) -> str | None:
    w = float(wn)
    for spec in PAPER_LABEL_REGIONS:
        lo, hi = float(spec["lo"]), float(spec["hi"])
        if spec["id"] == "oh_nh":
            if lo <= w <= hi:
                return str(spec["id"])
        elif spec["id"] == "ch":
            if lo <= w < hi:
                return str(spec["id"])
        elif spec["id"] == "co_aromatic":
            if lo <= w <= hi:
                return str(spec["id"])
        elif spec["id"] == "ring_cn":
            if lo <= w < hi:
                return str(spec["id"])
        elif spec["id"] == "co_fp":
            if lo <= w < hi:
                return str(spec["id"])
    return None


def region_title(region_id: str | None) -> str:
    if not region_id:
        return "unassigned"
    for spec in PAPER_LABEL_REGIONS:
        if spec["id"] == region_id:
            return str(spec["title"])
    return str(region_id)


def _in_range(wn: float, lo: float, hi: float) -> bool:
    return min(lo, hi) <= float(wn) <= max(lo, hi)


def _in_any_range(wn: float, ranges: list[tuple[float, float]]) -> bool:
    return any(_in_range(wn, lo, hi) for lo, hi in ranges)


def _distance_pts(wn: np.ndarray, min_distance_cm1: float) -> int:
    if wn.size < 2:
        return 1
    step = float(np.median(np.abs(np.diff(wn))))
    if step <= 0:
        return 1
    return max(1, int(round(min_distance_cm1 / step)))


def _interp_y(wn: np.ndarray, y: np.ndarray, wn_query: float) -> float:
    if wn.size < 2:
        return float(y[0]) if y.size else 0.0
    f = interp1d(wn, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    return float(f(wn_query))


def _isolation_bonus(wn: float, prominence: float, peaks: list[dict[str, Any]]) -> float:
    bonus = 1.0
    for other in peaks:
        ow = float(other["wavenumber_cm1"])
        op = float(other.get("prominence", 0))
        if abs(ow - wn) <= 30.0 and op > prominence * 1.15:
            bonus -= 0.35
    return max(0.0, min(1.0, bonus))


def compute_peak_score(prominence: float, height: float, isolation: float) -> float:
    return 0.55 * float(prominence) + 0.30 * float(height) + 0.15 * float(isolation)


def detect_candidate_peaks(
    wn: np.ndarray,
    y_norm: np.ndarray,
    *,
    config: PaperPeakSelectionConfig,
) -> list[dict[str, Any]]:
    wn = np.asarray(wn, dtype=float)
    y = np.asarray(y_norm, dtype=float)
    if wn.size < 5:
        return []

    dist = _distance_pts(wn, config.min_distance_cm1)
    idx, props = find_peaks(
        y,
        prominence=config.min_prominence,
        height=config.min_height,
        distance=dist,
    )
    prom_arr = peak_prominences(y, idx)[0] if idx.size else np.array([])
    candidates: list[dict[str, Any]] = []
    for i, peak_idx in enumerate(idx):
        wn_v = float(wn[peak_idx])
        height = float(y[peak_idx])
        prom = float(prom_arr[i]) if i < prom_arr.size else float(height)
        region = classify_label_region(wn_v)
        iso = _isolation_bonus(wn_v, prom, [])
        score = compute_peak_score(prom, height, iso)
        candidates.append(
            {
                "wavenumber_cm1": wn_v,
                "intensity": height,
                "height": height,
                "prominence": prom,
                "region": region or "",
                "region_title": region_title(region),
                "score": score,
                "detected": True,
                "label_selected": False,
                "reason_not_selected": "",
                "required": False,
                "preferred": False,
            }
        )

    for c in candidates:
        c["score"] = compute_peak_score(
            c["prominence"],
            c["height"],
            _isolation_bonus(c["wavenumber_cm1"], c["prominence"], candidates),
        )

    if config.use_shoulder_detection and candidates:
        widths = peak_widths(y, idx, rel_height=0.5)[0] if idx.size else np.array([])
        for i, c in enumerate(candidates):
            if i >= widths.size:
                break
            width_cm1 = float(widths[i]) * float(np.median(np.abs(np.diff(wn))))
            if width_cm1 >= 45.0 and float(c["prominence"]) >= config.min_prominence * 0.85:
                c["shoulder_like"] = True

    candidates.sort(key=lambda p: -float(p["score"]))
    return candidates


def _nearest_candidate(
    wn_target: float,
    candidates: list[dict[str, Any]],
    *,
    tolerance: float = 18.0,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_d = 1e9
    for c in candidates:
        d = abs(float(c["wavenumber_cm1"]) - wn_target)
        if d <= tolerance and d < best_d:
            best = c
            best_d = d
    return best


def _synthetic_candidate(
    wn: np.ndarray,
    y: np.ndarray,
    wn_target: float,
    *,
    required: bool = False,
    preferred: bool = False,
) -> dict[str, Any]:
    height = _interp_y(wn, y, wn_target)
    prom = max(height * 0.5, 0.02)
    region = classify_label_region(wn_target)
    return {
        "wavenumber_cm1": float(wn_target),
        "intensity": height,
        "height": height,
        "prominence": prom,
        "region": region or "",
        "region_title": region_title(region),
        "score": compute_peak_score(prom, height, 1.0),
        "detected": False,
        "label_selected": False,
        "reason_not_selected": "",
        "required": required,
        "preferred": preferred,
    }


def _spacing_ok(selected: list[dict[str, Any]], wn: float, min_spacing: float) -> bool:
    return all(abs(float(s["wavenumber_cm1"]) - wn) >= min_spacing for s in selected)


def select_labeled_peaks(
    wn: np.ndarray,
    y_norm: np.ndarray,
    *,
    config: PaperPeakSelectionConfig,
    overrides: PaperPeakOverrides | None = None,
) -> PaperPeakSelectionResult:
    ov = overrides or PaperPeakOverrides()
    ignore_ranges = list(config.ignore_label_ranges) + list(ov.suppress_ranges)
    candidates = detect_candidate_peaks(wn, y_norm, config=config)
    cand_by_wn = {round(float(c["wavenumber_cm1"]), 1): c for c in candidates}

    for req in ov.required_peaks:
        near = _nearest_candidate(req, candidates)
        if near:
            near["required"] = True
        else:
            syn = _synthetic_candidate(wn, y_norm, float(req), required=True)
            candidates.append(syn)
            cand_by_wn[round(float(req), 1)] = syn

    for pref in ov.preferred_peaks:
        near = _nearest_candidate(pref, candidates, tolerance=22.0)
        if near:
            near["preferred"] = True
            near["score"] = float(near["score"]) + 0.08
        else:
            syn = _synthetic_candidate(wn, y_norm, float(pref), preferred=True)
            candidates.append(syn)

    for sup in ov.suppressed_peaks:
        near = _nearest_candidate(sup, candidates, tolerance=15.0)
        if near:
            near["manually_suppressed"] = True

    for c in candidates:
        wn_v = float(c["wavenumber_cm1"])
        if _in_any_range(wn_v, ignore_ranges) and not c.get("required"):
            c["reason_not_selected"] = "outside labeled region"
        elif c.get("manually_suppressed"):
            c["reason_not_selected"] = "manually suppressed"
        elif not c.get("region") and not c.get("required"):
            c["reason_not_selected"] = "outside labeled region"

    selected: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []

    for spec in PAPER_LABEL_REGIONS:
        rid = str(spec["id"])
        max_labels = int(ov.region_quota_overrides.get(rid, spec["max_labels"]))
        min_spacing = float(spec["min_spacing"])
        lo, hi = float(spec["lo"]), float(spec["hi"])

        region_cands = [
            c
            for c in candidates
            if c.get("region") == rid
            and not c.get("reason_not_selected")
            and not c.get("manually_suppressed")
        ]
        region_cands.sort(
            key=lambda c: (
                -int(bool(c.get("required"))),
                -int(bool(c.get("preferred"))),
                -float(c["score"]),
            )
        )

        region_selected: list[dict[str, Any]] = []
        for c in region_cands:
            wn_v = float(c["wavenumber_cm1"])
            if not (lo <= wn_v <= hi or (rid == "ch" and lo <= wn_v < hi)):
                c["reason_not_selected"] = "outside labeled region"
                continue
            if len(region_selected) >= max_labels:
                c["reason_not_selected"] = "region quota exceeded"
                continue
            if not _spacing_ok(region_selected, wn_v, min_spacing):
                c["reason_not_selected"] = "crowded"
                continue
            c["label_selected"] = True
            region_selected.append(c)
            selected.append(c)

        for c in region_cands:
            if not c.get("label_selected") and not c.get("reason_not_selected"):
                c["reason_not_selected"] = "below threshold"

    required_unassigned = [
        c
        for c in candidates
        if c.get("required")
        and not c.get("label_selected")
        and not c.get("manually_suppressed")
    ]
    for c in required_unassigned:
        if len(selected) >= int(config.max_labels) and not c.get("required"):
            break
        if not _spacing_ok(selected, float(c["wavenumber_cm1"]), 25.0):
            c["reason_not_selected"] = "crowded"
            continue
        c["label_selected"] = True
        c["reason_not_selected"] = ""
        selected.append(c)

    if len(selected) > int(config.max_labels):
        selected.sort(
            key=lambda c: (
                -int(bool(c.get("required"))),
                -int(bool(c.get("preferred"))),
                -float(c["score"]),
            )
        )
        keep = selected[: int(config.max_labels)]
        drop = selected[int(config.max_labels) :]
        for c in drop:
            if c.get("required"):
                keep.append(c)
            else:
                c["label_selected"] = False
                c["reason_not_selected"] = "region quota exceeded"
        selected = keep

    selected_wn = {round(float(c["wavenumber_cm1"]), 1) for c in selected}
    for c in candidates:
        if c.get("label_selected"):
            continue
        reason = str(c.get("reason_not_selected") or "below threshold")
        c["reason_not_selected"] = reason
        suppressed.append(
            {
                **c,
                "reason_suppressed": reason,
            }
        )

    selected.sort(key=lambda c: -float(c["wavenumber_cm1"]))
    plot_peaks: list[dict[str, Any]] = []
    for c in selected:
        wn_v = float(c["wavenumber_cm1"])
        plot_peaks.append(
            {
                "wn": wn_v,
                "wn_cm1": wn_v,
                "y": float(c["intensity"]),
                "height": float(c["intensity"]),
                "prominence": float(c["prominence"]),
                "text": f"{wn_v:.0f}",
                "score": float(c["score"]),
                "region": c.get("region", ""),
            }
        )

    for c in candidates:
        c["label_selected"] = round(float(c["wavenumber_cm1"]), 1) in selected_wn

    return PaperPeakSelectionResult(
        candidates=candidates,
        selected=selected,
        suppressed=suppressed,
        selected_for_plot=plot_peaks,
    )


def match_transmittance_minima(
    wn_t: np.ndarray,
    y_pct: np.ndarray,
    selected_absorbance_peaks: list[dict[str, Any]],
    *,
    match_window_cm1: float = 25.0,
    min_dip_depth_pct: float = 0.35,
    baseline_pct: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_pct = np.asarray(y_pct, dtype=float)
    wn_t = np.asarray(wn_t, dtype=float)
    for peak in selected_absorbance_peaks:
        abs_wn = float(peak.get("wavenumber_cm1", peak.get("wn", 0)))
        lo = abs_wn - match_window_cm1
        hi = abs_wn + match_window_cm1
        mask = (wn_t >= lo) & (wn_t <= hi)
        if not np.any(mask):
            rows.append(
                {
                    "absorbance_peak_cm1": f"{abs_wn:.1f}",
                    "transmittance_label_cm1": "",
                    "transmittance_value": "",
                    "matched_within_cm1": "",
                    "label_shown": "no",
                    "note": "no transmittance data in match window",
                }
            )
            continue
        local_wn = wn_t[mask]
        local_y = y_pct[mask]
        idx = int(np.argmin(local_y))
        t_wn = float(local_wn[idx])
        t_val = float(local_y[idx])
        matched = abs(t_wn - abs_wn)
        if baseline_pct is not None:
            ref = float(baseline_pct)
            dip_depth = ref - t_val
            clear_dip = t_val < ref and dip_depth >= min_dip_depth_pct
            weak_note = "above baseline or insufficient dip below baseline"
        else:
            local_max = float(np.max(local_y))
            dip_depth = local_max - t_val
            clear_dip = dip_depth >= min_dip_depth_pct
            weak_note = "no clear transmittance dip"
        rows.append(
            {
                "absorbance_peak_cm1": f"{abs_wn:.1f}",
                "transmittance_label_cm1": f"{t_wn:.1f}" if clear_dip else "",
                "transmittance_value": f"{t_val:.2f}" if clear_dip else f"{t_val:.2f}",
                "matched_within_cm1": f"{matched:.1f}",
                "label_shown": "yes" if clear_dip else "no",
                "note": "" if clear_dip else weak_note,
                "wn": t_wn if clear_dip else abs_wn,
                "y": t_val if clear_dip else _interp_y(wn_t, y_pct, abs_wn),
                "text": f"{t_wn:.0f}" if clear_dip else f"{abs_wn:.0f}",
                "prominence": dip_depth if clear_dip else 0.0,
            }
        )
    return rows


def parse_ignore_label_ranges(spec: str | list[str] | None) -> list[tuple[float, float]]:
    if not spec:
        return list(DEFAULT_IGNORE_LABEL_RANGES)
    items = spec if isinstance(spec, list) else [spec]
    ranges: list[tuple[float, float]] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if ":" in text:
            a, b = text.split(":", 1)
        elif "-" in text:
            a, b = text.split("-", 1)
        else:
            continue
        try:
            lo, hi = float(a), float(b)
            ranges.append((min(lo, hi), max(lo, hi)))
        except ValueError:
            continue
    return ranges or list(DEFAULT_IGNORE_LABEL_RANGES)
