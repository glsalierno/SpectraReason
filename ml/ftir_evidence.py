"""
Spectral evidence extraction for evidence-first FTIR functional-group assignment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lib.peaks import find_peaks_simple
from ml.ftir_band_library import BAND_LIBRARY, load_band_library
from ml.ftir_interpretable_features import (
    INTERPRETABLE_REGIONS,
    _segment_stats,
    oh_broadness_metric,
    peaks_near_band,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "max_peaks": 80,
    "peak_sensitivity": "balanced",
    "peak_min_height": None,
    "peak_min_prominence": None,
    "peak_match_tolerance_cm1": 25.0,
    "region_signal_frac": 0.12,
    "min_region_points": 3,
}


def _norm_signal(value: float, y_max: float) -> float:
    if y_max <= 1e-12:
        return 0.0
    return float(np.clip(value / y_max, 0.0, 1.5))


def _enrich_peaks_with_quality(
    wn: np.ndarray,
    y: np.ndarray,
    peak_list: list[dict[str, Any]],
    *,
    y_range: float,
    window_cm1: float = 18.0,
) -> list[dict[str, Any]]:
    """
    Per-peak proxies: SNR-like, sharpness, width, isolation, local baseline stability.
    Used by v3 guardrails (nitrile / alkyne sharpness, noise rejection).
    """
    if not peak_list or wn.size < 8:
        return peak_list
    wn = np.asarray(wn, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    noise = float(np.nanstd(np.diff(y))) + 1e-9
    out: list[dict[str, Any]] = []
    for p in peak_list:
        row = dict(p)
        pw = float(row.get("wn_cm1", 0))
        idx = int(np.argmin(np.abs(wn - pw)))
        i0 = max(0, idx - 4)
        i1 = min(wn.size, idx + 5)
        local = y[i0:i1]
        base = float(np.nanmin(local))
        pk = float(row.get("height", local[min(idx - i0, local.size - 1)]))
        prominence = max(pk - base, 0.0)
        snr_proxy = float(prominence / (noise * y_range + 1e-9))

        w_lo = pw - window_cm1
        w_hi = pw + window_cm1
        m = (wn >= w_lo) & (wn <= w_hi)
        if int(np.count_nonzero(m)) < 5:
            row.update(
                {
                    "quality_snr_proxy": round(snr_proxy, 4),
                    "quality_sharpness": 0.0,
                    "quality_width_cm1": None,
                    "quality_isolation": 0.0,
                    "quality_baseline_stability": 0.0,
                    "quality_overlap_density": 0.0,
                }
            )
            out.append(row)
            continue
        seg = y[m]
        wseg = wn[m]
        apex = int(np.argmax(seg))
        hm = base + 0.5 * (float(np.max(seg)) - base)
        left = apex
        while left > 0 and seg[left] > hm:
            left -= 1
        right = apex
        while right < seg.size - 1 and seg[right] > hm:
            right += 1
        width_cm1 = float(wseg[right] - wseg[left]) if right > left else float(wseg[-1] - wseg[0])
        sharpness = float((np.max(seg) - base) / (width_cm1 + 1e-6))

        # isolation: peak height vs local shoulder max outside half-width
        half = max(2, (right - left) // 2)
        neigh = np.concatenate([seg[: max(0, left - half)], seg[min(seg.size, right + half) :]])
        neigh_max = float(np.max(neigh)) if neigh.size else base
        isolation = float((np.max(seg) - base) / (max(neigh_max - base, 1e-9) + 1e-9))

        # baseline stability: linear drift removed variance
        coef = np.polyfit(wseg - pw, seg, 1)
        resid = seg - np.polyval(coef, wseg - pw)
        stability = 1.0 / (float(np.nanstd(resid)) + 1e-6)

        row.update(
            {
                "quality_snr_proxy": round(snr_proxy, 4),
                "quality_sharpness": round(sharpness, 6),
                "quality_width_cm1": round(width_cm1, 2),
                "quality_isolation": round(min(isolation, 12.0), 4),
                "quality_baseline_stability": round(min(stability / 50.0, 1.5), 4),
            }
        )
        out.append(row)
    if len(out) >= 2:
        positions = [float(r.get("wn_cm1", 0.0)) for r in out]
        npos = len(positions)
        win = 35.0
        for i, row in enumerate(out):
            pw = positions[i]
            n_near = sum(1 for q in positions if abs(q - pw) <= win)
            row["quality_overlap_density"] = round((n_near - 1) / max(npos - 1, 1), 4)
    elif out:
        out[0]["quality_overlap_density"] = 0.0
    return out


def extract_spectral_evidence(
    wavenumber: np.ndarray,
    absorbance: np.ndarray,
    peaks: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract JSON-serializable spectral evidence: peaks, regions, band matches, ratios.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    wn = np.asarray(wavenumber, dtype=float).reshape(-1)
    y = np.asarray(absorbance, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]
    y_max = float(np.nanmax(y)) if y.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    y_range = max(y_max - y_min, 1e-9)

    from ml.ftir_peak_picking import (
        PEAK_SENSITIVITY_PRESETS,
        assign_peak_role,
        classify_peak_quality,
        normalize_peak_sensitivity,
        pick_spectral_peaks,
        resolve_peak_thresholds,
        rule_support_weight,
    )

    sens = normalize_peak_sensitivity(cfg.get("peak_sensitivity"))
    pmin_h = cfg.get("peak_min_height")
    pmin_p = cfg.get("peak_min_prominence")
    thresholds = resolve_peak_thresholds(
        sens,
        peak_min_height=float(pmin_h) if pmin_h is not None else None,
        peak_min_prominence=float(pmin_p) if pmin_p is not None else None,
    )
    if peaks is None:
        peak_list = pick_spectral_peaks(
            wn,
            y,
            sensitivity=sens,
            max_peaks=int(cfg.get("max_peaks", 80)),
            peak_min_height=float(pmin_h) if pmin_h is not None else None,
            peak_min_prominence=float(pmin_p) if pmin_p is not None else None,
        )
    else:
        peak_list = []
        for p in peaks:
            h = float(p.get("height", p.get("rel_height", 0)))
            row = {
                "wn_cm1": float(p.get("wn_cm1", p.get("wn", 0))),
                "height": h,
                "rel_height": _norm_signal(h, y_max),
            }
            for k in (
                "peak_quality",
                "peak_role",
                "rule_support_weight",
                "quality_snr_proxy",
                "quality_sharpness",
                "quality_width_cm1",
                "quality_isolation",
                "local_prominence",
            ):
                if k in p:
                    row[k] = p[k]
            peak_list.append(row)
        if not any(p.get("peak_quality") for p in peak_list):
            peak_list = _enrich_peaks_with_quality(wn, y, peak_list, y_range=y_range)
            preset = PEAK_SENSITIVITY_PRESETS[sens]
            qscale = float(preset["quality_scale"])
            for p in peak_list:
                p["_y_range"] = y_range
                pq = classify_peak_quality(p, quality_scale=qscale)
                p["peak_quality"] = pq
                p["peak_role"] = assign_peak_role(p, pq)
                p["rule_support_weight"] = rule_support_weight(p)

    regions: dict[str, dict[str, Any]] = {}
    for rname, lo, hi in INTERPRETABLE_REGIONS:
        st = _segment_stats(wn, y, float(lo), float(hi))
        regions[rname] = {
            "cm1_min": lo,
            "cm1_max": hi,
            "mean": st["mean"],
            "std": st["std"],
            "max": st["max"],
            "integral": st["integral"],
            "rel_max": _norm_signal(st["max"], y_max),
            "n_points": int(st["n_points"]),
        }

    oh_broad = oh_broadness_metric(wn, y)
    regions["oh_nh_broad"]["broadness"] = oh_broad

    band_matches: list[dict[str, Any]] = []
    tol = float(cfg["peak_match_tolerance_cm1"])
    sig_frac = float(cfg["region_signal_frac"])
    library = load_band_library(prefer_python=True)

    for band in library:
        lo = float(band["region_min_cm1"])
        hi = float(band["region_max_cm1"])
        near_all = peaks_near_band(peak_list, lo, hi, tolerance_cm1=tol)
        near_diag = [
            p
            for p in near_all
            if str(p.get("peak_role", "diagnostic_peak")) == "diagnostic_peak"
        ]
        m = (wn >= lo) & (wn <= hi)
        region_max = float(np.max(y[m])) if int(np.count_nonzero(m)) >= int(cfg["min_region_points"]) else 0.0
        rel_max = _norm_signal(region_max, y_max)
        peak_support = max(
            (
                float(p.get("rel_height", 0))
                * float(p.get("rule_support_weight", 1.0) or 0.0)
                for p in near_diag
            ),
            default=0.0,
        )
        support_score = max(rel_max, peak_support)
        matched = support_score >= sig_frac
        peak_qualities = [str(p.get("peak_quality", "")) for p in near_all if p.get("peak_quality")]
        band_matches.append(
            {
                "band_id": band["id"],
                "label": band["label"],
                "subclass": band.get("subclass", band["label"]),
                "region_min_cm1": lo,
                "region_max_cm1": hi,
                "mode": band["mode"],
                "importance": band["importance"],
                "specificity": band["specificity"],
                "region_rel_max": rel_max,
                "peak_support": peak_support,
                "support_score": support_score,
                "matched": matched,
                "peaks_near": near_all[:6],
                "diagnostic_peaks_near": near_diag[:6],
                "peak_quality": peak_qualities[0] if peak_qualities else "",
            }
        )

    def _r(num: str, den: str) -> float:
        a = regions.get(num, {}).get("rel_max", 0.0)
        b = regions.get(den, {}).get("rel_max", 0.0)
        return float(a / (b + 1e-9))

    ratios = {
        "oh_nh_to_fingerprint": _r("oh_nh_broad", "fingerprint"),
        "carbonyl_to_fingerprint": _r("carbonyl", "fingerprint"),
        "aromatic_to_ch_stretch": _r("aromatic_cc", "ch_stretch"),
        "siloxane_to_c_o": _r("si_o", "c_o_stretch"),
        "nitrile_to_fingerprint": _r("nitrile", "fingerprint"),
    }

    out: dict[str, Any] = {
        "peaks": peak_list,
        "regions": regions,
        "band_matches": band_matches,
        "ratios": ratios,
        "summary": {
            "n_peaks": len(peak_list),
            "n_diagnostic_peaks": sum(
                1 for p in peak_list if str(p.get("peak_role")) == "diagnostic_peak"
            ),
            "n_weak_peaks": sum(1 for p in peak_list if str(p.get("peak_quality")) == "weak"),
            "peak_sensitivity": sens,
            "peak_min_height": thresholds["peak_min_height"],
            "peak_min_prominence": thresholds["peak_min_prominence"],
            "y_max": y_max,
            "y_range": y_range,
            "oh_nh_broadness": oh_broad,
        },
    }
    ont = str(cfg.get("ontology") or "v3").lower()
    out["ontology"] = ont
    if ont == "v4":
        out.update(_partition_evidence_v4(out))
    return out


def _partition_evidence_v4(evidence: dict[str, Any]) -> dict[str, Any]:
    """Split evidence into local motifs, FG-supporting bands, and artifact summaries (v4 reports)."""
    from ml.ftir_ontology import LOCAL_MOTIF_BAND_IDS, LOCAL_MOTIF_REGION_KEYS

    match_map = {str(m.get("band_id")): m for m in evidence.get("band_matches") or []}
    regions = evidence.get("regions") or {}
    local_motifs: dict[str, Any] = {}
    for motif, bids in LOCAL_MOTIF_BAND_IDS.items():
        best = 0.0
        hits: list[dict[str, Any]] = []
        for bid in bids:
            m = match_map.get(bid)
            if not m or not m.get("matched"):
                continue
            sc = float(m.get("support_score", 0) or 0)
            best = max(best, sc)
            hits.append(m)
        for rkey in LOCAL_MOTIF_REGION_KEYS.get(motif, ()):
            rel = float((regions.get(rkey) or {}).get("rel_max", 0) or 0)
            if rel > best:
                best = rel
        if motif == "upper_mid_activity_region" and not hits and best < 0.08:
            rel_um = float((regions.get("upper_mid_activity") or {}).get("rel_max", 0) or 0)
            best = max(best, rel_um)
        local_motifs[motif] = {
            "support_score": round(best, 4),
            "band_ids": [str(h.get("band_id")) for h in hits],
            "bands": hits[:8],
        }

    fg_evidence: dict[str, Any] = {}
    for m in evidence.get("band_matches") or []:
        if not m.get("matched"):
            continue
        lab = str(m.get("label", "")).strip().lower()
        if not lab:
            continue
        fg_evidence.setdefault(lab, {"support_score": 0.0, "bands": []})
        sc = float(m.get("support_score", 0) or 0)
        block = fg_evidence[lab]
        block["support_score"] = round(max(float(block["support_score"]), sc), 4)
        if len(block["bands"]) < 8:
            block["bands"].append(m)

    arts = evidence.get("artifacts") or {}
    artifacts_block: dict[str, Any] = {
        "flags": dict(arts.get("flags") or {}),
        "cautions": list(arts.get("cautions") or [])[:12],
        "summary": str(arts.get("summary") or ""),
    }
    return {
        "local_motifs": local_motifs,
        "fg_evidence": fg_evidence,
        "ontology_artifacts": artifacts_block,
    }


def _spec_rank(spec: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(spec).lower(), 0)


def _imp_rank(imp: str) -> int:
    return {"required": 3, "supporting": 2, "weak": 1}.get(str(imp).lower(), 0)


def _assignments_dict(rule_assignments: dict[str, Any] | None) -> dict[str, Any]:
    if not rule_assignments:
        return {}
    return dict(rule_assignments.get("assignments") or {})


def _fg_rule_key_for_band(band: dict[str, Any], assignments: dict[str, Any]) -> str | None:
    """Map library band to rule-assignment key (lowercase FG label)."""
    candidates: list[str] = []
    for key in ("label", "subclass"):
        v = band.get(key)
        if v and str(v).strip():
            candidates.append(str(v).strip().lower())
    aliases = {
        "amide_carbonyl": "amide",
        "carboxylic_acid_oh": "carboxylic_acid",
        "broad_oh": "alcohol",
    }
    seen: set[str] = set()
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        mapped = aliases.get(raw, raw)
        if mapped in assignments:
            return mapped
        if raw in assignments:
            return raw
    if candidates:
        c0 = aliases.get(candidates[0], candidates[0])
        return c0 if c0 in assignments else None
    return None


def _band_matches_wavenumber(
    nu: float,
    band: dict[str, Any],
    detected_peaks: list[dict[str, Any]],
    tolerance_cm1: float,
) -> bool:
    lo = float(band["region_min_cm1"])
    hi = float(band["region_max_cm1"])
    if lo <= nu <= hi:
        return True
    for p in detected_peaks:
        pwn = float(p.get("wn_cm1", p.get("wn", 0)))
        if lo <= pwn <= hi and abs(pwn - nu) <= tolerance_cm1:
            return True
    return False


def _nearest_peak_info(
    nu: float,
    detected_peaks: list[dict[str, Any]],
    tolerance_cm1: float,
) -> tuple[bool, dict[str, Any] | None]:
    if not detected_peaks:
        return False, None
    best = min(detected_peaks, key=lambda p: abs(float(p.get("wn_cm1", p.get("wn", 0))) - nu))
    pwn = float(best.get("wn_cm1", best.get("wn", 0)))
    dist = abs(pwn - nu)
    near = dist <= tolerance_cm1
    return near, {"wn_cm1": pwn, "height": float(best.get("height", 0)), "distance_cm1": dist}


def _support_status_from_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.6:
        return "supported"
    if score >= 0.3:
        return "partial"
    return "not_supported"


def _status_display(st: str) -> str:
    return {
        "supported": "supported",
        "partial": "weak/partial",
        "not_supported": "not supported",
        "unknown": "unknown",
    }.get(st, st)


def _hover_v3_assignment_line(rule_key: str, assignments: dict[str, Any]) -> str:
    ent = assignments.get(rule_key) or {}
    cc = ent.get("confidence_class")
    if isinstance(cc, str) and cc.strip():
        return cc.strip()
    return _status_display(_support_status_from_score(ent.get("score")))


def _hover_assignment_line(rule_key: str, assignments: dict[str, Any]) -> str:
    ent = assignments.get(rule_key) or {}
    ist = ent.get("interpretation_strength")
    if isinstance(ist, str) and ist.strip():
        return ist.strip()
    return _hover_v3_assignment_line(rule_key, assignments)


def _evidence_elsewhere_lines(evidence: dict[str, Any] | None, nu: float) -> list[str]:
    if not evidence:
        return []
    lines: list[str] = []
    for m in evidence.get("band_matches") or []:
        if not m.get("matched"):
            continue
        lo = float(m.get("region_min_cm1", 0))
        hi = float(m.get("region_max_cm1", 0))
        if lo <= nu <= hi:
            continue
        lab = str(m.get("label", ""))
        mode = str(m.get("mode", ""))[:48]
        if "aromatic" in lab.lower() or "ring" in mode.lower():
            t = "Aromatic / ring-mode evidence is matched elsewhere in the spectrum."
            if t not in lines:
                lines.append(t)
        elif "oh" in mode.lower() or "o-h" in mode.lower() or lab.lower() in ("alcohol", "phenol", "phenol"):
            t = "O–H / hydrogen-bond envelope evidence is matched elsewhere."
            if t not in lines:
                lines.append(t)
        elif "si" in mode.lower() or "siloxane" in lab.lower():
            t = "Si–O / organosilicon evidence is matched elsewhere."
            if t not in lines:
                lines.append(t)
    return lines[:4]


def _upper_mid_motif_hover_lines(nu: float, evidence: dict[str, Any] | None) -> list[str]:
    """Region-activity hover (not FG assignment) for 3200–1800 cm⁻¹ sub-windows."""
    if not evidence:
        return []
    regions = evidence.get("regions") or {}
    lm = evidence.get("local_motifs") or {}
    lines: list[str] = []

    def _rel(name: str) -> float:
        return float((regions.get(name) or {}).get("rel_max", 0) or 0)

    if 3000.0 <= nu <= 3100.0 and max(
        _rel("aromatic_ch_stretch"),
        float((lm.get("aromatic_CH_region") or {}).get("support_score", 0) or 0),
    ) >= 0.04:
        lines.extend(
            [
                "<b>Region activity</b>: aromatic/sp² C–H stretch (3000–3100 cm⁻¹)",
                "Local motif — pair with ring C=C modes near 1450–1600 cm⁻¹",
                "Support: local / supporting only",
            ]
        )
    elif 2850.0 <= nu <= 2965.0 and max(
        _rel("aliphatic_ch"),
        float((lm.get("aliphatic_CH_region") or {}).get("support_score", 0) or 0),
    ) >= 0.04:
        lines.extend(
            [
                "<b>Region activity</b>: aliphatic C–H stretch (2850–2965 cm⁻¹)",
                "Local motif — possible alkyl / polymer backbone / organic matrix",
                "Support: local only; C–H stretch is not specific alone",
            ]
        )
    elif 2720.0 <= nu <= 2820.0 and max(_rel("aldehydic_ch"), 0.0) >= 0.04:
        carb = max(_rel("carbonyl"), _rel("amide_i"))
        lines.extend(
            [
                "<b>Region activity</b>: aldehydic C–H candidate (2720–2820 cm⁻¹)",
                "Not aromatic C–H — pair with C=O near 1650–1820 for aldehyde context",
            ]
        )
        if carb < 0.08:
            lines.append("C=O region weak here — aldehydic assignment remains tentative")
    elif 2260.0 <= nu < 2720.0 and max(_rel("upper_mid_activity"), float((lm.get("upper_mid_activity_region") or {}).get("support_score", 0) or 0)) >= 0.05:
        lines.extend(
            [
                "<b>Region activity</b>: upper mid-IR",
                "Usually weak / less diagnostic — do not assign a strong FG from this alone",
            ]
        )
    elif 3100.0 <= nu <= 3200.0 and _rel("nh_ch_transition") >= 0.05:
        lines.append("<b>Region activity</b>: N–H / aromatic C–H shoulder (3100–3200 cm⁻¹)")
    return lines


def _mid_region_noxide_hover_lines(nu: float, evidence: dict[str, Any] | None) -> list[str]:
    """1450–1650 cm⁻¹ overlap guidance (called from build_local_hover_context)."""
    if not evidence or not (1450.0 <= nu <= 1650.0):
        return []
    regions = evidence.get("regions") or {}
    lm = evidence.get("local_motifs") or {}
    match_map = {str(m.get("band_id")): m for m in (evidence.get("band_matches") or []) if m.get("matched")}

    def _rel(name: str) -> float:
        return float((regions.get(name) or {}).get("rel_max", 0) or 0)

    act = max(
        _rel("aromatic_cc"),
        _rel("amide_ii"),
        _rel("nitro_asym"),
        float((lm.get("heterocyclic_N_O_region") or {}).get("support_score", 0) or 0),
    )
    if act < 0.04:
        return []
    lines = [
        "<b>Local region</b>: C=C / amide II / N–O / NO₂ overlap",
        "Possible: aromatic C=C, amide II, enamine C=C–N, heterocyclic N–O",
    ]
    sym = match_map.get("nitro_sym")
    asym = match_map.get("nitro_asym")
    if asym and asym.get("matched") and not (sym and sym.get("matched")):
        lines.append("NO₂ asym only — nitro not supported without symmetric ~1320–1390 cm⁻¹")
    elif sym and sym.get("matched") and asym and asym.get("matched"):
        lines.append("Paired NO₂-like bands — nitro still needs rule/guardrail confirmation")
    return lines


def build_local_hover_context(
    wavenumber_value: float,
    absorbance_value: float,
    detected_peaks: list[dict[str, Any]],
    band_library: list[dict[str, Any]] | None = None,
    rule_assignments: dict[str, Any] | None = None,
    ml_assignments: dict[str, float] | None = None,
    evidence: dict[str, Any] | None = None,
    *,
    tolerance_cm1: float = 12.0,
    max_labels: int = 5,
    ontology: str | None = None,
) -> dict[str, Any]:
    """
    Build frequency-local, band-aware hover context (Plotly customdata).

    Does not include global top-k classifier probabilities; ML scores are only attached
    to functional-group keys that locally match ``wavenumber_value``.
    """
    nu = float(wavenumber_value)
    a = float(absorbance_value)
    library = list(band_library) if band_library is not None else load_band_library(prefer_python=True)
    assignments = _assignments_dict(rule_assignments)
    ml_map = dict(ml_assignments or {})
    peaks = [dict(p) for p in (detected_peaks or [])]
    ont = str(ontology or (evidence or {}).get("ontology") or "v3").lower()

    near_peak, nearest = _nearest_peak_info(nu, peaks, tolerance_cm1)
    peak_quality = str((nearest or {}).get("peak_quality", "") or "")
    peak_role = str((nearest or {}).get("peak_role", "") or "")

    matched_raw: list[tuple[dict[str, Any], float | None, float]] = []
    for band in library:
        if not _band_matches_wavenumber(nu, band, peaks, tolerance_cm1):
            continue
        key = _fg_rule_key_for_band(band, assignments)
        score = float(assignments[key]["score"]) if key and key in assignments else None
        # distance from nu to any peak inside band
        lo, hi = float(band["region_min_cm1"]), float(band["region_max_cm1"])
        d_peak = float("inf")
        for p in peaks:
            pwn = float(p.get("wn_cm1", p.get("wn", 0)))
            if lo <= pwn <= hi:
                d_peak = min(d_peak, abs(pwn - nu))
        if not np.isfinite(d_peak):
            d_peak = 0.0
        matched_raw.append((band, score, d_peak))

    matched_raw.sort(
        key=lambda t: (
            -_spec_rank(t[0].get("specificity", "low")),
            -_imp_rank(t[0].get("importance", "weak")),
            -(t[1] if t[1] is not None else -1.0),
            t[2],
        )
    )

    local_fg_keys: set[str] = set()
    for band, _, __ in matched_raw:
        k = _fg_rule_key_for_band(band, assignments)
        if k:
            local_fg_keys.add(k)
        else:
            lbl = str(band.get("label", "")).lower()
            if lbl:
                local_fg_keys.add(lbl)

    matching_bands: list[dict[str, Any]] = []
    caution_flags: list[str] = []

    for band, score, _ in matched_raw[: max(1, max_labels * 3)]:
        key = _fg_rule_key_for_band(band, assignments)
        fg_label = key or str(band.get("label", ""))
        sty = _support_status_from_score(score)
        ml_v: float | None = None
        show_ml = key and key in ml_map
        if show_ml:
            ml_v = float(ml_map[key])
        notes = str(band.get("notes") or "")
        matching_bands.append(
            {
                "label": str(band.get("label", "")),
                "subclass": band.get("subclass"),
                "range": [float(band["region_min_cm1"]), float(band["region_max_cm1"])],
                "mode": str(band.get("mode", "")),
                "importance": str(band.get("importance", "")),
                "specificity": str(band.get("specificity", "")),
                "support_status": sty,
                "rule_score": score,
                "ml_score": ml_v,
                "notes": notes,
                "band_id": band.get("id"),
            }
        )

    by_fg: dict[str, dict[str, Any]] = {}
    for row in matching_bands:
        fg = _fg_rule_key_for_band(
            {
                "label": row["label"],
                "subclass": row.get("subclass"),
            },
            assignments,
        ) or str(row["label"]).lower()
        cur = by_fg.get(fg) or {"rule_score": None, "support": "unknown", "rows": []}
        rs = row.get("rule_score")
        if rs is not None and (cur["rule_score"] is None or rs > cur["rule_score"]):
            cur["rule_score"] = rs
            cur["support"] = row["support_status"]
        cur["rows"].append(row)
        by_fg[fg] = cur

    fg_ranked_full = sorted(
        by_fg.items(),
        key=lambda kv: (
            -_spec_rank(kv[1]["rows"][0].get("specificity", "low")) if kv[1]["rows"] else 0,
            -(kv[1]["rule_score"] if kv[1]["rule_score"] is not None else -1.0),
        ),
    )
    fg_ranked = fg_ranked_full[:max_labels]

    meas = (evidence or {}).get("measurement") or {}
    art_flags = ((evidence or {}).get("artifacts") or {}).get("flags") or {}
    if (meas.get("is_atr") or art_flags.get("atr_crystal_fingerprint_overlap")) and any(
        str(b.get("band_id")) in ("siloxane_sio", "silicone_sic") for b in matching_bands
    ):
        caution_flags.append(
            "ATR-sensitive overlap region — competitors: organic C–O, aryl ether, ester C–O, "
            "siloxane, ATR/crystal overlap"
        )

    if len(matching_bands) >= 2:
        lows = {_spec_rank(b.get("specificity", "low")) for b in matching_bands[:6]}
        if min(lows) <= 1 and len(matching_bands) >= 3:
            caution_flags.append("overlap region — multiple low-specificity bands coincide; inspect paired evidence.")
    for row in matching_bands[:5]:
        if str(row.get("specificity", "")).lower() == "low":
            caution_flags.append("low specificity in this window — assignments are tentative.")
        if str(row.get("importance", "")).lower() == "required" and row.get("support_status") in ("not_supported", "partial"):
            caution_flags.append(f"“{row.get('mode','')}” often needs corroborating bands elsewhere — check pair rules.")
            break

    elsewhere = _evidence_elsewhere_lines(evidence, nu)
    obs_lines: list[str] = []
    if near_peak and nearest:
        obs_lines.append(f"detected peak near {nearest['wn_cm1']:.0f} cm⁻¹")
    obs_lines.extend(elsewhere)

    more_n = max(0, len(fg_ranked_full) - max_labels)

    peak_line = f"Peak: {'yes' if near_peak else 'no'}"
    if nearest:
        peak_line += f", nearest {nearest['wn_cm1']:.1f} cm⁻¹"
    elif peaks:
        peak_line += f" (nearest pick {_nearest_str(peaks, nu)})"

    region_activity_lines = _upper_mid_motif_hover_lines(nu, evidence)
    region_activity_lines.extend(_mid_region_noxide_hover_lines(nu, evidence))

    if not matching_bands:
        tail_parts = [peak_line]
        if region_activity_lines:
            tail_parts.extend(region_activity_lines)
        else:
            tail_parts.append(
                "No diagnostic FTIR band in current library at this frequency (within peak-linking tolerance)."
            )
        plotly_tail = "<br>".join(tail_parts)
        hover_text = "<br>".join([f"<b>ν={nu:.1f} cm⁻¹</b>", f"A={a:.3f}"] + tail_parts)
        caution_flags = list(dict.fromkeys(caution_flags))
        return {
            "nu": nu,
            "absorbance": a,
            "near_peak": near_peak,
            "nearest_peak": nearest,
            "peak_quality": peak_quality,
            "peak_role": peak_role,
            "matching_bands": [],
            "caution_flags": caution_flags,
            "hover_text": hover_text,
            "plotly_tail": plotly_tail,
        }

    region_bits = []
    for _, block in fg_ranked[:2]:
        for r in block["rows"][:1]:
            region_bits.append(f"{int(r['range'][0])}–{int(r['range'][1])} cm⁻¹, {r.get('mode', '')}")

    motif_lines: list[str] = []
    if ont == "v4" and evidence:
        from ml.ftir_ontology import ONTOLOGY_V4

        lm = evidence.get("local_motifs") or {}
        for mkey, mblock in sorted(lm.items()):
            if not isinstance(mblock, dict):
                continue
            if float(mblock.get("support_score", 0) or 0) < 0.12:
                continue
            span_txt = None
            for m in mblock.get("bands") or []:
                lo = float(m.get("region_min_cm1", 0))
                hi = float(m.get("region_max_cm1", 0))
                if lo - tolerance_cm1 <= nu <= hi + tolerance_cm1:
                    span_txt = f"{int(lo)}–{int(hi)} cm⁻¹"
                    break
            if span_txt is None:
                continue
            disp = ONTOLOGY_V4[mkey].display_name if mkey in ONTOLOGY_V4 else mkey.replace("_", " ")
            motif_lines.append(f"• {disp} ({span_txt})")
        motif_lines = motif_lines[:5]

    fg_lines = []
    for fg, block in fg_ranked:
        rs = block["rule_score"]
        ml_note = ""
        if ont != "v4":
            if fg in local_fg_keys and fg in ml_map:
                ml_note = f", ML={float(ml_map[fg]):.3f}"
        rs_s = f"{rs:.2f}" if rs is not None else "—"
        disp = str(block["rows"][0].get("label", fg)) if block["rows"] else fg
        v_st = _hover_assignment_line(fg, assignments)
        fam_note = ""
        if ont == "v4":
            ent_a = assignments.get(fg) or {}
            oc = str(ent_a.get("ontology_category") or "")
            if oc == "fallback":
                fam_note = " (family evidence)"
        fg_lines.append(f"• {disp} — {v_st}, rule S={rs_s}{ml_note}{fam_note}")

    conf_lines: list[str] = []
    for fg, block in fg_ranked[:5]:
        ent_a = assignments.get(fg) or {}
        cc_a = str(ent_a.get("confidence_class") or "—")
        conf_lines.append(f"• {fg}: {cc_a}")
    miss_lines: list[str] = []
    for fg, block in fg_ranked[:4]:
        ent_a = assignments.get(fg) or {}
        miss = ent_a.get("missing_expected_bands") or []
        if miss:
            miss_lines.append(f"• {fg}: " + ", ".join(str(x) for x in miss[:3]))

    tail_parts = [peak_line]
    if region_activity_lines:
        tail_parts.extend(region_activity_lines)
    if ont == "v4" and motif_lines:
        tail_parts.append("<b>Local motif</b><br>" + "<br>".join(motif_lines))
    elif region_bits:
        tail_parts.append("<b>Local band region</b><br>" + "<br>".join(region_bits[:2]))
    fg_hdr = "<b>Possible functional groups</b>" if ont == "v4" else "<b>Possible assignments</b>"
    if fg_lines:
        tail_parts.append(f"{fg_hdr}<br>" + "<br>".join(fg_lines))
    if conf_lines:
        tail_parts.append("<b>Confidence class</b><br>" + "<br>".join(conf_lines))
    if miss_lines:
        tail_parts.append("<b>Missing paired evidence</b><br>" + "<br>".join(miss_lines))
    if obs_lines:
        tail_parts.append("<b>Observed evidence</b><br>• " + "<br>• ".join(obs_lines))
    if more_n:
        tail_parts.append(f"+ {more_n} more possible overlapping bands")
    if caution_flags:
        uniq = list(dict.fromkeys(caution_flags))
        tail_parts.append("<b>Caution</b><br>" + "<br>".join(f"• {c[:220]}" for c in uniq[:3]))

    plotly_tail = "<br>".join(tail_parts)
    hover_text = "<br>".join([f"<b>ν={nu:.1f} cm⁻¹</b>", f"A={a:.3f}"] + tail_parts)
    caution_flags = list(dict.fromkeys(caution_flags))

    return {
        "nu": nu,
        "absorbance": a,
        "near_peak": near_peak,
        "nearest_peak": nearest,
        "peak_quality": peak_quality,
        "peak_role": peak_role,
        "matching_bands": matching_bands[: max_labels * 2],
        "caution_flags": caution_flags,
        "hover_text": hover_text,
        "plotly_tail": plotly_tail,
    }


def _nearest_str(peaks: list[dict[str, Any]], nu: float) -> str:
    if not peaks:
        return "n/a"
    best = min(peaks, key=lambda p: abs(float(p.get("wn_cm1", p.get("wn", 0))) - nu))
    return f"{float(best.get('wn_cm1', best.get('wn', 0))):.0f} cm⁻¹"


def format_peak_marker_hover(ctx: dict[str, Any], *, max_bands: int = 6) -> str:
    """Richer hover for picked-peak markers (local bands + plotly_tail body)."""
    nu = float(ctx["nu"])
    a = float(ctx["absorbance"])
    bands: list[dict[str, Any]] = list(ctx.get("matching_bands") or [])[:max_bands]
    tail = str(ctx.get("plotly_tail") or "")
    parts = tail.split("<br>", 1)
    peak_line = parts[0] if parts else f"Peak: {'yes' if ctx.get('near_peak') else 'no'}"
    rest = parts[1] if len(parts) > 1 else ""

    lines = [
        f"<b>Peak {nu:.1f} cm⁻¹</b>",
        f"A={a:.3f}",
        peak_line,
    ]
    pq = str(ctx.get("peak_quality") or "")
    pr = str(ctx.get("peak_role") or "")
    if pq == "weak" or pr == "weak_peak":
        lines.append("<i>weak local peak; not sufficient alone for assignment</i>")
    if bands:
        lines.append("<b>Matched bands</b>")
        for b in bands:
            lines.append(
                f"• {str(b.get('mode', ''))[:80]} — {_status_display(str(b.get('support_status', 'unknown')))}, "
                f"{b.get('importance', '')} importance"
            )
    if rest.strip():
        lines.append(rest)
    return "<br>".join(lines)


def rule_assignment_map(rule_assignments: dict[str, Any] | None) -> dict[str, Any]:
    """Flat label → assignment entry dict (for band / hover helpers)."""
    return _assignments_dict(rule_assignments)


def rule_key_for_band(band: dict[str, Any], rule_assignments: dict[str, Any] | None) -> str | None:
    """Map a band-library row to a rules assignment key."""
    return _fg_rule_key_for_band(band, _assignments_dict(rule_assignments))
