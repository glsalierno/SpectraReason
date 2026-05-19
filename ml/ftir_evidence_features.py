"""
Interpretable evidence feature layouts for structural FG SVM training.

``evidence_v1`` (legacy name ``spectral+evidence``): compact band/ratio/motif summary.
``evidence_v2`` (``spectral+evidence_v2``): regional stats, per-band peak metrics, quality, broadness, ratios, artifacts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ml.ftir_band_library import load_band_library
from ml.ftir_interpretable_features import INTERPRETABLE_REGIONS, oh_broadness_metric

EVIDENCE_FEATURE_VERSION_V1 = "evidence_v1"
EVIDENCE_FEATURE_VERSION_V2 = "evidence_v2"

# Regions exported as region_* features in v2 (must exist in evidence["regions"] when present).
V2_REGION_NAMES: tuple[str, ...] = tuple(r[0] for r in INTERPRETABLE_REGIONS)

# Extra ratio keys for v2 (merged with evidence.ratios; aliases deduped).
V2_EXTRA_RATIO_KEYS: tuple[tuple[str, str, str], ...] = (
    ("oh_to_fingerprint", "oh_nh_broad", "fingerprint"),
    ("c_o_to_carbonyl", "c_o_stretch", "carbonyl"),
    ("aromatic_to_c_o", "aromatic_cc", "c_o_stretch"),
    ("sio_overlap_to_organic_c_o", "si_o", "c_o_stretch"),
    ("nitro_asym_to_sym", "nitro_asym", "nitro_sym"),
    ("amide_to_carbonyl", "amide_i", "carbonyl"),
)

# Artifact flag keys → stable art_* feature names
ARTIFACT_FLAG_MAP: dict[str, str] = {
    "water_vapor_or_moisture_like": "art_water_moisture",
    "co2_region_elevated": "art_co2",
    "weak_nitrile_region_spike": "art_noise_spike",
    "baseline_drift_or_tilt": "art_baseline_instability",
    "edge_truncation": "art_edge_artifact",
    "possible_saturation": "art_saturated_peak",
    "fingerprint_crowding": "art_fingerprint_crowding",
}

_STABLE_V1_NAMES: list[str] | None = None
_STABLE_V2_NAMES: list[str] | None = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def evidence_feature_version_for_feature_set(feature_set: str | None) -> str:
    fs = str(feature_set or "").strip().lower().replace(" ", "")
    if "evidence_v2" in fs or fs in ("spectral+evidence_v2", "evidence_v2"):
        return EVIDENCE_FEATURE_VERSION_V2
    return EVIDENCE_FEATURE_VERSION_V1


def _build_evidence_v1(evidence: dict[str, Any]) -> tuple[list[float], list[str]]:
    names: list[str] = []
    vals: list[float] = []

    def add(prefix: str, key: str, val: float) -> None:
        names.append(f"{prefix}_{key}")
        vals.append(_safe_float(val))

    for m in evidence.get("band_matches") or []:
        bid = str(m.get("band_id", ""))
        add("band", bid, float(m.get("support_score", 0) or 0))

    ratios = evidence.get("ratios") or {}
    for rk, rv in sorted(ratios.items()):
        add("ratio", rk, float(rv))

    summ = evidence.get("summary") or {}
    add("sum", "oh_nh_broadness", float(summ.get("oh_nh_broadness", 0) or 0))
    add("sum", "n_peaks", float(summ.get("n_peaks", 0) or 0))

    art = evidence.get("artifacts") or {}
    flags = art.get("flags") or {}
    for fk, fv in sorted(flags.items()):
        add("art", fk, 1.0 if fv else 0.0)

    lm = evidence.get("local_motifs") or {}
    for mk, block in sorted(lm.items()):
        if isinstance(block, dict):
            add("motif", mk, float(block.get("support_score", 0) or 0))

    return vals, names


def _peaks_in_window(peaks: list[dict[str, Any]], lo: float, hi: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in peaks:
        w = _safe_float(p.get("wn_cm1", p.get("wn", 0)))
        if lo <= w <= hi:
            out.append(p)
    return out


def _band_center(band: dict[str, Any]) -> float:
    return 0.5 * (float(band["region_min_cm1"]) + float(band["region_max_cm1"]))


def _nearest_peak_stats(
    peaks: list[dict[str, Any]],
    lo: float,
    hi: float,
    *,
    tol: float = 25.0,
) -> dict[str, float]:
    near = _peaks_in_window(peaks, lo - tol, hi + tol)
    if not near:
        return {
            "peak_count": 0.0,
            "nearest_peak_distance": 999.0,
            "nearest_peak_height": 0.0,
            "max_peak_support": 0.0,
            "mean_peak_support": 0.0,
        }
    center = 0.5 * (lo + hi)
    dists: list[float] = []
    heights: list[float] = []
    supports: list[float] = []
    for p in near:
        w = _safe_float(p.get("wn_cm1", 0))
        dists.append(abs(w - center))
        heights.append(_safe_float(p.get("rel_height", p.get("height", 0))))
        supports.append(_safe_float(p.get("rel_height", p.get("height", 0))))
    return {
        "peak_count": float(len(near)),
        "nearest_peak_distance": float(min(dists)),
        "nearest_peak_height": float(max(heights)),
        "max_peak_support": float(max(supports)),
        "mean_peak_support": float(sum(supports) / len(supports)),
    }


def _quality_agg(peaks: list[dict[str, Any]], lo: float, hi: float) -> dict[str, float]:
    sub = _peaks_in_window(peaks, lo, hi)
    if not sub:
        return {
            "peak_width_mean": 0.0,
            "peak_width_max": 0.0,
            "peak_sharpness_mean": 0.0,
            "peak_sharpness_max": 0.0,
            "peak_isolation_mean": 0.0,
            "peak_isolation_max": 0.0,
            "peak_snr_proxy_mean": 0.0,
            "peak_snr_proxy_max": 0.0,
        }

    def col(key: str) -> list[float]:
        return [_safe_float(p.get(key, 0)) for p in sub if p.get(key) is not None]

    widths = col("quality_width_cm1")
    sharp = col("quality_sharpness")
    isol = col("quality_isolation")
    snr = col("quality_snr_proxy")
    out: dict[str, float] = {}
    for prefix, arr in (
        ("peak_width", widths),
        ("peak_sharpness", sharp),
        ("peak_isolation", isol),
        ("peak_snr_proxy", snr),
    ):
        if arr:
            out[f"{prefix}_mean"] = float(sum(arr) / len(arr))
            out[f"{prefix}_max"] = float(max(arr))
        else:
            out[f"{prefix}_mean"] = 0.0
            out[f"{prefix}_max"] = 0.0
    return out


def _region_area_norm(reg: dict[str, Any]) -> float:
    integral = _safe_float(reg.get("integral", 0))
    npt = max(_safe_float(reg.get("n_points", 0)), 1.0)
    return integral / npt


def _merged_ratios(evidence: dict[str, Any]) -> dict[str, float]:
    ratios = dict(evidence.get("ratios") or {})
    regions = evidence.get("regions") or {}

    def rel(name: str) -> float:
        return _safe_float((regions.get(name) or {}).get("rel_max", 0))

    alias = {
        "oh_to_fingerprint": ratios.get("oh_nh_to_fingerprint"),
        "ratio_oh_to_fingerprint": ratios.get("oh_nh_to_fingerprint"),
    }
    for k, v in alias.items():
        if v is not None and k not in ratios:
            ratios[k] = float(v)

    for out_key, num, den in V2_EXTRA_RATIO_KEYS:
        if out_key in ratios:
            continue
        a, b = rel(num), rel(den)
        ratios[out_key] = float(a / (b + 1e-9))

    return ratios


def _build_evidence_v2(evidence: dict[str, Any]) -> tuple[list[float], list[str]]:
    names: list[str] = []
    vals: list[float] = []

    def add(name: str, val: float) -> None:
        names.append(name)
        vals.append(_safe_float(val))

    # A. v1 core (bands, ratios, sum, motifs)
    v1_vals, v1_names = _build_evidence_v1(evidence)
    for n, v in zip(v1_names, v1_vals):
        add(n, v)

    regions = evidence.get("regions") or {}
    peaks = list(evidence.get("peaks") or [])
    y_max = _safe_float((evidence.get("summary") or {}).get("y_max", 1.0), 1.0)

    # B. Regional statistics
    for rname in V2_REGION_NAMES:
        reg = regions.get(rname) or {}
        add(f"region_{rname}_integral", _safe_float(reg.get("integral", 0)))
        add(f"region_{rname}_rel_max", _safe_float(reg.get("rel_max", 0)))
        add(f"region_{rname}_mean", _safe_float(reg.get("mean", 0)))
        add(f"region_{rname}_std", _safe_float(reg.get("std", 0)))
        add(f"region_{rname}_area_norm", _region_area_norm(reg))

    # C. Per-band peak statistics (library bands)
    library = load_band_library(prefer_python=True)
    for band in library:
        bid = str(band.get("id", ""))
        lo = float(band["region_min_cm1"])
        hi = float(band["region_max_cm1"])
        st = _nearest_peak_stats(peaks, lo, hi)
        add(f"peak_count_{bid}", st["peak_count"])
        add(f"nearest_peak_distance_{bid}", st["nearest_peak_distance"])
        add(f"nearest_peak_height_{bid}", st["nearest_peak_height"])
        add(f"max_peak_support_{bid}", st["max_peak_support"])
        add(f"mean_peak_support_{bid}", st["mean_peak_support"])

    # D. Peak quality per interpretable region + global
    for rname, lo, hi in INTERPRETABLE_REGIONS:
        qa = _quality_agg(peaks, float(lo), float(hi))
        for k, v in qa.items():
            add(f"{k}_{rname}", v)
    qa_g = _quality_agg(peaks, 400.0, 4000.0)
    for k, v in qa_g.items():
        add(f"{k}_global", v)

    # E. Broadness
    wn = evidence.get("_wn_cache")
    yy = evidence.get("_y_cache")
    if wn is not None and yy is not None:
        add("broadness_oh_nh", oh_broadness_metric(np.asarray(wn), np.asarray(yy)))
    else:
        add("broadness_oh_nh", _safe_float((regions.get("oh_nh_broad") or {}).get("broadness", 0)))
    add("broadness_acid_oh", _safe_float((regions.get("oh_nh_broad") or {}).get("broadness", 0)) * 0.5)
    add("broadness_nh", _safe_float((regions.get("oh_nh_broad") or {}).get("std", 0)) / (y_max + 1e-9))
    fp = regions.get("fingerprint") or {}
    add("broadness_global_fingerprint", _safe_float(fp.get("std", 0)) / (y_max + 1e-9))

    # F. Additional ratios (dedupe names already added in v1 block)
    merged = _merged_ratios(evidence)
    existing = set(names)
    for rk, rv in sorted(merged.items()):
        nm = f"ratio_{rk}"
        if nm in existing:
            continue
        add(nm, float(rv))
        existing.add(nm)

    # G. Artifact features (canonical names)
    flags = (evidence.get("artifacts") or {}).get("flags") or {}
    for flag_key, feat_name in ARTIFACT_FLAG_MAP.items():
        add(feat_name, 1.0 if flags.get(flag_key) else 0.0)

    return vals, names


def evidence_feature_vector(
    evidence: dict[str, Any],
    *,
    version: str | None = None,
    feature_set: str | None = None,
) -> tuple[list[float], list[str]]:
    """Deterministic evidence feature vector and names."""
    ver = version or evidence_feature_version_for_feature_set(feature_set)
    if ver == EVIDENCE_FEATURE_VERSION_V2:
        return _build_evidence_v2(evidence)
    return _build_evidence_v1(evidence)


def stable_evidence_feature_names(
    *,
    version: str | None = None,
    feature_set: str | None = None,
) -> list[str]:
    """Fixed column order for training (probe spectrum)."""
    global _STABLE_V1_NAMES, _STABLE_V2_NAMES
    ver = version or evidence_feature_version_for_feature_set(feature_set)
    if ver == EVIDENCE_FEATURE_VERSION_V2:
        if _STABLE_V2_NAMES is None:
            from ml.ftir_evidence import extract_spectral_evidence

            wn = np.linspace(400.0, 4000.0, 900)
            y = np.sin(wn / 220.0) * 0.02 + 0.05
            ev = extract_spectral_evidence(wn, y, peaks=None, config={"ontology": "v4"})
            try:
                from ml.ftir_artifacts import detect_spectral_artifacts

                ev["artifacts"] = detect_spectral_artifacts(wn, y, ev)
            except Exception:
                pass
            ev["_wn_cache"] = wn
            ev["_y_cache"] = y
            _STABLE_V2_NAMES = evidence_feature_vector(ev, version=EVIDENCE_FEATURE_VERSION_V2)[1]
        return list(_STABLE_V2_NAMES)
    if _STABLE_V1_NAMES is None:
        from ml.ftir_evidence import extract_spectral_evidence

        wn = np.linspace(400.0, 4000.0, 900)
        y = np.sin(wn / 220.0) * 0.02 + 0.05
        ev = extract_spectral_evidence(wn, y, peaks=None, config={"ontology": "v4"})
        _STABLE_V1_NAMES = evidence_feature_vector(ev, version=EVIDENCE_FEATURE_VERSION_V1)[1]
    return list(_STABLE_V1_NAMES)


def align_evidence_vector(vec: list[float], names: list[str], template: list[str]) -> list[float]:
    """Pad/truncate to template column order."""
    mp = {n: v for n, v in zip(names, vec)}
    return [float(mp.get(n, 0.0)) for n in template]


def feature_prefix_counts(feature_names: list[str]) -> dict[str, int]:
    """Count features by first token prefix for audit reports."""
    counts: dict[str, int] = {}
    for n in feature_names:
        if n.startswith("spectral_"):
            p = "spectral"
        elif n.startswith("band_"):
            p = "band"
        elif n.startswith("region_"):
            p = "region"
        elif n.startswith("peak_count_") or n.startswith("nearest_peak_") or n.startswith("max_peak_") or n.startswith("mean_peak_"):
            p = "band_peak"
        elif n.startswith("peak_width_") or n.startswith("peak_sharpness_") or n.startswith("peak_isolation_") or n.startswith("peak_snr_"):
            p = "peak_quality"
        elif n.startswith("ratio_"):
            p = "ratio"
        elif n.startswith("motif_"):
            p = "motif"
        elif n.startswith("sum_"):
            p = "sum"
        elif n.startswith("art_"):
            p = "artifact"
        elif n.startswith("broadness_"):
            p = "broadness"
        elif n == "has_structure_flag":
            p = "has_structure"
        else:
            p = "other"
        counts[p] = counts.get(p, 0) + 1
    return counts
