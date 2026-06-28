"""
Region-limited pseudo-Voigt deconvolution for interpretable FTIR ML features.

Conservative fitting in diagnostic windows only — not a replacement for evidence_v2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from lib.peaks import find_peaks_simple

DECONV_PROFILE_TYPE = "pseudo_voigt"
DECONV_MODES = ("off", "fast", "full")

# fast: skip aromatic_fingerprint; cap components; fewer optimizer iterations
FAST_SKIP_REGIONS = frozenset({"aromatic_fingerprint"})
FAST_MAX_COMPONENTS: dict[str, int] = {
    "oh_nh": 2,
    "carbonyl": 2,
    "nitrile_alkyne": 2,
    "nitro": 3,
    "c_o_sio_overlap": 3,
}
FAST_MAXFEV = 800
FULL_MAXFEV = 4000
FAST_MIN_REL_SIGNAL = 0.04

# region_id, wn_lo, wn_hi, max_components, min_sep_cm1, min_fwhm, max_fwhm
DECONV_REGIONS: tuple[tuple[str, float, float, int, float, float, float], ...] = (
    ("oh_nh", 3000.0, 3700.0, 3, 55.0, 60.0, 450.0),
    ("carbonyl", 1650.0, 1820.0, 3, 18.0, 8.0, 80.0),
    ("nitrile_alkyne", 2100.0, 2260.0, 2, 22.0, 5.0, 45.0),
    ("nitro", 1300.0, 1600.0, 4, 20.0, 10.0, 90.0),
    ("c_o_sio_overlap", 950.0, 1250.0, 5, 15.0, 8.0, 120.0),
    ("aromatic_fingerprint", 1450.0, 1650.0, 4, 18.0, 10.0, 100.0),
)

REGION_COMMON_SUFFIXES: tuple[str, ...] = (
    "n_components",
    "total_area",
    "max_height",
    "dominant_center",
    "dominant_fwhm",
    "mean_fwhm",
    "min_fwhm",
    "overlap_index",
    "fit_r2",
    "residual_norm",
    "fit_success",
)

EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "deconv_nitro_asym_area",
    "deconv_nitro_sym_area",
    "deconv_nitro_asym_sym_ratio",
    "deconv_nitro_pair_present",
    "deconv_carbonyl_dominant_center",
    "deconv_carbonyl_component_count",
    "deconv_carbonyl_broadness",
    "deconv_triple_bond_sharp_component_present",
    "deconv_triple_bond_center",
    "deconv_triple_bond_fwhm",
    "deconv_c_o_sio_component_count",
    "deconv_c_o_sio_overlap_index",
    "deconv_c_o_sio_broad_component_area",
    "deconv_c_o_sio_sharp_component_area",
)


@dataclass
class PeakComponent:
    center: float
    amplitude: float
    sigma: float
    eta: float
    fwhm: float
    area: float


@dataclass
class RegionFitResult:
    region: str
    success: bool
    r2: float
    residual_norm: float
    components: list[PeakComponent]
    overlap_index: float
    total_area: float
    max_height: float
    dominant_center: float
    dominant_fwhm: float
    mean_fwhm: float
    min_fwhm: float


def stable_deconv_feature_names() -> list[str]:
    names: list[str] = []
    for rid, *_ in DECONV_REGIONS:
        for suf in REGION_COMMON_SUFFIXES:
            names.append(f"deconv_{rid}_{suf}")
    names.extend(EXTRA_FEATURE_NAMES)
    return names


def deconv_meta(*, mode: str = "fast") -> dict[str, Any]:
    return {
        "n_deconv": len(stable_deconv_feature_names()),
        "deconv_profile_type": DECONV_PROFILE_TYPE,
        "deconv_regions": [r[0] for r in DECONV_REGIONS],
        "deconv_feature_names": stable_deconv_feature_names(),
        "deconv_mode": str(mode or "fast").lower(),
    }


def _regions_for_mode(mode: str) -> list[tuple[str, float, float, int, float, float, float]]:
    m = str(mode or "fast").lower()
    out: list[tuple[str, float, float, int, float, float, float]] = []
    for spec in DECONV_REGIONS:
        rid = spec[0]
        if m == "fast" and rid in FAST_SKIP_REGIONS:
            continue
        if m == "fast":
            max_n = min(spec[3], FAST_MAX_COMPONENTS.get(rid, spec[3]))
            out.append((rid, spec[1], spec[2], max_n, spec[4], spec[5], spec[6]))
        else:
            out.append(spec)
    return out


@dataclass
class DeconvExtractionStats:
    failure_count: int = 0
    spectrum_failures: int = 0
    region_failures: int = 0
    total_runtime_sec: float = 0.0
    n_calls: int = 0

    def mean_ms(self) -> float:
        if self.n_calls <= 0:
            return 0.0
        return 1000.0 * self.total_runtime_sec / self.n_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "deconv_failure_count": int(self.spectrum_failures),
            "deconv_region_failure_count": int(self.region_failures),
            "deconv_extraction_errors": int(self.failure_count),
            "deconv_runtime_total_sec": round(self.total_runtime_sec, 3),
            "deconv_runtime_mean_ms": round(self.mean_ms(), 3),
            "deconv_n_calls": int(self.n_calls),
        }


_GLOBAL_DECONV_STATS = DeconvExtractionStats()


def reset_deconv_stats() -> None:
    global _GLOBAL_DECONV_STATS
    _GLOBAL_DECONV_STATS = DeconvExtractionStats()


def get_deconv_stats() -> DeconvExtractionStats:
    return _GLOBAL_DECONV_STATS


def _sigma_to_fwhm(sigma: float, eta: float = 0.5) -> float:
    """Approximate FWHM for pseudo-Voigt from Gaussian sigma."""
    g_fwhm = 2.355 * max(sigma, 1e-6)
    l_fwhm = 2.0 * max(sigma, 1e-6) * 1.2
    return float(eta * l_fwhm + (1.0 - eta) * g_fwhm)


def _pseudo_voigt(x: np.ndarray, center: float, amplitude: float, sigma: float, eta: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    z = (x - center) / sigma
    gauss = np.exp(-0.5 * z**2)
    lorentz = 1.0 / (1.0 + z**2)
    return amplitude * (eta * lorentz + (1.0 - eta) * gauss)


def _multi_pseudo_voigt(x: np.ndarray, *params: float) -> np.ndarray:
    """params: triplets (center, amplitude, sigma) with fixed eta=0.5 per peak."""
    y = np.zeros_like(x, dtype=float)
    n = len(params) // 3
    for i in range(n):
        c, a, s = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        y += _pseudo_voigt(x, c, a, s, 0.5)
    return y


def _crop_region(wn: np.ndarray, y: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    m = (wn >= lo) & (wn <= hi)
    if int(np.count_nonzero(m)) < 8:
        return np.array([]), np.array([])
    return wn[m].astype(float), y[m].astype(float)


def _baseline_correct(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size < 4:
        return y - float(np.min(y))
    coef = np.polyfit(x, y, 1)
    base = np.polyval(coef, x)
    yc = y - base
    yc = yc - float(np.min(yc))
    return np.clip(yc, 0.0, None)


def _initial_centers(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_n: int,
    min_sep: float,
) -> list[float]:
    if x.size < 8:
        return []
    prom = max(1e-9, 0.02 * float(np.ptp(y)))
    height = float(np.min(y)) + 0.02 * float(np.ptp(y))
    peaks, _ = find_peaks(y, prominence=prom, height=height, distance=max(2, int(min_sep / max(float(np.mean(np.diff(x))), 1.0))))
    centers = [float(x[p]) for p in peaks]
    if not centers:
        pwn, _ph = find_peaks_simple(x, y, max_peaks=max_n)
        centers = list(pwn)
    if not centers and float(np.max(y)) > 1e-9:
        centers = [float(x[int(np.argmax(y))])]
    # second-derivative hints
    try:
        if x.size >= 15:
            ys = savgol_filter(y, window_length=min(11, x.size // 2 * 2 + 1), polyorder=2)
            d2 = np.gradient(np.gradient(ys, x), x)
            neg, _ = find_peaks(-d2, prominence=prom * 0.5, distance=max(2, int(min_sep / max(float(np.mean(np.diff(x))), 1.0))))
            for p in neg:
                centers.append(float(x[p]))
    except Exception:
        pass
    centers = sorted(centers, key=lambda c: -float(np.interp(c, x, y)))
    kept: list[float] = []
    for c in centers:
        if all(abs(c - k) >= min_sep for k in kept):
            kept.append(c)
        if len(kept) >= max_n:
            break
    return sorted(kept)


def _fit_region(
    x: np.ndarray,
    y: np.ndarray,
    *,
    region_id: str,
    max_components: int,
    min_sep: float,
    min_fwhm: float,
    max_fwhm: float,
    maxfev: int = FULL_MAXFEV,
) -> RegionFitResult:
    empty = RegionFitResult(
        region=region_id,
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
    if x.size < 12 or float(np.max(y)) < 1e-6:
        return empty

    y_rng = float(np.ptp(y))
    if y_rng < 1e-5:
        return empty

    centers = _initial_centers(x, y, max_n=max_components, min_sep=min_sep)
    if not centers:
        return empty

    best: RegionFitResult | None = None
    for n in range(min(len(centers), max_components), 0, -1):
        use_c = centers[:n]
        p0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for c in use_c:
            amp0 = float(np.interp(c, x, y))
            sig0 = max(min_fwhm / 2.355, min(max_fwhm / 2.355, (x[-1] - x[0]) / (4.0 * n)))
            p0.extend([c, max(amp0, 1e-6), sig0])
            lower.extend([x[0], 0.0, min_fwhm / 2.355])
            upper.extend([x[-1], y_rng * 3.0, max_fwhm / 2.355])
        try:
            popt, _ = curve_fit(
                _multi_pseudo_voigt,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=int(maxfev),
            )
        except Exception:
            continue
        comps: list[PeakComponent] = []
        for i in range(n):
            c, a, s = float(popt[3 * i]), float(popt[3 * i + 1]), float(popt[3 * i + 2])
            fwhm = _sigma_to_fwhm(s, 0.5)
            area = float(a * fwhm * 0.85)
            comps.append(PeakComponent(center=c, amplitude=a, sigma=s, eta=0.5, fwhm=fwhm, area=area))
        yhat = _multi_pseudo_voigt(x, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        res_norm = float(np.sqrt(ss_res) / (y_rng * np.sqrt(x.size) + 1e-12))
        if r2 < 0.20 or res_norm > 0.92:
            continue
        areas = [c.area for c in comps]
        total_a = float(sum(areas)) + 1e-12
        dom = max(comps, key=lambda c: c.area)
        overlap = 0.0
        if len(comps) > 1:
            overlap = float(1.0 - dom.area / total_a)
        cand = RegionFitResult(
            region=region_id,
            success=True,
            r2=float(np.clip(r2, 0.0, 1.0)),
            residual_norm=res_norm,
            components=comps,
            overlap_index=overlap,
            total_area=total_a,
            max_height=float(max(c.amplitude for c in comps)),
            dominant_center=dom.center,
            dominant_fwhm=dom.fwhm,
            mean_fwhm=float(np.mean([c.fwhm for c in comps])),
            min_fwhm=float(np.min([c.fwhm for c in comps])),
        )
        if best is None or cand.r2 > best.r2:
            best = cand
    return best if best is not None else empty


def _empty_region(rid: str) -> RegionFitResult:
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


def deconvolve_spectrum(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "fast",
) -> dict[str, Any]:
    """
    Fit configured regions. Returns dict region_id -> RegionFitResult (+ empty for skipped).
    """
    m = str(mode or "fast").lower()
    if m == "off":
        return {
            "regions": {r[0]: _empty_region(r[0]) for r in DECONV_REGIONS},
            "profile_type": DECONV_PROFILE_TYPE,
            "deconv_mode": "off",
        }

    wn = np.asarray(wn, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]
    y_max = float(np.nanmax(y)) if y.size else 0.0
    maxfev = FAST_MAXFEV if m == "fast" else FULL_MAXFEV

    regions_out: dict[str, Any] = {r[0]: _empty_region(r[0]) for r in DECONV_REGIONS}
    for rid, lo, hi, max_n, min_sep, min_fwhm, max_fwhm in _regions_for_mode(m):
        try:
            x, seg = _crop_region(wn, y, lo, hi)
            if x.size == 0:
                continue
            if m == "fast" and y_max > 1e-9:
                rel = float(np.nanmax(seg) / y_max)
                if rel < FAST_MIN_REL_SIGNAL:
                    continue
            seg_bc = _baseline_correct(x, seg)
            fit = _fit_region(
                x,
                seg_bc,
                region_id=rid,
                max_components=max_n,
                min_sep=min_sep,
                min_fwhm=min_fwhm,
                max_fwhm=max_fwhm,
                maxfev=maxfev,
            )
            regions_out[rid] = fit
        except Exception:
            regions_out[rid] = _empty_region(rid)

    return {"regions": regions_out, "profile_type": DECONV_PROFILE_TYPE, "deconv_mode": m}


def extract_deconv_features(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "fast",
    track_stats: bool = True,
) -> tuple[np.ndarray, list[str], bool]:
    """
    Safe deconv feature extraction: never raises; returns zeros on total failure.
    Third return value is True if spectrum-level extraction failed.
    """
    names = stable_deconv_feature_names()
    vec = np.zeros(len(names), dtype=float)
    m = str(mode or "fast").lower()
    if m == "off":
        return vec, names, False

    stats = _GLOBAL_DECONV_STATS if track_stats else None
    t0 = time.perf_counter()
    failed = False
    try:
        deconv_result = deconvolve_spectrum(wn, y, mode=m)
        vec, names = deconv_feature_vector(wn, y, deconv_result=deconv_result)
        if not np.all(np.isfinite(vec)):
            vec = np.zeros(len(names), dtype=float)
            failed = True
    except Exception:
        vec = np.zeros(len(names), dtype=float)
        failed = True
        if stats is not None:
            stats.failure_count += 1
            stats.spectrum_failures += 1

    if stats is not None:
        stats.n_calls += 1
        stats.total_runtime_sec += time.perf_counter() - t0

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names, failed


def _region_to_common_features(fit: RegionFitResult) -> dict[str, float]:
    n = len(fit.components)
    return {
        "n_components": float(n),
        "total_area": float(fit.total_area),
        "max_height": float(fit.max_height),
        "dominant_center": float(fit.dominant_center),
        "dominant_fwhm": float(fit.dominant_fwhm),
        "mean_fwhm": float(fit.mean_fwhm),
        "min_fwhm": float(fit.min_fwhm if n else 0.0),
        "overlap_index": float(fit.overlap_index),
        "fit_r2": float(fit.r2),
        "residual_norm": float(fit.residual_norm),
        "fit_success": 1.0 if fit.success else 0.0,
    }


def _extra_features(regions: dict[str, RegionFitResult]) -> dict[str, float]:
    out = {n: 0.0 for n in EXTRA_FEATURE_NAMES}
    nitro = regions.get("nitro")
    carbonyl = regions.get("carbonyl")
    triple = regions.get("nitrile_alkyne")
    cosio = regions.get("c_o_sio_overlap")

    if nitro and nitro.success:
        asym = [c for c in nitro.components if 1500 <= c.center <= 1570]
        sym = [c for c in nitro.components if 1320 <= c.center <= 1390]
        a_area = float(sum(c.area for c in asym))
        s_area = float(sum(c.area for c in sym))
        out["deconv_nitro_asym_area"] = a_area
        out["deconv_nitro_sym_area"] = s_area
        out["deconv_nitro_asym_sym_ratio"] = float(a_area / (s_area + 1e-9))
        out["deconv_nitro_pair_present"] = 1.0 if a_area > 0 and s_area > 0 else 0.0

    if carbonyl and carbonyl.success:
        out["deconv_carbonyl_dominant_center"] = float(carbonyl.dominant_center)
        out["deconv_carbonyl_component_count"] = float(len(carbonyl.components))
        out["deconv_carbonyl_broadness"] = float(carbonyl.mean_fwhm / (carbonyl.min_fwhm + 1e-9))

    if triple and triple.success:
        sharp = [c for c in triple.components if c.fwhm <= 35.0]
        out["deconv_triple_bond_sharp_component_present"] = 1.0 if sharp else 0.0
        dom = max(triple.components, key=lambda c: c.amplitude) if triple.components else None
        if dom:
            out["deconv_triple_bond_center"] = float(dom.center)
            out["deconv_triple_bond_fwhm"] = float(dom.fwhm)

    if cosio and cosio.success:
        out["deconv_c_o_sio_component_count"] = float(len(cosio.components))
        out["deconv_c_o_sio_overlap_index"] = float(cosio.overlap_index)
        broad = float(sum(c.area for c in cosio.components if c.fwhm >= 35.0))
        sharp = float(sum(c.area for c in cosio.components if c.fwhm < 35.0))
        out["deconv_c_o_sio_broad_component_area"] = broad
        out["deconv_c_o_sio_sharp_component_area"] = sharp

    return out


def deconv_feature_vector(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    deconv_result: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = stable_deconv_feature_names()
    vec = np.zeros(len(names), dtype=float)
    if deconv_result is None:
        deconv_result = deconvolve_spectrum(wn, y, mode="fast")
    regions_raw = deconv_result.get("regions") or {}
    idx = {n: i for i, n in enumerate(names)}

    for rid, *_ in DECONV_REGIONS:
        fit = regions_raw.get(rid)
        if isinstance(fit, dict) and "components" in fit:
            comps = [
                PeakComponent(
                    center=float(c.get("center", 0)),
                    amplitude=float(c.get("amplitude", 0)),
                    sigma=float(c.get("fwhm", 10)) / 2.355,
                    eta=0.5,
                    fwhm=float(c.get("fwhm", 0)),
                    area=float(c.get("area", 0)),
                )
                for c in fit.get("components") or []
            ]
            fit = RegionFitResult(
                region=rid,
                success=bool(fit.get("success")),
                r2=float(fit.get("r2", 0)),
                residual_norm=float(fit.get("residual_norm", 1)),
                components=comps,
                overlap_index=float(fit.get("overlap_index", 0)),
                total_area=float(fit.get("total_area", 0)),
                max_height=float(fit.get("max_height", 0)),
                dominant_center=float(fit.get("dominant_center", 0)),
                dominant_fwhm=float(fit.get("dominant_fwhm", 0)),
                mean_fwhm=float(fit.get("mean_fwhm", 0)),
                min_fwhm=float(fit.get("min_fwhm", 0)),
            )
        if not isinstance(fit, RegionFitResult):
            continue
        common = _region_to_common_features(fit)
        for suf, val in common.items():
            key = f"deconv_{rid}_{suf}"
            if key in idx:
                vec[idx[key]] = val

    extras = _extra_features({k: v for k, v in regions_raw.items() if isinstance(v, RegionFitResult)})
    for k, v in extras.items():
        if k in idx:
            vec[idx[k]] = v

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names


def region_fit_to_dict(fit: RegionFitResult) -> dict[str, Any]:
    return {
        "region": fit.region,
        "success": fit.success,
        "r2": fit.r2,
        "residual_norm": fit.residual_norm,
        "overlap_index": fit.overlap_index,
        "total_area": fit.total_area,
        "max_height": fit.max_height,
        "dominant_center": fit.dominant_center,
        "dominant_fwhm": fit.dominant_fwhm,
        "mean_fwhm": fit.mean_fwhm,
        "min_fwhm": fit.min_fwhm,
        "n_components": len(fit.components),
        "components": [
            {
                "center": c.center,
                "amplitude": c.amplitude,
                "fwhm": c.fwhm,
                "area": c.area,
            }
            for c in fit.components
        ],
    }


def deconv_to_evidence_dict(deconv_result: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable deconv block for evidence / guardrails."""
    regions_raw = deconv_result.get("regions") or {}
    regions: dict[str, Any] = {}
    for k, v in regions_raw.items():
        if isinstance(v, RegionFitResult):
            regions[k] = region_fit_to_dict(v)
        elif isinstance(v, dict):
            regions[k] = v
    return {
        "profile_type": deconv_result.get("profile_type", DECONV_PROFILE_TYPE),
        "regions": regions,
        "disclaimer": "fitted component evidence (not ground truth)",
    }


def deconv_summary_for_report(deconv_result: dict[str, Any]) -> dict[str, Any]:
    """Compact summary for audit/detail report sections (not ground truth)."""
    rows: list[dict[str, Any]] = []
    for rid, *_ in DECONV_REGIONS:
        fit = (deconv_result.get("regions") or {}).get(rid)
        if isinstance(fit, RegionFitResult):
            rows.append(
                {
                    "region": rid,
                    "fit_success": fit.success,
                    "fit_r2": round(fit.r2, 4),
                    "n_components": len(fit.components),
                    "dominant_center_cm1": round(fit.dominant_center, 2),
                    "dominant_fwhm_cm1": round(fit.dominant_fwhm, 2),
                    "note": "fitted component evidence (not ground truth)",
                }
            )
        elif isinstance(fit, dict) and fit:
            rows.append(
                {
                    "region": rid,
                    "fit_success": bool(fit.get("success")),
                    "fit_r2": round(float(fit.get("r2", 0)), 4),
                    "n_components": int(fit.get("n_components", 0)),
                    "dominant_center_cm1": round(float(fit.get("dominant_center", 0)), 2),
                    "dominant_fwhm_cm1": round(float(fit.get("dominant_fwhm", 0)), 2),
                    "note": "fitted component evidence (not ground truth)",
                }
            )
    return {"profile_type": DECONV_PROFILE_TYPE, "regions": rows}


def deconv_audit_table_html(evidence: dict[str, Any]) -> str:
    """HTML table for audit report density only."""
    block = evidence.get("deconv")
    if not block:
        return ""
    summ = deconv_summary_for_report(block)
    rows = summ.get("regions") or []
    if not rows:
        return ""
    trs = "".join(
        (
            f"<tr><td>{r['region']}</td><td>{'yes' if r['fit_success'] else 'no'}</td>"
            f"<td>{r['fit_r2']}</td><td>{r['n_components']}</td>"
            f"<td>{r['dominant_center_cm1']}</td><td>{r['dominant_fwhm_cm1']}</td></tr>"
        )
        for r in rows
    )
    return (
        "<h4 class='section-sub'>Fitted component evidence (deconvolution)</h4>"
        "<p class='muted small'>Region-limited pseudo-Voigt fits — advisory only, not ground truth. "
        "Poor R² means do not rely on component counts for assignment.</p>"
        "<table class='data'><thead><tr><th>Region</th><th>OK</th><th>R²</th>"
        "<th># comp</th><th>Center (cm⁻¹)</th><th>FWHM</th></tr></thead><tbody>"
        f"{trs}</tbody></table>"
    )
