"""
Interpretable spectral features for FTIR functional-group explanation.

Parallel to the 14-D training featurizer — designed for human-readable evidence mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lib.peaks import find_peaks_simple

# Finer windows than training 14-D block (cm⁻¹).
INTERPRETABLE_REGIONS: list[tuple[str, int, int]] = [
    ("oh_nh_broad", 2500, 3700),
    ("ch_stretch", 2800, 3100),
    ("nh_ch_transition", 3100, 3200),
    ("aromatic_ch_stretch", 3000, 3100),
    ("aliphatic_ch", 2850, 2965),
    ("aldehydic_ch", 2720, 2820),
    ("upper_mid_activity", 2260, 2800),
    ("carbonyl_overtone", 1820, 2100),
    ("alkyne_terminal_ch", 3280, 3340),
    ("alkyne_cc", 2100, 2260),
    ("nitrile", 2200, 2260),
    ("carbonyl", 1650, 1820),
    ("amide_i", 1630, 1690),
    ("ester_co", 1730, 1760),
    ("aromatic_cc", 1450, 1600),
    ("nitro_asym", 1500, 1570),
    ("c_o_stretch", 1000, 1300),
    ("si_o", 1000, 1150),
    ("aromatic_oop", 650, 900),
    ("fingerprint", 900, 1400),
]

# Spectrum plot shading (may overlap sub-windows for upper-mid coverage); see ml/ftir_shade_regions.py.
from ml.ftir_shade_regions import (  # noqa: E402
    spectrum_shade_evidence_keys,
    spectrum_shade_regions_legacy,
)

SPECTRUM_SHADE_REGIONS = spectrum_shade_regions_legacy()
SPECTRUM_SHADE_EVIDENCE_KEYS = spectrum_shade_evidence_keys()


def _segment_stats(wn: np.ndarray, y: np.ndarray, lo: float, hi: float) -> dict[str, float]:
    m = (wn >= lo) & (wn <= hi)
    if int(np.count_nonzero(m)) < 3:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "integral": 0.0, "n_points": 0.0}
    seg = y[m]
    wseg = wn[m]
    span = max(float(wseg[-1] - wseg[0]), 1e-6) if wseg.size > 1 else 1.0
    return {
        "mean": float(np.mean(seg)),
        "std": float(np.std(seg)),
        "max": float(np.max(seg)),
        "integral": float(np.trapz(seg, wseg)),
        "n_points": float(seg.size),
    }


def oh_broadness_metric(wn: np.ndarray, y: np.ndarray) -> float:
    """Higher → broader O-H/N-H envelope (relative std in 2500–3700)."""
    st = _segment_stats(wn, y, 2500, 3700)
    if st["mean"] < 1e-9:
        return 0.0
    return float(st["std"] / (st["mean"] + 1e-9))


def featurize_interpretable(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 48,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """
    Return (feature_vector, names, extras) where extras holds peak lists and per-region stats.
    """
    wn = np.asarray(wn, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]

    names: list[str] = []
    vec: list[float] = []
    region_stats: dict[str, dict[str, float]] = {}

    for rname, lo, hi in INTERPRETABLE_REGIONS:
        st = _segment_stats(wn, y, lo, hi)
        region_stats[rname] = st
        for k in ("mean", "std", "max", "integral"):
            names.append(f"interp_{rname}_{k}")
            vec.append(st[k])

    names.append("interp_oh_nh_broadness")
    vec.append(oh_broadness_metric(wn, y))

    y_rng = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    arom = region_stats.get("aromatic_cc", {})
    fp = region_stats.get("fingerprint", {})
    names.append("interp_aromatic_to_fingerprint_ratio")
    vec.append(float(arom.get("max", 0.0) / (fp.get("max", 0.0) + 1e-9)))

    pwn, ph = find_peaks_simple(wn, y, max_peaks=max_peaks)
    peak_records = [{"wn": float(a), "height": float(b)} for a, b in zip(pwn, ph)]

    # Top-N peak heights as sparse features (fixed slots by wavenumber bins)
    bin_edges = [650, 900, 1200, 1500, 1700, 2000, 2300, 2800, 3700]
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [p for p in peak_records if lo <= p["wn"] < hi]
        hmax = max((p["height"] for p in in_bin), default=0.0)
        names.append(f"interp_peak_max_{lo}_{hi}")
        vec.append(float(hmax))

    extras = {
        "region_stats": region_stats,
        "peaks": peak_records,
        "y_range": y_rng,
    }
    return np.asarray(vec, dtype=float), names, extras


def peaks_near_band(
    peaks: list[dict[str, Any]],
    lo: float,
    hi: float,
    *,
    tolerance_cm1: float = 25.0,
) -> list[dict[str, Any]]:
    out = []
    for p in peaks:
        w = float(p.get("wn_cm1", p.get("wn", 0)))
        if lo - tolerance_cm1 <= w <= hi + tolerance_cm1:
            out.append(p)
    return sorted(out, key=lambda x: -float(x.get("height", 0)))
