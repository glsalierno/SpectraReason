"""
Compact peak-codebook features for FTIR SVM (650–3700 cm⁻¹).

Binned peak statistics — interpretable, fixed dimension — not raw Kronecker deltas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lib.peaks import find_peaks_simple

PEAKCODEBOOK_WN_MIN = 650.0
PEAKCODEBOOK_WN_MAX = 3700.0
DEFAULT_N_BINS = 60

_STAT_SUFFIXES = (
    "peak_count_bin",
    "peak_max_height_bin",
    "peak_sum_height_bin",
    "peak_mean_sharpness_bin",
    "peak_max_support_bin",
)


def peakcodebook_bin_edges(
    *,
    wn_min: float = PEAKCODEBOOK_WN_MIN,
    wn_max: float = PEAKCODEBOOK_WN_MAX,
    n_bins: int = DEFAULT_N_BINS,
) -> np.ndarray:
    return np.linspace(float(wn_min), float(wn_max), int(n_bins) + 1)


def stable_peakcodebook_feature_names(
    *,
    n_bins: int = DEFAULT_N_BINS,
    wn_min: float = PEAKCODEBOOK_WN_MIN,
    wn_max: float = PEAKCODEBOOK_WN_MAX,
) -> list[str]:
    edges = peakcodebook_bin_edges(wn_min=wn_min, wn_max=wn_max, n_bins=n_bins)
    names: list[str] = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        for suf in _STAT_SUFFIXES:
            names.append(f"{suf}_{lo}_{hi}")
    return names


def _bin_index(wn: float, edges: np.ndarray) -> int:
    if wn < edges[0] or wn >= edges[-1]:
        return -1
    j = int(np.searchsorted(edges, wn, side="right") - 1)
    if j < 0 or j >= len(edges) - 1:
        return -1
    return j


def peakcodebook_feature_vector(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    evidence: dict[str, Any] | None = None,
    n_bins: int = DEFAULT_N_BINS,
    wn_min: float = PEAKCODEBOOK_WN_MIN,
    wn_max: float = PEAKCODEBOOK_WN_MAX,
    max_peaks: int = 48,
) -> tuple[np.ndarray, list[str]]:
    """
    Build binned peak statistics from detected peaks and optional band-match support.
    """
    wn = np.asarray(wn, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]

    edges = peakcodebook_bin_edges(wn_min=wn_min, wn_max=wn_max, n_bins=n_bins)
    names = stable_peakcodebook_feature_names(n_bins=n_bins, wn_min=wn_min, wn_max=wn_max)
    n_feat = len(names)
    vec = np.zeros(n_feat, dtype=float)

    pwn, ph = find_peaks_simple(wn, y, max_peaks=max_peaks)
    peaks = [{"wn": float(a), "height": float(b)} for a, b in zip(pwn, ph)]

    support_by_wn: dict[float, float] = {}
    if evidence:
        for bm in evidence.get("band_matches") or []:
            if not isinstance(bm, dict):
                continue
            pw = bm.get("peak_wn") or bm.get("nearest_peak_wn")
            if pw is None:
                continue
            sup = float(bm.get("support_score") or bm.get("score") or 0.0)
            support_by_wn[float(pw)] = max(support_by_wn.get(float(pw), 0.0), sup)

    # Per-bin accumulators
    counts = [0] * n_bins
    heights: list[list[float]] = [[] for _ in range(n_bins)]
    sharpness: list[list[float]] = [[] for _ in range(n_bins)]
    supports: list[list[float]] = [[] for _ in range(n_bins)]

    y_rng = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    noise = float(np.nanstd(np.diff(y))) + 1e-9 if y.size > 4 else 1e-9

    for p in peaks:
        pw = float(p["wn"])
        bi = _bin_index(pw, edges)
        if bi < 0:
            continue
        h = float(p["height"])
        counts[bi] += 1
        heights[bi].append(h)
        idx = int(np.argmin(np.abs(wn - pw)))
        local = y[max(0, idx - 3) : min(wn.size, idx + 4)]
        base = float(np.nanmin(local)) if local.size else 0.0
        prom = max(h - base, 0.0)
        sharp = float(prom / (noise * y_rng + 1e-9))
        sharpness[bi].append(sharp)
        sup = support_by_wn.get(pw, 0.0)
        supports[bi].append(sup)

    for bi in range(n_bins):
        base_i = bi * len(_STAT_SUFFIXES)
        vec[base_i + 0] = float(counts[bi])
        if heights[bi]:
            vec[base_i + 1] = float(max(heights[bi]))
            vec[base_i + 2] = float(sum(heights[bi]))
            vec[base_i + 3] = float(np.mean(sharpness[bi])) if sharpness[bi] else 0.0
            vec[base_i + 4] = float(max(supports[bi])) if supports[bi] else 0.0

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names


def peakcodebook_meta(
    *,
    n_bins: int = DEFAULT_N_BINS,
    wn_min: float = PEAKCODEBOOK_WN_MIN,
    wn_max: float = PEAKCODEBOOK_WN_MAX,
) -> dict[str, Any]:
    edges = peakcodebook_bin_edges(wn_min=wn_min, wn_max=wn_max, n_bins=n_bins)
    width = float((wn_max - wn_min) / n_bins) if n_bins else 0.0
    return {
        "n_peakcodebook": int(n_bins * len(_STAT_SUFFIXES)),
        "peakcodebook_n_bins": int(n_bins),
        "peakcodebook_bin_width": round(width, 4),
        "peakcodebook_wn_min": wn_min,
        "peakcodebook_wn_max": wn_max,
    }
