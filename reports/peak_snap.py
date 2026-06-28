"""Snap requested wavenumbers to local absorbance maxima or transmittance minima."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from reports.paper_peak_selection import classify_label_region, region_title

LabelMode = Literal["normalized_absorbance", "transmittance"]
SnapTarget = Literal["max_absorbance", "min_transmittance"]
DEFAULT_SNAP_WINDOW_CM1 = 25.0


def snap_peak_at_wavenumber(
    requested_cm1: float,
    wn: np.ndarray,
    y: np.ndarray,
    *,
    mode: LabelMode,
    window_cm1: float = DEFAULT_SNAP_WINDOW_CM1,
) -> dict[str, Any]:
    """Find nearest local extremum to *requested_cm1* within ±*window_cm1*."""
    wn = np.asarray(wn, dtype=float)
    y = np.asarray(y, dtype=float)
    if wn.size == 0 or y.size == 0:
        raise ValueError("empty spectrum for peak snap")

    req = float(requested_cm1)
    win = max(float(window_cm1), 1.0)
    mask = (wn >= req - win) & (wn <= req + win)
    snap_status = "local_extremum"
    if not np.any(mask):
        idx = int(np.argmin(np.abs(wn - req)))
        snap_status = "nearest_point"
    else:
        local_idx = np.where(mask)[0]
        if mode == "transmittance":
            idx = int(local_idx[int(np.argmin(y[local_idx]))])
        else:
            idx = int(local_idx[int(np.argmax(y[local_idx]))])

    snapped = float(wn[idx])
    intensity = float(y[idx])
    region = classify_label_region(snapped) or ""
    snap_target: SnapTarget = "min_transmittance" if mode == "transmittance" else "max_absorbance"
    return {
        "requested_wavenumber_cm1": req,
        "snapped_wavenumber_cm1": snapped,
        "wavenumber_cm1": snapped,
        "peak_y": intensity,
        "intensity": intensity,
        "mode": mode,
        "snap_target": snap_target,
        "snap_window_cm1": win,
        "region": region,
        "region_title": region_title(region) if region else "",
        "snap_status": snap_status,
    }
