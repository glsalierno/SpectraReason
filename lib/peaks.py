"""Peak picking for reports (delegates to ml.ftir_peak_picking presets)."""

from __future__ import annotations

import numpy as np


def find_peaks_simple(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    max_peaks: int = 40,
    *,
    sensitivity: str = "balanced",
) -> tuple[list[float], list[float]]:
    from ml.ftir_peak_picking import pick_peaks_simple_compat

    return pick_peaks_simple_compat(
        wavenumber_cm, absorbance, max_peaks=max_peaks, sensitivity=sensitivity
    )
