"""
Training-only spectrum perturbations (wavenumber + absorbance arrays).

Applied after molecule-level train/test split; never on held-out test molecules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AUGMENT_MODES = ("none", "light", "moderate")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def augment_spectrum(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "light",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return perturbed (wn, y) copies. ``none`` returns inputs unchanged.
    """
    mode = str(mode or "none").lower()
    if mode not in AUGMENT_MODES or mode == "none":
        return np.asarray(wn, float).copy(), np.asarray(y, float).copy()

    wn = np.asarray(wn, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(wn)
    wn, y = wn[order].copy(), y[order].copy()
    g = _rng(seed)

    if mode == "light":
        noise_scale = 0.012 * float(np.nanstd(y) + 1e-9)
        y = y + g.normal(0.0, noise_scale, size=y.shape)
        scale = float(g.uniform(0.92, 1.08))
        y = y * scale
        slope = float(g.uniform(-0.02, 0.02))
        y = y + slope * (wn - float(np.mean(wn))) / max(float(np.ptp(wn)), 1.0)
        shift = float(g.uniform(-3.0, 3.0))
        wn = wn + shift
    else:  # moderate
        noise_scale = 0.028 * float(np.nanstd(y) + 1e-9)
        y = y + g.normal(0.0, noise_scale, size=y.shape)
        scale = float(g.uniform(0.85, 1.15))
        y = y * scale
        c0, c1, c2 = g.uniform(-0.04, 0.04, size=3)
        xn = (wn - float(np.min(wn))) / max(float(np.ptp(wn)), 1.0)
        y = y + c0 + c1 * xn + c2 * (xn**2)
        shift = float(g.uniform(-8.0, 8.0))
        wn = wn + shift
        if wn.size > 32 and g.random() < 0.5:
            step = int(g.integers(2, 4))
            wn = wn[::step]
            y = y[::step]
        if wn.size > 16:
            win = int(g.choice([5, 7, 9]))
            if win % 2 == 0:
                win += 1
            from scipy.ndimage import uniform_filter1d

            y = uniform_filter1d(y, size=win, mode="nearest")

    y = np.clip(y, 0.0, None)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    wn = np.nan_to_num(wn, nan=0.0)
    order = np.argsort(wn)
    return wn[order], y[order]


def augmentation_settings(mode: str) -> dict[str, Any]:
    mode = str(mode or "none").lower()
    return {
        "augmentation_mode": mode,
        "augmentation_train_only": mode != "none",
        "augmentation_settings": {
            "light": "gaussian_noise, intensity_scale, baseline_slope, wn_shift",
            "moderate": "stronger_noise, curved_baseline, wn_shift, optional_downsample, smoothing",
        }.get(mode, "none"),
    }
