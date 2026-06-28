#!/usr/bin/env python3
"""
Phase 1–2 foundation for a Python FTIR → ML pipeline: unified IO, absorbance
conversion, baseline correction, Savitzky–Golay smoothing, and [0, 1] scaling.

ProSpecPy (optional): core preprocessing in this file uses NumPy/SciPy so runs
without anchor picking. For spline/anchor workflows, see prospecpy.baseline and
prospecpy.prospecpy.ProSpecPy in the installed package.

Example:
  python ftir_foundation.py
  python ftir_foundation.py --csv Dopamine_Polydopamine.CSV
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

IntensityMode = Literal["auto", "transmittance_percent", "transmittance_fraction", "absorbance"]
BaselineMode = Literal["movmin", "als"]
PreprocessBackend = Literal["native", "spectrochempy"]


def _read_jdx_xydata(text_lines: list[str]) -> tuple[dict[str, str], int | None]:
    meta: dict[str, str] = {}
    xy_idx: int | None = None
    for i, line in enumerate(text_lines):
        if line.startswith("##"):
            m = re.match(r"##([A-Z0-9_ -]+)=(.*)", line)
            if m:
                meta[m.group(1).strip().upper()] = m.group(2).strip()
            if line.upper().startswith("##XYDATA"):
                xy_idx = i + 1
                break
    return meta, xy_idx


def read_jdx_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Parse common NIST-style IR JCAMP with ##XYDATA=(X++(Y..Y)).

    Returns (wavenumber, intensity, y_units_hint) where intensity is raw
    (transmittance or absorbance per ##YUNITS / ##YFACTOR).
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta, xy_idx = _read_jdx_xydata(text)
    if xy_idx is None:
        raise ValueError(f"No ##XYDATA block in {path}")

    firstx = float(meta.get("FIRSTX", "0"))
    lastx = float(meta.get("LASTX", "0"))
    deltax = float(meta.get("DELTAX", "1"))
    yfac = float(meta.get("YFACTOR", "1"))
    yunits = meta.get("YUNITS", "TRANSMITTANCE").upper()

    xs: list[float] = []
    ys: list[float] = []
    sign = -1.0 if firstx > lastx else 1.0
    d = abs(deltax) if deltax != 0 else 1.0

    for line in text[xy_idx:]:
        if line.startswith("##"):
            break
        line = line.strip()
        if not line or line.startswith("$"):
            continue
        parts = [float(x) for x in line.replace(",", " ").split()]
        if len(parts) < 2:
            continue
        x0 = parts[0]
        for j, raw_y in enumerate(parts[1:]):
            x = x0 + sign * j * d
            yv = raw_y * yfac
            xs.append(x)
            ys.append(yv)

    wn = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    return wn, y, yunits


def read_csv_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load two-column wavenumber / intensity CSV.

    Supports files with a header row (e.g. ``Wavenumber_cm1,Delta_Absorbance``) by
    retrying with ``header=0``. Drops leading rows until both columns parse as finite
    floats. Requires at least ``min_points`` usable rows so short assignment tables are
    rejected as spectra.
    """
    min_points = 10

    def _xy_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if frame.shape[1] < 2:
            raise ValueError(f"Expected ≥2 columns (wavenumber, intensity) in {path}")
        wn = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        while wn.size > 0 and (not np.isfinite(wn[0]) or not np.isfinite(y[0])):
            wn, y = wn[1:], y[1:]
        m = np.isfinite(wn) & np.isfinite(y)
        wn, y = wn[m], y[m]
        # Drop zero/near-zero %T padding rows (common in instrument exports; breaks -log10(T)).
        if wn.size >= min_points:
            mx = float(np.nanmax(y))
            mn = float(np.nanmin(y))
            if mx <= 120.0 and mn >= 0.0:
                keep = y > 1e-3
                wn, y = wn[keep], y[keep]
        return wn, y

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    raw = pd.read_csv(path, header=None)
    candidates.append(_xy_from_frame(raw))
    try:
        hdr = pd.read_csv(path, header=0)
        candidates.append(_xy_from_frame(hdr))
    except Exception:
        pass

    best: tuple[np.ndarray, np.ndarray] | None = None
    for wn, y in candidates:
        if wn.size >= min_points:
            return wn, y
        if best is None or wn.size > best[0].size:
            best = (wn, y)

    if best is not None and best[0].size >= 2:
        raise ValueError(
            f"Too few numeric XY rows in {path} (got {best[0].size}, need ≥{min_points}); "
            "file may be a summary table rather than a spectrum."
        )
    raise ValueError(f"No finite numeric spectrum rows in {path}")


def read_spectrum(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, IntensityMode]:
    """
    Load a spectrum from .csv or .jdx.

    Returns (wavenumber, raw_intensity, suggested_mode) where suggested_mode is
    a hint for intensity interpretation (CSV always returns 'auto'; JDX may
    return 'absorbance' or 'transmittance_percent' from metadata).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()
    if suf == ".csv":
        wn, y = read_csv_spectrum(path)
        return wn, y, "auto"
    if suf == ".jdx":
        wn, y, yu = read_jdx_spectrum(path)
        if "ABSORB" in yu:
            return wn, y, "absorbance"
        if "TRANSMITT" in yu:
            return wn, y, "transmittance_percent"
        return wn, y, "auto"
    raise ValueError(f"Unsupported extension {path.suffix!r} (use .csv or .jdx)")


def infer_intensity_mode(y: np.ndarray) -> IntensityMode:
    """Best-effort guess for CSV exports without ##YUNITS."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return "absorbance"
    mx = float(np.nanmax(y))
    mn = float(np.nanmin(y))
    med = float(np.nanmedian(y))

    if mx <= 1.05 and mn >= 0.0:
        return "transmittance_fraction"
    # Typical %T plateaus near 100 with dips at peaks; absorbance rarely has median > 20.
    if 0.0 <= mn and mx <= 120.0 and (med > 25.0 or float(np.nanpercentile(y, 90)) > 60.0):
        return "transmittance_percent"
    return "absorbance"


def transmittance_to_absorbance(t: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Convert transmittance to absorbance: A = -log10(clip(T, eps) / 100) for percent input."""
    t_arr = np.asarray(t, dtype=float)
    mx = float(np.nanmax(t_arr)) if t_arr.size else 0.0
    if mx <= 1.05:
        t_clip = np.clip(t_arr, eps, None)
        return -np.log10(t_clip)
    t_clip = np.clip(t_arr, eps, None)
    return -np.log10(t_clip / 100.0)


def to_absorbance(y: np.ndarray, mode: IntensityMode = "auto") -> tuple[np.ndarray, IntensityMode]:
    """
    Convert intensity to absorbance. Returns (absorbance, resolved_mode).
    """
    y = np.asarray(y, dtype=float)
    if mode == "auto":
        mode = infer_intensity_mode(y)
    if mode == "absorbance":
        return y, mode
    if mode == "transmittance_fraction":
        return transmittance_to_absorbance(y), mode
    if mode == "transmittance_percent":
        return transmittance_to_absorbance(y), mode
    raise ValueError(f"Unknown intensity mode: {mode!r}")


def baseline_movmin(y: np.ndarray, window: int = 151) -> np.ndarray:
    """Rolling minimum baseline (fast, stable default for FTIR)."""
    window = max(3, int(window) | 1)
    pad = window // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = y[i] - np.min(yp[i : i + window])
    return out


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 10) -> np.ndarray:
    """
    Asymmetric least squares baseline (Eilers & Boelens). Returns y - baseline.
    """
    y = np.asarray(y, dtype=float)
    L = y.size
    # D: second differences (L x L)
    D = np.diff(np.eye(L), n=2, axis=0)
    D = csc_matrix(D)
    H = lam * (D.T @ D)
    w = np.ones(L, dtype=float)
    for _ in range(niter):
        W = csc_matrix((w, (np.arange(L), np.arange(L))))
        C = W + H
        z = spsolve(C, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return y - z


def preprocess_spectrum(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    *,
    intensity_mode: IntensityMode = "auto",
    baseline: BaselineMode = "movmin",
    movmin_window: int = 151,
    als_lambda: float = 1e5,
    als_p: float = 0.001,
    als_iters: int = 10,
    sg_window: int = 11,
    sg_poly: int = 2,
    normalize: bool = True,
    preprocess_backend: PreprocessBackend = "native",
    scp_derivative_order: int = 0,
    scp_resample_points: int | None = None,
    stash_pre_smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Full preprocessing: sort by wavenumber, absorbance conversion, baseline,
    Savitzky–Golay smoothing, optional min–max to [0, 1].

    Returns (wavenumber_sorted, absorbance_processed, info_dict).

    When ``stash_pre_smooth`` is True, ``info`` includes ``baseline_corrected_pre_sg``
    (absorbance after baseline correction, before Savitzky–Golay) and
    ``smoothed_pre_normalize`` (after SG / derivative / resample, before [0,1] scaling)
    for plotting overlays (e.g. denoise vs raw-ish curves on a common scale).
    """
    wn = np.asarray(wavenumber, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if wn.shape != y.shape:
        raise ValueError("wavenumber and intensity must have the same shape")

    o = np.argsort(wn)
    wn = wn[o]
    y = y[o]

    ab, used_mode = to_absorbance(y, intensity_mode)

    backend_used = "native"
    corrected = None
    if preprocess_backend == "spectrochempy":
        try:
            import spectrochempy as scp  # type: ignore

            # Optional backend: only used when available and successful.
            ds = scp.NDDataset(ab.copy())
            if baseline == "als" and hasattr(ds, "baseline"):
                try:
                    dsb = ds.baseline(model="asls")
                    corrected = np.asarray(dsb.data, dtype=float).ravel()
                except Exception:
                    corrected = None
            if corrected is None and baseline == "movmin":
                corrected = baseline_movmin(ab, movmin_window)
            if corrected is None and baseline == "als":
                corrected = baseline_als(ab, lam=als_lambda, p=als_p, niter=als_iters)
            backend_used = "spectrochempy"
        except Exception:
            corrected = None
    if corrected is None:
        if baseline == "movmin":
            corrected = baseline_movmin(ab, movmin_window)
        elif baseline == "als":
            corrected = baseline_als(ab, lam=als_lambda, p=als_p, niter=als_iters)
        else:
            raise ValueError(f"Unknown baseline mode: {baseline!r}")

    info_pre: dict[str, Any] = {}
    if stash_pre_smooth:
        info_pre["baseline_corrected_pre_sg"] = corrected.astype(float).copy()

    sg_window_use = int(sg_window) | 1
    sg_window_use = min(max(3, sg_window_use), len(corrected) - (1 - len(corrected) % 2))
    if len(corrected) < sg_window_use + 2:
        smoothed = corrected.astype(float)
    else:
        smoothed = savgol_filter(corrected, window_length=sg_window_use, polyorder=int(sg_poly))

    if int(scp_derivative_order) > 0:
        # optional derivative regardless of backend, controlled by new param
        smoothed = np.gradient(smoothed, edge_order=1)

    if scp_resample_points is not None and int(scp_resample_points) >= 32:
        npt = int(scp_resample_points)
        g = np.linspace(float(np.min(wn)), float(np.max(wn)), npt)
        smoothed = np.interp(g, wn, smoothed)
        wn = g

    out = smoothed.astype(float)
    info: dict = {
        "intensity_mode": used_mode,
        "baseline": baseline,
        "normalized": False,
        "preprocess_backend_requested": preprocess_backend,
        "preprocess_backend_used": backend_used,
        "derivative_order": int(max(0, scp_derivative_order)),
        "resample_points": int(scp_resample_points) if scp_resample_points is not None else None,
    }
    if stash_pre_smooth:
        info["baseline_corrected_pre_sg"] = info_pre["baseline_corrected_pre_sg"]
        info["smoothed_pre_normalize"] = out.astype(float).copy()
    if normalize:
        lo = float(np.nanmin(out))
        hi = float(np.nanmax(out))
        span = hi - lo
        if span <= 0 or not np.isfinite(span):
            raise ValueError("Cannot normalize: non-finite or flat spectrum after processing")
        out = (out - lo) / span
        info["normalized"] = True
        info["norm_min"] = lo
        info["norm_max"] = hi

    return wn, out, info


def try_prospecpy_raw_spline_baseline(
    wavenumber: np.ndarray, absorbance: np.ndarray
) -> np.ndarray | None:
    """
    Optional: use prospecpy.baseline.raw_spline to estimate a smooth baseline,
    subtract it after interpolation to original grid. Returns None if ProSpecPy
    is missing or fails.
    """
    try:
        from prospecpy.baseline import raw_spline  # type: ignore
    except Exception:
        return None
    try:
        xs, ys = raw_spline(np.asarray(wavenumber, float), np.asarray(absorbance, float))
        base = np.interp(wavenumber, xs, ys)
        return absorbance - base
    except Exception:
        return None


def demo(csv_path: Path) -> None:
    import matplotlib.pyplot as plt

    wn, raw, hint = read_spectrum(csv_path)
    wn_p, ab_p, info = preprocess_spectrum(wn, raw, intensity_mode=hint)

    print(f"File: {csv_path.name}")
    print(f"  Points: {len(wn)}  wavenumber range: {wn.min():.1f} - {wn.max():.1f} cm^-1")
    print(f"  Preprocess: {info}")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(wn, raw, color="0.55", lw=0.8, label="raw (CSV column 2)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (as loaded)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_dir = csv_path.parent / "ftir_foundation_demo"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "raw_column2.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 4.2))
    ax2.plot(wn_p, ab_p, color="0.1", lw=0.9, label="processed absorbance [0,1]")
    ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax2.set_ylabel("Processed (normalized)")
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "processed.png", dpi=150)
    plt.close(fig2)
    print(f"  Wrote figures under: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FTIR foundation: read + preprocess (Phase 1–2).")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("Dopamine_Polydopamine.CSV"),
        help="Sample two-column CSV (wavenumber, intensity)",
    )
    args = ap.parse_args()
    demo(args.csv.resolve())


if __name__ == "__main__":
    main()
