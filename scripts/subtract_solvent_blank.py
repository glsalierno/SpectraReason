#!/usr/bin/env python3
"""
Subtract a solvent blank spectrum from a sample (scaled absorbance difference).

Scaling matches blank intensity to sample in a reference band (default C–O ~1040 cm⁻¹)
on baseline-corrected absorbance before subtraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lib.ftir_foundation import baseline_movmin, read_spectrum, to_absorbance


def _band_median(wn: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    m = (wn >= lo) & (wn <= hi)
    vals = y[m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def subtract_scaled_blank(
    sample_path: Path,
    blank_path: Path,
    *,
    ref_lo: float = 1020.0,
    ref_hi: float = 1060.0,
    movmin_window: int = 151,
) -> tuple[np.ndarray, np.ndarray, dict]:
    wn_s, raw_s, hint_s = read_spectrum(sample_path)
    wn_b, raw_b, hint_b = read_spectrum(blank_path)
    if wn_s.shape != wn_b.shape or not np.allclose(wn_s, wn_b, rtol=0, atol=1e-3):
        raise ValueError(
            "Sample and blank must share the same wavenumber grid; "
            f"got {wn_s.size} vs {wn_b.size} points."
        )

    ab_s, mode_s = to_absorbance(raw_s, hint_s)
    ab_b, mode_b = to_absorbance(raw_b, hint_b)
    corr_s = baseline_movmin(ab_s, movmin_window)
    corr_b = baseline_movmin(ab_b, movmin_window)

    s_ref = _band_median(wn_s, corr_s, ref_lo, ref_hi)
    b_ref = _band_median(wn_s, corr_b, ref_lo, ref_hi)
    if not np.isfinite(s_ref) or not np.isfinite(b_ref) or b_ref <= 1e-12:
        scale = 1.0
    else:
        scale = float(s_ref / b_ref)

    diff = corr_s - scale * corr_b
    meta = {
        "sample": str(sample_path.resolve()),
        "blank": str(blank_path.resolve()),
        "sample_intensity_mode": mode_s,
        "blank_intensity_mode": mode_b,
        "reference_band_cm1": [ref_lo, ref_hi],
        "reference_band_label": "C-O stretch (~1040 cm-1)",
        "sample_median_absorbance_in_band": round(s_ref, 8) if np.isfinite(s_ref) else None,
        "blank_median_absorbance_in_band": round(b_ref, 8) if np.isfinite(b_ref) else None,
        "blank_scale_factor": round(scale, 6),
        "subtraction_stage": "baseline_corrected_absorbance",
        "n_points": int(wn_s.size),
    }
    return wn_s, diff, meta


def write_difference_csv(path: Path, wn: np.ndarray, absorbance: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x, a in zip(wn, absorbance):
            f.write(f"{float(x):.6e},{float(a):.6e}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Scaled solvent blank subtraction for FTIR CSV/JDX.")
    ap.add_argument("--sample", required=True, help="Sample spectrum path")
    ap.add_argument("--blank", required=True, help="Solvent blank spectrum path")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--meta-out", default="", help="Optional JSON metadata sidecar")
    ap.add_argument("--ref-lo", type=float, default=1020.0)
    ap.add_argument("--ref-hi", type=float, default=1060.0)
    args = ap.parse_args()

    wn, diff, meta = subtract_scaled_blank(
        Path(args.sample),
        Path(args.blank),
        ref_lo=float(args.ref_lo),
        ref_hi=float(args.ref_hi),
    )
    out = Path(args.out)
    write_difference_csv(out, wn, diff)
    print(json.dumps({"out": str(out.resolve()), **meta}, indent=2))
    if args.meta_out:
        Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
