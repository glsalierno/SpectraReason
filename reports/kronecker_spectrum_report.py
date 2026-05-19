#!/usr/bin/env python3
# Legacy/specialized entry point. Current production report is reports/structural_fg_svm_kronecker_report.py with --report-style product_v1.
"""
HTML report: **baseline-corrected + SG-smoothed** spectrum with **Kronecker / stem**
peak featurization (same spirit as ``advanced_ftir_html_report``), stacked vertically.

Optional **pre–Savitzky–Golay** overlay (baseline-corrected only) helps tune denoising:
raise ``--sg-window`` (odd) in small steps to suppress high-frequency noise while keeping
peak positions (polyorder 2 is typical).

Run from **FTIR_SVM** root::

    cd FTIR_SVM
    $env:PYTHONPATH = (Get-Location).Path  # FTIR_SVM
    python reports/kronecker_spectrum_report.py batch \\
      --inputs Dopamine_Powder.CSV Polydopamine_Powder.CSV \\
      --out-dir reports/powder_kronecker_sg19 \\
      --sg-window 19 --sg-poly 2 --show-pre-sg-overlay
"""

from __future__ import annotations

import argparse
import html
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scipy.signal import find_peaks

from lib.ftir_foundation import preprocess_spectrum, read_spectrum


def _resolve_under_chunks(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def find_peaks_scaled(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    *,
    prominence_scale: float = 1.0,
    max_peaks: int = 48,
) -> tuple[list[float], list[float]]:
    """Like ``find_peaks_simple`` but scales the relative prominence threshold."""
    x = np.asarray(wavenumber_cm, dtype=float)
    y = np.asarray(absorbance, dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if len(y) < 5:
        return [], []
    mx, mn = float(np.nanmax(y)), float(np.nanmin(y))
    prom = max(1e-9, 0.012 * float(prominence_scale) * (mx - mn))
    height = mn + 0.012 * float(prominence_scale) * (mx - mn)
    peaks, _props = find_peaks(y, prominence=prom, height=height)
    locs = x[peaks].tolist()
    heights = y[peaks].tolist()
    pairs = sorted(zip(locs, heights), key=lambda t: t[0])[:max_peaks]
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _stacked_figure_png(
    wn: np.ndarray,
    proc: np.ndarray,
    peak_wn: list[float],
    peak_h: list[float],
    *,
    title: str,
    show_pre_sg_overlay: bool,
    info: dict[str, Any],
) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    wn = np.asarray(wn, dtype=float)
    proc = np.asarray(proc, dtype=float)
    px = np.asarray(peak_wn, dtype=float)
    ph = np.asarray(peak_h, dtype=float)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0]},
    )
    if show_pre_sg_overlay and info.get("baseline_corrected_pre_sg") is not None:
        pre = np.asarray(info["baseline_corrected_pre_sg"], dtype=float)
        sm0 = np.asarray(info.get("smoothed_pre_normalize", proc), dtype=float)
        if pre.shape == sm0.shape == wn.shape:
            lo = min(float(np.nanmin(sm0)), float(np.nanmin(pre)))
            hi = max(float(np.nanmax(sm0)), float(np.nanmax(pre)))
            span = max(hi - lo, 1e-12)
            ax0.plot(wn, (pre - lo) / span, color="#ea580c", alpha=0.38, lw=0.75, label="Baseline-corr. (pre-SG)")
    ax0.plot(wn, proc, color="#0f172a", lw=0.95, label="Processed (norm., SG-smoothed)")
    if px.size:
        ax0.scatter(
            px,
            ph,
            color="#dc2626",
            s=32,
            zorder=5,
            marker="o",
            edgecolors="white",
            linewidths=0.45,
            label="Picked peaks",
        )
    ax0.set_ylabel("Normalized absorbance")
    ax0.set_title(title[:140], fontsize=10)
    ax0.legend(loc="upper right", fontsize=7.5)
    ax0.grid(True, alpha=0.25)
    ax0.invert_xaxis()

    if px.size:
        ax1.stem(px, ph, linefmt="C0-", markerfmt="D", basefmt=" ", label="Kronecker weights")
    else:
        ax1.text(0.5, 0.5, "(no peaks)", transform=ax1.transAxes, ha="center", va="center", fontsize=10, color="#64748b")
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Peak height\n(same units as top)")
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.grid(True, alpha=0.25)
    ax1.invert_xaxis()
    ax1.set_title("Kronecker / discrete impulses at picked ν", fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        plt.close(fig)
        return None


def _denoise_notes_html() -> str:
    return """
<h2>Denoising (keep peaks)</h2>
<ul>
  <li><strong>Savitzky–Golay</strong> (<code>--sg-window</code> odd, <code>--sg-poly 2</code>): widens the
      smoothing window to follow curvature; usually <strong>preserves peak positions</strong> better than a plain moving average.
      Try <code>11 → 15 → 19 → 21</code> and compare overlays.</li>
  <li><strong>Baseline</strong>: rolling minimum (<code>--baseline movmin</code>) vs ALS (<code>--baseline als</code>) if slow drift
      is mistaken for bands.</li>
  <li><strong>Fewer spurious peaks</strong>: raise <code>--prominence-scale</code> above 1.0 (stricter than the default 0.012×range rule).</li>
  <li><strong>Avoid heavy SG + tiny peaks</strong>: very large windows can merge close doublets—inspect the overlay.</li>
</ul>
"""


def run_batch(
    *,
    input_paths: list[Path],
    out_dir: Path,
    page_title: str,
    subtitle: str,
    baseline: str,
    movmin_window: int,
    sg_window: int,
    sg_poly: int,
    normalize: bool,
    show_pre_sg_overlay: bool,
    prominence_scale: float,
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    css = """
body { font-family: Arial, sans-serif; margin: 24px auto; max-width: 1100px; color: #1f2937; }
h1, h2 { margin-bottom: 6px; }
.muted { color: #6b7280; margin-top: 0; }
.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin: 14px 0 24px; }
img { max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }
pre.json { background: #f9fafb; padding: 10px; border-radius: 6px; font-size: 11px; overflow-x: auto; }
"""
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        f"<title>{html.escape(page_title)}</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>{html.escape(page_title)}</h1>",
        f"<p class='muted'>{html.escape(subtitle)}</p>",
        _denoise_notes_html(),
        "<h2>Per-spectrum stacked plots</h2>",
    ]

    for p in input_paths:
        p = p.resolve()
        wn, raw, hint = read_spectrum(p)
        wn_p, proc, info = preprocess_spectrum(
            wn,
            raw,
            intensity_mode=hint,
            baseline=baseline,  # type: ignore[arg-type]
            movmin_window=int(movmin_window),
            sg_window=int(sg_window),
            sg_poly=int(sg_poly),
            normalize=normalize,
            stash_pre_smooth=bool(show_pre_sg_overlay),
        )
        pwn, ph = find_peaks_scaled(
            wn_p,
            proc,
            prominence_scale=float(prominence_scale),
        )
        png = _stacked_figure_png(
            wn_p,
            proc,
            pwn,
            ph,
            title=p.name,
            show_pre_sg_overlay=show_pre_sg_overlay,
            info=info,
        )
        stem = p.stem.replace(" ", "_")
        png_name = f"kronecker_{stem}.png"
        if png:
            (out_dir / png_name).write_bytes(png)
            img_html = f"<img src='{html.escape(png_name)}' alt='{html.escape(p.name)}'/>"
        else:
            img_html = "<p>Matplotlib unavailable; figure skipped.</p>"

        meta = {
            "file": str(p),
            "preprocess": {k: info[k] for k in sorted(info) if k not in ("baseline_corrected_pre_sg", "smoothed_pre_normalize")},
            "peak_count": len(pwn),
            "cli": {
                "baseline": baseline,
                "movmin_window": movmin_window,
                "sg_window": sg_window,
                "sg_poly": sg_poly,
                "normalize": normalize,
                "show_pre_sg_overlay": show_pre_sg_overlay,
                "prominence_scale": prominence_scale,
            },
        }
        parts.append("<div class='card'>")
        parts.append(f"<h3>{html.escape(p.name)}</h3>")
        parts.append(img_html)
        parts.append("<h4>Parameters</h4>")
        parts.append(f"<pre class='json'>{html.escape(json.dumps(meta, indent=2))}</pre>")
        parts.append("</div>")

    parts.append("</body></html>")
    report_path = out_dir / "REPORT.html"
    report_path.write_text("".join(parts), encoding="utf-8")
    return report_path


def cmd_batch(args: argparse.Namespace) -> int:
    paths = [_resolve_under_chunks(Path(x)) for x in args.inputs]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not found: {p}")
    rp = run_batch(
        input_paths=paths,
        out_dir=_resolve_under_chunks(Path(args.out_dir)),
        page_title=args.title,
        subtitle=args.subtitle or "Kronecker featurization | ftir_foundation preprocess",
        baseline=str(args.baseline),
        movmin_window=int(args.movmin_window),
        sg_window=int(args.sg_window),
        sg_poly=int(args.sg_poly),
        normalize=not bool(args.no_normalize),
        show_pre_sg_overlay=bool(args.show_pre_sg_overlay),
        prominence_scale=float(args.prominence_scale),
    )
    print(json.dumps({"report": str(rp)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stacked spectrum + Kronecker peak report (no SVM)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_b = sub.add_parser("batch", help="One HTML report for arbitrary spectrum paths")
    p_b.add_argument("--inputs", nargs="+", required=True)
    p_b.add_argument("--out-dir", required=True)
    p_b.add_argument("--title", default="Kronecker spectrum report")
    p_b.add_argument("--subtitle", default="")
    p_b.add_argument("--baseline", choices=("movmin", "als"), default="movmin")
    p_b.add_argument("--movmin-window", type=int, default=151)
    p_b.add_argument("--sg-window", type=int, default=11, help="Odd SG window (ftir_foundation coerces to odd)")
    p_b.add_argument("--sg-poly", type=int, default=2)
    p_b.add_argument("--no-normalize", action="store_true", help="Skip [0,1] scaling after SG")
    p_b.add_argument(
        "--show-pre-sg-overlay",
        action="store_true",
        help="Plot baseline-corrected absorbance before SG (faint) under the smoothed curve",
    )
    p_b.add_argument(
        "--prominence-scale",
        type=float,
        default=1.0,
        help="Multiplies default prominence/height thresholds (>1 = fewer peaks)",
    )
    p_b.set_defaults(func=cmd_batch)
    ns = ap.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
