#!/usr/bin/env python3
"""Generate before/after static exports for layout polish review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_INPUT = _REPO / "examples" / "spectra" / "1H-Indol-5-ol-1953-54-4-IR.jdx"
OUT_ROOT = _REPO / "reports" / "layout_polish_before_after"


def _run_one(
    label: str,
    input_path: Path,
    out_dir: Path,
    *,
    peak_label_layout: str,
    auto_layout: bool,
    fingerprint_cluster_distance: float,
    presentation_mode: bool,
) -> None:
    from reports.structural_fg_svm_kronecker_report import run_batch

    out_dir.mkdir(parents=True, exist_ok=True)
    run_batch(
        input_paths=[input_path],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out_dir / "REPORT.html",
        page_title=f"Layout polish — {label}",
        subtitle=label,
        max_peaks=80,
        hover_top_fg=5,
        include_evidence=True,
        include_ml=False,
        report_style="product_v1",
        report_audience="front",
        visual_theme="matlab",
        show_region_ruler=True,
        export_static_figures=True,
        static_format="png",
        static_dpi=200,
        static_out=out_dir / "presentation" / "figures",
        peak_label_layout=peak_label_layout,
        auto_layout=auto_layout,
        fingerprint_cluster_distance=fingerprint_cluster_distance,
        presentation_mode=presentation_mode,
        peak_sensitivity="sensitive",
        peak_label_preset="sensitive",
        max_peak_labels=24,
        front_max_peak_labels=24,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    before = args.out.resolve() / "before_simple"
    after = args.out.resolve() / "after_smart"
    pres = args.out.resolve() / "after_presentation"

    _run_one(
        "before (simple layout, no auto-layout)",
        inp,
        before,
        peak_label_layout="simple",
        auto_layout=False,
        fingerprint_cluster_distance=0.0,
        presentation_mode=False,
    )
    _run_one(
        "after (smart layout + auto-layout + clustering)",
        inp,
        after,
        peak_label_layout="smart",
        auto_layout=True,
        fingerprint_cluster_distance=18.0,
        presentation_mode=False,
    )
    _run_one(
        "presentation mode",
        inp,
        pres,
        peak_label_layout="smart",
        auto_layout=True,
        fingerprint_cluster_distance=20.0,
        presentation_mode=True,
    )

    print(f"Wrote comparison under: {args.out.resolve()}")
    for sub in (before, after, pres):
        fig_dir = sub / "presentation" / "figures"
        if fig_dir.is_dir():
            for p in sorted(fig_dir.glob("*.png")):
                print(f"  {p.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
