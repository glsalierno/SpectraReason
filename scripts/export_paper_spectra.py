#!/usr/bin/env python3
"""Export publication-style transmittance and normalized absorbance spectrum figures.

Delegates to ``reports.paper_ftir_figures`` (horizontal leader-line peak labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reports.paper_ftir_figures import (
    PaperFigureConfig,
    export_paper_figures_for_spectrum,
    write_paper_figures_index,
)


def export_paper_figure(path: Path, out_dir: Path) -> dict:
    cfg = PaperFigureConfig(formats=("png",), max_peak_labels=10)
    return export_paper_figures_for_spectrum(path, out_dir, config=cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export paper-ready FTIR spectrum figures.")
    ap.add_argument("--out", required=True, help="Output directory for figure files")
    ap.add_argument("--max-labels", type=int, default=10)
    ap.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png", "svg", "pdf"],
    )
    ap.add_argument("inputs", nargs="+", help="Spectrum CSV/JDX paths")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PaperFigureConfig(
        formats=tuple(args.formats),
        max_peak_labels=int(args.max_labels),
    )
    manifest = [
        export_paper_figures_for_spectrum(Path(p), out_dir, config=cfg) for p in args.inputs
    ]
    write_paper_figures_index(out_dir, manifest)
    print(json.dumps({"out_dir": str(out_dir.resolve()), "figures": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())