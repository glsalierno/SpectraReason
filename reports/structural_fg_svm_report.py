#!/usr/bin/env python3
# Legacy/specialized entry point. Current production report is reports/structural_fg_svm_kronecker_report.py with --report-style product_v1.
"""
HTML report: **structural** FG SVM (spectra + RDKit + Mordred) on one or more spectra.

Uses ``hidden_peak_workbench.load_processed_spectrum`` (same CSV/JDX preprocessing
as other NIST tooling) and ``ml.structural_fg_svm.predict_proba_row``.

Run from **FTIR_SVM** root::

    cd FTIR_SVM
    $env:PYTHONPATH = (Get-Location).Path  # FTIR_SVM
    python reports/structural_fg_svm_report.py batch \\
      --inputs NIST/raw/Dopamine.CSV NIST/raw/Dopamine_Polydopamine.CSV \\
      --model ml/runs/struct_fg_svm_rdkit_mordred.joblib \\
      --out-dir reports/structural_fg_dopamine
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.spectrum import load_processed_spectrum
from ml.structural_fg_svm import predict_proba_row


def _resolve_under_chunks(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _probs_table_html(probs: dict[str, float]) -> str:
    rows = sorted(probs.items(), key=lambda kv: -kv[1])
    parts = [
        "<table class='tbl'><tr><th>Functional group</th><th>P(label)</th></tr>",
    ]
    for lab, p in rows:
        parts.append(
            "<tr><td>{}</td><td>{:.4f}</td></tr>".format(
                html.escape(str(lab)),
                float(p),
            )
        )
    parts.append("</table>")
    return "".join(parts)


def _save_spectrum_png(name: str, wn: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
    ax.plot(np.asarray(wn, float), np.asarray(y, float), color="#1d4ed8", lw=0.9)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance (processed)")
    ax.set_title(name[:120], fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _metadata_for_path(path: Path) -> dict[str, Any]:
    stem = path.stem
    title = stem.replace("_", " ")
    md: dict[str, Any] = {"title": title, "name": stem, "xunits": "1/CM"}
    low = stem.lower()
    if "dopamine" in low and "poly" not in low:
        md["cas"] = "51-61-6"
        # Free base dopamine: no halogens — constrains halide FG score at inference
        # (see ``ml.atom_content_mask``).
        md["formula"] = "C8H11NO2"
    elif "polydopamine" in low or ("dopamine" in low and "poly" in low):
        # Idealized / typical literature PDA repeat unit has no halogens; suppress
        # spectral false positives for halide unless you add a formula with Cl/Br etc.
        md["no_halogens"] = True
    return md


def write_structural_report_html(
    *,
    out_dir: Path,
    page_title: str,
    subtitle: str,
    model_path: Path,
    sections: list[dict[str, Any]],
) -> Path:
    css = """
body { font-family: Arial, sans-serif; margin: 24px auto; max-width: 1100px; color: #1f2937; }
h1, h2 { margin-bottom: 6px; }
.muted { color: #6b7280; margin-top: 0; }
.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin: 14px 0 24px; }
img { max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }
.tbl { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }
.tbl th, .tbl td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }
.tbl th { background: #f3f4f6; }
pre.json { background: #f9fafb; padding: 10px; border-radius: 6px; font-size: 11px; overflow-x: auto; }
"""
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        f"<title>{html.escape(page_title)}</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>{html.escape(page_title)}</h1>",
        f"<p class='muted'>{html.escape(subtitle)}</p>",
        f"<p class='muted'>Model: <code>{html.escape(str(model_path))}</code></p>",
        "<h2>Spectra and structural SVM functional-group scores</h2>",
    ]
    for sec in sections:
        parts.extend(
            [
                "<div class='card'>",
                f"<h3>{html.escape(sec['name'])}</h3>",
                f"<p class='muted'>Metadata used: <code>{html.escape(sec['meta_line'])}</code></p>",
                f"<img src='{html.escape(sec['png'])}' alt='{html.escape(sec['name'])}'/>",
                "<h4>Functional-group probabilities (OvR)</h4>",
                sec["table_html"],
                "<h4>Raw JSON</h4>",
                f"<pre class='json'>{html.escape(sec['json_block'])}</pre>",
                "</div>",
            ]
        )
    parts.append("</body></html>")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.html"
    report_path.write_text("".join(parts), encoding="utf-8")
    return report_path


def run_batch(
    *,
    input_paths: list[Path],
    model_path: Path,
    out_dir: Path,
    page_title: str,
    subtitle: str,
) -> Path:
    import joblib

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    artifact = joblib.load(model_path)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[dict[str, Any]] = []
    for p in input_paths:
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        ps = load_processed_spectrum(p)
        md = _metadata_for_path(p)
        probs = predict_proba_row(artifact, wn=ps.wn, y=ps.y, md=md)
        stem = p.stem.replace(" ", "_")
        png_name = f"spec_structural_{stem}.png"
        _save_spectrum_png(ps.name, ps.wn, ps.y, out_dir / png_name)
        meta_line = json.dumps(
            {k: md.get(k) for k in ("title", "name", "cas", "xunits") if md.get(k)},
            sort_keys=True,
        )
        json_block = json.dumps({"probabilities": probs}, indent=2)
        sections.append(
            {
                "name": ps.name,
                "png": png_name,
                "meta_line": meta_line,
                "table_html": _probs_table_html(probs),
                "json_block": json_block,
            }
        )

    return write_structural_report_html(
        out_dir=out_dir,
        page_title=page_title,
        subtitle=subtitle,
        model_path=model_path,
        sections=sections,
    )


def cmd_batch(args: argparse.Namespace) -> int:
    model = _resolve_under_chunks(Path(args.model))
    out_dir = _resolve_under_chunks(Path(args.out_dir))
    paths = [_resolve_under_chunks(Path(x)) for x in args.inputs]
    rp = run_batch(
        input_paths=paths,
        model_path=model,
        out_dir=out_dir,
        page_title=args.title,
        subtitle=args.subtitle or f"{len(paths)} spectra | structural FG SVM",
    )
    print(json.dumps({"report": str(rp)}, indent=2))
    return 0


def cmd_dopamine_preset(args: argparse.Namespace) -> int:
    model = _resolve_under_chunks(Path(args.model))
    out_dir = _resolve_under_chunks(Path(args.out_dir))
    paths = [
        _ROOT / "NIST" / "raw" / "Dopamine.CSV",
        _ROOT / "NIST" / "raw" / "Dopamine_Polydopamine.CSV",
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("Missing CSV files:\n" + "\n".join(missing))
    rp = run_batch(
        input_paths=[Path(p) for p in paths],
        model_path=model,
        out_dir=out_dir,
        page_title="Dopamine vs polydopamine — structural FG SVM",
        subtitle="NIST/raw CSVs | same preprocessing as peak workbench | ml.structural_fg_svm",
    )
    print(json.dumps({"preset": "dopamine_csv", "report": str(rp)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="HTML report for structural FG SVM on spectra")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("batch", help="One HTML report for arbitrary spectrum paths")
    p_b.add_argument("--inputs", nargs="+", required=True)
    p_b.add_argument("--model", required=True, help="models/*.joblib (default: models/struct_fg_v7_pubchem_mordred.joblib)")
    p_b.add_argument("--out-dir", required=True)
    p_b.add_argument("--title", default="Structural FG SVM report")
    p_b.add_argument("--subtitle", default="")
    p_b.set_defaults(func=cmd_batch)

    p_d = sub.add_parser(
        "dopamine-csv",
        help="Preset: NIST/raw/Dopamine.CSV + Dopamine_Polydopamine.CSV",
    )
    p_d.add_argument("--model", required=True)
    p_d.add_argument(
        "--out-dir",
        default="reports/structural_fg_dopamine_csv",
        help="Output directory under chunks (default: reports/structural_fg_dopamine_csv)",
    )
    p_d.set_defaults(func=cmd_dopamine_preset)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
