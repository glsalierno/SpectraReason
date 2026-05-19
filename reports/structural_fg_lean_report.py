#!/usr/bin/env python3
# Legacy/specialized entry point. Current production report is reports/structural_fg_svm_kronecker_report.py with --report-style product_v1.
"""
Lean HTML report: evidence-first functional groups + optional SVM refinement.

Compared to ``structural_fg_svm_kronecker_report.py``:
- No Plotly / Kronecker interactive panels (small static HTML)
- Matplotlib spectrum PNG per input
- Consensus + top rule assignments + cautions only

Default is evidence-only (``--ml-mode none``), matching pre-production guidance.

Run from **FTIR_SVM_v2** root::

    $env:PYTHONPATH = (Get-Location).Path
    python reports/structural_fg_lean_report.py batch \\
      --inputs examples/spectra/Catechol-120-80-9-IR.jdx \\
      --out-dir reports/lean_demo
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.peaks import find_peaks_simple
from lib.spectrum import load_processed_spectrum
from ml.ftir_pipeline import load_model_bundle, run_evidence_first_pipeline
from ml.ftir_report_sections import caution_block_html, rule_assignments_html
from ml.ftir_rule_config import load_rules_config
from reports.structural_fg_svm_kronecker_report import _metadata_for_path, _resolve_ml_mode


def _resolve_under_chunks(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _save_spectrum_png(name: str, wn: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    ax.plot(np.asarray(wn, float), np.asarray(y, float), color="#1d4ed8", lw=0.9)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance (processed)")
    ax.set_title(name[:100], fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _lean_consensus_table_html(
    consensus: dict[str, Any],
    *,
    top_n: int = 10,
    min_score: float = 0.12,
    show_ml: bool = False,
) -> str:
    items = consensus.get("top_labels") or []
    cols = "<tr><th>Functional group</th><th>Rule</th>"
    if show_ml:
        cols += "<th>ML P</th>"
    cols += "<th>Final</th><th>Status</th></tr>"
    parts = [
        "<h3>Consensus (evidence-first)</h3>",
        "<p class='hint'>Spectral rules are primary; ML is optional secondary signal. Not ground truth.</p>",
        f"<table class='tbl'>{cols}",
    ]
    shown = 0
    for lab, ent in items:
        if float(ent.get("final_score", 0)) < min_score:
            continue
        row = (
            f"<tr><td>{html.escape(str(lab))}</td>"
            f"<td>{float(ent.get('rule_score', 0)):.3f}</td>"
        )
        if show_ml:
            ml_vals = [
                ent.get("ml_probability_basic"),
                ent.get("ml_probability_subtle"),
                ent.get("ml_probability_legacy"),
            ]
            ml_max = max((float(v) for v in ml_vals if v is not None), default=0.0)
            row += f"<td>{ml_max:.3f}</td>" if ml_max > 0 else "<td>—</td>"
        row += (
            f"<td>{float(ent.get('final_score', 0)):.3f}</td>"
            f"<td>{html.escape(str(ent.get('agreement_status', '')))}</td></tr>"
        )
        parts.append(row)
        shown += 1
        if shown >= top_n:
            break
    parts.append("</table>")
    return "".join(parts)


_LEAN_CSS = """
body { font-family: Arial, sans-serif; margin: 24px auto; max-width: 960px; color: #1f2937; }
h1, h2, h3 { margin-bottom: 6px; }
.muted, .hint { color: #6b7280; font-size: 13px; }
.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin: 16px 0 22px; }
img { max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }
.tbl { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }
.tbl th, .tbl td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }
.tbl th { background: #f3f4f6; }
.caution { color: #92400e; font-size: 13px; }
ul.caution { margin: 6px 0 0 18px; }
"""


def write_lean_report_html(
    *,
    out_dir: Path,
    page_title: str,
    subtitle: str,
    meta_line: str,
    sections: list[dict[str, Any]],
) -> Path:
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        f"<title>{html.escape(page_title)}</title>",
        f"<style>{_LEAN_CSS}</style></head><body>",
        f"<h1>{html.escape(page_title)}</h1>",
        f"<p class='muted'>{html.escape(subtitle)}</p>",
        f"<p class='muted'>{html.escape(meta_line)}</p>",
    ]
    for sec in sections:
        parts.extend(
            [
                "<div class='card'>",
                f"<h2>{html.escape(sec['name'])}</h2>",
                f"<p class='muted'>Metadata: <code>{html.escape(sec['meta_line'])}</code></p>",
                f"<img src='{html.escape(sec['png'])}' alt='{html.escape(sec['name'])}'/>",
                sec["body_html"],
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
    out_dir: Path,
    page_title: str,
    subtitle: str,
    ml_mode: str = "none",
    fusion_mode: str = "annotate",
    basic_model_path: Path | None = None,
    subtle_model_path: Path | None = None,
    legacy_model_path: Path | None = None,
    rules_config: dict[str, Any] | None = None,
    top_n: int = 10,
    rule_top_n: int = 8,
    write_json: bool = False,
) -> Path:
    basic_art = load_model_bundle(basic_model_path) if basic_model_path else None
    subtle_art = load_model_bundle(subtle_model_path) if subtle_model_path else None
    legacy_art = load_model_bundle(legacy_model_path) if legacy_model_path else None

    resolved_mode = _resolve_ml_mode(
        ml_mode,
        has_legacy=legacy_art is not None,
        has_basic=basic_art is not None,
        has_subtle=subtle_art is not None,
        include_evidence=True,
    )
    show_ml = resolved_mode != "none"

    meta_bits = [f"ml_mode={resolved_mode}", f"fusion={fusion_mode}"]
    if basic_model_path:
        meta_bits.append(f"basic={basic_model_path.name}")
    if legacy_model_path:
        meta_bits.append(f"legacy={legacy_model_path.name}")
    if rules_config and rules_config.get("description"):
        meta_bits.append(str(rules_config["description"]))

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[dict[str, Any]] = []
    json_records: list[dict[str, Any]] = []

    for p in input_paths:
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        ps = load_processed_spectrum(p)
        md = _metadata_for_path(p)
        pwn, ph = find_peaks_simple(ps.wn, ps.y)
        peaks = [{"wn_cm1": a, "height": b} for a, b in zip(pwn, ph)]

        pipeline = run_evidence_first_pipeline(
            ps.wn,
            ps.y,
            md=md,
            peaks=peaks,
            ml_mode=resolved_mode,  # type: ignore[arg-type]
            fusion_mode=fusion_mode,  # type: ignore[arg-type]
            basic_model=basic_art,
            subtle_model=subtle_art,
            legacy_model=legacy_art,
            rules_config=rules_config,
            guardrails_mode="v2",
        )

        stem = p.stem.replace(" ", "_")
        png_name = f"spec_lean_{stem}.png"
        _save_spectrum_png(ps.name, ps.wn, ps.y, out_dir / png_name)

        body = (
            _lean_consensus_table_html(
                pipeline["consensus"],
                top_n=top_n,
                show_ml=show_ml,
            )
            + rule_assignments_html(pipeline["rule_assignments"], top_n=rule_top_n)
            + caution_block_html(pipeline)
        )
        if pipeline.get("warnings"):
            body += (
                "<p class='muted'>Notes: "
                + html.escape("; ".join(pipeline["warnings"]))
                + "</p>"
            )

        meta_line = json.dumps(
            {k: md.get(k) for k in ("title", "name", "cas", "formula", "xunits") if md.get(k)},
            sort_keys=True,
        )
        sections.append(
            {
                "name": ps.name,
                "png": png_name,
                "meta_line": meta_line,
                "body_html": body,
            }
        )
        if write_json:
            json_records.append(
                {
                    "spectrum": ps.name,
                    "path": str(p),
                    "consensus": pipeline["consensus"],
                    "rule_assignments": pipeline["rule_assignments"],
                    "ml_mode": resolved_mode,
                    "warnings": pipeline.get("warnings"),
                }
            )

    report_path = write_lean_report_html(
        out_dir=out_dir,
        page_title=page_title,
        subtitle=subtitle,
        meta_line=" | ".join(meta_bits),
        sections=sections,
    )
    if write_json:
        (out_dir / "lean_results.json").write_text(
            json.dumps(json_records, indent=2),
            encoding="utf-8",
        )
    return report_path


def _rules_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    preset = (getattr(args, "rules_preset", "") or "").strip()
    rc_path = (getattr(args, "rules_config", "") or "").strip()
    if preset or rc_path:
        return load_rules_config(preset=preset or None, config_path=rc_path or None)
    return None


def write_compare_index(
    out_dir: Path,
    *,
    evidence_href: str = "evidence_only/REPORT.html",
    svm_href: str = "svm_legacy/REPORT.html",
) -> Path:
    body = f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>Lean FTIR FG — compare</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px auto; max-width: 720px; }}
a {{ display: block; margin: 12px 0; font-size: 18px; }}
.muted {{ color: #6b7280; font-size: 14px; }}
</style></head><body>
<h1>Lean FTIR functional-group reports</h1>
<p class='muted'>Same spectrum(s), two modes — open both to compare.</p>
<a href="{html.escape(evidence_href)}">Evidence only</a>
<span class='muted'>Rules + band library; no SVM column.</span>
<a href="{html.escape(svm_href)}">Evidence + legacy SVM (annotate)</a>
<span class='muted'>Adds ML P column; final scores stay rule-led unless fusion changes them.</span>
</body></html>"""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    index_path.write_text(body, encoding="utf-8")
    return index_path


def cmd_batch(args: argparse.Namespace) -> int:
    paths = [_resolve_under_chunks(Path(x)) for x in args.inputs]
    out_dir = _resolve_under_chunks(Path(args.out_dir))

    basic_p = Path(args.basic_model) if getattr(args, "basic_model", "") else None
    subtle_p = Path(args.subtle_model) if getattr(args, "subtle_model", "") else None
    legacy_p = Path(args.model) if getattr(args, "model", "") else None
    ml_mode = getattr(args, "ml_mode", "none")
    if ml_mode in ("legacy", "basic") and legacy_p is None and basic_p is None:
        legacy_p = Path("models/struct_fg_v7_pubchem_mordred.joblib")

    rules_config = _rules_config_from_args(args)
    preset = (getattr(args, "rules_preset", "") or "").strip()

    rp = run_batch(
        input_paths=paths,
        out_dir=out_dir,
        page_title=args.title,
        subtitle=args.subtitle or f"{len(paths)} spectra | lean evidence-first FTIR FG report",
        ml_mode=ml_mode,
        fusion_mode=getattr(args, "fusion_mode", "annotate"),
        basic_model_path=_resolve_under_chunks(basic_p) if basic_p else None,
        subtle_model_path=_resolve_under_chunks(subtle_p) if subtle_p else None,
        legacy_model_path=_resolve_under_chunks(legacy_p) if legacy_p else None,
        rules_config=rules_config,
        top_n=int(args.top_n),
        rule_top_n=int(args.rule_top_n),
        write_json=bool(args.write_json),
    )
    out: dict[str, Any] = {"report": str(rp)}
    if args.write_json:
        out["json"] = str(out_dir / "lean_results.json")
    if preset:
        out["rules_preset"] = preset
    print(json.dumps(out, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    paths = [_resolve_under_chunks(Path(x)) for x in args.inputs]
    out_dir = _resolve_under_chunks(Path(args.out_dir))
    model_path = _resolve_under_chunks(
        Path(args.model or "models/struct_fg_v7_pubchem_mordred.joblib")
    )
    rules_config = _rules_config_from_args(args)
    preset = (getattr(args, "rules_preset", "") or "").strip()
    write_json = bool(args.write_json)
    fusion = getattr(args, "fusion_mode", "annotate")
    n = len(paths)

    ev_dir = out_dir / "evidence_only"
    svm_dir = out_dir / "svm_legacy"
    ev_report = run_batch(
        input_paths=paths,
        out_dir=ev_dir,
        page_title="Lean FTIR FG — evidence only",
        subtitle=f"{n} spectra | rules only (no ML)",
        ml_mode="none",
        rules_config=rules_config,
        top_n=int(args.top_n),
        rule_top_n=int(args.rule_top_n),
        write_json=write_json,
    )
    svm_report = run_batch(
        input_paths=paths,
        out_dir=svm_dir,
        page_title="Lean FTIR FG — evidence + legacy SVM",
        subtitle=f"{n} spectra | annotate fusion | {model_path.name}",
        ml_mode="legacy",
        fusion_mode=fusion,
        legacy_model_path=model_path,
        rules_config=rules_config,
        top_n=int(args.top_n),
        rule_top_n=int(args.rule_top_n),
        write_json=write_json,
    )
    index_path = write_compare_index(out_dir)
    result: dict[str, Any] = {
        "index": str(index_path),
        "evidence_only": str(ev_report),
        "svm_legacy": str(svm_report),
        "model": str(model_path),
    }
    if preset:
        result["rules_preset"] = preset
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Lean evidence-first FTIR functional-group HTML report")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("batch", help="Build one lean HTML report")
    p_b.add_argument("--inputs", nargs="+", required=True)
    p_b.add_argument("--out-dir", required=True)
    p_b.add_argument("--model", default="", help="Legacy/basic SVM .joblib (optional)")
    p_b.add_argument("--basic-model", default="")
    p_b.add_argument("--subtle-model", default="")
    p_b.add_argument(
        "--ml-mode",
        choices=("none", "basic", "subtle", "both", "legacy", "auto"),
        default="none",
        help="Default none = evidence only",
    )
    p_b.add_argument(
        "--fusion-mode",
        choices=("annotate", "weighted", "gate", "ml_only"),
        default="annotate",
    )
    p_b.add_argument(
        "--rules-preset",
        default="conservative",
        help="Rule preset (default: conservative). Use '' for library defaults.",
    )
    p_b.add_argument("--rules-config", default="", help="Optional JSON overrides")
    p_b.add_argument("--top-n", type=int, default=10, help="Max consensus rows")
    p_b.add_argument("--rule-top-n", type=int, default=8, help="Max rule-assignment rows")
    p_b.add_argument("--write-json", action="store_true", help="Write lean_results.json")
    p_b.add_argument("--title", default="FTIR functional groups — lean report")
    p_b.add_argument("--subtitle", default="")
    p_b.set_defaults(func=cmd_batch)

    p_c = sub.add_parser(
        "compare",
        help="Build evidence-only + SVM lean reports side by side",
    )
    p_c.add_argument("--inputs", nargs="+", required=True)
    p_c.add_argument("--out-dir", required=True, help="Parent dir (creates evidence_only/ and svm_legacy/)")
    p_c.add_argument(
        "--model",
        default="models/struct_fg_v7_pubchem_mordred.joblib",
        help="Legacy SVM for the refinement report",
    )
    p_c.add_argument(
        "--fusion-mode",
        choices=("annotate", "weighted", "gate", "ml_only"),
        default="annotate",
    )
    p_c.add_argument("--rules-preset", default="conservative")
    p_c.add_argument("--rules-config", default="")
    p_c.add_argument("--top-n", type=int, default=10)
    p_c.add_argument("--rule-top-n", type=int, default=8)
    p_c.add_argument("--write-json", action="store_true")
    p_c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
