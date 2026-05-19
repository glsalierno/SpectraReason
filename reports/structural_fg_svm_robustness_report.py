#!/usr/bin/env python3
# Legacy/specialized entry point. Current production report is reports/structural_fg_svm_kronecker_report.py with --report-style product_v1.
"""
Evidence-first robustness report: perturb spectra, measure rule and optional ML stability.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.spectrum import load_processed_spectrum
from ml.ftir_export import export_pipeline_batch_csv
from ml.ftir_pipeline import load_model_bundle, run_evidence_first_pipeline
from ml.ftir_rule_config import load_rules_config
from ml.ftir_robustness import evaluate_evidence_robustness_one_spectrum, evaluate_robustness_one_spectrum
from reports.structural_fg_svm_kronecker_report import _metadata_for_path, _resolve_ml_mode


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_ROOT / p).resolve()


def run_batch(
    *,
    input_paths: list[Path],
    out_dir: Path,
    ml_mode: str = "none",
    fusion_mode: str = "annotate",
    basic_model_path: Path | None = None,
    subtle_model_path: Path | None = None,
    legacy_model_path: Path | None = None,
    rules_config: dict[str, Any] | None = None,
    export_csv_extra: Path | None = None,
) -> dict[str, Path]:
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basic_art = load_model_bundle(basic_model_path)
    subtle_art = load_model_bundle(subtle_model_path)
    legacy_art = load_model_bundle(legacy_model_path)
    resolved = _resolve_ml_mode(
        ml_mode,
        has_legacy=legacy_art is not None,
        has_basic=basic_art is not None,
        has_subtle=subtle_art is not None,
        include_evidence=True,
    )

    ml_models: dict[str, Any] = {}
    if basic_art is not None:
        ml_models["basic"] = basic_art
    if subtle_art is not None:
        ml_models["subtle"] = subtle_art
    if legacy_art is not None:
        ml_models["legacy"] = legacy_art
    if resolved in ("basic", "both") and "basic" not in ml_models and legacy_art is not None:
        ml_models["basic"] = legacy_art

    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    pipeline_batch: list[dict[str, Any]] = []

    for p in input_paths:
        p = _resolve(p)
        ps = load_processed_spectrum(p)
        md = _metadata_for_path(p)

        ev_lf, ev_sum = evaluate_evidence_robustness_one_spectrum(ps.wn, ps.y)
        for row in ev_lf:
            row["spectrum"] = ps.name
        long_rows.extend(ev_lf)

        ml_summ = {}
        if resolved != "none" and ml_models:
            ml_lf, ml_summ = evaluate_robustness_one_spectrum(ps.wn, ps.y, md, ml_models)
            for row in ml_lf:
                row["spectrum"] = ps.name
                row["layer"] = row.get("layer", "ml")
            long_rows.extend(ml_lf)

        base_pipe = run_evidence_first_pipeline(
            ps.wn,
            ps.y,
            md=md,
            ml_mode=resolved,  # type: ignore[arg-type]
            fusion_mode=fusion_mode,  # type: ignore[arg-type]
            basic_model=basic_art,
            subtle_model=subtle_art,
            legacy_model=legacy_art,
            rules_config=rules_config,
            guardrails_mode="v2",
        )
        pipeline_batch.append({"spectrum": ps.name, "path": str(p), "pipeline": base_pipe})
        top_consensus = [
            t[0]
            for t in sorted(
                (base_pipe.get("consensus") or {}).get("per_label", {}).items(),
                key=lambda kv: -kv[1].get("final_score", 0),
            )[:1]
        ]

        summaries.append(
            {
                "spectrum": ps.name,
                "path": str(p),
                "ml_mode": resolved,
                "evidence_robustness": ev_sum.get("robustness_score"),
                "ml_robustness": ml_summ.get("overall_robustness_score") if ml_summ else None,
                "consensus_top1": top_consensus,
                "overall_robustness_score": float(
                    (ev_sum.get("robustness_score", 0) + (ml_summ.get("overall_robustness_score") or 0))
                    / (2 if ml_summ else 1)
                ),
            }
        )

    summary_csv = out_dir / "robustness_summary.csv"
    long_csv = out_dir / "robustness_longform.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spectrum",
                "path",
                "ml_mode",
                "evidence_robustness",
                "ml_robustness",
                "consensus_top1",
                "overall_robustness_score",
            ],
        )
        w.writeheader()
        for s in summaries:
            w.writerow({**s, "consensus_top1": json.dumps(s.get("consensus_top1"))})

    if long_rows:
        keys = sorted({k for r in long_rows for k in r})
        with long_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(long_rows)

    html_path = out_dir / "REPORT.html"
    cards = []
    for s in summaries:
        score = float(s.get("overall_robustness_score", 0))
        color = "#16a34a" if score >= 0.85 else ("#ca8a04" if score >= 0.65 else "#dc2626")
        cards.append(
            f"<section class='card'><h2>{html.escape(s['spectrum'])}</h2>"
            f"<p>ml_mode={html.escape(str(s.get('ml_mode')))}</p>"
            f"<span class='badge' style='background:{color}'>Overall {score:.2f}</span>"
            f"<ul><li>Evidence rule stability: {s.get('evidence_robustness', 0):.2f}</li>"
            f"<li>ML stability: {s.get('ml_robustness')}</li>"
            f"<li>Consensus top-1: {html.escape(str(s.get('consensus_top1')))}</li></ul></section>"
        )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>FTIR robustness (evidence-first)</title>"
        "<style>body{font-family:system-ui;margin:24px}.card{border:1px solid #ddd;"
        "border-radius:8px;padding:14px;margin:12px 0}.badge{color:#fff;padding:4px 8px;border-radius:4px}"
        "</style></head><body><h1>Robustness report</h1>"
        + "".join(cards)
        + "</body></html>",
        encoding="utf-8",
    )
    out_paths = {"summary_csv": summary_csv, "longform_csv": long_csv, "html": html_path}
    if export_csv_extra is not None:
        out_paths.update(export_pipeline_batch_csv(pipeline_batch, export_csv_extra, prefix="robust"))
    return out_paths


def cmd_batch(args: argparse.Namespace) -> int:
    ins = args.inputs or getattr(args, "input_alt", None)
    if not ins:
        raise SystemExit("Provide --inputs or --input")
    paths = [_resolve(Path(x)) for x in ins]
    out = _resolve(Path(args.out_dir))
    basic = Path(args.basic_model) if args.basic_model else None
    subtle = Path(args.subtle_model) if args.subtle_model else None
    legacy = Path(args.model) if args.model else None
    rules_config = None
    if (args.rules_preset or "").strip():
        rules_config = load_rules_config(preset=args.rules_preset.strip())
    export_extra = _resolve(Path(args.export_csv)) if (args.export_csv or "").strip() else None
    paths_out = run_batch(
        input_paths=paths,
        out_dir=out,
        ml_mode=args.ml_mode,
        fusion_mode=args.fusion_mode,
        basic_model_path=_resolve(basic) if basic else None,
        subtle_model_path=_resolve(subtle) if subtle else None,
        legacy_model_path=_resolve(legacy) if legacy else None,
        rules_config=rules_config,
        export_csv_extra=export_extra,
    )
    print(json.dumps({k: str(v) for k, v in paths_out.items()}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Evidence-first FTIR robustness report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("batch")
    p.add_argument("--inputs", nargs="+", default=None)
    p.add_argument("--input", nargs="+", dest="input_alt", default=None)
    p.add_argument("--out-dir", "--output", dest="out_dir", required=True)
    p.add_argument("--ml-mode", choices=("none", "basic", "subtle", "both", "legacy"), default="none")
    p.add_argument("--fusion-mode", choices=("annotate", "weighted", "gate", "ml_only"), default="annotate")
    p.add_argument("--basic-model", default="")
    p.add_argument("--subtle-model", default="")
    p.add_argument("--model", default="", help="Legacy model path")
    p.add_argument("--rules-preset", default="", help="Optional rule preset for evidence layer")
    p.add_argument("--export-csv", default="", help="Optional extra export directory")
    p.set_defaults(func=cmd_batch)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
