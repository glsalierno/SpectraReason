#!/usr/bin/env python3
"""
Optional batch scan: rule scores across spectra + suggested preset hints.

Does not change defaults; use output to pick ``--rules-preset`` or edit JSON overrides.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.spectrum import load_processed_spectrum
from ml.ftir_export import export_pipeline_batch_csv
from ml.ftir_pipeline import run_evidence_first_pipeline
from ml.ftir_rule_config import list_presets, load_rules_config
from reports.structural_fg_svm_kronecker_report import _metadata_for_path


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_ROOT / p).resolve()


def run_calibration_batch(
    input_paths: list[Path],
    *,
    rules_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from ml.ftir_rule_config import evidence_config_from_rules, rules_assign_config_from_rules

    ev_cfg = evidence_config_from_rules(rules_config)
    ru_cfg = rules_assign_config_from_rules(rules_config)
    summary_rows: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []

    for p in input_paths:
        p = _resolve(p)
        ps = load_processed_spectrum(p)
        md = _metadata_for_path(p)
        pipe = run_evidence_first_pipeline(
            ps.wn, ps.y, md=md, ml_mode="none",
            evidence_config=ev_cfg,
            rules_config=ru_cfg,
            guardrails_mode="v2",
        )
        assigns = (pipe.get("rule_assignments") or {}).get("assignments") or {}
        phenol_s = float(assigns.get("phenol", {}).get("score", 0))
        alcohol_s = float(assigns.get("alcohol", {}).get("score", 0))
        top = (pipe.get("consensus") or {}).get("top_labels") or []
        top3 = [t[0] for t in top[:3]]
        summary_rows.append(
            {
                "spectrum": ps.name,
                "path": str(p),
                "top1_rule": top3[0] if top3 else "",
                "top2_rule": top3[1] if len(top3) > 1 else "",
                "top3_rule": top3[2] if len(top3) > 2 else "",
                "phenol_score": phenol_s,
                "alcohol_score": alcohol_s,
                "phenol_confidence": assigns.get("phenol", {}).get("confidence"),
                "alcohol_confidence": assigns.get("alcohol", {}).get("confidence"),
            }
        )
        batch.append({"spectrum": ps.name, "path": str(p), "pipeline": pipe})
    return summary_rows, batch


def cmd_batch(args: argparse.Namespace) -> int:
    paths = [_resolve(Path(x)) for x in args.inputs]
    rules_config = None
    if getattr(args, "rules_preset", None):
        rules_config = load_rules_config(preset=args.rules_preset)
    if getattr(args, "rules_config", None):
        rules_config = load_rules_config(
            preset=args.rules_preset if getattr(args, "rules_preset", None) else None,
            config_path=args.rules_config,
        )
    summary_rows, batch = run_calibration_batch(paths, rules_config=rules_config)
    out_dir = _resolve(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration_summary.csv"
    with cal_path.open("w", newline="", encoding="utf-8") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    exported: dict[str, str] = {"calibration_summary": str(cal_path)}
    if args.export_csv:
        exp = export_pipeline_batch_csv(batch, out_dir / "csv", prefix="cal")
        exported.update({k: str(v) for k, v in exp.items()})
    if args.suggest_preset:
        phenol_hits = sum(1 for r in summary_rows if float(r.get("phenol_score", 0)) >= 0.35)
        alc_hits = sum(1 for r in summary_rows if float(r.get("alcohol_score", 0)) >= 0.35)
        both = sum(
            1 for r in summary_rows
            if float(r.get("phenol_score", 0)) >= 0.3 and float(r.get("alcohol_score", 0)) >= 0.3
        )
        suggestion = "conservative" if both > len(summary_rows) * 0.3 else "default"
        if both > 0:
            suggestion = "phenol_alcohol_strict"
        sug_path = out_dir / "suggested_preset.txt"
        sug_path.write_text(
            f"suggested_preset={suggestion}\n"
            f"phenol_hits={phenol_hits} alcohol_hits={alc_hits} both_ambiguous={both}\n"
            f"Use: --rules-preset {suggestion}\n",
            encoding="utf-8",
        )
        exported["suggested_preset"] = str(sug_path)
    print(json.dumps(exported, indent=2))
    return 0


def cmd_list_presets(_args: argparse.Namespace) -> int:
    print(json.dumps({"presets": list_presets()}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Optional rule calibration / batch scan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("batch", help="Scan spectra with rule-only pipeline")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rules-preset", default="", help=f"One of: {', '.join(list_presets())}")
    p.add_argument("--rules-config", default="", help="Extra JSON overrides")
    p.add_argument("--export-csv", action="store_true", help="Also write consensus/rules CSV")
    p.add_argument("--suggest-preset", action="store_true", help="Write suggested_preset.txt")
    p.set_defaults(func=cmd_batch)
    p2 = sub.add_parser("list-presets")
    p2.set_defaults(func=cmd_list_presets)
    args = ap.parse_args()
    if getattr(args, "rules_preset", None) == "":
        args.rules_preset = None
    if getattr(args, "rules_config", None) == "":
        args.rules_config = None
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
