"""
Optional CSV export for evidence-first pipeline results (publication supplements).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_pipeline_batch_csv(
    batch: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """
    ``batch`` items: ``{spectrum, path, pipeline}`` from report or calibrate CLI.

    Writes:
    - consensus_longform.csv
    - rules_longform.csv
    - consensus_wide.csv (one row per spectrum, top-N columns)
    - band_matches_longform.csv (optional, can be large)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = f"{prefix}_" if prefix else ""

    consensus_rows: list[dict[str, Any]] = []
    rules_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for item in batch:
        spec = item.get("spectrum", "")
        path = item.get("path", "")
        pipe = item.get("pipeline") or {}
        for lab, ent in (pipe.get("consensus") or {}).get("per_label", {}).items():
            consensus_rows.append(
                {
                    "spectrum": spec,
                    "path": path,
                    "label": lab,
                    "rule_score": ent.get("rule_score"),
                    "ml_probability_basic": ent.get("ml_probability_basic"),
                    "ml_probability_subtle": ent.get("ml_probability_subtle"),
                    "ml_probability_legacy": ent.get("ml_probability_legacy"),
                    "final_score": ent.get("final_score"),
                    "agreement_status": ent.get("agreement_status"),
                    "confidence_class": ent.get("confidence_class"),
                    "evidence_completeness": ent.get("evidence_completeness"),
                    "assignment_type": ent.get("assignment_type"),
                    "ml_mode": pipe.get("ml_mode"),
                    "fusion_mode": pipe.get("fusion_mode"),
                    "guardrails_mode": pipe.get("guardrails_mode"),
                    "ml_guardrails": pipe.get("ml_guardrails"),
                }
            )
        for lab, ent in (pipe.get("rule_assignments") or {}).get("assignments", {}).items():
            rules_rows.append(
                {
                    "spectrum": spec,
                    "path": path,
                    "label": lab,
                    "rule_score": ent.get("score"),
                    "confidence": ent.get("confidence"),
                    "confidence_class": ent.get("confidence_class"),
                    "evidence_completeness": ent.get("evidence_completeness"),
                    "assignment_type": ent.get("assignment_type"),
                    "summary": ent.get("human_readable_summary"),
                    "caution_flags": json.dumps(ent.get("caution_flags") or []),
                    "missing_bands": json.dumps(ent.get("missing_expected_bands") or []),
                }
            )
        for m in (pipe.get("evidence") or {}).get("band_matches") or []:
            if not m.get("matched"):
                continue
            band_rows.append(
                {
                    "spectrum": spec,
                    "path": path,
                    "band_id": m.get("band_id"),
                    "label": m.get("label"),
                    "region_min_cm1": m.get("region_min_cm1"),
                    "region_max_cm1": m.get("region_max_cm1"),
                    "support_score": m.get("support_score"),
                }
            )
        top = (pipe.get("consensus") or {}).get("top_labels") or []
        wide: dict[str, Any] = {"spectrum": spec, "path": path}
        for i, (lab, ent) in enumerate(top[:8]):
            wide[f"top{i+1}_label"] = lab
            wide[f"top{i+1}_final_score"] = ent.get("final_score")
            wide[f"top{i+1}_agreement"] = ent.get("agreement_status")
        wide_rows.append(wide)

    paths: dict[str, Path] = {}
    paths["consensus_longform"] = _write_csv(out_dir / f"{pre}consensus_longform.csv", consensus_rows)
    paths["rules_longform"] = _write_csv(out_dir / f"{pre}rules_longform.csv", rules_rows)
    paths["consensus_wide"] = _write_csv(out_dir / f"{pre}consensus_wide.csv", wide_rows)
    if band_rows:
        paths["band_matches_longform"] = _write_csv(out_dir / f"{pre}band_matches_longform.csv", band_rows)
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path
