#!/usr/bin/env python3
"""
Archive or remove temporal report outputs, old example runs, and caches.

Preserves (see CANONICAL_OUTPUTS.md / docs/COMMANDS.md):
  reports/reference_snapshots/
  reports/product_v1_front_demo/
  reports/product_v1_debug_demo/
  reports/ftir_powder_pda_eg_con_new_matlab/
  reports/examples_matlab_pyrrole_indol/
  reports/confounder_coverage_summary.*
  reports/external_dataset_expansion_audit.md
  reports/dataset_ingestion_audit.md
  reports/*.py (report modules)
  reports/*CONTRACT*.md, FIGURES_AND_EXPORT.md, CURATION_AND_CHUNKS.md, report_regression_checklist.md
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_MONTH = "2026-05"
ARCHIVE_DIR = ROOT / "reports" / "_archive" / ARCHIVE_MONTH
AUDIT_ARCHIVE = ARCHIVE_DIR / "development_audits"

# Entire directories to move under reports/_archive/2026-05/
ARCHIVE_DIRS: list[tuple[str, str]] = [
    ("reports/examples_pyrrole_2_carboxylic_acid_front", "one-off pyrrole front report"),
    ("reports/ftir_powder_pda_eg_con_new_front", "superseded by ftir_powder_pda_eg_con_new_matlab"),
    ("reports/dopamine_polydopamine_powder_060526", "superseded article one-off export"),
    ("reports/dopamine_polydopamine_powder_060526_svm", "superseded article one-off export"),
    ("reports/nylon_pda_oda_t_paper", "superseded paper-only export"),
    ("reports/pda_eg_con_new_minus_air_scaled", "superseded POC one-off report"),
    ("examples/_evidence_pipeline_report", "legacy examples HTML/CSV bundle"),
    ("examples/_evidence_pipeline_report_deconv", "legacy deconv examples bundle"),
]

# Glob patterns under reports/
ARCHIVE_GLOBS: list[tuple[str, str]] = [
    ("reports/model_training_*", "ephemeral training run HTML snapshots"),
]

# Loose files in reports/ → development_audits/
ARCHIVE_FILES: list[tuple[str, str]] = [
    ("reports/codebase_tidy_audit.md", "superseded tidy audit"),
    ("reports/feature_audit_v4.md", "v4 feature audit"),
    ("reports/peak_sensitivity_audit.md", "peak sensitivity experiment"),
    ("reports/product_v1_design_audit.md", "design notes"),
    ("reports/report_stabilization_audit.md", "old stabilization notes"),
    ("reports/upper_mid_ch_shading_audit.md", "shading experiment"),
    ("reports/v4_benchmark_operations.md", "benchmark ops notes"),
    ("reports/v4_classification_improvement_audit.md", "classification experiment"),
    ("reports/v4_deconv_feature_benchmark.md", "deconv benchmark"),
    ("reports/v4_deconv_training_benchmark.md", "deconv training benchmark"),
    ("reports/v4_evidence_v2_retraining_audit.md", "retraining audit"),
    ("reports/v4_retraining_variety_audit.md", "retraining variety audit"),
    ("reports/v4_smarts_label_audit.md", "SMARTS label audit"),
    ("reports/vulture_dead_code_audit.md", "regenerable vulture audit"),
    ("reports/vulture_raw_output.txt", "regenerable vulture raw"),
    ("reports/examples_svm_predictions.json", "old examples prediction dump"),
    ("reports/release_stabilization_audit.md", "superseded by cleanup manifest"),
]

# Delete outright (safe to regenerate)
DELETE_PATHS: list[str] = [
    ".pytest_cache",
    "reports/_smoke",
    "reports/_paper_export_smoke",
    "ml/runs/_job_monitor_status.json",
]


def _move(src: Path, dest: Path, *, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dry_run:
        print(f"  would move: {src} -> {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  SKIP (exists): {dest}")
        return False
    shutil.move(str(src), str(dest))
    print(f"  archived: {src.name} -> {dest.parent.name}/")
    return True


def _delete(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        print(f"  would delete: {path}")
        return True
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink()
    except OSError as exc:
        print(f"  WARN could not delete {path.relative_to(ROOT)}: {exc}")
        return False
    print(f"  deleted: {path.relative_to(ROOT)}")
    return True


def cleanup(*, dry_run: bool = False) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_ARCHIVE.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "reports" / f"cleanup_manifest_{date.today():%Y%m%d}.csv"
    rows: list[dict[str, str]] = []

    for pattern, reason in ARCHIVE_GLOBS:
        for src in sorted(ROOT.glob(pattern)):
            if not src.is_dir():
                continue
            dest = ARCHIVE_DIR / src.name
            if _move(src, dest, dry_run=dry_run):
                rows.append({"action": "archive", "source": str(src), "destination": str(dest), "reason": reason})

    for rel, reason in ARCHIVE_DIRS:
        src = ROOT / rel
        dest = ARCHIVE_DIR / src.name
        if _move(src, dest, dry_run=dry_run):
            rows.append({"action": "archive", "source": str(src), "destination": str(dest), "reason": reason})

    for rel, reason in ARCHIVE_FILES:
        src = ROOT / rel
        dest = AUDIT_ARCHIVE / src.name
        if _move(src, dest, dry_run=dry_run):
            rows.append({"action": "archive", "source": str(src), "destination": str(dest), "reason": reason})

    # Old archive manifests (keep latest cleanup only in reports root)
    for src in sorted(ROOT.glob("reports/archive_manifest_*.csv")):
        dest = AUDIT_ARCHIVE / src.name
        if _move(src, dest, dry_run=dry_run):
            rows.append({"action": "archive", "source": str(src), "destination": str(dest), "reason": "old archive manifest"})

    for rel in DELETE_PATHS:
        if _delete(ROOT / rel, dry_run=dry_run):
            rows.append({"action": "delete", "source": str(ROOT / rel), "destination": "", "reason": "ephemeral cache"})

    # __pycache__ under project (not .venv)
    for cache in ROOT.rglob("__pycache__"):
        if ".venv" in cache.parts or "venv" in cache.parts:
            continue
        if _delete(cache, dry_run=dry_run):
            rows.append({"action": "delete", "source": str(cache), "destination": "", "reason": "pycache"})

    if not dry_run:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["action", "source", "destination", "reason"])
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'Dry run: ' if dry_run else ''}{len(rows)} items processed. Manifest: {manifest.name}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive temporal reports and delete caches")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cleanup(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
