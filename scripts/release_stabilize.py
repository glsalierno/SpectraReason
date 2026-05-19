#!/usr/bin/env python3
"""
SpectraReason production release stabilization helper.

Creates folders, reference snapshots, archive manifest + moves, vulture audit,
and release_stabilization_audit.md. Safe to re-run (skips existing snapshots unless --force).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE_MONTH = "2026-05"
ARCHIVE_DIR = ROOT / "reports" / "_archive" / ARCHIVE_MONTH

REFERENCE_SPECTRA: list[tuple[str, str]] = [
    ("catechol", str(ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx")),
    ("nylon_amide", str(ROOT / "examples" / "spectra" / "Nylon_T.CSV")),
    ("benzoic_acid", str(ROOT / "examples" / "spectra" / "Benzoic acid - 65-85-0-IR.jdx")),
    ("pyrrole_carboxylic", str(ROOT / "examples" / "spectra" / "1H-Pyrrole-2-carboxylic acid-634-97-9-IR.jdx")),
    ("indol_5_ol", str(ROOT / "examples" / "spectra" / "1H-Indol-5-ol-1953-54-4-IR.jdx")),
    ("indole", str(ROOT / "examples" / "spectra" / "Indole_120-72-9-IR.jdx")),
    ("pyrrole", str(ROOT / "examples" / "spectra" / "Pyrrole_109-97-7-IR.jdx")),
]

ARCHIVE_CANDIDATES: list[tuple[str, str]] = [
    ("reports/_prof_test", "smoke / profiling one-off"),
    ("reports/layout_polish_before_after", "layout A/B experiment"),
    ("reports/peak_label_threshold_noxide_test", "peak label threshold experiment"),
    ("reports/peak_sensitivity_validation", "peak sensitivity sweep"),
    ("reports/peak_threshold_005_test", "low threshold experiment"),
    ("reports/region_ruler_peak_label_test", "ruler + peak label test"),
    ("reports/upper_mid_ch_shading_test", "upper-mid shading experiment"),
    ("reports/ftir_powder_upper_mid_shading", "powder shading experiment"),
    ("reports/ftir_powder_v4_deconv", "deconv comparison (non-production model)"),
    ("reports/ftir_powder_v4_evidence_first", "superseded evidence-first report"),
    ("reports/ftir_powder_pda_eg_con_new", "duplicate pre-matlab powder report"),
    ("reports/product_v1_demo", "superseded by product_v1_front_demo"),
]

PROD_FAM = ROOT / "ml" / "runs" / "struct_fg_family_v4_ontology_latest.joblib"
PROD_SPEC = ROOT / "ml" / "runs" / "struct_fg_specific_v4_ontology_latest.joblib"


def ensure_dirs() -> None:
    for rel in (
        "configs/production",
        "configs/experiments",
        "docs",
        "reports/reference_snapshots/front",
        "reports/reference_snapshots/debug",
        "reports/reference_snapshots/static_figures",
        "reports/_archive",
        "ml/runs/production",
        "ml/runs/experiments",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
        gitkeep = ROOT / rel / ".gitkeep"
        if rel.startswith(("configs/", "ml/runs/")) and not any((ROOT / rel).iterdir()):
            gitkeep.touch(exist_ok=True)


def _run_report(
    *,
    inputs: list[Path],
    out: Path,
    audience: str,
    export_static: bool,
    static_out: Path | None = None,
) -> None:
    from reports.structural_fg_svm_kronecker_report import run_batch

    if not inputs:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    run_batch(
        input_paths=inputs,
        model_path=None,
        basic_model_path=PROD_FAM if PROD_FAM.is_file() else None,
        subtle_model_path=PROD_SPEC if PROD_SPEC.is_file() else None,
        out_path=out,
        page_title=f"Reference snapshots ({audience})",
        subtitle="SpectraReason production defaults",
        max_peaks=80,
        hover_top_fg=8,
        ml_mode="both" if PROD_FAM.is_file() else "none",
        fusion_mode="annotate",
        ml_guardrails="strict",
        guardrails_mode="v3",
        report_style="product_v1",
        report_audience=audience,
        report_density="balanced" if audience == "front" else "audit",
        visual_theme="matlab",
        show_region_ruler=True,
        peak_sensitivity="sensitive",
        show_weak_peaks=True,
        export_static_figures=export_static,
        static_out=static_out,
        rules_config={"ontology": "v4"},
        anonymize_metadata=True,
    )


def generate_snapshots(*, force: bool = False) -> list[Path]:
    existing_inputs = [Path(p) for _, p in REFERENCE_SPECTRA if Path(p).is_file()]
    missing = [n for n, p in REFERENCE_SPECTRA if not Path(p).is_file()]
    if missing:
        print("WARN missing spectra:", ", ".join(missing))

    outs: list[Path] = []
    front_out = ROOT / "reports" / "reference_snapshots" / "front" / "REPORT.html"
    debug_out = ROOT / "reports" / "reference_snapshots" / "debug" / "REPORT.html"
    static_out = ROOT / "reports" / "reference_snapshots" / "static_figures" / "REPORT.html"
    static_dir = ROOT / "reports" / "reference_snapshots" / "static_figures" / "presentation" / "figures"

    if force or not front_out.is_file():
        _run_report(inputs=existing_inputs, out=front_out, audience="front", export_static=False)
        outs.append(front_out)
    if force or not debug_out.is_file():
        _run_report(inputs=existing_inputs, out=debug_out, audience="debug", export_static=False)
        outs.append(debug_out)
    if force or not static_out.is_file():
        subset = existing_inputs[:3] if len(existing_inputs) > 3 else existing_inputs
        _run_report(
            inputs=subset,
            out=static_out,
            audience="front",
            export_static=True,
            static_out=static_dir,
        )
        outs.append(static_out)
    return outs


def archive_clutter(*, dry_run: bool = False) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "reports" / f"archive_manifest_{date.today():%Y%m%d}.csv"
    rows: list[dict[str, str]] = []
    for rel, reason in ARCHIVE_CANDIDATES:
        src = ROOT / rel
        if not src.exists():
            continue
        dest = ARCHIVE_DIR / src.name
        rows.append({"source": str(src), "destination": str(dest), "reason": reason})
        if dry_run:
            continue
        if dest.exists():
            print(f"SKIP archive (dest exists): {dest}")
            continue
        shutil.move(str(src), str(dest))
        print(f"ARCHIVED {src} -> {dest}")
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "destination", "reason"])
        w.writeheader()
        w.writerows(rows)
    return manifest


def run_vulture() -> tuple[Path, Path]:
    raw_path = ROOT / "reports" / "vulture_raw_output.txt"
    md_path = ROOT / "reports" / "vulture_dead_code_audit.md"
    cmd = [
        sys.executable,
        "-m",
        "vulture",
        str(ROOT),
        "--min-confidence",
        "80",
        "--exclude",
        ".venv,env,__pycache__,reports/_archive,ml/runs,data,*.joblib,*.npz",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=False)
    except FileNotFoundError:
        raw_path.write_text("vulture not installed; pip install -r requirements-dev.txt\n", encoding="utf-8")
        md_path.write_text(
            "# Vulture dead-code audit\n\nVulture was not installed. Install with:\n\n"
            "```powershell\npip install -r requirements-dev.txt\n"
            f"python -m vulture . --min-confidence 80\n```\n",
            encoding="utf-8",
        )
        return raw_path, md_path
    raw = (proc.stdout or "") + (proc.stderr or "")
    raw_path.write_text(raw or "(no findings at confidence >= 80)\n", encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    md = [
        "# Vulture dead-code audit",
        "",
        f"Command: `{' '.join(cmd)}`",
        f"Exit code: {proc.returncode}",
        f"Findings: {len(lines)}",
        "",
        "## Classification policy",
        "",
        "- **Keep (A):** CLI entry points, dynamic sklearn fields, report HTML builders — see `tools/vulture_whitelist.py`",
        "- **Deprecate (B):** old report helpers — list in `docs/DEPRECATED.md`",
        "- **Remove (C):** only duplicate private helpers confirmed by grep + tests",
        "",
        "## Raw findings",
        "",
        "```",
        raw.strip() or "(none)",
        "```",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return raw_path, md_path


def write_reference_readme() -> Path:
    readme = ROOT / "reports" / "reference_snapshots" / "README.md"
    lines = [
        "# Reference snapshots (production defaults)",
        "",
        "Bundled HTML examples for **SpectraReason** production settings. Open in a",
        "browser without a server.",
        "",
        "## Spectra",
        "",
        "| ID | Relative path | Role |",
        "|----|---------------|------|",
    ]
    for sid, p in REFERENCE_SPECTRA:
        rel = Path(p)
        try:
            rel = rel.relative_to(ROOT)
        except ValueError:
            pass
        lines.append(f"| {sid} | `{rel.as_posix()}` | canonical |")
    lines += [
        "",
        "## Regenerate",
        "",
        "```bash",
        "export PYTHONPATH=\"$(pwd)\"",
        "python scripts/release_stabilize.py --snapshots-only",
        "```",
        "",
        "## Outputs",
        "",
        "- Front: `reports/reference_snapshots/front/REPORT.html`",
        "- Debug: `reports/reference_snapshots/debug/REPORT.html`",
        "- Static figures: `reports/reference_snapshots/static_figures/`",
        "",
        "## Expected qualitative behavior",
        "",
        "- **Catechol / indole / pyrrole:** aromatic + O–H/N–H; heteroaromatic cautions; no supported nitro from mid-region alone",
        "- **Nylon:** amide I/II pattern; amide supported when paired bands present",
        "- **Pyrrole / heteroaromatic set:** N–H and fingerprint overlap cards in front mode",
        "- **Benzoic acid:** carboxylic C=O/O–H; not confused with nitro",
        "",
        "## Known ambiguities",
        "",
        "- 1450–1650 cm⁻¹: C=C vs amide II vs heterocyclic N–O (ruler + guardrails)",
        "- ATR fingerprint: C–O vs Si–O overlap without siloxane call",
        "- Pyrrole-carboxylic: amide/enamine/pyrrole overlap cards in front mode",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def write_release_audit(
    *,
    manifest: Path | None,
    vulture_md: Path,
    snapshots: list[Path],
) -> Path:
    from reports.reproducibility_meta import _sha256_file, git_commit_hash

    audit = ROOT / "reports" / "release_stabilization_audit.md"
    archived = []
    if manifest and manifest.is_file():
        with manifest.open(encoding="utf-8") as f:
            archived = list(csv.DictReader(f))
    body = [
        "# Release stabilization audit",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Git commit:** `{git_commit_hash(ROOT)}`",
        "",
        "## Production models",
        "",
        f"| Model | Path | SHA-256 (prefix) |",
        f"|-------|------|------------------|",
    ]
    for label, p in (("family", PROD_FAM), ("specific", PROD_SPEC)):
        rel = p
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            pass
        body.append(f"| {label} | `{rel.as_posix()}` | `{_sha256_file(p)}` |")
    body += [
        "",
        "## Documentation created/updated",
        "",
        "- `docs/PRODUCTION_DEFAULTS.md`",
        "- `docs/REPRODUCIBILITY.md`",
        "- `docs/COMMANDS.md` (production + archive commands)",
        "- `docs/CODEMAP.md`, `docs/DEPRECATED.md`",
        "- `CANONICAL_OUTPUTS.md`",
        "- `reports/reference_snapshots/README.md`",
        "",
        "## Reference snapshots",
        "",
    ]
    for p in snapshots:
        body.append(f"- `{p}`")
    body += [
        "",
        "## Archived folders",
        "",
        f"Manifest: `{manifest}`" if manifest else "(no manifest)",
        "",
    ]
    for row in archived:
        body.append(f"- `{row.get('source')}` → `{row.get('destination')}` ({row.get('reason')})")
    body += [
        "",
        "## Vulture",
        "",
        f"See `reports/vulture_dead_code_audit.md` and `reports/vulture_raw_output.txt`.",
        "",
        "## Files deleted in this pass",
        "",
        "None (safe cleanup only: reproducibility metadata wiring; no production code removed).",
        "",
        "## Tests",
        "",
        "```powershell",
        "python -m pytest ml/tests -q",
        "```",
        "",
        "## Remaining debt",
        "",
        "- Promote `configs/production/*.yaml` when CLI reads pinned presets",
        "- Review vulture B-class items before any deletion",
        "- Deconv model remains experimental under `ml/runs/experiments/`",
        "",
        "## Next safe cleanup",
        "",
        "- Unused imports flagged at 100% confidence only",
        "- Whitelist dynamic report hooks in vulture config",
        "",
    ]
    audit.write_text("\n".join(body), encoding="utf-8")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-only", action="store_true")
    ap.add_argument("--force-snapshots", action="store_true")
    ap.add_argument("--dry-run-archive", action="store_true")
    ap.add_argument("--skip-vulture", action="store_true")
    ap.add_argument("--skip-archive", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    write_reference_readme()

    snapshots: list[Path] = []
    manifest: Path | None = None
    if args.snapshots_only:
        snapshots = generate_snapshots(force=args.force_snapshots)
        print("Snapshots:", snapshots)
        return 0

    snapshots = generate_snapshots(force=args.force_snapshots)
    if not args.skip_archive:
        manifest = archive_clutter(dry_run=args.dry_run_archive)
    vulture_md = ROOT / "reports" / "vulture_dead_code_audit.md"
    if not args.skip_vulture:
        _, vulture_md = run_vulture()
    write_release_audit(manifest=manifest, vulture_md=vulture_md, snapshots=snapshots)
    print("Done. Audit:", ROOT / "reports" / "release_stabilization_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
