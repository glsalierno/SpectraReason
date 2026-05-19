#!/usr/bin/env python3
"""
Build (if needed), train, and benchmark v4 specific SVMs: baseline / deconv / peakcodebook / combined.
Does not promote *_latest without explicit --promote-winner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIST = ROOT.parent / "NIST" / "reference_libraries" / "nistchemdata_ir_index_v7_fresh.sqlite"
PUBCHEM = ROOT / "ml" / "runs" / "pubchem_train_writable.json"
EXP = ROOT / "ml" / "runs" / "experiments"
REPORT = ROOT / "reports" / "v4_deconv_training_benchmark.md"
BASELINE_MODEL = ROOT / "ml" / "runs" / "struct_fg_specific_v4_ontology_latest.joblib"

PC_DS = EXP / "v4_classification_improvement" / "ds_peakcodebook_specific"
DECONV_DS = EXP / "v4_deconv_specific" / "ds_deconv"
PC_TRAIN = EXP / "v4_peakcodebook_specific"
DECONV_TRAIN = EXP / "v4_deconv_specific"
COMB_DS = EXP / "v4_peakcodebook_deconv_specific" / "ds_combined"
COMB_TRAIN = EXP / "v4_peakcodebook_deconv_specific"

HIGH_RISK = [
    "siloxane",
    "silicone_or_silane",
    "nitro",
    "nitrile",
    "alkyne",
    "phenol",
    "amide",
    "ester",
    "aryl_ether",
    "heteroaromatic",
    "pyrrole_like_NH",
    "cyclic_amine",
    "carboxylic_acid",
    "carbonate",
    "urethane",
]


def run(cmd: list[str]) -> float:
    print("\n>>>", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return time.perf_counter() - t0


def collect_train_metrics(out_dir: Path) -> dict:
    metas = sorted(out_dir.glob("*_training_metadata.json"))
    if not metas:
        return {}
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    ms = meta.get("metrics_summary") or {}
    hn = meta.get("hard_negative_summary") or {}
    joblibs = sorted(out_dir.glob("struct_fg_*.joblib"))
    return {
        "macro_f1": ms.get("macro_f1"),
        "micro_f1": None,
        "n_features": ms.get("n_features"),
        "n_train": meta.get("split", {}).get("n_train"),
        "n_test": meta.get("split", {}).get("n_test"),
        "hard_negative_total_fp": hn.get("total_fp"),
        "feature_set": meta.get("feature_set"),
        "deconv_mode": meta.get("deconv_mode"),
        "deconv_failure_count": meta.get("deconv_failure_count"),
        "deconv_runtime_mean_ms": meta.get("deconv_runtime_mean_ms"),
        "model_path": str(joblibs[-1].resolve()) if joblibs else "",
        "train_meta": str(metas[-1].resolve()),
    }


def collect_baseline() -> dict:
    import joblib

    if not BASELINE_MODEL.is_file():
        return {"experiment": "A_baseline", "feature_set": "spectral+evidence_v2", "macro_f1": None}
    art = joblib.load(BASELINE_MODEL)
    meta_path = sorted(BASELINE_MODEL.parent.glob("struct_fg_specific*v4*training_metadata.json"))
    m = {}
    if meta_path:
        m = json.loads(meta_path[-1].read_text(encoding="utf-8"))
    else:
        m = {"metrics_summary": {"macro_f1": 0.54, "n_features": 434}, "feature_set": "spectral+evidence_v2"}
    ev = art.get("meta") or {}
    ms = m.get("metrics_summary") or {}
    return {
        "experiment": "A_baseline_production",
        "feature_set": ev.get("feature_set") or "spectral+evidence_v2",
        "macro_f1": ms.get("macro_f1") or art.get("macro_f1"),
        "n_features": len(art.get("feature_names") or []) or ms.get("n_features"),
        "model_path": str(BASELINE_MODEL.resolve()),
        "hard_negative_total_fp": None,
    }


def per_label_csv(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    import csv

    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = str(row.get("label", "")).lower()
            if lab:
                out[lab] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--promote-winner", action="store_true")
    args = ap.parse_args()

    if not NIST.is_file():
        print(f"Missing NIST index: {NIST}", file=sys.stderr)
        return 1

    py = sys.executable
    build_common = [
        py,
        "-m",
        "ml.structural_fg_svm",
        "build-dataset",
        "--nist-index",
        str(NIST),
        "--label-source",
        "smarts",
        "--ontology",
        "v4",
        "--pipeline-version",
        "v4_ontology",
        "--model-kind",
        "specific",
        "--min-label-positives",
        "10",
        "--require-structure",
        "--enrich-pubchem",
        "--pubchem-cache",
        str(PUBCHEM),
        "--pubchem-offline-only",
        "--store-raw-spectra",
        "--progress-every",
        "500",
    ]
    train_common = [
        py,
        "-m",
        "ml.structural_fg_svm",
        "train",
        "--model-kind",
        "specific",
        "--ontology",
        "v4",
        "--pipeline-version",
        "v4_ontology",
        "--label-source",
        "smarts",
        "--calibration",
        "sigmoid",
        "--split",
        "molecule",
        "--min-label-positives",
        "10",
        "--hard-negative-mode",
        "on",
        "--threshold-objective",
        "balanced_guarded",
        "--random-state",
        "13",
        "--memory-safe",
        "--n-jobs",
        "1",
        "--no-update-latest",
    ]

    timings: dict[str, float] = {}
    rows: list[dict] = [collect_baseline()]

    if not args.skip_build:
        if not PC_DS.with_suffix(".npz").is_file():
            timings["build_peakcodebook"] = run(
                build_common
                + [
                    "--out-prefix",
                    str(PC_DS),
                    "--feature-set",
                    "spectral+evidence_v2+peakcodebook",
                ]
            )
        else:
            print(f"Reusing peakcodebook dataset: {PC_DS.with_suffix('.npz')}", flush=True)
        timings["build_deconv"] = run(
            build_common
            + [
                "--out-prefix",
                str(DECONV_DS),
                "--feature-set",
                "spectral+evidence_v2+deconv",
                "--deconv-mode",
                "fast",
            ]
        )
        meta_d = json.loads(Path(str(DECONV_DS) + ".meta.json").read_text(encoding="utf-8"))
        print(json.dumps({"deconv_build_verify": {k: meta_d.get(k) for k in ("feature_dim", "n_deconv", "deconv_failure_count", "deconv_runtime_mean_ms")}}, indent=2))

        timings["build_combined"] = run(
            build_common
            + [
                "--out-prefix",
                str(COMB_DS),
                "--feature-set",
                "spectral+evidence_v2+peakcodebook+deconv",
                "--deconv-mode",
                "fast",
            ]
        )

    if not args.skip_train:
        # Sequential: deconv train first (after build); peakcodebook/combined only if datasets exist
        if DECONV_DS.with_suffix(".npz").is_file():
            timings["train_deconv"] = run(
                train_common
                + [
                    "--dataset-prefix",
                    str(DECONV_DS),
                    "--out",
                    str(DECONV_TRAIN),
                    "--deconv-mode",
                    "fast",
                ]
            )
            rows.append({"experiment": "C_deconv", **collect_train_metrics(DECONV_TRAIN)})
        if PC_DS.with_suffix(".npz").is_file():
            timings["train_peakcodebook"] = run(
                train_common + ["--dataset-prefix", str(PC_DS), "--out", str(PC_TRAIN)]
            )
            rows.append({"experiment": "B_peakcodebook", **collect_train_metrics(PC_TRAIN)})
        if COMB_DS.with_suffix(".npz").is_file():
            timings["train_combined"] = run(
                train_common
                + [
                    "--dataset-prefix",
                    str(COMB_DS),
                    "--out",
                    str(COMB_TRAIN),
                    "--deconv-mode",
                    "fast",
                ]
            )
            rows.append({"experiment": "D_combined", **collect_train_metrics(COMB_TRAIN)})

    baseline_f1 = float(rows[0].get("macro_f1") or 0.54)
    winner = max(
        (r for r in rows if r.get("macro_f1") is not None),
        key=lambda r: float(r["macro_f1"]),
        default=rows[0],
    )

    lines = [
        "# v4 deconvolution training & benchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Why the earlier deconv build stopped",
        "",
        "The prior background job (`740115`) did **not** crash with a Python traceback — it was **interrupted** around SQLite row ~1500 while PubChem cache checkpoints were still running. Full `deconvolve_spectrum` per spectrum on ~18k rows is CPU-heavy without fast mode.",
        "",
        "## Fixes applied",
        "",
        "- `--deconv-mode fast|full|off` (default **fast** for builds)",
        "- Per-spectrum `extract_deconv_features()` — failures → zero vector, `fit_success=0`, build continues",
        "- Fast mode: skips `aromatic_fingerprint`, caps components, `maxfev=800`, skips low-signal regions",
        "- Build metadata: `deconv_failure_count`, `deconv_runtime_mean_ms`",
        "",
        "## Metrics summary",
        "",
        "| Experiment | feature_set | macro-F1 | Δ vs baseline | n_features | hard-neg FP | deconv failures | model |",
        "|------------|-------------|----------|---------------|------------|-------------|-----------------|-------|",
    ]
    for r in rows:
        f1 = r.get("macro_f1")
        delta = ""
        if f1 is not None:
            try:
                delta = f"{float(f1) - baseline_f1:+.4f}"
            except (TypeError, ValueError):
                delta = ""
        lines.append(
            f"| {r.get('experiment')} | `{r.get('feature_set')}` | {f1} | {delta} | {r.get('n_features')} | "
            f"{r.get('hard_negative_total_fp')} | {r.get('deconv_failure_count', '—')} | `{r.get('model_path', '')}` |"
        )

    lines += ["", "## Promotion decision", ""]
    promote_ok = False
    if winner.get("experiment") != "A_baseline_production" and float(winner.get("macro_f1") or 0) >= baseline_f1 + 0.005:
        lines.append(
            f"Candidate **{winner.get('experiment')}** leads on macro-F1 ({winner.get('macro_f1')}). "
            "Review hard-negative CSVs before promotion."
        )
        promote_ok = bool(args.promote_winner)
    else:
        lines.append("**Keep production baseline** (`struct_fg_specific_v4_ontology_latest.joblib`, 434-D). No promotion.")

    if promote_ok and winner.get("model_path"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive = ROOT / "ml" / "runs" / f"archive_struct_fg_specific_v4_ontology_{ts}.joblib"
        shutil.copy2(BASELINE_MODEL, archive)
        shutil.copy2(winner["model_path"], BASELINE_MODEL)
        lines.append(f"- Backed up prior latest to `{archive.resolve()}`")
        lines.append(f"- Promoted `{winner['model_path']}` → `{BASELINE_MODEL.resolve()}`")

    lines += [
        "",
        "## Timings (seconds)",
        "",
        json.dumps(timings, indent=2),
        "",
        "## Commands",
        "",
        f"Runner: `{REPORT.parent / 'run_v4_deconv_training_benchmark.py'}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(REPORT.resolve()), "winner": winner.get("experiment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
