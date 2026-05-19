#!/usr/bin/env python3
"""Benchmark deconv feature sets (A–D) for v4 specific model; writes reports/v4_deconv_feature_benchmark.md"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIST = ROOT.parent / "NIST" / "reference_libraries" / "nistchemdata_ir_index_v7_fresh.sqlite"
EXP = ROOT / "ml" / "runs" / "experiments" / "v4_deconv_benchmark"
REPORT = ROOT / "reports" / "v4_deconv_feature_benchmark.md"
PUBCHEM = ROOT / "ml" / "runs" / "pubchem_train_writable.json"

EXPERIMENTS = (
    ("A_baseline", "spectral+evidence_v2"),
    ("B_peakcodebook", "spectral+evidence_v2+peakcodebook"),
    ("C_deconv", "spectral+evidence_v2+deconv"),
    ("D_combined", "spectral+evidence_v2+peakcodebook+deconv"),
)


def run(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return time.perf_counter() - t0


def collect_metrics(out_dir: Path) -> dict:
    meta_files = list(out_dir.glob("*_training_metadata.json"))
    if not meta_files:
        return {}
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    ms = meta.get("metrics_summary") or {}
    hn = meta.get("hard_negative_summary") or {}
    return {
        "macro_f1": ms.get("macro_f1"),
        "n_train": meta.get("split", {}).get("n_train"),
        "n_test": meta.get("split", {}).get("n_test"),
        "n_features": ms.get("n_features"),
        "hard_negative_total_fp": hn.get("total_fp"),
        "feature_set": meta.get("feature_set"),
    }


def main() -> int:
    if not NIST.is_file():
        print(f"NIST index missing: {NIST}", file=sys.stderr)
        return 1

    py = sys.executable
    EXP.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for tag, fs in EXPERIMENTS:
        ds = EXP / f"ds_{tag}"
        out = EXP / tag
        out.mkdir(parents=True, exist_ok=True)
        build_cmd = [
            py,
            "-m",
            "ml.structural_fg_svm",
            "build-dataset",
            "--nist-index",
            str(NIST),
            "--out-prefix",
            str(ds),
            "--model-kind",
            "specific",
            "--label-source",
            "smarts",
            "--feature-set",
            fs,
            "--ontology",
            "v4",
            "--pipeline-version",
            "v4_ontology",
            "--min-label-positives",
            "10",
            "--require-structure",
            "--enrich-pubchem",
            "--pubchem-cache",
            str(PUBCHEM),
            "--pubchem-offline-only",
            "--store-raw-spectra",
        ]
        t_build = run(build_cmd)
        train_cmd = [
            py,
            "-m",
            "ml.structural_fg_svm",
            "train",
            "--dataset-prefix",
            str(ds),
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
            "--no-update-latest",
            "--out",
            str(out),
        ]
        t_train = run(train_cmd)
        m = collect_metrics(out)
        m["experiment"] = tag
        m["feature_set"] = fs
        m["build_seconds"] = round(t_build, 1)
        m["train_seconds"] = round(t_train, 1)
        rows.append(m)

    baseline_f1 = float(rows[0].get("macro_f1") or 0) if rows else 0.0
    lines = [
        "# v4 deconvolution feature benchmark",
        "",
        f"**Workspace:** `{ROOT}`",
        "",
        "## Results (specific model, molecule split, seed=13)",
        "",
        "| Experiment | Feature set | macro-F1 | Δ vs A | n_features | hard-neg FP | train (s) |",
        "|------------|-------------|----------|--------|------------|-------------|-----------|",
    ]
    best = max(rows, key=lambda r: float(r.get("macro_f1") or 0), default={})
    for r in rows:
        f1 = r.get("macro_f1")
        delta = ""
        if f1 is not None and baseline_f1:
            delta = f"{float(f1) - baseline_f1:+.4f}"
        lines.append(
            f"| {r['experiment']} | `{r['feature_set']}` | {f1} | {delta} | {r.get('n_features')} | "
            f"{r.get('hard_negative_total_fp')} | {r.get('train_seconds')} |"
        )

    lines += [
        "",
        "## Recommendation",
        "",
    ]
    if best:
        lines.append(
            f"- **Best macro-F1 in this run:** `{best.get('experiment')}` (`{best.get('feature_set')}`, F1={best.get('macro_f1')})."
        )
    lines += [
        "- **Production default remains** `spectral+evidence_v2` on existing `struct_fg_specific_v4_ontology_latest.joblib` unless deconv/combined beats baseline on macro-F1 **and** does not increase hard-negative FP on phenol/ester/nitrile/nitro pairs.",
        "- Deconv features are **optional**; enable in training with `--feature-set spectral+evidence_v2+deconv` or combined with peakcodebook.",
        "- Reports: deconv tables appear only in **`--report-density audit`** (fitted component evidence, not ground truth).",
        "- Evidence/rules stay primary; ML advisory; guardrails apply soft deconv boosts/caps via `ml/ftir_guardrails.apply_deconv_soft_guardrails`.",
        "",
        "## Artifacts",
        "",
        f"`{EXP}`",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(REPORT.resolve()), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
