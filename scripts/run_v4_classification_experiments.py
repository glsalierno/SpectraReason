#!/usr/bin/env python3
"""
Run controlled v4 classification improvement experiments (A–E).

Does not overwrite production *_latest.joblib (uses --no-update-latest).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIST_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent
    / "NIST"
    / "reference_libraries"
    / "nistchemdata_ir_index_v7_fresh.sqlite"
)
EXP_DIR = ROOT / "ml" / "runs" / "experiments" / "v4_classification_improvement"
PUBCHEM = ROOT / "ml" / "runs" / "pubchem_train_writable.json"


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    nist = Path(sys.argv[1]) if len(sys.argv) > 1 else NIST_DEFAULT
    if not nist.is_file():
        print(f"NIST index not found: {nist}", file=sys.stderr)
        return 1

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    common_build = [
        py,
        "-m",
        "ml.structural_fg_svm",
        "build-dataset",
        "--nist-index",
        str(nist),
        "--label-source",
        "smarts",
        "--ontology",
        "v4",
        "--pipeline-version",
        "v4_ontology",
        "--require-structure",
        "--enrich-pubchem",
        "--pubchem-cache",
        str(PUBCHEM),
        "--pubchem-offline-only",
        "--min-label-positives",
        "10",
    ]
    common_train = [
        py,
        "-m",
        "ml.structural_fg_svm",
        "train",
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
        "--hard-negative-mode",
        "on",
        "--threshold-objective",
        "balanced_guarded",
        "--hard-negative-weight",
        "1.5",
        "--random-state",
        "13",
        "--no-update-latest",
    ]

    ds_base_fam = EXP_DIR / "ds_baseline_family"
    ds_base_spec = EXP_DIR / "ds_baseline_specific"
    ds_pc_fam = EXP_DIR / "ds_peakcodebook_family"
    ds_pc_spec = EXP_DIR / "ds_peakcodebook_specific"

    for kind, prefix, min_pos in (
        ("family", ds_base_fam, "20"),
        ("specific", ds_base_spec, "10"),
    ):
        run(
            common_build
            + [
                "--out-prefix",
                str(prefix),
                "--model-kind",
                kind,
                "--feature-set",
                "spectral+evidence_v2",
                "--min-label-positives",
                min_pos,
            ]
        )

    for kind, prefix, min_pos in (
        ("family", ds_pc_fam, "20"),
        ("specific", ds_pc_spec, "10"),
    ):
        run(
            common_build
            + [
                "--out-prefix",
                str(prefix),
                "--model-kind",
                kind,
                "--feature-set",
                "spectral+evidence_v2+peakcodebook",
                "--store-raw-spectra",
                "--min-label-positives",
                min_pos,
            ]
        )

    experiments = [
        ("A_baseline_family", ds_base_fam, "family", "spectral+evidence_v2", "none", False),
        ("A_baseline_specific", ds_base_spec, "specific", "spectral+evidence_v2", "none", False),
        ("B_peakcodebook_family", ds_pc_fam, "family", "spectral+evidence_v2+peakcodebook", "none", False),
        ("B_peakcodebook_specific", ds_pc_spec, "specific", "spectral+evidence_v2+peakcodebook", "none", False),
        ("C_peakcodebook_aug_light_specific", ds_pc_spec, "specific", "spectral+evidence_v2+peakcodebook", "light", False),
        ("D_peakcodebook_aug_moderate_specific", ds_pc_spec, "specific", "spectral+evidence_v2+peakcodebook", "moderate", False),
        ("E_peakcodebook_benchmark_specific", ds_pc_spec, "specific", "spectral+evidence_v2+peakcodebook", "none", True),
    ]

    results: list[dict] = []
    for name, ds_prefix, kind, _fs, aug, bench in experiments:
        out = EXP_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = (
            common_train
            + [
                "--dataset-prefix",
                str(ds_prefix),
                "--model-kind",
                kind,
                "--min-label-positives",
                "20" if kind == "family" else "10",
                "--augment",
                aug,
                "--out",
                str(out),
            ]
        )
        if bench:
            cmd += [
                "--benchmark",
                "--benchmark-models",
                "linear_svm",
                "extra_trees",
                "hist_gradient_boosting",
                "logistic_elasticnet",
            ]
        run(cmd)
        meta_glob = list(out.glob("struct_fg_*_training_metadata.json"))
        if meta_glob:
            meta = json.loads(meta_glob[0].read_text(encoding="utf-8"))
            results.append(
                {
                    "experiment": name,
                    "macro_f1": meta.get("metrics_summary", {}).get("macro_f1"),
                    "n_train": meta.get("split", {}).get("n_train"),
                    "n_augmented": meta.get("augmentation", {}).get("n_augmented_rows", 0),
                    "feature_set": meta.get("feature_set"),
                }
            )

    summary_path = EXP_DIR / "experiment_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(summary_path.resolve()), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
