#!/usr/bin/env python3
"""
Train a multi-label functional-group SVM on spectra stored in a NistChemData SQLite index.

Artifact format matches ``heuristics/fg_svm_plugin.predict_fg_priors_with_svm``:
``joblib.load`` returns ``{"model": OneVsRestClassifier, "labels": list[str]}``.

Labels are inferred from compound **name / title / formula** metadata using conservative
keyword rules (no RDKit). Refine ``FG_RULES`` for your chemistry domain.

Usage (from ``chunks`` root, ``PYTHONPATH=.``)::

    python -m ml.ftir_fg_svm build-labels --nist-index NIST/reference_libraries/nistchemdata_ir_index_v7_fresh.sqlite --out ml/runs/nist_fg_labels.csv
    python -m ml.ftir_fg_svm train --labels ml/runs/nist_fg_labels.csv --nist-index NIST/reference_libraries/nistchemdata_ir_index_v7_fresh.sqlite --model-out ml/runs/nist_fg_svm.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- Feature vector must stay aligned with heuristics/fg_svm_plugin._featurize ---


def featurize(wn: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(wn, dtype=float)
    yy = np.asarray(y, dtype=float)
    f = [
        float(np.mean(yy)),
        float(np.std(yy)),
        float(np.max(yy)),
        float(np.min(yy)),
    ]
    for lo, hi in ((2500, 3700), (1650, 1820), (1200, 1700), (900, 1400), (650, 900)):
        m = (x >= lo) & (x <= hi)
        if int(np.count_nonzero(m)) < 3:
            f.extend([0.0, 0.0])
            continue
        seg = yy[m]
        f.extend([float(np.mean(seg)), float(np.std(seg))])
    return np.asarray(f, dtype=float).reshape(1, -1)


# (label, keywords) — substring match on lowercased metadata text; order defines CSV columns
FG_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("alcohol", ("hydroxy", " alcohol", "alcohol,", "diol", "phenol", " glycol")),
    ("amine", ("amine", "amino", " azide", " hydrazine")),
    ("carbonyl", ("ketone", "aldehyde", " quinone", " lactam")),
    ("carboxylic_acid", ("carboxylic", "acid,", "acid ", "carboxyl")),
    ("ester", ("ester", "lactone")),
    ("ether", (" ether", "ether,", "epoxide", " furan", " pyran")),
    ("aromatic", ("benzene", "phenyl", "tolyl", "naphth", "anthrac", "pyridine", "furane")),
    ("halide", ("chloro", "bromo", "fluoro", "iodo", " chloride", " bromide")),
    ("nitrile", ("nitrile", "cyano")),
    ("nitro", ("nitro")),
    ("alkene", ("ethylene", " propene", "butene", "pentene", " alkene", "olefin")),
    ("alkyne", ("yne", " acetylene", "alkyne")),
]


def _metadata_text(md: dict[str, Any]) -> str:
    parts = [md.get("name"), md.get("title"), md.get("formula")]
    return " ".join(str(p or "") for p in parts).lower()


def infer_fg_vector(md: dict[str, Any]) -> dict[str, int]:
    text = _metadata_text(md)
    out: dict[str, int] = {}
    for label, kws in FG_RULES:
        out[label] = int(any(kw in text for kw in kws))
    return out


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    p = path.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"SQLite index not found: {p}")
    return sqlite3.connect(str(p))


def cmd_build_labels(args: argparse.Namespace) -> int:
    db = Path(args.nist_index)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_names = [t[0] for t in FG_RULES]

    conn = _connect_sqlite(db)
    try:
        cur = conn.execute("SELECT reference_id, metadata_json FROM spectra")
        rows_out: list[dict[str, Any]] = []
        for rid, mj in cur:
            try:
                md = json.loads(mj) if mj else {}
            except json.JSONDecodeError:
                md = {}
            vec = infer_fg_vector(md)
            row = {"reference_id": rid}
            for k in label_names:
                row[k] = vec[k]
            rows_out.append(row)
    finally:
        conn.close()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["reference_id", *label_names])
        w.writeheader()
        w.writerows(rows_out)

    n_any = sum(1 for r in rows_out if sum(int(r[k]) for k in label_names) > 0)
    print(
        json.dumps(
            {
                "wrote": str(out_path.resolve()),
                "n_rows": len(rows_out),
                "n_with_any_label": n_any,
                "label_names": label_names,
            },
            indent=2,
        )
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        import joblib
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
    except ImportError as e:
        print(
            "Missing dependency (install in the same environment): pip install scikit-learn joblib",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    labels_csv = Path(args.labels)
    db_path = Path(args.nist_index)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    if not labels_csv.is_file():
        raise SystemExit(
            f"Labels file not found: {labels_csv.resolve()}\n"
            "Create it first (from repo root, PYTHONPATH=.):\n"
            f'  python -m ml.ftir_fg_svm build-labels --nist-index "{args.nist_index}" '
            f'--out "{args.labels}"'
        )

    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "reference_id" not in reader.fieldnames:
            raise SystemExit("labels CSV must have a reference_id column")
        label_cols = [c for c in reader.fieldnames if c != "reference_id"]
        if not label_cols:
            raise SystemExit("no label columns in CSV (columns besides reference_id)")
        rows = list(reader)

    conn = _connect_sqlite(db_path)
    try:
        X_list: list[np.ndarray] = []
        Y_list: list[list[int]] = []
        skipped_no_row = 0
        skipped_parse = 0
        skipped_no_label = 0

        for row in rows:
            rid = row["reference_id"]
            cur = conn.execute(
                "SELECT wn_json, y_json FROM spectra WHERE reference_id = ?", (rid,)
            )
            got = cur.fetchone()
            if got is None:
                skipped_no_row += 1
                continue
            wn_s, y_s = got
            try:
                wn = np.asarray(json.loads(wn_s), dtype=float)
                yy = np.asarray(json.loads(y_s), dtype=float)
            except (json.JSONDecodeError, TypeError):
                skipped_parse += 1
                continue
            y_vec = [int(row[c]) for c in label_cols]
            if sum(y_vec) == 0:
                skipped_no_label += 1
                continue
            X_list.append(featurize(wn, yy).ravel())
            Y_list.append(y_vec)

        if len(X_list) < 10:
            raise SystemExit(
                f"too few training rows after filtering ({len(X_list)}); "
                f"skipped_no_row={skipped_no_row} skipped_parse={skipped_parse} "
                f"skipped_no_label={skipped_no_label}"
            )

        X = np.vstack(X_list)
        Y = np.asarray(Y_list, dtype=int)

        # Drop labels with no positive samples (shouldn't happen after filter) or only one class
        keep_mask = np.ones(len(label_cols), dtype=bool)
        for j, name in enumerate(label_cols):
            col = Y[:, j]
            if col.max() == 0 or col.min() == col.max():
                keep_mask[j] = False
        if not np.all(keep_mask):
            dropped = [label_cols[j] for j in range(len(label_cols)) if not keep_mask[j]]
            print(json.dumps({"dropped_labels_no_variance": dropped}, indent=2))
            label_cols = [label_cols[j] for j in range(len(label_cols)) if keep_mask[j]]
            Y = Y[:, keep_mask]

        if Y.shape[1] == 0:
            raise SystemExit("no label columns with both positives and negatives")

        # Default n_jobs=1: sklearn/joblib parallel OvR on Windows + Python 3.13 can hit
        # multiprocessing backend errors (e.g. stray posix subprocess paths). Use --n-jobs > 1 only if it works on your stack.
        n_jobs = max(1, int(args.n_jobs))
        clf = OneVsRestClassifier(
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
            n_jobs=n_jobs,
        )
        clf.fit(X, Y)

        artifact = {"model": clf, "labels": [str(x).lower() for x in label_cols]}
        joblib.dump(artifact, model_out)

        print(
            json.dumps(
                {
                    "model_out": str(model_out.resolve()),
                    "n_train": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "labels": artifact["labels"],
                    "n_jobs": n_jobs,
                    "skipped_no_row": skipped_no_row,
                    "skipped_parse": skipped_parse,
                    "skipped_no_label": skipped_no_label,
                },
                indent=2,
            )
        )
    finally:
        conn.close()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FTIR functional-group SVM (NistChem SQLite index)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_bl = sub.add_parser("build-labels", help="Write CSV of heuristic FG labels from index metadata")
    p_bl.add_argument("--nist-index", required=True, help="Path to nistchemdata *.sqlite")
    p_bl.add_argument("--out", required=True, help="Output CSV path")
    p_bl.set_defaults(func=cmd_build_labels)

    p_tr = sub.add_parser("train", help="Train OvR SVM and save joblib artifact")
    p_tr.add_argument("--labels", required=True, help="CSV from build-labels")
    p_tr.add_argument("--nist-index", required=True, help="Same SQLite index used for spectra")
    p_tr.add_argument("--model-out", required=True, help="Output .joblib path")
    p_tr.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel OvR fits (default 1; avoids joblib multiprocessing issues on some Windows/Python builds)",
    )
    p_tr.set_defaults(func=cmd_train)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
