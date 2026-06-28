"""Build experimental NPZ dataset from external SQLite index (does not touch production)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_external_dataset(
    *,
    sqlite_index: Path,
    out_prefix: Path,
    model_kind: str = "family",
    ontology: str = "v4",
    enrich_pubchem: bool = False,
    pubchem_cache: Path | None = None,
) -> int:
    """Delegate to ``ml.structural_fg_svm build-dataset`` with experimental paths."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "ml.structural_fg_svm",
        "build-dataset",
        "--nist-index",
        str(sqlite_index.resolve()),
        "--out-prefix",
        str(out_prefix),
        "--model-kind",
        model_kind,
        "--label-source",
        "smarts",
        "--feature-set",
        "spectral+evidence_v2",
        "--ontology",
        ontology,
        "--pipeline-version",
        "external_experimental",
        "--min-label-positives",
        "5",
    ]
    if enrich_pubchem:
        cache = pubchem_cache or Path("ml/runs/pubchem_train_writable.json")
        cmd.extend(["--enrich-pubchem", "--pubchem-cache", str(cache)])
    print("[build_external_dataset]", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build experimental NPZ from external SQLite")
    p.add_argument("--sqlite-index", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, default=Path("ml/runs/experimental/ds_external"))
    p.add_argument("--model-kind", default="family")
    p.add_argument("--ontology", default="v4")
    p.add_argument("--enrich-pubchem", action="store_true")
    args = p.parse_args(argv)
    rc = build_external_dataset(
        sqlite_index=args.sqlite_index,
        out_prefix=args.out_prefix,
        model_kind=args.model_kind,
        ontology=args.ontology,
        enrich_pubchem=args.enrich_pubchem,
    )
    meta = {
        "tier": "experimental",
        "sqlite": str(args.sqlite_index.resolve()),
        "out_prefix": str(args.out_prefix),
    }
    Path(str(args.out_prefix) + ".external_build.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return rc
