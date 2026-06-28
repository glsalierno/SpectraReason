"""CLI for external dataset ingestion (``python -m ml.external``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.dataset_quality import audit_index_cli, write_audit_markdown
from ml.external.build_confounder_benchmarks import build_confounder_benchmarks
from ml.external.build_external_dataset import build_external_dataset
from ml.external.import_jcamp_folder import ingest_jcamp_folder
from ml.external.import_open_polymer_ftir import ingest_open_polymer_ftir
from ml.external.import_sdbs import ingest_sdbs
from ml.external.library_adapter import ingest_library
from ml.external.merge_indexes import merge_indexes
from ml.external.ingest_common import load_source_registry


def cmd_ingest_sdbs(args: argparse.Namespace) -> int:
    stats, db = ingest_sdbs(
        Path(args.raw_dir) if args.raw_dir else None,
        Path(args.out_db) if args.out_db else None,
        recursive=not args.no_recursive,
        manifest_path=Path(args.manifest) if getattr(args, "manifest", None) else None,
    )
    print(json.dumps({"stats": stats.__dict__, "sqlite": str(db)}, indent=2))
    return 0


def cmd_ingest_jcamp(args: argparse.Namespace) -> int:
    lib_path = args.library_path or args.folder
    if lib_path is None:
        raise SystemExit("Provide folder positional arg or --library-path")
    if args.library_path or (args.library_source and args.library_source != "jcamp"):
        stats, db = ingest_library(
            library_path=Path(lib_path),
            library_source=args.library_source or "jcamp",
            out_db=Path(args.out_db),
            source_id=args.source_id,
        )
    else:
        stats, db = ingest_jcamp_folder(
            Path(lib_path),
            Path(args.out_db),
            source_id=args.source_id or "user_jcamp",
            source_name=args.source_name,
            source_license=args.license or "user-provided",
        )
    print(json.dumps({"stats": stats.__dict__, "sqlite": str(db)}, indent=2))
    return 0


def cmd_ingest_polymer(args: argparse.Namespace) -> int:
    stats, db = ingest_open_polymer_ftir(
        Path(args.raw_dir) if args.raw_dir else None,
        Path(args.out_db) if args.out_db else None,
        source_id=args.source_id or "open_polymer_atr",
    )
    print(json.dumps({"stats": stats.__dict__, "sqlite": str(db)}, indent=2))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    summary = merge_indexes([Path(p) for p in args.inputs], Path(args.out_db))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    return build_external_dataset(
        sqlite_index=Path(args.sqlite_index),
        out_prefix=Path(args.out_prefix),
        model_kind=args.model_kind,
        ontology=args.ontology,
        enrich_pubchem=args.enrich_pubchem,
    )


def cmd_benchmarks(args: argparse.Namespace) -> int:
    summary = build_confounder_benchmarks(Path(args.sqlite_index), Path(args.out_dir))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    report = audit_index_cli(Path(args.sqlite_index), Path(args.out_report))
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_summarize_coverage(args: argparse.Namespace) -> int:
    from ml.external.summarize_confounder_coverage import run_summarize

    summary = run_summarize(
        sqlite_index=Path(args.sqlite_index) if args.sqlite_index else None,
        experimental_dir=Path(args.experimental_dir) if args.experimental_dir else None,
        out_json=Path(args.out_json) if args.out_json else None,
        out_md=Path(args.out_md) if args.out_md else None,
        update_manifests=not args.no_update_manifests,
        update_expansion_audit=not args.no_update_audit,
    )
    print(json.dumps({"total_spectra": summary["total_spectra"], "gaps": len(summary["coverage_gaps"])}, indent=2))
    return 0


def cmd_list_sources(_args: argparse.Namespace) -> int:
    for row in load_source_registry():
        print(
            f"{row.get('source_id')}: {row.get('source_name')} "
            f"[{row.get('ingestion_status')}] ~{row.get('spectrum_count_estimate')} spectra"
        )
    return 0


def cmd_full_pipeline(args: argparse.Namespace) -> int:
    """Ingest examples → merge → QA → benchmarks → optional NPZ."""
    examples = Path("examples/spectra")
    out_db = Path(args.out_db or "data/experimental/merged_external_index.sqlite")
    stats, _ = ingest_jcamp_folder(
        examples,
        out_db,
        source_id="examples_reference",
        source_name="FTIR_SVM examples (reference JDX)",
        source_license="project examples — not for external redistribution",
        redistribution_allowed=False,
    )
    report = audit_index_cli(out_db, Path("reports/dataset_ingestion_audit.md"))
    build_confounder_benchmarks(out_db)
    print(json.dumps({"ingest": stats.__dict__, "qa": report.to_dict()}, indent=2))
    if args.build_npz:
        return build_external_dataset(sqlite_index=out_db, out_prefix=Path(args.out_prefix))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml.external", description="External FTIR dataset ingestion")
    sub = p.add_subparsers(dest="command", required=True)

    p_sdbs = sub.add_parser("ingest-sdbs", help="Ingest user-downloaded SDBS JCAMP folder")
    p_sdbs.add_argument("--raw-dir", type=Path, default=None)
    p_sdbs.add_argument("--out-db", type=Path, default=None)
    p_sdbs.add_argument("--no-recursive", action="store_true")
    p_sdbs.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest (default: raw/sdbs/sdbs_download_manifest.csv)",
    )
    p_sdbs.set_defaults(func=cmd_ingest_sdbs)

    p_j = sub.add_parser("ingest-jcamp-folder", help="Ingest JCAMP folder or library plugin")
    p_j.add_argument("folder", type=Path, nargs="?", default=None)
    p_j.add_argument("--out-db", type=Path, required=True)
    p_j.add_argument("--source-id", default=None)
    p_j.add_argument("--source-name", default=None)
    p_j.add_argument("--license", default=None)
    p_j.add_argument("--library-path", type=Path, default=None)
    p_j.add_argument("--library-source", default="jcamp")
    p_j.set_defaults(func=cmd_ingest_jcamp)

    p_poly = sub.add_parser("ingest-open-polymer", help="Ingest open polymer ATR downloads")
    p_poly.add_argument("--raw-dir", type=Path, default=None)
    p_poly.add_argument("--out-db", type=Path, default=None)
    p_poly.add_argument("--source-id", default=None)
    p_poly.set_defaults(func=cmd_ingest_polymer)

    p_m = sub.add_parser("merge-indexes", help="Merge SQLite indexes")
    p_m.add_argument("inputs", nargs="+", type=Path)
    p_m.add_argument("--out-db", type=Path, required=True)
    p_m.set_defaults(func=cmd_merge)

    p_b = sub.add_parser("build-external-dataset", help="Build experimental NPZ (not production)")
    p_b.add_argument("--sqlite-index", type=Path, required=True)
    p_b.add_argument("--out-prefix", type=Path, default="ml/runs/experimental/ds_external")
    p_b.add_argument("--model-kind", default="family")
    p_b.add_argument("--ontology", default="v4")
    p_b.add_argument("--enrich-pubchem", action="store_true")
    p_b.set_defaults(func=cmd_build_dataset)

    p_bm = sub.add_parser("build-confounder-benchmarks", help="Build benchmark JSON subsets")
    p_bm.add_argument("--sqlite-index", type=Path, required=True)
    p_bm.add_argument("--out-dir", type=Path, default="data/benchmark_sets")
    p_bm.set_defaults(func=cmd_benchmarks)

    p_qa = sub.add_parser("dataset-qa", help="Run QA audit on SQLite index")
    p_qa.add_argument("--sqlite-index", type=Path, required=True)
    p_qa.add_argument("--out-report", type=Path, default="reports/dataset_ingestion_audit.md")
    p_qa.set_defaults(func=cmd_qa)

    p_cov = sub.add_parser(
        "summarize-confounder-coverage",
        help="Report targeted confounder coverage and update manifests/audit",
    )
    p_cov.add_argument("--sqlite-index", type=Path, default=None, help="Single index (default: all in experimental/)")
    p_cov.add_argument("--experimental-dir", type=Path, default=None)
    p_cov.add_argument("--out-json", type=Path, default=None)
    p_cov.add_argument("--out-md", type=Path, default=None)
    p_cov.add_argument("--no-update-manifests", action="store_true")
    p_cov.add_argument("--no-update-audit", action="store_true")
    p_cov.set_defaults(func=cmd_summarize_coverage)

    p_ls = sub.add_parser("list-sources", help="List registered open sources")
    p_ls.set_defaults(func=cmd_list_sources)

    p_demo = sub.add_parser("demo-ingest-examples", help="Ingest examples/spectra for validation")
    p_demo.add_argument("--out-db", type=Path, default=None)
    p_demo.add_argument("--build-npz", action="store_true")
    p_demo.add_argument("--out-prefix", type=Path, default="ml/runs/experimental/ds_examples")
    p_demo.set_defaults(func=cmd_full_pipeline)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
