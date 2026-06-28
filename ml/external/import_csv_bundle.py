"""Import CSV spectra (optionally with manifest metadata) into SQLite."""

from __future__ import annotations

import csv
from pathlib import Path

from ml.external.ingest_common import (
    IngestStats,
    build_ingest_result,
    preprocess_for_index,
    registry_entry,
)
from ml.external.spectrum_index import ensure_db, set_index_meta, upsert_many, write_manifest
from ml.external.tagging import tag_spectrum
from lib.ftir_foundation import read_csv_spectrum


def _load_manifest(folder: Path, manifest_csv: Path | None) -> dict[str, dict[str, str]]:
    mpath = manifest_csv or (folder / "manifest.csv")
    if not mpath.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with mpath.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("file") or row.get("filename") or row.get("path")
            if key:
                out[Path(key).name] = row
    return out


def ingest_csv_bundle(
    folder: Path,
    out_db: Path,
    *,
    source_id: str,
    source_name: str | None = None,
    source_license: str | None = None,
    manifest_csv: Path | None = None,
) -> tuple[IngestStats, Path]:
    reg = registry_entry(source_id) or {}
    source_name = source_name or str(reg.get("source_name") or source_id)
    source_license = source_license or str(reg.get("license") or "user-provided")
    manifest = _load_manifest(folder, manifest_csv)

    stats = IngestStats()
    conn = ensure_db(out_db)
    rows = []
    for path in sorted(folder.rglob("*.csv")):
        if path.name.lower() == "manifest.csv":
            continue
        stats.attempted += 1
        try:
            wn_raw, y_raw = read_csv_spectrum(path)
            md = {"title": path.stem, "name": path.stem, "xunits": "1/CM"}
            if path.name in manifest:
                for k, v in manifest[path.name].items():
                    if v:
                        md[k.lower()] = v
            prepped = preprocess_for_index(wn_raw, y_raw, md, intensity_mode="auto")
            if prepped is None:
                stats.bump_failure("preprocess_rejected")
                continue
            wn, y, md_out = prepped
            row = build_ingest_result(
                source_id=source_id,
                source_name=source_name,
                source_license=source_license,
                original_identifier=path.stem,
                source_path=path,
                wn=wn,
                y=y,
                md=md_out,
            )
            if row is None:
                stats.bump_failure("qa_rejected")
                continue
            row.tags = tag_spectrum(row.metadata, wn, y)
            row.metadata["dataset_tags"] = row.tags
            rows.append(row)
            stats.ingested += 1
        except Exception as exc:
            stats.bump_failure(type(exc).__name__)

    upsert_many(conn, rows)
    set_index_meta(conn, "source_id", source_id)
    set_index_meta(conn, "ingestion_adapter", "import_csv_bundle")
    conn.commit()
    conn.close()
    write_manifest(out_db, out_db.with_suffix(".manifest.json"), {"source_id": source_id})
    return stats, out_db
