"""Import a folder of JCAMP (.jdx/.dx) spectra into an experimental SQLite index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.external.ingest_common import (
    IngestStats,
    build_ingest_result,
    load_raw_spectrum,
    preprocess_for_index,
    registry_entry,
)
from ml.external.spectrum_index import ensure_db, set_index_meta, upsert_many, write_manifest
from ml.external.tagging import tag_spectrum


def _iter_jcamp_files(root: Path, recursive: bool) -> list[Path]:
    patterns = ("*.jdx", "*.JDX", "*.dx", "*.DX", "*.jcm", "*.JCM")
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat) if recursive else root.glob(pat))
    return sorted({p.resolve() for p in files})


def ingest_jcamp_folder(
    folder: Path,
    out_db: Path,
    *,
    source_id: str,
    source_name: str | None = None,
    source_license: str | None = None,
    source_url: str | None = None,
    redistribution_allowed: bool | None = None,
    recursive: bool = True,
    tags: list[str] | None = None,
    manifest_path: Path | None = None,
) -> tuple[IngestStats, Path]:
    reg = registry_entry(source_id) or {}
    source_name = source_name or str(reg.get("source_name") or source_id)
    source_license = source_license or str(reg.get("license") or "unknown")
    source_url = source_url or reg.get("url")
    if redistribution_allowed is None and "redistribution_allowed" in reg:
        redistribution_allowed = bool(reg.get("redistribution_allowed"))

    stats = IngestStats()
    conn = ensure_db(out_db)
    rows: list[Any] = []
    seen_ids: set[str] = set()

    for path in _iter_jcamp_files(folder, recursive):
        stats.attempted += 1
        try:
            wn_raw, y_raw, md, hint = load_raw_spectrum(path)
            prepped = preprocess_for_index(wn_raw, y_raw, md, intensity_mode=hint)
            if prepped is None:
                stats.bump_failure("preprocess_rejected")
                continue
            wn, y, md_out = prepped
            oid = path.stem
            row = build_ingest_result(
                source_id=source_id,
                source_name=source_name,
                source_license=source_license,
                original_identifier=oid,
                source_path=path,
                wn=wn,
                y=y,
                md=md_out,
                source_url=source_url,
                redistribution_allowed=redistribution_allowed,
            )
            if row is None:
                stats.bump_failure("qa_rejected")
                continue
            if row.reference_id in seen_ids:
                stats.bump_failure("duplicate_reference_id")
                continue
            seen_ids.add(row.reference_id)
            row.tags = tag_spectrum(row.metadata, wn, y, tags)
            row.metadata["dataset_tags"] = row.tags
            rows.append(row)
            stats.ingested += 1
        except Exception as exc:
            stats.bump_failure(type(exc).__name__)

    upsert_many(conn, rows)
    set_index_meta(conn, "source_id", source_id)
    set_index_meta(conn, "ingestion_adapter", "import_jcamp_folder")
    conn.commit()
    conn.close()

    mpath = manifest_path or out_db.with_suffix(".manifest.json")
    write_manifest(out_db, mpath, {"source_id": source_id, "stats": stats.__dict__})
    return stats, out_db
