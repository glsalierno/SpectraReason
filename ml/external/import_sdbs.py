"""
Import SDBS (AIST Japan) spectra from **user-downloaded** local JCAMP files.

Layout (recommended first batch, 25–35 spectra):

  data/external_sources/raw/sdbs/
    nitro_positive/
    n_oxide_hard_negative/
    amide_positive/
    amide_hard_negative/
    siloxane_confounds/
    sdbs_download_manifest.csv

SDBS terms: https://sdbs.db.aist.go.jp — research use with citation; no bulk scrape.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.external.import_jcamp_folder import _iter_jcamp_files
from ml.external.ingest_common import (
    IngestStats,
    build_ingest_result,
    load_raw_spectrum,
    preprocess_for_index,
    registry_entry,
)
from ml.external.sdbs_batch import (
    DEFAULT_MANIFEST,
    SDBS_SUBFOLDERS,
    enrich_sdbs_metadata,
    load_sdbs_manifest,
)
from ml.external.spectrum_index import ensure_db, set_index_meta, upsert_many, write_manifest
from ml.external.tagging import tag_spectrum

SDBS_SOURCE_ID = "sdbs_aist"
DEFAULT_RAW = Path("data/external_sources/raw/sdbs")
DEFAULT_OUT = Path("data/experimental/sdbs_ir_index.sqlite")


def ingest_sdbs(
    raw_dir: Path | None = None,
    out_db: Path | None = None,
    *,
    recursive: bool = True,
    manifest_path: Path | None = None,
) -> tuple[IngestStats, Path]:
    reg = registry_entry(SDBS_SOURCE_ID) or {}
    raw = (raw_dir or DEFAULT_RAW).resolve()
    out = (out_db or DEFAULT_OUT).resolve()
    mpath = (manifest_path or raw / "sdbs_download_manifest.csv").resolve()

    if not raw.is_dir():
        raise FileNotFoundError(
            f"SDBS raw directory not found: {raw}\n"
            "Create subfolders per docs/SDBS_FIRST_BATCH.md and add JCAMP exports."
        )

    manifest = load_sdbs_manifest(mpath if mpath.is_file() else DEFAULT_MANIFEST)
    stats = IngestStats()
    conn = ensure_db(out)
    rows = []
    seen_ids: set[str] = set()
    per_folder: dict[str, int] = {}

    for path in _iter_jcamp_files(raw, recursive):
        stats.attempted += 1
        try:
            wn_raw, y_raw, md, hint = load_raw_spectrum(path)
            md = enrich_sdbs_metadata(md, path, raw, manifest)
            prepped = preprocess_for_index(wn_raw, y_raw, md, intensity_mode=hint)
            if prepped is None:
                stats.bump_failure("preprocess_rejected")
                continue
            wn, y, md_out = prepped

            oid = str(md_out.get("original_identifier") or md_out.get("sdbs_id") or path.stem)
            manual_tags = list(md_out.get("batch_tags") or [])
            row = build_ingest_result(
                source_id=SDBS_SOURCE_ID,
                source_name=str(reg.get("source_name") or "SDBS (AIST Japan)"),
                source_license=str(reg.get("license") or "SDBS terms — research use, cite AIST"),
                original_identifier=oid,
                source_path=path,
                wn=wn,
                y=y,
                md=md_out,
                source_url=str(reg.get("url") or "https://sdbs.db.aist.go.jp"),
                redistribution_allowed=bool(reg.get("redistribution_allowed", False)),
                tags=manual_tags,
            )
            if row is None:
                stats.bump_failure("qa_rejected")
                continue
            if row.reference_id in seen_ids:
                stats.bump_failure("duplicate_reference_id")
                continue
            seen_ids.add(row.reference_id)
            row.tags = tag_spectrum(row.metadata, wn, y, manual_tags)
            row.metadata["dataset_tags"] = row.tags
            rows.append(row)
            stats.ingested += 1
            folder = str(md_out.get("batch_folder") or "_root")
            per_folder[folder] = per_folder.get(folder, 0) + 1
        except Exception as exc:
            stats.bump_failure(type(exc).__name__)

    upsert_many(conn, rows)
    set_index_meta(conn, "source_id", SDBS_SOURCE_ID)
    set_index_meta(conn, "ingestion_adapter", "import_sdbs")
    set_index_meta(conn, "sdbs_manifest", str(mpath))
    conn.commit()
    conn.close()

    audit_path = raw / "sdbs_ingest_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "stats": stats.__dict__,
                "per_folder": per_folder,
                "manifest": str(mpath),
                "subfolders_expected": list(SDBS_SUBFOLDERS.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_manifest(
        out,
        out.with_suffix(".manifest.json"),
        {"source_id": SDBS_SOURCE_ID, "stats": stats.__dict__, "per_folder": per_folder},
    )
    return stats, out
