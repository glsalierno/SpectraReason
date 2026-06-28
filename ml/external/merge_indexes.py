"""Merge multiple compatible SQLite spectral indexes into one experimental index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ml.external.spectrum_index import ensure_db, set_index_meta


def merge_indexes(
    inputs: list[Path],
    out_db: Path,
    *,
    dedupe_by_reference_id: bool = True,
) -> dict[str, int]:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    conn_out = ensure_db(out_db)
    merged = 0
    skipped_dup = 0
    per_source: dict[str, int] = {}

    for db_in in inputs:
        if not db_in.is_file():
            continue
        conn_in = sqlite3.connect(str(db_in))
        try:
            rows = conn_in.execute(
                "SELECT reference_id, source_path, source_archive, internal_path, virtual_path, "
                "file_size, mtime, metadata_json, wn_json, y_json FROM spectra"
            ).fetchall()
            for row in rows:
                rid = row[0]
                if dedupe_by_reference_id:
                    exists = conn_out.execute(
                        "SELECT 1 FROM spectra WHERE reference_id = ?", (rid,)
                    ).fetchone()
                    if exists:
                        skipped_dup += 1
                        continue
                conn_out.execute(
                    """
                    INSERT OR REPLACE INTO spectra VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    row,
                )
                merged += 1
                try:
                    md = json.loads(row[7])
                    sid = str(md.get("source_id") or "unknown")
                except Exception:
                    sid = "unknown"
                per_source[sid] = per_source.get(sid, 0) + 1
        finally:
            conn_in.close()

    set_index_meta(conn_out, "merge_inputs", json.dumps([str(p) for p in inputs]))
    conn_out.commit()
    conn_out.close()
    return {"merged": merged, "skipped_duplicate": skipped_dup, "per_source": per_source}
