"""SQLite spectral index compatible with NIST ``spectra`` schema."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ml.external.ingest_common import IngestResult


def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spectra (
            reference_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_archive TEXT,
            internal_path TEXT,
            virtual_path TEXT,
            file_size INTEGER NOT NULL,
            mtime INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            wn_json TEXT NOT NULL,
            y_json TEXT NOT NULL
        )
        """
    )
    for col, ctype in (
        ("source_archive", "TEXT"),
        ("internal_path", "TEXT"),
        ("virtual_path", "TEXT"),
    ):
        try:
            conn.execute(f"ALTER TABLE spectra ADD COLUMN {col} {ctype}")
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spectra_source_path ON spectra(source_path)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return conn


def set_index_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)",
        (key, value),
    )


def upsert_spectrum(conn: sqlite3.Connection, row: IngestResult) -> None:
    p = Path(row.source_path)
    st = p.stat()
    conn.execute(
        """
        INSERT OR REPLACE INTO spectra (
            reference_id, source_path, source_archive, internal_path, virtual_path,
            file_size, mtime, metadata_json, wn_json, y_json
        ) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.reference_id,
            row.source_path,
            row.source_path,
            int(st.st_size),
            int(st.st_mtime),
            json.dumps(row.metadata, ensure_ascii=False),
            json.dumps(np.asarray(row.wn_cm1, dtype=float).tolist()),
            json.dumps(np.asarray(row.y_processed, dtype=float).tolist()),
        ),
    )


def upsert_many(conn: sqlite3.Connection, rows: Iterable[IngestResult]) -> int:
    n = 0
    for row in rows:
        upsert_spectrum(conn, row)
        n += 1
    return n


def count_spectra(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0])
    finally:
        conn.close()


def load_all_metadata(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT reference_id, metadata_json FROM spectra")
        out: list[dict[str, Any]] = []
        for rid, mj in cur:
            md = json.loads(mj) if mj else {}
            md["reference_id"] = rid
            out.append(md)
        return out
    finally:
        conn.close()


def write_manifest(db_path: Path, manifest_path: Path, extra: dict[str, Any] | None = None) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sqlite": str(db_path.resolve()),
        "spectrum_count": count_spectra(db_path),
        "metadata_samples": load_all_metadata(db_path)[:5],
    }
    if extra:
        payload.update(extra)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
