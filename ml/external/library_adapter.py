"""Plugin-style library adapters (JCAMP folder, SQLite, CSV bundle)."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from ml.external.import_jcamp_folder import ingest_jcamp_folder
from ml.external.import_csv_bundle import ingest_csv_bundle
from ml.external.ingest_common import IngestResult, IngestStats, registry_entry
from ml.external.spectrum_index import count_spectra, ensure_db


class LibraryAdapter(ABC):
    source_id: str

    @abstractmethod
    def ingest(self, *, out_db: Path, **kwargs: Any) -> tuple[IngestStats, Path]:
        ...


class JcampFolderAdapter(LibraryAdapter):
    def __init__(self, source_id: str = "user_jcamp") -> None:
        self.source_id = source_id

    def ingest(self, *, out_db: Path, library_path: Path, **kwargs: Any) -> tuple[IngestStats, Path]:
        reg = registry_entry(self.source_id) or {}
        stats, _ = ingest_jcamp_folder(
            library_path,
            out_db,
            source_id=self.source_id,
            source_name=str(kwargs.get("library_source") or reg.get("source_name") or "User JCAMP"),
            source_license=str(reg.get("license") or "user-provided"),
            recursive=bool(kwargs.get("recursive", True)),
            tags=kwargs.get("tags"),
        )
        return stats, out_db


class CsvBundleAdapter(LibraryAdapter):
    def __init__(self, source_id: str = "user_csv") -> None:
        self.source_id = source_id

    def ingest(self, *, out_db: Path, library_path: Path, **kwargs: Any) -> tuple[IngestStats, Path]:
        reg = registry_entry(self.source_id) or {}
        stats, _ = ingest_csv_bundle(
            library_path,
            out_db,
            source_id=self.source_id,
            source_name=str(kwargs.get("library_source") or reg.get("source_name") or "User CSV"),
            source_license=str(reg.get("license") or "user-provided"),
            manifest_csv=kwargs.get("manifest_csv"),
        )
        return stats, out_db


class SqlitePassthroughAdapter(LibraryAdapter):
    """Use an existing compatible SQLite index as-is."""

    def __init__(self, source_id: str = "sqlite_index") -> None:
        self.source_id = source_id

    def ingest(self, *, out_db: Path, library_path: Path, **kwargs: Any) -> tuple[IngestStats, Path]:
        import shutil

        shutil.copy2(library_path, out_db)
        n = count_spectra(out_db)
        return IngestStats(attempted=n, ingested=n), out_db


_ADAPTERS: dict[str, type[LibraryAdapter]] = {
    "jcamp": JcampFolderAdapter,
    "jcamp_folder": JcampFolderAdapter,
    "csv": CsvBundleAdapter,
    "csv_bundle": CsvBundleAdapter,
    "sqlite": SqlitePassthroughAdapter,
}


def get_adapter(library_source: str, source_id: str | None = None) -> LibraryAdapter:
    key = library_source.lower().strip()
    cls = _ADAPTERS.get(key)
    if cls is None:
        raise ValueError(f"Unknown library_source {library_source!r}; choose from {sorted(_ADAPTERS)}")
    return cls(source_id=source_id or f"user_{key}")


def ingest_library(
    *,
    library_path: Path,
    library_source: str,
    out_db: Path,
    source_id: str | None = None,
    **kwargs: Any,
) -> tuple[IngestStats, Path]:
    adapter = get_adapter(library_source, source_id=source_id)
    return adapter.ingest(out_db=out_db, library_path=library_path, library_source=library_source, **kwargs)
