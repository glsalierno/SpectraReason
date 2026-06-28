"""
Import open polymer / ATR FTIR datasets from local downloads (Zenodo, university repos).

Registered open targets (see ``source_registry.json``):
- zenodo_polymer_atr_ftir (CC BY — verify record before redistribution)
- user_polymer_jcamp (local licensed/user-owned libraries)

No automatic download — user places files under ``data/external_sources/raw/open_polymer/``.
"""

from __future__ import annotations

from pathlib import Path

from ml.external.import_jcamp_folder import ingest_jcamp_folder
from ml.external.import_csv_bundle import ingest_csv_bundle
from ml.external.ingest_common import IngestStats, registry_entry

DEFAULT_RAW = Path("data/external_sources/raw/open_polymer")
DEFAULT_OUT = Path("data/experimental/open_polymer_ir_index.sqlite")


def ingest_open_polymer_ftir(
    raw_dir: Path | None = None,
    out_db: Path | None = None,
    *,
    source_id: str = "open_polymer_atr",
) -> tuple[IngestStats, Path]:
    reg = registry_entry(source_id) or {}
    raw = (raw_dir or DEFAULT_RAW).resolve()
    out = (out_db or DEFAULT_OUT).resolve()
    if not raw.is_dir():
        raise FileNotFoundError(
            f"Open polymer raw directory not found: {raw}\n"
            "Place Zenodo/university ATR polymer JCAMP or CSV bundles here."
        )

    jdx_stats, _ = ingest_jcamp_folder(
        raw,
        out,
        source_id=source_id,
        source_name=str(reg.get("source_name") or "Open polymer FTIR"),
        source_license=str(reg.get("license") or "see source record"),
        source_url=reg.get("url"),
        redistribution_allowed=reg.get("redistribution_allowed"),
        recursive=True,
        tags=["polymer", "atr"],
    )

    csv_dir = raw / "csv"
    if csv_dir.is_dir():
        csv_stats, _ = ingest_csv_bundle(
            csv_dir,
            out,
            source_id=f"{source_id}_csv",
            source_name=str(reg.get("source_name") or "Open polymer FTIR CSV"),
            source_license=str(reg.get("license") or "see source record"),
        )
        jdx_stats.attempted += csv_stats.attempted
        jdx_stats.ingested += csv_stats.ingested
        jdx_stats.rejected += csv_stats.rejected
        for k, v in csv_stats.failures.items():
            jdx_stats.failures[k] = jdx_stats.failures.get(k, 0) + v

    return jdx_stats, out
