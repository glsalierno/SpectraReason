"""Provenance and licensing fields for externally ingested spectra."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

PREPROCESSING_VERSION = "ftir_foundation_movmin_sg_v1"

PROVENANCE_KEYS = (
    "source_id",
    "source_name",
    "source_license",
    "source_url",
    "original_identifier",
    "ingestion_date",
    "preprocessing_version",
    "redistribution_allowed",
    "dataset_tier",
)


def today_iso() -> str:
    return date.today().isoformat()


def make_reference_id(source_id: str, original_identifier: str) -> str:
    key = f"{source_id}|{original_identifier}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def provenance_summary(md: dict[str, Any]) -> dict[str, Any]:
    """Subset of metadata for debug reports / exports."""
    keys = (
        "source_id",
        "source_name",
        "source_license",
        "source_url",
        "original_identifier",
        "ingestion_date",
        "preprocessing_version",
        "dataset_tier",
        "dataset_tags",
        "acquisition_mode",
        "redistribution_allowed",
    )
    return {k: md[k] for k in keys if md.get(k) is not None}


def attach_provenance(
    md: dict[str, Any],
    *,
    source_id: str,
    source_name: str,
    source_license: str,
    original_identifier: str,
    source_url: str | None = None,
    redistribution_allowed: bool | None = None,
    dataset_tier: str = "experimental",
    ingestion_date: str | None = None,
) -> dict[str, Any]:
    """Merge provenance fields into spectrum metadata (NIST-compatible keys)."""
    out = dict(md)
    out["source_id"] = source_id
    out["source_name"] = source_name
    out["source_license"] = source_license
    out["source"] = source_name
    if source_url:
        out["source_url"] = source_url
    out["original_identifier"] = original_identifier
    out["ingestion_date"] = ingestion_date or today_iso()
    out["preprocessing_version"] = PREPROCESSING_VERSION
    if redistribution_allowed is not None:
        out["redistribution_allowed"] = bool(redistribution_allowed)
    out["dataset_tier"] = dataset_tier
    return out
