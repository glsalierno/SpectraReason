"""SDBS first-batch layout: subfolders, manifest, and metadata enrichment."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

# Subfolder under raw/sdbs/ → dataset tags + target batch class hint
SDBS_SUBFOLDERS: dict[str, dict[str, Any]] = {
    "nitro_positive": {
        "tags": ["nitro", "batch_nitro_positive"],
        "target_class_hint": "nitro_positive",
    },
    "n_oxide_hard_negative": {
        "tags": ["n_oxide", "heteroaromatic", "nitroso", "batch_nitro_hn"],
        "target_class_hint": "nitro_hn",
    },
    "amide_positive": {
        "tags": ["amide", "batch_amide_positive"],
        "target_class_hint": "amide_positive",
    },
    "amide_hard_negative": {
        "tags": ["amide", "enamine", "heteroaromatic", "batch_amide_hn"],
        "target_class_hint": "amide_hn",
    },
    "siloxane_confounds": {
        "tags": ["siloxane", "polymer", "ether", "batch_siloxane_confound"],
        "target_class_hint": "siloxane_confound",
    },
}

DEFAULT_MANIFEST = Path("data/external_sources/raw/sdbs/sdbs_download_manifest.csv")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return str(v[0]).strip() if v else ""
    return str(v).strip()


def load_sdbs_manifest(path: Path | None = None) -> dict[str, dict[str, str]]:
    """
    Load manifest keyed by normalized compound_name and local_filename stem.

    CSV columns: batch_folder, compound_name, sdbs_id, sdbs_url, local_filename,
    cas, notes, download_date, downloaded
    """
    path = path or DEFAULT_MANIFEST
    if not path.is_file():
        return {}
    by_key: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = {k: _cell(v) for k, v in row.items()}
            for key in (
                _norm(rec.get("compound_name", "")),
                _norm(rec.get("local_filename", "")),
                _norm(Path(rec.get("local_filename", "")).stem),
            ):
                if key:
                    by_key[key] = rec
    return by_key


def lookup_manifest_row(
    path: Path,
    manifest: dict[str, dict[str, str]],
) -> dict[str, str]:
    keys = [_norm(path.stem), _norm(path.name)]
    for k in keys:
        if k in manifest:
            return manifest[k]
    return {}


def batch_meta_for_path(path: Path, raw_root: Path) -> dict[str, Any]:
    """Infer batch folder tags from path relative to raw/sdbs/."""
    try:
        rel = path.resolve().relative_to(raw_root.resolve())
        parts = rel.parts
    except ValueError:
        parts = ()
    extra: dict[str, Any] = {}
    if parts:
        folder = parts[0].lower()
        spec = SDBS_SUBFOLDERS.get(folder)
        if spec:
            extra["batch_folder"] = folder
            extra["target_class_hint"] = spec["target_class_hint"]
            extra["batch_tags"] = list(spec["tags"])
    return extra


def enrich_sdbs_metadata(
    md: dict[str, Any],
    path: Path,
    raw_root: Path,
    manifest: dict[str, dict[str, str]],
) -> dict[str, Any]:
    out = dict(md)
    out.update(batch_meta_for_path(path, raw_root))
    row = lookup_manifest_row(path, manifest)
    if row:
        out["batch_folder"] = row.get("batch_folder") or out.get("batch_folder")
        if row.get("compound_name"):
            out["batch_compound"] = row["compound_name"]
            out.setdefault("title", row["compound_name"])
            out.setdefault("name", row["compound_name"])
        if row.get("sdbs_id"):
            out["sdbs_id"] = row["sdbs_id"]
            out["original_identifier"] = f"sdbs:{row['sdbs_id']}"
        elif row.get("compound_name"):
            out.setdefault("original_identifier", _norm(row["compound_name"]))
        if row.get("sdbs_url"):
            out["sdbs_url"] = row["sdbs_url"]
        if row.get("cas"):
            out["cas"] = row["cas"]
        if row.get("notes"):
            out["batch_notes"] = row["notes"]
        if row.get("download_date"):
            out["sdbs_access_date"] = row["download_date"]
    if out.get("batch_tags"):
        existing = list(out.get("dataset_tags") or [])
        merged = existing + [t for t in out["batch_tags"] if t not in existing]
        out["dataset_tags"] = merged
    return out
