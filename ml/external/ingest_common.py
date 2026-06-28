"""Shared ingestion: unit normalization, preprocessing, validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lib.ftir_foundation import (
    IntensityMode,
    preprocess_spectrum,
    read_jdx_spectrum,
    read_spectrum,
)
from ml.external.provenance import PREPROCESSING_VERSION, attach_provenance, make_reference_id
from ml.structural_fg_svm import prepare_nist_ftir_cm1

AcquisitionMode = Literal["ATR", "transmission", "gas_phase"] | None


@dataclass
class IngestResult:
    reference_id: str
    wn_cm1: np.ndarray
    y_processed: np.ndarray
    metadata: dict[str, Any]
    source_path: str
    tags: list[str] = field(default_factory=list)
    qa_flags: list[str] = field(default_factory=list)


@dataclass
class IngestStats:
    attempted: int = 0
    ingested: int = 0
    rejected: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def bump_failure(self, reason: str) -> None:
        self.rejected += 1
        self.failures[reason] = self.failures.get(reason, 0) + 1


def parse_jcamp_metadata_lines(lines: list[str]) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for ln in lines:
        s = ln.strip()
        if not s.startswith("##") or "=" not in s:
            continue
        key, val = s[2:].split("=", 1)
        raw[key.strip().upper()] = val.strip()
    md = {
        "title": raw.get("TITLE"),
        "name": raw.get("TITLE"),
        "formula": raw.get("MOLFORM"),
        "cas": raw.get("CAS REGISTRY NO") or raw.get("CAS"),
        "inchi": raw.get("INCHI"),
        "inchikey": raw.get("INCHIKEY"),
        "state": raw.get("STATE"),
        "phase": raw.get("PHASE"),
        "sampling_procedure": raw.get("SAMPLING PROCEDURE"),
        "xunits": raw.get("XUNITS"),
        "yunits": raw.get("YUNITS"),
        "spectrum_type": raw.get("SPECTRUM TYPE"),
    }
    md["acquisition_mode"] = infer_acquisition_mode(md)
    return md


def infer_acquisition_mode(md: dict[str, Any]) -> AcquisitionMode:
    tokens = " ".join(
        str(md.get(k) or "")
        for k in ("sampling_procedure", "state", "phase", "title", "name")
    ).lower()
    if "atr" in tokens or "attenuated total reflect" in tokens:
        return "ATR"
    if "transmission" in tokens or "kbr" in tokens or "pellet" in tokens or "mull" in tokens:
        return "transmission"
    if "gas" in tokens or "vapor" in tokens:
        return "gas_phase"
    return None


def intensity_hint_from_metadata(md: dict[str, Any], yunits: str | None = None) -> IntensityMode:
    yu = (yunits or str(md.get("yunits") or "")).upper()
    if "ABSORB" in yu:
        return "absorbance"
    if "TRANSMITT" in yu:
        return "transmittance_percent"
    return "auto"


def load_raw_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], IntensityMode]:
    suf = path.suffix.lower()
    if suf in (".jdx", ".dx", ".jcm"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        md = parse_jcamp_metadata_lines(lines)
        wn, y, yunits = read_jdx_spectrum(path)
        md["yunits"] = md.get("yunits") or yunits
        hint = intensity_hint_from_metadata(md, yunits)
        return wn, y, md, hint
    wn, y, hint = read_spectrum(path)
    md = {"title": path.stem, "name": path.stem, "xunits": "1/CM"}
    return wn, y, md, hint


def preprocess_for_index(
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
    *,
    intensity_mode: IntensityMode = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    wn_p, y_p, info = preprocess_spectrum(wn, y, intensity_mode=intensity_mode)
    md_out = dict(md)
    md_out["preprocess_intensity_mode"] = info.get("intensity_mode")
    md_out["preprocess_baseline"] = info.get("baseline")
    md_out["preprocessing_version"] = PREPROCESSING_VERSION
    prepared = prepare_nist_ftir_cm1(wn_p, y_p, md_out)
    if prepared is None:
        return None
    return prepared[0], prepared[1], md_out


def basic_spectrum_qa(wn: np.ndarray, y: np.ndarray) -> list[str]:
    flags: list[str] = []
    if wn.size < 32:
        flags.append("too_few_points")
    span = float(np.nanmax(wn) - np.nanmin(wn)) if wn.size else 0.0
    if span < 200.0:
        flags.append("narrow_span")
    if not np.all(np.isfinite(wn)) or not np.all(np.isfinite(y)):
        flags.append("non_finite")
    yr = float(np.nanmax(y) - np.nanmin(y))
    if yr < 1e-6:
        flags.append("flat_spectrum")
    if float(np.nanmax(y)) > 1.05 or float(np.nanmin(y)) < -0.05:
        flags.append("absorbance_out_of_range")
    dy = np.diff(y)
    if dy.size and float(np.nanstd(dy)) < 1e-8 and yr < 0.01:
        flags.append("suspiciously_flat")
    return flags


def build_ingest_result(
    *,
    source_id: str,
    source_name: str,
    source_license: str,
    original_identifier: str,
    source_path: Path,
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
    tags: list[str] | None = None,
    source_url: str | None = None,
    redistribution_allowed: bool | None = None,
    reject_on_qa: bool = True,
    qa_blocklist: frozenset[str] = frozenset(
        {"too_few_points", "narrow_span", "non_finite", "flat_spectrum"}
    ),
) -> IngestResult | None:
    qa = basic_spectrum_qa(wn, y)
    if reject_on_qa and any(f in qa_blocklist for f in qa):
        return None
    rid = make_reference_id(source_id, original_identifier)
    md_full = attach_provenance(
        md,
        source_id=source_id,
        source_name=source_name,
        source_license=source_license,
        original_identifier=original_identifier,
        source_url=source_url,
        redistribution_allowed=redistribution_allowed,
    )
    if tags:
        md_full["dataset_tags"] = list(tags)
    return IngestResult(
        reference_id=rid,
        wn_cm1=wn,
        y_processed=y,
        metadata=md_full,
        source_path=str(source_path.resolve()),
        tags=tags or [],
        qa_flags=qa,
    )


def load_source_registry(registry_path: Path | None = None) -> list[dict[str, Any]]:
    if registry_path is None:
        registry_path = Path("data/external_sources/source_registry.json")
    if not registry_path.is_file():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return list(data.get("sources", []))


def registry_entry(source_id: str, registry_path: Path | None = None) -> dict[str, Any] | None:
    for row in load_source_registry(registry_path):
        if str(row.get("source_id")) == source_id:
            return row
    return None


def keyword_tags_from_metadata(md: dict[str, Any]) -> list[str]:
    blob = " ".join(
        str(md.get(k) or "")
        for k in ("title", "name", "formula", "state", "sampling_procedure")
    ).lower()
    tags: list[str] = []
    rules = [
        ("polymer", ("polymer", "nylon", "polyethylene", "polystyrene", "silicone", "pdms")),
        ("heteroaromatic", ("pyrid", "pyrrol", "imidaz", "indol", "furan", "thiophen", "oxazole")),
        ("nitro", ("nitro", "no2")),
        ("n_oxide", ("n-oxide", "n oxide", "n-oxy", "pyridine n-oxide")),
        ("nitroso", ("nitroso",)),
        ("phenol", ("phenol", "catechol", "hydroxy")),
        ("amide", ("amide", "imide", "lactam")),
        ("siloxane", ("siloxane", "silicone", "pdms", "polysilox", "si-o-si")),
        ("enamine", ("enamine",)),
        ("moisture_like", ("water", "moisture", "humid")),
    ]
    for tag, kws in rules:
        if any(k in blob for k in kws):
            tags.append(tag)
    mode = str(md.get("acquisition_mode") or "").upper()
    if mode == "ATR":
        tags.append("atr")
    elif mode == "transmission":
        tags.append("transmission")
    return tags


def merge_manual_tags(md: dict[str, Any], extra: list[str] | None) -> list[str]:
    base = list(md.get("dataset_tags") or [])
    base.extend(keyword_tags_from_metadata(md))
    if extra:
        base.extend(extra)
    seen: set[str] = set()
    out: list[str] = []
    for t in base:
        t = str(t).strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
