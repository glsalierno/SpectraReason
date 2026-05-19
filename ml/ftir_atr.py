"""
ATR measurement metadata: inference from paths/metadata and interpretation flags.

No aggressive spectral correction — used for guardrails and reporting only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

MeasurementMode = Literal["ATR", "transmission", "unknown"]
AtrCrystal = Literal["diamond", "ZnSe", "Ge", "Si", "unknown"]

_ATR_PATH_PATTERNS = (
    r"\batr\b",
    r"[_\-\.]atr[_\-\.]|[_\-\.]atr$|^atr[_\-\.]",
    r"attenuated\s+total\s+reflection",
    r"\bdiamond\b",
    r"\bznse\b",
    r"zinc\s+selenide",
    r"\bge\s+crystal\b",
    r"germanium\s+crystal",
)
_CRYSTAL_PATTERNS: tuple[tuple[str, AtrCrystal], ...] = (
    (r"\bdiamond\b", "diamond"),
    (r"\bznse\b|zinc\s+selenide", "ZnSe"),
    (r"\bge\b(?!\w)|germanium", "Ge"),
    (r"\bsi\b(?!\w)|silicon\s+crystal", "Si"),
)


def _text_blob(path: Path | None, md: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if path is not None:
        parts.extend([path.name, path.stem, str(path.parent)])
    if md:
        for k in ("title", "name", "technique", "measurement_mode", "mode", "sample_type", "comment", "notes"):
            v = md.get(k)
            if v:
                parts.append(str(v))
    return " ".join(parts).lower()


def infer_measurement_mode(
    path: Path | str | None = None,
    md: dict[str, Any] | None = None,
) -> MeasurementMode:
    """Infer ATR vs transmission from explicit metadata or path/title heuristics."""
    md = md or {}
    explicit = str(
        md.get("measurement_mode") or md.get("mode") or md.get("technique") or ""
    ).strip().lower()
    if explicit in ("atr", "attenuated total reflection"):
        return "ATR"
    if explicit in ("transmission", "trans", "kbr", "nujol"):
        return "transmission"
    if md.get("is_atr") is True or md.get("atr") is True:
        return "ATR"
    if md.get("is_atr") is False:
        return "transmission"

    blob = _text_blob(Path(path) if path else None, md)
    if "transmission" in blob or re.search(r"\bkbr\b", blob):
        return "transmission"
    for pat in _ATR_PATH_PATTERNS:
        if re.search(pat, blob, re.I):
            return "ATR"
    if "powder" in blob and "atr" not in blob:
        return "unknown"
    return "unknown"


def infer_atr_crystal(
    path: Path | str | None = None,
    md: dict[str, Any] | None = None,
) -> AtrCrystal:
    md = md or {}
    explicit = str(md.get("atr_crystal") or md.get("crystal") or "").strip().lower()
    crystal_map: dict[str, AtrCrystal] = {
        "diamond": "diamond",
        "znse": "ZnSe",
        "zinc selenide": "ZnSe",
        "ge": "Ge",
        "germanium": "Ge",
        "si": "Si",
        "silicon": "Si",
    }
    if explicit in crystal_map:
        return crystal_map[explicit]
    blob = _text_blob(Path(path) if path else None, md)
    for pat, crystal in _CRYSTAL_PATTERNS:
        if re.search(pat, blob, re.I):
            return crystal
    return "unknown"


def resolve_atr_context(
    *,
    path: Path | str | None = None,
    md: dict[str, Any] | None = None,
    mode: str | None = None,
    atr_crystal: str | None = None,
    atr_aware: bool | None = None,
) -> dict[str, Any]:
    """
    Build measurement context for pipeline / guardrails.

    ``atr_aware`` defaults True when mode is ATR or inferred ATR.
    """
    md = dict(md or {})
    if mode and str(mode).lower() not in ("", "unknown"):
        m = str(mode).strip().lower()
        meas: MeasurementMode = "ATR" if m in ("atr", "attenuated total reflection") else (
            "transmission" if m in ("transmission", "trans") else "unknown"
        )
        mode_inferred = False
    else:
        meas = infer_measurement_mode(path, md)
        mode_inferred = meas == "ATR" and not md.get("measurement_mode")

    if atr_crystal and str(atr_crystal).lower() != "unknown":
        crystal = infer_atr_crystal(path, {**md, "atr_crystal": atr_crystal})
        crystal_inferred = False
    else:
        crystal = infer_atr_crystal(path, md)
        crystal_inferred = crystal != "unknown" and not md.get("atr_crystal")

    is_atr = meas == "ATR"
    aware = bool(atr_aware) if atr_aware is not None else is_atr

    return {
        "mode": meas,
        "atr_crystal": crystal,
        "atr_aware": aware,
        "is_atr": is_atr,
        "mode_inferred": mode_inferred,
        "crystal_inferred": crystal_inferred,
    }


def atr_sensitive_interpretation(evidence: dict[str, Any]) -> bool:
    """True when ATR-aware guardrails should apply to Si-O / siloxane."""
    meas = evidence.get("measurement") or {}
    if not isinstance(meas, dict):
        meas = {}
    flags = (evidence.get("artifacts") or {}).get("flags") or {}
    if flags.get("atr_crystal_fingerprint_overlap"):
        return True
    if not meas.get("atr_aware", meas.get("is_atr")):
        return False
    return str(meas.get("mode") or "").upper() == "ATR"


def merge_measurement_into_metadata(md: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    out = dict(md)
    out["measurement_mode"] = measurement.get("mode")
    out["atr_crystal"] = measurement.get("atr_crystal")
    out["atr_aware"] = measurement.get("atr_aware")
    out["is_atr"] = measurement.get("is_atr")
    return out
