"""
Report-level suppression of nitro / NO₂ chemistry (samples known to lack R–NO₂).
"""

from __future__ import annotations

import re
from typing import Any

NITRO_DISABLED_LABELS = frozenset(
    {
        "nitro",
        "nitro_family",
        "NO2_asym_region",
        "NO2_sym_region",
        "N_O_NO2_overlap",
        "n_oxide_confounded_region",
        "heterocyclic_N_O_region",
        "heterocyclic_N_oxide",
        "pyrrole_N_oxide_like",
    }
)

NITRO_DISABLED_BAND_IDS = frozenset({"nitro_asym", "nitro_sym"})

_NITRO_TEXT = re.compile(r"\bnitro\b|no[\u2082₂2]|nitro_asym|nitro_sym", re.I)

_UNSAT_MID_HOVER_NO_NITRO = (
    "Aromatic C=C, amide II, enamine C=C–N, heterocyclic N–O / N-oxide-like modes"
)
_UNSAT_MID_RULER_LABEL_LINES = ("C=C / amide II", "N–O")


def ruler_hover_note(spec: Any, *, suppress_nitro: bool) -> str:
    if suppress_nitro and getattr(spec, "id", "") == "unsat_mid":
        return _UNSAT_MID_HOVER_NO_NITRO
    return str(getattr(spec, "hover_note", "") or getattr(spec, "short_label", ""))


def ruler_label_lines(region_id: str, short_label: str, *, suppress_nitro: bool) -> list[str] | None:
    if suppress_nitro and region_id == "unsat_mid":
        return list(_UNSAT_MID_RULER_LABEL_LINES)
    return None


def nitro_reporting_suppressed(pipeline_or_config: dict[str, Any] | None) -> bool:
    if not pipeline_or_config:
        return False
    if pipeline_or_config.get("suppress_nitro_reporting"):
        return True
    rs = pipeline_or_config.get("report_suppression") or {}
    return bool(rs.get("nitro"))


def mentions_nitro_chemistry(text: str) -> bool:
    return bool(_NITRO_TEXT.search(str(text or "")))


def nitro_suppression_rules_patch() -> dict[str, Any]:
    return {
        "suppress_nitro_reporting": True,
        "disabled_labels": sorted(NITRO_DISABLED_LABELS),
        "disabled_band_ids": sorted(NITRO_DISABLED_BAND_IDS),
        "report_suppression": {"nitro": True},
        "description": "Nitro / NO₂ reporting suppressed (sample context: no nitro compounds).",
    }


def filter_nitro_band_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in matches:
        bid = str(m.get("band_id") or "")
        if bid in NITRO_DISABLED_BAND_IDS:
            continue
        if mentions_nitro_chemistry(str(m.get("label") or "")):
            continue
        out.append(m)
    return out


def _scrub_assignment_row(ent: dict[str, Any]) -> None:
    ent["caution_flags"] = [
        c for c in (ent.get("caution_flags") or []) if not mentions_nitro_chemistry(str(c))
    ]
    ent["evidence"] = [e for e in (ent.get("evidence") or []) if not mentions_nitro_chemistry(str(e))]
    if mentions_nitro_chemistry(str(ent.get("human_readable_summary") or "")):
        ent["human_readable_summary"] = ""
    notes = ent.get("notes")
    if isinstance(notes, list):
        ent["notes"] = [n for n in notes if not mentions_nitro_chemistry(str(n))]


def apply_nitro_suppression(pipeline: dict[str, Any]) -> None:
    """Remove nitro-related labels, band matches, and prose from a finished pipeline."""
    evidence = pipeline.setdefault("evidence", {})
    evidence["band_matches"] = filter_nitro_band_matches(list(evidence.get("band_matches") or []))

    ra = pipeline.setdefault("rule_assignments", {})
    assigns = ra.setdefault("assignments", {})
    for lab in list(assigns.keys()):
        if lab in NITRO_DISABLED_LABELS:
            del assigns[lab]
        else:
            ent = assigns.get(lab)
            if isinstance(ent, dict):
                _scrub_assignment_row(ent)

    ra["ambiguity_labels"] = [
        a
        for a in (ra.get("ambiguity_labels") or [])
        if isinstance(a, dict) and not mentions_nitro_chemistry(str(a.get("title") or ""))
    ]

    cons = pipeline.setdefault("consensus", {})
    per = cons.setdefault("per_label", {})
    for lab in list(per.keys()):
        if lab in NITRO_DISABLED_LABELS:
            del per[lab]
    cons["top_labels"] = [
        (lab, ent)
        for lab, ent in (cons.get("top_labels") or [])
        if lab not in NITRO_DISABLED_LABELS
    ]

    ml_ref = pipeline.setdefault("ml_refinement", {})
    for key in ("basic", "subtle", "legacy"):
        block = ml_ref.get(key)
        if not isinstance(block, dict):
            continue
        pl = block.get("per_label") or {}
        for lab in list(pl.keys()):
            if lab in NITRO_DISABLED_LABELS:
                del pl[lab]

    pipeline["report_suppression"] = {"nitro": True}
    pipeline["suppress_nitro_reporting"] = True
