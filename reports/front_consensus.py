"""
Front-facing consensus presentation: professional labels, ambiguity-aware filtering.

Debug/audit reports keep raw ontology keys and confidence classes.
"""

from __future__ import annotations

import re
from typing import Any

from ml.report_suppression import nitro_reporting_suppressed
from reports.kronecker_pi_layout import _esc, _truncate
from reports.product_v1_report import chemistry_label

MARKER_CONSENSUS_TABLE = "<!-- report-feature:consensus-interpretation-table -->"

# Standalone local motifs hidden from front consensus / key evidence (unless nitro supported).
_FRONT_HIDE_LOCAL_STANDALONE = frozenset(
    {
        "NO2_asym_region",
        "NO2_sym_region",
        "C_O_fingerprint_region",
        "carbonyl_region",
        "broad_OH_NH_region",
        "nitrile_alkyne_region",
        "CH_stretch_region",
        "aliphatic_CH_region",
        "aromatic_CH_region",
        "upper_mid_activity_region",
        "nh_ch_transition_region",
        "fingerprint_crowding_region",
        "Si_O_overlap_region",
        "amide_II_region",
        "enamine_region",
        "heterocyclic_N_O_region",
        "n_oxide_confounded_region",
        "aromatic_CC_region",
    }
)

_FRONT_AMBIGUITY_MOTIFS = frozenset(
    {
        "N_O_NO2_overlap",
        "n_oxide_confounded_region",
        "heterocyclic_N_O_region",
        "heterocyclic_N_oxide",
        "pyrrole_N_oxide_like",
        "amide_II_region",
        "enamine_region",
        "Si_O_overlap_region",
        "fingerprint_C_O_or_Si_O_overlap",
    }
)

_FRONT_DISPLAY_NAMES: dict[str, str] = {
    "nitro": "Nitro",
    "NO2_asym_region": "N–O / NO₂ overlap region",
    "NO2_sym_region": "N–O / NO₂ overlap region",
    "N_O_NO2_overlap": "N–O / NO₂ overlap",
    "heterocyclic_N_O_region": "Heterocyclic N–O / N-oxide-like",
    "heterocyclic_N_oxide": "Heterocyclic N–O / N-oxide-like",
    "pyrrole_N_oxide_like": "Pyrrole N-oxide-like",
    "n_oxide_confounded_region": "N–O / NO₂ overlap",
    "amide": "Amide",
    "amide_II_region": "Amide II / enamine / pyrrole-like overlap",
    "secondary_amine": "Secondary amine",
    "pyrrole_like_NH": "Pyrrole-like N–H",
    "enamine_region": "Enamine / C=C–N overlap",
    "C_O_fingerprint_region": "C–O / fingerprint activity",
    "fingerprint_C_O_or_Si_O_overlap": "C–O / Si–O overlap",
    "aromatic_CC_region": "Aromatic / heteroaromatic ring modes",
    "heteroaromatic": "Heteroaromatic",
    "aromatic": "Aromatic",
    "ether": "Ether",
    "aryl_ether": "Aryl ether",
    "siloxane": "Siloxane (ATR-sensitive)",
    "phenol": "Phenol",
    "alcohol": "Alcohol",
}


def front_display_name(lab: str, pipeline: dict[str, Any], ent: dict[str, Any] | None = None) -> str:
    lb = str(lab)
    if lb in _FRONT_DISPLAY_NAMES:
        return _FRONT_DISPLAY_NAMES[lb]
    if lb.endswith("_region"):
        base = lb.replace("_region", "").replace("_", " ")
        return f"{base.replace('c o', 'C–O')} region activity"
    return chemistry_label(lb, pipeline, ent)


def nitro_is_supported(pipeline: dict[str, Any]) -> bool:
    ent = ((pipeline.get("rule_assignments") or {}).get("assignments") or {}).get("nitro") or {}
    if not isinstance(ent, dict):
        return False
    sc = float(ent.get("score", 0) or 0)
    cc = str(ent.get("confidence_class") or "").lower()
    ec = str(ent.get("evidence_completeness") or "").lower()
    if sc < 0.38:
        return False
    if cc in ("local_possible", "local_motif_only", "overlap_limited", "not_supported"):
        return False
    if ec in ("single_band", "partial"):
        return False
    return cc in ("strong", "supported") and ec == "complete"


def professional_confidence_label(ent: dict[str, Any], pipeline: dict[str, Any], lab: str) -> str:
    """Human-readable confidence for front-facing tables (not raw ontology)."""
    cc = str(ent.get("confidence_class") or "").lower()
    ec = str(ent.get("evidence_completeness") or "").lower()
    oc = str(ent.get("ontology_category") or "").lower()
    sc = float(ent.get("score", 0) or 0)
    if lab == "nitro" and not nitro_is_supported(pipeline):
        return "Insufficient evidence"
    if oc == "local_motif" or cc in ("local_possible", "local_motif_only"):
        return "Local overlap only"
    if cc == "overlap_limited" or ec == "single_band":
        return "Supported with overlap"
    if cc == "strong" and ec == "complete":
        return "Strong match"
    if cc == "supported":
        return "Supported" if ec == "complete" else "Supported with overlap"
    if cc == "tentative":
        return "Tentative subclass"
    if sc < 0.2:
        return "Insufficient evidence"
    return "Supported with overlap"


def should_show_front_consensus_row(lab: str, ent: dict[str, Any], pipeline: dict[str, Any]) -> bool:
    """Filter rows for front consensus / key evidence."""
    if not isinstance(ent, dict):
        return False
    lb = str(lab)
    oc = str(ent.get("ontology_category") or "").lower()
    cc = str(ent.get("confidence_class") or "").lower()
    sc = float(ent.get("score", 0) or 0)
    if sc < 0.18 and lb not in _FRONT_AMBIGUITY_MOTIFS:
        return False
    if cc == "not_supported" and sc < 0.25:
        return False
    if lb in ("NO2_asym_region", "NO2_sym_region"):
        return False
    if lb == "nitro":
        return nitro_is_supported(pipeline)
    if lb == "amide" and cc in ("overlap_limited", "local_possible", "tentative"):
        return False
    if lb in _FRONT_HIDE_LOCAL_STANDALONE:
        return False
    if oc == "local_motif" and lb not in _FRONT_AMBIGUITY_MOTIFS:
        return False
    if oc == "artifact":
        return False
    if lb.endswith("_region") and oc == "local_motif":
        return False
    return True


def collect_front_ambiguities(pipeline: dict[str, Any], *, max_items: int = 5) -> list[str]:
    """Concise ambiguity / confounder lines for consensus table."""
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    items: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = text.strip()
        if t and t not in seen:
            seen.add(t)
            items.append(t)

    if not nitro_is_supported(pipeline):
        asym = float((assigns.get("NO2_asym_region") or {}).get("score", 0) or 0)
        sym = float((assigns.get("NO2_sym_region") or {}).get("score", 0) or 0)
        if asym >= 0.15 or sym >= 0.12:
            _add("N–O / NO₂ overlap (paired nitro bands not confirmed)")

    mid = max(
        float((assigns.get("heterocyclic_N_O_region") or {}).get("score", 0) or 0),
        float((assigns.get("heterocyclic_N_oxide") or {}).get("score", 0) or 0),
        float((assigns.get("pyrrole_N_oxide_like") or {}).get("score", 0) or 0),
        float((assigns.get("N_O_NO2_overlap") or {}).get("score", 0) or 0),
    )
    if mid >= 0.15:
        _add("Heterocyclic N–O / N-oxide-like overlap (1250–1650 cm⁻¹)")

    amide_sc = float((assigns.get("amide") or {}).get("score", 0) or 0)
    amide_ii = float((assigns.get("amide_II_region") or {}).get("score", 0) or 0)
    if amide_ii >= 0.15 and (amide_sc < 0.35 or str((assigns.get("amide") or {}).get("confidence_class")) in (
        "tentative",
        "local_possible",
        "overlap_limited",
    )):
        _add("Amide II / enamine / pyrrole-like overlap")

    enamine = float((assigns.get("enamine_region") or {}).get("score", 0) or 0)
    if enamine >= 0.15:
        _add("Enamine / C=C–N modes in mid-IR")

    sio = float((assigns.get("Si_O_overlap_region") or {}).get("score", 0) or 0)
    co = float((assigns.get("ether") or {}).get("score", 0) or 0)
    if sio >= 0.15 or (co >= 0.2 and float((assigns.get("siloxane") or {}).get("score", 0) or 0) >= 0.15):
        _add("C–O / Si–O fingerprint overlap")

    for amb in (pipeline.get("rule_assignments") or {}).get("ambiguity_labels") or []:
        if isinstance(amb, dict):
            title = str(amb.get("title") or "")
            if title and "NO₂" not in title.upper() and "nitro" not in title.lower():
                _add(title[:90])
        if len(items) >= max_items:
            break

    return items[:max_items]


def build_consensus_interpretation_text(pipeline: dict[str, Any], *, ml_enabled: bool) -> str:
    """Prose consensus for spectrum card and batch table."""
    from reports.front_facing_report import spectrum_quality_limited

    limited, lim_msg = spectrum_quality_limited(pipeline)
    if limited:
        return (
            f"This spectrum has limited reliable assignment. {lim_msg} "
            "Review the plot and technical details for band-level context."
        )

    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    from reports.v4_evidence_report import evidence_ranked_assignments

    ranked = [
        (lab, ent)
        for lab, ent in evidence_ranked_assignments(pipeline, top_n=14)
        if should_show_front_consensus_row(lab, ent, pipeline)
    ]

    lead_parts: list[str] = []
    for lab, ent in ranked[:4]:
        disp = front_display_name(lab, pipeline, ent)
        conf = professional_confidence_label(ent, pipeline, lab)
        if conf in ("Strong match", "Supported", "Supported with overlap"):
            lead_parts.append(disp.lower() if conf != "Tentative subclass" else f"tentative {disp.lower()}")
    if not lead_parts:
        lead_parts.append("hydroxy- and C–O-containing aromatic/heteroaromatic chemistry")

    parts = [
        f"This spectrum is most consistent with {', '.join(lead_parts[:3])}.",
    ]

    phenol = float((assigns.get("phenol") or {}).get("score", 0) or 0)
    alcohol = float((assigns.get("alcohol") or {}).get("score", 0) or 0)
    if phenol >= 0.3 or alcohol >= 0.3:
        parts.append("Phenol/alcohol subclassing remains tentative without stronger paired diagnostic bands.")

    if not nitro_reporting_suppressed(pipeline) and not nitro_is_supported(pipeline):
        asym = float((assigns.get("NO2_asym_region") or {}).get("score", 0) or 0)
        sym = float((assigns.get("NO2_sym_region") or {}).get("score", 0) or 0)
        if asym >= 0.12 or sym >= 0.1:
            parts.append(
                "Bands near 1500 and 1320 cm⁻¹ are better described as N–O/NO₂ overlap "
                "unless paired nitro evidence is unambiguous."
            )

    amide_ent = assigns.get("amide") or {}
    if float(amide_ent.get("score", 0) or 0) >= 0.2 and str(amide_ent.get("confidence_class")) in (
        "tentative",
        "local_possible",
        "overlap_limited",
    ):
        parts.append("Amide calls require paired carbonyl and N–H/amide II context.")

    text = " ".join(parts[:4])
    return text[:480] + ("…" if len(text) > 480 else "")


def collect_supported_evidence_phrases(pipeline: dict[str, Any], *, max_phrases: int = 3) -> str:
    from reports.front_facing_report import format_key_spectral_evidence
    from reports.v4_evidence_report import evidence_ranked_assignments

    phrases: list[str] = []
    for lab, ent in evidence_ranked_assignments(pipeline, top_n=16):
        if not should_show_front_consensus_row(lab, ent, pipeline):
            continue
        phrases.append(format_key_spectral_evidence(ent, pipeline, lab))
        if len(phrases) >= max_phrases:
            break
    return "; ".join(phrases) if phrases else "Diagnostic region activity (see plot)"


def build_front_consensus_table_html(*, rows: list[dict[str, Any]], ml_enabled: bool) -> str:
    """Top-of-report consensus table (replaces Summary)."""
    parts = [
        MARKER_CONSENSUS_TABLE,
        "<section id='summary-table' class='summary-table-section product-summary card front-summary consensus-section'>",
        "<h2 class='summary-table-heading'>Consensus interpretation</h2>",
        "<p class='muted small'>Evidence-first consensus; ML is advisory when enabled.</p>",
    ]
    if not any(nitro_reporting_suppressed(r.get("_pipeline") or {}) for r in rows):
        parts.append(
            "<p class='muted small'>"
            "Local overlap motifs (NO₂ regions, fingerprint windows) appear under ambiguities or technical details, "
            "not as standalone supported chemistry."
            "</p>"
        )
    parts.extend([
        "<div class='table-scroll summary-table-wrap'>"
        "<table class='tbl tbl-zebra tbl-sticky tbl-summary tbl-front-consensus'><thead><tr>"
        "<th>Spectrum</th>",
        "<th>Consensus interpretation</th>",
        "<th>Supported evidence</th>",
        "<th>Ambiguities / confounders</th>",
        "<th>Confidence</th>",
        "<th></th>",
        "</tr></thead><tbody>",
    ])
    for r in rows:
        pipeline = r.get("_pipeline") or {}
        consensus = build_consensus_interpretation_text(pipeline, ml_enabled=ml_enabled)
        ev = collect_supported_evidence_phrases(pipeline)
        amb = "; ".join(collect_front_ambiguities(pipeline))
        from reports.v4_evidence_report import evidence_ranked_assignments

        conf = "Supported with overlap"
        for lab, ent in evidence_ranked_assignments(pipeline, top_n=8):
            if should_show_front_consensus_row(lab, ent, pipeline):
                c = professional_confidence_label(ent, pipeline, lab)
                if c == "Strong match":
                    conf = c
                    break
                if c in ("Supported", "Supported with overlap"):
                    conf = c
        parts.append(
            f"<tr><td class='td-spectrum'>{_esc(r['name'])}</td>"
            f"<td>{_esc(_truncate(consensus, 200))}</td>"
            f"<td>{_esc(_truncate(ev, 120))}</td>"
            f"<td>{_esc(_truncate(amb or '—', 140))}</td>"
            f"<td>{_esc(conf)}</td>"
            f"<td><a href='#{_esc(r['anchor'])}'>View</a></td></tr>"
        )
    parts.append("</tbody></table></div></section>")
    return "".join(parts)


def build_front_ambiguity_cards_html(pipeline: dict[str, Any], *, anchor: str = "") -> str:
    items = collect_front_ambiguities(pipeline, max_items=4)
    if not items:
        return ""
    lis = "".join(f"<li>{_esc(x)}</li>" for x in items)
    eid = f"{_esc(anchor)}-ambiguity" if anchor else "ambiguity-cards"
    return (
        f"<section class='ambiguity-cards' id='{eid}'>"
        "<h3>Ambiguities &amp; confounders</h3>"
        f"<ul class='caution-front'>{lis}</ul></section>"
    )
