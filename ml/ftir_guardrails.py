"""
v3_guarded false-positive guardrails: paired-band logic, overlap caps, soft competitor suppression.

Does not hard-zero scores; adds confidence classes, completeness, cautions, and ambiguity fallbacks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Evidence group = OR over band_ids (any band in group with sufficient match counts as group satisfied).
EVIDENCE_GROUPS: dict[str, list[str]] = {
    "nitro_asym_g": ["nitro_asym"],
    "nitro_sym_g": ["nitro_sym"],
    "ester_co_g": ["ester_co"],
    "ester_coo_g": ["ester_co_o"],
    "amide_co_g": ["amide_co"],
    "amide_nh_g": ["amide_nh"],
    "amide_nh_or_ii_g": ["amide_nh", "amide_ii"],
    "acid_co_g": ["carboxylic_co"],
    "acid_oh_g": ["acid_oh_broad", "broad_oh"],
    "phenol_oh_g": ["phenol_oh", "broad_oh"],
    "aromatic_cc_g": ["aromatic_cc"],
    "phenolic_co_g": ["phenolic_co", "aryl_ether_co"],
    "siloxane_sio_g": ["siloxane_sio"],
    "silicone_sic_g": ["silicone_sic"],
    "nitrile_cn_g": ["nitrile_cn"],
    "alkyne_cc_g": ["alkyne_cc"],
    "ketone_co_g": ["ketone_co"],
    "aldehyde_co_g": ["aldehyde_co"],
    "ether_co_g": ["ether_co"],
    "aryl_ether_co_g": ["aryl_ether_co"],
    "alcohol_oh_g": ["alcohol_oh", "broad_oh"],
    "hetero_g": ["heteroaromatic"],
    "pyrrole_nh_g": ["pyrrole_nh"],
    "cyclic_nh_g": ["cyclic_amine_nh"],
    "carbonate_co_g": ["carbonate_co"],
    "urethane_co_g": ["urethane_co"],
    "amine_nh_g": ["amine_nh", "amine_nh2"],
}

# Per label: required group keys (AND), optional caps/suppression.
V3_GUARDRAILS: dict[str, dict[str, Any]] = {
    "nitro": {
        "required_groups": ["nitro_asym_g", "nitro_sym_g"],
        "missing_required_cap": 0.4,
        "single_band_cap": 0.28,
        "competitors": [
            "aromatic",
            "heteroaromatic",
            "pyrrole_like_NH",
            "heterocyclic_N_O_region",
            "enamine_region",
            "aromatic_CC_region",
            "amide_II_region",
        ],
        "competitor_suppression_factor": 0.45,
        "caution_messages": ["NO₂ assignment requires paired asymmetric + symmetric evidence."],
    },
    "ester": {
        "required_groups": ["ester_co_g", "ester_coo_g"],
        "missing_required_cap": 0.4,
        "single_band_cap": 0.3,
        "competitors": ["ketone", "aldehyde", "amide", "carboxylic_acid", "ether"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": ["Ester needs C=O and C–O ester fingerprint support together."],
    },
    "amide": {
        "required_groups": ["amide_co_g", "amide_nh_or_ii_g"],
        "missing_required_cap": 0.4,
        "single_band_cap": 0.3,
        "competitors": ["amine", "ketone", "ester", "carboxylic_acid", "primary_amine", "secondary_amine"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": ["Amide I without N–H or amide II context is often ambiguous vs ketone/ester."],
    },
    "carboxylic_acid": {
        "required_groups": ["acid_co_g", "acid_oh_g"],
        "missing_required_cap": 0.4,
        "single_band_cap": 0.3,
        "competitors": ["alcohol", "phenol", "ester"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": ["Carboxylic acid needs carbonyl plus broad acid O–H envelope."],
    },
    "phenol": {
        "required_groups": ["phenol_oh_g", "aromatic_cc_g", "phenolic_co_g"],
        "missing_required_cap": 0.38,
        "single_band_cap": 0.3,
        "competitors": ["alcohol", "aryl_ether", "ether"],
        "competitor_suppression_factor": 0.5,
        "artifact_confounders": ["water_vapor_or_moisture_like"],
        "caution_messages": ["Phenol needs aromatic + phenolic O–H / aryl C–O context, not O–H alone."],
    },
    "alcohol": {
        "required_groups": ["alcohol_oh_g", "ether_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.28,
        "competitors": ["phenol", "carboxylic_acid"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": ["Alcohol needs O–H plus C–O stretch context; O–H alone is hydroxy-family only."],
    },
    "primary_amine": {
        "required_groups": ["amine_nh_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.28,
        "competitors": ["amide", "pyrrole_like_NH"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "secondary_amine": {
        "required_groups": ["amine_nh_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.26,
        "competitors": ["amide", "pyrrole_like_NH"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "siloxane": {
        "required_groups": ["siloxane_sio_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.30,
        "competitors": ["phenol", "alcohol", "ether", "aryl_ether", "ester"],
        "competitor_suppression_factor": 0.35,
        "ratio_min": {"ratio": "siloxane_to_c_o", "min": 0.42, "otherwise_cap": 0.25},
        "caution_messages": [
            "Siloxane vs organic C-O requires Si-O dominance (ratio) plus a second Si-related region (e.g. Si-C); "
            "one 1000-1150 cm-1 band alone is never high-confidence siloxane."
        ],
    },
    "silicone_or_silane": {
        "required_groups": ["siloxane_sio_g"],
        "missing_required_cap": 0.4,
        "single_band_cap": 0.30,
        "competitors": ["ether", "aryl_ether", "ester", "phenol"],
        "competitor_suppression_factor": 0.35,
        "caution_messages": [
            "Organosilicon vs organic C-O: require siloxane fingerprint plus Si-C or second Si-O evidence; "
            "not from a single overlap band."
        ],
    },
    "ether": {
        "required_groups": ["ether_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.3,
        "competitors": ["ester", "aryl_ether", "siloxane"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "aryl_ether": {
        "required_groups": ["aryl_ether_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.3,
        "competitors": ["phenol", "ester", "ether", "siloxane"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "nitrile": {
        "required_groups": ["nitrile_cn_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.26,
        "competitors": ["alkyne"],
        "competitor_suppression_factor": 0.45,
        "min_peak_sharpness": 0.012,
        "artifact_confounders": ["co2_region_elevated", "weak_nitrile_region_spike"],
        "caution_messages": ["Nitrile C≡N should be sharp and isolated; check alkyne and artifacts."],
    },
    "alkyne": {
        "required_groups": ["alkyne_cc_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.28,
        "competitors": ["nitrile"],
        "competitor_suppression_factor": 0.45,
        "min_peak_sharpness": 0.01,
        "artifact_confounders": ["weak_nitrile_region_spike", "co2_region_elevated"],
        "caution_messages": [],
    },
    "ketone": {
        "required_groups": ["ketone_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.3,
        "competitors": ["ester", "amide", "carboxylic_acid", "aldehyde"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "aldehyde": {
        "required_groups": ["aldehyde_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.3,
        "competitors": ["ketone", "ester"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "aromatic": {
        "required_groups": ["aromatic_cc_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.32,
        "competitors": ["heteroaromatic", "nitro"],
        "competitor_suppression_factor": 0.55,
        "caution_messages": [],
    },
    "heteroaromatic": {
        "required_groups": ["hetero_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.28,
        "competitors": ["aromatic", "primary_amine", "pyrrole_like_NH"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "pyrrole_like_NH": {
        "required_groups": ["pyrrole_nh_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.28,
        "competitors": ["cyclic_amine", "amide", "primary_amine"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "cyclic_amine": {
        "required_groups": ["cyclic_nh_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.26,
        "competitors": ["pyrrole_like_NH"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "carbonate": {
        "required_groups": ["carbonate_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.28,
        "competitors": ["ester", "ketone", "urethane"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
    "urethane": {
        "required_groups": ["urethane_co_g"],
        "missing_required_cap": 0.45,
        "single_band_cap": 0.26,
        "competitors": ["amide", "ester", "ketone"],
        "competitor_suppression_factor": 0.5,
        "caution_messages": [],
    },
}

OVERLAP_SINGLE_BAND_LABELS = frozenset(
    {
        "siloxane",
        "silicone_or_silane",
        "ether",
        "aryl_ether",
        "ester",
        "phenol",
        "nitro",
        "nitrile",
        "alkyne",
        "heteroaromatic",
    }
)

LOCAL_POSSIBILITY_FLOOR = 0.10


def _match_map(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["band_id"]: m for m in evidence.get("band_matches") or []}


def _group_satisfied(
    match_map: dict[str, dict[str, Any]],
    group_key: str,
    *,
    min_support: float = 0.08,
) -> tuple[bool, float]:
    bids = EVIDENCE_GROUPS.get(group_key) or []
    best = 0.0
    for bid in bids:
        m = match_map.get(bid)
        if not m:
            continue
        if not m.get("matched"):
            continue
        sc = float(m.get("support_score", 0))
        best = max(best, sc)
        if sc >= min_support:
            return True, sc
    return best >= min_support, best


def _competitor_score(assignments: dict[str, Any], name: str) -> float:
    """Resolve aggregate competitors like 'amine'."""
    amine_keys = ("primary_amine", "secondary_amine", "tertiary_amine")
    if name == "amine":
        return max(float((assignments.get(k) or {}).get("score", 0) or 0) for k in amine_keys)
    ent = assignments.get(name)
    if not ent:
        return 0.0
    return float(ent.get("score", 0) or 0)


SILICONE_FG_LABELS = frozenset({"siloxane", "silicone_or_silane"})
_SILICON_BAND_IDS = ("siloxane_sio", "silicone_sic")
_ORGANIC_CO_COMPETITORS = ("alcohol", "phenol", "ether", "aryl_ether", "ester")
_ATR_SILOXANE_CAUTION = (
    "ATR/fingerprint overlap can mimic Si-O/Si-O-Si; paired silicon evidence not observed."
)


def _silicon_evidence_region_count(match_map: dict[str, dict[str, Any]], min_sup: float) -> int:
    """Count distinct silicon-related evidence regions (Si-O-Si window + Si-C / Si-CH3)."""
    n = 0
    for bid in _SILICON_BAND_IDS:
        m = match_map.get(bid)
        if not m or not m.get("matched"):
            continue
        sup = float(m.get("support_score", 0) or 0)
        if sup < max(min_sup, 0.12):
            continue
        if not (m.get("peaks_near") or sup >= 0.22):
            continue
        n += 1
    return n


def _organic_co_competitor_max(assignments: dict[str, Any]) -> float:
    mx = max(float((assignments.get(k) or {}).get("score", 0) or 0) for k in _ORGANIC_CO_COMPETITORS)
    co = assignments.get("C_O_containing")
    if isinstance(co, dict):
        mx = max(mx, float(co.get("score", 0) or 0))
    return mx


_CH_LOCAL_MOTIF_LABELS = frozenset(
    {
        "aliphatic_CH_region",
        "aromatic_CH_region",
        "upper_mid_activity_region",
        "nh_ch_transition_region",
        "CH_stretch_region",
    }
)
_CH_FALLBACK_LABELS = frozenset({"aliphatic_CH_present"})


_NITRO_NOXIDE_CAUTION = (
    "Nitro requires paired NO₂ bands; heterocyclic N–O can mimic this pattern."
)
_NITRO_FRONT_LABEL = "N–O / NO₂ overlap"
_AMIDE_OVERLAP_CAUTION = (
    "Amide II region overlaps enamine, heteroaromatic, and pyrrole-like modes; "
    "subclass requires paired carbonyl/N–H evidence."
)

_NO2_LOCAL_KEYS = ("NO2_asym_region", "NO2_sym_region")


def _assignment_score(assignments: dict[str, Any], key: str) -> float:
    ent = assignments.get(key)
    if not isinstance(ent, dict):
        return 0.0
    return float(ent.get("score", 0) or 0)


def _mid_ir_confounder_score(assignments: dict[str, Any]) -> float:
    return max(
        _assignment_score(assignments, "heteroaromatic"),
        _assignment_score(assignments, "pyrrole_like_NH"),
        _assignment_score(assignments, "aromatic"),
        _assignment_score(assignments, "heterocyclic_N_O_region"),
        _assignment_score(assignments, "heterocyclic_N_oxide"),
        _assignment_score(assignments, "pyrrole_N_oxide_like"),
        _assignment_score(assignments, "N_O_NO2_overlap"),
        _assignment_score(assignments, "n_oxide_confounded_region"),
        _assignment_score(assignments, "enamine_region"),
        _assignment_score(assignments, "amide_II_region"),
        _assignment_score(assignments, "aromatic_CC_region"),
    )


def _demote_no2_local_motifs(assignments: dict[str, Any], *, nitro_supported: bool) -> None:
    """NO₂ region rows are local motifs unless paired nitro FG is supported."""
    if nitro_supported:
        return
    for key in _NO2_LOCAL_KEYS:
        ent = assignments.get(key)
        if not isinstance(ent, dict):
            continue
        sc = max(float(ent.get("score", 0) or 0), 0.22)
        ent["score"] = round(min(sc, 0.38), 4)
        ent["ontology_category"] = "local_motif"
        ent["confidence_class"] = "overlap_limited"
        ent["evidence_completeness"] = "single_band"
        ent["interpretation_strength"] = "local_motif_only"
        ent["assignment_type"] = "local_band_only"
        ent["human_readable_summary"] = (
            f"{_NITRO_FRONT_LABEL} — paired nitro bands not confirmed."
        )
        if _NITRO_NOXIDE_CAUTION not in (ent.get("caution_flags") or []):
            ent.setdefault("caution_flags", []).append(_NITRO_NOXIDE_CAUTION)


def _ensure_overlap_motif(
    assignments: dict[str, Any],
    key: str,
    *,
    score: float,
    summary: str,
) -> None:
    ent = assignments.get(key)
    if not isinstance(ent, dict):
        assignments[key] = {
            "score": round(score, 4),
            "ontology_category": "local_motif",
            "confidence_class": "overlap_limited",
            "evidence_completeness": "single_band",
            "interpretation_strength": "local_motif_only",
            "human_readable_summary": summary,
            "caution_flags": [_NITRO_NOXIDE_CAUTION],
        }
        return
    ent["score"] = round(max(float(ent.get("score", 0) or 0), score), 4)
    ent["ontology_category"] = "local_motif"
    ent["confidence_class"] = "overlap_limited"
    ent.setdefault("caution_flags", []).append(_NITRO_NOXIDE_CAUTION)


def apply_nitro_noxide_confounder_guardrails(
    assignments: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_band_support: float = 0.08,
) -> None:
    """Require paired NO₂ bands for nitro; demote local NO₂ motifs when ambiguous."""
    match_map = _match_map(evidence)
    asym_ok, _ = _group_satisfied(match_map, "nitro_asym_g", min_support=min_band_support)
    sym_ok, _ = _group_satisfied(match_map, "nitro_sym_g", min_support=min_band_support)
    paired = bool(asym_ok and sym_ok)
    confounder = _mid_ir_confounder_score(assignments)
    ent_n = assignments.get("nitro")
    nitro_supported = False
    if isinstance(ent_n, dict):
        nitro_supported = (
            paired
            and confounder < 0.42
            and float(ent_n.get("score", 0) or 0) >= 0.38
            and str(ent_n.get("confidence_class") or "") in ("strong", "supported")
        )

    _demote_no2_local_motifs(assignments, nitro_supported=nitro_supported)

    if not isinstance(ent_n, dict):
        if asym_ok or sym_ok:
            _ensure_overlap_motif(
                assignments,
                "N_O_NO2_overlap",
                score=0.28,
                summary=f"{_NITRO_FRONT_LABEL} — nitro requires paired asymmetric + symmetric bands.",
            )
        return

    if not paired or confounder >= 0.22:
        cap = 0.22 if confounder >= 0.28 else 0.28
        if float(ent_n.get("score", 0) or 0) > cap:
            ent_n["score"] = round(min(float(ent_n["score"]), cap), 4)
        ent_n["confidence_class"] = "overlap_limited"
        ent_n["evidence_completeness"] = "single_band" if not paired else "partial"
        ent_n["interpretation_strength"] = "ambiguous_family"
        if not paired:
            ent_n["human_readable_summary"] = (
                f"{_NITRO_FRONT_LABEL} — paired nitro bands not confirmed."
            )
        if _NITRO_NOXIDE_CAUTION not in (ent_n.get("caution_flags") or []):
            ent_n.setdefault("caution_flags", []).append(_NITRO_NOXIDE_CAUTION)
        if asym_ok or sym_ok or _assignment_score(assignments, "NO2_asym_region") >= 0.15:
            _ensure_overlap_motif(
                assignments,
                "N_O_NO2_overlap",
                score=max(0.26, _assignment_score(assignments, "NO2_asym_region"), _assignment_score(assignments, "NO2_sym_region")),
                summary=f"{_NITRO_FRONT_LABEL} — heterocyclic N–O / enamine / amide II can mimic NO₂.",
            )
        if confounder >= 0.18:
            _ensure_overlap_motif(
                assignments,
                "heterocyclic_N_oxide",
                score=max(0.24, confounder * 0.85),
                summary="Heterocyclic N–O / N-oxide-like overlap (1250–1650 cm⁻¹).",
            )
            if _assignment_score(assignments, "pyrrole_like_NH") >= 0.15:
                _ensure_overlap_motif(
                    assignments,
                    "pyrrole_N_oxide_like",
                    score=max(0.22, _assignment_score(assignments, "pyrrole_like_NH") * 0.7),
                    summary="Pyrrole N-oxide-like / heterocyclic N–O confounder.",
                )
        return

    if confounder >= 0.38 and float(ent_n.get("score", 0) or 0) > 0.32:
        ent_n["score"] = round(min(float(ent_n["score"]), 0.32), 4)
        ent_n["confidence_class"] = "overlap_limited"
        ent_n.setdefault("caution_flags", []).append(_NITRO_NOXIDE_CAUTION)


def apply_amide_overlap_guardrails(
    assignments: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_band_support: float = 0.08,
) -> None:
    """Amide II–like bands without carbonyl/N–H context → overlap, not supported amide."""
    match_map = _match_map(evidence)
    co_ok, _ = _group_satisfied(match_map, "amide_co_g", min_support=min_band_support)
    nh_ok, _ = _group_satisfied(match_map, "amide_nh_or_ii_g", min_support=min_band_support)
    ent = assignments.get("amide")
    amide_ii = _assignment_score(assignments, "amide_II_region")
    enamine = _assignment_score(assignments, "enamine_region")
    pyrrole = _assignment_score(assignments, "pyrrole_like_NH")
    if not isinstance(ent, dict):
        if amide_ii >= 0.15 and not co_ok:
            _ensure_overlap_motif(
                assignments,
                "amide_II_region",
                score=max(amide_ii, 0.24),
                summary="Amide II / enamine / pyrrole-like overlap.",
            )
        return

    weak_context = not co_ok or (not nh_ok and amide_ii >= 0.15)
    hetero_overlap = max(enamine, pyrrole, _assignment_score(assignments, "heteroaromatic")) >= 0.2
    if weak_context or (amide_ii >= 0.18 and hetero_overlap):
        cap = 0.26 if not co_ok else 0.32
        if float(ent.get("score", 0) or 0) > cap:
            ent["score"] = round(min(float(ent["score"]), cap), 4)
        ent["confidence_class"] = "overlap_limited"
        ent["evidence_completeness"] = "partial" if co_ok else "single_band"
        ent["human_readable_summary"] = "Amide II / enamine / pyrrole-like overlap"
        if _AMIDE_OVERLAP_CAUTION not in (ent.get("caution_flags") or []):
            ent.setdefault("caution_flags", []).append(_AMIDE_OVERLAP_CAUTION)
        if amide_ii >= 0.12:
            motif = assignments.get("amide_II_region")
            if isinstance(motif, dict):
                motif["ontology_category"] = "local_motif"
                motif["confidence_class"] = "overlap_limited"
                motif["score"] = round(max(float(motif.get("score", 0) or 0), amide_ii, 0.22), 4)
            else:
                assignments["amide_II_region"] = {
                    "score": round(max(amide_ii, 0.22), 4),
                    "ontology_category": "local_motif",
                    "confidence_class": "overlap_limited",
                    "human_readable_summary": "Amide II / enamine / pyrrole-like overlap.",
                    "caution_flags": [_AMIDE_OVERLAP_CAUTION],
                }


def apply_ch_motif_guardrails(assignments: dict[str, Any]) -> None:
    """C–H / upper-mid motifs stay local; never promoted to supported structural FG alone."""
    for lab in _CH_LOCAL_MOTIF_LABELS:
        ent = assignments.get(lab)
        if not isinstance(ent, dict):
            continue
        sc = float(ent.get("score", 0) or 0)
        if sc <= 0:
            continue
        cap = 0.32 if lab == "upper_mid_activity_region" else 0.35
        ent["ontology_category"] = "local_motif"
        ent["confidence_class"] = "local_possible"
        ent["interpretation_strength"] = "local_motif_only"
        ent["assignment_type"] = "local_band_only"
        ent["evidence_completeness"] = "single_band"
        ent["score"] = round(min(sc, cap), 4)
        msg = "C–H / upper-mid region activity — local motif only; not a standalone structural assignment."
        if msg not in (ent.get("caution_flags") or []):
            ent.setdefault("caution_flags", []).append(msg)

    for lab in _CH_FALLBACK_LABELS:
        ent = assignments.get(lab)
        if not isinstance(ent, dict):
            continue
        sc = float(ent.get("score", 0) or 0)
        if sc <= 0:
            continue
        ent["ontology_category"] = "fallback"
        ent["confidence_class"] = "local_possible"
        ent["interpretation_strength"] = "ambiguous_family"
        ent["assignment_type"] = "local_band_only"
        ent["evidence_completeness"] = "single_band"
        ent["score"] = round(min(sc, 0.28), 4)
        if str(ent.get("human_readable_summary", "")).lower().startswith("spectral evidence supports"):
            ent["human_readable_summary"] = (
                "Aliphatic C–H stretch observed — supports organic material; not alkane confirmed."
            )
        msg = "Aliphatic C–H present — supporting evidence only; pair with more diagnostic bands."
        if msg not in (ent.get("caution_flags") or []):
            ent.setdefault("caution_flags", []).append(msg)


def apply_silicon_soft_gates(
    assignments: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_band_support: float = 0.08,
) -> None:
    """
    Soft gating for siloxane / silicone_or_silane: never high-confidence from one 1000-1150 cm-1
    region alone; keep scores + cautions (no hard deletion).
    """
    from ml.ftir_atr import atr_sensitive_interpretation

    match_map = _match_map(evidence)
    n_si = _silicon_evidence_region_count(match_map, min_band_support)
    org_mx = _organic_co_competitor_max(assignments)
    atr_sensitive = atr_sensitive_interpretation(evidence)
    single_region_cap = 0.20 if atr_sensitive else 0.25
    overlap_caution = (
        "Si-O region overlaps organic C-O / C-O-C / aryl ether / alcohol / ester fingerprint bands."
    )

    for label in SILICONE_FG_LABELS:
        ent = assignments.get(label)
        if not isinstance(ent, dict):
            continue
        score = float(ent.get("score", 0) or 0)
        if score <= 0:
            continue

        if n_si >= 2:
            ratio = float((evidence.get("ratios") or {}).get("siloxane_to_c_o", 0) or 0)
            if ratio < 0.42 or org_mx >= 0.50:
                score = min(score, 0.35)
                ent["confidence_class"] = "tentative"
                ent["evidence_completeness"] = "artifact_limited"
                ent["assignment_type"] = "local_band_only"
                ent.setdefault("caution_flags", []).append(
                    "Two Si-related bands matched but organic C–O dominates or siloxane/C–O ratio is low."
                )
            elif atr_sensitive and org_mx >= 0.38:
                score = min(score, 0.38)
                ent["confidence_class"] = "tentative"
                ent["evidence_completeness"] = "artifact_limited"
                ent["assignment_type"] = "local_band_only"
                ent.setdefault("caution_flags", []).append(
                    "ATR context: paired Si regions present but organic C–O competitors remain strong."
                )
            elif atr_sensitive and str(ent.get("confidence_class") or "") in ("strong", "supported"):
                score = min(score, 0.55)
                ent["confidence_class"] = "tentative"
                ent["evidence_completeness"] = "artifact_limited"
            if overlap_caution not in (ent.get("caution_flags") or []):
                ent.setdefault("caution_flags", []).append(overlap_caution)
            if atr_sensitive and _ATR_SILOXANE_CAUTION not in (ent.get("caution_flags") or []):
                ent.setdefault("caution_flags", []).append(_ATR_SILOXANE_CAUTION)
            ent["score"] = round(score, 4)
            continue

        score = min(score, single_region_cap)
        if org_mx >= 0.45:
            score *= 0.25
            if not ent.get("competing_explanation"):
                ent["competing_explanation"] = (
                    "Stronger organic C-O / ether / ester explanation vs fewer than two silicon-related regions"
                )
            msg = (
                "Stronger organic C-O explanation (>=0.60) while silicon evidence regions < 2; "
                "silicon score suppressed."
            )
            if msg not in (ent.get("caution_flags") or []):
                ent.setdefault("caution_flags", []).append(msg)

        score = max(LOCAL_POSSIBILITY_FLOOR, float(score))
        ent["score"] = round(score, 4)

        if atr_sensitive:
            ent["confidence_class"] = "local_motif_only"
            ent["evidence_completeness"] = "artifact_limited"
            ent["assignment_type"] = "artifact_limited"
            ent["interpretation_strength"] = "local_motif_only"
            if _ATR_SILOXANE_CAUTION not in (ent.get("caution_flags") or []):
                ent.setdefault("caution_flags", []).append(_ATR_SILOXANE_CAUTION)
        elif ent["score"] < 0.12:
            ent["confidence_class"] = "not_supported"
        else:
            ent["confidence_class"] = "local_possible"
            ent["evidence_completeness"] = "single_band"
            ent["assignment_type"] = "local_band_only"

        if overlap_caution not in (ent.get("caution_flags") or []):
            ent.setdefault("caution_flags", []).append(overlap_caution)
        sg = (
            "Single Si-O-like region without paired organosilicon evidence "
            "(Si-O-Si plus Si-C or a second Si-related band); score soft-gated."
        )
        if sg not in (ent.get("caution_flags") or []):
            ent.setdefault("caution_flags", []).append(sg)
        if n_si < 2 and not atr_sensitive and str(ent.get("assignment_type") or "") == "specific":
            ent["assignment_type"] = "local_band_only"


def _apply_si_o_overlap_motif_gate(
    assignments: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_band_support: float = 0.08,
) -> None:
    """Si_O_overlap_region and incomplete silicon FGs stay local_motif_only."""
    match_map = _match_map(evidence)
    n_si = _silicon_evidence_region_count(match_map, min_band_support)
    org_mx = _organic_co_competitor_max(assignments)

    for motif_lab in ("Si_O_overlap_region", "fingerprint_C_O_or_Si_O_overlap"):
        ent = assignments.get(motif_lab)
        if not isinstance(ent, dict):
            continue
        sc = float(ent.get("score", 0) or 0)
        if sc <= 0:
            continue
        ent["ontology_category"] = "local_motif"
        ent["assignment_type"] = "local_band_only"
        ent["confidence_class"] = "local_possible"
        ent["evidence_completeness"] = "single_band"
        ent["interpretation_strength"] = "local_motif_only"
        ent["score"] = round(min(sc, 0.35), 4)
        ent.setdefault("caution_flags", []).append(
            "Si–O vs C–O overlap — local motif only; not a supported organosilicon assignment."
        )

    if n_si < 2:
        for lab in SILICONE_FG_LABELS:
            ent = assignments.get(lab)
            if not isinstance(ent, dict):
                continue
            if org_mx >= 0.45 and float(ent.get("score", 0) or 0) > 0.12:
                fb = assignments.get("fingerprint_C_O_or_Si_O_overlap")
                if isinstance(fb, dict):
                    fb["score"] = round(max(float(fb.get("score", 0) or 0), org_mx * 0.85), 4)
                    fb.setdefault(
                        "caution_flags",
                        [],
                    ).append("Reporting fingerprint C–O / Si–O overlap instead of supported siloxane.")


def _best_peak_metric_in_window(
    peaks: list[dict[str, Any]],
    lo: float,
    hi: float,
    key: str,
) -> float:
    best = 0.0
    for p in peaks:
        w = float(p.get("wn_cm1", 0))
        if lo <= w <= hi:
            best = max(best, float(p.get(key, 0) or 0))
    return best


def _classify_assignment(
    score: float,
    *,
    complete: bool,
    single_band: bool,
    conflict: bool,
    artifact_limited: bool,
) -> tuple[str, str, str]:
    """confidence_class, evidence_completeness, assignment_type."""
    if artifact_limited:
        evc = "artifact_limited"
    elif conflict:
        evc = "conflicting"
    elif single_band:
        evc = "single_band"
    elif complete:
        evc = "complete"
    else:
        evc = "partial"

    if score < 0.08:
        return "not_supported", evc, "specific"
    if complete and score >= 0.85:
        return "strong", evc, "specific"
    if complete and score >= 0.6:
        return "supported", evc, "specific"
    if score >= 0.35 or (not complete and score >= 0.28):
        return "tentative", evc, "specific"
    if score >= 0.12:
        return "local_possible", evc, "local_band_only"
    return "not_supported", evc, "specific"


def apply_v3_guardrails(
    assignments: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_band_support: float = 0.08,
    ontology: str | None = None,
) -> dict[str, Any]:
    """
    Mutates assignment entries in place with v3 fields; returns ambiguity + diagnostics.

    Expected ``evidence`` to optionally include ``artifacts`` from ``ml.ftir_artifacts``.
    """
    match_map = _match_map(evidence)
    peaks = evidence.get("peaks") or []
    ratios = evidence.get("ratios") or {}
    artifacts = evidence.get("artifacts") or {}
    art_flags = artifacts.get("flags") or {}
    artifact_limited = bool(
        art_flags.get("water_vapor_or_moisture_like")
        or art_flags.get("co2_region_elevated")
        or art_flags.get("fingerprint_crowding")
        or art_flags.get("atr_crystal_fingerprint_overlap")
    )

    diagnostics: list[dict[str, Any]] = []
    ambiguity_labels: list[dict[str, Any]] = []

    for label, spec in V3_GUARDRAILS.items():
        ent = assignments.get(label)
        if not ent:
            continue
        ent.pop("competing_explanation", None)
        score = float(ent.get("score", 0))
        req_groups = list(spec.get("required_groups") or [])
        sat: list[bool] = []
        for gk in req_groups:
            ok, _ = _group_satisfied(match_map, gk, min_support=min_band_support)
            sat.append(ok)
        n_req = len(req_groups)
        req_frac = float(sum(1 for x in sat if x) / n_req) if n_req else 1.0
        complete = n_req == 0 or all(sat)

        notes: list[str] = []
        ratio_failed = False
        if n_req >= 2 and req_frac < 1.0:
            cap = float(spec.get("missing_required_cap", 0.4)) * req_frac
            if score > cap:
                notes.append(
                    f"Incomplete required motif ({req_frac:.0%} groups met); score capped toward {cap:.2f} (tentative)."
                )
                score = min(score, cap)

        # Ratio gate (siloxane family)
        rg = spec.get("ratio_min")
        if rg:
            rval = float(ratios.get(str(rg.get("ratio", "")), 0))
            if rval < float(rg.get("min", 0)):
                ratio_failed = True
                cap2 = float(rg.get("otherwise_cap", 0.25))
                if score > cap2:
                    notes.append(f"Ratio {rg['ratio']}={rval:.2f} below {rg['min']:.2f}; organosilicon score capped.")
                    score = min(score, cap2)

        motif_complete = complete and not ratio_failed

        # Single low/med specificity band
        matched_specificity_low = 0
        matched_bands = 0
        for bid in sum((EVIDENCE_GROUPS.get(gk, []) for gk in req_groups), []):
            m = match_map.get(bid)
            if m and m.get("matched"):
                matched_bands += 1
                sp = str(m.get("specificity", "")).lower()
                if sp in ("low", "medium"):
                    matched_specificity_low += 1
        single_band = label in OVERLAP_SINGLE_BAND_LABELS and matched_bands == 1
        # Si-O-Si + Si-C (or two counted Si regions): not "one overlap band" for organosilicon.
        if label in SILICONE_FG_LABELS and _silicon_evidence_region_count(match_map, min_band_support) >= 2:
            single_band = False
        if single_band:
            sbc = float(spec.get("single_band_cap", 0.3))
            if score > sbc:
                notes.append(
                    "Only one overlapping diagnostic region observed; assignment remains tentative."
                )
                score = min(score, sbc)

        # Nitrile / alkyne sharpness
        msharp = float(spec.get("min_peak_sharpness", 0) or 0)
        if msharp > 0 and label in ("nitrile", "alkyne"):
            lo, hi = (2200.0, 2265.0) if label == "nitrile" else (2100.0, 2260.0)
            sh = _best_peak_metric_in_window(peaks, lo, hi, "quality_sharpness")
            if sh < msharp and score > 0.32:
                notes.append(f"Low peak sharpness in diagnostic window (sharpness={sh:.4f}); capped nitrile/alkyne-like score.")
                score = min(score, 0.32)

        # Label-specific artifact confounders (soft caps; see ``artifact_confounders`` in ``V3_GUARDRAILS``)
        for akey in list(spec.get("artifact_confounders") or []):
            if not art_flags.get(akey):
                continue
            if label == "phenol" and akey == "water_vapor_or_moisture_like" and score > 0.25:
                score = min(score, max(LOCAL_POSSIBILITY_FLOOR, score * 0.55))
                notes.append("Moisture-like broad O–H may overlap phenolic assignment — check aryl C–O and ring modes.")
            if label in ("nitrile", "alkyne"):
                if akey == "co2_region_elevated" and score > 0.30:
                    score = min(score, max(LOCAL_POSSIBILITY_FLOOR, score * 0.52))
                    notes.append("CO₂ / atmospheric region elevated — nitrile vs alkyne calls need corroboration.")
                if akey == "weak_nitrile_region_spike" and score > 0.29:
                    score = min(score, 0.29)
                    notes.append("Weak isolated nitrile-window feature — prefer tentative / local_possible assignment.")

        incomplete_for_comp = (req_frac < 1.0) or single_band or ratio_failed

        # Competitor suppression
        comp_factor = float(spec.get("competitor_suppression_factor", 0.5))
        competitors = list(spec.get("competitors") or [])
        best_comp = 0.0
        best_name = ""
        for c in competitors:
            cs = _competitor_score(assignments, c)
            if cs > best_comp:
                best_comp = cs
                best_name = c
        if best_comp >= 0.42 and incomplete_for_comp and score > LOCAL_POSSIBILITY_FLOOR:
            prev = score
            new_score = max(LOCAL_POSSIBILITY_FLOOR, score * comp_factor)
            if new_score < prev - 1e-6:
                ent["competing_explanation"] = f"{best_name} (score≈{best_comp:.2f})"
                notes.append("Stronger competing explanation detected.")
                if best_name:
                    notes.append(f"Leading competitor: {best_name} (score {best_comp:.2f}).")
            score = new_score

        if artifact_limited and score > 0.45:
            score = min(score, 0.48)
            notes.append("Artifact / overlap context limits confidence.")

        conflict = best_comp >= 0.5 and score < best_comp - 0.08

        conf_cls, ev_cmp, assign_type = _classify_assignment(
            score,
            complete=motif_complete,
            single_band=single_band,
            conflict=conflict,
            artifact_limited=artifact_limited,
        )

        ent["score"] = round(score, 4)
        ent["confidence_class"] = conf_cls
        ent["evidence_completeness"] = ev_cmp
        ent["assignment_type"] = assign_type
        ent["required_evidence_fraction"] = round(req_frac, 4)
        for n in notes:
            ent.setdefault("caution_flags", []).append(n)
        for msg in spec.get("caution_messages") or []:
            if msg and msg not in (ent.get("caution_flags") or []):
                ent.setdefault("caution_flags", []).append(msg)
        if artifacts.get("cautions"):
            for ac in artifacts["cautions"][:3]:
                if ac not in (ent.get("caution_flags") or []):
                    ent.setdefault("caution_flags", []).append(ac)

        diagnostics.append(
            {
                "label": label,
                "score": ent["score"],
                "confidence_class": conf_cls,
                "evidence_completeness": ev_cmp,
                "required_fraction": req_frac,
                "competitor": best_name or None,
                "competitor_score": round(best_comp, 4) if best_comp else None,
            }
        )

    apply_nitro_noxide_confounder_guardrails(assignments, evidence, min_band_support=min_band_support)
    apply_amide_overlap_guardrails(assignments, evidence, min_band_support=min_band_support)
    apply_silicon_soft_gates(assignments, evidence, min_band_support=min_band_support)
    _apply_si_o_overlap_motif_gate(assignments, evidence, min_band_support=min_band_support)
    apply_ch_motif_guardrails(assignments)

    # --- Ambiguity fallbacks (family labels, not SVM heads) ---
    def _add_fallback(fid: str, title: str, reason: str, related: list[str]) -> None:
        ambiguity_labels.append(
            {
                "id": fid,
                "title": title,
                "reason": reason,
                "related_labels": related,
            }
        )

    ph = float((assignments.get("phenol") or {}).get("score", 0))
    alc = float((assignments.get("alcohol") or {}).get("score", 0))
    if ph >= 0.22 and alc >= 0.22 and abs(ph - alc) < 0.12:
        if (assignments.get("phenol") or {}).get("evidence_completeness") != "complete" and (
            assignments.get("alcohol") or {}
        ).get("evidence_completeness") != "complete":
            _add_fallback(
                "hydroxy_containing",
                "Hydroxy-containing (ambiguous)",
                "Hydroxy family supported; alcohol vs phenol depends on aromatic / aryl C–O evidence.",
                ["phenol", "alcohol"],
            )

    es = float((assignments.get("ester") or {}).get("score", 0))
    et = float((assignments.get("ether") or {}).get("score", 0))
    ae = float((assignments.get("aryl_ether") or {}).get("score", 0))
    if max(es, et, ae) >= 0.28 and len({round(x, 2) for x in (es, et, ae) if x > 0.15}) >= 2:
        _add_fallback(
            "C_O_containing",
            "C–O-containing (ambiguous)",
            "C–O family supported; ether vs aryl ether vs siloxane not uniquely resolved.",
            ["ester", "ether", "aryl_ether", "siloxane"],
        )

    amide_s = float((assignments.get("amide") or {}).get("score", 0))
    amax = max(
        float((assignments.get("primary_amine") or {}).get("score", 0)),
        float((assignments.get("secondary_amine") or {}).get("score", 0)),
    )
    if amide_s >= 0.22 and amax >= 0.22 and (assignments.get("amide") or {}).get("evidence_completeness") != "complete":
        _add_fallback(
            "nitrogen_containing",
            "Nitrogen-containing (ambiguous)",
            "Nitrogen family supported; amide vs amine requires paired N–H / amide II evidence.",
            ["amide", "primary_amine", "secondary_amine"],
        )

    sil = float((assignments.get("siloxane") or {}).get("score", 0))
    sil2 = float((assignments.get("silicone_or_silane") or {}).get("score", 0))
    co_rich = max(es, et, ae, ph)
    if max(sil, sil2) >= 0.18 and co_rich >= 0.35 and max(sil, sil2) < co_rich - 0.05:
        _add_fallback(
            "fingerprint_C_O_or_Si_O_overlap",
            "Fingerprint C–O / Si–O overlap",
            "Fingerprint C–O / Si–O overlap — prefer this over siloxane when paired Si evidence is absent.",
            ["siloxane", "silicone_or_silane", "ether", "aryl_ether"],
        )

    carb = max(
        float((assignments.get("ketone") or {}).get("score", 0)),
        float((assignments.get("ester") or {}).get("score", 0)),
        float((assignments.get("amide") or {}).get("score", 0)),
        float((assignments.get("aldehyde") or {}).get("score", 0)),
    )
    if carb >= 0.4:
        tops = sorted(
            [
                ("ketone", float((assignments.get("ketone") or {}).get("score", 0))),
                ("ester", float((assignments.get("ester") or {}).get("score", 0))),
                ("amide", float((assignments.get("amide") or {}).get("score", 0))),
                ("aldehyde", float((assignments.get("aldehyde") or {}).get("score", 0))),
            ],
            key=lambda t: -t[1],
        )
        if tops[0][1] - tops[1][1] < 0.08 and tops[1][1] > 0.25:
            _add_fallback(
                "carbonyl_containing",
                "Carbonyl-containing (ambiguous)",
                "Carbonyl family supported; ester / amide / ketone subclass needs paired fingerprint evidence.",
                [t[0] for t in tops[:3]],
            )

    ar = float((assignments.get("aromatic") or {}).get("score", 0))
    het = float((assignments.get("heteroaromatic") or {}).get("score", 0))
    if ar >= 0.28 and het >= 0.28 and abs(ar - het) < 0.1:
        _add_fallback("aromatic_system", "Aromatic system (ambiguous)", "Carbocyclic vs heteroaromatic overlap.", ["aromatic", "heteroaromatic"])

    apply_deconv_soft_guardrails(assignments, evidence)

    return {
        "version": "v3_guarded",
        "diagnostics": diagnostics,
        "ambiguity_labels": ambiguity_labels,
    }


def _deconv_region(evidence: dict[str, Any], region_id: str) -> dict[str, Any]:
    deconv = evidence.get("deconv") or {}
    return dict((deconv.get("regions") or {}).get(region_id) or {})


def apply_deconv_soft_guardrails(assignments: dict[str, Any], evidence: dict[str, Any]) -> None:
    """
  Soft use of fitted-component evidence. Poor fits add caution; good fits allow modest boosts.
  Never replaces band evidence; does not hard-delete assignments.
    """
    if not evidence.get("deconv"):
        return

    def _caution(ent: dict[str, Any], msg: str) -> None:
        ent.setdefault("caution_flags", [])
        if msg not in ent["caution_flags"]:
            ent["caution_flags"].append(msg)

    triple = _deconv_region(evidence, "nitrile_alkyne")
    nitro = _deconv_region(evidence, "nitro")
    oh = _deconv_region(evidence, "oh_nh")
    cosio = _deconv_region(evidence, "c_o_sio_overlap")

    triple_ok = bool(triple.get("success")) and float(triple.get("fit_r2", 0)) >= 0.4
    sharp_tb = float(triple.get("min_fwhm", 99)) <= 28.0 and int(triple.get("n_components", 0)) >= 1

    for lab in ("nitrile", "alkyne"):
        ent = assignments.get(lab)
        if not ent:
            continue
        sc = float(ent.get("score", 0))
        if triple_ok and sharp_tb and sc > 0.28 and sc < 0.55:
            ent["score"] = round(min(0.58, sc * 1.08), 4)
            ent.setdefault("notes", []).append("Deconv: sharp triple-bond component supports tentative call.")
        elif triple_ok and not sharp_tb and sc > 0.35:
            ent["score"] = round(min(sc, 0.34), 4)
            _caution(ent, "Deconv: no sharp fitted component in 2100–2260 cm⁻¹; nitrile/alkyne down-weighted.")

    ent_n = assignments.get("nitro")
    if ent_n:
        sc = float(ent_n.get("score", 0))
        pair = float(nitro.get("n_components", 0)) >= 2 and bool(nitro.get("success"))
        asym = any(
            1500 <= float(c.get("center", 0)) <= 1570
            for c in (nitro.get("components") or [])
        )
        sym = any(1320 <= float(c.get("center", 0)) <= 1390 for c in (nitro.get("components") or []))
        if pair and asym and sym and sc > 0.25 and sc < 0.6:
            ent_n["score"] = round(min(0.62, sc * 1.1), 4)
            ent_n.setdefault("notes", []).append("Deconv: paired NO₂-like components support nitro assignment.")
        elif sc > 0.4 and not (asym and sym):
            ent_n["score"] = round(min(sc, 0.38), 4)
            _caution(ent_n, "Deconv: nitro region lacks paired asym/sym fitted components.")

    for lab in SILICONE_FG_LABELS:
        ent = assignments.get(lab)
        if not ent:
            continue
        sc = float(ent.get("score", 0))
        n_comp = int(cosio.get("n_components", 0) or 0)
        if bool(cosio.get("success")) and n_comp >= 2 and sc > 0.32 and sc < 0.55:
            ent["score"] = round(min(0.52, sc * 1.05), 4)
        elif sc > 0.45 and n_comp < 2:
            ent["score"] = round(min(sc, 0.42), 4)
            _caution(ent, "Deconv: C–O/Si–O overlap region shows <2 components; siloxane remains tentative.")

    for lab in ("alcohol", "phenol", "carboxylic_acid_OH"):
        ent = assignments.get(lab)
        if not ent:
            continue
        sc = float(ent.get("score", 0))
        if bool(oh.get("success")) and float(oh.get("mean_fwhm", 0)) >= 80 and sc > 0.2 and sc < 0.5:
            ent["score"] = round(min(0.52, sc * 1.06), 4)

    for lab, ent in list(assignments.items()):
        if not isinstance(ent, dict):
            continue
        sc = float(ent.get("score", 0))
        for rid, reg in (evidence.get("deconv") or {}).get("regions", {}).items():
            if not isinstance(reg, dict):
                continue
            if bool(reg.get("success")):
                continue
            if float(reg.get("fit_r2", 0)) < 0.25 and sc > 0.45:
                ent["score"] = round(min(sc, 0.44), 4)
                _caution(
                    ent,
                    f"Deconv: poor fit in {rid} (R²={float(reg.get('fit_r2', 0)):.2f}); interpret with caution.",
                )
                break


def attach_interpretation_strength_v4(assignments: dict[str, Any], *, ontology: str | None) -> None:
    """Add v4 standardized interpretation_strength alongside legacy confidence_class."""
    from ml.ftir_ontology import is_v4, map_confidence_class_to_interpretation_strength

    if not is_v4(ontology):
        return
    for label, ent in assignments.items():
        if not isinstance(ent, dict):
            continue
        cc = str(ent.get("confidence_class") or "").lower()
        if ent.get("ontology_category") == "local_motif":
            ent["interpretation_strength"] = "local_motif_only"
            continue
        if str(label) in SILICONE_FG_LABELS and (
            cc in ("local_possible", "local_motif_only")
            or str(ent.get("evidence_completeness") or "") in ("single_band", "artifact_limited")
            or str(ent.get("assignment_type") or "") == "artifact_limited"
        ):
            ent["interpretation_strength"] = "local_motif_only"
            continue
        base = map_confidence_class_to_interpretation_strength(
            ent.get("confidence_class"),
            assignment_type=ent.get("assignment_type"),
            evidence_completeness=ent.get("evidence_completeness"),
        )
        if ent.get("ontology_category") == "fallback" and cc == "tentative":
            ent["interpretation_strength"] = "ambiguous_family"
        else:
            ent["interpretation_strength"] = base


def annotate_non_guarded_labels(assignments: dict[str, Any], evidence: dict[str, Any]) -> None:
    """Fill confidence_class etc. for labels without dedicated v3 spec (backward compat)."""
    match_map = _match_map(evidence)
    for label, ent in assignments.items():
        if label in V3_GUARDRAILS or not isinstance(ent, dict):
            continue
        if ent.get("confidence_class"):
            continue
        req = ent.get("missing_expected_bands") or []
        complete = not bool(req)
        sc = float(ent.get("score", 0))
        cc, evc, at = _classify_assignment(sc, complete=complete, single_band=False, conflict=False, artifact_limited=False)
        ent["confidence_class"] = cc
        ent["evidence_completeness"] = "complete" if complete else "partial"
        ent["assignment_type"] = at
        _ = match_map  # reserved for future per-label completeness
