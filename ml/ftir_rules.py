"""
Rule-based functional-group assignment from spectral evidence.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np

from ml.ftir_band_library import format_band_row, load_band_library

Confidence = Literal["high", "medium", "low"]

# Per-FG rule: required band_ids (must match), supporting band_ids, conflicting labels, min score
_FG_RULES: dict[str, dict[str, Any]] = {
    "phenol": {
        "required": ["phenol_oh", "aromatic_cc"],
        "supporting": ["phenolic_co", "aryl_ether_co"],
        "conflicts": [],
        "min_score": 0.35,
        "caution_template": "Alcohol/phenol distinction depends on aromatic and phenolic C-O evidence.",
    },
    "alcohol": {
        "required": ["alcohol_oh"],
        "supporting": ["ether_co"],
        "conflicts": ["phenol"],
        "min_score": 0.30,
        "caution_template": "Broad O-H overlaps phenol and acid; check aromatic bands.",
    },
    "primary_amine": {
        "required": ["amine_nh"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.28,
        "caution_template": "Weak N-H alone is insufficient; check for competing amide carbonyl bands.",
    },
    "secondary_amine": {
        "required": ["amine_nh2"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.25,
        "caution_template": "Secondary amine N-H is often weak in FTIR.",
    },
    "tertiary_amine": {
        "required": [],
        "supporting": ["amine_nh"],
        "conflicts": [],
        "min_score": 0.15,
        "caution_template": "Tertiary amines often lack N-H stretch; assignment is tentative.",
    },
    "amide": {
        "required_any_groups": [["amide_co"], ["amide_nh", "amide_ii"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.38,
        "caution_template": "Amide needs amide I plus N–H stretch or amide II; ester carbonyl can partially overlap.",
    },
    "pyrrole_like_NH": {
        "required": ["pyrrole_nh"],
        "supporting": ["heteroaromatic"],
        "conflicts": ["cyclic_amine"],
        "min_score": 0.32,
        "caution_template": "Distinguish from cyclic aliphatic amine using heteroaromatic fingerprint.",
    },
    "cyclic_amine": {
        "required": ["cyclic_amine_nh"],
        "supporting": [],
        "conflicts": ["pyrrole_like_NH"],
        "min_score": 0.25,
        "caution_template": "Do not confuse with pyrrole unless heteroaromatic support is absent.",
    },
    "aromatic": {
        "required": ["aromatic_cc"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.30,
    },
    "heteroaromatic": {
        "required": ["heteroaromatic"],
        "supporting": ["aromatic_cc"],
        "conflicts": [],
        "min_score": 0.32,
        "min_required_support": 0.45,
        "caution_template": "Distinguish from carbocyclic aromatic using heteroaromatic ring modes.",
    },
    "ketone": {
        "required": ["ketone_co"],
        "supporting": [],
        "conflicts": ["ester", "carboxylic_acid", "amide"],
        "min_score": 0.32,
    },
    "aldehyde": {
        "required": ["aldehyde_co"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.32,
    },
    "ester": {
        "required": ["ester_co", "ester_co_o"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.36,
        "caution_template": "Ester requires carbonyl and C-O region support together.",
    },
    "carboxylic_acid": {
        "required": ["carboxylic_co", "acid_oh_broad"],
        "supporting": ["broad_oh"],
        "conflicts": [],
        "min_score": 0.38,
        "caution_template": "Broad acid O-H dimer region is distinctive but overlaps alcohol/phenol.",
    },
    "ether": {
        "required": ["ether_co"],
        "supporting": [],
        "conflicts": ["ester"],
        "min_score": 0.28,
    },
    "aryl_ether": {
        "required": ["aryl_ether_co"],
        "supporting": ["aromatic_cc"],
        "conflicts": [],
        "min_score": 0.30,
    },
    "nitrile": {
        "required": ["nitrile_cn"],
        "supporting": [],
        "conflicts": ["alkyne"],
        "min_score": 0.40,
        "min_required_support": 0.55,
        "caution_template": "Nitrile C≡N should be a sharp band near 2200–2260 cm⁻¹, not broad fingerprint noise.",
    },
    "nitro": {
        "required": ["nitro_asym", "nitro_sym"],
        "required_band_min": {"nitro_asym": 0.45, "nitro_sym": 0.22},
        "supporting": [],
        "conflicts": [],
        "min_score": 0.38,
        "caution_template": "Nitro assignment requires both asymmetric and symmetric NO₂ band evidence.",
    },
    "alkene": {
        "required": ["alkene_cc"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.25,
    },
    "alkyne": {
        "required": ["alkyne_cc"],
        "supporting": ["alkyne_ch"],
        "conflicts": ["nitrile"],
        "min_score": 0.32,
        "min_required_support": 0.55,
        "caution_template": "Alkyne C≡C overlaps nitrile region; check sharp nitrile band near 2200-2260 cm⁻¹.",
    },
    "siloxane": {
        "required": ["siloxane_sio"],
        "supporting": ["silicone_sic"],
        "conflicts": ["ether"],
        "min_score": 0.42,
        "caution_template": "Siloxane needs Si-O-Si plus a second Si-related region (e.g. Si-C); "
        "one 1000-1150 cm-1 band alone is never high-confidence siloxane (overlaps organic C-O).",
    },
    "silicone_or_silane": {
        "required": ["siloxane_sio"],
        "supporting": ["silicone_sic"],
        "conflicts": [],
        "min_score": 0.42,
        "ratio_gates": [{"ratio": "siloxane_to_c_o", "min": 0.55}],
        "caution_template": "Organosilicon needs paired Si evidence (Si-O-Si + Si-C or two Si regions), not fingerprint C-O alone.",
    },
    "carbonate": {
        "required": ["carbonate_co"],
        "supporting": [],
        "conflicts": ["ester", "ketone"],
        "min_score": 0.28,
        "caution_template": "Carbonate C=O overlaps ester/ketone carbonyl region; confirm with other bands.",
    },
    "urethane": {
        "required": ["urethane_co"],
        "supporting": ["amide_nh"],
        "conflicts": ["amide", "ester"],
        "min_score": 0.26,
        "caution_template": "Urethane/urea carbonyl overlaps amide I; check N-H and fingerprint context.",
    },
}

# v4 ontology: additional scored layers (families, fallbacks, local motifs). Keys must not shadow v3 FG rules.
_V4_ONTOLOGY_RULES: dict[str, dict[str, Any]] = {
    "broad_OH_NH_region": {
        "required_any_groups": [["broad_oh", "alcohol_oh", "phenol_oh", "acid_oh_broad", "amine_nh", "amide_nh"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.22,
        "ontology_category": "local_motif",
        "caution_template": "Broad O–H / N–H envelope — local spectral motif, not a specific FG assignment.",
    },
    "carbonyl_region": {
        "required_any_groups": [
            ["ketone_co", "aldehyde_co", "ester_co", "carboxylic_co", "amide_co", "carbonate_co", "urethane_co"]
        ],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.22,
        "ontology_category": "local_motif",
        "caution_template": "Carbonyl stretch region — classify with paired fingerprint evidence.",
    },
    "C_O_fingerprint_region": {
        "required_any_groups": [["ether_co", "ester_co_o", "aryl_ether_co", "phenolic_co"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.20,
        "ontology_category": "local_motif",
        "caution_template": "C–O fingerprint window — overlaps ethers, esters, and phenolic C–O.",
    },
    "aromatic_CC_region": {
        "required_any_groups": [["aromatic_cc", "heteroaromatic"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.22,
        "ontology_category": "local_motif",
        "caution_template": "Aromatic ring modes — local motif; heteroaromatic vs carbocyclic needs context.",
    },
    "nitrile_alkyne_region": {
        "required_any_groups": [["nitrile_cn", "alkyne_cc"]],
        "supporting": ["alkyne_ch"],
        "conflicts": [],
        "min_score": 0.20,
        "ontology_category": "local_motif",
        "caution_template": "Triple-bond / nitrile window — nitrile vs alkyne is spectrally ambiguous without sharpness.",
    },
    "NO2_asym_region": {
        "required": ["nitro_asym"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.18,
        "ontology_category": "local_motif",
        "caution_template": "NO₂ asymmetric region alone — pair with symmetric NO₂ band for nitro FG.",
    },
    "NO2_sym_region": {
        "required": ["nitro_sym"],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.16,
        "ontology_category": "local_motif",
        "caution_template": "NO₂ symmetric region — pair with asymmetric NO₂ for nitro FG.",
    },
    "enamine_region": {
        "required": ["enamine_c_c_cn"],
        "supporting": ["aromatic_cc", "amide_ii"],
        "conflicts": [],
        "min_score": 0.16,
        "ontology_category": "local_motif",
        "caution_template": "Enamine / C=C–N overlap in mid-IR — not a standalone amide call.",
    },
    "heterocyclic_N_O_region": {
        "required_any_groups": [["heterocyclic_n_oxide", "n_oxide_high", "n_oxide_low", "pyrrole_n_oxide_like"]],
        "supporting": ["nitro_asym"],
        "conflicts": [],
        "min_score": 0.16,
        "ontology_category": "local_motif",
        "caution_template": "Heterocyclic N–O / N-oxide-like — confounds nitro and amide II region.",
    },
    "heterocyclic_N_oxide": {
        "required_any_groups": [["heterocyclic_n_oxide", "n_oxide_high", "n_oxide_low"]],
        "supporting": ["pyrrole_n_oxide_like"],
        "conflicts": [],
        "min_score": 0.14,
        "ontology_category": "local_motif",
        "caution_template": "Heterocyclic N-oxide-like ambiguity — not a high-confidence FG.",
    },
    "pyrrole_N_oxide_like": {
        "required": ["pyrrole_n_oxide_like"],
        "supporting": ["pyrrole_nh", "heterocyclic_n_oxide"],
        "conflicts": [],
        "min_score": 0.14,
        "ontology_category": "local_motif",
        "caution_template": "Pyrrole N-oxide-like / heterocyclic N–O confounder.",
    },
    "N_O_NO2_overlap": {
        "required_any_groups": [["nitro_asym", "nitro_sym", "heterocyclic_n_oxide", "n_oxide_high"]],
        "supporting": ["amide_ii", "enamine_c_c_cn"],
        "conflicts": [],
        "min_score": 0.14,
        "ontology_category": "local_motif",
        "caution_template": "N–O / NO₂ overlap — nitro requires paired bands; N-oxide can mimic.",
    },
    "n_oxide_confounded_region": {
        "required_any_groups": [["heterocyclic_n_oxide", "n_oxide_high", "nitro_asym"]],
        "supporting": ["nitro_sym"],
        "conflicts": [],
        "min_score": 0.14,
        "ontology_category": "local_motif",
        "caution_template": "N–O / NO₂ overlap region — paired nitro not confirmed.",
    },
    "Si_O_overlap_region": {
        "required_any_groups": [["siloxane_sio", "ether_co", "ester_co_o"]],
        "supporting": ["silicone_sic"],
        "conflicts": [],
        "min_score": 0.18,
        "ontology_category": "local_motif",
        "caution_template": "Si–O vs C–O overlap — local motif; organosilicon needs Si–O dominance + paired evidence.",
    },
    "amide_II_region": {
        "required": ["amide_ii"],
        "supporting": ["amide_co", "amide_nh"],
        "conflicts": [],
        "min_score": 0.18,
        "ontology_category": "local_motif",
        "caution_template": "Amide II / N–H bend context — combine with amide I / N–H stretch for amide FG.",
    },
    "hydroxy_containing": {
        "required_any_groups": [["broad_oh", "alcohol_oh", "phenol_oh", "acid_oh_broad"]],
        "supporting": ["aromatic_cc"],
        "conflicts": [],
        "min_score": 0.24,
        "ontology_category": "fallback",
        "caution_template": "Hydroxy-containing family — subclass (alcohol vs phenol vs acid) needs extra bands.",
    },
    "carbonyl_containing": {
        "required_any_groups": [
            ["ketone_co", "aldehyde_co", "ester_co", "carboxylic_co", "amide_co", "carbonate_co", "urethane_co"]
        ],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.26,
        "ontology_category": "fallback",
        "caution_template": "Carbonyl family — ketone/ester/amide distinction needs fingerprint pairing.",
    },
    "nitrogen_containing": {
        "required_any_groups": [["amine_nh", "amine_nh2", "amide_nh", "amide_ii", "nitrile_cn", "nitro_asym"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.24,
        "ontology_category": "fallback",
        "caution_template": "Nitrogen-containing family — resolve amine vs amide vs nitrile with paired evidence.",
    },
    "aromatic_system": {
        "required_any_groups": [["aromatic_cc", "heteroaromatic"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.24,
        "ontology_category": "fallback",
        "caution_template": "Aromatic system — heteroaromatic vs carbocyclic needs heteroatom ring modes.",
    },
    "C_O_containing": {
        "required_any_groups": [["ether_co", "ester_co_o", "aryl_ether_co", "phenolic_co"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.22,
        "ontology_category": "fallback",
        "caution_template": "C–O fingerprint family — ether vs ester vs aryl ether needs carbonyl context.",
    },
    "unsaturation_possible": {
        "required_any_groups": [["alkene_cc", "alkyne_cc", "nitrile_cn"]],
        "supporting": ["alkyne_ch"],
        "conflicts": [],
        "min_score": 0.22,
        "ontology_category": "fallback",
        "caution_template": "Unsaturation / triple-bond region — alkene vs alkyne vs nitrile needs sharpness checks.",
    },
    "fingerprint_C_O_or_Si_O_overlap": {
        "required_any_groups": [["ether_co", "ester_co_o", "siloxane_sio"]],
        "supporting": ["silicone_sic"],
        "conflicts": [],
        "min_score": 0.20,
        "ontology_category": "fallback",
        "caution_template": "Fingerprint C-O vs Si-O overlap — use siloxane only with two silicon-related regions and Si-O dominance vs organic C-O.",
    },
    "triple_bond_region_possible": {
        "required_any_groups": [["nitrile_cn", "alkyne_cc"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.20,
        "ontology_category": "fallback",
        "caution_template": "Sharp feature near 2100–2260 cm⁻¹ — nitrile vs alkyne ambiguity common.",
    },
    "aliphatic_CH_region": {
        "required_any_groups": [["aliphatic_ch_asym", "aliphatic_ch_sym"]],
        "supporting": ["aromatic_ch_stretch"],
        "conflicts": [],
        "min_score": 0.14,
        "ontology_category": "local_motif",
        "caution_template": "Aliphatic C–H stretch — local motif; supports organic matrix, not a specific FG alone.",
    },
    "aromatic_CH_region": {
        "required_any_groups": [["aromatic_ch_stretch"]],
        "supporting": ["aromatic_cc"],
        "conflicts": [],
        "min_score": 0.14,
        "ontology_category": "local_motif",
        "caution_template": "Aromatic/sp² C–H stretch — pair with ring C=C modes below 1650 cm⁻¹.",
    },
    "upper_mid_activity_region": {
        "region_keys": ["upper_mid_activity"],
        "min_region_rel": 0.10,
        "required_any_groups": [],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.12,
        "ontology_category": "local_motif",
        "caution_template": "Upper mid-IR activity — usually weak/less diagnostic; spectral activity only.",
    },
    "nh_ch_transition_region": {
        "required_any_groups": [["amide_nh", "pyrrole_nh", "amine_nh", "aromatic_ch_stretch"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.12,
        "ontology_category": "local_motif",
        "caution_template": "N–H / aromatic C–H shoulder region — local activity, not a standalone assignment.",
    },
    "CH_stretch_region": {
        "required_any_groups": [["aliphatic_ch_asym", "aliphatic_ch_sym", "aromatic_ch_stretch"]],
        "supporting": [],
        "conflicts": [],
        "min_score": 0.12,
        "ontology_category": "local_motif",
        "caution_template": "C–H stretch region — local spectral motif.",
    },
    "aliphatic_CH_present": {
        "required_any_groups": [["aliphatic_ch_asym", "aliphatic_ch_sym"]],
        "supporting": ["aromatic_ch_stretch"],
        "conflicts": [],
        "min_score": 0.12,
        "ontology_category": "fallback",
        "caution_template": "Aliphatic C–H present — supporting evidence for organic material, not alkane confirmed.",
    },
}


def get_effective_fg_rules(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Base rules merged with optional ``label_overrides`` from rules config."""
    rules = copy.deepcopy(_FG_RULES)
    if str((config or {}).get("ontology") or "v3").lower() == "v4":
        rules.update(copy.deepcopy(_V4_ONTOLOGY_RULES))
    overrides = (config or {}).get("label_overrides") or {}
    for label, patch in overrides.items():
        if label in rules:
            rules[label] = {**rules[label], **patch}
        else:
            rules[label] = dict(patch)
    return rules


def _apply_post_rules(assignments: dict[str, Any], config: dict[str, Any] | None) -> None:
    """Optional score caps after main assignment (e.g. alcohol vs phenol)."""
    post = (config or {}).get("post_rules") or {}
    cap = post.get("alcohol_max_if_phenol_aromatic")
    if not cap:
        return
    if_label = str(cap.get("if_label", "phenol"))
    cap_label = str(cap.get("cap_label", "alcohol"))
    if_min = float(cap.get("if_min_score", 0.45))
    cap_to = float(cap.get("cap_to", 0.25))
    phenol = assignments.get(if_label)
    alcohol = assignments.get(cap_label)
    if phenol and alcohol and float(phenol.get("score", 0)) >= if_min:
        if float(alcohol.get("score", 0)) > cap_to:
            alcohol["score"] = round(cap_to, 4)
            alcohol["confidence"] = "low"
            alcohol.setdefault("caution_flags", []).append(
                f"Score capped: strong {if_label} evidence; aliphatic alcohol less likely."
            )


def _match_map(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["band_id"]: m for m in evidence.get("band_matches") or []}


def _band_hit(match_map: dict[str, dict[str, Any]], band_id: str, *, min_score: float = 0.08) -> tuple[bool, float, dict[str, Any] | None]:
    m = match_map.get(band_id)
    if not m:
        return False, 0.0, None
    sc = float(m.get("support_score", 0))
    return sc >= min_score, sc, m


def assign_functional_groups_from_evidence(
    evidence: dict[str, Any],
    band_library: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return per-label assignments with scores, confidence, evidence lists, cautions.
    """
    _ = band_library or load_band_library(prefer_python=True)
    cfg = config or {}
    min_hit = float(cfg.get("min_band_support", 0.08))
    match_map = _match_map(evidence)
    assignments: dict[str, Any] = {}
    active_rules = get_effective_fg_rules(cfg)
    disabled_labels = frozenset((cfg or {}).get("disabled_labels") or [])
    ratios = evidence.get("ratios") or {}

    for label, spec in active_rules.items():
        if label in disabled_labels:
            continue
        required = list(spec.get("required") or [])
        required_any_groups: list[list[str]] = list(spec.get("required_any_groups") or [])
        supporting = list(spec.get("supporting") or [])
        req_hits: list[dict[str, Any]] = []
        sup_hits: list[dict[str, Any]] = []
        missing: list[str] = []
        req_scores: list[float] = []

        min_req_sup = float(spec.get("min_required_support", 0.0))
        per_band_min: dict[str, float] = dict(spec.get("required_band_min") or {})

        if required_any_groups:
            seen_ids: set[str] = set()
            sat_n = 0
            for group in required_any_groups:
                best_sc = 0.0
                best_m: dict[str, Any] | None = None
                group_ok = False
                for bid in group:
                    bid_min = float(per_band_min.get(bid, min_req_sup))
                    ok, sc, m = _band_hit(match_map, bid, min_score=min_hit)
                    if ok and m and sc >= bid_min:
                        group_ok = True
                        if sc > best_sc:
                            best_sc, best_m = sc, m
                    elif ok and m and sc < bid_min:
                        missing.append(f"Band {bid} below minimum support {bid_min:.2f} (got {sc:.2f})")
                if group_ok and best_m:
                    sat_n += 1
                    bid = str(best_m.get("band_id", ""))
                    if bid and bid not in seen_ids:
                        req_hits.append(best_m)
                        req_scores.append(best_sc)
                        seen_ids.add(bid)
                else:
                    labels_s = ", ".join(group)
                    missing.append(f"Expected at least one of: {labels_s} (amide I + N–H or amide II context)")
            n_req_groups = len(required_any_groups)
            req_frac = float(sat_n / n_req_groups) if n_req_groups else 1.0
        else:
            for bid in required:
                bid_min = float(per_band_min.get(bid, min_req_sup))
                ok, sc, m = _band_hit(match_map, bid, min_score=min_hit)
                if ok and m and sc >= bid_min:
                    req_hits.append(m)
                    req_scores.append(sc)
                elif ok and m and sc < bid_min:
                    missing.append(f"Band {bid} below minimum support {bid_min:.2f} (got {sc:.2f})")
                else:
                    band_def = next((b for b in load_band_library() if b["id"] == bid), None)
                    if band_def:
                        missing.append(
                            f"Expected {format_band_row(band_def)} ({band_def.get('importance', 'required')})"
                        )

            if required:
                req_frac = len(req_hits) / len(required)
            else:
                req_frac = 1.0 if sup_hits else 0.0

        for bid in supporting:
            ok, sc, m = _band_hit(match_map, bid, min_score=min_hit)
            if ok and m:
                sup_hits.append(m)

        region_keys = list(spec.get("region_keys") or [])
        if region_keys:
            regions = evidence.get("regions") or {}
            min_rr = float(spec.get("min_region_rel", 0.10))
            from ml.ftir_interpretable_features import INTERPRETABLE_REGIONS

            reg_bounds = {name: (lo, hi) for name, lo, hi in INTERPRETABLE_REGIONS}
            for rk in region_keys:
                block = regions.get(rk) or {}
                rel = float(block.get("rel_max", 0) or 0)
                if rel < min_rr:
                    continue
                lo, hi = reg_bounds.get(rk, (0.0, 0.0))
                pseudo = {
                    "band_id": f"region_{rk}",
                    "label": rk,
                    "support_score": rel,
                    "region_rel_max": rel,
                    "region_min_cm1": lo,
                    "region_max_cm1": hi,
                    "mode": "Regional spectral activity",
                    "matched": True,
                    "peaks_near": [],
                }
                req_hits.append(pseudo)
                req_scores.append(rel)
            if region_keys and req_scores and not required_any_groups and not required:
                req_frac = 1.0

        if not required_any_groups and not required and not region_keys:
            req_frac = 1.0 if sup_hits else 0.0

        sup_bonus = min(0.25, 0.08 * len(sup_hits))
        mean_req = float(sum(req_scores) / len(req_scores)) if req_scores else 0.0
        n_req_effective = len(required_any_groups) if required_any_groups else len(required)
        if required_any_groups or required:
            score = float(np.clip(req_frac * 0.55 + mean_req * 0.35 + sup_bonus, 0.0, 1.0))
            if n_req_effective > 1 and req_frac < 1.0:
                score = float(min(score, 0.40 * req_frac))
        else:
            score = float(np.clip(mean_req * 0.5 + sup_bonus, 0.0, 1.0))

        min_score = float(spec.get("min_score", 0.25))
        if score < min_score * 0.5:
            confidence: Confidence = "low"
        elif score >= min_score and req_frac >= 1.0:
            confidence = "high" if score >= min_score + 0.15 else "medium"
        elif score >= min_score * 0.75:
            confidence = "medium"
        else:
            confidence = "low"

        lib = load_band_library()
        supporting_bands = []
        supporting_band_ids: list[str] = []
        for m in req_hits + sup_hits:
            bid = str(m.get("band_id", ""))
            if bid:
                supporting_band_ids.append(bid)
            b = next((x for x in lib if x["id"] == m["band_id"]), None)
            if b:
                supporting_bands.append(
                    f"{format_band_row(b)} (support {float(m.get('support_score', 0)):.2f})"
                )
        supporting_peaks = []
        for m in req_hits + sup_hits:
            for p in m.get("peaks_near") or []:
                supporting_peaks.append(f"{p.get('wn_cm1', 0):.0f} cm⁻¹")

        cautions: list[str] = []
        for gate in spec.get("ratio_gates") or []:
            rname = str(gate.get("ratio", ""))
            rmin = float(gate.get("min", 0))
            rval = float(ratios.get(rname, 0))
            if rval < rmin:
                score = float(min(score, rmin * 0.5))
                cautions.append(
                    f"Ratio {rname}={rval:.2f} below expected minimum {rmin:.2f} for {label}."
                )
        if spec.get("caution_template"):
            cautions.append(str(spec["caution_template"]))
        if missing:
            cautions.append("Some expected band regions lack clear support.")
        if required and req_frac < 1.0:
            cautions.append("Not all required band criteria are met.")

        conflicting: list[str] = []
        for clab in spec.get("conflicts") or []:
            other = assignments.get(clab)
            if other and float(other.get("score", 0)) > score + 0.1:
                conflicting.append(f"Higher evidence for {clab} (score {other['score']:.2f})")

        evidence_lines = []
        for m in req_hits + sup_hits:
            lo, hi = m["region_min_cm1"], m["region_max_cm1"]
            evidence_lines.append(
                f"{m.get('mode', 'band')} in {lo:.0f}-{hi:.0f} cm⁻¹ (support {m.get('support_score', 0):.2f})"
            )

        if score >= min_score:
            summary = (
                f"Spectral evidence supports {label} (score {score:.2f}, {confidence} confidence). "
                + "; ".join(evidence_lines[:3])
                + ("." if evidence_lines else "")
            )
        else:
            summary = (
                f"Insufficient spectral evidence for {label} (score {score:.2f}). "
                "Assignment is tentative if cited."
            )

        row: dict[str, Any] = {
            "score": round(score, 4),
            "confidence": confidence,
            "supporting_bands": supporting_bands,
            "supporting_band_ids": list(dict.fromkeys(supporting_band_ids)),
            "supporting_peaks": supporting_peaks[:8],
            "missing_expected_bands": missing,
            "conflicting_evidence": conflicting,
            "caution_flags": cautions,
            "human_readable_summary": summary,
            "evidence": evidence_lines,
        }
        oc = spec.get("ontology_category")
        if oc:
            row["ontology_category"] = str(oc)
        elif str(cfg.get("ontology") or "v3").lower() == "v4" and label in _FG_RULES:
            row["ontology_category"] = "specific_fg"
        assignments[label] = row

    _apply_post_rules(assignments, cfg)

    mode = str(cfg.get("guardrails_mode", "v2") or "v2").lower()
    ambiguity_labels: list[dict[str, Any]] = []
    guardrails_diagnostics: list[dict[str, Any]] = []
    guardrails_version = mode
    if mode == "v3":
        from ml.ftir_guardrails import apply_v3_guardrails, annotate_non_guarded_labels

        extra = apply_v3_guardrails(
            assignments,
            evidence,
            ontology=str(cfg.get("ontology") or "v3"),
            suppress_nitro_reporting=bool(cfg.get("suppress_nitro_reporting")),
        )
        ambiguity_labels = list(extra.get("ambiguity_labels") or [])
        guardrails_diagnostics = list(extra.get("diagnostics") or [])
        guardrails_version = str(extra.get("version") or "v3_guarded")
        annotate_non_guarded_labels(assignments, evidence)
    elif mode == "v2":
        from ml.ftir_guardrails import annotate_non_guarded_labels

        annotate_non_guarded_labels(assignments, evidence)

    from ml.ftir_guardrails import attach_interpretation_strength_v4

    attach_interpretation_strength_v4(assignments, ontology=str(cfg.get("ontology") or "v3"))

    return {
        "assignments": assignments,
        "top_labels": sorted(
            assignments.items(),
            key=lambda kv: -float(kv[1].get("score", 0)),
        )[:12],
        "ambiguity_labels": ambiguity_labels,
        "guardrails_diagnostics": guardrails_diagnostics,
        "guardrails_version": guardrails_version,
        "ontology": str(cfg.get("ontology") or "v3").lower(),
    }
