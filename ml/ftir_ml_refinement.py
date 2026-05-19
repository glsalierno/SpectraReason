"""Optional ML refinement layer — secondary to rule-based evidence assignment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ml.structural_fg_svm import predict_proba_row

FusionMode = Literal["annotate", "weighted", "gate", "ml_only"]
AgreementStatus = Literal[
    "rule_and_ml_agree",
    "ml_only_warning",
    "rule_only",
    "conflict",
    "unknown_model_label",
    "insufficient_evidence",
]


def _map_ml_label_to_rule(ml_label: str) -> list[str]:
    """Map SVM label names to rule-assignment keys."""
    lab = ml_label.lower().strip()
    mapping = {
        "alcohol": ["alcohol"],
        "alcohol_or_phenol": ["alcohol", "phenol"],
        "amine": ["primary_amine", "secondary_amine", "tertiary_amine"],
        "amine_or_amide": ["primary_amine", "secondary_amine", "tertiary_amine", "amide"],
        "carbonyl": ["ketone", "aldehyde", "ester", "carboxylic_acid", "amide"],
        "aromatic": ["aromatic", "heteroaromatic"],
        "halide": [],
        "nitro": ["nitro"],
        "nitrile": ["nitrile"],
        "alkene": ["alkene"],
        "alkyne": ["alkyne"],
        "ether": ["ether", "aryl_ether"],
        "ester": ["ester"],
        "carboxylic_acid": ["carboxylic_acid"],
        "ether_or_ester": ["ether", "aryl_ether", "ester"],
        "silicone_or_siloxane": ["siloxane", "silicone_or_silane"],
        "silicon_oxygen": ["siloxane", "silicone_or_silane"],
        "hydroxy_containing": ["hydroxy_containing", "alcohol", "phenol"],
        "carbonyl_containing": ["carbonyl_containing", "ketone", "aldehyde", "ester", "amide", "carboxylic_acid"],
        "nitrogen_containing": ["nitrogen_containing", "primary_amine", "secondary_amine", "tertiary_amine", "amide", "nitrile", "nitro"],
        "aromatic_system": ["aromatic_system", "aromatic", "heteroaromatic"],
        "C_O_containing": ["C_O_containing", "ether", "aryl_ether", "ester"],
        "unsaturation_possible": ["unsaturation_possible", "alkene", "alkyne", "nitrile"],
        "silicon_oxygen_family": ["siloxane", "silicone_or_silane", "silicon_oxygen_family"],
    }
    return mapping.get(lab, [lab])


def _ml_display_name(model_bundle: dict[str, Any] | None) -> str:
    """Human-readable ML column label for reports (matches saved calibration metadata)."""
    if model_bundle is None:
        return "SVM score"
    meta = model_bundle.get("meta") or {}
    cal = model_bundle.get("calibration") or meta.get("calibration") or {}
    fitted = bool(cal.get("fitted")) if isinstance(cal, dict) else False
    mk = str(model_bundle.get("ml_score_kind") or meta.get("ml_score_kind") or "").lower()
    if mk == "calibrated_probability" and fitted:
        return "calibrated SVM probability"
    if mk == "calibrated_probability" and not fitted:
        return "SVM score"
    return "SVM score"


def refine_assignments_with_ml(
    rule_assignments: dict[str, Any],
    evidence: dict[str, Any],
    *,
    wn: Any = None,
    y: Any = None,
    md: dict[str, Any] | None = None,
    model_bundle: dict[str, Any] | None = None,
    model_kind: str = "basic",
    fusion_mode: FusionMode = "annotate",
    alpha: float = 0.65,
    beta: float = 0.35,
    gate_min_rule_score: float = 0.15,
    ml_threshold: float = 0.45,
    agree_delta: float = 0.12,
    ml_guardrails: Literal["strict", "moderate", "off"] = "strict",
    ontology: str | None = None,
) -> dict[str, Any]:
    """
    Merge rule scores with optional ML probabilities. Default ``annotate`` keeps rules primary.
    """
    md = md or {}
    onto = str(ontology or rule_assignments.get("ontology") or "v3").lower()
    base = dict(rule_assignments.get("assignments") or {})
    all_labels = set(base.keys())

    ml_probs: dict[str, float] = {}
    if model_bundle is not None and wn is not None and y is not None:
        try:
            ml_probs = predict_proba_row(model_bundle, wn=wn, y=y, md=md)
        except Exception as exc:
            return {
                "fusion_mode": fusion_mode,
                "model_kind": model_kind,
                "ml_error": str(exc),
                "per_label": {},
                "ml_probabilities": {},
            }
        for ml_lab in ml_probs:
            for rlab in _map_ml_label_to_rule(ml_lab):
                all_labels.add(rlab)

    from ml.ftir_ontology import ONTOLOGY_V4, is_v4

    ml_unknown_heads = [k for k in ml_probs if is_v4(onto) and k not in ONTOLOGY_V4]

    ml_display = _ml_display_name(model_bundle)
    ml_is_proba = "probability" in ml_display.lower()
    thr_map: dict[str, float] = {}
    if model_bundle:
        thr_map = (model_bundle.get("meta") or {}).get("per_label_thresholds") or {}

    per_label: dict[str, Any] = {}
    for lab in sorted(all_labels):
        rule_entry = dict(base.get(lab) or {})
        rule_score = float(rule_entry.get("score", 0.0))

        ml_raw = 0.0
        best_ml = ""
        for ml_lab, p in ml_probs.items():
            if lab in _map_ml_label_to_rule(ml_lab):
                pv = float(p)
                if pv > ml_raw:
                    ml_raw = pv
                    best_ml = str(ml_lab)

        if is_v4(onto):
            ocat = str(rule_entry.get("ontology_category") or "")
            if ocat in ("local_motif", "artifact"):
                ml_raw = 0.0
                best_ml = ""

        ml_thr = float(thr_map.get(lab, ml_threshold if ml_is_proba else 0.0))
        ml_p = ml_raw
        if ml_is_proba:
            ml_high = ml_raw >= ml_thr
        else:
            ml_high = ml_raw > 0.0

        if fusion_mode == "ml_only":
            final_score = ml_p
        elif fusion_mode == "weighted":
            w_ml = ml_p if ml_is_proba else float(np.tanh(ml_p))
            final_score = alpha * rule_score + beta * w_ml
        elif fusion_mode == "gate":
            if rule_score < gate_min_rule_score:
                final_score = rule_score
            else:
                w_ml = ml_p if ml_is_proba else float(np.tanh(ml_p))
                final_score = max(rule_score, min(w_ml, rule_score + 0.25))
        else:
            final_score = rule_score

        cls = str(rule_entry.get("confidence_class") or "")
        evc = str(rule_entry.get("evidence_completeness") or "complete")
        weak_cls = cls in ("tentative", "local_possible", "not_supported") and cls != ""
        weak_ev = evc in ("partial", "single_band", "conflicting", "artifact_limited")
        mg = str(ml_guardrails or "strict").lower()
        if mg == "strict" and fusion_mode in ("weighted", "gate", "ml_only") and (weak_cls or weak_ev):
            cap = min(0.58, max(rule_score + 0.05, 0.42))
            final_score = min(float(final_score), float(cap))
        elif mg == "moderate" and fusion_mode in ("weighted", "gate") and weak_ev and not weak_cls:
            final_score = min(float(final_score), max(rule_score + 0.18, 0.62))

        rule_high = rule_score >= ml_threshold
        agreement: AgreementStatus = "insufficient_evidence"
        if rule_high and ml_high:
            if ml_is_proba:
                agreement = "rule_and_ml_agree" if abs(rule_score - ml_p) <= agree_delta else "conflict"
            else:
                agreement = "rule_and_ml_agree"
        elif (not rule_high) and ml_high and rule_score < gate_min_rule_score:
            agreement = "ml_only_warning"
        elif rule_high and (not ml_high):
            agreement = "rule_only"

        cautions = list(rule_entry.get("caution_flags") or [])

        if is_v4(onto) and best_ml and best_ml not in ONTOLOGY_V4 and ml_high:
            agreement = "unknown_model_label"  # type: ignore[assignment]
            cautions.append(f"ML head {best_ml!r} is not in the v4 ontology; treat output as exploratory.")

        if agreement == "ml_only_warning":
            cautions.append(
                "ML-only warning: the model output is comparatively high while spectral rule evidence is weak; "
                "manual band review is required (FTIR evidence remains primary)."
            )
        if agreement == "conflict":
            cautions.append("Rule score and ML output disagree; assignment is tentative.")

        per_label[lab] = {
            "rule_score": round(rule_score, 4),
            "ml_score": round(ml_p, 6),
            "ml_score_label": ml_display,
            "ml_probability": round(ml_p, 4) if model_bundle and ml_is_proba else None,
            "final_score": round(float(final_score), 4),
            "agreement_status": agreement,
            "confidence": rule_entry.get("confidence", "low"),
            "supporting_bands": rule_entry.get("supporting_bands", []),
            "supporting_peaks": rule_entry.get("supporting_peaks", []),
            "missing_expected_bands": rule_entry.get("missing_expected_bands", []),
            "conflicting_evidence": rule_entry.get("conflicting_evidence", []),
            "caution_flags": cautions,
            "human_readable_summary": _summary_text(
                lab,
                rule_entry,
                rule_score,
                ml_p,
                agreement,
                ml_display=ml_display,
            ),
        }

    if is_v4(onto) and str(ml_guardrails or "strict").lower() != "off":
        apply_ml_advisory_soft_gates(
            per_label,
            evidence,
            base,
            ml_probs=ml_probs,
            ml_guardrails=str(ml_guardrails or "strict"),
            thr_map=thr_map,
            ml_threshold=ml_threshold,
            ml_is_proba=ml_is_proba,
        )

    if is_v4(onto):
        silicon_like = ("siloxane", "silicone_or_silane", "silicon_oxygen_family")
        for lab in silicon_like:
            pl = per_label.get(lab)
            if not pl:
                continue
            rs = float(pl.get("rule_score", 0))
            ml_raw = float(pl.get("ml_score", 0) or 0)
            if ml_raw >= 0.55 and rs < 0.22:
                pl["agreement_status"] = "ml_only_warning"
                pl["final_score"] = round(min(float(pl.get("final_score", 0)), rs + 0.1, 0.32), 4)
                sc2 = (
                    "Silicon ML output elevated with weak spectral siloxane evidence; "
                    "final score not promoted above tentative."
                )
                if sc2 not in (pl.get("caution_flags") or []):
                    pl.setdefault("caution_flags", []).append(sc2)

    return {
        "fusion_mode": fusion_mode,
        "model_kind": model_kind,
        "ml_probabilities": ml_probs,
        "ml_score_display": ml_display,
        "ml_guardrails": ml_guardrails,
        "ontology": onto,
        "ml_unknown_heads": ml_unknown_heads,
        "per_label": per_label,
        "top_consensus": sorted(
            per_label.items(),
            key=lambda kv: -float(kv[1].get("final_score", 0)),
        )[:12],
    }


# High-risk labels: minimum evidence groups (from ftir_guardrails) before ML may promote scores.
_ML_GATE_REQUIRED: dict[str, list[str]] = {
    "phenol": ["phenol_oh_g", "aromatic_cc_g", "phenolic_co_g"],
    "alcohol": ["alcohol_oh_g", "ether_co_g"],
    "amide": ["amide_co_g", "amide_nh_or_ii_g"],
    "ester": ["ester_co_g", "ester_coo_g"],
    "ether": ["ether_co_g"],
    "aryl_ether": ["aryl_ether_co_g", "aromatic_cc_g"],
    "siloxane": ["siloxane_sio_g"],
    "silicone_or_silane": ["siloxane_sio_g"],
    "nitro": ["nitro_asym_g", "nitro_sym_g"],
    "nitrile": ["nitrile_cn_g"],
    "alkyne": ["alkyne_cc_g"],
    "heteroaromatic": ["hetero_g", "aromatic_cc_g"],
    "carboxylic_acid": ["acid_co_g", "acid_oh_g"],
    "carbonate": ["carbonate_co_g"],
    "urethane": ["urethane_co_g"],
    "pyrrole_like_NH": ["pyrrole_nh_g"],
    "cyclic_amine": ["cyclic_nh_g"],
}


def _evidence_group_support(evidence: dict[str, Any], group_key: str, *, min_support: float = 0.08) -> float:
    from ml.ftir_guardrails import _group_satisfied, _match_map

    ok, sc = _group_satisfied(_match_map(evidence), group_key, min_support=min_support)
    return sc if ok else 0.0


def _evidence_support_tier(
    evidence: dict[str, Any],
    label: str,
) -> Literal["absent", "partial", "supported"]:
    groups = _ML_GATE_REQUIRED.get(label)
    if not groups:
        return "supported"
    scores = [_evidence_group_support(evidence, g) for g in groups]
    n_ok = sum(1 for s in scores if s >= 0.08)
    if n_ok == 0:
        return "absent"
    if n_ok < len(groups):
        return "partial"
    return "supported"


def apply_ml_advisory_soft_gates(
    per_label: dict[str, Any],
    evidence: dict[str, Any],
    rule_assignments: dict[str, Any],
    *,
    ml_probs: dict[str, float],
    ml_guardrails: str,
    thr_map: dict[str, float],
    ml_threshold: float,
    ml_is_proba: bool,
) -> None:
    """
    Cap ML-influenced scores when spectral evidence is weak (strict/moderate).
    Rules remain primary; does not remove rule assignments.
    """
    mg = str(ml_guardrails or "strict").lower()
    for lab, pl in per_label.items():
        if lab not in _ML_GATE_REQUIRED:
            continue
        tier = _evidence_support_tier(evidence, lab)
        ml_raw = float(pl.get("ml_score", 0) or 0)
        ml_thr = float(thr_map.get(lab, ml_threshold if ml_is_proba else 0.0))
        ml_high = (ml_raw >= ml_thr) if ml_is_proba else (ml_raw > 0.0)
        if not ml_high:
            continue
        rs = float(pl.get("rule_score", 0) or 0)
        cap_supported = 0.92
        cap_tentative = 0.58 if mg == "strict" else 0.68
        cap_local = 0.42 if mg == "strict" else 0.52
        if tier == "supported":
            cap = cap_supported
        elif tier == "partial":
            cap = cap_tentative
        else:
            cap = cap_local
        new_final = min(float(pl.get("final_score", rs)), cap)
        if tier != "supported" and ml_high and rs < 0.22:
            pl["agreement_status"] = "ml_only_warning"
            msg = (
                f"ML advisory for {lab}: elevated model score with "
                f"{'no' if tier == 'absent' else 'incomplete'} spectral evidence — not promoted above tentative."
            )
            if msg not in (pl.get("caution_flags") or []):
                pl.setdefault("caution_flags", []).append(msg)
        pl["final_score"] = round(new_final, 4)
        pl["evidence_support_tier"] = tier

    # Silicon: require two Si regions for supported ML promotion
    from ml.ftir_guardrails import _match_map, _silicon_evidence_region_count

    n_si = _silicon_evidence_region_count(_match_map(evidence), 0.08)
    for lab in ("siloxane", "silicone_or_silane"):
        pl = per_label.get(lab)
        if not pl:
            continue
        if n_si < 2:
            pl["final_score"] = round(min(float(pl.get("final_score", 0)), 0.25), 4)
            pl["agreement_status"] = "ml_only_warning"
            pl.setdefault(
                "caution_flags",
                [],
            ).append("ML cannot promote silicon above tentative without two silicon-related regions.")
            if n_si < 1:
                pl["evidence_support_tier"] = "absent"
            else:
                pl["evidence_support_tier"] = "partial"


def _summary_text(
    label: str,
    rule_entry: dict[str, Any],
    rule_score: float,
    ml_p: float,
    agreement: str,
    *,
    ml_display: str,
) -> str:
    parts: list[str] = []
    if rule_score >= 0.2:
        parts.append(
            f"Spectral rule evidence provides partial support for {label} (rule score {rule_score:.2f}); "
            "this is not definitive proof of presence."
        )
    if ml_p != 0 and "probability" in ml_display.lower():
        parts.append(
            f"The secondary ML stage reports a comparatively high {ml_display.split('(')[0].strip().lower()} "
            f"for related model heads (~{ml_p:.2f} on a 0–1 scale where fitted)."
        )
    elif ml_p != 0:
        parts.append(
            f"The secondary ML stage outputs a comparatively large SVM margin score ({ml_p:.3f}); "
            "this is not a calibrated probability."
        )
    if agreement == "rule_and_ml_agree":
        parts.append("Evidence-first rules and the optional ML stage broadly agree; interpretation remains tentative.")
    elif agreement == "ml_only_warning":
        parts.append(
            "Caution: ML output is elevated while FTIR band evidence is weak — treat as ML-only flag, not a spectral assignment."
        )
    elif agreement == "conflict":
        parts.append("Rule and ML signals disagree; manual review is recommended.")
    base = rule_entry.get("human_readable_summary", "")
    blob = " ".join(parts)
    if base and base not in blob:
        blob = f"{blob} {base}"
    return blob.strip()
