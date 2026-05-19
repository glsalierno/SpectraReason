"""
Evidence-first FTIR functional-group pipeline orchestrator.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ml.ftir_evidence import extract_spectral_evidence
from ml.ftir_ml_refinement import refine_assignments_with_ml
from ml.ftir_rule_config import evidence_config_from_rules, rules_assign_config_from_rules
from ml.ftir_rules import assign_functional_groups_from_evidence

MlMode = Literal["none", "basic", "subtle", "both", "legacy"]
FusionMode = Literal["annotate", "weighted", "gate", "ml_only"]


def run_evidence_first_pipeline(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    md: dict[str, Any] | None = None,
    peaks: list[dict[str, Any]] | None = None,
    ml_mode: MlMode = "none",
    fusion_mode: FusionMode = "annotate",
    basic_model: dict[str, Any] | None = None,
    subtle_model: dict[str, Any] | None = None,
    legacy_model: dict[str, Any] | None = None,
    evidence_config: dict[str, Any] | None = None,
    rules_config: dict[str, Any] | None = None,
    guardrails_mode: str = "v2",
    ml_guardrails: Literal["strict", "moderate", "off"] = "strict",
    measurement_mode: str | None = None,
    atr_crystal: str | None = None,
    atr_aware: bool | None = None,
    spectrum_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: evidence → rules → optional ML → consensus.

    Returns structured dict suitable for reports and JSON export.
    """
    md = md or {}
    ev_cfg = {**evidence_config_from_rules(rules_config), **(evidence_config or {})}
    ru_cfg = rules_assign_config_from_rules(rules_config)
    if rules_config and rules_config.get("ontology") is not None:
        on = str(rules_config.get("ontology") or "v3").lower()
        ru_cfg["ontology"] = on
        ev_cfg.setdefault("ontology", on)
    if rules_config and rules_config.get("label_overrides"):
        ru_cfg["label_overrides"] = {
            **(ru_cfg.get("label_overrides") or {}),
            **rules_config["label_overrides"],
        }
    if rules_config and rules_config.get("post_rules"):
        ru_cfg["post_rules"] = rules_config["post_rules"]

    from ml.ftir_atr import merge_measurement_into_metadata, resolve_atr_context

    measurement = resolve_atr_context(
        path=spectrum_path,
        md=md,
        mode=measurement_mode or ev_cfg.get("measurement_mode"),
        atr_crystal=atr_crystal or ev_cfg.get("atr_crystal"),
        atr_aware=atr_aware if atr_aware is not None else ev_cfg.get("atr_aware"),
    )
    md = merge_measurement_into_metadata(md, measurement)
    evidence = extract_spectral_evidence(wn, y, peaks=peaks, config=ev_cfg)
    evidence["measurement"] = measurement
    try:
        from ml.ftir_artifacts import detect_spectral_artifacts

        evidence["artifacts"] = detect_spectral_artifacts(wn, y, evidence, measurement=measurement)
    except Exception:
        evidence["artifacts"] = {"flags": {}, "cautions": [], "summary": "none"}

    if ev_cfg.get("compute_deconv") or ev_cfg.get("include_deconv_for_guardrails"):
        try:
            from ml.ftir_deconvolution import deconv_to_evidence_dict, deconvolve_spectrum

            evidence["deconv"] = deconv_to_evidence_dict(deconvolve_spectrum(wn, y))
        except Exception:
            evidence.setdefault("deconv", {"regions": {}, "disclaimer": "deconv unavailable"})

    ru_cfg["guardrails_mode"] = str(guardrails_mode or "v2").lower()
    rule_result = assign_functional_groups_from_evidence(evidence, config=ru_cfg)

    ml_basic: dict[str, Any] | None = None
    ml_subtle: dict[str, Any] | None = None
    ml_legacy: dict[str, Any] | None = None
    warnings_list: list[str] = []

    if ml_mode in ("basic", "both", "legacy") and basic_model is None and legacy_model is None:
        warnings_list.append("basic/legacy ML requested but no model loaded; evidence-only for ML refinement.")
    if ml_mode in ("subtle", "both") and subtle_model is None:
        warnings_list.append("subtle ML requested but no subtle model loaded; skipping subtle refinement.")

    basic_bundle = basic_model or (
        legacy_model if ml_mode in ("basic", "both") and basic_model is None else None
    )
    if basic_bundle is not None and basic_model is None and legacy_model is not None and ml_mode in ("basic", "both"):
        warnings_list.append(
            "Broad-label SVM: using legacy model bundle for basic ML refinement (same artifact as v7)."
        )

    basic_kind = str((basic_bundle or {}).get("model_kind") or (basic_bundle or {}).get("meta", {}).get("model_kind") or "basic")
    subtle_kind = str((subtle_model or {}).get("model_kind") or (subtle_model or {}).get("meta", {}).get("model_kind") or "subtle")

    if ml_mode in ("basic", "both") and basic_bundle is not None:
        ml_basic = refine_assignments_with_ml(
            rule_result,
            evidence,
            wn=wn,
            y=y,
            md=md,
            model_bundle=basic_bundle,
            model_kind=basic_kind,
            fusion_mode=fusion_mode,
            ml_guardrails=ml_guardrails,
            ontology=str(rule_result.get("ontology") or "v3"),
        )
    elif ml_mode == "legacy" and legacy_model is not None:
        ml_legacy = refine_assignments_with_ml(
            rule_result,
            evidence,
            wn=wn,
            y=y,
            md=md,
            model_bundle=legacy_model,
            model_kind="legacy",
            fusion_mode=fusion_mode,
            ml_guardrails=ml_guardrails,
            ontology=str(rule_result.get("ontology") or "v3"),
        )

    if ml_mode in ("subtle", "both") and subtle_model is not None:
        ml_subtle = refine_assignments_with_ml(
            rule_result,
            evidence,
            wn=wn,
            y=y,
            md=md,
            model_bundle=subtle_model,
            model_kind=subtle_kind,
            fusion_mode=fusion_mode,
            ml_guardrails=ml_guardrails,
            ontology=str(rule_result.get("ontology") or "v3"),
        )

    consensus = _build_consensus(rule_result, ml_basic, ml_subtle, ml_legacy, fusion_mode=fusion_mode)

    return {
        "evidence": evidence,
        "rule_assignments": rule_result,
        "ml_refinement": {
            "basic": ml_basic,
            "subtle": ml_subtle,
            "legacy": ml_legacy,
        },
        "consensus": consensus,
        "ml_mode": ml_mode,
        "fusion_mode": fusion_mode,
        "rules_config_description": (rules_config or {}).get("description", ""),
        "warnings": warnings_list,
        "guardrails_mode": guardrails_mode,
        "ml_guardrails": ml_guardrails,
        "ontology": str((rule_result or {}).get("ontology") or ru_cfg.get("ontology") or "v3").lower(),
        "measurement": measurement,
    }


def _build_consensus(
    rule_result: dict[str, Any],
    ml_basic: dict[str, Any] | None,
    ml_subtle: dict[str, Any] | None,
    ml_legacy: dict[str, Any] | None,
    *,
    fusion_mode: FusionMode,
) -> dict[str, Any]:
    labels: set[str] = set((rule_result.get("assignments") or {}).keys())
    for block in (ml_basic, ml_subtle, ml_legacy):
        if block:
            labels.update((block.get("per_label") or {}).keys())

    per_label: dict[str, Any] = {}
    for lab in labels:
        rule_score = float((rule_result.get("assignments") or {}).get(lab, {}).get("score", 0))
        ml_b = float(
            (((ml_basic or {}).get("per_label") or {}).get(lab, {}).get("ml_probability"))
            or (((ml_basic or {}).get("per_label") or {}).get(lab, {}).get("ml_score"))
            or 0
        )
        ml_s = float(
            (((ml_subtle or {}).get("per_label") or {}).get(lab, {}).get("ml_probability"))
            or (((ml_subtle or {}).get("per_label") or {}).get(lab, {}).get("ml_score"))
            or 0
        )
        ml_l = float(
            (((ml_legacy or {}).get("per_label") or {}).get(lab, {}).get("ml_probability"))
            or (((ml_legacy or {}).get("per_label") or {}).get(lab, {}).get("ml_score"))
            or 0
        )
        ml_max = max(ml_b, ml_s, ml_l)

        if fusion_mode == "ml_only" and ml_max > 0:
            final = ml_max
        elif fusion_mode == "weighted" and ml_max > 0:
            final = 0.65 * rule_score + 0.35 * ml_max
        else:
            ref = ml_basic or ml_legacy or ml_subtle
            if ref and lab in (ref.get("per_label") or {}):
                final = float(ref["per_label"][lab].get("final_score", rule_score))
            else:
                final = rule_score

        rent = (rule_result.get("assignments") or {}).get(lab, {}) or {}
        per_label[lab] = {
            "rule_score": rule_score,
            "ml_probability_basic": ml_b or None,
            "ml_probability_subtle": ml_s or None,
            "ml_probability_legacy": ml_l or None,
            "final_score": round(final, 4),
            "agreement_status": _consensus_agreement(rule_score, ml_max),
            "confidence_class": rent.get("confidence_class"),
            "evidence_completeness": rent.get("evidence_completeness"),
            "assignment_type": rent.get("assignment_type"),
        }

    onto = str((rule_result or {}).get("ontology") or "v3").lower()

    def _rank_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float]:
        lab, ent = item
        rent = (rule_result.get("assignments") or {}).get(lab, {}) or {}
        cat = str(rent.get("ontology_category") or rent.get("assignment_type") or "")
        if onto == "v4":
            from ml.ftir_ontology import NON_TRAINABLE_V4, TRAINABLE_SPECIFIC_V4

            if lab in NON_TRAINABLE_V4 or cat == "local_motif" or cat == "artifact":
                tier = 3
            elif cat == "specific_fg" or lab in TRAINABLE_SPECIFIC_V4:
                tier = 0
            elif cat in ("family", "fallback") or lab.endswith("_containing") or lab.endswith("_family"):
                tier = 1
            else:
                tier = 2
        else:
            tier = 1 if cat in ("fallback", "family") else 0
        return (tier, -float(ent.get("final_score", 0)))

    top_sorted = sorted(per_label.items(), key=_rank_key)[:12]
    return {
        "per_label": per_label,
        "top_labels": top_sorted,
    }


def _consensus_agreement(rule_score: float, ml_p: float) -> str:
    if rule_score >= 0.35 and ml_p >= 0.45 and abs(rule_score - ml_p) <= 0.15:
        return "rule_and_ml_agree"
    if rule_score < 0.15 and ml_p >= 0.45:
        return "ml_only_warning"
    if rule_score >= 0.35 and ml_p < 0.2:
        return "rule_only"
    if rule_score >= 0.35 and ml_p >= 0.45:
        return "conflict"
    return "insufficient_evidence"


def load_model_bundle(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        warnings.warn(f"Model not found (continuing evidence-only): {p}")
        return None
    import joblib

    return joblib.load(p)
