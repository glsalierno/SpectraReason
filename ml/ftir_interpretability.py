"""
Per-spectrum functional-group explanation layer (spectral evidence, not causal claims).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.ftir_band_library import bands_for_label, format_band_row
from ml.ftir_interpretable_features import featurize_interpretable, peaks_near_band
from ml.structural_fg_svm import (
    explain_structural_fg_model,
    predict_proba_row,
    spectral_band_reference,
)


def _label_coef_hints(artifact: dict[str, Any], label: str, top_k: int) -> list[dict[str, Any]]:
    try:
        rep = explain_structural_fg_model(artifact, topk=top_k)
        block = rep.get("per_functional_group", {}).get(label.lower(), {})
        pos = block.get("top_positive_z") or []
        return [{"feature": r["feature"], "coef_z": r["coef_z"], "direction": "positive"} for r in pos[:top_k]]
    except Exception:
        return []


def _region_evidence_from_interp(
    label: str,
    extras: dict[str, Any],
    *,
    prob: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Map interpretable region stats + band library to evidence lists."""
    peaks = extras.get("peaks") or []
    rstats = extras.get("region_stats") or {}
    evidence_regions: list[dict[str, Any]] = []
    evidence_peaks: list[dict[str, Any]] = []
    missing: list[str] = []

    for band in bands_for_label(label):
        lo = float(band["region_min_cm1"])
        hi = float(band["region_max_cm1"])
        near = peaks_near_band(peaks, lo, hi)
        # Heuristic strength from interpretable windows overlapping band
        strength = 0.0
        for rname, st in rstats.items():
            # loose overlap check via region table names
            if any(k in rname for k in ("oh", "carbonyl", "nitrile", "aromatic", "nitro", "si_o", "c_o")):
                strength = max(strength, float(st.get("max", 0.0)))
        entry = {
            "region": format_band_row(band),
            "region_min_cm1": lo,
            "region_max_cm1": hi,
            "mode": band.get("mode"),
            "library_strength": band.get("strength"),
            "library_specificity": band.get("specificity"),
            "observed_region_max": strength,
            "notes": band.get("notes", ""),
        }
        evidence_regions.append(entry)
        if near:
            for p in near[:5]:
                evidence_peaks.append(
                    {
                        "wn_cm1": p["wn"],
                        "height": p["height"],
                        "band_label": label,
                        "region": f"{lo}–{hi}",
                    }
                )
        elif prob >= 0.35 and band.get("strength") in ("strong", "medium"):
            missing.append(f"Expected activity near {lo}–{hi} cm⁻¹ ({band.get('mode')}) not clearly supported by picked peaks.")

    return evidence_regions, evidence_peaks, missing


def explain_prediction(
    wn: np.ndarray,
    y: np.ndarray,
    model_bundle: dict[str, Any],
    *,
    md: dict[str, Any] | None = None,
    top_k: int = 5,
    label: str | None = None,
) -> dict[str, Any]:
    """
    Structured explanation for one spectrum.

    Returns a dict for one label (if ``label`` set) or top-k labels by probability.
    """
    md = md or {}
    probs = predict_proba_row(model_bundle, wn=wn, y=y, md=md)
    if label:
        targets = [label.lower()]
    else:
        targets = [t[0] for t in sorted(probs.items(), key=lambda kv: -kv[1])[: max(1, top_k)]]

    interp_vec, interp_names, extras = featurize_interpretable(wn, y)
    # Top interpretable features by magnitude (local salience without full model Jacobian)
    idx = np.argsort(-np.abs(interp_vec))[:12]
    supporting = [
        {"feature": interp_names[i], "value": float(interp_vec[i])} for i in idx if interp_vec[i] != 0
    ]

    explanations: list[dict[str, Any]] = []
    for lab in targets:
        p = float(probs.get(lab, 0.0))
        ev_reg, ev_pk, missing = _region_evidence_from_interp(lab, extras, prob=p)
        coef_hints = _label_coef_hints(model_bundle, lab, top_k=6)
        caution: list[str] = []
        if p < 0.2:
            caution.append("Low model probability; evidence below is exploratory.")
        if missing:
            caution.append("Some expected band regions lack clear peak support.")
        if not coef_hints:
            caution.append("Global linear coefficients unavailable for this label head.")

        # Human-readable one-liner
        reg_bits = []
        for er in ev_reg[:3]:
            reg_bits.append(er["region"])
        peak_bits = [f"{ep['wn_cm1']:.0f} cm⁻¹" for ep in ev_pk[:3]]
        summary = (
            f"P({lab})={p:.3f}: spectral evidence associated with this prediction includes "
        )
        if reg_bits:
            summary += ", ".join(reg_bits[:2])
        else:
            summary += "weak regional alignment to the band library"
        if peak_bits:
            summary += f"; nearby observed peaks at {', '.join(peak_bits)}"
        summary += ". Interpretation is associative, not proof of functional group presence."

        explanations.append(
            {
                "label": lab,
                "probability": p,
                "evidence_regions": ev_reg,
                "evidence_peaks": ev_pk,
                "supporting_features": supporting,
                "model_coef_hints": coef_hints,
                "missing_expected_features": missing,
                "caution_flags": caution,
                "human_readable_summary": summary,
            }
        )

    return {
        "functional_group_probabilities": probs,
        "model_kind": (model_bundle.get("meta") or {}).get("model_kind", model_bundle.get("model_kind", "unknown")),
        "spectral_windows_reference": spectral_band_reference(),
        "explanations": explanations,
    }


def explain_top_labels_html_cards(explanation_bundle: dict[str, Any]) -> str:
    """Minimal HTML fragment for report embedding."""
    import html as html_mod

    parts = ["<div class='expl-cards'>"]
    for ex in explanation_bundle.get("explanations") or []:
        lab = html_mod.escape(str(ex["label"]))
        p = float(ex["probability"])
        summary = html_mod.escape(str(ex.get("human_readable_summary", "")))
        cautions = ex.get("caution_flags") or []
        chtml = ""
        if cautions:
            chtml = "<ul class='caution'>" + "".join(
                f"<li>{html_mod.escape(str(c))}</li>" for c in cautions
            ) + "</ul>"
        parts.append(
            f"<div class='expl-card'><h4>{lab} <span class='prob'>{p:.3f}</span></h4>"
            f"<p>{summary}</p>{chtml}</motion-div>"
        )
    parts.append("</div>")
    return "".join(parts)
