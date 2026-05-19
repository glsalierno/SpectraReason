"""v3_guarded guardrail behavior (caps, competitors, nitrile sharpness)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_rules import assign_functional_groups_from_evidence


def _bm(
    band_id: str,
    *,
    matched: bool,
    support: float,
    specificity: str = "medium",
) -> dict:
    return {
        "band_id": band_id,
        "label": band_id,
        "subclass": band_id,
        "region_min_cm1": 0,
        "region_max_cm1": 4000,
        "mode": band_id,
        "importance": "required",
        "specificity": specificity,
        "region_rel_max": support,
        "peak_support": support,
        "support_score": support,
        "matched": matched,
        "peaks_near": [],
    }


def test_v3_siloxane_capped_when_ratio_low_and_ether_high() -> None:
    """Organic C-O-rich fingerprint without Si-O dominance should not yield high siloxane."""
    ev = {
        "band_matches": [
            _bm("siloxane_sio", matched=True, support=0.35, specificity="high"),
            _bm("ether_co", matched=True, support=0.55, specificity="low"),
            _bm("aryl_ether_co", matched=True, support=0.45, specificity="medium"),
            _bm("aromatic_cc", matched=True, support=0.5),
        ],
        "peaks": [{"wn_cm1": 1080, "height": 0.5, "rel_height": 0.4, "quality_sharpness": 0.02}],
        "ratios": {"siloxane_to_c_o": 0.15},
        "regions": {},
        "artifacts": {"flags": {"fingerprint_crowding": True}, "cautions": [], "summary": "fingerprint_crowding"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    sil = float(r["assignments"]["siloxane"]["score"])
    assert sil <= 0.25, sil
    assert r["assignments"]["siloxane"]["confidence_class"] in ("local_possible", "not_supported")


def test_v3_nitro_requires_paired_bands() -> None:
    ev = {
        "band_matches": [
            _bm("nitro_asym", matched=True, support=0.5),
            _bm("nitro_sym", matched=False, support=0.05),
        ],
        "peaks": [],
        "ratios": {},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    assert float(r["assignments"]["nitro"]["score"]) < 0.35


def test_v3_nitrile_weak_sharpness_caps() -> None:
    ev = {
        "band_matches": [_bm("nitrile_cn", matched=True, support=0.55)],
        "peaks": [{"wn_cm1": 2240, "height": 0.1, "quality_sharpness": 0.001}],
        "ratios": {},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    assert float(r["assignments"]["nitrile"]["score"]) <= 0.33


def test_v2_skips_v3_diagnostics() -> None:
    ev = {
        "band_matches": [_bm("nitrile_cn", matched=True, support=0.2)],
        "peaks": [],
        "ratios": {},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v2"})
    assert not r.get("guardrails_diagnostics")


def test_v3_amide_requires_nh_or_amide_ii_with_co() -> None:
    """Amide I alone should not yield a confident amide score after rules + v3."""
    ev = {
        "band_matches": [
            _bm("amide_co", matched=True, support=0.62),
            _bm("amide_nh", matched=False, support=0.02),
            _bm("amide_ii", matched=False, support=0.02),
        ],
        "peaks": [],
        "ratios": {},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    assert float(r["assignments"]["amide"]["score"]) < 0.35


def test_v3_competing_explanation_note_and_field() -> None:
    ev = {
        "band_matches": [
            _bm("nitro_asym", matched=True, support=0.5),
            _bm("nitro_sym", matched=False, support=0.05),
            _bm("aromatic_cc", matched=True, support=0.72),
        ],
        "peaks": [],
        "ratios": {},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    nit = r["assignments"].get("nitro") or {}
    flags = " ".join(nit.get("caution_flags") or [])
    assert "Stronger competing explanation detected." in flags


def test_hover_uses_v3_confidence_class_when_present() -> None:
    from ml.ftir_evidence import build_local_hover_context

    rule_assignments = {
        "assignments": {
            "phenol": {
                "score": 0.72,
                "confidence_class": "tentative",
                "confidence": "medium",
            }
        }
    }
    peaks = [{"wn_cm1": 1240.0, "height": 0.5}]
    ctx = build_local_hover_context(
        1240.0,
        0.66,
        peaks,
        band_library=None,
        rule_assignments=rule_assignments,
        ml_assignments=None,
        evidence=None,
        tolerance_cm1=80.0,
    )
    assert "tentative" in ctx.get("hover_text", "")


def test_ml_strict_caps_weighted_mode() -> None:
    from unittest.mock import patch

    from ml.ftir_ml_refinement import refine_assignments_with_ml

    rule = {
        "assignments": {
            "ester": {
                "score": 0.25,
                "confidence": "low",
                "confidence_class": "tentative",
                "evidence_completeness": "partial",
                "supporting_peaks": [],
                "supporting_bands": [],
                "missing_expected_bands": ["x"],
                "caution_flags": [],
                "human_readable_summary": "",
            }
        },
        "top_labels": [],
    }
    ev = {"band_matches": [], "peaks": [], "ratios": {}, "regions": {}}
    bundle = {"meta": {"ml_score_kind": "calibrated_probability", "per_label_thresholds": {"ester": 0.3}}}
    with patch("ml.ftir_ml_refinement.predict_proba_row", return_value={"ester": 0.95}):
        out = refine_assignments_with_ml(
            rule,
            ev,
            wn=np.linspace(4000, 400, 200),
            y=np.random.RandomState(0).rand(200) * 0.1,
            model_bundle=bundle,
            fusion_mode="weighted",
            ml_guardrails="strict",
        )
    fs = float(out["per_label"]["ester"]["final_score"])
    assert fs <= 0.58


def test_silicon_single_sio_region_soft_gated() -> None:
    """One Si-O-like region (no Si-C) + strong ether: siloxane stays low / local_possible, not supported."""
    ev = {
        "band_matches": [
            _bm("siloxane_sio", matched=True, support=0.55, specificity="low"),
            _bm("ether_co", matched=True, support=0.62, specificity="low"),
        ],
        "peaks": [],
        "ratios": {"siloxane_to_c_o": 0.5},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) <= 0.30
    assert sil["confidence_class"] in ("local_possible", "not_supported")


def test_silicon_two_regions_not_single_band_gated() -> None:
    """Si-O-Si + Si-C evidence: silicon soft-gate path skipped (no forced local_possible from that gate)."""
    ev = {
        "band_matches": [
            _bm("siloxane_sio", matched=True, support=0.58, specificity="low"),
            _bm("silicone_sic", matched=True, support=0.48, specificity="medium"),
            _bm("ether_co", matched=True, support=0.30, specificity="low"),
        ],
        "peaks": [],
        "ratios": {"siloxane_to_c_o": 0.7},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) > 0.25
    assert sil["confidence_class"] in ("supported", "strong", "tentative", "local_possible")
