"""Tests for v4 evidence-first report helpers and silicon guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_rules import assign_functional_groups_from_evidence
from reports.v4_evidence_report import (
    SCORE_TOOLTIP,
    assignment_evidence_rank,
    build_band_evidence_map_html,
    build_evidence_first_assignments_table_html,
    evidence_ranked_assignments,
)


def _bm(band_id: str, *, matched: bool = True, support: float = 0.5, wn: float = 1080.0) -> dict:
    return {
        "band_id": band_id,
        "label": band_id,
        "subclass": band_id,
        "region_min_cm1": wn - 40,
        "region_max_cm1": wn + 40,
        "mode": band_id,
        "importance": "required",
        "specificity": "medium",
        "region_rel_max": support,
        "peak_support": support,
        "support_score": support,
        "matched": matched,
        "peaks_near": [{"wn_cm1": wn, "rel_height": 0.4}],
    }


def test_siloxane_capped_without_paired_si() -> None:
    ev = {
        "band_matches": [
            _bm("siloxane_sio", support=0.4, wn=1080),
            _bm("ether_co", support=0.55, wn=1280),
            _bm("aryl_ether_co", support=0.45, wn=1250),
        ],
        "peaks": [{"wn_cm1": 1080, "height": 0.5, "rel_height": 0.4}],
        "ratios": {"siloxane_to_c_o": 0.15},
        "regions": {},
        "artifacts": {"flags": {"fingerprint_crowding": True}, "cautions": [], "summary": "fingerprint_crowding"},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3", "ontology": "v4"})
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) <= 0.25
    assert sil["confidence_class"] in ("local_possible", "not_supported")
    assert sil.get("assignment_type") in ("local_band_only", "artifact_limited")


def test_evidence_rank_prefers_supported_over_high_tentative() -> None:
    pipeline = {
        "rule_assignments": {
            "assignments": {
                "phenol": {"score": 0.48, "confidence_class": "tentative", "evidence_completeness": "artifact_limited"},
                "aromatic": {"score": 0.62, "confidence_class": "supported", "evidence_completeness": "complete"},
            }
        }
    }
    ranked = evidence_ranked_assignments(pipeline, min_score=0.1, top_n=2)
    assert ranked[0][0] == "aromatic"


def test_assignments_table_uses_evidence_labels() -> None:
    pipeline = {
        "ontology": "v4",
        "rule_assignments": {"assignments": {"phenol": {"score": 0.48, "confidence_class": "tentative"}}},
        "consensus": {"per_label": {}},
        "ml_refinement": {},
    }
    html = build_evidence_first_assignments_table_html(pipeline, top_n=5)
    assert "Evidence score" in html
    assert "Report ranking score" in html
    assert SCORE_TOOLTIP[:20] in html
    assert "not probabilities" in html


def test_band_evidence_map_renders_rows() -> None:
    pipeline = {
        "evidence": {
            "band_matches": [_bm("phenolic_co", wn=1284)],
            "fg_evidence": {},
        },
        "rule_assignments": {
            "assignments": {
                "phenol": {
                    "score": 0.55,
                    "confidence_class": "supported",
                    "supporting_bands": ["phenolic_co"],
                }
            }
        },
        "canonical_peaks": {
            "evidence_rows": [
                {
                    "peak_cm1": 1284,
                    "band_region": "1260–1310 cm⁻¹",
                    "possible_functional_groups": ["phenol"],
                    "band_id": "phenolic_co",
                    "peak_id": "p1",
                    "source": "observed_peak",
                }
            ],
            "peak_by_id": {"p1": {"center_cm1": 1284.0, "quality_class": "sharp"}},
        },
    }
    html = build_band_evidence_map_html(pipeline, audience="debug")
    assert "1284" in html
    assert "Band evidence map" in html
