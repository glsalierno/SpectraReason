"""Nitro / NO₂ report suppression for samples without R–NO₂."""

from __future__ import annotations

from ml.report_suppression import apply_nitro_suppression, mentions_nitro_chemistry, nitro_reporting_suppressed


def test_apply_nitro_suppression_strips_labels_and_text():
    pipeline = {
        "evidence": {
            "band_matches": [
                {"band_id": "nitro_asym", "matched": True, "label": "NO2 asymmetric", "support_score": 0.6},
                {"band_id": "amide_ii", "matched": True, "label": "amide II", "support_score": 0.4},
            ]
        },
        "rule_assignments": {
            "assignments": {
                "nitro": {
                    "score": 0.22,
                    "human_readable_summary": "Spectral evidence supports nitro (score 0.71).",
                    "caution_flags": ["Nitro requires paired NO₂ bands."],
                    "evidence": ["NO₂ asymmetric in 1500-1570 cm⁻¹"],
                },
                "amide": {"score": 0.5, "human_readable_summary": "Amide supported.", "caution_flags": []},
            },
            "ambiguity_labels": [{"title": "N–O / NO₂ overlap"}],
        },
        "consensus": {
            "per_label": {"nitro": {"final_score": 0.1}, "amide": {"final_score": 0.5}},
            "top_labels": [("nitro", {"final_score": 0.1}), ("amide", {"final_score": 0.5})],
        },
        "ml_refinement": {"basic": {"per_label": {"nitro": {"ml_score": 0.2}}}},
    }
    apply_nitro_suppression(pipeline)
    assert nitro_reporting_suppressed(pipeline)
    bids = [m["band_id"] for m in pipeline["evidence"]["band_matches"]]
    assert "nitro_asym" not in bids
    assert "nitro" not in pipeline["rule_assignments"]["assignments"]
    assert not mentions_nitro_chemistry(pipeline["rule_assignments"]["assignments"]["amide"]["human_readable_summary"])
    assert pipeline["consensus"]["top_labels"] == [("amide", {"final_score": 0.5})]
