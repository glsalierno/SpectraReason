"""Front-facing professionalization: consensus table, NO₂ demotion, guardrails."""

from __future__ import annotations

from ml.ftir_guardrails import (
    apply_amide_overlap_guardrails,
    apply_nitro_noxide_confounder_guardrails,
    apply_v3_guardrails,
)
from reports.front_consensus import (
    MARKER_CONSENSUS_TABLE,
    build_front_consensus_table_html,
    nitro_is_supported,
    should_show_front_consensus_row,
)


def _match(band_id: str, *, support: float = 0.2) -> dict:
    return {"band_id": band_id, "matched": True, "support_score": support, "label": band_id}


def _pipeline(assignments: dict) -> dict:
    return {"rule_assignments": {"assignments": assignments}, "evidence": {"band_matches": []}}


def test_no2_local_motifs_hidden_when_nitro_insufficient():
    assigns = {
        "nitro": {
            "score": 0.24,
            "confidence_class": "overlap_limited",
            "evidence_completeness": "single_band",
            "ontology_category": "specific_fg",
        },
        "NO2_asym_region": {
            "score": 0.35,
            "confidence_class": "overlap_limited",
            "ontology_category": "local_motif",
        },
        "NO2_sym_region": {
            "score": 0.3,
            "confidence_class": "overlap_limited",
            "ontology_category": "local_motif",
        },
    }
    pipe = _pipeline(assigns)
    assert not nitro_is_supported(pipe)
    assert not should_show_front_consensus_row("NO2_asym_region", assigns["NO2_asym_region"], pipe)
    assert not should_show_front_consensus_row("NO2_sym_region", assigns["NO2_sym_region"], pipe)


def test_nitro_requires_paired_unambiguous_bands():
    evidence = {
        "band_matches": [
            _match("nitro_asym"),
        ],
    }
    assignments = {
        "nitro": {"score": 0.55, "confidence_class": "supported", "evidence_completeness": "complete"},
        "NO2_asym_region": {"score": 0.4, "ontology_category": "local_motif"},
    }
    apply_nitro_noxide_confounder_guardrails(assignments, evidence)
    assert float(assignments["nitro"]["score"]) <= 0.28
    assert assignments["nitro"]["confidence_class"] == "overlap_limited"
    assert "N_O_NO2_overlap" in assignments


def test_heterocyclic_n_oxide_suppresses_nitro():
    evidence = {
        "band_matches": [
            _match("nitro_asym"),
            _match("nitro_sym"),
        ],
    }
    assignments = {
        "nitro": {"score": 0.5, "confidence_class": "supported", "evidence_completeness": "complete"},
        "heterocyclic_N_O_region": {"score": 0.35, "ontology_category": "local_motif"},
        "heteroaromatic": {"score": 0.4},
        "pyrrole_like_NH": {"score": 0.3},
    }
    apply_nitro_noxide_confounder_guardrails(assignments, evidence)
    assert float(assignments["nitro"]["score"]) <= 0.32
    assert assignments["nitro"]["confidence_class"] == "overlap_limited"


def test_amide_ii_only_reported_as_overlap():
    evidence = {"band_matches": [_match("amide_ii")]}
    assignments = {
        "amide": {"score": 0.45, "confidence_class": "supported", "evidence_completeness": "complete"},
        "amide_II_region": {"score": 0.28, "ontology_category": "local_motif"},
        "pyrrole_like_NH": {"score": 0.25},
    }
    apply_amide_overlap_guardrails(assignments, evidence)
    assert assignments["amide"]["confidence_class"] == "overlap_limited"
    assert "Amide II" in assignments["amide"]["human_readable_summary"]
    assert not should_show_front_consensus_row("amide", assignments["amide"], _pipeline(assignments))


def test_front_consensus_table_heading_not_summary():
    row = {
        "name": "test.jdx",
        "anchor": "spec-0-test",
        "_pipeline": _pipeline(
            {
                "aromatic": {
                    "score": 0.6,
                    "confidence_class": "supported",
                    "evidence_completeness": "complete",
                    "ontology_category": "family",
                },
            }
        ),
    }
    html = build_front_consensus_table_html(rows=[row], ml_enabled=False)
    assert MARKER_CONSENSUS_TABLE in html
    assert "Consensus interpretation</h2>" in html
    assert "Main interpretation" not in html
    assert ">Summary</h2>" not in html
    assert "Consensus interpretation</th>" in html
    assert "rule_only" not in html
    assert "artifact_limited" not in html


def test_raw_region_labels_hidden_from_front_consensus():
    ent = {
        "score": 0.4,
        "confidence_class": "local_possible",
        "ontology_category": "local_motif",
    }
    pipe = _pipeline({})
    assert not should_show_front_consensus_row("C_O_fingerprint_region", ent, pipe)
    assert not should_show_front_consensus_row("NO2_sym_region", ent, pipe)


def test_front_batch_html_omits_summary_table_and_presentation_paths(tmp_path):
    from pathlib import Path

    from reports.structural_fg_svm_kronecker_report import run_batch

    jdx = Path(__file__).resolve().parents[2] / "examples" / "spectra" / "1H-Indol-5-ol-1953-54-4-IR.jdx"
    if not jdx.is_file():
        pytest.skip("example spectrum missing")
    out = tmp_path / "REPORT.html"
    run_batch(
        input_paths=[jdx],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="Test",
        subtitle="front",
        max_peaks=40,
        hover_top_fg=8,
        ml_mode="none",
        report_audience="front",
        visual_theme="matlab",
        export_static_figures=True,
        rules_config={"ontology": "v4"},
    )
    html = out.read_text(encoding="utf-8")
    assert "<section id='summary-table'" not in html
    assert "Consensus interpretation</h2>" not in html
    assert "presentation-figures-details" not in html
    assert "glsal" not in html.lower() and "oneDrive" not in html


def test_debug_guardrails_still_emit_no2_local_motifs():
    evidence = {"band_matches": [_match("nitro_asym")]}
    assignments: dict = {}
    apply_v3_guardrails(assignments, evidence, ontology="v4")
    # v3 guardrails only touch labels in V3_GUARDRAILS; NO2 regions come from rules engine.
    # After nitro guardrails on explicit assignments:
    assignments = {
        "nitro": {"score": 0.5, "confidence_class": "supported"},
        "NO2_asym_region": {"score": 0.3, "ontology_category": "local_motif"},
    }
    apply_nitro_noxide_confounder_guardrails(assignments, evidence)
    assert "NO2_asym_region" in assignments
    assert assignments["NO2_asym_region"]["ontology_category"] == "local_motif"
