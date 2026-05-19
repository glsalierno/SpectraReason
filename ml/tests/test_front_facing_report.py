"""Front-facing vs debug report audience."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reports.front_facing_report import (
    MARKER_METADATA_HIDDEN,
    MARKER_SPECTROSCOPIST_SUMMARY,
    MARKER_FRONT_TECHNICAL,
    build_front_card_stack,
    build_spectroscopist_summary,
    format_key_spectral_evidence,
    front_ml_check_line,
    is_front_audience,
    resolve_report_audience,
    sanitize_run_settings_line,
)
from reports.structural_fg_svm_kronecker_report import run_batch

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = [
    _ROOT / "examples" / "spectra" / "Pyrrole_109-97-7-IR.jdx",
    _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx",
    _ROOT / "examples" / "spectra" / "Indole_120-72-9-IR.jdx",
    _ROOT / "examples" / "spectra" / "Indoleacetic acid-87-51-4-IR.jdx",
]


def test_resolve_report_audience_defaults():
    assert resolve_report_audience(None, report_style="product_v1", report_density="balanced") == "front"
    assert resolve_report_audience(None, report_style="product_v1", report_density="audit") == "debug"
    assert resolve_report_audience("debug", report_style="product_v1") == "debug"
    assert resolve_report_audience(None, front_facing_flag=True) == "front"


def test_spectroscopist_summary_no_score_jargon():
    pipeline = {
        "rule_assignments": {
            "assignments": {
                "aromatic": {"score": 0.8, "confidence_class": "supported", "ontology_category": "family"},
                "hydroxy_containing": {
                    "score": 0.7,
                    "confidence_class": "tentative",
                    "ontology_category": "family",
                },
            }
        },
        "evidence": {"artifacts": {"flags": {"water_vapor_or_moisture_like": True}}},
        "consensus": {"top_labels": []},
    }
    text = build_spectroscopist_summary(pipeline, ml_enabled=True)
    assert "probability" not in text.lower()
    assert "ontology" not in text.lower()
    assert len(text.split(".")) <= 4


def test_format_key_spectral_evidence_no_broken_parens():
    ent = {
        "evidence": ["Broad O–H/N–H stretch (support 0.4)"],
        "supporting_bands": ["3200.0–3600.0 cm⁻¹ (", "1450.0–1600.0 cm⁻¹ ("],
    }
    phrase = format_key_spectral_evidence(ent, {}, "hydroxy_containing")
    assert "(" not in phrase[:3] or "3200" in phrase
    assert "3200.0–3600.0 cm⁻¹ (" not in phrase


def test_front_ml_check_line_no_mixed():
    line = front_ml_check_line({"consensus": {"top_labels": []}}, ml_enabled=True)
    assert "Mixed" not in line
    assert line.startswith("ML check:")


def test_sanitize_run_settings_strips_absolute_paths():
    raw = (
        "peak_sensitivity=sensitive | "
        r"family=C:\Users\secret\models\family.joblib | "
        "report_audience=front"
    )
    safe = sanitize_run_settings_line(raw)
    assert r"C:\Users" not in safe
    assert "family.joblib" in safe
    assert "report_audience=front" in safe


def test_front_card_stack_markers():
    html = build_front_card_stack(
        pipeline={
            "rule_assignments": {"assignments": {}},
            "evidence": {"artifacts": {"flags": {}}},
            "consensus": {"top_labels": []},
        },
        anchor="spec-0-test",
        ml_enabled=False,
        include_evidence=True,
        audit_html="",
        band_map_html="",
        justify_html="",
        explain_html="",
        fg_block="",
        just_block="",
        meta_html="<pre>meta</pre>",
    )
    assert MARKER_SPECTROSCOPIST_SUMMARY in html
    assert MARKER_FRONT_TECHNICAL in html
    assert MARKER_METADATA_HIDDEN not in html
    assert "<details" in html


@pytest.mark.slow
def test_front_debug_html_reports(tmp_path: Path):
    if not all(p.is_file() for p in _EXAMPLES):
        pytest.skip("example spectra missing")
    family = _ROOT / "ml" / "runs" / "struct_fg_family_v4_ontology_latest.joblib"
    specific = _ROOT / "ml" / "runs" / "struct_fg_specific_v4_ontology_latest.joblib"
    if not family.is_file() or not specific.is_file():
        pytest.skip("v4 ML models not on disk")

    front_out = tmp_path / "front"
    debug_out = tmp_path / "debug"
    common = dict(
        input_paths=_EXAMPLES,
        model_path=None,
        basic_model_path=family,
        subtle_model_path=specific,
        page_title="Test",
        subtitle="Test",
        max_peaks=80,
        hover_top_fg=8,
        ml_mode="both",
        include_evidence=True,
        include_ml=True,
        report_style="product_v1",
        show_region_ruler=True,
        peak_sensitivity="sensitive",
        show_weak_peaks=True,
        max_peak_labels=30,
        label_all_diagnostic_peaks=True,
    )
    run_batch(
        **common,
        out_path=front_out / "REPORT.html",
        report_audience="front",
        front_max_peak_labels=10,
        peak_label_preset="sensitive",
    )
    run_batch(
        **common,
        out_path=debug_out / "REPORT.html",
        report_audience="debug",
        report_density="audit",
    )

    front_html = (front_out / "REPORT.html").read_text(encoding="utf-8")
    debug_html = (debug_out / "REPORT.html").read_text(encoding="utf-8")

    assert 'class="product-v1 front-audience"' in front_html or "product-v1 front-audience" in front_html
    assert MARKER_SPECTROSCOPIST_SUMMARY in front_html
    assert MARKER_METADATA_HIDDEN not in front_html
    assert "ML Mixed" not in front_html
    assert "Models: <code>" not in front_html
    assert MARKER_FRONT_TECHNICAL in front_html
    assert "Technical details" in front_html
    assert "<details" in front_html
    assert "max_peak_labels=10" in front_html or "front_max_peak_labels=10" in front_html
    assert "peak_label_preset=sensitive" in front_html
    assert "report_audience=front" in front_html
    assert 'class="peak-labeling-summary"' not in front_html
    assert "<!-- report-feature:peak-labeling-summary -->" not in front_html
    assert not re.search(r'file://[A-Za-z]:/', front_html)
    assert not re.search(r'>[^<]*C:\\Users[^<]*<', front_html)

    assert MARKER_METADATA_HIDDEN in debug_html or "Metadata</summary>" in debug_html
    assert MARKER_SPECTROSCOPIST_SUMMARY not in debug_html
    assert "peak-labeling-summary" in debug_html
    assert "max_peak_labels=30" in debug_html or "label_all_diagnostic_peaks=on" in debug_html
