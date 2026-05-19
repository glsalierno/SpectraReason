"""Smoke tests for Kronecker batch HTML report density modes."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports.kronecker_pi_layout import (
    build_most_likely_fg_table_html,
    build_spectrum_body_html,
    build_summary_table_html,
    spectrum_summary_row,
)
from reports.structural_fg_svm_kronecker_report import write_interactive_report_html


def _minimal_pipeline() -> dict:
    return {
        "evidence": {
            "band_matches": [],
            "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
        },
        "rule_assignments": {
            "assignments": {
                "aromatic": {
                    "score": 0.55,
                    "confidence": "medium",
                    "confidence_class": "supported",
                    "evidence_completeness": "complete",
                    "assignment_type": "specific",
                    "supporting_bands": [],
                    "supporting_peaks": [],
                    "missing_expected_bands": [],
                    "caution_flags": [],
                    "human_readable_summary": "test",
                },
            },
            "ambiguity_labels": [],
            "guardrails_diagnostics": [],
            "guardrails_version": "v3_guarded",
        },
        "consensus": {
            "top_labels": [
                (
                    "aromatic",
                    {
                        "rule_score": 0.55,
                        "final_score": 0.55,
                        "agreement_status": "insufficient_evidence",
                        "ml_probability_basic": None,
                        "ml_probability_subtle": None,
                    },
                )
            ],
        },
        "ml_refinement": {"basic": None, "subtle": None, "legacy": None},
        "ml_mode": "none",
        "fusion_mode": "annotate",
        "warnings": [],
    }


def test_summary_table_lists_all_spectra() -> None:
    rows = [
        {
            "name": "A",
            "anchor": "spec-0-A",
            "status": "strong_support",
            "families_text": "—",
            "specifics_text": "aromatic",
            "amb_titles": [],
            "caut_major": [],
            "caut_minor": [],
            "caut_short": [],
            "ml_agree": "N/A",
        },
        {
            "name": "B",
            "anchor": "spec-1-B",
            "status": "attention",
            "families_text": "hydroxy_containing",
            "specifics_text": "—",
            "amb_titles": ["X"],
            "caut_major": ["c1"],
            "caut_minor": ["overlap note"],
            "caut_short": ["c1"],
            "ml_agree": "N/A",
        },
    ]
    html = build_summary_table_html(rows=rows, ml_enabled=False)
    assert "summary-table" in html
    assert "Summary Table" in html
    assert "spec-0-A" in html and "spec-1-B" in html
    assert "Supported families" in html


def test_balanced_plot_first_wraps_evidence_and_consensus() -> None:
    html = build_spectrum_body_html(
        pipeline=_minimal_pipeline(),
        anchor="spec-0-x",
        gr="v3",
        density="balanced",
        top_n_summary=5,
        include_evidence=True,
        include_ml=False,
        include_consensus=True,
        resolved_mode="none",
        legacy_probs=None,
        show_ambiguity_labels=True,
        show_artifact_flags=True,
        omit_quick_interpretation=True,
        plot_first_layout=True,
    )
    assert "spec-0-x-evidence-pack" in html
    assert "spec-0-x-cons-caut" in html
    assert "id='spec-0-x-figure'" not in html


def test_most_likely_fg_table_contains_consensus_row() -> None:
    html = build_most_likely_fg_table_html(
        _minimal_pipeline(),
        top_n=8,
        include_consensus=True,
    )
    assert "Leading assignments" in html or "Consensus" in html
    assert "aromatic" in html


def test_write_interactive_plot_wrap_before_tables_html(tmp_path: Path) -> None:
    fig = MagicMock()
    fig.to_html.return_value = "<div class='PLOT_MARKER'></div>"
    out = tmp_path / "order.html"
    write_interactive_report_html(
        out_path=out,
        page_title="t",
        subtitle="s",
        model_path=None,
        model_paths_line="ml_mode=none",
        sections=[
            {
                "name": "One",
                "anchor": "spec-0-one",
                "meta_line": "{}",
                "figure": fig,
                "figure_note_html": "",
                "table_html": "",
                "tables_html": "<div class='AFTER_PLOT'>tables-stack</div>",
                "interpret_html": "",
                "robustness_badge": "",
                "sidebar_badge_html": "",
            }
        ],
        summary_table_html="",
        report_density_label="balanced",
    )
    text = out.read_text(encoding="utf-8")
    p_plot = text.find("PLOT_MARKER")
    p_after = text.find("tables-stack")
    assert p_plot != -1 and p_after != -1
    assert p_plot < p_after
    assert "spec-0-one-figure" in text


def test_balanced_body_collapses_diagnostics() -> None:
    html = build_spectrum_body_html(
        pipeline=_minimal_pipeline(),
        anchor="spec-0-x",
        gr="v3",
        density="balanced",
        top_n_summary=5,
        include_evidence=True,
        include_ml=False,
        include_consensus=True,
        resolved_mode="none",
        legacy_probs=None,
        show_ambiguity_labels=True,
        show_artifact_flags=True,
        omit_quick_interpretation=True,
    )
    assert "spec-0-x-diagnostics" in html
    assert "<details" in html
    assert "spec-0-x-just" in html
    assert ">Details</summary>" in html


def test_summary_mode_wraps_full_audit() -> None:
    html = build_spectrum_body_html(
        pipeline=_minimal_pipeline(),
        anchor="spec-0-x",
        gr="v3",
        density="summary",
        top_n_summary=3,
        include_evidence=True,
        include_ml=False,
        include_consensus=True,
        resolved_mode="none",
        legacy_probs=None,
        show_ambiguity_labels=True,
        show_artifact_flags=True,
        omit_quick_interpretation=True,
    )
    assert "Details (full audit)" in html
    assert "spec-0-x-audit" in html


def test_write_interactive_metadata_not_inline(tmp_path: Path) -> None:
    fig = MagicMock()
    fig.to_html.return_value = "<div id='plot'></div>"
    out = tmp_path / "smoke.html"
    html_path = write_interactive_report_html(
        out_path=out,
        page_title="t",
        subtitle="s",
        model_path=None,
        model_paths_line="ml_mode=none",
        sections=[
            {
                "name": "One",
                "anchor": "spec-0-one",
                "meta_line": '{"cas": "1-2-3"}',
                "figure": fig,
                "figure_note_html": "",
                "table_html": "",
                "tables_html": "<details class='metadata-details'><summary>Metadata</summary>"
                "<pre class='mono'>{\"cas\": \"1-2-3\"}</pre></details>",
                "interpret_html": "",
                "robustness_badge": "",
                "sidebar_badge_html": "<span class='badge-sci badge-strong'>Strong</span>",
            }
        ],
        summary_table_html="<section id='summary-table'><p>x</p></section>",
        report_density_label="balanced",
    )
    text = html_path.read_text(encoding="utf-8")
    assert "summary-table" in text
    assert "Spectra needing review" not in text
    assert "Executive summary" not in text
    assert "metadata-details" in text


def test_spectrum_summary_row_ml_na_when_no_ml() -> None:
    row = spectrum_summary_row(
        name="n",
        anchor="a",
        pipeline=_minimal_pipeline(),
        ml_enabled=False,
    )
    assert row["ml_agree"] == "N/A"
    assert "families_text" in row
    assert "specifics_text" in row


def test_path_for_publish_anonymize_basename() -> None:
    from reports.structural_fg_svm_kronecker_report import _path_for_publish

    p = Path("C:/Users/example/SecretFolder/Spec.CSV")
    assert _path_for_publish(p, anonymize=True) == "Spec.CSV"
    assert "SecretFolder" not in _path_for_publish(p, anonymize=True)


def test_write_interactive_redacted_paths_notice(tmp_path: Path) -> None:
    fig = MagicMock()
    fig.to_html.return_value = "<div id='plot'></div>"
    out = tmp_path / "redact.html"
    write_interactive_report_html(
        out_path=out,
        page_title="t",
        subtitle="s",
        model_path=None,
        model_paths_line="basic=model.joblib | ml_mode=basic",
        sections=[
            {
                "name": "One",
                "anchor": "spec-0-one",
                "meta_line": "{}",
                "figure": fig,
                "figure_note_html": "",
                "table_html": "",
                "tables_html": "",
                "interpret_html": "",
                "robustness_badge": "",
                "sidebar_badge_html": "",
            }
        ],
        summary_table_html="",
        report_density_label="balanced",
        redacted_paths_notice=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "redacted for sharing" in text


def test_upper_mid_shade_coverage_no_large_gaps() -> None:
    from ml.ftir_shade_regions import upper_mid_coverage_gaps

    gaps = upper_mid_coverage_gaps(1800.0, 3200.0, max_gap_cm1=80.0)
    assert not gaps, f"unexpected gaps 1800–3200 cm⁻¹: {gaps}"


def test_traditional_region_shading_multiple_active() -> None:
    import numpy as np

    from reports.structural_fg_svm_kronecker_report import _traditional_region_shading_shapes

    wn = np.linspace(400.0, 4000.0, 800)
    y = np.zeros_like(wn)
    y += 0.5 * np.exp(-((wn - 1700) ** 2) / 8000)
    y += 0.4 * np.exp(-((wn - 1100) ** 2) / 6000)
    y += 0.3 * np.exp(-((wn - 3400) ** 2) / 12000)
    evidence = {
        "regions": {
            "carbonyl": {"rel_max": 0.6},
            "c_o_stretch": {"rel_max": 0.5},
            "oh_nh_broad": {"rel_max": 0.45},
        },
        "summary": {"y_max": 1.0},
    }
    shapes, names = _traditional_region_shading_shapes(
        evidence=evidence,
        wn=wn,
        y=y,
        y_min=0.0,
        y_max=1.0,
        min_rel_max=0.10,
    )
    assert len(shapes) >= 3
    assert any("C=O" in n for n in names)
    assert any("O–H" in n for n in names)
