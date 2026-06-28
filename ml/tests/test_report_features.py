"""Regression tests for Kronecker report feature contract (see reports/REPORT_FEATURE_CONTRACT.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports.front_consensus import MARKER_CONSENSUS_TABLE
from reports.report_render import (
    FORBIDDEN_REPORT_STRINGS,
    MARKER_BAND_EVIDENCE_MAP,
    MARKER_BAND_SHADING,
    MARKER_EVIDENCE_TABLE,
    MARKER_KRONECKER_STEMS,
    MARKER_LOCAL_HOVER,
    MARKER_PLOTLY_SPECTRUM,
    MARKER_SUMMARY_TABLE,
    render_band_shading_shapes,
)
from reports.structural_fg_svm_kronecker_report import run_batch

_EXAMPLE = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"


def _assert_contract(html: str, *, expect_shading: bool, audience: str = "debug") -> None:
    for bad in FORBIDDEN_REPORT_STRINGS:
        assert bad not in html, f"forbidden section present: {bad}"
    if audience == "front":
        assert MARKER_CONSENSUS_TABLE in html or "Consensus interpretation" in html
    else:
        assert MARKER_SUMMARY_TABLE in html
        assert "summary-table" in html
    assert MARKER_PLOTLY_SPECTRUM in html
    assert MARKER_LOCAL_HOVER in html
    assert MARKER_KRONECKER_STEMS in html
    assert MARKER_EVIDENCE_TABLE in html or "Functional group assignments" in html
    assert MARKER_BAND_EVIDENCE_MAP in html or "Band evidence map" in html
    assert "plotly-graph-div" in html or "Plotly.newPlot" in html
    assert "customdata" in html or "%{customdata}" in html
    if expect_shading:
        assert MARKER_BAND_SHADING in html
        assert 'data-report-feature="band-shading"' in html
        assert "fillcolor" in html
        assert "layer" in html and "below" in html
    else:
        assert MARKER_BAND_SHADING not in html
        assert 'data-report-feature="band-shading"' not in html


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_batch_report_with_band_shading(tmp_path: Path) -> None:
    out = tmp_path / "with_shade.html"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="shade test",
        subtitle="1 spectrum",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        fusion_mode="annotate",
        include_evidence=True,
        include_ml=False,
        include_consensus=True,
        rules_config={"ontology": "v4"},
        guardrails_mode="v3",
        show_band_shading=True,
        report_density="balanced",
        report_audience="debug",
    )
    html = out.read_text(encoding="utf-8")
    _assert_contract(html, expect_shading=True, audience="debug")
    idx_summary = html.index(MARKER_SUMMARY_TABLE)
    idx_plot = html.index(MARKER_PLOTLY_SPECTRUM)
    assert idx_summary < idx_plot, "Summary Table must appear before spectrum plot"


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_batch_report_without_band_shading(tmp_path: Path) -> None:
    out = tmp_path / "no_shade.html"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="no shade test",
        subtitle="1 spectrum",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        fusion_mode="annotate",
        include_evidence=True,
        include_ml=False,
        include_consensus=True,
        rules_config={"ontology": "v4"},
        guardrails_mode="v3",
        show_band_shading=False,
        report_density="balanced",
        report_audience="debug",
    )
    html = out.read_text(encoding="utf-8")
    _assert_contract(html, expect_shading=False, audience="debug")


def test_band_shading_shapes_independent_of_rule_score() -> None:
    import numpy as np

    wn = np.linspace(400.0, 4000.0, 800)
    y = np.zeros_like(wn)
    y += 0.6 * np.exp(-((wn - 1700) ** 2) / 8000)
    y += 0.5 * np.exp(-((wn - 1100) ** 2) / 6000)
    evidence = {
        "regions": {"carbonyl": {"rel_max": 0.55}, "c_o_stretch": {"rel_max": 0.48}},
        "summary": {"y_max": 1.0},
    }
    shapes, names = render_band_shading_shapes(
        evidence=evidence, wn=wn, y=y, y_min=0.0, y_max=1.0, min_rel_max=0.10
    )
    assert len(shapes) >= 2
    assert all(s.get("layer") == "below" for s in shapes)
    assert "xref" not in shapes[0]
