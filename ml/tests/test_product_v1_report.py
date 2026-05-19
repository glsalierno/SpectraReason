"""Regression tests for product_v1 report style (see reports/REPORT_PRODUCT_CONTRACT.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports.product_v1_report import (
    MARKER_KEY_EVIDENCE,
    MARKER_METADATA_HIDDEN,
    MARKER_PRODUCT_AUDIT,
    MARKER_PRODUCT_DETAILS,
    MARKER_PRODUCT_INTERPRETATION,
    MARKER_SPECTRUM_ANNOTATIONS,
    compress_caution,
)
from reports.report_render import (
    FORBIDDEN_REPORT_STRINGS,
    MARKER_BAND_SHADING,
    MARKER_LOCAL_HOVER,
    MARKER_PLOTLY_SPECTRUM,
    MARKER_SUMMARY_TABLE,
)
from reports.structural_fg_svm_kronecker_report import run_batch

_EXAMPLE = _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"


def test_compress_caution_shortens_verbose_text() -> None:
    long = "phenol: Moisture-like broad O–H may overlap phenolic assignment — check aryl C–O."
    short = compress_caution(long)
    assert len(short) < len(long)
    assert "moisture" in short.lower() or "O–H" in short


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_product_v1_report_contract(tmp_path: Path) -> None:
    out = tmp_path / "product.html"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="product test",
        subtitle="1 spectrum",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        fusion_mode="annotate",
        include_evidence=True,
        include_ml=False,
        rules_config={"ontology": "v4"},
        guardrails_mode="v3",
        show_band_shading=True,
        label_band_shading=True,
        report_density="balanced",
        report_style="product_v1",
        report_audience="debug",
    )
    html = out.read_text(encoding="utf-8")
    for bad in FORBIDDEN_REPORT_STRINGS:
        assert bad not in html
    assert MARKER_SUMMARY_TABLE in html
    assert MARKER_PLOTLY_SPECTRUM in html
    assert MARKER_BAND_SHADING in html
    assert MARKER_LOCAL_HOVER in html
    assert MARKER_PRODUCT_INTERPRETATION in html
    assert MARKER_KEY_EVIDENCE in html
    assert MARKER_SPECTRUM_ANNOTATIONS in html
    assert MARKER_PRODUCT_DETAILS in html
    assert MARKER_PRODUCT_AUDIT in html
    assert MARKER_METADATA_HIDDEN in html
    assert "product-v1" in html
    assert html.index(MARKER_SUMMARY_TABLE) < html.index(MARKER_PLOTLY_SPECTRUM)
    assert "Broad O–H" not in html or "product-details" in html
    assert "metadata-details" in html
    assert 'metadata-details" open' not in html.replace("metadata-details product", "")


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_legacy_style_still_available(tmp_path: Path) -> None:
    out = tmp_path / "legacy.html"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="legacy test",
        subtitle="1 spectrum",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        report_style="legacy",
        rules_config={"ontology": "v4"},
        show_band_shading=True,
    )
    html = out.read_text(encoding="utf-8")
    assert MARKER_PRODUCT_INTERPRETATION not in html
    assert "Band evidence map" in html or "report-feature:band-evidence-map" in html
