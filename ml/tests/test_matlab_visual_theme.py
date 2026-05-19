"""MATLAB visual theme, static exports, and MATLAB postprocessing bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports.matlab_visual_theme import (
    MARKER_MATLAB_THEME,
    normalize_visual_theme,
    write_make_figures_m,
)
from reports.report_render import (
    MARKER_LOCAL_HOVER,
    MARKER_PEAK_LABELS,
    MARKER_PLOTLY_SPECTRUM,
    MARKER_REGION_RULER,
)
from reports.structural_fg_svm_kronecker_report import run_batch

_EXAMPLE = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"


def test_normalize_visual_theme() -> None:
    assert normalize_visual_theme("matlab") == "matlab"
    assert normalize_visual_theme("unknown") == "default"


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_matlab_theme_preserves_hover_and_ruler(tmp_path: Path) -> None:
    out = tmp_path / "matlab.html"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="matlab theme",
        subtitle="1 spectrum",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        rules_config={"ontology": "v4"},
        report_style="product_v1",
        report_audience="front",
        visual_theme="matlab",
        show_region_ruler=True,
        peak_sensitivity="sensitive",
    )
    html = out.read_text(encoding="utf-8")
    assert MARKER_PLOTLY_SPECTRUM in html
    assert MARKER_PEAK_LABELS in html or "Labeled peaks" in html
    assert MARKER_LOCAL_HOVER in html
    assert "customdata" in html
    assert MARKER_REGION_RULER in html
    assert MARKER_MATLAB_THEME in html
    assert "visual-matlab" in html
    assert "hovertemplate" in html


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_default_theme_still_works(tmp_path: Path) -> None:
    out = tmp_path / "default.html"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="default",
        subtitle="1",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        rules_config={"ontology": "v4"},
        report_style="product_v1",
        visual_theme="default",
    )
    html = out.read_text(encoding="utf-8")
    assert MARKER_PLOTLY_SPECTRUM in html
    assert "<body class='product-v1 front-audience visual-matlab'>" not in html
    assert 'visual-matlab"' not in html.split("<body", 1)[-1].split(">", 1)[0]
    assert MARKER_MATLAB_THEME not in html


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_static_export_and_matlab_bundle(tmp_path: Path) -> None:
    out = tmp_path / "run" / "REPORT.html"
    static_dir = tmp_path / "figures"
    matlab_dir = tmp_path / "matlab_export"
    run_batch(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        out_path=out,
        page_title="export test",
        subtitle="1",
        max_peaks=24,
        hover_top_fg=6,
        ml_mode="none",
        rules_config={"ontology": "v4"},
        report_style="product_v1",
        visual_theme="matlab",
        export_static_figures=True,
        static_format="svg",
        static_out=static_dir,
        matlab_export_dir=matlab_dir,
        show_region_ruler=True,
    )
    html = out.read_text(encoding="utf-8")
    assert "static-figures-note" not in html
    assert "visualization-only and do not change assignments" not in html

    stem = "Dopamine_Powder"
    assert (matlab_dir / f"{stem}_spectrum.csv").is_file()
    assert (matlab_dir / f"{stem}_peaks.csv").is_file()
    assert (matlab_dir / f"{stem}_ruler_regions.csv").is_file()
    assert (matlab_dir / f"{stem}_key_evidence.csv").is_file()
    assert (matlab_dir / f"{stem}_annotations.csv").is_file()
    assert (matlab_dir / "make_figures.m").is_file()

    assert static_dir.is_dir()
    has_static = any(static_dir.glob(f"{stem}_*"))
    has_json_fallback = any((static_dir / "plotly_json").glob("*.json")) if (static_dir / "plotly_json").is_dir() else False
    assert has_static or has_json_fallback or (static_dir / "STATIC_EXPORT_README.txt").is_file()


def test_make_figures_m_generation(tmp_path: Path) -> None:
    path = write_make_figures_m(tmp_path / "matlab", ["Test_Spec"])
    text = path.read_text(encoding="utf-8")
    assert "showPeakLabels" in text
    assert "exportgraphics" in text
    assert "Test_Spec" in text
