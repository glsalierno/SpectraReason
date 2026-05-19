"""Tests for publication-quality annotation layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reports.annotation_layout import (
    apply_peak_label_layout,
    cluster_peaks_for_labeling,
    compute_figure_layout,
    ruler_font_size,
    validate_layout,
    wrap_ruler_label,
    write_layout_validation,
)


def test_wrap_ruler_unsat_mid_two_lines() -> None:
    text = wrap_ruler_label("unsat_mid", "C=C / amide II / N–O", wn_lo=1450, wn_hi=1650)
    assert "<br>" in text
    assert "amide II" in text


def test_ruler_font_size_scales_with_span() -> None:
    narrow = ruler_font_size(1000, 1100, front=True)
    wide = ruler_font_size(1000, 1400, front=True)
    assert 7 <= narrow <= 11
    assert wide >= narrow


def test_cluster_peaks_keeps_strongest_in_fingerprint() -> None:
    peaks = [
        {"wn_cm1": 1100.0, "height": 0.2, "peak_quality": "moderate"},
        {"wn_cm1": 1110.0, "height": 0.5, "peak_quality": "strong"},
        {"wn_cm1": 3000.0, "height": 0.3, "peak_quality": "moderate"},
    ]
    kept, stats = cluster_peaks_for_labeling(peaks, cluster_distance_cm1=18.0)
    wns = {float(p["wn_cm1"]) for p in kept}
    assert 1110.0 in wns
    assert 1100.0 not in wns
    assert 3000.0 in wns
    assert stats.get("fingerprint_cluster_suppressed", 0) >= 1


def test_smart_layout_shifts_before_hiding() -> None:
    anns = [
        {"wn": 1000.0, "y": 0.5, "text": "1000", "_peak": {"height": 0.5, "peak_quality": "strong"}},
        {"wn": 1005.0, "y": 0.48, "text": "1005", "_peak": {"height": 0.45, "peak_quality": "strong"}},
        {"wn": 1010.0, "y": 0.52, "text": "1010", "_peak": {"height": 0.4, "peak_quality": "moderate"}},
    ]
    laid, stats = apply_peak_label_layout(anns, mode="smart", y_max=1.0)
    assert len(laid) >= 2
    assert stats.get("labels_shifted", 0) >= 0 or stats.get("collision_suppressed", 0) >= 0


def test_auto_layout_increases_height_with_many_labels() -> None:
    few = compute_figure_layout(n_labeled_peaks=5, use_ruler=True, auto_layout=True)
    many = compute_figure_layout(n_labeled_peaks=28, use_ruler=True, auto_layout=True)
    assert many["height"] > few["height"]


def test_presentation_mode_taller_than_default() -> None:
    base = compute_figure_layout(n_labeled_peaks=10, auto_layout=True, presentation=False)
    pres = compute_figure_layout(n_labeled_peaks=10, auto_layout=True, presentation=True)
    assert pres["height"] >= base["height"]


def test_validate_layout_json_fields() -> None:
    v = validate_layout(
        {
            "label_layout_stats": {
                "n_labels": 10,
                "labeled_peaks_count": 8,
                "labels_shifted": 3,
                "collision_suppressed": 2,
            },
            "figure_height": 1200,
        }
    )
    assert v["n_labels"] == 10
    assert v["n_labels_shifted"] == 3


def test_write_layout_validation(tmp_path: Path) -> None:
    p = write_layout_validation(tmp_path, {"n_labels": 5, "n_labeled_peaks": 4})
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["n_labels"] == 5
    assert (tmp_path / "export_layout_audit.md").is_file()


def test_ruler_labels_fit_box_estimate() -> None:
    """Wrapped unsat_mid label should be shorter per line than single-line original."""
    wrapped = wrap_ruler_label("unsat_mid", "C=C / amide II / N–O / NO2", wn_lo=1450, wn_hi=1650)
    lines = wrapped.replace("<br>", "\n").split("\n")
    assert all(len(line) <= 28 for line in lines)


def test_ruler_multi_line_rows_get_taller_bands() -> None:
    from reports.annotation_layout import allocate_ruler_y_bands, plan_ruler_row_layouts, ruler_row_line_weight

    one = ruler_row_line_weight("C=O")
    two = ruler_row_line_weight("line1<br>line2")
    assert two > one
    bands = allocate_ruler_y_bands([one, two, one])
    h_single = bands[0][1] - bands[0][0]
    h_double = bands[1][1] - bands[1][0]
    assert h_double > h_single * 1.25

    from ml.ftir_region_ruler import FTIR_RULER_REGIONS

    layouts, total_w = plan_ruler_row_layouts(FTIR_RULER_REGIONS, front=True)
    unsat = next(x for x in layouts if x["spec"].id == "unsat_mid")
    carbonyl = next(x for x in layouts if x["spec"].id == "carbonyl")
    assert unsat["n_lines"] >= 2
    assert unsat["band_height"] > carbonyl["band_height"]
    assert total_w > len(FTIR_RULER_REGIONS)


def test_static_export_writes_separate_spectrum_and_region_panels(tmp_path) -> None:
    from ml.canonical_peaks import build_canonical_peak_table
    from ml.ftir_evidence import extract_spectral_evidence
    from reports.static_figure_export import export_static_matplotlib_bundle

    wn = np.linspace(4000, 400, 800)
    y = 0.35 * np.exp(-((wn - 1700) ** 2) / (2 * 80**2))
    ev = extract_spectral_evidence(wn, y, config={"ontology": "v4", "peak_sensitivity": "sensitive"})
    pipeline = {"ontology": "v4", "evidence": ev, "rule_assignments": {"assignments": {}}}
    pack = build_canonical_peak_table(pipeline, label_all_above_height=0.05)
    out = tmp_path / "figs"
    result = export_static_matplotlib_bundle(
        spectrum_name="test.csv",
        wn=wn,
        y=y,
        pipeline=pipeline,
        canonical_pack=pack,
        out_dir=out,
        fmt="png",
        dpi=72,
        static_label_policy="key",
        spectrum_label_policy="all",
        max_static_labels=5,
        show_ruler=True,
    )
    names = {Path(f).name for f in result.get("files") or []}
    assert "test_region_guide.png" in names
    assert "test_spectrum_peaks.png" in names
    assert result.get("spectrum_label_policy") == "all"


def test_build_stacked_passes_layout_meta() -> None:
    from reports.structural_fg_svm_kronecker_report import _build_stacked_interactive_figure

    wn = np.linspace(4000, 400, 800)
    y = 0.3 * np.exp(-((wn - 1700) ** 2) / (2 * 80**2)) + 0.1 * np.exp(-((wn - 1100) ** 2) / (2 * 40**2))
    peaks = [
        {"wn_cm1": float(w), "height": float(h), "peak_quality": "strong", "label_reason": "height_prominence"}
        for w, h in [(1700, 0.35), (1100, 0.15), (1108, 0.14), (1115, 0.13)]
    ]
    pipeline = {"evidence": {"band_matches": [], "peaks": peaks}, "ontology": "v4"}
    fig, meta = _build_stacked_interactive_figure(
        name="layout_test",
        wn=wn,
        y=y,
        peak_wn=[p["wn_cm1"] for p in peaks],
        peak_h=[p["height"] for p in peaks],
        pipeline=pipeline,
        peaks_dicts=peaks,
        show_region_ruler=True,
        report_style="product_v1",
        report_audience="front",
        visual_theme="matlab",
        peaks_labeled=peaks,
        peaks_plotted=peaks,
        max_peak_labels=10,
        presentation_mode=False,
        peak_label_layout="smart",
        auto_layout=True,
        fingerprint_cluster_distance=18.0,
    )
    assert meta.get("peak_label_layout") == "smart"
    assert int(meta.get("figure_height", 0)) >= 900
    assert fig.layout.height is not None
