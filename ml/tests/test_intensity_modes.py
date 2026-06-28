"""Tests for explicit FTIR intensity classification and transmittance export rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib.intensity_modes import (
    absorbance_to_apparent_transmittance,
    classify_intensity,
    has_transmittance_panel,
    is_difference_filename,
    plan_transmittance_panel,
    transmittance_to_absorbance,
)
from reports.paper_ftir_figures import export_paper_figures_for_spectrum, PaperFigureConfig


def test_transmittance_absorbance_roundtrip() -> None:
    t = np.array([100.0, 50.0, 10.0])
    a = transmittance_to_absorbance(t)
    t_back = absorbance_to_apparent_transmittance(a)
    np.testing.assert_allclose(t_back, t, rtol=1e-6)


def test_difference_filename_markers() -> None:
    assert is_difference_filename("ODA_in_Ethanol_blank_subtracted")
    assert is_difference_filename("pda_eg_con_new_minus_air_scaled")
    assert is_difference_filename("sample_diff_run")
    assert not is_difference_filename("Dopamine")


def test_classify_forced_difference_stem() -> None:
    raw = np.array([0.01, -0.02, 0.03])
    c = classify_intensity(Path("ODA_in_Ethanol_blank_subtracted.CSV"), "auto", raw)
    assert c.category == "absorbance_difference"
    assert not c.is_native_transmittance


def test_native_transmittance_plan() -> None:
    raw = np.linspace(95, 70, 50)
    c = classify_intensity(Path("demo.CSV"), "auto", raw)
    plan = plan_transmittance_panel(raw, c)
    assert has_transmittance_panel(plan)
    assert not plan.is_apparent


def test_difference_skips_transmittance_by_default() -> None:
    raw = np.array([0.05, 0.1, 0.08])
    c = classify_intensity(Path("blank_subtracted.CSV"), "auto", raw)
    plan = plan_transmittance_panel(raw, c, allow_apparent=False)
    assert not has_transmittance_panel(plan)
    assert plan.skip_reason
    assert plan.banner_html


def test_absorbance_apparent_with_flag() -> None:
    raw = np.array([0.5, 1.2, 2.0, 1.5])
    c = classify_intensity(Path("processed_abs.CSV"), "auto", raw)
    assert c.category == "absorbance"
    plan = plan_transmittance_panel(raw, c, allow_apparent=True)
    assert has_transmittance_panel(plan)
    assert plan.is_apparent
    assert plan.warning


def test_export_native_and_absorbance(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 200)
    pct = 92.0 - 10.0 * np.exp(-((wn - 1600) ** 2) / (2 * 60**2))
    csv = tmp_path / "pct.CSV"
    csv.write_text(
        "Wavenumber,Transmittance\n"
        + "\n".join(f"{a:.2f},{b:.4f}" for a, b in zip(wn, pct)),
        encoding="utf-8",
    )
    result = export_paper_figures_for_spectrum(
        csv, tmp_path / "out", config=PaperFigureConfig(formats=("png",))
    )
    assert result["transmittance_valid"]
    assert not result.get("transmittance_is_apparent")
    assert "transmittance_peaks" in result["figures"]


def test_difference_apparent_with_flag_offsets_negative_values() -> None:
    raw = np.array([-0.02, 0.05, 0.11, 0.03])
    c = classify_intensity(Path("ODA_in_Ethanol_blank_subtracted.CSV"), "auto", raw)
    plan = plan_transmittance_panel(raw, c, allow_apparent=True)
    assert has_transmittance_panel(plan)
    assert plan.is_apparent
    assert plan.warning
    assert "Blank-subtracted difference" in str(plan.warning)


def test_apparent_transmittance_baseline_clips_positive_peaks() -> None:
    from reports.paper_ftir_figures import (
        DEFAULT_APPARENT_TRANSMITTANCE_BASELINE_PCT,
        apparent_transmittance_baseline_pct,
        prepare_transmittance_trace,
    )

    raw = np.array([0.02, 0.05, 0.11, 0.03])
    c = classify_intensity(Path("ODA_in_Ethanol_blank_subtracted.CSV"), "auto", raw)
    plan = plan_transmittance_panel(raw, c, allow_apparent=True)
    baseline = apparent_transmittance_baseline_pct(plan)
    assert baseline == DEFAULT_APPARENT_TRANSMITTANCE_BASELINE_PCT
    t_app = absorbance_to_apparent_transmittance(raw - float(np.min(raw)))
    clipped = prepare_transmittance_trace(t_app, baseline_pct=baseline)
    assert float(np.max(clipped)) <= baseline + 1e-9


def test_match_transmittance_minima_respects_baseline() -> None:
    from reports.paper_peak_selection import match_transmittance_minima

    wn = np.linspace(4000, 400, 200)
    y = np.full_like(wn, 95.0)
    y[80:90] = 90.0
    rows = match_transmittance_minima(
        wn,
        y,
        [{"wavenumber_cm1": float(wn[85])}],
        baseline_pct=95.0,
        min_dip_depth_pct=0.35,
    )
    assert rows[0]["label_shown"] == "yes"
    rows_flat = match_transmittance_minima(
        wn,
        y,
        [{"wavenumber_cm1": float(wn[10])}],
        baseline_pct=95.0,
        min_dip_depth_pct=0.35,
    )
    assert rows_flat[0]["label_shown"] == "no"


def test_export_difference_skips_without_flag(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 100)
    ab = 0.05 * np.sin(np.linspace(0, 3, 100))
    csv = tmp_path / "ODA_in_Ethanol_blank_subtracted.CSV"
    csv.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.2f},{b:.6f}" for a, b in zip(wn, ab)),
        encoding="utf-8",
    )
    result = export_paper_figures_for_spectrum(
        csv, tmp_path / "out2", config=PaperFigureConfig(formats=("png",))
    )
    assert not result["transmittance_valid"]
    assert "transmittance_peaks" not in result["figures"]
    assert "absorbance" in result["transmittance_note"].lower() or "difference" in result[
        "transmittance_note"
    ].lower()
