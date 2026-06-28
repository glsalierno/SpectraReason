"""Tests for publication paper FTIR figure export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reports.annotation_layout import apply_horizontal_leader_label_layout
from reports.paper_ftir_figures import (
    PaperFigureConfig,
    export_paper_figures_for_spectrum,
    resolve_intensity_mode,
    transmittance_valid,
)
from reports.paper_peak_selection import (
    PaperPeakSelectionConfig,
    detect_candidate_peaks,
    parse_ignore_label_ranges,
    select_labeled_peaks,
)


def test_transmittance_valid_modes() -> None:
    assert transmittance_valid("transmittance_percent")
    assert not transmittance_valid("absorbance")


def test_horizontal_leader_layout_places_above() -> None:
    peaks = [
        {"wn": 1600.0, "y": 0.8, "text": "1600", "prominence": 0.5},
        {"wn": 1700.0, "y": 0.75, "text": "1700", "prominence": 0.4},
    ]
    laid, stats = apply_horizontal_leader_label_layout(
        peaks,
        y_min=0.0,
        y_max=1.0,
        label_side="above",
        max_labels=2,
    )
    assert stats["labels_placed"] >= 1
    for item in laid:
        assert float(item["label_y"]) > float(item["y"])


def test_ignore_label_range_excludes_fingerprint(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 800)
    y = np.zeros_like(wn)
    y += 0.08 * np.exp(-((wn - 1600) ** 2) / (2 * 40**2))
    y += 0.10 * np.exp(-((wn - 700) ** 2) / (2 * 30**2))
    cfg = PaperPeakSelectionConfig(ignore_label_ranges=[(400.0, 900.0)])
    candidates = detect_candidate_peaks(wn, y, config=cfg)
    selection = select_labeled_peaks(wn, y, config=cfg)
    selected_wn = [float(p["wavenumber_cm1"]) for p in selection.selected]
    assert all(not (400 <= w <= 900) for w in selected_wn)
    assert any(1500 <= w <= 1800 for w in selected_wn)


def test_export_paper_figures_transmittance_and_absorbance(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 500)
    pct = 90.0 - 15.0 * np.exp(-((wn - 1600) ** 2) / (2 * 80**2))
    csv = tmp_path / "demo_pct.CSV"
    csv.write_text(
        "Wavenumber,Transmittance\n"
        + "\n".join(f"{a:.2f},{b:.4f}" for a, b in zip(wn, pct)),
        encoding="utf-8",
    )
    cfg = PaperFigureConfig(
        formats=("png",),
        max_peak_labels=8,
        wn_min=400,
        wn_max=4000,
    )
    result = export_paper_figures_for_spectrum(csv, tmp_path / "paper", config=cfg)
    assert result["transmittance_valid"]
    assert "normalized_absorbance_peaks" in result["figures"]
    assert "transmittance_peaks" in result["figures"]
    assert Path(result["peak_tables"]["selected"]).is_file()
    assert Path(result["peak_tables"]["all_candidates"]).is_file()
    assert Path(result["peak_tables"]["suppressed"]).is_file()
    assert Path(result["peak_tables"]["transmittance_minima"]).is_file()
    assert Path(result["peak_tables"]["manual"]).is_file()
    assert Path(result["feedback"]["files"]["txt"]).is_file()


def test_absorbance_only_skips_transmittance(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 200)
    ab = np.linspace(0.1, 0.9, 200)
    csv = tmp_path / "blank_subtracted.CSV"
    csv.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.2f},{b:.4f}" for a, b in zip(wn, ab)),
        encoding="utf-8",
    )
    result = export_paper_figures_for_spectrum(
        csv,
        tmp_path / "paper2",
        config=PaperFigureConfig(formats=("png",), max_peak_labels=6),
    )
    assert not result["transmittance_valid"]
    assert result["transmittance_note"]
    assert "transmittance_peaks" not in result["figures"]


def test_horizontal_leader_suppresses_on_overlap() -> None:
    peaks = [
        {"wn": 1600.0, "y": 0.5, "text": "1600", "prominence": 0.9},
        {"wn": 1602.0, "y": 0.49, "text": "1602", "prominence": 0.1},
    ]
    laid, stats = apply_horizontal_leader_label_layout(
        peaks,
        y_min=0.0,
        y_max=1.0,
        max_labels=2,
    )
    assert stats["labels_placed"] >= 1


def test_resolve_intensity_mode_forces_absorbance_on_difference() -> None:
    from lib.intensity_modes import classify_intensity

    raw = np.array([0.1, 0.2, 0.3])
    c = classify_intensity(Path("sample_minus_blank.CSV"), "auto", raw)
    assert c.category == "absorbance_difference"


def test_parse_ignore_label_ranges() -> None:
    ranges = parse_ignore_label_ranges(["900:400"])
    assert ranges == [(400.0, 900.0)]
