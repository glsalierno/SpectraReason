"""Tests for interactive curation and region stack exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reports.discussion_regions import default_discussion_regions, load_discussion_regions
from reports.interactive_curation import build_curation_plotly_figure
from reports.label_overrides import label_record, merge_overrides_with_auto, overrides_to_laid_peaks
from reports.region_stack_export import export_region_stacks, spectra_from_batch, StackSpectrum


def test_merge_overrides_preserves_manual() -> None:
    auto = [
        label_record(
            mode="normalized_absorbance",
            wavenumber_cm1=1600.0,
            peak_y=0.5,
            label_text="1600",
        )
    ]
    saved = {
        "labels": [
            {
                "mode": "normalized_absorbance",
                "wavenumber_cm1": 1600.0,
                "label_text": "1600*",
                "show_label": True,
                "xshift_px": 4,
                "yshift_px": 20,
                "source": "auto",
            }
        ]
    }
    merged = merge_overrides_with_auto(auto, saved)
    assert merged[0]["label_text"] == "1600*"
    assert float(merged[0]["xshift_px"]) == 4.0


def test_overrides_to_laid_peaks_above_absorbance() -> None:
    labels = [
        label_record(
            mode="normalized_absorbance",
            wavenumber_cm1=1600.0,
            peak_y=0.6,
            label_text="1600",
            yshift_px=12,
        )
    ]
    laid = overrides_to_laid_peaks(labels, mode="normalized_absorbance", y_min=0.0, y_max=1.0)
    assert laid[0]["show_label"]
    assert float(laid[0]["label_y"]) > float(laid[0]["y"])


def test_curation_plotly_figure_reversed_xaxis() -> None:
    wn = np.linspace(4000, 400, 100)
    y = np.linspace(0.1, 0.9, 100)
    fig = build_curation_plotly_figure(
        wn,
        y,
        [],
        mode="normalized_absorbance",
        title="test",
    )
    assert float(fig.layout.xaxis.range[0]) == 4000.0
    assert float(fig.layout.xaxis.range[1]) == 400.0


def test_default_discussion_regions_count() -> None:
    regions = default_discussion_regions()
    assert len(regions) == 5
    assert regions[0].name == "OH_NH_stretch"


def test_export_region_stack_smoke(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 200)
    ab = 0.1 + 0.5 * np.exp(-((wn - 1600) ** 2) / (2 * 80**2))
    p1 = tmp_path / "a.CSV"
    p2 = tmp_path / "b.CSV"
    p1.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.1f},{b:.4f}" for a, b in zip(wn, ab)),
        encoding="utf-8",
    )
    p2.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.1f},{b:.4f}" for a, b in zip(wn, ab * 0.8)),
        encoding="utf-8",
    )
    specs = [
        StackSpectrum(stem="a", label="A", path=p1),
        StackSpectrum(stem="b", label="B", path=p2),
    ]
    regions = [r for r in default_discussion_regions() if r.name == "C_O_aromatic_NH"]
    manifest = export_region_stacks(
        spectra=specs,
        out_dir=tmp_path / "out",
        regions=regions,
        stack_modes=("normalized_absorbance",),
        formats=("png",),
        offset_gap=0.15,
    )
    key = "C_O_aromatic_NH_normalized_absorbance_stack"
    assert key in manifest.get("outputs", {})
    assert Path(manifest["outputs"][key][0]).is_file()


def test_load_discussion_regions_missing_file() -> None:
    regions = load_discussion_regions(Path("/nonexistent/regions.yaml"))
    assert len(regions) == 5
