"""Tests for manual peak snapping."""

from __future__ import annotations

import numpy as np

from reports.label_overrides import build_auto_curation_labels, manual_label_from_snap, merge_overrides_with_auto
from reports.peak_snap import snap_peak_at_wavenumber


def test_snap_absorbance_maximum() -> None:
    wn = np.linspace(4000, 400, 400)
    y = np.exp(-((wn - 1605) ** 2) / (2 * 25**2))
    snap = snap_peak_at_wavenumber(1610, wn, y, mode="normalized_absorbance", window_cm1=30)
    assert abs(float(snap["snapped_wavenumber_cm1"]) - 1605) < 8
    assert snap["snap_target"] == "max_absorbance"


def test_snap_transmittance_minimum() -> None:
    wn = np.linspace(4000, 400, 400)
    y = 95 - 10 * np.exp(-((wn - 1700) ** 2) / (2 * 20**2))
    snap = snap_peak_at_wavenumber(1695, wn, y, mode="transmittance", window_cm1=30)
    assert abs(float(snap["snapped_wavenumber_cm1"]) - 1700) < 8
    assert snap["snap_target"] == "min_transmittance"


def test_merge_keeps_manual_only_transmittance() -> None:
    auto = build_auto_curation_labels(type("Sel", (), {"selected": [], "candidates": []})(), [])
    manual = manual_label_from_snap(
        {
            "mode": "transmittance",
            "wavenumber_cm1": 1700.0,
            "peak_y": 82.0,
            "requested_wavenumber_cm1": 1695.0,
            "snapped_wavenumber_cm1": 1700.0,
            "snap_window_cm1": 25.0,
            "snap_target": "min_transmittance",
        },
        added_by="typed",
    )
    merged = merge_overrides_with_auto(auto, {"labels": [manual]})
    assert any(l["mode"] == "transmittance" and l["source"] == "manual" for l in merged)
