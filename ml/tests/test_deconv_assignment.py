"""Tests for advisory deconvolution → FG candidate mapping and report hooks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_band_library import load_band_library
from ml.ftir_deconv_assignment import (
    AMBIGUITY_1500_1600,
    DeconvReportConfig,
    build_component_rows,
    enrich_deconv_pack_for_pipeline,
    rank_candidates_for_component,
    run_report_deconvolution,
)
from reports.deconv_report import MARKER_DECONV_OVERLAY, build_deconv_candidates_table_html

_EXAMPLE = _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"


def _synth_gaussian(x: np.ndarray, center: float, amp: float, width: float = 12.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def _pipeline_stub(*, atr: bool = False) -> dict:
    return {
        "rule_assignments": {"assignments": {"siloxane": {"score": 0.4}}},
        "evidence": {
            "measurement": {"atr_aware": atr, "mode": "atr" if atr else "transmission"},
            "artifacts": {"flags": {"atr_crystal_band": atr}},
        },
    }


def test_component_near_1710_maps_carbonyl_candidates() -> None:
    lib = load_band_library(prefer_python=True)
    comp = {"center": 1710.0, "height": 0.25, "fwhm": 18.0, "area": 1.0, "rel_area": 0.8}
    cands = rank_candidates_for_component(comp, lib, tolerance_cm1=30.0)
    labels = " ".join(str(c.get("label", "")).lower() for c in cands)
    assert any(
        k in labels
        for k in ("ketone", "ester", "amide", "carboxylic", "aldehyde", "carbonyl", "urethane")
    )


def test_component_near_1540_maps_mid_region_ambiguity() -> None:
    lib = load_band_library(prefer_python=True)
    comp = {"center": 1540.0, "height": 0.2, "fwhm": 22.0, "area": 1.0, "rel_area": 0.7}
    cands = rank_candidates_for_component(comp, lib, tolerance_cm1=30.0)
    labels = " ".join(str(c.get("label", "")).lower() for c in cands)
    assert any(k in labels for k in ("amide", "aromatic", "nitro", "heteroaromatic", "enamine"))
    rows = build_component_rows(
        {"components": [comp]},
        _pipeline_stub(),
        library=lib,
        config=DeconvReportConfig(),
    )
    assert rows
    assert AMBIGUITY_1500_1600.split("(")[0].strip() in rows[0]["caution"] or "overlap" in rows[0]["caution"].lower()


def test_siloxane_candidate_tentative_under_atr_caution() -> None:
    lib = load_band_library(prefer_python=True)
    comp = {"center": 1050.0, "height": 0.18, "fwhm": 40.0, "area": 1.0, "rel_area": 0.6}
    rows = build_component_rows(
        {"components": [comp]},
        _pipeline_stub(atr=True),
        library=lib,
    )
    assert rows
    caut = rows[0]["caution"].lower()
    assert "siloxane" in caut or "si–o" in caut or "atr" in caut
    assert rows[0]["evidence_role"] in ("candidate", "weak candidate", "local motif")


def test_show_deconvolution_adds_table_marker() -> None:
    html = build_deconv_candidates_table_html(
        [
            {
                "center": 1710.0,
                "width": 15.0,
                "rel_area": 0.5,
                "top_candidates": "ester, ketone",
                "evidence_role": "candidate",
                "caution": "—",
                "linked_assignment": "—",
            }
        ],
        anchor="spec-test",
    )
    assert MARKER_DECONV_OVERLAY in html
    assert "Deconvoluted peak candidates" in html
    assert "candidate" in html.lower()


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_deconvolve_report_on_example_spectrum() -> None:
    from lib.spectrum import load_processed_spectrum

    ps = load_processed_spectrum(_EXAMPLE)
    pack = run_report_deconvolution(
        ps.wn,
        ps.y,
        _pipeline_stub(),
        config=DeconvReportConfig(region_preset="auto", min_component_height=0.02),
    )
    assert pack.get("components")
    assert pack.get("curves", {}).get("segments")


@pytest.mark.skipif(not _EXAMPLE.is_file(), reason="example spectrum missing")
def test_report_with_and_without_deconv_flag(tmp_path: Path) -> None:
    from reports.structural_fg_svm_kronecker_report import run_batch

    common = dict(
        input_paths=[_EXAMPLE],
        model_path=None,
        basic_model_path=None,
        subtle_model_path=None,
        page_title="deconv smoke",
        subtitle="1",
        max_peaks=40,
        hover_top_fg=6,
        ml_mode="none",
        rules_config={"ontology": "v4"},
        report_style="product_v1",
        report_audience="front",
        include_evidence=True,
        include_ml=False,
    )
    out_off = tmp_path / "off.html"
    run_batch(**common, out_path=out_off, show_deconvolution=False)
    html_off = out_off.read_text(encoding="utf-8")
    assert MARKER_DECONV_OVERLAY not in html_off

    out_on = tmp_path / "on.html"
    run_batch(
        **common,
        out_path=out_on,
        show_deconvolution=True,
        deconv_regions="auto",
        deconv_min_component_height=0.02,
    )
    html_on = out_on.read_text(encoding="utf-8")
    assert MARKER_DECONV_OVERLAY in html_on
    assert "show_deconvolution=on" in html_on or "Deconvoluted peak candidates" in html_on
