"""Tests for region-limited deconvolution features."""

from __future__ import annotations

import numpy as np
import pytest

from ml.ftir_deconvolution import (
    deconv_feature_vector,
    deconvolve_spectrum,
    stable_deconv_feature_names,
    _pseudo_voigt,
)


def test_deterministic_features():
    wn = np.linspace(650, 3700, 500)
    y = 0.02 + 0.5 * np.exp(-((wn - 1710) ** 2) / (2 * 25**2))
    v1, n1 = deconv_feature_vector(wn, y)
    v2, n2 = deconv_feature_vector(wn, y)
    assert n1 == n2
    assert np.allclose(v1, v2)
    assert not np.any(np.isnan(v1))
    assert not np.any(np.isinf(v1))


def test_flat_spectrum_no_fake_peaks():
    wn = np.linspace(1000, 2000, 200)
    y = np.ones_like(wn) * 0.01 + np.random.default_rng(0).normal(0, 1e-5, wn.size)
    res = deconvolve_spectrum(wn, y)
    carb = res["regions"]["carbonyl"]
    assert carb.success is False or len(carb.components) == 0
    vec, names = deconv_feature_vector(wn, y, deconv_result=res)
    idx = {n: i for i, n in enumerate(names)}
    assert vec[idx["deconv_carbonyl_fit_success"]] == 0.0


def test_synthetic_single_peak():
    wn = np.linspace(1650, 1820, 180)
    c0 = 1710.0
    y = _pseudo_voigt(wn, c0, 0.8, 12.0, 0.5) + 0.02
    res = deconvolve_spectrum(wn, y)
    fit = res["regions"]["carbonyl"]
    assert fit.success
    assert len(fit.components) >= 1
    assert abs(fit.dominant_center - c0) < 25.0


def test_synthetic_overlapping_peaks():
    wn = np.linspace(1650, 1820, 200)
    y = (
        _pseudo_voigt(wn, 1710, 0.5, 10, 0.5)
        + _pseudo_voigt(wn, 1755, 0.45, 11, 0.5)
        + 0.02
    )
    res = deconvolve_spectrum(wn, y)
    fit = res["regions"]["carbonyl"]
    assert fit.success
    assert len(fit.components) >= 2


def test_nitrile_sharp_component():
    wn = np.linspace(2100, 2260, 120)
    y = _pseudo_voigt(wn, 2240, 0.9, 4.0, 0.4) + 0.01
    vec, names = deconv_feature_vector(wn, y)
    idx = {n: i for i, n in enumerate(names)}
    assert vec[idx["deconv_triple_bond_sharp_component_present"]] >= 1.0
    assert vec[idx["deconv_nitrile_alkyne_fit_success"]] >= 1.0


def test_broad_oh_component():
    wn = np.linspace(3000, 3700, 200)
    y = _pseudo_voigt(wn, 3350, 0.7, 80.0, 0.6) + 0.02
    res = deconvolve_spectrum(wn, y)
    oh = res["regions"]["oh_nh"]
    assert oh.success
    assert oh.mean_fwhm >= 50.0


def test_poor_fit_caution_fields():
    wn = np.linspace(1300, 1600, 100)
    y = np.random.default_rng(1).normal(0.05, 0.002, wn.size)
    vec, names = deconv_feature_vector(wn, y)
    idx = {n: i for i, n in enumerate(names)}
    assert vec[idx["deconv_nitro_fit_success"]] == 0.0 or vec[idx["deconv_nitro_fit_r2"]] < 0.5


def test_feature_names_length():
    names = stable_deconv_feature_names()
    wn = np.linspace(650, 3700, 400)
    y = 0.1 * np.exp(-((wn - 1700) ** 2) / (2 * 30**2))
    vec, n2 = deconv_feature_vector(wn, y)
    assert len(vec) == len(names) == len(n2)


def test_extract_deconv_features_never_raises():
    from ml.ftir_deconvolution import extract_deconv_features, reset_deconv_stats

    reset_deconv_stats()
    wn = np.array([1.0, 2.0, 3.0])
    y = np.array([np.nan, np.inf, -1.0])
    vec, names, failed = extract_deconv_features(wn, y, mode="fast")
    assert len(vec) == len(names)
    assert np.all(np.isfinite(vec))


def test_build_layout_includes_deconv_only_when_requested():
    from ml.structural_fg_svm import build_feature_row_layout

    wn = np.linspace(650, 3700, 400)
    y = 0.05 + 0.4 * np.exp(-((wn - 1710) ** 2) / (2 * 25**2))
    base, _ = build_feature_row_layout(
        wn, y, {}, feature_set="spectral+evidence_v2", calc=None, mordred_dim=0, smarts_feature_labels=[]
    )
    full, _ = build_feature_row_layout(
        wn,
        y,
        {},
        feature_set="spectral+evidence_v2+deconv",
        calc=None,
        mordred_dim=0,
        smarts_feature_labels=[],
    )
    assert full.shape[0] > base.shape[0]
    assert full.shape[0] - base.shape[0] == len(stable_deconv_feature_names())
