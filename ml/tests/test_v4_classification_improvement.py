"""Tests for v4 classification improvements (peakcodebook, augmentation, diagnostics)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.ftir_peakcodebook_features import (
    peakcodebook_feature_vector,
    peakcodebook_meta,
    stable_peakcodebook_feature_names,
)
from ml.spectrum_augmentation import augment_spectrum
from ml.training_diagnostics import (
    HARD_NEGATIVE_PAIRS,
    compute_hard_negative_sample_weights,
    tune_per_label_thresholds,
)


def _synthetic_spectrum(n: int = 400):
    wn = np.linspace(650, 3700, n)
    y = 0.05 + 0.4 * np.exp(-((wn - 1700) ** 2) / (2 * 80**2))
    y += 0.25 * np.exp(-((wn - 3300) ** 2) / (2 * 120**2))
    return wn, y


def test_peakcodebook_deterministic():
    wn, y = _synthetic_spectrum()
    v1, n1 = peakcodebook_feature_vector(wn, y)
    v2, n2 = peakcodebook_feature_vector(wn, y)
    assert n1 == n2
    assert np.allclose(v1, v2)
    assert not np.any(np.isnan(v1))
    assert not np.any(np.isinf(v1))


def test_peakcodebook_names_match_length():
    names = stable_peakcodebook_feature_names()
    meta = peakcodebook_meta()
    assert len(names) == int(meta["n_peakcodebook"])
    wn, y = _synthetic_spectrum()
    vec, nms = peakcodebook_feature_vector(wn, y)
    assert len(vec) == len(names) == len(nms)


def test_peakcodebook_only_when_requested():
    from ml.structural_fg_svm import build_feature_row_layout

    wn, y = _synthetic_spectrum()
    row_base, _ = build_feature_row_layout(
        wn, y, {}, feature_set="spectral+evidence_v2", calc=None, mordred_dim=0, smarts_feature_labels=[]
    )
    row_pc, _ = build_feature_row_layout(
        wn,
        y,
        {},
        feature_set="spectral+evidence_v2+peakcodebook",
        calc=None,
        mordred_dim=0,
        smarts_feature_labels=[],
    )
    assert row_pc.shape[0] > row_base.shape[0]
    assert row_pc.shape[0] - row_base.shape[0] == int(peakcodebook_meta()["n_peakcodebook"])


def test_augmentation_changes_spectrum():
    wn, y = _synthetic_spectrum()
    wn2, y2 = augment_spectrum(wn, y, mode="light", seed=42)
    assert not np.allclose(wn, wn2) or not np.allclose(y, y2)


def test_hard_negative_pairs_expanded():
    assert "amide" in HARD_NEGATIVE_PAIRS
    assert "tertiary_amine" in HARD_NEGATIVE_PAIRS["amide"]


def test_hard_negative_sample_weights():
    labels = ["phenol", "alcohol", "aromatic"]
    Y = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    w = compute_hard_negative_sample_weights(labels, Y, weight=1.5)
    assert w.shape == (3,)
    assert float(w.max()) >= 1.5


def test_threshold_objectives():
    import csv
    import io

    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.55, 0.9, 0.2, 0.7])
    thr_f1, rows_f1 = tune_per_label_thresholds(
        ["x"], y.reshape(-1, 1), s.reshape(-1, 1), score_kind="calibrated_probability", objective="f1"
    )
    thr_pb, _ = tune_per_label_thresholds(
        ["x"], y.reshape(-1, 1), s.reshape(-1, 1), score_kind="calibrated_probability", objective="precision_biased"
    )
    assert len(thr_f1) == 1
    assert len(thr_pb) == 1
    assert "threshold_objective" in rows_f1[0]
    buf = io.StringIO()
    fields = list(rows_f1[0].keys())
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerow(rows_f1[0])
    assert "threshold_objective" in buf.getvalue()
