"""Tests for evidence_v2 features, v4 family/specific training labels, and ML soft gates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.ftir_evidence import extract_spectral_evidence
from ml.ftir_evidence_features import (
    EVIDENCE_FEATURE_VERSION_V2,
    align_evidence_vector,
    evidence_feature_vector,
    feature_prefix_counts,
    stable_evidence_feature_names,
)
from ml.ftir_ml_refinement import apply_ml_advisory_soft_gates
from ml.ftir_ontology import (
    NON_TRAINABLE_V4,
    TRAINABLE_FAMILY_V4,
    TRAINABLE_SPECIFIC_V4,
    trainable_labels_v4,
)
from ml.structural_fg_svm import (
    MODEL_KIND_FAMILY,
    MODEL_KIND_SPECIFIC,
    resolve_training_label_names,
)


def _probe_evidence() -> dict:
    wn = np.linspace(400.0, 4000.0, 900)
    y = np.sin(wn / 220.0) * 0.02 + 0.05
    ev = extract_spectral_evidence(wn, y, peaks=None, config={"ontology": "v4"})
    try:
        from ml.ftir_artifacts import detect_spectral_artifacts

        ev["artifacts"] = detect_spectral_artifacts(wn, y, ev)
    except Exception:
        pass
    ev["_wn_cache"] = wn
    ev["_y_cache"] = y
    return ev


def test_evidence_v2_deterministic_no_nan():
    ev = _probe_evidence()
    v1, n1 = evidence_feature_vector(ev, feature_set="spectral+evidence_v2")
    v2, n2 = evidence_feature_vector(ev, feature_set="spectral+evidence_v2")
    assert n1 == n2
    assert v1 == v2
    assert len(v1) == len(stable_evidence_feature_names(feature_set="spectral+evidence_v2"))
    for x in v1:
        assert math.isfinite(float(x))
    tmpl = stable_evidence_feature_names(feature_set="spectral+evidence_v2")
    aligned = align_evidence_vector(v1, n1, tmpl)
    assert len(aligned) == len(tmpl)


def test_evidence_v2_includes_regions_peaks_artifacts():
    names = stable_evidence_feature_names(feature_set="spectral+evidence_v2")
    assert any(n.startswith("region_") and n.endswith("_integral") for n in names)
    assert any(n.startswith("peak_count_") for n in names)
    assert any(n.startswith("peak_width_mean_") for n in names)
    assert any(n.startswith("art_") for n in names)
    assert "broadness_oh_nh" in names
    counts = feature_prefix_counts(names)
    assert counts.get("region", 0) > 0
    assert counts.get("artifact", 0) >= 1


def test_v1_backward_compatible_feature_set():
    ev = _probe_evidence()
    v_old, n_old = evidence_feature_vector(ev, feature_set="spectral+evidence")
    assert len(v_old) == len(n_old)
    assert len(n_old) < len(stable_evidence_feature_names(feature_set="spectral+evidence_v2"))


def test_trainable_label_sets_v4():
    fam = trainable_labels_v4(MODEL_KIND_FAMILY)
    spec = trainable_labels_v4(MODEL_KIND_SPECIFIC)
    assert set(fam) == set(TRAINABLE_FAMILY_V4)
    assert set(spec) == set(TRAINABLE_SPECIFIC_V4)
    for lab in fam + spec:
        assert lab not in NON_TRAINABLE_V4
    assert "broad_OH_NH_region" not in fam
    assert "water_moisture_artifact" not in spec


def test_resolve_training_labels_smarts_no_smarts_in_x_names():
    fam = resolve_training_label_names(MODEL_KIND_FAMILY, "smarts", ontology="v4")
    spec = resolve_training_label_names(MODEL_KIND_SPECIFIC, "smarts", ontology="v4")
    assert len(fam) >= 8
    assert len(spec) > 9
    assert "phenol" in spec
    assert "hydroxy_containing" in fam


def test_ml_soft_gate_caps_high_ml_low_evidence():
    evidence = {"band_matches": [], "regions": {}, "ratios": {}, "artifacts": {"flags": {}}}
    per_label = {
        "phenol": {
            "rule_score": 0.05,
            "ml_score": 0.82,
            "final_score": 0.82,
            "agreement_status": "ml_only_warning",
            "caution_flags": [],
        }
    }
    apply_ml_advisory_soft_gates(
        per_label,
        evidence,
        {},
        ml_probs={"phenol": 0.82},
        ml_guardrails="strict",
        thr_map={"phenol": 0.5},
        ml_threshold=0.45,
        ml_is_proba=True,
    )
    assert per_label["phenol"]["final_score"] <= 0.42
    assert per_label["phenol"]["agreement_status"] == "ml_only_warning"


def test_siloxane_single_region_capped_in_ml_gate():
    evidence = {
        "band_matches": [
            {
                "band_id": "siloxane_sio",
                "matched": True,
                "support_score": 0.25,
            }
        ],
        "regions": {},
        "ratios": {},
        "artifacts": {"flags": {}},
    }
    per_label = {
        "siloxane": {
            "rule_score": 0.2,
            "ml_score": 0.7,
            "final_score": 0.7,
            "agreement_status": "rule_only",
            "caution_flags": [],
        }
    }
    apply_ml_advisory_soft_gates(
        per_label,
        evidence,
        {},
        ml_probs={"siloxane": 0.7},
        ml_guardrails="strict",
        thr_map={},
        ml_threshold=0.45,
        ml_is_proba=True,
    )
    assert per_label["siloxane"]["final_score"] <= 0.32
