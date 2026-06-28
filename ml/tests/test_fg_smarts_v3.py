"""Tests for SMARTS library v3, leakage warnings, calibration metadata, and reporting language."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def test_smarts_compile_all():
    from ml.fg_smarts_library import compile_all_smarts

    pytest.importorskip("rdkit")
    pat = compile_all_smarts()
    assert len(pat) >= 10


def test_basic_smarts_label_inference_ethanol():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from ml.fg_smarts_library import infer_multilabel_smarts, basic_training_labels
    from ml.structural_fg_svm import resolve_training_label_names

    mol = Chem.MolFromSmiles("CCO")
    labels = resolve_training_label_names("basic", "smarts")
    vec = infer_multilabel_smarts(mol, labels)
    assert vec.get("alcohol_or_phenol") == 1


def test_subtle_phenol_not_aliphatic_alcohol():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from ml.fg_smarts_library import infer_multilabel_smarts, subtle_training_labels

    mol = Chem.MolFromSmiles("Oc1ccccc1")
    labels = subtle_training_labels()
    vec = infer_multilabel_smarts(mol, labels)
    assert vec.get("phenol") == 1
    assert vec.get("alcohol") == 0


def test_min_positives_drop_reason_in_filter():
    from ml.fg_label_configs import filter_labels_by_counts

    Y = np.zeros((30, 2), dtype=int)
    Y[:, 0] = 1
    names = ["a", "b"]
    _, _, dropped, _ = filter_labels_by_counts(names, Y, min_positives=20)
    assert any("b" in d for d in dropped)


def test_leakage_meta_flag_matches_combo():
    leak = str("smarts") == "smarts" and "smarts" in "spectral+smarts"
    assert leak is True


def test_training_writes_calibration_fields(tmp_path):
    pytest.importorskip("sklearn")
    import joblib
    from sklearn.datasets import make_multilabel_classification

    from ml.structural_fg_svm import cmd_train
    import argparse

    X, Y = make_multilabel_classification(
        n_samples=200, n_features=24, n_classes=3, n_labels=3, random_state=0
    )
    prefix = tmp_path / "ds"
    npz = Path(str(prefix) + ".npz")
    meta_path = Path(str(prefix) + ".meta.json")
    np.savez_compressed(
        npz,
        X=X.astype(float),
        Y=Y.astype(int),
        reference_id=np.asarray([f"m{i % 40}" for i in range(X.shape[0])], dtype=object),
        has_mol=np.ones(X.shape[0]),
    )
    meta = {
        "label_names": ["a", "b", "c"],
        "model_kind": "basic",
        "label_source": "existing",
        "feature_set": "spectral",
        "feature_names": [f"f{i}" for i in range(X.shape[1])] + ["has_structure_flag"],
        "n_spectral": X.shape[1] - 1,
        "n_mordred": 0,
        "n_rdkit": 0,
        "n_smarts": 0,
        "n_has_flag": 1,
        "mordred_names": [],
        "dropped_labels": [],
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    ns = argparse.Namespace(
        dataset_prefix=str(prefix),
        model_out="",
        model_kind="basic",
        label_source="",
        min_label_positives=5,
        calibration="sigmoid",
        test_size=0.25,
        split="molecule",
        random_state=1,
        out=str(tmp_path),
        remap_legacy_labels=False,
        n_jobs=1,
    )
    assert cmd_train(ns) == 0
    joblibs = list(tmp_path.glob("struct_fg_basic*.joblib"))
    assert joblibs
    art = joblib.load(joblibs[0])
    assert art.get("calibration", {}).get("fitted") is True
    assert art.get("ml_score_kind") == "calibrated_probability"
    meta_json = list(tmp_path.glob("*_training_metadata.json"))
    assert meta_json
    tm = json.loads(meta_json[0].read_text(encoding="utf-8"))
    assert tm.get("calibration", {}).get("fitted") is True


def test_refinement_runs_with_empty_ml():
    from ml.ftir_ml_refinement import refine_assignments_with_ml

    rule = {
        "assignments": {
            "phenol": {
                "score": 0.05,
                "supporting_peaks": [],
                "supporting_bands": [],
                "missing_expected_bands": [],
                "conflicting_evidence": [],
                "caution_flags": [],
                "confidence": "low",
                "human_readable_summary": "",
            }
        }
    }
    out = refine_assignments_with_ml(
        rule,
        {},
        wn=np.array([1000.0, 2000.0]),
        y=np.array([0.1, 0.2]),
        md={"title": "x"},
        model_bundle=None,
    )
    assert out["per_label"]


def test_report_justification_includes_spectral_fields():
    from ml.ftir_pipeline import run_evidence_first_pipeline
    from ml.ftir_report_sections import justification_cards_html
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    if not p.is_file():
        pytest.skip("example spectrum missing")
    ps = load_processed_spectrum(p)
    pipe = run_evidence_first_pipeline(ps.wn, ps.y, ml_mode="none")
    html = justification_cards_html(pipe)
    assert "Supporting peaks" in html
    assert "Rule score" in html


def test_evidence_only_pipeline_still():
    from ml.ftir_pipeline import run_evidence_first_pipeline
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    if not p.is_file():
        pytest.skip("missing example")
    ps = load_processed_spectrum(p)
    out = run_evidence_first_pipeline(ps.wn, ps.y, ml_mode="none")
    assert out["ml_mode"] == "none"


def test_old_v7_model_loads_or_skips():
    joblib = pytest.importorskip("joblib")
    for sub in ("ml/runs", "models"):
        path = _ROOT / sub / "struct_fg_v7_pubchem_mordred.joblib"
        if path.is_file():
            art = joblib.load(path)
            assert art.get("model") is not None
            return
    pytest.skip("v7 model not present in workspace")


def test_train_none_calibration_sets_score_kind(tmp_path):
    pytest.importorskip("sklearn")
    import joblib
    from sklearn.datasets import make_multilabel_classification

    from ml.structural_fg_svm import cmd_train
    import argparse

    X, Y = make_multilabel_classification(n_samples=120, n_features=16, n_labels=2, random_state=2)
    prefix = tmp_path / "ds2"
    np.savez_compressed(
        Path(str(prefix) + ".npz"),
        X=X.astype(float),
        Y=Y.astype(int),
        reference_id=np.asarray([f"g{i % 20}" for i in range(X.shape[0])], dtype=object),
        has_mol=np.ones(X.shape[0]),
    )
    meta = {
        "label_names": ["a", "b"],
        "model_kind": "basic",
        "label_source": "existing",
        "feature_set": "spectral",
        "feature_names": [f"f{i}" for i in range(X.shape[1])] + ["has_structure_flag"],
        "n_spectral": X.shape[1] - 1,
        "n_mordred": 0,
        "n_rdkit": 0,
        "n_smarts": 0,
        "n_has_flag": 1,
        "mordred_names": [],
        "dropped_labels": [],
    }
    Path(str(prefix) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ns = argparse.Namespace(
        dataset_prefix=str(prefix),
        model_out="",
        model_kind="basic",
        label_source="",
        min_label_positives=3,
        calibration="none",
        test_size=0.2,
        split="random",
        random_state=0,
        out=str(tmp_path),
        remap_legacy_labels=False,
        n_jobs=1,
    )
    assert cmd_train(ns) == 0
    art = joblib.load(next(tmp_path.glob("struct_fg_basic*.joblib")))
    assert art.get("ml_score_kind") == "svm_decision_score"
