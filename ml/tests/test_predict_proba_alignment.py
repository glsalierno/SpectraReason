"""SVM inference aligns to saved model feature_names when evidence schema grows."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from lib.spectrum import load_processed_spectrum
from ml.structural_fg_svm import build_feature_row_layout, predict_proba_row


def test_predict_proba_row_aligns_to_saved_feature_names():
    model_path = Path("ml/runs/struct_fg_family_v4_ontology_latest.joblib")
    if not model_path.is_file():
        return
    artifact = joblib.load(model_path)
    model_fnames = list(artifact.get("feature_names") or (artifact.get("meta") or {}).get("feature_names") or [])
    assert len(model_fnames) == int(artifact["scaler"].n_features_in_)

    example = Path("examples/spectra/Dopamine_Powder.CSV")
    if not example.is_file():
        return
    ps = load_processed_spectrum(example)
    x_row, _ = build_feature_row_layout(
        ps.wn,
        ps.y,
        {},
        feature_set="spectral+evidence_v2",
        calc=None,
        mordred_dim=0,
        smarts_feature_labels=[],
    )
    assert x_row.shape[0] >= len(model_fnames)

    probs = predict_proba_row(artifact, wn=ps.wn, y=ps.y, md={})
    assert probs
    assert all(0.0 <= float(v) <= 1.0 for v in probs.values())
    assert max(probs.values()) > 0.01
