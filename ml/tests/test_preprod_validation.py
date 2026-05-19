"""Pre-production validation tests (evidence-first pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
DEMO_SPECTRUM = _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"
V7 = _ROOT / "models" / "struct_fg_v7_pubchem_mordred.joblib"


@pytest.fixture
def demo_spectrum():
    from lib.spectrum import load_processed_spectrum

    if not DEMO_SPECTRUM.is_file():
        pytest.skip("Catechol-120-80-9-IR.jdx missing")
    return load_processed_spectrum(DEMO_SPECTRUM)


def test_evidence_json_serializable(demo_spectrum):
    from ml.ftir_evidence import extract_spectral_evidence

    ev = extract_spectral_evidence(demo_spectrum.wn, demo_spectrum.y)
    json.dumps(ev)
    assert "band_matches" in ev and "peaks" in ev


def test_rule_assignment_structure(demo_spectrum):
    from ml.ftir_evidence import extract_spectral_evidence
    from ml.ftir_rules import assign_functional_groups_from_evidence

    rules = assign_functional_groups_from_evidence(
        extract_spectral_evidence(demo_spectrum.wn, demo_spectrum.y)
    )
    a = rules["assignments"]["primary_amine"]
    for key in (
        "score",
        "confidence",
        "supporting_bands",
        "missing_expected_bands",
        "caution_flags",
        "human_readable_summary",
    ):
        assert key in a


def test_ml_refinement_none(demo_spectrum):
    from ml.ftir_evidence import extract_spectral_evidence
    from ml.ftir_ml_refinement import refine_assignments_with_ml
    from ml.ftir_rules import assign_functional_groups_from_evidence

    ev = extract_spectral_evidence(demo_spectrum.wn, demo_spectrum.y)
    rules = assign_functional_groups_from_evidence(ev)
    out = refine_assignments_with_ml(rules, ev, model_bundle=None, fusion_mode="annotate")
    assert out["fusion_mode"] == "annotate"
    assert out["per_label"]


def test_ml_missing_model_path_graceful(demo_spectrum):
    from ml.ftir_pipeline import load_model_bundle, run_evidence_first_pipeline

    bad = load_model_bundle(_ROOT / "models" / "nonexistent.joblib")
    assert bad is None
    pipe = run_evidence_first_pipeline(
        demo_spectrum.wn,
        demo_spectrum.y,
        ml_mode="basic",
        basic_model=bad,
    )
    assert pipe["ml_mode"] == "basic"
    assert any("subtle" in w.lower() or "basic" in w.lower() or "no" in w.lower() for w in pipe.get("warnings", [])) or pipe["ml_refinement"]["basic"] is None


@pytest.mark.parametrize("fusion_mode", ["annotate", "weighted", "gate", "ml_only"])
def test_fusion_modes(demo_spectrum, fusion_mode):
    joblib = pytest.importorskip("joblib")
    from ml.ftir_pipeline import run_evidence_first_pipeline

    if not V7.is_file():
        pytest.skip("v7 model missing")
    art = joblib.load(V7)
    pipe = run_evidence_first_pipeline(
        demo_spectrum.wn,
        demo_spectrum.y,
        ml_mode="legacy",
        fusion_mode=fusion_mode,  # type: ignore[arg-type]
        legacy_model=art,
    )
    assert pipe["fusion_mode"] == fusion_mode
    assert pipe["consensus"]["per_label"]


def test_dopamine_no_silicone_nitro_top_hits(demo_spectrum):
    from ml.ftir_pipeline import run_evidence_first_pipeline

    pipe = run_evidence_first_pipeline(demo_spectrum.wn, demo_spectrum.y, ml_mode="none")
    a = pipe["rule_assignments"]["assignments"]
    assert float(a.get("silicone_or_silane", {}).get("score", 0)) < 0.5
    assert float(a.get("nitro", {}).get("score", 0)) < 0.5


def test_language_avoids_certainty_claims(demo_spectrum):
    from ml.ftir_pipeline import run_evidence_first_pipeline

    pipe = run_evidence_first_pipeline(demo_spectrum.wn, demo_spectrum.y, ml_mode="none")
    text = json.dumps(pipe)
    for bad in ("confirmed", "proves", "proof of", "definitive identification", "ground truth"):
        assert bad not in text.lower()
