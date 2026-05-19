"""Smoke tests for basic/subtle FG SVM, interpretability, and robustness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def test_basic_label_building():
    from ml.fg_label_configs import infer_fg_vector, MODEL_KIND_BASIC

    md = {"title": "Ethanol", "name": "ethanol"}
    fg = infer_fg_vector(md, model_kind=MODEL_KIND_BASIC)
    assert fg.get("alcohol_or_phenol") == 1


def test_subtle_label_smarts_phenol():
    from ml.fg_label_configs import infer_fg_vector, MODEL_KIND_SUBTLE

    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("Oc1ccccc1")
    fg = infer_fg_vector({"title": "phenol"}, model_kind=MODEL_KIND_SUBTLE, mol=mol)
    assert fg.get("phenol") == 1


def test_filter_insufficient_positives():
    from ml.fg_label_configs import filter_labels_by_counts

    Y = np.array([[1, 0], [1, 0], [0, 1]], dtype=int)
    names = ["common", "rare"]
    keep, Yk, dropped, _ = filter_labels_by_counts(names, Y, min_positives=2)
    assert "common" in keep
    assert any("rare" in d for d in dropped)


def test_load_legacy_joblib():
    joblib = pytest.importorskip("joblib")
    p = _ROOT / "models" / "struct_fg_v7_pubchem_mordred.joblib"
    if not p.is_file():
        pytest.skip("bundled v7 model missing")
    art = joblib.load(p)
    assert art.get("labels")
    assert art.get("model") is not None


def test_explain_one_spectrum():
    joblib = pytest.importorskip("joblib")
    from ml.ftir_interpretability import explain_prediction
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    model = _ROOT / "models" / "struct_fg_v7_pubchem_mordred.joblib"
    if not p.is_file() or not model.is_file():
        pytest.skip("example spectrum or model missing")
    ps = load_processed_spectrum(p)
    art = joblib.load(model)
    out = explain_prediction(ps.wn, ps.y, art, md={"title": "Dopamine"}, top_k=3)
    assert out["explanations"]
    assert out["explanations"][0].get("human_readable_summary")


def test_robustness_one_spectrum():
    joblib = pytest.importorskip("joblib")
    from ml.ftir_robustness import evaluate_robustness_one_spectrum
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    model = _ROOT / "models" / "struct_fg_v7_pubchem_mordred.joblib"
    if not p.is_file() or not model.is_file():
        pytest.skip("example spectrum or model missing")
    ps = load_processed_spectrum(p)
    art = joblib.load(model)
    lf, summ = evaluate_robustness_one_spectrum(ps.wn, ps.y, {"title": "Dopamine"}, {"model": art})
    assert summ.get("overall_robustness_score") is not None
    assert len(lf) > 0


def test_band_library_loads():
    from ml.ftir_band_library import bands_for_label

    bands = bands_for_label("phenol")
    assert bands and bands[0].get("region_min_cm1")


def test_evidence_only_pipeline():
    from ml.ftir_evidence import extract_spectral_evidence
    from ml.ftir_pipeline import run_evidence_first_pipeline
    from ml.ftir_rules import assign_functional_groups_from_evidence
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"
    if not p.is_file():
        pytest.skip("catechol example missing")
    ps = load_processed_spectrum(p)
    ev = extract_spectral_evidence(ps.wn, ps.y)
    rules = assign_functional_groups_from_evidence(ev)
    assert "phenol" in (rules.get("assignments") or {})
    pipe = run_evidence_first_pipeline(ps.wn, ps.y, ml_mode="none")
    assert pipe["rule_assignments"]
    assert pipe["ml_mode"] == "none"


def test_pipeline_without_model():
    from ml.ftir_pipeline import run_evidence_first_pipeline
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    if not p.is_file():
        pytest.skip("missing example")
    ps = load_processed_spectrum(p)
    out = run_evidence_first_pipeline(ps.wn, ps.y, ml_mode="none", basic_model=None)
    assert out["consensus"]


def test_rules_preset_conservative():
    from ml.ftir_rule_config import load_rules_config
    from ml.ftir_rules import get_effective_fg_rules

    cfg = load_rules_config(preset="conservative")
    rules = get_effective_fg_rules({"label_overrides": cfg["label_overrides"]})
    assert float(rules["phenol"]["min_score"]) >= 0.4


def test_export_csv():
    from ml.ftir_export import export_pipeline_batch_csv
    from ml.ftir_pipeline import run_evidence_first_pipeline
    from lib.spectrum import load_processed_spectrum
    import tempfile

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    if not p.is_file():
        pytest.skip("missing example")
    ps = load_processed_spectrum(p)
    pipe = run_evidence_first_pipeline(ps.wn, ps.y, ml_mode="none")
    with tempfile.TemporaryDirectory() as td:
        paths = export_pipeline_batch_csv(
            [{"spectrum": ps.name, "path": str(p), "pipeline": pipe}],
            td,
        )
        assert paths["consensus_longform"].is_file()


def test_phenol_alcohol_fields():
    from ml.ftir_evidence import extract_spectral_evidence
    from ml.ftir_rules import assign_functional_groups_from_evidence
    from lib.spectrum import load_processed_spectrum

    p = _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"
    if not p.is_file():
        pytest.skip("missing catechol")
    ps = load_processed_spectrum(p)
    rules = assign_functional_groups_from_evidence(extract_spectral_evidence(ps.wn, ps.y))
    phenol = rules["assignments"]["phenol"]
    alcohol = rules["assignments"]["alcohol"]
    assert "score" in phenol and "confidence" in phenol
    assert "score" in alcohol


def test_lean_report_smoke():
    from reports.structural_fg_lean_report import run_batch
    import tempfile

    p = _ROOT / "examples" / "spectra" / "Dopamine_Powder.CSV"
    if not p.is_file():
        pytest.skip("missing example")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rp = run_batch(
            input_paths=[p],
            out_dir=out,
            page_title="lean test",
            subtitle="",
            ml_mode="none",
        )
        assert rp.is_file()
        html = rp.read_text(encoding="utf-8")
        assert "Consensus (evidence-first)" in html
        assert (out / "spec_lean_Dopamine_Powder.png").is_file()
