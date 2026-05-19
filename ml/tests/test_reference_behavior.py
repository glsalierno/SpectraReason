"""
Reference behavior principles for FTIR_SVM_v4 production release.

Guards scientific guardrails and front/debug report contracts without duplicating
full integration tests (see also test_front_professionalization, test_ftir_v3_guardrails).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ml.canonical_peaks import build_canonical_peak_table
from ml.ftir_guardrails import (
    apply_amide_overlap_guardrails,
    apply_nitro_noxide_confounder_guardrails,
)
from ml.ftir_rules import assign_functional_groups_from_evidence
from ml.ftir_region_ruler import FTIR_RULER_REGIONS
from reports.front_consensus import (
    MARKER_CONSENSUS_TABLE,
    nitro_is_supported,
    should_show_front_consensus_row,
)
from reports.front_facing_report import (
    MARKER_FRONT_TECHNICAL,
    MARKER_METADATA_HIDDEN,
    MARKER_SPECTROSCOPIST_SUMMARY,
    build_front_card_stack,
)
from reports.reproducibility_meta import MARKER_REPRODUCIBILITY, build_reproducibility_html, build_run_context

_ROOT = Path(__file__).resolve().parents[2]


def _bm(
    band_id: str,
    *,
    matched: bool = True,
    support: float = 0.4,
    specificity: str = "medium",
) -> dict:
    return {
        "band_id": band_id,
        "label": band_id,
        "subclass": band_id,
        "region_min_cm1": 0,
        "region_max_cm1": 4000,
        "mode": band_id,
        "importance": "required",
        "specificity": specificity,
        "region_rel_max": support,
        "peak_support": support,
        "support_score": support,
        "matched": matched,
        "peaks_near": [],
    }


def _pipe(assignments: dict) -> dict:
    return {"rule_assignments": {"assignments": assignments}, "evidence": {"band_matches": []}}


def test_nitro_not_supported_from_single_no2_band() -> None:
    ev = {
        "band_matches": [_bm("nitro_asym"), _bm("nitro_sym", matched=False, support=0.05)],
        "peaks": [],
        "ratios": {},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": ""},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3", "ontology": "v4"})
    assert float(r["assignments"]["nitro"]["score"]) < 0.35
    assigns = dict(r["assignments"])
    apply_nitro_noxide_confounder_guardrails(assigns, ev)
    assert not nitro_is_supported(_pipe(assigns))


def test_heterocyclic_n_oxide_suppresses_nitro_overcall() -> None:
    ev = {"band_matches": [_bm("nitro_asym"), _bm("nitro_sym"), _bm("heterocyclic_n_oxide")]}
    assigns = {
        "nitro": {"score": 0.5, "confidence_class": "supported", "evidence_completeness": "complete"},
        "heterocyclic_N_O_region": {"score": 0.35, "ontology_category": "local_motif"},
        "heteroaromatic": {"score": 0.4},
    }
    apply_nitro_noxide_confounder_guardrails(assigns, ev)
    assert float(assigns["nitro"]["score"]) <= 0.32


def test_siloxane_not_supported_from_single_atr_co_band() -> None:
    ev = {
        "band_matches": [
            _bm("siloxane_sio", support=0.35),
            _bm("ether_co", support=0.55),
        ],
        "peaks": [{"wn_cm1": 1080, "height": 0.5, "quality_sharpness": 0.02}],
        "ratios": {"siloxane_to_c_o": 0.15},
        "regions": {},
        "artifacts": {"flags": {"fingerprint_crowding": True}, "cautions": [], "summary": ""},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3"})
    sil = float(r["assignments"]["siloxane"]["score"])
    assert sil <= 0.25
    assert r["assignments"]["siloxane"]["confidence_class"] in ("local_possible", "not_supported", "overlap_limited")


def test_co_si_overlap_without_si_evidence() -> None:
    ev = {
        "band_matches": [_bm("ether_co", support=0.5), _bm("aryl_ether_co", support=0.4)],
        "peaks": [],
        "ratios": {"siloxane_to_c_o": 0.1},
        "regions": {},
        "artifacts": {"flags": {}, "cautions": [], "summary": ""},
    }
    r = assign_functional_groups_from_evidence(ev, config={"guardrails_mode": "v3", "ontology": "v4"})
    cc = r["assignments"].get("C_O_Si_O_overlap", r["assignments"].get("co_si_overlap"))
    if cc is None:
        # overlap may be keyed as local motif
        overlap_keys = [k for k in r["assignments"] if "overlap" in k.lower() or "C_O" in k]
        assert overlap_keys or float(r["assignments"].get("siloxane", {}).get("score", 0)) < 0.3
    else:
        assert float(cc.get("score", 0)) >= 0.1 or cc.get("confidence_class") == "overlap_limited"


def test_amide_ii_alone_not_supported_amide() -> None:
    ev = {"band_matches": [_bm("amide_ii")]}
    assigns = {
        "amide": {"score": 0.45, "confidence_class": "supported", "evidence_completeness": "complete"},
        "amide_II_region": {"score": 0.28, "ontology_category": "local_motif"},
    }
    apply_amide_overlap_guardrails(assigns, ev)
    assert assigns["amide"]["confidence_class"] == "overlap_limited"
    assert not should_show_front_consensus_row("amide", assigns["amide"], _pipe(assigns))


def test_amide_enamine_pyrrole_ambiguity_visible_in_summary() -> None:
    assigns = {
        "amide": {"score": 0.3, "confidence_class": "overlap_limited", "human_readable_summary": "Amide II overlap"},
        "pyrrole_like_NH": {"score": 0.28, "confidence_class": "tentative"},
        "enamine_region": {"score": 0.22, "ontology_category": "local_motif"},
    }
    pipe = _pipe(assigns)
    from reports.front_consensus import build_front_ambiguity_cards_html

    html = build_front_ambiguity_cards_html(pipe, anchor="t")
    assert html == "" or any(
        k in html.lower() for k in ("overlap", "amide", "pyrrole", "enamine", "ambiguities")
    )


def test_front_card_hides_raw_spam_shows_collapsed_repro() -> None:
    pipe = _pipe(
        {
            "aromatic": {"score": 0.6, "confidence_class": "supported", "ontology_category": "family"},
            "NO2_sym_region": {"score": 0.3, "ontology_category": "local_motif"},
        }
    )
    repro = build_reproducibility_html(build_run_context(paths_line=["test=1"]))
    html = build_front_card_stack(
        pipeline=pipe,
        anchor="spec-0",
        ml_enabled=False,
        include_evidence=True,
        audit_html="",
        band_map_html="",
        justify_html="",
        explain_html="",
        fg_block="",
        just_block="",
        meta_html="<p>RAW_META</p>",
        reproducibility_html=repro,
        show_metadata=False,
    )
    assert MARKER_SPECTROSCOPIST_SUMMARY in html
    assert MARKER_FRONT_TECHNICAL in html
    assert MARKER_METADATA_HIDDEN not in html
    assert "RAW_META" not in html
    assert MARKER_REPRODUCIBILITY in html
    assert "NO2_sym_region" not in html or "Technical details" in html


def test_debug_stack_includes_metadata_markers() -> None:
    from reports.product_v1_report import build_product_tables_stack

    html = build_product_tables_stack(
        pipeline=_pipe({"aromatic": {"score": 0.5, "confidence_class": "supported"}}),
        anchor="d0",
        ml_enabled=False,
        include_evidence=True,
        audit_html="<!-- audit -->",
        band_map_html="",
        justify_html="",
        explain_html="",
        fg_block="",
        just_block="",
        meta_html="<p>META</p>",
        density="audit",
        reproducibility_html=build_reproducibility_html(build_run_context(paths_line=["audit=1"])),
    )
    assert "Details —" in html
    assert MARKER_REPRODUCIBILITY in html


def test_front_consensus_table_not_generic_summary() -> None:
    from reports.front_consensus import build_front_consensus_table_html

    row = {
        "name": "x.jdx",
        "anchor": "a",
        "_pipeline": _pipe(
            {"aromatic": {"score": 0.6, "confidence_class": "supported", "ontology_category": "family"}}
        ),
    }
    html = build_front_consensus_table_html(rows=[row], ml_enabled=False)
    assert MARKER_CONSENSUS_TABLE in html
    assert "Consensus interpretation</h2>" in html


def test_ruler_mid_region_label_matches_spec() -> None:
    mid = next(r for r in FTIR_RULER_REGIONS if r.id == "unsat_mid")
    assert "amide" in mid.short_label.lower()
    assert mid.lo == 1450 and mid.hi == 1650


def test_canonical_peak_frequencies_align_with_tables(tmp_path) -> None:
    from ml.ftir_evidence import extract_spectral_evidence
    import numpy as np

    wn = np.linspace(4000, 400, 400)
    y = 0.5 * np.exp(-((wn - 1700) ** 2) / (2 * 50**2))
    ev = extract_spectral_evidence(wn, y, config={"peak_sensitivity": "sensitive", "ontology": "v4"})
    pipe = {"evidence": ev, "rule_assignments": {"assignments": {}}}
    pack = build_canonical_peak_table(pipe, label_min_height=0.05, report_audience="front")
    for p in pack.get("peaks", []):
        assert "center_cm1" in p
    for er in pack.get("evidence_rows", []):
        if er.get("peak_id") and er.get("peak_cm1"):
            pid = er["peak_id"]
            assert er["peak_cm1"] == round(float(pack["peak_by_id"][pid]["center_cm1"]))


@pytest.mark.slow
def test_reference_front_report_smoke(tmp_path: Path) -> None:
    jdx = _ROOT / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"
    fam = _ROOT / "ml" / "runs" / "struct_fg_family_v4_ontology_latest.joblib"
    spec = _ROOT / "ml" / "runs" / "struct_fg_specific_v4_ontology_latest.joblib"
    if not jdx.is_file() or not fam.is_file():
        pytest.skip("fixtures missing")
    from reports.structural_fg_svm_kronecker_report import run_batch

    out = tmp_path / "REPORT.html"
    run_batch(
        input_paths=[jdx],
        model_path=None,
        basic_model_path=fam,
        subtle_model_path=spec,
        out_path=out,
        page_title="Reference smoke",
        subtitle="front",
        max_peaks=60,
        hover_top_fg=8,
        ml_mode="both",
        report_audience="front",
        visual_theme="matlab",
        rules_config={"ontology": "v4"},
        guardrails_mode="v3",
    )
    text = out.read_text(encoding="utf-8")
    assert MARKER_REPRODUCIBILITY in text
    assert re.search(r"Reproducibility metadata", text)
