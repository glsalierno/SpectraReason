"""ATR-aware Si-O / siloxane guardrail tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.ftir_atr import infer_measurement_mode, resolve_atr_context
from ml.ftir_rules import assign_functional_groups_from_evidence


def _bm(band_id: str, *, matched: bool = True, support: float = 0.5, wn: float = 1080.0) -> dict:
    return {
        "band_id": band_id,
        "label": band_id,
        "subclass": band_id,
        "region_min_cm1": wn - 40,
        "region_max_cm1": wn + 40,
        "mode": band_id,
        "importance": "required",
        "specificity": "medium",
        "region_rel_max": support,
        "peak_support": support,
        "support_score": support,
        "matched": matched,
        "peaks_near": [{"wn_cm1": wn, "rel_height": 0.4}],
    }


def _atr_evidence(*bands: dict, ratios: dict | None = None) -> dict:
    flags = {
        "fingerprint_crowding": True,
        "atr_crystal_fingerprint_overlap": True,
    }
    return {
        "band_matches": list(bands),
        "peaks": [{"wn_cm1": 1080, "height": 0.5, "rel_height": 0.4}],
        "ratios": ratios or {"siloxane_to_c_o": 0.15},
        "regions": {"fingerprint": {"rel_max": 0.6}, "c_o_stretch": {"rel_max": 0.4}},
        "measurement": {"mode": "ATR", "atr_aware": True, "is_atr": True, "atr_crystal": "diamond"},
        "artifacts": {"flags": flags, "cautions": [], "summary": "atr"},
    }


def _run(ev: dict) -> dict:
    return assign_functional_groups_from_evidence(
        ev, config={"guardrails_mode": "v3", "ontology": "v4"}
    )


def test_infer_atr_from_path() -> None:
    p = Path("examples/spectra/Dopamine_Powder.CSV")
    assert infer_measurement_mode(p, {}) in ("ATR", "unknown")
    ctx = resolve_atr_context(path="polydopamine_ATR_SCRAPED.CSV", md={})
    assert ctx["mode"] == "ATR"
    assert ctx["atr_aware"] is True


def test_atr_single_sio_band_not_supported_siloxane() -> None:
    ev = _atr_evidence(_bm("siloxane_sio", support=0.55, wn=1050))
    r = _run(ev)
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) <= 0.20
    assert sil["confidence_class"] == "local_motif_only"
    assert sil.get("evidence_completeness") == "artifact_limited"
    assert sil.get("assignment_type") == "artifact_limited"
    assert sil["confidence_class"] != "supported"


def test_atr_organic_co_competitors_suppress_siloxane() -> None:
    ev = _atr_evidence(
        _bm("siloxane_sio", support=0.5, wn=1050),
        _bm("ether_co", support=0.62, wn=1280),
        _bm("aryl_ether_co", support=0.58, wn=1250),
        _bm("ester_co_o", support=0.52, wn=1180),
        ratios={"siloxane_to_c_o": 0.12},
    )
    r = _run(ev)
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) <= 0.20
    assert sil["confidence_class"] in ("local_motif_only", "local_possible", "not_supported")


def test_atr_paired_si_regions_can_reach_tentative() -> None:
    ev = _atr_evidence(
        _bm("siloxane_sio", support=0.58, wn=1050),
        _bm("silicone_sic", support=0.52, wn=1280),
        ratios={"siloxane_to_c_o": 0.72},
    )
    r = _run(ev)
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) >= 0.28
    assert sil["confidence_class"] in ("tentative", "supported", "strong")


def test_transmission_less_restrictive_single_band() -> None:
    ev = {
        "band_matches": [_bm("siloxane_sio", support=0.55, wn=1050)],
        "peaks": [{"wn_cm1": 1050, "height": 0.5}],
        "ratios": {"siloxane_to_c_o": 0.5},
        "regions": {},
        "measurement": {"mode": "transmission", "atr_aware": False, "is_atr": False},
        "artifacts": {"flags": {}, "cautions": [], "summary": "none"},
    }
    r = _run(ev)
    sil = r["assignments"]["siloxane"]
    assert float(sil["score"]) <= 0.25
    assert sil["confidence_class"] in ("local_possible", "tentative")
    assert sil.get("confidence_class") != "local_motif_only" or float(sil["score"]) > 0.20
