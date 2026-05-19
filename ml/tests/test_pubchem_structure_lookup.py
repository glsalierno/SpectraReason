"""Unit tests for PubChem metadata extraction and normalization (no network)."""

from __future__ import annotations

from ml.pubchem_structure_lookup import (
    LOOKUP_STRATEGY,
    CACHE_SCHEMA,
    cas_checksum_is_valid,
    extract_cas_candidates,
    extract_pubchem_query_bundle,
    generate_name_variants,
    normalize_chemical_name,
    score_against_pubchem_synonyms,
    uninvert_registry_name,
    _parse_cache_entry,
    _cache_entry,
)


def test_cas_checksum_valid_ethanol():
    assert cas_checksum_is_valid("64-17-5")


def test_cas_checksum_invalid():
    assert not cas_checksum_is_valid("64-17-4")


def test_extract_cas_from_suffix_filename():
    md = {}
    # Word boundaries allow CAS tokens separated from stems by space/punctuation, not underscore.
    cas_list, inv = extract_cas_candidates(md, r"C:\data\ethanol 64-17-5 IR.jdx")
    queries = [x["query"] for x in cas_list]
    assert "64-17-5" in queries
    assert inv == 0


def test_normalize_preserves_substituent_parentheses():
    s = "2-(O-CHLOROPHENYL)BENZOXAZONE-4"
    assert normalize_chemical_name(s, "noise") == s
    long_s = "ETHANOL, 2,2'-(5-CHLORO-2-ETHOXY PHENYLIMIDO) DI"
    assert "CHLORO" in (normalize_chemical_name(long_s, "noise") or "")


def test_normalize_noise_strips_ir_parenthetical():
    assert normalize_chemical_name("ACETONE (IR)", "noise") == "ACETONE"


def test_metadata_aliases_compound_name():
    md = {"compound_name": "TEST COMPOUND", "cas": "64-17-5"}
    bundle = extract_pubchem_query_bundle(md, None)
    assert any("64-17-5" == x["query"] for x in bundle["cas_candidates"])
    assert len(bundle["name_candidates"]) >= 1


def test_cache_schema_retry_stale_strategy():
    raw = _cache_entry(status="unresolved", resolved={}, attempts=[])
    raw["lookup_strategy"] = "old_strategy"
    _, status, _, current = _parse_cache_entry(raw)
    assert status == "unresolved"
    assert current is False


def test_cache_hit_resolved_roundtrip():
    resolved_payload = {"SMILES": "CCO", "pubchem_cid": 702}
    raw = _cache_entry(status="resolved", resolved=resolved_payload, attempts=[])
    assert raw["cache_schema"] == CACHE_SCHEMA
    assert raw["lookup_strategy"] == LOOKUP_STRATEGY


def test_uninvert_registry_style():
    assert uninvert_registry_name("BENZOIC ACID, 4-CHLORO-").lower() == "4-chlorobenzoic acid"
    assert uninvert_registry_name("ACETIC ACID, TRICHLORO-").lower() == "trichloroacetic acid"
    assert uninvert_registry_name("ETHANOL, 2,2'-IMINODI-").lower() == "2,2'-iminodiethanol"
    mid = "2,2'-(TEST)"
    assert uninvert_registry_name(f"ETHANOL, {mid} DI").endswith("diethanol")


def test_generate_variants_valence_and_noise():
    qs = [x["query"].lower() for x in generate_name_variants("FERROUS OXALATE", "t")]
    assert any("iron(ii)" in q or "iron(2+)" in q for q in qs)
    qs2 = [x["query"] for x in generate_name_variants("MANGANOUS OXALATE", "t")]
    assert any("manganese(ii)" in q.lower() for q in qs2)
    assert normalize_chemical_name("ACETONE (IR)", "noise") == "ACETONE"


def test_score_against_pubchem_synonyms_prefers_list():
    props = {"IUPACName": "2-aminoethanol"}
    syns = ["ethanolamine", "2-aminoethyl alcohol"]
    score, reasons = score_against_pubchem_synonyms("ETHANOLAMINE", props, syns)
    assert score >= 0.30
    assert any("synonym_evidence" in r for r in reasons)
