"""
Chemically coherent FTIR functional-group ontology (v4_ontology).

Separates families, specific functional groups, local spectral motifs,
ambiguity/fallback labels, and artifact/confounder labels for evidence-first
interpretation and optional ML heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Iterable

Category = Literal["family", "specific_fg", "local_motif", "fallback", "artifact"]


@dataclass(frozen=True)
class OntologyEntry:
    label: str
    display_name: str
    category: Category
    parent_labels: tuple[str, ...] = ()
    child_labels: tuple[str, ...] = ()
    related_labels: tuple[str, ...] = ()
    mutually_competing_labels: tuple[str, ...] = ()
    required_evidence_groups: tuple[str, ...] = ()
    supporting_evidence_groups: tuple[str, ...] = ()
    high_risk_false_positive: bool = False
    report_priority: int = 50
    trainable_basic: bool = False
    trainable_subtle: bool = False
    reportable: bool = True
    default_threshold: float = 0.35
    caution_text: str = ""


# --- Trainable SVM targets (v4) -------------------------------------------------

# Family-level ML heads (broad ontology fallbacks / families)
TRAINABLE_FAMILY_V4: tuple[str, ...] = (
    "hydroxy_containing",
    "carbonyl_containing",
    "nitrogen_containing",
    "aromatic_system",
    "C_O_containing",
    "unsaturation_possible",
    "nitro_family",
    "silicon_oxygen_family",
)

# Backward-compatible alias
TRAINABLE_BASIC_V4: tuple[str, ...] = TRAINABLE_FAMILY_V4

# Specific FG ML heads (fine SMARTS when structure available)
TRAINABLE_SPECIFIC_V4: tuple[str, ...] = (
    "alcohol",
    "phenol",
    "carboxylic_acid_OH",
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "aniline_like_amine",
    "aliphatic_amine",
    "amide",
    "pyrrole_like_NH",
    "cyclic_amine",
    "ketone",
    "aldehyde",
    "ester",
    "carboxylic_acid",
    "carbonate",
    "urethane",
    "ether",
    "aryl_ether",
    "nitrile",
    "nitro",
    "alkene",
    "alkyne",
    "aromatic",
    "heteroaromatic",
    "siloxane",
    "silicone_or_silane",
)

# Backward-compatible alias (subtle ≈ specific in v4)
TRAINABLE_SUBTLE_V4: tuple[str, ...] = TRAINABLE_SPECIFIC_V4

TRAINABLE_COMBINED_V4: tuple[str, ...] = tuple(
    dict.fromkeys([*TRAINABLE_FAMILY_V4, *TRAINABLE_SPECIFIC_V4])
)

# Labels that must never be SVM training columns
NON_TRAINABLE_V4: frozenset[str] = frozenset(
    {
        "broad_OH_NH_region",
        "carbonyl_region",
        "C_O_fingerprint_region",
        "aromatic_CC_region",
        "nitrile_alkyne_region",
        "NO2_asym_region",
        "NO2_sym_region",
        "N_O_NO2_overlap",
        "heterocyclic_N_oxide",
        "pyrrole_N_oxide_like",
        "n_oxide_confounded_region",
        "enamine_region",
        "heterocyclic_N_O_region",
        "Si_O_overlap_region",
        "amide_II_region",
        "CH_stretch_region",
        "aliphatic_CH_region",
        "aromatic_CH_region",
        "upper_mid_activity_region",
        "nh_ch_transition_region",
        "aliphatic_CH_present",
        "fingerprint_crowding_region",
        "water_moisture_artifact",
        "CO2_artifact",
        "noise_spike",
        "baseline_artifact",
        "edge_artifact",
        "saturated_peak",
        "preprocessing_sensitive",
        "atr_crystal_fingerprint_overlap",
    }
)

HIGH_RISK_SPECIFIC_V4: frozenset[str] = frozenset(
    {
        "siloxane",
        "silicone_or_silane",
        "nitro",
        "nitrile",
        "alkyne",
        "phenol",
        "amide",
        "ester",
        "aryl_ether",
        "heteroaromatic",
        "pyrrole_like_NH",
        "cyclic_amine",
        "carboxylic_acid",
        "carbonate",
        "urethane",
    }
)

# SMARTS for v4 basic weak labels (structure-only; spectral evidence still required in reports)
V4_FAMILY_SMARTS: tuple[tuple[str, str], ...] = (
    ("hydroxy_containing", "[OX2H1]"),
    ("carbonyl_containing", "[CX3]=[OX1]"),
    (
        "nitrogen_containing",
        "[$([NX3][CX3]=[OX1]),$([NX3;H2,H1;!$(NC=O)]),$([NX3;H0;!$(NC=O)]),$([NX3;H1;!$(NC=O)])]",
    ),
    ("aromatic_system", "a1aaaaa1"),
    (
        "C_O_containing",
        "[$([OD2]([#6])[#6]),$([CX3](=O)[OX2;!$(OC=O)])]",
    ),
    ("unsaturation_possible", "[$([CX3]=[CX3]),$([CX2]#[CX2])]"),
    ("nitro_family", "[$([NX3+](=O)[O-]),$([NX3](=O)=O)]"),
    ("silicon_oxygen_family", "[Si][OX2]"),
)

V4_BASIC_SMARTS = V4_FAMILY_SMARTS

# Band library ids contributing to each local motif (spectral evidence, not FG targets)
LOCAL_MOTIF_BAND_IDS: dict[str, tuple[str, ...]] = {
    "broad_OH_NH_region": ("broad_oh", "alcohol_oh", "phenol_oh", "amine_nh", "amine_nh2", "amide_nh"),
    "carbonyl_region": ("ketone_co", "aldehyde_co", "ester_co", "carboxylic_co", "amide_co", "carbonate_co", "urethane_co"),
    "C_O_fingerprint_region": ("ether_co", "ester_co_o", "aryl_ether_co", "phenolic_co"),
    "aromatic_CC_region": ("aromatic_cc", "heteroaromatic"),
    "nitrile_alkyne_region": ("nitrile_cn", "alkyne_cc"),
    "NO2_asym_region": ("nitro_asym",),
    "NO2_sym_region": ("nitro_sym",),
    "Si_O_overlap_region": ("siloxane_sio", "ether_co", "ester_co_o"),
    "amide_II_region": ("amide_ii",),
    "enamine_region": ("enamine_c_c_cn",),
    "heterocyclic_N_O_region": ("heterocyclic_n_oxide", "n_oxide_high", "n_oxide_low", "pyrrole_n_oxide_like"),
    "n_oxide_confounded_region": ("heterocyclic_n_oxide", "n_oxide_high", "nitro_asym"),
    "heterocyclic_N_oxide": ("heterocyclic_n_oxide", "n_oxide_high", "n_oxide_low"),
    "pyrrole_N_oxide_like": ("pyrrole_n_oxide_like", "pyrrole_nh"),
    "N_O_NO2_overlap": ("nitro_asym", "nitro_sym", "heterocyclic_n_oxide", "n_oxide_high"),
    "CH_stretch_region": ("aliphatic_ch_asym", "aliphatic_ch_sym", "aromatic_ch_stretch"),
    "aliphatic_CH_region": ("aliphatic_ch_asym", "aliphatic_ch_sym"),
    "aromatic_CH_region": ("aromatic_ch_stretch",),
    "upper_mid_activity_region": (),
    "nh_ch_transition_region": ("amide_nh", "pyrrole_nh", "amine_nh"),
    "fingerprint_crowding_region": (),
}

# Motifs scored from evidence["regions"] when band hits are weak (see ftir_evidence._partition_evidence_v4).
LOCAL_MOTIF_REGION_KEYS: dict[str, tuple[str, ...]] = {
    "CH_stretch_region": ("ch_stretch", "aliphatic_ch", "aromatic_ch_stretch"),
    "aliphatic_CH_region": ("aliphatic_ch", "ch_stretch"),
    "aromatic_CH_region": ("aromatic_ch_stretch", "ch_stretch"),
    "upper_mid_activity_region": ("upper_mid_activity",),
    "nh_ch_transition_region": ("nh_ch_transition", "oh_nh_broad"),
    "carbonyl_region": ("carbonyl", "amide_i"),
}


def band_id_to_local_motifs(band_id: str) -> list[str]:
    """Reverse index: which local motifs a band library id can contribute to."""
    bid = str(band_id)
    out: list[str] = []
    for motif, bids in LOCAL_MOTIF_BAND_IDS.items():
        if bid in bids:
            out.append(motif)
    return out


def _e(
    label: str,
    display_name: str,
    category: Category,
    *,
    parents: tuple[str, ...] = (),
    children: tuple[str, ...] = (),
    related: tuple[str, ...] = (),
    competing: tuple[str, ...] = (),
    req_ev: tuple[str, ...] = (),
    sup_ev: tuple[str, ...] = (),
    hr_fp: bool = False,
    rp: int = 50,
    tb: bool = False,
    ts: bool = False,
    rep: bool = True,
    thr: float = 0.35,
    caution: str = "",
) -> OntologyEntry:
    return OntologyEntry(
        label=label,
        display_name=display_name,
        category=category,
        parent_labels=parents,
        child_labels=children,
        related_labels=related,
        mutually_competing_labels=competing,
        required_evidence_groups=req_ev,
        supporting_evidence_groups=sup_ev,
        high_risk_false_positive=hr_fp,
        report_priority=rp,
        trainable_basic=tb,
        trainable_subtle=ts,
        reportable=rep,
        default_threshold=thr,
        caution_text=caution,
    )


def build_v4_ontology() -> dict[str, OntologyEntry]:
    """Full v4 ontology entries keyed by label."""
    o: dict[str, OntologyEntry] = {}

    # A. Families
    for lab, disp, ch, tb in (
        ("hydroxy_family", "Hydroxy / hydrogen-bonding family", ("alcohol", "phenol", "carboxylic_acid_OH"), False),
        (
            "carbonyl_family",
            "Carbonyl family",
            ("ketone", "aldehyde", "ester", "amide", "carboxylic_acid", "carbonate", "urethane"),
            False,
        ),
        (
            "nitrogen_family",
            "Nitrogen functional family",
            ("primary_amine", "secondary_amine", "tertiary_amine", "amide", "nitrile", "nitro"),
            False,
        ),
        ("aromatic_family", "Aromatic systems", ("aromatic", "heteroaromatic"), False),
        ("ether_C_O_family", "C–O / ether family", ("ether", "aryl_ether", "ester"), False),
        ("unsaturation_family", "Unsaturation family", ("alkene", "alkyne"), False),
        ("silicon_oxygen_family", "Silicon–oxygen family", ("siloxane", "silicone_or_silane"), True),
        ("nitro_family", "Nitro family", ("nitro",), False),
        ("sulfur_family", "Sulfur functional family (reserved)", (), False),
        ("halogenated_family", "Halogenated motifs (reserved)", (), False),
    ):
        o[lab] = _e(lab, disp, "family", children=ch, rp=40, tb=tb)

    # B. Specific FGs (subset; aligns with rule engine keys)
    specifics: list[tuple[str, str, tuple[str, ...], tuple[str, ...], bool]] = [
        ("alcohol", "Aliphatic alcohol", ("hydroxy_family",), ("phenol", "carboxylic_acid"), False),
        ("phenol", "Phenol", ("hydroxy_family", "aromatic_family"), ("alcohol",), True),
        ("carboxylic_acid_OH", "Carboxylic acid O–H", ("hydroxy_family", "carbonyl_family"), (), True),
        ("primary_amine", "Primary amine", ("nitrogen_family",), ("amide",), False),
        ("secondary_amine", "Secondary amine", ("nitrogen_family",), ("amide",), False),
        ("tertiary_amine", "Tertiary amine", ("nitrogen_family",), (), False),
        ("aniline_like_amine", "Aniline-like amine", ("nitrogen_family", "aromatic_family"), ("primary_amine",), True),
        ("aliphatic_amine", "Aliphatic amine", ("nitrogen_family",), ("aniline_like_amine",), False),
        ("amide", "Amide", ("carbonyl_family", "nitrogen_family"), ("ester", "ketone"), True),
        ("pyrrole_like_NH", "Pyrrole-like N–H", ("nitrogen_family", "aromatic_family"), ("cyclic_amine",), True),
        ("cyclic_amine", "Cyclic amine", ("nitrogen_family",), ("pyrrole_like_NH",), False),
        ("ketone", "Ketone", ("carbonyl_family",), ("ester", "amide"), False),
        ("aldehyde", "Aldehyde", ("carbonyl_family",), ("ketone",), False),
        ("ester", "Ester", ("carbonyl_family", "ether_C_O_family"), ("ether",), True),
        ("carboxylic_acid", "Carboxylic acid", ("carbonyl_family", "hydroxy_family"), ("ester",), True),
        ("carbonate", "Carbonate", ("carbonyl_family",), ("ester", "ketone"), True),
        ("urethane", "Urethane / carbamate", ("carbonyl_family", "nitrogen_family"), ("amide", "ester"), True),
        ("ether", "Ether", ("ether_C_O_family",), ("ester", "aryl_ether"), False),
        ("aryl_ether", "Aryl ether", ("ether_C_O_family", "aromatic_family"), ("phenol",), True),
        ("nitrile", "Nitrile", ("nitrogen_family",), ("alkyne",), True),
        ("nitro", "Nitro", ("nitro_family",), (), True),
        ("alkene", "Alkene", ("unsaturation_family",), (), False),
        ("alkyne", "Alkyne", ("unsaturation_family",), ("nitrile",), True),
        ("aromatic", "Carbocyclic aromatic", ("aromatic_family",), ("heteroaromatic",), False),
        ("heteroaromatic", "Heteroaromatic", ("aromatic_family",), ("aromatic",), True),
        ("siloxane", "Siloxane", ("silicon_oxygen_family",), ("ether", "aryl_ether"), True),
        ("silicone_or_silane", "Silicone / organosilicon", ("silicon_oxygen_family",), ("siloxane",), True),
    ]
    for lab, disp, par, comp, hr in specifics:
        tb = lab in TRAINABLE_BASIC_V4
        ts = lab in TRAINABLE_SUBTLE_V4
        o[lab] = _e(lab, disp, "specific_fg", parents=par, competing=comp, hr_fp=hr, rp=70, tb=tb, ts=ts)

    # C. Local motifs
    for lab, disp in (
        ("broad_OH_NH_region", "Broad O–H / N–H stretch envelope"),
        ("carbonyl_region", "Carbonyl stretch region"),
        ("C_O_fingerprint_region", "C–O fingerprint / ester-ether window"),
        ("aromatic_CC_region", "Aromatic C=C ring modes"),
        ("nitrile_alkyne_region", "Nitrile / alkyne triple-bond window"),
        ("NO2_asym_region", "Nitro asymmetric region"),
        ("NO2_sym_region", "Nitro symmetric region"),
        ("Si_O_overlap_region", "Si–O vs C–O overlap fingerprint"),
        ("amide_II_region", "Amide II / N–H bend region"),
        ("CH_stretch_region", "C–H stretch region"),
        ("aliphatic_CH_region", "Aliphatic C–H stretch"),
        ("aromatic_CH_region", "Aromatic/sp² C–H stretch"),
        ("upper_mid_activity_region", "Upper mid-IR activity"),
        ("nh_ch_transition_region", "N–H / aromatic C–H shoulder"),
        ("fingerprint_crowding_region", "Crowded fingerprint baseline"),
        ("enamine_region", "Enamine / conjugated C=C–N region"),
        ("heterocyclic_N_O_region", "Heterocyclic N–O / N-oxide-like region"),
        ("n_oxide_confounded_region", "N–O / NO₂ overlap region (nitro requires paired bands)"),
        ("heterocyclic_N_oxide", "Heterocyclic N–O / N-oxide-like"),
        ("pyrrole_N_oxide_like", "Pyrrole N-oxide-like"),
        ("N_O_NO2_overlap", "N–O / NO₂ overlap"),
    ):
        o[lab] = _e(lab, disp, "local_motif", rp=30, tb=False, ts=False, caution="Spectral motif only — not a standalone functional-group assignment.")

    # D. Fallback / ambiguity
    for lab, disp, rel in (
        ("hydroxy_containing", "Hydroxy-containing (ambiguous)", ("alcohol", "phenol", "carboxylic_acid")),
        (
            "nitrogen_containing",
            "Nitrogen-containing (ambiguous)",
            ("primary_amine", "secondary_amine", "tertiary_amine", "amide", "nitrile", "nitro"),
        ),
        ("carbonyl_containing", "Carbonyl-containing (ambiguous)", ("ketone", "ester", "amide", "aldehyde")),
        ("C_O_containing", "C–O-containing (ambiguous)", ("ether", "aryl_ether", "ester")),
        ("aromatic_system", "Aromatic system (ambiguous)", ("aromatic", "heteroaromatic")),
        ("unsaturation_possible", "Unsaturation possible", ("alkene", "alkyne", "nitrile")),
        ("fingerprint_C_O_or_Si_O_overlap", "Fingerprint C–O / Si–O overlap", ("ether", "aryl_ether", "siloxane")),
        ("triple_bond_region_possible", "Triple-bond region signal", ("nitrile", "alkyne")),
        (
            "aliphatic_CH_present",
            "Aliphatic C–H present",
            ("aliphatic_CH_region", "CH_stretch_region"),
        ),
    ):
        par_fam = {
            "hydroxy_containing": ("hydroxy_family",),
            "nitrogen_containing": ("nitrogen_family",),
            "carbonyl_containing": ("carbonyl_family",),
            "C_O_containing": ("ether_C_O_family",),
            "aromatic_system": ("aromatic_family",),
            "unsaturation_possible": ("unsaturation_family",),
            "fingerprint_C_O_or_Si_O_overlap": ("ether_C_O_family", "silicon_oxygen_family"),
            "triple_bond_region_possible": ("unsaturation_family", "nitrogen_family"),
            "aliphatic_CH_present": ("CH_stretch_region", "aliphatic_CH_region"),
        }.get(lab, ())
        caution_fb = (
            "C–H stretch observed — supports organic material but is not specific alone."
            if lab == "aliphatic_CH_present"
            else "Broad fallback label to reduce false positives when subclass evidence is incomplete."
        )
        o[lab] = _e(
            lab,
            disp,
            "fallback",
            parents=par_fam,
            related=rel,
            rp=35,
            tb=lab in TRAINABLE_BASIC_V4,
            ts=False,
            thr=0.18 if lab == "aliphatic_CH_present" else 0.22,
            caution=caution_fb,
        )

    # E. Artifacts
    for lab, disp in (
        ("water_moisture_artifact", "Water / moisture interference"),
        ("CO2_artifact", "Atmospheric CO₂ interference"),
        ("noise_spike", "Noise spike / weak isolated feature"),
        ("baseline_artifact", "Baseline / tilt artifact"),
        ("edge_artifact", "Spectral edge truncation"),
        ("saturated_peak", "Possible detector saturation"),
        ("preprocessing_sensitive", "Preprocessing-sensitive region"),
        (
            "atr_crystal_fingerprint_overlap",
            "ATR / crystal fingerprint overlap",
        ),
    ):
        caution = (
            "ATR/contact/crystal/fingerprint-region overlap may mimic Si–O or C–O bands; "
            "not a supported organosilicon assignment without paired silicon evidence."
            if lab == "atr_crystal_fingerprint_overlap"
            else "Confounder — reduces confidence but does not erase chemical evidence."
        )
        o[lab] = _e(lab, disp, "artifact", rp=10, tb=False, ts=False, caution=caution)

    return o


ONTOLOGY_V4: dict[str, OntologyEntry] = build_v4_ontology()

V4_CONFIDENCE_TERMS = (
    "strong_support",
    "supported",
    "tentative",
    "local_motif_only",
    "ambiguous_family",
    "artifact_limited",
    "not_supported",
)


def ontology_version_key(ontology: str | None) -> str:
    return str(ontology or "v3").lower().strip()


def is_v4(ontology: str | None) -> bool:
    return ontology_version_key(ontology) == "v4"


def get_ontology_entry(label: str, *, ontology: str | None = "v4") -> OntologyEntry | None:
    if not is_v4(ontology):
        return None
    return ONTOLOGY_V4.get(label)


def trainable_labels_v4(model_kind: str) -> list[str]:
    mk = str(model_kind).lower()
    if mk in ("family", "basic"):
        return list(TRAINABLE_FAMILY_V4)
    if mk in ("specific", "subtle"):
        return list(TRAINABLE_SPECIFIC_V4)
    if mk == "combined":
        return list(TRAINABLE_COMBINED_V4)
    raise ValueError(f"Unknown v4 model_kind: {model_kind!r}")


def is_trainable_v4_label(label: str) -> bool:
    lab = str(label)
    if lab in NON_TRAINABLE_V4:
        return False
    ent = ONTOLOGY_V4.get(lab)
    if ent and ent.category in ("local_motif", "artifact"):
        return False
    return lab in TRAINABLE_COMBINED_V4 or lab in TRAINABLE_FAMILY_V4


def label_category(label: str) -> Category | None:
    ent = ONTOLOGY_V4.get(label)
    return ent.category if ent else None


def map_confidence_class_to_interpretation_strength(
    confidence_class: str | None,
    *,
    assignment_type: str | None = None,
    evidence_completeness: str | None = None,
) -> str:
    """Map legacy v3 confidence_class strings to standardized v4 vocabulary."""
    cc = str(confidence_class or "").lower().strip()
    at = str(assignment_type or "").lower()
    evc = str(evidence_completeness or "").lower()
    if evc == "artifact_limited" or cc == "artifact_limited":
        return "artifact_limited"
    if cc == "strong":
        return "strong_support"
    if cc == "supported":
        return "supported"
    if cc == "tentative":
        return "tentative"
    if cc == "local_possible" or at == "local_band_only":
        return "local_motif_only"
    if cc == "not_supported" or not cc:
        return "not_supported"
    if cc in V4_CONFIDENCE_TERMS:
        return cc
    return "tentative"


def validate_ontology_graph(ontology: dict[str, OntologyEntry] | None = None) -> list[str]:
    """Return list of validation errors (empty if OK)."""
    ont = ontology or ONTOLOGY_V4
    errors: list[str] = []
    labels = set(ont.keys())
    if len(labels) != len(ont):
        errors.append("duplicate_label_keys")
    seen_lab: set[str] = set()
    for lab, ent in ont.items():
        if lab in seen_lab:
            errors.append(f"duplicate_id:{lab}")
        seen_lab.add(lab)
        for p in ent.parent_labels:
            if p not in labels:
                errors.append(f"{lab}:missing_parent:{p}")
        for c in ent.child_labels:
            if c not in labels:
                errors.append(f"{lab}:missing_child:{c}")
    for lab in TRAINABLE_FAMILY_V4:
        ent = ont.get(lab)
        if not ent:
            errors.append(f"missing_trainable_family:{lab}")
            continue
        if ent.category == "local_motif" or ent.category == "artifact":
            errors.append(f"trainable_family_bad_category:{lab}")
    for lab in TRAINABLE_SPECIFIC_V4:
        ent = ont.get(lab)
        if not ent:
            continue
        if ent.category == "local_motif" or ent.category == "artifact":
            errors.append(f"trainable_subtle_bad_category:{lab}")
    return errors


def _compile_smarts_patterns(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, Any, str]]:
    from rdkit import Chem

    out: list[tuple[str, Any, str]] = []
    for label, smarts in pairs:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise ValueError(f"Bad SMARTS for {label}: {smarts}")
        out.append((label, pat, smarts))
    return out


_V4_SMARTS_CACHE: list[tuple[str, Any, str]] | None = None


def get_v4_family_smarts_compiled() -> list[tuple[str, Any, str]]:
    global _V4_SMARTS_CACHE
    if _V4_SMARTS_CACHE is None:
        _V4_SMARTS_CACHE = _compile_smarts_patterns(V4_FAMILY_SMARTS)
    return _V4_SMARTS_CACHE


get_v4_basic_smarts_compiled = get_v4_family_smarts_compiled


def infer_v4_family_smarts(mol: Any) -> dict[str, int]:
    """Binary labels for v4 family training columns (structure-derived)."""
    if mol is None:
        return {lab: 0 for lab, _, _ in get_v4_family_smarts_compiled()}
    from rdkit import Chem

    m = Chem.Mol(mol)
    out: dict[str, int] = {}
    for lab, pat, _s in get_v4_family_smarts_compiled():
        try:
            out[lab] = int(m.HasSubstructMatch(pat))
        except Exception:
            out[lab] = 0
    return out


infer_v4_basic_smarts = infer_v4_family_smarts


def infer_v4_specific_smarts(mol: Any, label_names: list[str] | None = None) -> dict[str, int]:
    """Binary labels for v4 specific FG columns via fg_smarts_library."""
    from ml.fg_smarts_library import infer_multilabel_smarts

    targets = list(label_names or TRAINABLE_SPECIFIC_V4)
    if mol is None:
        return {lab: 0 for lab in targets}
    raw = infer_multilabel_smarts(mol, targets)
    out: dict[str, int] = {}
    for lab in targets:
        v = int(raw.get(lab, 0))
        if lab == "carboxylic_acid_OH" and v == 0:
            v = max(v, int(raw.get("carboxylic_acid", 0)))
        if lab == "aliphatic_amine" and v == 0:
            v = max(v, int(raw.get("primary_amine", 0)), int(raw.get("secondary_amine", 0)))
        if lab == "aniline_like_amine" and v == 0:
            v = max(v, int(raw.get("primary_amine", 0)))
        out[lab] = v
    return out


def evidence_feature_vector(
    evidence: dict[str, Any],
    *,
    version: str | None = None,
    feature_set: str | None = None,
) -> tuple[list[float], list[str]]:
    """Spectral evidence features for ML (v1 compact or v2 rich)."""
    from ml.ftir_evidence_features import evidence_feature_vector as _ev_vec

    return _ev_vec(evidence, version=version, feature_set=feature_set)


def smarts_labels_allowed_for_ontology(ontology: str | None) -> set[str] | None:
    """Labels that may appear as SMARTS row targets; None = no restriction (v3)."""
    if not is_v4(ontology):
        return None
    allowed = set(TRAINABLE_BASIC_V4) | set(TRAINABLE_SUBTLE_V4)
    # motifs/artifacts explicitly disallowed
    for lab, ent in ONTOLOGY_V4.items():
        if ent.category in ("local_motif", "artifact") and lab not in allowed:
            continue
    return allowed


def assert_smarts_label_allowed(label: str, *, ontology: str | None) -> None:
    if not is_v4(ontology):
        return
    ent = ONTOLOGY_V4.get(label)
    if ent is None:
        raise ValueError(f"SMARTS/structural label not defined in v4 ontology: {label!r}")
    if ent.category in ("local_motif", "artifact"):
        raise ValueError(f"SMARTS entry must not target local_motif/artifact label: {label}")
