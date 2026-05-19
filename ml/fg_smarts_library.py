"""
SMARTS library for structural functional-group labeling and optional binary features (v3).

Overlapping SMARTS matches are **expected** (e.g., phenol ∩ aromatic, ester ∩ ether_or_ester).
Labels are chemistry-informed heuristics, not ground truth; see per-entry limitations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

ModelKindStr = Literal["basic", "subtle", "both"]


@dataclass(frozen=True)
class SmartEntry:
    label: str
    smarts: str
    model_kind: ModelKindStr
    description: str
    limitations: str
    priority: int
    parent_label: str | None = None
    ontology_label: str | None = None
    parent_family: str | None = None


def _entries() -> list[SmartEntry]:
    # Broad / basic labels (composite patterns; overlaps are intentional).
    basic: list[SmartEntry] = [
        SmartEntry(
            label="alcohol_or_phenol",
            smarts="[OX2H1]",
            model_kind="basic",
            description="O–H on sp3 or aromatic oxygen (alcohols and phenols).",
            limitations="Acids can contain O–H; does not distinguish alcohol vs phenol vs enol.",
            priority=60,
            parent_label=None,
        ),
        SmartEntry(
            label="amine_or_amide",
            smarts="[$([NX3][CX3]=[OX1]),$([NX3;H2,H1;!$(NC=O)]),$([NX3;H0;!$(NC=O)])]",
            model_kind="basic",
            description="Amide linkage or neutral amine nitrogens (primary/secondary/tertiary), excluding nitro-like patterns by omission.",
            limitations="Quaternary ammonium (NX4) missed; imines/imides can confuse heuristics; nitro handling is imperfect.",
            priority=55,
            parent_label=None,
        ),
        SmartEntry(
            label="carbonyl",
            smarts="[CX3]=[OX1]",
            model_kind="basic",
            description="C=O including ketone, aldehyde, ester, acid, amide, etc.",
            limitations="Very broad; does not identify carbonyl class without subtle labels.",
            priority=50,
            parent_label=None,
        ),
        SmartEntry(
            label="aromatic",
            smarts="a1aaaaa1",
            model_kind="basic",
            description="At least one aromatic ring (carbo- or heteroaromatic).",
            limitations="Requires a full aromatic ring match; some fused systems may need manual review.",
            priority=52,
            parent_label=None,
        ),
        SmartEntry(
            label="ether_or_ester",
            smarts="[$([OD2]([#6])[#6]),$([CX3](=O)[OX2;!$(OC=O)])]",
            model_kind="basic",
            description="Ether oxygen or ester C(=O)–O–C.",
            limitations="Esters also match carbonyl basic label; epoxides are ring ethers but pattern-dependent.",
            priority=48,
            parent_label=None,
        ),
        SmartEntry(
            label="carboxylic_acid",
            smarts="[CX3](=O)[OX2H1]",
            model_kind="basic",
            description="Carboxylic acid motif C(=O)OH.",
            limitations="Salts and zwitterions may not match; overlaps carbonyl/alcohol regions spectrally.",
            priority=58,
            parent_label=None,
        ),
        SmartEntry(
            label="nitrile",
            smarts="[CX2]#[NX1]",
            model_kind="basic",
            description="Carbon–nitrogen triple bond (nitrile / cyano).",
            limitations="May overlap spectroscopically with alkyne stretches in some contexts; SMARTS is structure-only.",
            priority=62,
            parent_label=None,
        ),
        SmartEntry(
            label="nitro",
            smarts="[$([NX3+](=O)[O-]),$([NX3](=O)=O)]",
            model_kind="basic",
            description="Nitro group (multiple formal representations).",
            limitations="Nitro aromatic vs aliphatic not separated at basic level; use subtle patterns for that split.",
            priority=63,
            parent_label=None,
        ),
        SmartEntry(
            label="alkene",
            smarts="[CX3]=[CX3]",
            model_kind="basic",
            description="C=C double bond.",
            limitations="Conjugation/cis–trans not encoded; overlaps many partial structures.",
            priority=45,
            parent_label=None,
        ),
        SmartEntry(
            label="alkyne",
            smarts="[CX2]#[CX2]",
            model_kind="basic",
            description="C≡C triple bond (terminal or internal).",
            limitations="Terminal alkyne C–H not required; can be ambiguous vs nitrile in FTIR without context.",
            priority=46,
            parent_label=None,
        ),
        SmartEntry(
            label="silicon_oxygen",
            smarts="[Si][OX2]",
            model_kind="basic",
            description="Si–O linkage (siloxanyl / silanol / many silicones).",
            limitations="Does not distinguish siloxane vs silanol vs silicate; organosilicon cases vary widely.",
            priority=47,
            parent_label=None,
        ),
    ]

    subtle: list[SmartEntry] = [
        SmartEntry(
            label="alcohol",
            smarts="[OX2H1;!$(O[c])]",
            model_kind="subtle",
            description="Aliphatic / non-aromatic-attached hydroxyl.",
            limitations="May miss some strained/heteroatom adjacency edge cases; overlaps phenol spectrally.",
            priority=70,
            parent_label="alcohol_or_phenol",
        ),
        SmartEntry(
            label="phenol",
            smarts="[OX2H1;$(O[c])]",
            model_kind="subtle",
            description="Hydroxyl attached to an aromatic carbon (phenol-like).",
            limitations="Polyphenols match; salts/phenoxide forms may not.",
            priority=72,
            parent_label="alcohol_or_phenol",
        ),
        SmartEntry(
            label="primary_alcohol",
            smarts="[OX2H1;!$(O[c]);$([CH2][OX2H1])]",
            model_kind="subtle",
            description="Primary alcohol R–CH₂–OH.",
            limitations="Ambiguous in complex scaffolds; SMARTS is approximate.",
            priority=68,
            parent_label="alcohol",
        ),
        SmartEntry(
            label="secondary_alcohol",
            smarts="[OX2H1;!$(O[c]);$([CH1]([#6])([#6]))]",
            model_kind="subtle",
            description="Secondary alcohol.",
            limitations="Steric/congested centers may deviate; overlaps primary/tertiary in borderline cases.",
            priority=66,
            parent_label="alcohol",
        ),
        SmartEntry(
            label="tertiary_alcohol",
            smarts="[OX2H1;!$(O[c]);$([CX4]([#6])([#6])([#6])[OX2H1])]",
            model_kind="subtle",
            description="Tertiary alcohol.",
            limitations="Quaternary centers adjacent to OH can be tricky; keep manual review in mind.",
            priority=64,
            parent_label="alcohol",
        ),
        SmartEntry(
            label="primary_amine",
            smarts="[NX3;H2;!$(NC=O)]",
            model_kind="subtle",
            description="Primary amine (two H on neutral amine).",
            limitations="Salts/zwitterions may not match; overlaps amide N–H in FTIR if mis-assigned.",
            priority=61,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="secondary_amine",
            smarts="[NX3;H1;!$(NC=O)]",
            model_kind="subtle",
            description="Secondary amine.",
            limitations="Often weak N–H in FTIR; can be masked by broad O–H.",
            priority=59,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="tertiary_amine",
            smarts="[NX3;H0;!$(NC=O)]",
            model_kind="subtle",
            description="Tertiary amine.",
            limitations="No N–H stretch; FTIR evidence is indirect; many false negatives in spectra-only workflows.",
            priority=57,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="aniline_like_amine",
            smarts="[NX3;H2,H1;$(a)]",
            model_kind="subtle",
            description="Amino substituent on aromatic ring (aniline-like).",
            limitations="Overlaps other aromatic amines; does not encode electronics of the ring.",
            priority=60,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="aliphatic_amine",
            smarts="[NX3;H2,H1,H0;!$(a);!$(NC=O)]",
            model_kind="subtle",
            description="Aliphatic amine nitrogen.",
            limitations="Broad; includes tertiary aliphatic amines (no H).",
            priority=58,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="amide",
            smarts="[NX3][CX3]=[OX1]",
            model_kind="subtle",
            description="Amide N–C(=O) pattern (amide/urea-like linkage).",
            limitations="Overlaps carbamates/ureas depending on graph; spectroscopy still required for class.",
            priority=65,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="pyrrole_like_NH",
            smarts="[nH]",
            model_kind="subtle",
            description="Heteroaromatic N–H (e.g., pyrrole/indole-like).",
            limitations="Many false positives possible without context; overlaps broad O–H/N–H envelope.",
            priority=62,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="cyclic_amine",
            smarts="[NX3;R]",
            model_kind="subtle",
            description="Aliphatic amine in a ring.",
            limitations="Includes lactams if N matches; verify with carbonyl evidence for lactams.",
            priority=56,
            parent_label="amine_or_amide",
        ),
        SmartEntry(
            label="ketone",
            smarts="[#6][CX3](=[OX1])[#6]",
            model_kind="subtle",
            description="Ketone-like carbonyl with heavy-atom substituents on both sides (heuristic).",
            limitations="Some conjugated / cyclic ketones may still be ambiguous; aldehydes are not targeted.",
            priority=63,
            parent_label="carbonyl",
        ),
        SmartEntry(
            label="aldehyde",
            smarts="[CH1](=O)",
            model_kind="subtle",
            description="Aldehyde C=O.",
            limitations="Sensitive to tautomerism representation; overlap with masked aldehyde motifs.",
            priority=64,
            parent_label="carbonyl",
        ),
        SmartEntry(
            label="ester",
            smarts="[CX3](=O)[OX2][#6;!$(OC(=O))]",
            model_kind="subtle",
            description="Ester linkage (acyl C=O bonded to alkoxy O, excluding carboxylic acids).",
            limitations="Lactones match; distinction from carbonate/anhydride needs extra checks.",
            priority=66,
            parent_label="ether_or_ester",
        ),
        SmartEntry(
            label="carbonate",
            smarts="[OX2][CX3](=[OX1])[OX2]",
            model_kind="subtle",
            description="Carbonate-like O–C(=O)–O.",
            limitations="Overlaps esters in some representations; carbamates can look similar.",
            priority=60,
            parent_label="carbonyl",
        ),
        SmartEntry(
            label="urethane",
            smarts="[NX3][CX3](=[OX1])[OX2]",
            model_kind="subtle",
            description="Urethane / carbamate ester N–C(=O)–O.",
            limitations="Overlaps amides and esters; confirm with band evidence.",
            priority=58,
            parent_label="ether_or_ester",
        ),
        SmartEntry(
            label="ether",
            smarts="[OD2]([#6])[#6]",
            model_kind="subtle",
            description="Dialkyl/alkyl–alkyl ether oxygen (R–O–R′).",
            limitations="Does not require distinguishing aryl vs alkyl without aryl_ether pattern.",
            priority=54,
            parent_label="ether_or_ester",
        ),
        SmartEntry(
            label="aryl_ether",
            smarts="[OD2]([#6a])[#6]",
            model_kind="subtle",
            description="Aryl ether (Ar–O–R).",
            limitations="Ether band overlaps ester C–O; aromatic support should be checked in spectra.",
            priority=56,
            parent_label="ether_or_ester",
        ),
        SmartEntry(
            label="nitrile",
            smarts="[CX2]#[NX1]",
            model_kind="subtle",
            description="Nitrile / cyano.",
            limitations="Terminal alkyne vs nitrile confusion is spectral, not SMARTS.",
            priority=70,
            parent_label="nitrile",
        ),
        SmartEntry(
            label="nitro_aromatic",
            smarts="a[$([NX3+](=O)[O-]),$([NX3](=O)=O)]",
            model_kind="subtle",
            description="Nitro attached to aromatic system.",
            limitations="Depends on aromatic detection; some nitro formats may differ in drawings.",
            priority=71,
            parent_label="nitro",
        ),
        SmartEntry(
            label="nitro_aliphatic",
            smarts="[$([NX3+](=O)[O-]),$([NX3](=O)=O)]-!@[CX4]",
            model_kind="subtle",
            description="Nitro attached to aliphatic carbon.",
            limitations="Sparse in many libraries; may be imbalanced for ML.",
            priority=69,
            parent_label="nitro",
        ),
        SmartEntry(
            label="siloxane",
            smarts="[Si][OX2][Si]",
            model_kind="subtle",
            description="Si–O–Si siloxane linkage.",
            limitations="Does not cover all silicone formulations; chain ends/silanols differ.",
            priority=66,
            parent_label="silicon_oxygen",
        ),
        SmartEntry(
            label="silicone_or_silane",
            smarts="[Si]",
            model_kind="subtle",
            description="Any organosilicon center.",
            limitations="Extremely broad; includes silanes not necessarily silicone polymers.",
            priority=50,
            parent_label="silicon_oxygen",
        ),
        SmartEntry(
            label="aromatic",
            smarts="a1aaaaa1",
            model_kind="subtle",
            description="Aromatic ring present.",
            limitations="Same as basic aromatic; heteroatoms allowed via aromatic flags.",
            priority=53,
            parent_label="aromatic",
        ),
        SmartEntry(
            label="heteroaromatic",
            smarts="[a;$(a@[!#6])]",
            model_kind="subtle",
            description="Aromatic atom bonded to a heteroatom in-ring / attachment heuristic.",
            limitations="Heuristic SMARTS; some heteroaromatics may be missed while fused systems vary.",
            priority=55,
            parent_label="aromatic",
        ),
        SmartEntry(
            label="carboxylic_acid",
            smarts="[CX3](=O)[OX2H1]",
            model_kind="subtle",
            description="Carboxylic acid.",
            limitations="Salts not matched; overlaps ester if mis-drawn.",
            priority=74,
            parent_label="carboxylic_acid",
        ),
        SmartEntry(
            label="alkene",
            smarts="[CX3]=[CX3]",
            model_kind="subtle",
            description="Alkene.",
            limitations="Does not encode cis/trans or conjugation depth.",
            priority=51,
            parent_label="alkene",
        ),
        SmartEntry(
            label="alkyne",
            smarts="[CX2]#[CX2]",
            model_kind="subtle",
            description="Alkyne.",
            limitations="Nitrile ambiguity remains spectral.",
            priority=52,
            parent_label="alkyne",
        ),
    ]

    both_tags: list[SmartEntry] = []
    return basic + subtle + both_tags


SMARTS_ENTRIES: tuple[SmartEntry, ...] = tuple(_entries())

# v3 SMARTS row labels → v4 ontology keys (structure/evidence semantics; not spectral motifs).
SMARTS_DEFAULT_ONTOLOGY_LABEL_V4: dict[str, str] = {
    "alcohol_or_phenol": "hydroxy_containing",
    "amine_or_amide": "nitrogen_containing",
    "carbonyl": "carbonyl_containing",
    "ether_or_ester": "C_O_containing",
    "silicon_oxygen": "silicon_oxygen_family",
    "primary_alcohol": "alcohol",
    "secondary_alcohol": "alcohol",
    "tertiary_alcohol": "alcohol",
    "aniline_like_amine": "primary_amine",
    "aliphatic_amine": "nitrogen_containing",
    "nitro_aromatic": "nitro",
    "nitro_aliphatic": "nitro",
}

_COMPILED_CACHE: list[tuple[SmartEntry, Any]] | None = None


def v4_effective_ontology_label(entry: SmartEntry) -> str:
    """Resolve SMARTS library row to a v4 ontology label (never a spectral motif)."""
    if entry.ontology_label:
        return str(entry.ontology_label)
    return SMARTS_DEFAULT_ONTOLOGY_LABEL_V4.get(entry.label, entry.label)


def smarts_library_version_hash() -> str:
    """Stable hash for metadata: depends on label+smarts+model_kind ordering."""
    payload = [
        {"label": e.label, "smarts": e.smarts, "model_kind": e.model_kind} for e in sorted(SMARTS_ENTRIES, key=lambda x: (x.model_kind, x.label))
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def compile_all_smarts() -> list[tuple[SmartEntry, Any]]:
    from rdkit import Chem

    out: list[tuple[SmartEntry, Any]] = []
    for e in SMARTS_ENTRIES:
        pat = Chem.MolFromSmarts(e.smarts)
        if pat is None:
            raise ValueError(f"SMARTS compile failed for {e.label!r}: {e.smarts!r}")
        out.append((e, pat))
    return out


def get_compiled_patterns() -> list[tuple[SmartEntry, Any]]:
    global _COMPILED_CACHE
    if _COMPILED_CACHE is None:
        _COMPILED_CACHE = compile_all_smarts()
    return _COMPILED_CACHE


def labels_for_model_kind(model_kind: str) -> list[str]:
    mk = str(model_kind).lower()
    if mk == "basic":
        return list(basic_training_labels())
    if mk == "subtle":
        return list(subtle_training_labels())
    if mk == "both":
        seen: set[str] = set()
        out: list[str] = []
        for e in SMARTS_ENTRIES:
            if e.label not in seen:
                seen.add(e.label)
                out.append(e.label)
        return out
    raise ValueError(f"Unknown model_kind: {model_kind!r}")


def basic_training_labels() -> list[str]:
    return [e.label for e in SMARTS_ENTRIES if e.model_kind == "basic"]


def subtle_training_labels() -> list[str]:
    return [e.label for e in SMARTS_ENTRIES if e.model_kind == "subtle"]


def match_entry(mol: Any, pat: Any) -> bool:
    if mol is None:
        return False
    try:
        from rdkit import Chem

        m = Chem.Mol(mol)
        return bool(m.HasSubstructMatch(pat))
    except Exception:
        return False


def binary_vector_for_labels(mol: Any | None, labels: Iterable[str]) -> tuple[list[int], list[str]]:
    """Return (values, feature_names) with names 'smarts_<label>'."""
    label_list = list(labels)
    if mol is None:
        return [0] * len(label_list), [f"smarts_{x}" for x in label_list]
    compiled_by_label: dict[str, Any] = {}
    for e, pat in get_compiled_patterns():
        if e.label in label_list and e.label not in compiled_by_label:
            compiled_by_label[e.label] = pat
    out: list[int] = []
    for lab in label_list:
        pat = compiled_by_label.get(lab)
        out.append(1 if pat is not None and match_entry(mol, pat) else 0)
    return out, [f"smarts_{x}" for x in label_list]


def infer_multilabel_smarts(mol: Any | None, labels: Iterable[str]) -> dict[str, int]:
    vec, _ = binary_vector_for_labels(mol, labels)
    return {l: int(v) for l, v in zip(labels, vec)}


def docs_overlaps_note() -> str:
    return (
        "Overlapping SMARTS labels are expected and informative: for example, phenol ⊆ aromatic for many structures, "
        "and esters match both ether-like C–O and a carbonyl. Reports should not treat overlaps as errors."
    )


def validate_smarts_entries_for_ontology(*, ontology: str | None = "v4") -> list[str]:
    """Return validation errors (empty if OK). SMARTS must not target spectral-only motifs/artifacts."""
    from ml.ftir_ontology import assert_smarts_label_allowed, is_v4

    errs: list[str] = []
    if not is_v4(ontology):
        return errs
    for e in SMARTS_ENTRIES:
        lab = v4_effective_ontology_label(e)
        try:
            assert_smarts_label_allowed(lab, ontology=ontology)
        except ValueError as exc:
            errs.append(f"{e.label!r} → {lab!r}: {exc}")
    return errs
