"""
Targeted confounder expansion classes for external FTIR indexing.

Each class defines how to recognize spectra (tags, keywords, SMARTS) and minimum
coverage targets for gap analysis. See ``docs/TARGET_EXTERNAL_EXPANSION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["true_positive", "hard_negative", "supporting"]


@dataclass(frozen=True)
class TargetClass:
    class_id: str
    role: Role
    problem: str  # nitro | amide | siloxane
    label: str
    example_compounds: tuple[str, ...]
    tag_rules: tuple[str, ...]  # dataset_tags or synthetic tags from matcher
    keyword_any: tuple[str, ...] = ()
    keyword_all: tuple[str, ...] = ()
    smarts: str | None = None
    preferred_source: str = "sdbs_aist"
    expected_ambiguity: str = ""
    minimum_count: int = 5
    benchmark_manifest: str | None = None


# --- Nitro problem set ---
NITRO_POSITIVE = TargetClass(
    class_id="nitro_positive",
    role="true_positive",
    problem="nitro",
    label="True nitro (R–NO₂)",
    example_compounds=(
        "nitrobenzene",
        "p-nitrotoluene",
        "2,4-dinitrotoluene",
        "nitromethane",
        "m-nitroaniline",
    ),
    tag_rules=("nitro",),
    keyword_any=("nitro", "dinitro", "trinitro"),
    smarts="[N+](=O)[O-]",
    preferred_source="sdbs_aist",
    expected_ambiguity="ν_as(NO₂) ~1520–1550, ν_s ~1340–1370; aromatic NO₂ vs aliphatic shift.",
    minimum_count=10,
    benchmark_manifest="nitro_vs_noxide_manifest.json",
)

NITRO_HN_N_OXIDE = TargetClass(
    class_id="nitro_hn_n_oxide",
    role="hard_negative",
    problem="nitro",
    label="Heterocyclic N-oxide",
    example_compounds=(
        "pyridine 1-oxide",
        "pyridazine 1-oxide",
        "quinoline 1-oxide",
        "morpholine n-oxide",
    ),
    tag_rules=("n_oxide",),
    keyword_any=("n-oxide", "n oxide", "pyridine n-oxide", "pyridazine"),
    smarts="[n+][O-]",
    preferred_source="sdbs_aist",
    expected_ambiguity="N–O stretch overlaps nitro region; lacks symmetric NO₂ pair.",
    minimum_count=8,
    benchmark_manifest="nitro_vs_noxide_manifest.json",
)

NITRO_HN_NITROSO = TargetClass(
    class_id="nitro_hn_nitroso",
    role="hard_negative",
    problem="nitro",
    label="Nitroso (R–N=O)",
    example_compounds=("nitrosobenzene", "cyclohexanone oxime", "nitrosodimethylamine"),
    tag_rules=("nitroso",),
    keyword_any=("nitroso", "n-oxo"),
    smarts="[NX2]=O",
    preferred_source="sdbs_aist",
    expected_ambiguity="N=O ~1450–1500 cm⁻¹; weaker than nitro; no second NO stretch.",
    minimum_count=5,
    benchmark_manifest="nitro_vs_noxide_manifest.json",
)

NITRO_HN_HETEROAROMATIC = TargetClass(
    class_id="nitro_hn_heteroaromatic",
    role="hard_negative",
    problem="nitro",
    label="Heteroaromatic (no nitro)",
    example_compounds=("pyridine", "pyrrole", "imidazole", "indole", "furan", "thiophene"),
    tag_rules=("heteroaromatic",),
    keyword_any=("pyridin", "pyrrol", "imidaz", "indol", "furan", "thiophen", "oxazole"),
    smarts="a1naaaa1",
    preferred_source="sdbs_aist",
    expected_ambiguity="Ring breathing 1400–1600; can crowd fingerprint near nitro bands.",
    minimum_count=10,
    benchmark_manifest="nitro_vs_noxide_manifest.json",
)

NITRO_HN_ENAMINE = TargetClass(
    class_id="nitro_hn_enamine",
    role="hard_negative",
    problem="nitro",
    label="Enamine / vinylogous amine",
    example_compounds=("morpholine enamine", "dimethylaminocrotonitrile"),
    tag_rules=("enamine",),
    keyword_any=("enamine", "enamino", "vinylogous"),
    smarts="[CX3]=[CX3][NX3]",
    preferred_source="sdbs_aist",
    expected_ambiguity="C=C + C–N; no NO₂; N–H if secondary enamine.",
    minimum_count=5,
    benchmark_manifest="nitro_vs_noxide_manifest.json",
)

# --- Amide problem set ---
AMIDE_POSITIVE = TargetClass(
    class_id="amide_positive",
    role="true_positive",
    problem="amide",
    label="Primary/secondary/tertiary amide",
    example_compounds=(
        "acetamide",
        "benzamide",
        "caprolactam",
        "nylon 6",
        "dmf",
    ),
    tag_rules=("amide",),
    keyword_any=("amide", "lactam", "nylon"),
    smarts="C(=O)N",
    preferred_source="sdbs_aist",
    expected_ambiguity="Amide I ~1650–1680, Amide II ~1550; polymer broadening in nylon.",
    minimum_count=10,
    benchmark_manifest="amide_vs_enamine_manifest.json",
)

AMIDE_HN_ENAMINE = TargetClass(
    class_id="amide_hn_enamine",
    role="hard_negative",
    problem="amide",
    label="Enamine",
    example_compounds=("morpholine enamine", "pyrrolidine enamine"),
    tag_rules=("enamine",),
    keyword_any=("enamine", "enamino"),
    smarts="[CX3]=[CX3][NX3]",
    preferred_source="sdbs_aist",
    expected_ambiguity="C=C + N lone pair; weak or absent amide I carbonyl.",
    minimum_count=8,
    benchmark_manifest="amide_vs_enamine_manifest.json",
)

AMIDE_HN_PYRROLE = TargetClass(
    class_id="amide_hn_pyrrole",
    role="hard_negative",
    problem="amide",
    label="Pyrrole / indole N–H",
    example_compounds=("pyrrole", "indole", "carbazole", "1H-indol-5-ol"),
    tag_rules=("heteroaromatic",),
    keyword_any=("pyrrol", "indol", "carbazol"),
    smarts="[nH]1cccc1",
    preferred_source="sdbs_aist",
    expected_ambiguity="N–H broad ~3200–3400; no amide I; heteroaromatic fingerprint.",
    minimum_count=8,
    benchmark_manifest="amide_vs_enamine_manifest.json",
)

AMIDE_HN_IMIDE = TargetClass(
    class_id="amide_hn_imide",
    role="hard_negative",
    problem="amide",
    label="Imide / cyclic imide",
    example_compounds=("phthalimide", "succinimide", "n-hydroxysuccinimide"),
    tag_rules=("amide",),
    keyword_any=("imide", "succinimide", "phthalimide"),
    smarts="C(=O)NC(=O)",
    preferred_source="sdbs_aist",
    expected_ambiguity="Twin carbonyls; amide I often split/broadened vs simple amide.",
    minimum_count=5,
    benchmark_manifest="amide_vs_enamine_manifest.json",
)

AMIDE_HN_CONJUGATED = TargetClass(
    class_id="amide_hn_conjugated_amide",
    role="hard_negative",
    problem="amide",
    label="Conjugated / aromatic amide",
    example_compounds=("nicotinamide", "acrylamide", "cinnamamide"),
    tag_rules=("amide", "heteroaromatic"),
    keyword_any=("nicotinamide", "acrylamide", "cinnam"),
    smarts="C(=O)Nc",
    preferred_source="sdbs_aist",
    expected_ambiguity="Lower amide I frequency; aromatic bands overlap amide II.",
    minimum_count=5,
    benchmark_manifest="amide_vs_enamine_manifest.json",
)

# --- Siloxane problem set ---
SILOXANE_POSITIVE = TargetClass(
    class_id="siloxane_positive",
    role="true_positive",
    problem="siloxane",
    label="Siloxane / silicone (Si–O–Si)",
    example_compounds=(
        "polydimethylsiloxane",
        "hexamethyldisiloxane",
        "octamethylcyclotetrasiloxane",
        "silicone oil",
    ),
    tag_rules=("siloxane",),
    keyword_any=("siloxane", "silicone", "pdms", "disiloxane", "polysilox"),
    smarts="[Si][OX2][Si]",
    preferred_source="open_polymer_atr",
    expected_ambiguity="Si–O–Si ~1000–1100 cm⁻¹; CH₃ rocking ~1260.",
    minimum_count=8,
    benchmark_manifest="siloxane_vs_CO_manifest.json",
)

SILOXANE_HN_ETHER_ESTER = TargetClass(
    class_id="siloxane_hn_ether_ester",
    role="hard_negative",
    problem="siloxane",
    label="Ether / ester C–O",
    example_compounds=("diethyl ether", "ethyl acetate", "peg", "cellulose acetate"),
    tag_rules=(),
    keyword_any=("ether", "ester", "acetate", "polyethylene glycol", "peg"),
    smarts="[OD2]([#6])[#6]",
    preferred_source="sdbs_aist",
    expected_ambiguity="C–O ~1050–1150; lacks Si–CH₃ 1260 shoulder pattern.",
    minimum_count=10,
    benchmark_manifest="siloxane_vs_CO_manifest.json",
)

SILOXANE_HN_POLYMER_CO = TargetClass(
    class_id="siloxane_hn_polymer_co",
    role="hard_negative",
    problem="siloxane",
    label="C–O-rich polymer (non-silicone)",
    example_compounds=("nylon", "pet", "pmma", "pva", "epoxy"),
    tag_rules=("polymer",),
    keyword_any=("nylon", "polyester", "pmma", "polyvinyl", "epoxy", "polyurethane"),
    preferred_source="open_polymer_atr",
    expected_ambiguity="Strong C=O or C–O; ester/amide carbonyl; no Si–O–Si.",
    minimum_count=8,
    benchmark_manifest="siloxane_vs_CO_manifest.json",
)

SILOXANE_HN_ATR_POLYMER = TargetClass(
    class_id="siloxane_hn_atr_polymer",
    role="supporting",
    problem="siloxane",
    label="ATR polymer (contact / baseline)",
    example_compounds=("nylon 6 ATR", "pdms ATR", "epoxy coating ATR"),
    tag_rules=("polymer", "atr"),
    keyword_any=("atr", "powder", "film", "coating"),
    preferred_source="open_polymer_atr",
    expected_ambiguity="ATR distortion; baseline drift; moisture-like 1600–1700.",
    minimum_count=6,
    benchmark_manifest="siloxane_vs_CO_manifest.json",
)

ALL_TARGET_CLASSES: tuple[TargetClass, ...] = (
    NITRO_POSITIVE,
    NITRO_HN_N_OXIDE,
    NITRO_HN_NITROSO,
    NITRO_HN_HETEROAROMATIC,
    NITRO_HN_ENAMINE,
    AMIDE_POSITIVE,
    AMIDE_HN_ENAMINE,
    AMIDE_HN_PYRROLE,
    AMIDE_HN_IMIDE,
    AMIDE_HN_CONJUGATED,
    SILOXANE_POSITIVE,
    SILOXANE_HN_ETHER_ESTER,
    SILOXANE_HN_POLYMER_CO,
    SILOXANE_HN_ATR_POLYMER,
)

TARGET_BY_ID: dict[str, TargetClass] = {c.class_id: c for c in ALL_TARGET_CLASSES}


def _blob(md: dict[str, Any]) -> str:
    return " ".join(str(md.get(k) or "") for k in ("title", "name", "formula", "state")).lower()


def _tags(md: dict[str, Any]) -> set[str]:
    return {str(t).lower() for t in (md.get("dataset_tags") or [])}


def _smarts_match(smiles: str, smarts: str) -> bool:
    try:
        from rdkit import Chem
    except ImportError:
        return False
    mol = Chem.MolFromSmiles(smiles)
    pat = Chem.MolFromSmarts(smarts)
    if mol is None or pat is None:
        return False
    return mol.HasSubstructMatch(pat)


def _exclude_blob(blob: str, tc: TargetClass) -> bool:
    """Reject nitroso/n-oxide strings when matching true nitro positives."""
    if tc.class_id == "nitro_positive":
        if "nitroso" in blob or "n-oxide" in blob or "n oxide" in blob:
            return True
    if tc.class_id == "nitro_hn_heteroaromatic":
        if "nitro" in blob and "nitroso" not in blob:
            if any(x in blob for x in ("dinitro", "trinitro", "nitrobenz", "nitrotol")):
                return True
    return False


def match_target_class(md: dict[str, Any], tc: TargetClass) -> bool:
    tags = _tags(md)
    blob = _blob(md)
    if _exclude_blob(blob, tc):
        return False

    if tc.tag_rules:
        tag_hit = all(t in tags for t in tc.tag_rules)
    else:
        tag_hit = False
    kw_hit = bool(tc.keyword_any and any(k in blob for k in tc.keyword_any))
    if tc.keyword_all and not all(k in blob for k in tc.keyword_all):
        kw_hit = False

    smiles = md.get("SMILES") or md.get("smiles")
    smarts_hit = bool(tc.smarts and smiles and _smarts_match(str(smiles), tc.smarts))

    if tc.tag_rules:
        return tag_hit or (kw_hit and not tc.tag_rules) or smarts_hit
    return kw_hit or smarts_hit


def classify_spectrum(md: dict[str, Any]) -> list[str]:
    """Return all matching target class_ids (multi-label allowed)."""
    return [tc.class_id for tc in ALL_TARGET_CLASSES if match_target_class(md, tc)]


def manifest_spec(manifest_name: str) -> dict[str, Any]:
    """Build benchmark manifest JSON structure."""
    classes = [c for c in ALL_TARGET_CLASSES if c.benchmark_manifest == manifest_name]
    if not classes:
        return {}
    problem = classes[0].problem
    return {
        "manifest_version": "1.0",
        "name": manifest_name.replace("_manifest.json", "").replace(".json", ""),
        "problem": problem,
        "description": f"Targeted confounder expansion for {problem} classification.",
        "true_positive_classes": [c.class_id for c in classes if c.role == "true_positive"],
        "hard_negative_classes": [c.class_id for c in classes if c.role == "hard_negative"],
        "supporting_classes": [c.class_id for c in classes if c.role == "supporting"],
        "classes": [
            {
                "class_id": c.class_id,
                "role": c.role,
                "label": c.label,
                "example_compounds": list(c.example_compounds),
                "tag_rules": list(c.tag_rules),
                "keyword_any": list(c.keyword_any),
                "smarts": c.smarts,
                "preferred_source": c.preferred_source,
                "expected_ambiguity": c.expected_ambiguity,
                "minimum_count": c.minimum_count,
            }
            for c in classes
        ],
        "expected_behavior": _expected_behavior(problem),
        "known_ambiguities": _known_ambiguities(problem),
        "ingested_spectra": [],
        "coverage": {},
    }


def _expected_behavior(problem: str) -> str:
    return {
        "nitro": "Separate nitro (symmetric NO₂ pair) from N-oxide/nitroso/heteroaromatic false positives.",
        "amide": "Separate amide I/II from enamine, pyrrole N–H, and imide twin carbonyls.",
        "siloxane": "Separate Si–O–Si from ether/ester C–O and carbonyl-rich polymers.",
    }.get(problem, "")


def _known_ambiguities(problem: str) -> list[str]:
    return {
        "nitro": [
            "Aromatic nitro vs heterocyclic N-oxide in 1300–1600 cm⁻¹",
            "Nitroso N=O vs nitro asymmetric stretch",
        ],
        "amide": [
            "Imide vs simple amide carbonyl splitting",
            "Pyrrole N–H vs secondary amine N–H",
            "Conjugated amide lowered amide I",
        ],
        "siloxane": [
            "Ester C–O near 1100 vs Si–O–Si",
            "Silicate filler vs silicone",
            "ATR baseline distortion in polymers",
        ],
    }.get(problem, [])
