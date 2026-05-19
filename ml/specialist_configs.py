"""Specialist v4 model kinds: chemically related label subsets + hard-negative groups."""

from __future__ import annotations

SPECIALIST_MODEL_KINDS: tuple[str, ...] = (
    "hydroxy_specialist",
    "carbonyl_specialist",
    "nitrogen_specialist",
    "c_o_specialist",
    "triple_bond_specialist",
    "silicon_specialist",
    "nitro_specialist",
)

SPECIALIST_LABELS: dict[str, list[str]] = {
    "hydroxy_specialist": [
        "alcohol",
        "phenol",
        "carboxylic_acid_OH",
        "hydroxy_containing",
    ],
    "carbonyl_specialist": [
        "ketone",
        "aldehyde",
        "ester",
        "amide",
        "carboxylic_acid",
        "carbonate",
        "urethane",
        "carbonyl_containing",
    ],
    "nitrogen_specialist": [
        "primary_amine",
        "secondary_amine",
        "tertiary_amine",
        "amide",
        "pyrrole_like_NH",
        "cyclic_amine",
        "nitrogen_containing",
    ],
    "c_o_specialist": [
        "ether",
        "aryl_ether",
        "ester",
        "alcohol",
        "phenol",
        "siloxane",
        "C_O_containing",
    ],
    "triple_bond_specialist": [
        "nitrile",
        "alkyne",
        "unsaturation_possible",
    ],
    "silicon_specialist": [
        "siloxane",
        "silicone_or_silane",
        "silicon_oxygen_family",
    ],
    "nitro_specialist": [
        "nitro",
        "nitro_family",
    ],
}

SPECIALIST_HARD_NEGATIVES: dict[str, dict[str, list[str]]] = {
    "hydroxy_specialist": {
        "phenol": ["alcohol", "aromatic", "aryl_ether"],
        "alcohol": ["phenol", "ether"],
    },
    "carbonyl_specialist": {
        "amide": ["primary_amine", "secondary_amine", "ketone", "ester", "carboxylic_acid"],
        "ester": ["ether", "ketone", "aldehyde", "amide", "carboxylic_acid"],
    },
    "nitrogen_specialist": {
        "amide": ["ketone", "ester", "carboxylic_acid"],
        "pyrrole_like_NH": ["primary_amine", "heteroaromatic"],
    },
    "c_o_specialist": {
        "ester": ["ether", "ketone", "aryl_ether"],
        "ether": ["ester", "siloxane"],
    },
    "triple_bond_specialist": {
        "nitrile": ["alkyne"],
        "alkyne": ["nitrile"],
    },
    "silicon_specialist": {
        "siloxane": ["ether", "aryl_ether", "alcohol", "phenol", "ester"],
        "silicone_or_silane": ["ether", "aryl_ether", "alcohol", "phenol", "ester"],
    },
    "nitro_specialist": {
        "nitro": ["aromatic", "heteroaromatic"],
    },
}


def is_specialist_model_kind(model_kind: str) -> bool:
    return str(model_kind) in SPECIALIST_MODEL_KINDS


def specialist_label_names(model_kind: str) -> list[str]:
    mk = str(model_kind)
    if mk not in SPECIALIST_LABELS:
        raise ValueError(f"Unknown specialist model_kind: {mk!r}")
    return list(SPECIALIST_LABELS[mk])
