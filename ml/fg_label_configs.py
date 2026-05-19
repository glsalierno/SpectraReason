"""
Functional-group label definitions for **basic** vs **subtle** structural FG SVM modes.

- **basic**: broad chemistry classes (compatible with legacy v7 reports where possible).
- **subtle**: finer classes from RDKit SMARTS (when structure parses) plus conservative keyword fallbacks.

Training uses the same weak-supervision idea as ``ml.ftir_fg_svm``: metadata text + optional structure.
"""

from __future__ import annotations

from typing import Any, Callable

# Re-export legacy rules for backward-compatible loads
from ml.ftir_fg_svm import FG_RULES as LEGACY_FG_RULES
from ml.ftir_fg_svm import infer_fg_vector as infer_legacy_fg_vector

MODEL_KIND_BASIC = "basic"
MODEL_KIND_SUBTLE = "subtle"
MODEL_KIND_FAMILY = "family"
MODEL_KIND_SPECIFIC = "specific"
MODEL_KIND_COMBINED = "combined"
MODEL_KINDS = (
    MODEL_KIND_BASIC,
    MODEL_KIND_SUBTLE,
    MODEL_KIND_FAMILY,
    MODEL_KIND_SPECIFIC,
    MODEL_KIND_COMBINED,
)

# --- Basic (broad) keyword rules -------------------------------------------------

BASIC_FG_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("alcohol_or_phenol", ("hydroxy", " alcohol", "alcohol,", "ethanol", "methanol", "diol", "phenol", " glycol", " catechol")),
    ("amine_or_amide", ("amine", "amino", "amide", " lactam", " hydrazine", " azide")),
    ("carbonyl", ("ketone", "aldehyde", " quinone", " carbonyl")),
    ("carboxylic_acid", ("carboxylic", "acid,", "acid ", "carboxyl")),
    ("ester", ("ester", "lactone")),
    ("ether", (" ether", "ether,", "epoxide", " furan", " pyran", "oxirane")),
    ("aromatic", ("benzene", "phenyl", "tolyl", "naphth", "anthrac", "pyridine", "indole", "pyrrole")),
    ("halide", ("chloro", "bromo", "fluoro", "iodo", " chloride", " bromide")),
    ("nitrile", ("nitrile", "cyano")),
    ("nitro", ("nitro")),
    ("alkene", ("ethylene", " propene", "butene", "pentene", " alkene", "olefin")),
    ("alkyne", ("yne", " acetylene", "alkyne")),
    ("silicone_or_siloxane", ("siloxane", "silicone", " silane", "polysilox")),
]

# Map legacy v7 label columns → basic labels (for evaluating old NPZ without rebuild)
LEGACY_TO_BASIC: dict[str, str] = {
    "alcohol": "alcohol_or_phenol",
    "amine": "amine_or_amide",
    "carbonyl": "carbonyl",
    "carboxylic_acid": "carboxylic_acid",
    "ester": "ester",
    "ether": "ether",
    "aromatic": "aromatic",
    "halide": "halide",
    "nitrile": "nitrile",
    "nitro": "nitro",
    "alkene": "alkene",
    "alkyne": "alkyne",
}

# --- Subtle keyword fallbacks (when no RDKit mol) --------------------------------

SUBTLE_KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("phenol", ("phenol", "catechol", "hydroquinone", " naphthol")),
    ("alcohol", ("hydroxy", " alcohol", "alcohol,", "diol", " glycol")),
    ("primary_amine", ("primary amine", "1-amine", " amino ")),
    ("secondary_amine", ("secondary amine", "2-amine")),
    ("tertiary_amine", ("tertiary amine", "3-amine")),
    ("amide", ("amide", " lactam")),
    ("aniline_like_amine", ("aniline", " toluidine")),
    ("ketone", ("ketone", " quinone")),
    ("aldehyde", ("aldehyde", " formaldehyde")),
    ("ester", ("ester", "lactone")),
    ("carboxylic_acid", ("carboxylic", "acid,", "carboxyl")),
    ("ether", (" ether", "ether,", "epoxide")),
    ("aryl_ether", ("anisole", " aryl ether", "phenoxy")),
    ("aromatic", ("benzene", "phenyl", "tolyl", "naphth")),
    ("heteroaromatic", ("pyridine", "pyrrole", "furan", "thiophene", "indole", "imidazole")),
    ("nitrile", ("nitrile", "cyano")),
    ("nitro", ("nitro")),
    ("nitro_aromatic", ("nitrobenz", " dinitro", " trinitro")),
    ("alkene", ("ethylene", " alkene", "olefin")),
    ("alkyne", ("yne", "alkyne", "acetylene")),
    ("siloxane", ("siloxane", "polysilox")),
    ("silane_or_silicone", ("silicone", " silane")),
]

# SMARTS patterns: (label, smarts, description)
SUBTLE_SMARTS: list[tuple[str, str, str]] = [
    ("phenol", "[OX2H1;$(a)]", "Ar-OH"),
    ("alcohol", "[OX2H1;!$(a)]", "aliphatic OH"),
    ("primary_alcohol", "[OX2H1;!$(a);$([CH2][OX2H1])]", "primary OH"),
    ("secondary_alcohol", "[OX2H1;!$(a);$([CH][OX2H1])]", "secondary OH"),
    ("tertiary_alcohol", "[OX2H1;!$(a);$([C][OX2H1])]", "tertiary OH"),
    ("primary_amine", "[NX3;H2;!$(NC=O)]", "primary amine"),
    ("secondary_amine", "[NX3;H1;!$(NC=O)]", "secondary amine"),
    ("tertiary_amine", "[NX3;H0;!$(NC=O)]", "tertiary amine"),
    ("amide", "[NX3][CX3]=O", "amide"),
    ("pyrrole_like_NH", "[nH]", "pyrrole-like NH"),
    ("cyclic_amine", "[NX3;R]", "cyclic amine"),
    ("aniline_like_amine", "[NX3;H2,H1;$(a)]", "aniline-like"),
    ("aliphatic_amine", "[NX3;H2,H1,H0;!$(a);!$(NC=O)]", "aliphatic amine"),
    ("ketone", "[CX3]=[OX1]", "ketone/aldehyde carbonyl"),
    ("aldehyde", "[CH1](=O)", "aldehyde"),
    ("ester", "[CX3](=O)[OX2]", "ester/lactone"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", "COOH"),
    ("carbonate", "[OX2][CX3](=[OX1])[OX2]", "carbonate"),
    ("urethane", "[NX3][CX3](=[OX1])[OX2]", "urethane/urea-like"),
    ("ether", "[OD2]([#6])[#6]", "aliphatic ether"),
    ("aryl_ether", "[OD2]([#6a])[#6]", "aryl ether"),
    ("aromatic", "a1aaaaa1", "aromatic ring"),
    ("heteroaromatic", "a1aa1aa1", "heteroaromatic"),
    ("nitrile", "[CX2]#N", "nitrile"),
    ("nitro", "[$([NX3](=O)=O)]", "nitro"),
    ("nitro_aromatic", "a[$([NX3](=O)=O)]", "aromatic nitro"),
    ("alkene", "[CX3]=[CX3]", "alkene"),
    ("alkyne", "[CX2]#[CX2]", "alkyne"),
    ("siloxane", "[Si][OX2][Si]", "siloxane"),
    ("silane_or_silicone", "[Si]", "organosilicon"),
]


def _metadata_text(md: dict[str, Any]) -> str:
    parts = [md.get("name"), md.get("title"), md.get("formula")]
    return " ".join(str(p or "") for p in parts).lower()


def _match_keywords(rules: list[tuple[str, tuple[str, ...]]], text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, kws in rules:
        out[label] = int(any(kw in text for kw in kws))
    return out


def _require_rdkit():
    from rdkit import Chem  # noqa: F401


def _compile_smarts_patterns() -> list[tuple[str, Any, str]]:
    _require_rdkit()
    from rdkit import Chem

    compiled: list[tuple[str, Any, str]] = []
    for label, smarts, desc in SUBTLE_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled.append((label, pat, desc))
    return compiled


_SMARTS_COMPILED: list[tuple[str, Any, str]] | None = None


def get_smarts_compiled() -> list[tuple[str, Any, str]]:
    global _SMARTS_COMPILED
    if _SMARTS_COMPILED is None:
        _SMARTS_COMPILED = _compile_smarts_patterns()
    return _SMARTS_COMPILED


def infer_fg_from_smarts(mol: Any) -> dict[str, int]:
    """Structure-based subtle labels (binary)."""
    if mol is None:
        return {label: 0 for label, _, _ in SUBTLE_SMARTS}
    _require_rdkit()
    from rdkit import Chem

    mol_use = Chem.Mol(mol)
    out: dict[str, int] = {}
    for label, pat, _desc in get_smarts_compiled():
        try:
            out[label] = int(mol_use.HasSubstructMatch(pat))
        except Exception:
            out[label] = 0
    return out


def featurize_subtle_smarts(mol: Any | None) -> tuple[list[int], list[str]]:
    """Binary SMARTS match vector (same order as ``subtle_label_names()``)."""
    names = subtle_label_names()
    if mol is None:
        return [0] * len(names), [f"smarts_{n}" for n in names]
    vec = infer_fg_from_smarts(mol)
    return [int(vec.get(n, 0)) for n in names], [f"smarts_{n}" for n in names]


def basic_label_names() -> list[str]:
    return [t[0] for t in BASIC_FG_RULES]


def subtle_label_names() -> list[str]:
    """Ordered unique labels: SMARTS-defined first, then keyword-only extras."""
    seen: set[str] = set()
    names: list[str] = []
    for label, _, _ in SUBTLE_SMARTS:
        if label not in seen:
            seen.add(label)
            names.append(label)
    for label, _ in SUBTLE_KEYWORD_RULES:
        if label not in seen:
            seen.add(label)
            names.append(label)
    return names


def get_label_rules(model_kind: str) -> list[tuple[str, tuple[str, ...]]]:
    if model_kind == MODEL_KIND_BASIC:
        return list(BASIC_FG_RULES)
    if model_kind == MODEL_KIND_SUBTLE:
        return list(SUBTLE_KEYWORD_RULES)
    raise ValueError(f"Unknown model_kind: {model_kind!r}")


def infer_fg_vector(
    md: dict[str, Any],
    *,
    model_kind: str = MODEL_KIND_BASIC,
    mol: Any | None = None,
) -> dict[str, int]:
    """
  Weak labels for one compound.

  - **basic**: keyword rules only (broad).
  - **subtle**: SMARTS on ``mol`` when available; keyword rules fill gaps; union OR logic.
    """
    if model_kind == MODEL_KIND_BASIC:
        return _match_keywords(BASIC_FG_RULES, _metadata_text(md))

    kw = _match_keywords(SUBTLE_KEYWORD_RULES, _metadata_text(md))
    sm = infer_fg_from_smarts(mol) if mol is not None else {}
    names = subtle_label_names()
    out: dict[str, int] = {}
    for lab in names:
        out[lab] = int(max(kw.get(lab, 0), sm.get(lab, 0)))
    return out


def remap_legacy_y_columns(
    legacy_names: list[str],
    Y: Any,
    *,
    target_kind: str = MODEL_KIND_BASIC,
) -> tuple[list[str], Any]:
    """
    Map an NPZ label matrix trained on legacy 12-D names to basic label space.

    Used for quick smoke training without NIST rebuild.
    """
    import numpy as np

    Y = np.asarray(Y, dtype=int)
    if target_kind != MODEL_KIND_BASIC:
        raise ValueError("remap_legacy_y_columns only supports basic kind")
    target_names = basic_label_names()
    out = np.zeros((Y.shape[0], len(target_names)), dtype=int)
    name_to_j = {n: i for i, n in enumerate(legacy_names)}
    for j_new, tname in enumerate(target_names):
        for leg, t in LEGACY_TO_BASIC.items():
            if t != tname:
                continue
            j_old = name_to_j.get(leg)
            if j_old is not None:
                out[:, j_new] = np.maximum(out[:, j_new], Y[:, j_old])
    return target_names, out


def filter_labels_by_counts(
    label_names: list[str],
    Y: Any,
    *,
    min_positives: int,
) -> tuple[list[str], Any, list[str], dict[str, int]]:
    """Drop labels with fewer than ``min_positives`` positives or no variance."""
    import numpy as np

    Y = np.asarray(Y, dtype=int)
    counts = {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)}
    keep: list[str] = []
    dropped: list[str] = []
    for j, lab in enumerate(label_names):
        col = Y[:, j]
        n_pos = int(col.sum())
        if n_pos < min_positives:
            dropped.append(f"{lab}:only_{n_pos}_positives")
            continue
        if col.max() == 0 or col.min() == col.max():
            dropped.append(f"{lab}:no_variance")
            continue
        keep.append(lab)
    if not keep:
        return [], Y[:, :0], dropped, counts
    idx = [label_names.index(l) for l in keep]
    return keep, Y[:, idx], dropped, counts
