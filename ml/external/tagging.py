"""Dataset tagging via metadata heuristics, SMARTS, and manual overrides."""

from __future__ import annotations

from typing import Any

from ml.external.ingest_common import keyword_tags_from_metadata, merge_manual_tags

# Optional structure-derived tags when RDKit + SMILES available
_SMARTS_TAG_RULES: list[tuple[str, str]] = [
    ("nitro", "[N+](=O)[O-]"),
    ("n_oxide", "[n+][O-]"),
    ("phenol", "[OX2H1][c]"),
    ("amide", "C(=O)N"),
    ("siloxane", "[Si][OX2][Si]"),
    ("heteroaromatic", "a1naaaa1"),
]


def smarts_tags_from_smiles(smiles: str) -> list[str]:
    try:
        from rdkit import Chem
    except ImportError:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    tags: list[str] = []
    for tag, smarts in _SMARTS_TAG_RULES:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            tags.append(tag)
    return tags


def spectral_heuristic_tags(wn: Any, y: Any) -> list[str]:
    import numpy as np

    wn = np.asarray(wn, dtype=float)
    y = np.asarray(y, dtype=float)
    tags: list[str] = []
    if wn.size < 2:
        return tags
    dy = np.diff(y)
    if float(np.nanstd(dy)) > 0.08:
        tags.append("noisy")
    # crude baseline drift: low-frequency slope vs high-frequency mean
    n = wn.size
    lo = y[: max(1, n // 8)].mean()
    hi = y[-max(1, n // 8) :].mean()
    if abs(float(lo - hi)) > 0.25:
        tags.append("baseline_drift")
    return tags


def tag_spectrum(
    md: dict[str, Any],
    wn: Any = None,
    y: Any = None,
    manual_tags: list[str] | None = None,
) -> list[str]:
    tags = merge_manual_tags(md, manual_tags)
    if wn is not None and y is not None:
        tags = merge_manual_tags({"dataset_tags": tags}, spectral_heuristic_tags(wn, y))
    smiles = md.get("SMILES") or md.get("smiles")
    if smiles:
        tags = merge_manual_tags({"dataset_tags": tags}, smarts_tags_from_smiles(str(smiles)))
    return tags
