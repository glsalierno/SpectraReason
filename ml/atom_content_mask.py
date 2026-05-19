"""
Optional **atom-aware masking** of structural FG SVM probabilities at inference time.

Training labels come from keyword rules on metadata text, which can correlate with
spectral shape in ways that do not respect true elemental composition. When the
caller supplies an empirical **formula** or explicit **elements**, we can zero out
labels that require atoms that are absent (currently: **halide** if no F/Cl/Br/I/At).
"""

from __future__ import annotations

import re
from typing import Any

_HALOGENS = frozenset({"F", "Cl", "Br", "I", "At"})

# Only these two-letter tokens are consumed as pairs. (Avoids ``NO2`` → Nobelium ``No``.)
_TWO_LETTER = frozenset(
    {
        "Cl",
        "Br",
        "Si",
        "Se",
        "Na",
        "Mg",
        "Al",
        "Ca",
        "Fe",
        "Cu",
        "Zn",
        "Co",
        "Ni",
        "Mn",
        "Cr",
        "Li",
        "Be",
        "Pb",
        "Sn",
        "As",
        "Sb",
        "Te",
        "Bi",
        "Cd",
        "Hg",
        "Ag",
        "Au",
        "Pt",
        "Pd",
        "Ru",
        "Rh",
        "Ir",
        "Ta",
        "W",
        "Mo",
        "Nb",
        "V",
        "Ti",
        "Sc",
        "Ge",
        "Ga",
        "Kr",
        "Xe",
        "Ar",
        "He",
        "Ba",
        "Sr",
        "Ra",
        "Cs",
        "Rb",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Ac",
        "Th",
        "Pa",
    }
)
_TWO_LOWER = {x.lower() for x in _TWO_LETTER}


def _one_letter_elements() -> frozenset[str]:
    from rdkit.Chem import GetPeriodicTable

    pt = GetPeriodicTable()
    return frozenset(pt.GetElementSymbol(z) for z in range(1, 119) if len(pt.GetElementSymbol(z)) == 1)


def elements_from_empirical_formula(formula: str) -> set[str] | None:
    """
    Tokenize a Hill-style empirical formula (e.g. ``C8H11NO2``).

    Two-letter symbols are only recognized if in a **chemistry whitelist** (so ``N`` + ``O``
    in ``NO2`` is not read as Nobelium ``No``). Returns ``None`` if nothing was parsed.
    """
    s = "".join(str(formula).split())
    if not s:
        return None
    singles = _one_letter_elements()
    i = 0
    found: set[str] = set()
    n = len(s)
    while i < n:
        if s[i].isdigit():
            i += 1
            continue
        if i + 2 <= n and s[i].isupper() and s[i + 1].islower():
            pair = s[i : i + 2]
            if pair.lower() in _TWO_LOWER:
                canon = next(x for x in _TWO_LETTER if x.lower() == pair.lower())
                found.add(canon)
                i += 2
                continue
        if s[i].isupper() and s[i] in singles:
            found.add(s[i])
            i += 1
            continue
        i += 1
    return found if found else None


def infer_element_symbols(md: dict[str, Any]) -> set[str] | None:
    """
    Derive element symbols from metadata.

    Recognized keys:

    - ``elements`` / ``ELEMENTS`` / ``atom_symbols``: comma/space-separated symbols
      (e.g. ``"C, H, N, O"``).
    - ``formula`` / ``FORMULA`` / ``MOLFORMULA``: empirical formula string.

    If ``skip_atom_fg_mask`` is truthy, returns ``None`` (no masking).
    """
    if not md or md.get("skip_atom_fg_mask"):
        return None

    raw_el = md.get("elements") or md.get("ELEMENTS") or md.get("atom_symbols")
    if raw_el is not None and str(raw_el).strip():
        parts = re.split(r"[,;\s]+", str(raw_el).strip())
        out: set[str] = set()
        from rdkit.Chem import GetPeriodicTable

        pt = GetPeriodicTable()
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Normalize to periodic-table casing (Cl not CL)
            z = pt.GetAtomicNumber(p)
            if z > 0:
                out.add(pt.GetElementSymbol(z))
        return out if out else None

    for key in ("formula", "FORMULA", "MOLFORMULA", "formula_string"):
        v = md.get(key)
        if v and str(v).strip():
            return elements_from_empirical_formula(str(v).strip())
    return None


def mask_fg_probs_by_atom_content(probs: dict[str, float], md: dict[str, Any]) -> dict[str, float]:
    """
    Copy ``probs`` and zero out **halide** when inferred elements contain no halogen.

    If composition cannot be inferred, returns ``probs`` unchanged.

    Explicit flags (no formula needed):

    - ``no_halogens`` / ``halogen_free`` truthy → force halide probability to 0.
    - ``has_halogens: false`` → same.
    """
    if md.get("no_halogens") or md.get("halogen_free") or md.get("has_halogens") is False:
        out = dict(probs)
        if "halide" in out:
            out["halide"] = 0.0
        return out

    elems = infer_element_symbols(md)
    if not elems:
        return dict(probs)
    if elems & _HALOGENS:
        return dict(probs)
    out = dict(probs)
    if "halide" in out:
        out["halide"] = 0.0
    return out
