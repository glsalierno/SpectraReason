"""
Structured FTIR band knowledge for evidence-first functional-group assignment.

Primary source: Python ``BAND_LIBRARY`` (conservative cm⁻¹ ranges).
Optional YAML overlay via ``load_band_library()`` for legacy interpretability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

Importance = Literal["required", "supporting", "weak"]
Specificity = Literal["high", "medium", "low"]


def _band(
    *,
    band_id: str,
    label: str,
    region_min_cm1: float,
    region_max_cm1: float,
    mode: str,
    importance: Importance = "supporting",
    specificity: Specificity = "medium",
    subclass: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": band_id,
        "label": label,
        "subclass": subclass or label,
        "region_min_cm1": float(region_min_cm1),
        "region_max_cm1": float(region_max_cm1),
        "mode": mode,
        "importance": importance,
        "specificity": specificity,
        "notes": notes,
    }


# Conservative FTIR reference bands (ranges, not single frequencies).
BAND_LIBRARY: list[dict[str, Any]] = [
    _band(band_id="broad_oh", label="broad_OH", region_min_cm1=3200, region_max_cm1=3600,
          mode="O-H / N-H stretch envelope", importance="supporting", specificity="low",
          notes="Broad; alcohol, phenol, acid, amine overlap."),
    _band(band_id="alcohol_oh", label="alcohol", region_min_cm1=3200, region_max_cm1=3650,
          mode="O-H stretch (aliphatic)", importance="required", specificity="low"),
    _band(band_id="phenol_oh", label="phenol", region_min_cm1=3200, region_max_cm1=3600,
          mode="O-H stretch (phenolic)", importance="required", specificity="medium"),
    _band(band_id="acid_oh_broad", label="carboxylic_acid_OH", region_min_cm1=2500, region_max_cm1=3300,
          mode="broad acid O-H", importance="supporting", specificity="medium",
          notes="Often very broad, overlaps fingerprint."),
    _band(band_id="amine_nh", label="primary_amine", region_min_cm1=3300, region_max_cm1=3500,
          mode="N-H stretch", importance="supporting", specificity="medium", subclass="primary_amine"),
    _band(band_id="amine_nh2", label="secondary_amine", region_min_cm1=3300, region_max_cm1=3500,
          mode="N-H stretch", importance="weak", specificity="low", subclass="secondary_amine"),
    _band(band_id="amide_nh", label="amide", region_min_cm1=3100, region_max_cm1=3500,
          mode="N-H stretch", importance="supporting", specificity="medium"),
    _band(band_id="pyrrole_nh", label="pyrrole_like_NH", region_min_cm1=3200, region_max_cm1=3450,
          mode="N-H (heteroaromatic)", importance="supporting", specificity="medium"),
    _band(band_id="cyclic_amine_nh", label="cyclic_amine", region_min_cm1=3300, region_max_cm1=3500,
          mode="N-H (cyclic)", importance="weak", specificity="low"),
    _band(band_id="aromatic_cc", label="aromatic", region_min_cm1=1450, region_max_cm1=1600,
          mode="C=C ring stretches", importance="required", specificity="medium"),
    _band(band_id="heteroaromatic", label="heteroaromatic", region_min_cm1=1400, region_max_cm1=1650,
          mode="heteroaromatic ring modes", importance="supporting", specificity="low"),
    _band(band_id="ketone_co", label="ketone", region_min_cm1=1700, region_max_cm1=1728,
          mode="C=O stretch", importance="required", specificity="medium"),
    _band(band_id="aldehyde_co", label="aldehyde", region_min_cm1=1720, region_max_cm1=1745,
          mode="C=O stretch", importance="required", specificity="medium"),
    _band(band_id="ester_co", label="ester", region_min_cm1=1730, region_max_cm1=1758,
          mode="C=O stretch", importance="required", specificity="medium"),
    _band(band_id="ester_co_o", label="ester", region_min_cm1=1150, region_max_cm1=1250,
          mode="C-O stretch", importance="supporting", specificity="medium"),
    _band(band_id="carboxylic_co", label="carboxylic_acid", region_min_cm1=1690, region_max_cm1=1725,
          mode="C=O stretch", importance="required", specificity="medium"),
    _band(band_id="amide_co", label="amide_carbonyl", region_min_cm1=1630, region_max_cm1=1690,
          mode="amide I", importance="required", specificity="medium"),
    _band(band_id="amide_ii", label="amide", region_min_cm1=1510, region_max_cm1=1575,
          mode="amide II (N–H bend / C–N)", importance="supporting", specificity="medium",
          notes="Supports amide when N–H stretch is absent (e.g. tertiary amide)."),
    _band(band_id="carbonate_co", label="carbonate", region_min_cm1=1740, region_max_cm1=1770,
          mode="C=O stretch", importance="supporting", specificity="medium"),
    _band(band_id="urethane_co", label="urethane", region_min_cm1=1690, region_max_cm1=1740,
          mode="C=O (urethane/urea)", importance="supporting", specificity="low"),
    _band(band_id="ether_co", label="ether", region_min_cm1=1050, region_max_cm1=1150,
          mode="C-O stretch", importance="required", specificity="low"),
    _band(band_id="aryl_ether_co", label="aryl_ether", region_min_cm1=1180, region_max_cm1=1280,
          mode="Ar-O-C stretch", importance="supporting", specificity="medium"),
    _band(band_id="phenolic_co", label="phenol", region_min_cm1=1180, region_max_cm1=1260,
          mode="phenolic C-O", importance="supporting", specificity="medium", subclass="phenolic_C-O"),
    _band(band_id="nitrile_cn", label="nitrile", region_min_cm1=2200, region_max_cm1=2260,
          mode="C≡N stretch", importance="required", specificity="high"),
    _band(band_id="nitro_asym", label="nitro", region_min_cm1=1500, region_max_cm1=1570,
          mode="NO₂ asymmetric", importance="required", specificity="medium"),
    _band(band_id="nitro_sym", label="nitro", region_min_cm1=1320, region_max_cm1=1390,
          mode="NO₂ symmetric", importance="supporting", specificity="medium", subclass="nitro_sym"),
    _band(
        band_id="enamine_c_c_cn",
        label="enamine",
        region_min_cm1=1480,
        region_max_cm1=1650,
        mode="enamine / conjugated C=C–N stretch",
        importance="supporting",
        specificity="low",
        notes="Overlaps aromatic C=C, amide II, NO₂, heteroaromatic modes.",
    ),
    _band(
        band_id="heterocyclic_n_oxide",
        label="heterocyclic_N_O",
        region_min_cm1=1250,
        region_max_cm1=1650,
        mode="heterocyclic N–O / N-oxide-related",
        importance="supporting",
        specificity="low",
        notes="Can overlap nitro symmetric/asymmetric and heteroaromatic ring modes.",
    ),
    _band(
        band_id="n_oxide_high",
        label="heterocyclic_N_O",
        region_min_cm1=1450,
        region_max_cm1=1600,
        mode="N-oxide-like (high wavenumber)",
        importance="weak",
        specificity="low",
        notes="Pyrrole/pyridine N-oxide-like; confounds nitro without paired NO₂ bands.",
    ),
    _band(
        band_id="n_oxide_low",
        label="heterocyclic_N_O",
        region_min_cm1=1250,
        region_max_cm1=1350,
        mode="N-oxide-like (low wavenumber)",
        importance="weak",
        specificity="low",
    ),
    _band(
        band_id="pyrrole_n_oxide_like",
        label="pyrrole_N_oxide",
        region_min_cm1=1250,
        region_max_cm1=1600,
        mode="pyrrole/pyridine N-oxide-like",
        importance="weak",
        specificity="low",
        notes="Possible nitro confounder; require paired NO₂ before nitro assignment.",
    ),
    _band(band_id="alkene_cc", label="alkene", region_min_cm1=1620, region_max_cm1=1680,
          mode="C=C stretch", importance="supporting", specificity="medium"),
    _band(band_id="alkyne_cc", label="alkyne", region_min_cm1=2100, region_max_cm1=2260,
          mode="C≡C stretch", importance="supporting", specificity="medium"),
    _band(band_id="alkyne_ch", label="alkyne", region_min_cm1=3280, region_max_cm1=3340,
          mode="≡C-H stretch", importance="weak", specificity="medium", subclass="terminal_alkyne"),
    _band(
        band_id="aliphatic_ch_asym",
        label="aliphatic_CH",
        region_min_cm1=2920,
        region_max_cm1=2965,
        mode="aliphatic C–H asymmetric stretch",
        importance="supporting",
        specificity="low",
        notes="Supports organic matrix / alkyl backbone; not specific alone.",
    ),
    _band(
        band_id="aliphatic_ch_sym",
        label="aliphatic_CH",
        region_min_cm1=2850,
        region_max_cm1=2885,
        mode="aliphatic C–H symmetric stretch",
        importance="supporting",
        specificity="low",
    ),
    _band(
        band_id="aromatic_ch_stretch",
        label="aromatic_CH",
        region_min_cm1=3000,
        region_max_cm1=3100,
        mode="aromatic/sp² C–H stretch",
        importance="supporting",
        specificity="medium",
        notes="Pair with aromatic C=C ring modes near 1450–1600 cm⁻¹.",
    ),
    _band(
        band_id="aldehydic_ch_fermi",
        label="aldehyde",
        region_min_cm1=2720,
        region_max_cm1=2820,
        mode="aldehydic C–H / Fermi doublet",
        importance="supporting",
        specificity="low",
        notes="Meaningful mainly with C=O near 1650–1820; do not confuse with aromatic C–H at ~3000.",
    ),
    _band(
        band_id="siloxane_sio",
        label="siloxane",
        region_min_cm1=1000,
        region_max_cm1=1150,
        mode="Si-O-Si asymmetric",
        importance="supporting",
        specificity="low",
        notes="1000-1150 cm-1 overlaps C-O, C-O-C, aryl ether, alcohol, ester, and carbohydrate fingerprint; "
        "not sufficient alone for high-confidence siloxane; pair with Si-C or second Si-O evidence.",
    ),
    _band(band_id="silicone_sic", label="silicone_or_silane", region_min_cm1=1250, region_max_cm1=1350,
          mode="Si-C / Si-CH₃", importance="supporting", specificity="medium"),
]

# Fix duplicate band_id for ester - second entry used ester_co_o in call above

_LIBRARY_BY_ID: dict[str, dict[str, Any]] = {b["id"]: b for b in BAND_LIBRARY}
_LIBRARY_BY_LABEL: dict[str, list[dict[str, Any]]] = {}
for _b in BAND_LIBRARY:
    _LIBRARY_BY_LABEL.setdefault(str(_b["label"]).lower(), []).append(_b)

_CACHE_YAML: list[dict[str, Any]] | None = None
_LIBRARY_PATH = Path(__file__).resolve().parent / "ftir_band_library.yaml"


def load_band_library(path: Path | None = None, *, prefer_python: bool = True) -> list[dict[str, Any]]:
    """Return structured band entries (Python library by default)."""
    if prefer_python:
        return list(BAND_LIBRARY)
    global _CACHE_YAML
    p = path or _LIBRARY_PATH
    if path is None and _CACHE_YAML is not None:
        return _CACHE_YAML
    if not p.is_file():
        return list(BAND_LIBRARY)
    import yaml

    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    bands = list(data.get("bands") or [])
    if path is None:
        _CACHE_YAML = bands
    return bands


def band_by_id(band_id: str) -> dict[str, Any] | None:
    return _LIBRARY_BY_ID.get(band_id)


def bands_for_label(label: str) -> list[dict[str, Any]]:
    lab = str(label).lower().strip()
    out = list(_LIBRARY_BY_LABEL.get(lab, []))
    for b in BAND_LIBRARY:
        subs = [str(s).lower() for s in (b.get("subclasses") or []) if isinstance(b.get("subclasses"), list)]
        if lab in subs and b not in out:
            out.append(b)
    return out


def format_band_row(b: dict[str, Any]) -> str:
    lo = b.get("region_min_cm1")
    hi = b.get("region_max_cm1")
    mode = b.get("mode", "")
    return f"{lo}–{hi} cm⁻¹ ({mode})"


def all_fg_labels() -> list[str]:
    """Unique functional-group labels referenced in the band library."""
    seen: set[str] = set()
    labels: list[str] = []
    for b in BAND_LIBRARY:
        lab = str(b["label"]).lower()
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
    return labels
