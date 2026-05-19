"""
Traditional FTIR spectrum shading windows: upper-mid coverage, tiered activity thresholds.

Shading indicates **region activity**, not confirmed functional-group assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShadeRegionSpec:
    id: str
    lo: float
    hi: float
    label: str
    evidence_keys: tuple[str, ...]
    strong_min: float = 0.10
    faint_min: float = 0.05
    activity_only: bool = False
    legend_suffix: str = "region activity"


# Order: high → low wavenumber (Plotly x reversed later; rects use cm⁻¹ directly).
SPECTRUM_SHADE_REGION_SPECS: tuple[ShadeRegionSpec, ...] = (
    ShadeRegionSpec("oh_nh", 3200, 3700, "O–H / N–H", ("oh_nh_broad",), 0.10, 0.05),
    ShadeRegionSpec(
        "nh_ch_transition",
        3100,
        3200,
        "N–H / aromatic C–H shoulder",
        ("oh_nh_broad", "aromatic_ch_stretch", "ch_stretch"),
        0.10,
        0.05,
    ),
    ShadeRegionSpec(
        "ch_stretch",
        2800,
        3100,
        "C–H stretch",
        ("ch_stretch", "aromatic_ch_stretch", "aliphatic_ch"),
        0.10,
        0.04,
    ),
    ShadeRegionSpec(
        "aliphatic_ch",
        2850,
        2965,
        "aliphatic C–H stretch",
        ("aliphatic_ch", "ch_stretch"),
        0.10,
        0.04,
    ),
    ShadeRegionSpec(
        "aromatic_ch",
        3000,
        3100,
        "aromatic/sp² C–H stretch",
        ("aromatic_ch_stretch", "ch_stretch"),
        0.10,
        0.05,
    ),
    ShadeRegionSpec(
        "upper_mid_activity",
        2260,
        2800,
        "upper mid-IR activity",
        ("upper_mid_activity",),
        0.10,
        0.05,
        activity_only=True,
        legend_suffix="spectral activity (low specificity)",
    ),
    ShadeRegionSpec("triple_bond", 2100, 2260, "C≡C / C≡N", ("alkyne_cc", "nitrile"), 0.10, 0.05),
    ShadeRegionSpec(
        "carbonyl_overtone_gap",
        1820,
        2100,
        "weak combination / overtone",
        ("carbonyl_overtone", "carbonyl"),
        0.12,
        0.06,
        activity_only=True,
        legend_suffix="spectral activity (weak overtone)",
    ),
    ShadeRegionSpec(
        "carbonyl",
        1650,
        1820,
        "C=O",
        ("carbonyl", "amide_i", "ester_co"),
        0.10,
        0.05,
    ),
    ShadeRegionSpec(
        "unsat_mid",
        1450,
        1649,
        "C=C / amide II / NO₂",
        ("aromatic_cc", "nitro_asym", "amide_i"),
        0.10,
        0.05,
    ),
    ShadeRegionSpec(
        "co_fingerprint",
        1000,
        1449,
        "C–O / fingerprint",
        ("c_o_stretch", "si_o", "fingerprint"),
        0.10,
        0.05,
    ),
    ShadeRegionSpec(
        "fingerprint_low",
        650,
        999,
        "Fingerprint (low ν)",
        ("fingerprint", "aromatic_oop"),
        0.10,
        0.05,
    ),
)

STRONG_FILL = "rgba(59,130,246,0.17)"
FAINT_FILL = "rgba(100,116,139,0.09)"
ACTIVITY_STRONG_FILL = "rgba(148,163,184,0.14)"
ACTIVITY_FAINT_FILL = "rgba(148,163,184,0.07)"


def spectrum_shade_regions_legacy() -> list[tuple[str, int, int, str]]:
    """Backward-compatible tuples for product_v1 region annotations."""
    return [(s.id, int(s.lo), int(s.hi), s.label) for s in SPECTRUM_SHADE_REGION_SPECS]


def spectrum_shade_evidence_keys() -> dict[str, tuple[str, ...]]:
    return {s.id: s.evidence_keys for s in SPECTRUM_SHADE_REGION_SPECS}


def resolve_shade_thresholds(
    spec: ShadeRegionSpec,
    *,
    shade_strong_min: float | None = None,
    shade_faint_min: float | None = None,
    shade_sensitive: bool = False,
) -> tuple[float, float]:
    """Return (strong_min, faint_min) for one region."""
    strong = float(shade_strong_min if shade_strong_min is not None else spec.strong_min)
    faint = float(shade_faint_min if shade_faint_min is not None else spec.faint_min)
    if shade_sensitive:
        faint = min(faint, 0.03)
    if spec.id in ("ch_stretch", "aliphatic_ch", "aromatic_ch"):
        faint = min(faint, 0.04)
    return strong, faint


def upper_mid_coverage_gaps(lo: float = 1800.0, hi: float = 3200.0, *, max_gap_cm1: float = 80.0) -> list[tuple[float, float]]:
    """Return uncovered intervals within [lo, hi] given current shade windows."""
    windows = [(max(s.lo, lo), min(s.hi, hi)) for s in SPECTRUM_SHADE_REGION_SPECS if s.hi > lo and s.lo < hi]
    windows.sort()
    merged: list[tuple[float, float]] = []
    for a, b in windows:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    gaps: list[tuple[float, float]] = []
    cur = lo
    for a, b in merged:
        if a - cur > max_gap_cm1:
            gaps.append((cur, a))
        cur = max(cur, b)
    if hi - cur > max_gap_cm1:
        gaps.append((cur, hi))
    return gaps


def render_tiered_band_shading(
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    y_min: float,
    y_max: float,
    *,
    shade_strong_min: float | None = None,
    shade_faint_min: float | None = None,
    shade_sensitive: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    from reports.report_render import region_activity

    shapes: list[dict[str, Any]] = []
    names: list[str] = []
    for spec in SPECTRUM_SHADE_REGION_SPECS:
        strong_min, faint_min = resolve_shade_thresholds(
            spec,
            shade_strong_min=shade_strong_min,
            shade_faint_min=shade_faint_min,
            shade_sensitive=shade_sensitive,
        )
        rel = region_activity(
            evidence, wn, y, spec.lo, spec.hi, evidence_keys=spec.evidence_keys
        )
        if rel < faint_min:
            continue
        tier = "strong" if rel >= strong_min else "faint"
        if spec.activity_only:
            fill = ACTIVITY_STRONG_FILL if tier == "strong" else ACTIVITY_FAINT_FILL
        else:
            fill = STRONG_FILL if tier == "strong" else FAINT_FILL
        suffix = spec.legend_suffix if spec.activity_only else "region activity"
        names.append(
            f"{spec.label} ({int(spec.lo)}–{int(spec.hi)} cm⁻¹, {tier} {suffix})"
        )
        shapes.append(
            {
                "type": "rect",
                "x0": float(spec.lo),
                "x1": float(spec.hi),
                "y0": y_min,
                "y1": y_max,
                "fillcolor": fill,
                "line": {"width": 0},
                "layer": "below",
                "_shade_tier": tier,
                "_shade_region_id": spec.id,
            }
        )
    return shapes, names
