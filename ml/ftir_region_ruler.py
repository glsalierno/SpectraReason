"""
Compact FTIR interpretation ruler (horizontal range bars above the spectrum).

Shows traditional tentative assignment windows — not functional-group confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RulerRegionSpec:
    id: str
    lo: float
    hi: float
    short_label: str
    evidence_keys: tuple[str, ...]
    hover_note: str = ""


FTIR_RULER_REGIONS: tuple[RulerRegionSpec, ...] = (
    RulerRegionSpec("oh_nh", 3200, 3700, "O–H/N–H", ("oh_nh_broad",), "Broad O–H / N–H"),
    RulerRegionSpec(
        "aromatic_ch",
        3000,
        3100,
        "sp² C–H",
        ("aromatic_ch_stretch",),
        "Aromatic/sp² C–H (~3000–3100); not 2800 cm⁻¹",
    ),
    RulerRegionSpec(
        "aliphatic_ch",
        2850,
        2965,
        "sp³ C–H",
        ("aliphatic_ch",),
        "Aliphatic C–H asymmetric/symmetric",
    ),
    RulerRegionSpec(
        "aldehydic_ch",
        2720,
        2820,
        "aldehyde C–H",
        ("aldehydic_ch",),
        "Aldehydic C–H / Fermi doublet; pair with C=O for aldehyde",
    ),
    RulerRegionSpec("triple_bond", 2100, 2260, "C≡N/C≡C", ("alkyne_cc", "nitrile")),
    RulerRegionSpec("carbonyl", 1650, 1820, "C=O", ("carbonyl", "amide_i", "ester_co")),
    RulerRegionSpec(
        "unsat_mid",
        1450,
        1650,
        "C=C / amide II / N–O",
        (
            "aromatic_cc",
            "nitro_asym",
            "amide_i",
            "amide_ii",
            "enamine_c_c_cn",
            "heterocyclic_n_oxide",
            "n_oxide_high",
        ),
        "Aromatic C=C, amide II, enamine C=C–N, heterocyclic N–O / N-oxide-like modes; "
        "NO₂ asym overlap — paired 1320–1390 cm⁻¹ required for nitro",
    ),
    RulerRegionSpec("co_fingerprint", 1000, 1450, "C–O / fingerprint", ("c_o_stretch", "fingerprint")),
    RulerRegionSpec("fingerprint_low", 650, 1000, "fingerprint", ("fingerprint", "aromatic_oop")),
)


def ruler_region_activity(
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    spec: RulerRegionSpec,
) -> float:
    from reports.report_render import region_activity

    return region_activity(
        evidence, wn, y, spec.lo, spec.hi, evidence_keys=spec.evidence_keys
    )


def ruler_activity_tier(rel: float, *, active_min: float = 0.05, strong_min: float = 0.10) -> str:
    if rel >= strong_min:
        return "strong"
    if rel >= active_min:
        return "active"
    return "muted"


def _tier_style(tier: str, *, front: bool = False) -> tuple[str, str, float]:
    if tier == "strong":
        return ("rgba(226,232,240,0.95)", "#1e293b", 2.4 if front else 2.2)
    if tier == "active":
        return ("rgba(241,245,249,0.92)", "#475569", 1.8 if front else 1.5)
    return ("rgba(248,250,252,0.88)", "#cbd5e1", 1.2 if front else 1.0)


def add_ftir_region_ruler(
    fig: Any,
    *,
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    row: int = 1,
    col: int = 1,
    active_min: float = 0.05,
    strong_min: float = 0.10,
    front_mode: bool = False,
    suppress_nitro_reporting: bool = False,
) -> list[dict[str, Any]]:
    """Add stacked horizontal ruler bars to an existing Plotly subplot row."""
    import plotly.graph_objects as go

    if not FTIR_RULER_REGIONS:
        return []

    from ml.report_suppression import ruler_hover_note
    from reports.annotation_layout import plan_ruler_row_layouts, ruler_font_size_for_band

    row_layouts, _total_weight = plan_ruler_row_layouts(
        FTIR_RULER_REGIONS,
        front=front_mode,
        suppress_nitro_reporting=suppress_nitro_reporting,
    )
    activities: list[dict[str, Any]] = []
    for layout in row_layouts:
        spec = layout["spec"]
        y0 = float(layout["y0"])
        y1 = float(layout["y1"])
        rel = ruler_region_activity(evidence, wn, y, spec)
        tier = ruler_activity_tier(rel, active_min=active_min, strong_min=strong_min)
        fill, line_col, lw = _tier_style(tier, front=front_mode)
        note = ruler_hover_note(spec, suppress_nitro=suppress_nitro_reporting)
        hover = (
            f"<b>{spec.short_label}</b><br>"
            f"{int(spec.lo)}–{int(spec.hi)} cm⁻¹<br>"
            f"Region activity: {rel:.2f} ({tier})<br>"
            f"{note}<br>"
            "<i>Tentative range — not FG assignment</i>"
        )
        fig.add_trace(
            go.Scatter(
                x=[spec.lo, spec.hi, spec.hi, spec.lo, spec.lo],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=fill,
                line=dict(color=line_col, width=lw),
                mode="lines",
                name=spec.short_label,
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        cx = (spec.lo + spec.hi) / 2.0
        cy = (y0 + y1) / 2.0
        label_text = layout["label_text"]
        fsize = ruler_font_size_for_band(
            float(spec.lo),
            float(spec.hi),
            n_lines=int(layout["n_lines"]),
            band_height=float(layout["band_height"]),
            front=front_mode,
            tier=tier,
        )
        fig.add_annotation(
            x=cx,
            y=cy,
            xref="x",
            yref="y",
            text=label_text,
            showarrow=False,
            align="center",
            valign="middle",
            font=dict(
                size=fsize,
                color="#334155" if tier == "strong" else ("#475569" if tier != "muted" else "#94a3b8"),
                family="Segoe UI, system-ui, sans-serif",
            ),
            row=row,
            col=col,
        )
        activities.append(
            {
                "id": spec.id,
                "lo": spec.lo,
                "hi": spec.hi,
                "rel_activity": round(rel, 4),
                "tier": tier,
            }
        )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[0, 1],
        fixedrange=True,
        row=row,
        col=col,
    )
    fig.update_xaxes(showticklabels=False, row=row, col=col)
    return activities
