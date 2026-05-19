"""
Collision-aware layout for numeric peak labels on FTIR spectrum plots.
"""

from __future__ import annotations

import math
from typing import Any


def _priority(p: dict[str, Any]) -> float:
    h = float(p.get("height", p.get("rel_height", 0)) or 0)
    q = str(p.get("peak_quality") or "moderate")
    mult = {"strong": 1.4, "moderate": 1.1, "weak": 0.85}.get(q, 1.0)
    if p.get("label_reason") == "key_evidence":
        mult *= 1.25
    return h * mult


def _box_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _estimate_box(
    wn: float,
    y: float,
    *,
    y_span: float,
    text: str,
    textangle: float = 0.0,
    yshift_px: float = 0.0,
) -> tuple[float, float, float, float]:
    span = max(float(y_span), 1e-9)
    w_half = 18.0 + len(text) * 1.8
    if abs(textangle) >= 45:
        w_half = 12.0
    y_off = (yshift_px / 280.0) * span
    y_top = y + y_off + 0.04 * span
    y_bot = y + y_off - 0.01 * span
    return (wn - w_half, y_bot, wn + w_half, y_top)


def apply_collision_layout(
    annotations: list[dict[str, Any]],
    *,
    y_max: float,
    y_min: float = 0.0,
    wn_min: float | None = None,
    wn_max: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Assign textposition / textangle / yshift / leader lines; drop lowest-priority on collision.
    """
    if not annotations:
        return [], {}
    y_span = max(float(y_max) - float(y_min), 1e-9)
    ranked = sorted(annotations, key=lambda a: -_priority(a.get("_peak", {})))
    placed_boxes: list[tuple[float, float, float, float]] = []
    out: list[dict[str, Any]] = []
    suppressed = 0

    placements = (
        {"textposition": "top center", "textangle": 0, "yshift": 8, "showarrow": False},
        {"textposition": "top center", "textangle": 0, "yshift": 22, "showarrow": False},
        {"textposition": "top center", "textangle": 0, "yshift": 36, "showarrow": False},
        {"textposition": "middle right", "textangle": -45, "yshift": 6, "showarrow": False},
        {"textposition": "top center", "textangle": -90, "yshift": 10, "showarrow": False},
        {
            "textposition": "top center",
            "textangle": 0,
            "yshift": 14,
            "showarrow": True,
            "arrowwidth": 0.8,
            "arrowcolor": "#64748b",
        },
    )

    for ann in ranked:
        wn = float(ann.get("wn", 0))
        y = float(ann.get("y", 0))
        if not (math.isfinite(wn) and math.isfinite(y)):
            suppressed += 1
            continue
        if wn_min is not None and wn < wn_min - 1:
            suppressed += 1
            continue
        if wn_max is not None and wn > wn_max + 1:
            suppressed += 1
            continue
        text = ann.get("text") or f"{wn:.0f}"
        placed = False
        for pl in placements:
            box = _estimate_box(
                wn,
                y,
                y_span=y_span,
                text=text,
                textangle=float(pl.get("textangle", 0)),
                yshift_px=float(pl.get("yshift", 0)),
            )
            if any(_box_overlaps(box, b) for b in placed_boxes):
                continue
            placed_boxes.append(box)
            row = dict(ann)
            row.update(pl)
            out.append(row)
            placed = True
            break
        if not placed:
            suppressed += 1

    stats: dict[str, int] = {"n_labels": len(annotations)}
    if suppressed:
        stats["collision_suppressed"] = suppressed
    stats["labeled_peaks_count"] = len(out)
    return out, stats


def apply_peak_label_layout(
    annotations: list[dict[str, Any]],
    *,
    mode: str = "smart",
    y_max: float,
    y_min: float = 0.0,
    wn_min: float | None = None,
    wn_max: float | None = None,
    presentation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Delegate to annotation_layout (smart) or local simple collision layout."""
    if str(mode or "smart").lower() == "simple":
        return apply_collision_layout(
            annotations, y_max=y_max, y_min=y_min, wn_min=wn_min, wn_max=wn_max
        )
    from reports.annotation_layout import apply_peak_label_layout as _smart

    return _smart(
        annotations,
        mode="smart",
        y_max=y_max,
        y_min=y_min,
        wn_min=wn_min,
        wn_max=wn_max,
        presentation=presentation,
    )
