"""
Publication-quality annotation layout for FTIR Plotly figures.

Handles peak labels, ruler text, dynamic figure size, export margins, and validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Ruler labels that need forced line breaks (cm⁻¹ span is often narrow).
RULER_LABEL_LINES: dict[str, list[str]] = {
    "unsat_mid": ["C=C / amide II", "N–O / NO₂"],
    "co_fingerprint": ["C–O /", "fingerprint"],
    "triple_bond": ["C≡N /", "C≡C"],
    "oh_nh": ["O–H /", "N–H"],
}

RULER_WRAP_SPLITS = (" / ", " · ", ", ")


def _box_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def peak_priority(peak: dict[str, Any]) -> float:
    reason = str(peak.get("label_reason") or "")
    if reason == "key_evidence":
        tier = 5.0
    elif reason in ("diagnostic", "forced", "audit"):
        tier = 4.0
    elif reason == "height_prominence":
        tier = 3.0
    else:
        tier = 2.0
    q = str(peak.get("peak_quality") or "moderate")
    qmult = {"strong": 1.4, "moderate": 1.1, "weak": 0.85}.get(q, 1.0)
    h = float(peak.get("height", peak.get("rel_height", 0)) or 0)
    return tier * qmult * (0.5 + h)


def cluster_peaks_for_labeling(
    peaks: list[dict[str, Any]],
    *,
    cluster_distance_cm1: float = 18.0,
    fingerprint_lo: float = 1000.0,
    fingerprint_hi: float = 1450.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    In dense fingerprint regions, label only the strongest peak per cluster.
    Others remain in data for hover via plotted markers.
    """
    if not peaks or cluster_distance_cm1 <= 0:
        return list(peaks), {}
    sorted_p = sorted(peaks, key=lambda p: -peak_priority(p))
    kept: list[dict[str, Any]] = []
    clusters_suppressed = 0
    used_centers: list[float] = []

    def _in_fingerprint(wn: float) -> bool:
        return fingerprint_lo <= wn <= fingerprint_hi

    for p in sorted_p:
        wn = float(p.get("wn_cm1", 0))
        if not _in_fingerprint(wn):
            kept.append(p)
            continue
        if any(abs(wn - c) <= cluster_distance_cm1 for c in used_centers):
            clusters_suppressed += 1
            continue
        used_centers.append(wn)
        kept.append(p)
    stats = {"fingerprint_cluster_suppressed": clusters_suppressed} if clusters_suppressed else {}
    return kept, stats


def _estimate_label_box(
    wn: float,
    y: float,
    *,
    y_span: float,
    text: str,
    textangle: float = 0.0,
    yshift_px: float = 0.0,
    xshift_px: float = 0.0,
    wn_span_data: float = 4000.0,
) -> tuple[float, float, float, float]:
    span = max(float(y_span), 1e-9)
    y_off = (yshift_px / 280.0) * span
    x_off = (xshift_px / 800.0) * wn_span_data * 0.02
    if abs(textangle) >= 89:
        x_half = max(6.0, 8.0) * (wn_span_data / max(span * 50, 1.0)) * 0.06
        x_half += abs(xshift_px) * (wn_span_data / 800.0)
        char_h = 0.026 * span * max(len(text), 3)
        y_top = y + y_off + char_h + 0.02 * span
        y_bot = y + y_off - 0.01 * span
        return (wn - x_half + x_off, y_bot, wn + x_half + x_off, y_top)
    w_half = max(12.0, 14.0 + len(text) * 1.6)
    if abs(textangle) >= 45:
        w_half = max(10.0, 8.0 + len(text) * 0.9)
    x_half = w_half * (wn_span_data / max(span * 50, 1.0)) * 0.15
    x_half += abs(xshift_px) * (wn_span_data / 800.0)
    y_top = y + y_off + 0.045 * span
    y_bot = y + y_off - 0.012 * span
    return (wn - w_half + x_off, y_bot, wn + w_half + x_off, y_top)


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
    """Collision-aware peak label placement (smart) or minimal offsets (simple)."""
    if not annotations:
        return [], {}
    if str(mode or "smart").lower() == "simple":
        from reports.peak_label_layout import apply_collision_layout

        return apply_collision_layout(
            annotations, y_max=y_max, y_min=y_min, wn_min=wn_min, wn_max=wn_max
        )

    y_span = max(float(y_max) - float(y_min), 1e-9)
    wn_lo = float(wn_min) if wn_min is not None else -1e9
    wn_hi = float(wn_max) if wn_max is not None else 1e9
    wn_data_span = max(wn_hi - wn_lo, 400.0)
    y_ceil = float(y_max) + 0.18 * y_span

    ranked = sorted(annotations, key=lambda a: -peak_priority(a.get("_peak", {})))
    placed_boxes: list[tuple[float, float, float, float]] = []
    out: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "n_labels": len(annotations),
        "labels_shifted": 0,
        "collision_suppressed": 0,
        "outside_view": 0,
    }

    base_shifts = (10, 20, 32, 44, 58, 72, 88, 104) if not presentation else (12, 24, 38, 52, 68)
    placements: list[dict[str, Any]] = []
    for ysh in base_shifts:
        placements.append(
            {"textposition": "top center", "textangle": -90, "yshift": ysh, "showarrow": False}
        )
    for ysh in base_shifts[:4]:
        placements.append(
            {"textposition": "top center", "textangle": 0, "yshift": ysh, "showarrow": False}
        )
    for ysh in (14, 26, 38):
        placements.append(
            {
                "textposition": "top center",
                "textangle": -90,
                "yshift": ysh,
                "showarrow": True,
                "arrowwidth": 0.8,
                "arrowcolor": "#64748b",
            }
        )

    for ann in ranked:
        wn = float(ann.get("wn", 0))
        y = float(ann.get("y", 0))
        if not (math.isfinite(wn) and math.isfinite(y)):
            stats["outside_view"] += 1
            continue
        if wn < wn_lo - 5 or wn > wn_hi + 5:
            stats["outside_view"] += 1
            continue
        text = ann.get("text") or f"{wn:.0f}"
        placed = False
        for pl in placements:
            box = _estimate_label_box(
                wn,
                y,
                y_span=y_span,
                text=text,
                textangle=float(pl.get("textangle", 0)),
                yshift_px=float(pl.get("yshift", 0)),
                wn_span_data=wn_data_span,
            )
            if box[3] > y_ceil and abs(float(pl.get("textangle", 0))) < 89:
                continue
            if any(_box_overlaps(box, b) for b in placed_boxes):
                continue
            placed_boxes.append(box)
            row = dict(ann)
            row.update(pl)
            out.append(row)
            if pl.get("yshift", 8) != 8 or pl.get("showarrow"):
                stats["labels_shifted"] += 1
            placed = True
            break
        if not placed:
            stats["collision_suppressed"] += 1

    stats["labeled_peaks_count"] = len(out)
    return out, {k: v for k, v in stats.items() if v}


def wrap_ruler_label(
    region_id: str,
    short_label: str,
    *,
    wn_lo: float,
    wn_hi: float,
) -> str:
    """Multi-line ruler label using <br> for Plotly annotations."""
    if region_id in RULER_LABEL_LINES:
        return "<br>".join(RULER_LABEL_LINES[region_id])
    span = max(float(wn_hi) - float(wn_lo), 1.0)
    max_chars = max(6, int(span / 22))
    if len(short_label) <= max_chars:
        return short_label
    if " / " in short_label:
        parts = [p.strip() for p in short_label.split(" / ") if p.strip()]
        lines: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current} / {part}".strip(" /") if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = part
        if current:
            lines.append(current)
        if lines:
            return "<br>".join(lines[:3])
    if len(short_label) > max_chars + 4:
        mid = len(short_label) // 2
        for sep in RULER_WRAP_SPLITS:
            idx = short_label.find(sep, mid - 8)
            if idx > 0:
                return short_label[:idx].strip() + "<br>" + short_label[idx + len(sep) :].strip()
    return short_label


def ruler_font_size(wn_lo: float, wn_hi: float, *, front: bool = False, tier: str = "active") -> int:
    span = max(float(wn_hi) - float(wn_lo), 1.0)
    base = 10 if front else 9
    if tier == "muted":
        base -= 1
    elif tier == "strong":
        base += 0
    size = int(base + min(2, span / 120.0))
    return max(7, min(11, size))


def ruler_row_line_weight(label_text: str) -> float:
    """Vertical weight for a ruler row from wrapped label line count."""
    n_lines = label_text.count("<br>") + 1
    return 1.0 + 0.72 * max(0, n_lines - 1)


def allocate_ruler_y_bands(weights: list[float], *, margin_frac: float = 0.025) -> list[tuple[float, float]]:
    """Top-to-bottom y bands on the ruler subplot (0–1), inset for label padding."""
    if not weights:
        return []
    usable = max(0.5, 1.0 - 2.0 * margin_frac)
    total = sum(weights) or 1.0
    bands: list[tuple[float, float]] = []
    y_top = 1.0 - margin_frac
    for w in weights:
        h = usable * (w / total)
        y1 = y_top
        y0 = y_top - h
        inset = min(0.14 * h, 0.018)
        bands.append((y0 + inset, y1 - inset))
        y_top = y0
    return bands


def ruler_font_size_for_band(
    wn_lo: float,
    wn_hi: float,
    *,
    n_lines: int,
    band_height: float,
    front: bool = False,
    tier: str = "active",
) -> int:
    """Cap font size so multi-line labels fit inside the row band height."""
    base = ruler_font_size(wn_lo, wn_hi, front=front, tier=tier)
    line_budget = float(band_height) / max(int(n_lines), 1)
    if line_budget < 0.024:
        cap = 6
    elif line_budget < 0.030:
        cap = 7
    elif line_budget < 0.036:
        cap = 8
    elif line_budget < 0.044:
        cap = 9
    else:
        cap = 11
    return max(6, min(base, cap))


def ruler_subplot_height_fraction(
    *,
    n_regions: int,
    total_line_weight: float | None = None,
    front: bool = False,
) -> float:
    """Suggested fraction of figure height for the ruler subplot row."""
    n = max(int(n_regions), 1)
    weight = float(total_line_weight) if total_line_weight is not None else float(n)
    avg = weight / n
    base = 0.15 if front else 0.14
    if avg <= 1.05:
        return base
    extra = min(0.10, (avg - 1.0) * 0.022 * n)
    return min(0.26 if front else 0.24, base + extra)


def split_row_heights_with_ruler(
    ruler_frac: float,
    spec_frac: float,
    kron_frac: float,
) -> list[float]:
    """Renormalize spectrum + Kronecker rows after expanding ruler height."""
    ruler_frac = max(0.08, min(0.28, float(ruler_frac)))
    rest = 1.0 - ruler_frac
    denom = max(spec_frac + kron_frac, 1e-9)
    return [ruler_frac, rest * spec_frac / denom, rest * kron_frac / denom]


def plan_ruler_row_layouts(
    regions: tuple[Any, ...],
    *,
    front: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """
    Per-region y band, wrapped label, and font size for ruler panels.

    Returns (layouts, total_line_weight) for subplot height sizing.
    """
    staged: list[dict[str, Any]] = []
    weights: list[float] = []
    for spec in regions:
        label_text = wrap_ruler_label(
            spec.id,
            spec.short_label,
            wn_lo=float(spec.lo),
            wn_hi=float(spec.hi),
        )
        w = ruler_row_line_weight(label_text)
        weights.append(w)
        staged.append(
            {
                "spec": spec,
                "label_text": label_text,
                "n_lines": label_text.count("<br>") + 1,
            }
        )
    bands = allocate_ruler_y_bands(weights)
    layouts: list[dict[str, Any]] = []
    for item, (y0, y1) in zip(staged, bands):
        spec = item["spec"]
        band_h = y1 - y0
        layouts.append(
            {
                "spec": spec,
                "label_text": item["label_text"],
                "n_lines": item["n_lines"],
                "y0": y0,
                "y1": y1,
                "band_height": band_h,
                "font_size": ruler_font_size_for_band(
                    float(spec.lo),
                    float(spec.hi),
                    n_lines=item["n_lines"],
                    band_height=band_h,
                    front=front,
                ),
            }
        )
    return layouts, sum(weights)


def compute_figure_layout(
    *,
    n_labeled_peaks: int = 0,
    use_ruler: bool = False,
    show_deconv: bool = False,
    presentation: bool = False,
    auto_layout: bool = True,
    panel: str = "full",
    base_height: int = 1410,
) -> dict[str, Any]:
    """Dynamic height and margins for crowded annotations."""
    height = int(base_height)
    margin_t, margin_b, margin_l, margin_r = 88, 56, 64, 28
    vert_spacing = 0.03
    legend_y = 1.02

    if not auto_layout:
        return {
            "height": height,
            "margin": dict(t=margin_t, b=margin_b, l=margin_l, r=margin_r),
            "vertical_spacing": vert_spacing,
            "legend_y": legend_y,
        }

    if use_ruler and panel == "full":
        try:
            from ml.ftir_region_ruler import FTIR_RULER_REGIONS

            _, ruler_weight = plan_ruler_row_layouts(FTIR_RULER_REGIONS, front=presentation)
            rf = ruler_subplot_height_fraction(
                n_regions=len(FTIR_RULER_REGIONS),
                total_line_weight=ruler_weight,
                front=presentation,
            )
            height += int(36 + max(0.0, rf - 0.14) * 160)
        except Exception:
            height += 40
        margin_t += 12
    if show_deconv:
        height += 30
    if presentation:
        height = int(height * 1.05)
        margin_t += 16
        margin_b += 8
        legend_y = 1.04
    extra = max(0, n_labeled_peaks - 12)
    if extra > 0:
        height += min(120, extra * 6)
        margin_t += min(40, extra * 2)
    if n_labeled_peaks > 20:
        margin_t += 20

    if panel != "full":
        height = max(320, int(height * 0.55))

    return {
        "height": height,
        "margin": dict(t=margin_t, b=margin_b, l=margin_l, r=margin_r),
        "vertical_spacing": vert_spacing,
        "legend_y": legend_y,
    }


def apply_figure_layout(fig: Any, layout_params: dict[str, Any], *, n_rows: int = 3) -> None:
    m = layout_params.get("margin") or {}
    fig.update_layout(
        height=int(layout_params.get("height", 1100)),
        margin=dict(
            t=int(m.get("t", 88)),
            b=int(m.get("b", 56)),
            l=int(m.get("l", 64)),
            r=int(m.get("r", 28)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=float(layout_params.get("legend_y", 1.02)),
            x=0,
            xanchor="left",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=0,
        ),
    )
    if hasattr(fig.layout, "grid") and fig.layout.grid and n_rows > 1:
        try:
            fig.update_layout(
                grid=dict(rows=n_rows, columns=1),
                vertical_spacing=float(layout_params.get("vertical_spacing", 0.03)),
            )
        except Exception:
            pass


def finalize_export_layout(fig: Any, layout_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extra top margin so peak labels and titles are not clipped in PNG/SVG."""
    meta = dict(layout_meta or {})
    n_lab = int(meta.get("n_labeled_peaks", meta.get("labeled_peaks_count", 0)) or 0)
    extra_top = 24 + min(48, n_lab * 2)
    if meta.get("show_region_ruler"):
        extra_top += 16
    m = fig.layout.margin
    t = int(m.t) if m and m.t else 88
    b = int(m.b) if m and m.b else 56
    fig.update_layout(
        margin=dict(t=t + extra_top, b=b + 8, l=64, r=32),
        title=dict(font=dict(size=13)),
    )
    export_meta = {"export_margin_top_added": extra_top}
    return export_meta


def validate_layout(fig_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize layout pass for export audit."""
    meta = fig_meta or {}
    label_stats = meta.get("label_layout_stats") or meta.get("unlabeled_reason_counts") or {}
    return {
        "n_labels": int(label_stats.get("n_labels", meta.get("n_labeled_peaks", 0)) or 0),
        "n_labeled_peaks": int(
            label_stats.get("labeled_peaks_count", meta.get("n_labeled_peaks", 0)) or 0
        ),
        "n_overlaps_detected": int(label_stats.get("collision_suppressed", 0) or 0),
        "n_labels_shifted": int(label_stats.get("labels_shifted", 0) or 0),
        "n_labels_hidden": int(label_stats.get("collision_suppressed", 0) or 0),
        "n_outside_view": int(label_stats.get("outside_view", 0) or 0),
        "fingerprint_cluster_suppressed": int(label_stats.get("fingerprint_cluster_suppressed", 0) or 0),
        "figure_height": meta.get("figure_height"),
        "show_region_ruler": bool(meta.get("show_region_ruler")),
        "peak_label_layout": meta.get("peak_label_layout", "smart"),
        "auto_layout": meta.get("auto_layout", True),
        "presentation_mode": bool(meta.get("presentation_mode")),
        "unresolved_overlaps": int(label_stats.get("collision_suppressed", 0) or 0) > 3,
    }


def write_layout_validation(
    out_dir: Path,
    validation: dict[str, Any],
    *,
    also_markdown: bool = True,
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "layout_validation.json"
    json_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if also_markdown:
        md = out_dir / "export_layout_audit.md"
        lines = [
            "# Export layout audit\n",
            f"- Labels requested: {validation.get('n_labels', '—')}\n",
            f"- Labels placed: {validation.get('n_labeled_peaks', '—')}\n",
            f"- Labels shifted (stagger/rotate/leader): {validation.get('n_labels_shifted', '—')}\n",
            f"- Labels hidden (lowest priority): {validation.get('n_labels_hidden', '—')}\n",
            f"- Fingerprint cluster (hover only): {validation.get('fingerprint_cluster_suppressed', '—')}\n",
            f"- Figure height (px): {validation.get('figure_height', '—')}\n",
            f"- Unresolved overlap warning: {validation.get('unresolved_overlaps', False)}\n",
        ]
        md.write_text("".join(lines), encoding="utf-8")
    return json_path
