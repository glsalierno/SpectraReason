"""
Modular Kronecker report rendering helpers and stable HTML feature markers.

Layout code in ``kronecker_pi_layout`` / ``v4_evidence_report``; spectrum figures here.
See ``reports/REPORT_FEATURE_CONTRACT.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Stable HTML markers for regression tests (do not remove without updating tests).
MARKER_SUMMARY_TABLE = "<!-- report-feature:summary-table -->"
MARKER_PLOTLY_SPECTRUM = "<!-- report-feature:plotly-spectrum -->"
MARKER_BAND_SHADING = "<!-- report-feature:band-shading -->"
MARKER_REGION_RULER = "<!-- report-feature:region-ruler -->"
MARKER_PEAK_LABELS = "<!-- report-feature:peak-labels -->"
MARKER_KRONECKER_STEMS = "<!-- report-feature:kronecker-stems -->"
MARKER_LOCAL_HOVER = "<!-- report-feature:local-hover -->"
MARKER_EVIDENCE_TABLE = "<!-- report-feature:evidence-table -->"
MARKER_BAND_EVIDENCE_MAP = "<!-- report-feature:band-evidence-map -->"
MARKER_ML_FAMILY = "<!-- report-feature:ml-family-column -->"
MARKER_ML_SPECIFIC = "<!-- report-feature:ml-specific-column -->"
MARKER_CONSENSUS = "<!-- report-feature:consensus-column -->"
MARKER_AMBIGUITY = "<!-- report-feature:ambiguity-labels -->"
MARKER_ARTIFACT_FLAGS = "<!-- report-feature:artifact-flags -->"

FORBIDDEN_REPORT_STRINGS = (
    "Executive Summary",
    "Needs Review",
    "PI Summary",
)

_SHADE_COLORS: tuple[str, ...] = (
    "rgba(59,130,246,0.14)",
    "rgba(16,185,129,0.12)",
    "rgba(245,158,11,0.12)",
    "rgba(236,72,153,0.1)",
    "rgba(139,92,246,0.1)",
)


def region_activity(
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
    *,
    evidence_keys: tuple[str, ...] = (),
) -> float:
    """Relative activity in [lo, hi]: evidence regions and/or direct segment max."""
    regions = evidence.get("regions") or {}
    rel = 0.0
    for key in evidence_keys:
        block = regions.get(key)
        if isinstance(block, dict):
            rel = max(rel, float(block.get("rel_max", 0) or 0))
    from ml.ftir_interpretable_features import _segment_stats

    st = _segment_stats(wn, y, lo, hi)
    y_max = float((evidence.get("summary") or {}).get("y_max", 0) or 0)
    if y_max <= 0:
        y_max = float(np.nanmax(y)) if y.size else 1.0
    seg_rel = float(st["max"] / (y_max + 1e-9))
    return max(rel, seg_rel)


def render_band_shading_shapes(
    *,
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    y_min: float,
    y_max: float,
    min_rel_max: float = 0.10,
    shade_faint_min: float | None = None,
    shade_sensitive: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Traditional FTIR windows with tiered shading (strong/faint) by regional activity.
    Activity shading is independent of functional-group assignment score.
    """
    from ml.ftir_shade_regions import render_tiered_band_shading

    faint = 0.05 if shade_faint_min is None else float(shade_faint_min)
    return render_tiered_band_shading(
        evidence,
        wn,
        y,
        y_min,
        y_max,
        shade_strong_min=float(min_rel_max),
        shade_faint_min=faint,
        shade_sensitive=shade_sensitive,
    )


def render_spectrum_figure(
    *,
    name: str,
    wn: np.ndarray,
    y: np.ndarray,
    peak_wn: list[float],
    peak_h: list[float],
    pipeline: dict[str, Any],
    peaks_dicts: list[dict[str, Any]],
    show_band_shading: bool = False,
    include_ml: bool = True,
    hover_top_fg: int = 5,
    hover_tolerance_cm1: float = 12.0,
    library: list[dict[str, Any]] | None = None,
    report_density: str = "balanced",
    ontology: str | None = None,
) -> Any:
    """Stacked spectrum + Kronecker stems; local hover; optional traditional region shading."""
    from reports.structural_fg_svm_kronecker_report import _build_stacked_interactive_figure

    fig, _meta = _build_stacked_interactive_figure(
        name=name,
        wn=wn,
        y=y,
        peak_wn=peak_wn,
        peak_h=peak_h,
        pipeline=pipeline,
        peaks_dicts=peaks_dicts,
        show_band_shading=show_band_shading,
        include_ml=include_ml,
        hover_top_fg=hover_top_fg,
        hover_tolerance_cm1=hover_tolerance_cm1,
        library=library,
        report_density=report_density,
        ontology=ontology,
    )
    return fig


def render_summary_table(rows: list[dict[str, Any]], *, ml_enabled: bool) -> str:
    from reports.kronecker_pi_layout import build_summary_table_html

    return MARKER_SUMMARY_TABLE + build_summary_table_html(rows=rows, ml_enabled=ml_enabled)


def render_evidence_table(pipeline: dict[str, Any], *, anchor: str = "") -> str:
    ont = str(pipeline.get("ontology") or "").lower()
    if ont == "v4":
        from reports.v4_evidence_report import build_evidence_first_assignments_table_html

        body = build_evidence_first_assignments_table_html(pipeline, top_n=24, anchor=anchor)
    else:
        from reports.kronecker_pi_layout import build_explainable_assignments_table_html

        body = build_explainable_assignments_table_html(pipeline, anchor=anchor)
    if MARKER_EVIDENCE_TABLE not in body:
        body = MARKER_EVIDENCE_TABLE + body
    return body


def render_band_evidence_map(
    pipeline: dict[str, Any],
    *,
    anchor: str = "",
    audience: str = "front",
) -> str:
    from reports.v4_evidence_report import build_band_evidence_map_html

    body = build_band_evidence_map_html(pipeline, anchor=anchor, audience=audience)
    if MARKER_BAND_EVIDENCE_MAP not in body:
        body = MARKER_BAND_EVIDENCE_MAP + body
    return body


def render_consensus_table(pipeline: dict[str, Any], *, top_n: int = 12) -> str:
    from reports.kronecker_pi_layout import build_most_likely_fg_table_html

    body = build_most_likely_fg_table_html(pipeline, top_n=top_n, include_consensus=True)
    if MARKER_CONSENSUS not in body:
        body = MARKER_CONSENSUS + body
    return body


def render_ml_columns_note(*, has_family: bool, has_specific: bool) -> str:
    parts: list[str] = []
    if has_family:
        parts.append(MARKER_ML_FAMILY)
    if has_specific:
        parts.append(MARKER_ML_SPECIFIC)
    return "".join(parts)


def wrap_plotly_spectrum_html(
    plot_html: str,
    *,
    band_shading: bool = False,
    region_ruler: bool = False,
    peak_labels: bool = False,
) -> str:
    parts = [MARKER_PLOTLY_SPECTRUM, MARKER_LOCAL_HOVER, MARKER_KRONECKER_STEMS]
    if region_ruler:
        parts.append(MARKER_REGION_RULER)
        parts.append('<span data-report-feature="region-ruler" hidden></span>')
    if peak_labels:
        parts.append(MARKER_PEAK_LABELS)
    if band_shading:
        parts.append(MARKER_BAND_SHADING)
        parts.append('<span data-report-feature="band-shading" hidden></span>')
    return "".join(parts) + plot_html