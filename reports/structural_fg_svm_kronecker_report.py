#!/usr/bin/env python3
"""
Interactive HTML report: **structural FG SVM** + **Kronecker** peak featurization.

Each spectrum is a **stacked Plotly figure** (processed trace + discrete peak impulses).
Spectrum hover is **local and band-aware** (library windows, peaks, rule support, cautions).
Global scores appear in tables below the figure, not repeated on every point.
Optional ``--show-band-shading`` tints non-overlapping traditional FTIR windows (O–H, C=O, C≡C/C≡N, etc.) when regional activity is present.

Run from **FTIR_SVM** root::

    cd FTIR_SVM
    $env:PYTHONPATH = (Get-Location).Path  # FTIR_SVM
    python reports/structural_fg_svm_kronecker_report.py batch \\
      --inputs Dopamine_Powder.CSV Nylon_T.CSV \\
      --model ml/runs/struct_fg_v7_pubchem_mordred.joblib \\
      --out-dir reports/svm_kronecker_interactive

For a **shareable** HTML+CSV export (no machine directories in the artifact), add ``--anonymize-metadata``.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.spectrum import load_processed_spectrum
from ml.ftir_evidence import build_local_hover_context, format_peak_marker_hover
from ml.ftir_peak_picking import (
    QUALITY_VISUAL,
    peaks_for_display,
    peaks_for_kronecker,
    peaks_for_label,
)
from ml.ftir_band_library import load_band_library
from ml.ftir_export import export_pipeline_batch_csv
from ml.ftir_pipeline import load_model_bundle, run_evidence_first_pipeline
from ml.ftir_rule_config import load_rules_config
from reports.kronecker_pi_layout import (
    KRONECKER_PI_EXTRA_CSS,
    build_explainable_assignments_table_html,
    build_fg_justification_intro_html,
    build_most_likely_fg_table_html,
    build_spectrum_body_html,
    build_summary_table_html,
    spectrum_card_title_row_html,
    spectrum_summary_quick_details_html,
    spectrum_summary_row,
    status_badge_html,
    _truncate,
)
from ml.ftir_robustness import evaluate_robustness_one_spectrum
from ml.structural_fg_svm import predict_proba_row


def _path_for_publish(p: Path, *, anonymize: bool) -> str:
    """Full resolved path for local runs; basename only when building shareable artifacts."""
    if anonymize:
        return p.name
    return str(p.resolve())


def _metadata_for_path(path: Path) -> dict[str, Any]:
    """Same heuristics as ``structural_fg_svm_report`` (CAS, formula, halide mask hints)."""
    prov_sidecar = path.with_suffix(path.suffix + ".provenance.json")
    if not prov_sidecar.is_file():
        prov_sidecar = path.with_name(path.name + ".provenance.json")
    stem = path.stem
    title = stem.replace("_", " ")
    md: dict[str, Any] = {"title": title, "name": stem, "xunits": "1/CM"}
    low = stem.lower()
    if "dopamine" in low and "poly" not in low:
        md["cas"] = "51-61-6"
        md["formula"] = "C8H11NO2"
    elif "polydopamine" in low or ("dopamine" in low and "poly" in low):
        md["no_halogens"] = True
    # Nylon: no halogens in typical PA-6 repeat unit
    elif "nylon" in low:
        md["no_halogens"] = True
    if any(k in low for k in ("powder", "_atr", "atr_", "polydopamine")):
        md.setdefault("sample_type", "powder")
    if "atr" in low or "scraped" in low:
        md.setdefault("measurement_mode", "ATR")
    if prov_sidecar.is_file():
        try:
            import json as _json

            extra = _json.loads(prov_sidecar.read_text(encoding="utf-8"))
            if isinstance(extra, dict):
                md.update(extra)
        except Exception:
            pass
    return md

_SHADE_COLORS: tuple[str, ...] = (
    "rgba(59,130,246,0.14)",
    "rgba(16,185,129,0.12)",
    "rgba(245,158,11,0.12)",
    "rgba(236,72,153,0.1)",
    "rgba(139,92,246,0.1)",
)

def _merged_ml_by_label(pipeline: dict[str, Any]) -> dict[str, float]:
    """Max ML probability or score per FG from refinement blocks (local hover uses subset)."""
    out: dict[str, float] = {}
    for key in ("basic", "subtle", "legacy"):
        block = (pipeline.get("ml_refinement") or {}).get(key) or {}
        if not isinstance(block, dict):
            continue
        for lab, ent in (block.get("per_label") or {}).items():
            v = ent.get("ml_probability")
            if v is None:
                v = ent.get("ml_score")
            if v is None:
                continue
            fv = float(v)
            out[str(lab)] = max(out.get(str(lab), 0.0), fv)
    return out


def _peaks_as_dicts(peak_wn: list[float], peak_h: list[float]) -> list[dict[str, Any]]:
    return [{"wn_cm1": float(a), "height": float(b)} for a, b in zip(peak_wn, peak_h)]


def _region_rel_max(
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
    *,
    evidence_keys: tuple[str, ...] = (),
) -> float:
    """Relative peak activity in a window (0–1); delegates to report_render.region_activity."""
    from reports.report_render import region_activity

    return region_activity(evidence, wn, y, lo, hi, evidence_keys=evidence_keys)


def _traditional_region_shading_shapes(
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
    """Traditional FTIR window shading; see ``reports.report_render.render_band_shading_shapes``."""
    from reports.report_render import render_band_shading_shapes

    return render_band_shading_shapes(
        evidence=evidence,
        wn=wn,
        y=y,
        y_min=y_min,
        y_max=y_max,
        min_rel_max=min_rel_max,
        shade_faint_min=shade_faint_min,
        shade_sensitive=shade_sensitive,
    )


def _resolve_under_chunks(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _build_stacked_interactive_figure(
    *,
    name: str,
    wn: np.ndarray,
    y: np.ndarray,
    peak_wn: list[float],
    peak_h: list[float],
    pipeline: dict[str, Any],
    peaks_dicts: list[dict[str, Any]],
    show_band_shading: bool = False,
    show_region_ruler: bool | None = None,
    include_ml: bool = True,
    hover_top_fg: int = 5,
    hover_tolerance_cm1: float = 12.0,
    library: list[dict[str, Any]] | None = None,
    report_density: str = "balanced",
    report_style: str = "legacy",
    ontology: str | None = None,
    label_band_shading: bool = False,
    show_weak_peaks: bool = False,
    max_peak_labels: int = 24,
    label_all_diagnostic_peaks: bool = False,
    peaks_plotted: list[dict[str, Any]] | None = None,
    peaks_labeled: list[dict[str, Any]] | None = None,
    shade_min_activity: float = 0.10,
    shade_faint_min: float = 0.05,
    shade_sensitive: bool = False,
    report_audience: str = "debug",
    visual_theme: str = "default",
    panel_mode: str = "full",
    label_all_above_height: float | None = None,
    presentation_mode: bool = False,
    show_deconvolution: bool = False,
    deconv_pack: dict[str, Any] | None = None,
    peak_label_layout: str = "smart",
    auto_layout: bool | None = None,
    fingerprint_cluster_distance: float | None = 18.0,
) -> tuple[Any, dict[str, Any]]:
    """Stacked spectrum + Kronecker bars; hover is local/band-aware (no global prob block)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from reports.matlab_visual_theme import get_plotly_theme, normalize_visual_theme

    dens = str(report_density or "balanced").lower()
    style = str(report_style or "legacy").lower()
    audience = str(report_audience or "debug").lower()
    is_front = audience == "front"
    if auto_layout is None:
        auto_layout = is_front
    theme_key = normalize_visual_theme(visual_theme)
    theme_cfg = dict(get_plotly_theme(theme_key))
    if presentation_mode:
        theme_cfg["peak_label_font"] = int(theme_cfg.get("peak_label_font", 9)) + 2
        theme_cfg["spectrum_width"] = float(theme_cfg.get("spectrum_width", 1.1)) + 0.35
        theme_cfg["title_font_size"] = int(theme_cfg.get("title_font_size") or 13) + 1
    panel = str(panel_mode or "full").lower()
    use_ruler = show_region_ruler if show_region_ruler is not None else style == "product_v1"
    if panel == "kronecker":
        use_ruler = False
    elif panel == "spectrum":
        pass
    fig_meta: dict[str, Any] = {
        "show_region_ruler": use_ruler,
        "show_band_shading": show_band_shading,
        "n_labeled_peaks": 0,
    }
    # Base heights (px); product_v1 uses larger spectrum panel.
    if style == "product_v1":
        _h_scale = 2.35
        _base_h = {"summary": 400, "balanced": 600, "audit": 660}
    else:
        _h_scale = 1.8
        _base_h = {"summary": 320, "balanced": 470, "audit": 540}
    fig_height = int(_base_h.get(dens, 470) * _h_scale)
    ruler_frac = 0.14
    if use_ruler:
        from ml.ftir_region_ruler import FTIR_RULER_REGIONS
        from ml.report_suppression import nitro_reporting_suppressed
        from reports.annotation_layout import plan_ruler_row_layouts, ruler_subplot_height_fraction

        _, ruler_weight = plan_ruler_row_layouts(
            FTIR_RULER_REGIONS,
            front=is_front,
            suppress_nitro_reporting=nitro_reporting_suppressed(pipeline),
        )
        ruler_frac = ruler_subplot_height_fraction(
            n_regions=len(FTIR_RULER_REGIONS),
            total_line_weight=ruler_weight,
            front=is_front,
        )
        fig_height += int(55 + max(0.0, ruler_frac - 0.14) * 220)
    if is_front and not auto_layout and not presentation_mode and fig_height > 1100:
        fig_height = 1100
    wn = np.asarray(wn, dtype=float)
    y = np.asarray(y, dtype=float)
    y_min = float(np.nanmin(y)) if y.size else 0.0
    y_max = float(np.nanmax(y)) if y.size else 1.0
    y_span = max(y_max - y_min, 1e-9)

    lib = library if library is not None else load_band_library(prefer_python=True)
    ont_hover = str(ontology or pipeline.get("ontology") or "v3").lower()
    ml_map: dict[str, float] = {}
    if include_ml and pipeline.get("ml_mode") != "none" and ont_hover != "v4":
        ml_map = _merged_ml_by_label(pipeline)

    max_labels = min(5, max(1, int(hover_top_fg)))
    custom_spec: list[str] = []
    for xv, yv in zip(wn, y):
        ctx = build_local_hover_context(
            float(xv),
            float(yv),
            peaks_dicts,
            band_library=lib,
            rule_assignments=pipeline.get("rule_assignments"),
            ml_assignments=ml_map if ml_map else None,
            evidence=pipeline.get("evidence"),
            tolerance_cm1=hover_tolerance_cm1,
            max_labels=max_labels,
            ontology=ont_hover,
        )
        custom_spec.append(ctx.get("plotly_tail") or ctx.get("hover_text", ""))

    if panel == "kronecker":
        ruler_row, spectrum_row, kron_row = 0, 0, 1
        kron_title = "Picked peaks" if is_front else "Kronecker impulses (picked peaks)"
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(kron_title,),
        )
    elif panel == "spectrum":
        if use_ruler:
            from reports.annotation_layout import split_row_heights_with_ruler

            ruler_row, spectrum_row, kron_row = 1, 2, 0
            ruler_title = "FTIR region guide" if is_front else "FTIR region ruler (tentative ranges)"
            spec_title = name if is_front else f"{name} — processed absorbance"
            rh_spec = split_row_heights_with_ruler(ruler_frac, 0.86, 0.0)
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[rh_spec[0], rh_spec[1]],
                subplot_titles=(ruler_title, spec_title),
            )
        else:
            ruler_row, spectrum_row, kron_row = 0, 1, 0
            spec_title = name if is_front else f"{name} — processed absorbance"
            fig = make_subplots(rows=1, cols=1, subplot_titles=(spec_title,))
    elif use_ruler:
        from reports.annotation_layout import split_row_heights_with_ruler

        ruler_row, spectrum_row, kron_row = 1, 2, 3
        if is_front:
            row_heights = split_row_heights_with_ruler(ruler_frac, 0.74, 0.16)
            kron_title = "Picked peaks"
            ruler_title = "FTIR region guide"
            spec_title = name
        else:
            row_heights = split_row_heights_with_ruler(ruler_frac, 0.57, 0.32)
            kron_title = "Kronecker impulses (picked peaks)"
            ruler_title = "FTIR region ruler (tentative ranges)"
            spec_title = f"{name} — processed absorbance"
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(ruler_title, spec_title, kron_title),
        )
    else:
        ruler_row, spectrum_row, kron_row = 0, 1, 2
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.68, 0.32],
            subplot_titles=(
                f"{name} — processed absorbance",
                "Kronecker impulses (picked peaks)",
            ),
        )

    if use_ruler and ruler_row > 0 and pipeline.get("evidence"):
        from ml.ftir_region_ruler import add_ftir_region_ruler
        from ml.report_suppression import nitro_reporting_suppressed

        fig_meta["ruler_activities"] = add_ftir_region_ruler(
            fig,
            evidence=pipeline["evidence"],
            wn=wn,
            y=y,
            row=ruler_row,
            col=1,
            front_mode=is_front,
            suppress_nitro_reporting=nitro_reporting_suppressed(pipeline),
        )

    shade_names: list[str] = []
    shade_palette = theme_cfg.get("shade_colors") or _SHADE_COLORS
    if show_band_shading and pipeline.get("evidence") and spectrum_row > 0:
        sh, shade_names = _traditional_region_shading_shapes(
            evidence=pipeline["evidence"],
            wn=wn,
            y=y,
            y_min=y_min - 0.02 * y_span,
            y_max=y_max + 0.02 * y_span,
            min_rel_max=float(shade_min_activity),
            shade_faint_min=float(shade_faint_min),
            shade_sensitive=shade_sensitive,
        )
        for s in sh:
            plotly_shape = {k: v for k, v in s.items() if not str(k).startswith("_shade")}
            fig.add_shape(plotly_shape, row=spectrum_row, col=1)
        if label_band_shading and shade_names:
            from reports.product_v1_report import add_band_shading_labels

            add_band_shading_labels(fig, shade_names, sh, row=spectrum_row, col=1)

    if spectrum_row > 0:
        fig.add_trace(
            go.Scatter(
                x=wn,
                y=y,
                mode="lines",
                name="Spectrum",
                line=dict(
                    color=theme_cfg["spectrum_line"],
                    width=theme_cfg["spectrum_width"],
                ),
                customdata=np.array(custom_spec, dtype=object),
                hovertemplate=(
                    "<b>ν=%{x:.1f} cm⁻¹</b><br>"
                    "A=%{y:.3f}<br><br>"
                    "%{customdata}<extra></extra>"
                ),
            ),
            row=spectrum_row,
            col=1,
        )

        for i, nm in enumerate(shade_names[:12]):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=shade_palette[i % len(shade_palette)],
                        symbol="square",
                    ),
                    name=nm,
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=spectrum_row,
                col=1,
            )

    if peak_wn and (spectrum_row > 0 or kron_row > 0):
        peaks_all = peaks_dicts if peaks_dicts else _peaks_as_dicts(peak_wn, peak_h)
        peaks_d = (
            list(peaks_plotted)
            if peaks_plotted is not None
            else peaks_for_display(
                peaks_all,
                show_weak_peaks=show_weak_peaks or dens == "audit",
                report_density=dens,
                include_noise=dens == "audit",
            )
        )
        peaks_lab = (
            list(peaks_labeled)
            if peaks_labeled is not None
            else peaks_for_label(
                peaks_all,
                show_weak_peaks=show_weak_peaks or dens == "audit",
                max_peak_labels=max_peak_labels,
                label_all_diagnostic=label_all_diagnostic_peaks,
                report_density=dens,
            )
        )
        fig_meta["n_labeled_peaks"] = len(peaks_lab)
        evidence = pipeline.get("evidence") or {}
        if spectrum_row > 0 and ont_hover == "v4" and evidence and peaks_lab:
            from reports.v4_evidence_report import peak_annotation_specs

            ann_cap = max(1, int(max_peak_labels))
            if peaks_lab and all(p.get("peak_id") for p in peaks_lab):
                ann_cap = len(peaks_lab)
            elif label_all_above_height is not None and peaks_lab:
                ann_cap = max(ann_cap, len(peaks_lab))
            ann, layout_stats = peak_annotation_specs(
                peaks_lab,
                evidence,
                max_peaks=ann_cap,
                report_density=dens,
                include_weak=True,
                y_max=y_max,
                y_min=y_min,
                wn_min=float(np.nanmin(wn)) if wn.size else None,
                wn_max=float(np.nanmax(wn)) if wn.size else None,
                peak_label_layout=peak_label_layout,
                fingerprint_cluster_distance=fingerprint_cluster_distance,
                presentation=presentation_mode,
            )
            if layout_stats:
                fig_meta["label_layout_stats"] = layout_stats
                ur = fig_meta.setdefault("unlabeled_reason_counts", {})
                for k, v in layout_stats.items():
                    ur[k] = ur.get(k, 0) + int(v)
            if ann:
                plain = [a for a in ann if not a.get("showarrow")]
                arrows = [a for a in ann if a.get("showarrow")]
                scatterable = [
                    a
                    for a in plain
                    if float(a.get("textangle", 0) or 0) == 0.0
                    and str(a.get("textposition", "top center")) == "top center"
                    and int(a.get("yshift", 8) or 8) <= 12
                ]
                layout_ann = [a for a in plain if a not in scatterable] + arrows
                if scatterable:
                    acol = [a["color"] for a in scatterable]
                    if theme_key in ("matlab", "dark"):
                        acol = [theme_cfg["spectrum_line"]] * len(scatterable)
                    label_sym = "triangle-up" if theme_key == "matlab" else "diamond"
                    fig.add_trace(
                        go.Scatter(
                            x=np.array([a["wn"] for a in scatterable], dtype=float),
                            y=np.array([a["y"] for a in scatterable], dtype=float),
                            mode="markers+text",
                            text=[a.get("text", f"{a['wn']:.0f}") for a in scatterable],
                            textposition="top center",
                            textfont=dict(
                                size=theme_cfg["peak_label_font"],
                                color=theme_cfg["peak_label_color"],
                            ),
                            name="Labeled peaks",
                            marker=dict(
                                color=acol,
                                size=8 if theme_key == "matlab" else 9,
                                symbol=label_sym,
                                line=dict(
                                    width=0.5,
                                    color=theme_cfg["peak_marker_line"],
                                ),
                            ),
                            customdata=np.array([a["hover"] for a in scatterable], dtype=object),
                            hovertemplate="%{customdata}<extra></extra>",
                        ),
                        row=spectrum_row,
                        col=1,
                    )
                for a in layout_ann:
                    yshift = int(a.get("yshift", 14) or 14)
                    fig.add_annotation(
                        x=float(a["wn"]),
                        y=float(a["y"]),
                        text=a.get("text", f"{a['wn']:.0f}"),
                        showarrow=bool(a.get("showarrow")),
                        arrowhead=2 if a.get("showarrow") else 0,
                        arrowwidth=float(a.get("arrowwidth", 0.8)),
                        arrowcolor=str(a.get("arrowcolor", "#64748b")),
                        ax=0,
                        ay=-yshift if not a.get("showarrow") else yshift,
                        textangle=float(a.get("textangle", 0) or 0),
                        font=dict(
                            size=theme_cfg["peak_label_font"],
                            color=theme_cfg["peak_label_color"],
                        ),
                        row=spectrum_row,
                        col=1,
                    )
                fig_meta["has_peak_labels"] = True
        if (
            show_deconvolution
            and deconv_pack
            and spectrum_row > 0
            and panel in ("full", "spectrum")
        ):
            from reports.deconv_report import add_deconv_overlay_traces, deconv_legend_buttons

            deconv_names = add_deconv_overlay_traces(
                fig,
                deconv_pack,
                spectrum_row=spectrum_row,
                col=1,
                audit=dens == "audit",
                theme_line=theme_cfg.get("spectrum_line", "#94a3b8"),
            )
            fig_meta["deconv_overlay"] = True
            if deconv_names:
                fig_meta["deconv_trace_names"] = deconv_names
                menus = list(fig.layout.updatemenus or ()) if fig.layout.updatemenus else []
                menus.extend(
                    [
                        {
                            "type": "dropdown",
                            "direction": "down",
                            "x": 0.02,
                            "y": 1.12,
                            "xanchor": "left",
                            "buttons": deconv_legend_buttons(deconv_names),
                        }
                    ]
                )
                fig.update_layout(updatemenus=menus)
        if spectrum_row > 0:
            for qname in ("strong", "moderate", "weak"):
                subset = [p for p in peaks_d if str(p.get("peak_quality", "moderate")) == qname]
                if not subset:
                    continue
                vis = QUALITY_VISUAL.get(qname, QUALITY_VISUAL["moderate"])
                tc = theme_cfg.get(f"peak_{qname}")
                marker_color = tc if tc else vis["color"]
                marker_sym = "circle" if theme_key == "matlab" else vis["symbol"]
                sx = np.array([float(p["wn_cm1"]) for p in subset], dtype=float)
                sy = np.array([float(p["height"]) for p in subset], dtype=float)
                peak_custom: list[str] = []
                for xv, yv in zip(sx, sy):
                    ctx = build_local_hover_context(
                        float(xv),
                        float(yv),
                        peaks_d,
                        band_library=lib,
                        rule_assignments=pipeline.get("rule_assignments"),
                        ml_assignments=ml_map if ml_map else None,
                        evidence=evidence,
                        tolerance_cm1=hover_tolerance_cm1,
                        max_labels=max_labels,
                        ontology=ont_hover,
                    )
                    peak_custom.append(format_peak_marker_hover(ctx))
                fig.add_trace(
                    go.Scatter(
                        x=sx,
                        y=sy,
                        mode="markers",
                        name=f"Peaks ({qname})",
                        marker=dict(
                            color=marker_color,
                            size=vis["size"] if theme_key == "default" else max(5, vis["size"] - 1),
                            symbol=marker_sym,
                            line=dict(
                                width=0.3,
                                color=theme_cfg["peak_marker_line"],
                            ),
                        ),
                        customdata=np.array(peak_custom, dtype=object),
                        hovertemplate="%{customdata}<extra></extra>",
                        opacity=vis["opacity"],
                    ),
                    row=spectrum_row,
                    col=1,
                )
        kron_peaks = peaks_d
        kx = np.asarray([float(p["wn_cm1"]) for p in kron_peaks], dtype=float)
        ky = np.asarray([float(p["height"]) for p in kron_peaks], dtype=float)
        k_custom: list[str] = []
        for xv, yv in zip(kx, ky):
            ctx = build_local_hover_context(
                float(xv),
                float(yv),
                peaks_d,
                band_library=lib,
                rule_assignments=pipeline.get("rule_assignments"),
                ml_assignments=ml_map if ml_map else None,
                evidence=evidence,
                tolerance_cm1=hover_tolerance_cm1,
                max_labels=max_labels,
                ontology=ont_hover,
            )
            k_custom.append(format_peak_marker_hover(ctx))
        if kron_row > 0:
            if theme_cfg.get("kron_fill"):
                kron_marker = dict(
                    color=theme_cfg["kron_fill"],
                    line=dict(
                        width=theme_cfg["kron_line_width"] or 0.25,
                        color=theme_cfg["kron_line"],
                    ),
                )
            elif is_front:
                kron_marker = dict(color="#93c5fd", line=dict(width=0.25, color="#bfdbfe"))
            else:
                kron_marker = dict(color="#2563eb", line=dict(width=0.4, color="#1e3a8a"))
            fig.add_trace(
                go.Bar(
                    x=kx,
                    y=ky,
                    name="Picked peaks" if is_front else "Kronecker",
                    marker=kron_marker,
                    customdata=np.array(k_custom, dtype=object),
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                row=kron_row,
                col=1,
            )
    elif kron_row > 0:
        xref = "x domain" if kron_row == 1 else f"x{kron_row} domain"
        yref = "y domain" if kron_row == 1 else f"y{kron_row} domain"
        fig.add_annotation(
            text="No peaks picked",
            xref=xref,
            yref=yref,
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, color="#64748b"),
        )

    spike_col = theme_cfg.get("spike_color") or "#94a3b8"
    if kron_row > 0:
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", autorange="reversed", row=kron_row, col=1)
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor=spike_col,
            spikethickness=1,
            row=kron_row,
            col=1,
        )
        fig.update_yaxes(title_text="Peak height", row=kron_row, col=1)
    if spectrum_row > 0:
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor=spike_col,
            spikethickness=1,
            row=spectrum_row,
            col=1,
        )
        if kron_row <= 0:
            fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", autorange="reversed", row=spectrum_row, col=1)
        fig.update_yaxes(title_text="Normalized absorbance", row=spectrum_row, col=1)
    if style == "product_v1" and pipeline.get("evidence") and not use_ruler:
        from reports.product_v1_report import add_spectrum_annotations, region_annotation_specs

        ann_specs = region_annotation_specs(pipeline, wn, y, max_annotations=6)
        if ann_specs:
            add_spectrum_annotations(fig, ann_specs, y_max=y_max, row=spectrum_row, col=1)

    n_layout_rows = fig.layout.grid.rows if hasattr(fig.layout, "grid") and fig.layout.grid else (
        3 if use_ruler and panel == "full" else (2 if panel == "spectrum" and use_ruler else 1)
    )
    try:
        n_layout_rows = int(fig._grid_ref.shape[0])  # type: ignore[attr-defined]
    except Exception:
        n_layout_rows = 3 if use_ruler and panel == "full" else 2

    from reports.annotation_layout import apply_figure_layout, compute_figure_layout

    n_placed = int(
        (fig_meta.get("label_layout_stats") or {}).get(
            "labeled_peaks_count", fig_meta.get("n_labeled_peaks", 0)
        )
        or 0
    )
    layout_params = compute_figure_layout(
        n_labeled_peaks=n_placed,
        use_ruler=use_ruler and ruler_row > 0,
        show_deconv=bool(show_deconvolution and deconv_pack),
        presentation=presentation_mode,
        auto_layout=bool(auto_layout),
        panel=panel,
        base_height=int(fig_height),
    )
    fig_height = int(layout_params["height"])
    if panel != "full":
        fig_height = max(280, int(fig_height * 0.55))

    fig.update_layout(
        template="plotly_white",
        height=fig_height,
        hovermode="x",
        hoverlabel=dict(
            bgcolor=theme_cfg["hover_bg"],
            font_size=12 if not presentation_mode else 13,
            bordercolor=theme_cfg["hover_border"],
        ),
    )
    apply_figure_layout(fig, layout_params, n_rows=n_layout_rows)
    leg_y = float(layout_params.get("legend_y", theme_cfg.get("legend_y", 1.02)))
    if use_ruler and ruler_row > 0 and panel == "full":
        leg_y = min(leg_y, 0.98)
    fig.update_layout(legend=dict(y=leg_y))
    from reports.matlab_visual_theme import apply_plotly_theme_to_figure

    apply_plotly_theme_to_figure(
        fig,
        theme_key,
        n_rows=n_layout_rows,
        spectrum_row=spectrum_row or 1,
        kron_row=kron_row or n_layout_rows,
    )
    fig_meta["visual_theme"] = theme_key
    fig_meta["panel_mode"] = panel
    fig_meta["figure_height"] = fig_height
    fig_meta["peak_label_layout"] = peak_label_layout
    fig_meta["auto_layout"] = bool(auto_layout)
    fig_meta["presentation_mode"] = presentation_mode
    fig_meta["fingerprint_cluster_distance"] = fingerprint_cluster_distance
    return fig, fig_meta


def write_interactive_report_html(
    *,
    out_path: Path,
    page_title: str,
    subtitle: str,
    model_path: Path | None,
    model_paths_line: str = "",
    sections: list[dict[str, Any]],
    summary_table_html: str = "",
    report_density_label: str = "",
    redacted_paths_notice: bool = False,
    include_summary_table: bool = True,
    report_style: str = "legacy",
    report_audience: str = "debug",
    visual_theme: str = "default",
    export_static_figures: bool = False,
    editable_text: bool = False,
    extra_css: str = "",
    extra_body_html: str = "",
    extra_script: str = "",
) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    css = """
body { font-family: system-ui, Segoe UI, Arial, sans-serif; margin: 0; color: #1f2937; }
.layout { display: flex; min-height: 100vh; }
#sidebar { width: 272px; background: #f8fafc; border-right: 1px solid #e2e8f0; padding: 16px; position: sticky; top: 0; align-self: flex-start; max-height: 100vh; overflow-y: auto; }
#sidebar h2 { font-size: 0.95rem; margin: 0 0 8px; }
#sidebar ul { list-style: none; padding: 0; margin: 0; font-size: 0.88rem; }
#sidebar li { margin: 6px 0; }
#sidebar a { color: #1d4ed8; text-decoration: none; }
#sidebar a:hover { text-decoration: underline; }
main { flex: 1; padding: 20px 28px 48px; max-width: 1180px; }
h1 { font-size: 1.35rem; margin: 0 0 6px; }
.muted { color: #6b7280; margin: 0 0 14px; font-size: 0.92rem; }
.card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 18px 22px; margin: 18px 0 28px; background: #fff; }
.hint { font-size: 0.85rem; color: #64748b; margin: 0 0 12px; }
.tbl { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 12px; }
.tbl th, .tbl td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }
.tbl th { background: #f3f4f6; }
.plot-wrap { margin-top: 8px; }
.expl-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; margin: 10px 0; }
.expl-card h4 { margin: 0 0 6px; font-size: 0.95rem; }
.expl-card .prob { color: #1d4ed8; font-weight: 600; }
.badge-rob { display: inline-block; padding: 3px 10px; border-radius: 6px; color: #fff; font-size: 12px; margin-left: 8px; }
.caution { color: #78716c; font-size: 12px; margin: 4px 0 0 16px; }
details { margin: 12px 0; }
summary { cursor: pointer; font-weight: 600; color: #334155; }
.spec-card .spec-card-title { font-size: 1.05rem; }
#all-spectra { scroll-margin-top: 12px; }
"""
    css += KRONECKER_PI_EXTRA_CSS
    style_key = str(report_style or "legacy").lower()
    audience_key = str(report_audience or "debug").lower()
    body_classes = []
    if style_key == "product_v1":
        body_classes.append("product-v1")
    if audience_key == "front":
        body_classes.append("front-audience")
    if style_key == "product_v1":
        from reports.product_v1_report import PRODUCT_V1_CSS

        css += PRODUCT_V1_CSS
    if audience_key == "front":
        from reports.front_facing_report import FRONT_EXTRA_CSS

        css += FRONT_EXTRA_CSS
    from reports.matlab_visual_theme import (
        MARKER_MATLAB_THEME,
        body_class_for_theme,
        extra_css_for_theme,
        normalize_visual_theme,
    )

    theme_key = normalize_visual_theme(visual_theme)
    theme_body = body_class_for_theme(theme_key)
    if theme_body:
        body_classes.append(theme_body)
        css += extra_css_for_theme(theme_key)
    body_class = " ".join(body_classes)

    nav_items = ["<li><a href='#all-spectra'>Spectra</a></li>"]
    if include_summary_table and summary_table_html:
        nav_label = "Consensus interpretation" if "consensus-interpretation-table" in summary_table_html else "Summary Table"
        nav_items.insert(0, f"<li><a href='#summary-table'>{nav_label}</a></li>")
    nav_report = "<p class='nav-section-title'>Report</p><ul>" + "".join(nav_items) + "</ul>"
    nav_specs: list[str] = []
    for sec in sections:
        badge = sec.get("sidebar_badge_html") or ""
        nav_specs.append(
            "<li>"
            f"<a href='#{html.escape(sec['anchor'])}'>{html.escape(sec['name'])}</a> {badge}"
            "</li>"
        )

    body_parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        f"<title>{html.escape(page_title)}</title>",
        f"<style>{css}</style>",
        extra_css,
        f"</head><body class='{body_class}'>",
        "<div class='layout'>",
        "<nav id='sidebar'>",
        nav_report,
        "<h2>Spectra</h2><ul>",
        *nav_specs,
        "</ul></nav>",
        "<main>",
        f"<h1>{html.escape(page_title)}</h1>",
        f"<p class='muted'>{html.escape(subtitle)}</p>",
    ]
    if audience_key != "front":
        body_parts.append(
            f"<p class='muted'>Models: <code>{html.escape(model_paths_line or str(model_path or ''))}</code></p>"
        )
    elif model_paths_line and audience_key != "front":
        body_parts.append(
            "<p class='muted settings-appendix'><details><summary>Run settings</summary>"
            f"<code>{html.escape(model_paths_line)}</code></details></p>"
        )
    if redacted_paths_notice:
        body_parts.append(
            "<p class='muted'>Host directory paths are redacted for sharing; spectrum filenames, "
            "model file names, and analysis settings (fusion, guardrails, density) are shown as used.</p>"
        )
    if report_density_label and audience_key != "front":
        body_parts.append(
            f"<p class='muted'>Report density: <b>{html.escape(report_density_label)}</b> "
            "(summary = compact; balanced = default; audit = full tables and more sections open).</p>"
        )
    if style_key != "product_v1":
        body_parts.append(
            "<p class='hint'>Spectrum hover is local to wavenumber (bands + evidence). "
            "Global scores appear in tables below each figure.</p>"
        )
    if theme_key == "matlab":
        body_parts.append(MARKER_MATLAB_THEME)
    from reports.report_render import MARKER_SUMMARY_TABLE

    if summary_table_html:
        if MARKER_SUMMARY_TABLE not in summary_table_html:
            body_parts.append(MARKER_SUMMARY_TABLE)
        body_parts.append(summary_table_html)
    body_parts.append("<div id='all-spectra'>")

    plotly_js_included = False
    for sec in sections:
        kw: dict[str, Any] = {
            "include_plotlyjs": not plotly_js_included,
            "full_html": False,
            "config": {"displayModeBar": True, "responsive": True},
        }
        from reports.report_render import wrap_plotly_spectrum_html

        plot_html = wrap_plotly_spectrum_html(
            sec["figure"].to_html(**kw),
            band_shading=bool(sec.get("band_shading")),
            region_ruler=bool(sec.get("region_ruler")),
            peak_labels=bool(sec.get("peak_labels")),
        )
        plotly_js_included = True
        figure_note = sec.get("figure_note_html", "")
        tables = sec.get("tables_html", "")
        anc = html.escape(sec["anchor"])
        title_html = sec.get("title_html", "")
        body_parts.extend(
            [
                f"<section id='{anc}' class='card spec-card'>",
                title_html,
                figure_note,
                f"<div id='{anc}-figure' class='plot-wrap'>",
                plot_html,
                "</div>",
                sec.get("robustness_badge", ""),
                tables,
                sec.get("table_html", ""),
                sec.get("interpret_html", ""),
                "</section>",
            ]
        )
    if editable_text:
        from reports.front_facing_report import EDITABLE_TEXT_SCRIPT

        body_parts.append(EDITABLE_TEXT_SCRIPT)
    if extra_body_html:
        body_parts.append(extra_body_html)
    if extra_script:
        body_parts.append(extra_script)
    body_parts.append("</div></main></div></body></html>")

    out_path.write_text("".join(body_parts), encoding="utf-8")
    return out_path


def _robustness_badge_html(score: float) -> str:
    color = "#16a34a" if score >= 0.85 else ("#ca8a04" if score >= 0.65 else "#dc2626")
    return (
        f"<span class='badge-rob' style='background:{color}'>"
        f"Robustness {score:.2f}</span>"
    )


def _resolve_ml_mode(
    ml_mode: str,
    *,
    has_legacy: bool,
    has_basic: bool,
    has_subtle: bool,
    include_evidence: bool,
) -> str:
    if ml_mode != "auto":
        return ml_mode
    if include_evidence and not (has_legacy or has_basic or has_subtle):
        return "none"
    if has_basic and has_subtle:
        return "both"
    if has_basic:
        return "basic"
    if has_subtle:
        return "subtle"
    if has_legacy:
        return "legacy"
    return "none"


def run_batch(
    *,
    input_paths: list[Path],
    model_path: Path | None,
    basic_model_path: Path | None,
    subtle_model_path: Path | None,
    out_path: Path,
    page_title: str,
    subtitle: str,
    max_peaks: int,
    hover_top_fg: int,
    ml_mode: str = "auto",
    fusion_mode: str = "annotate",
    include_evidence: bool = True,
    include_ml: bool = True,
    include_consensus: bool = True,
    include_robustness: bool = False,
    rules_config: dict[str, Any] | None = None,
    export_csv_dir: Path | None = None,
    show_band_shading: bool = False,
    show_region_ruler: bool | None = None,
    label_all_diagnostic_peaks: bool = False,
    label_all_above_height: float | None = None,
    peak_label_preset: str | None = None,
    presentation_mode: bool = False,
    peak_label_layout: str = "smart",
    auto_layout: bool | None = None,
    fingerprint_cluster_distance: float | None = 18.0,
    static_peak_label_policy: str = "key",
    max_static_peak_labels: int = 12,
    guardrails_mode: str = "v3",
    ml_guardrails: str = "strict",
    show_ambiguity_labels: bool = True,
    show_artifact_flags: bool = True,
    report_density: str = "balanced",
    top_n_summary: int = 5,
    rules_preset_label: str = "",
    rules_config_path_label: str = "",
    anonymize_metadata: bool = False,
    include_summary_table: bool = True,
    measurement_mode: str = "",
    atr_crystal: str = "",
    atr_aware: bool | None = None,
    report_style: str = "product_v1",
    label_band_shading: bool = False,
    peak_sensitivity: str = "balanced",
    show_weak_peaks: bool = False,
    max_peak_labels: int = 24,
    peak_min_height: float | None = None,
    peak_min_prominence: float | None = None,
    peak_label_min_height: float | None = None,
    peak_label_min_prominence: float | None = None,
    shade_min_activity: float = 0.10,
    shade_faint_min: float = 0.05,
    shade_sensitive: bool = False,
    report_audience: str | None = None,
    front_max_peak_labels: int = 30,
    show_metadata: bool = False,
    front_facing: bool = False,
    visual_theme: str = "default",
    export_static_figures: bool = False,
    static_format: str = "png",
    static_dpi: int = 300,
    static_out: Path | None = None,
    export_paper_figures: bool = False,
    paper_figure_modes: tuple[str, ...] = ("transmittance", "normalized_absorbance"),
    paper_label_style: str = "horizontal-leader",
    paper_report_style: str = "both",
    paper_formats: tuple[str, ...] = ("png", "svg", "pdf"),
    max_paper_peak_labels: int = 10,
    paper_out: Path | None = None,
    min_peak_prominence: float = 0.04,
    min_peak_height: float = 0.05,
    min_peak_distance_cm1: float = 20.0,
    ignore_label_ranges: tuple[str, ...] | None = None,
    use_shoulder_detection: bool = False,
    override_file: Path | None = None,
    export_spectrum_feedback: bool = True,
    export_interactive_curation: bool = False,
    export_region_stacks: bool = False,
    regions_file: Path | None = None,
    export_chunk_data: bool = True,
    export_chunk_collage: bool = True,
    label_overrides_dir: Path | None = None,
    save_label_overrides: bool = True,
    apply_label_overrides: bool = False,
    interactive_png_export: bool = True,
    show_peak_markers: bool = False,
    allow_apparent_transmittance: bool = False,
    force_intensity_mode: str | None = None,
    apparent_transmittance_label: str = "Apparent Transmittance (%)",
    offset_gap: float = 0.15,
    stack_modes: tuple[str, ...] = ("normalized_absorbance", "transmittance"),
    region_labels: str = "selected",
    matlab_export_dir: Path | None = None,
    editable_text: bool | None = None,
    interpretation_notes: Path | None = None,
    show_deconvolution: bool = False,
    deconv_max_components_per_region: int = 6,
    deconv_min_component_height: float = 0.03,
    deconv_model: str = "pseudo_voigt",
    deconv_regions: str = "auto",
) -> tuple[Path, list[dict[str, Any]]]:
    basic_art = load_model_bundle(basic_model_path) if basic_model_path else None
    subtle_art = load_model_bundle(subtle_model_path) if subtle_model_path else None
    legacy_art = load_model_bundle(model_path) if model_path else None

    resolved_mode = _resolve_ml_mode(
        ml_mode,
        has_legacy=legacy_art is not None,
        has_basic=basic_art is not None,
        has_subtle=subtle_art is not None,
        include_evidence=include_evidence,
    )

    dens_raw = str(report_density or "balanced").lower()
    density_key = dens_raw if dens_raw in ("summary", "balanced", "audit") else "balanced"
    style_key = str(report_style or "product_v1").lower()
    if style_key not in ("legacy", "product_v1"):
        style_key = "product_v1"
    from reports.front_consensus import build_front_consensus_table_html
    from reports.front_facing_report import (
        build_front_card_stack,
        build_front_spec_title,
        interpretation_override_for_spectrum,
        is_front_audience,
        load_interpretation_notes,
        resolve_report_audience,
    )

    audience = resolve_report_audience(
        report_audience,
        report_style=style_key,
        report_density=density_key,
        front_facing_flag=front_facing,
    )
    is_front = is_front_audience(audience)
    use_editable_text = is_front if editable_text is None else bool(editable_text)
    interpretation_notes_map = load_interpretation_notes(interpretation_notes)
    notes_template_path = out_path.parent / "INTERPRETATION_NOTES.txt"
    from ml.ftir_deconv_assignment import DeconvReportConfig

    deconv_cfg = DeconvReportConfig(
        max_components_per_region=int(deconv_max_components_per_region),
        min_component_height=float(deconv_min_component_height),
        model=str(deconv_model or "pseudo_voigt"),  # type: ignore[arg-type]
        region_preset=str(deconv_regions or "auto"),
        front_max_table_rows=12,
    )
    effective_label_all = bool(label_all_diagnostic_peaks) and not is_front
    effective_max_peak_labels = int(max_peak_labels)
    if label_all_above_height is not None:
        effective_max_peak_labels = max(effective_max_peak_labels, 999)
    elif is_front and not effective_label_all:
        effective_max_peak_labels = min(effective_max_peak_labels, int(front_max_peak_labels))
    if presentation_mode:
        effective_max_peak_labels = min(effective_max_peak_labels, 14)
        if fingerprint_cluster_distance is None:
            fingerprint_cluster_distance = 20.0
    if auto_layout is None:
        auto_layout = is_front

    paths_line: list[str] = [f"ml_mode={resolved_mode}", f"fusion={fusion_mode}"]
    if basic_model_path:
        paths_line.append(f"family={_path_for_publish(basic_model_path, anonymize=anonymize_metadata)}")
    if subtle_model_path:
        paths_line.append(f"specific={_path_for_publish(subtle_model_path, anonymize=anonymize_metadata)}")
    if model_path:
        paths_line.append(f"legacy={_path_for_publish(model_path, anonymize=anonymize_metadata)}")
    paths_line.append(f"guardrails={guardrails_mode}")
    paths_line.append(f"ml_guardrails={ml_guardrails}")
    paths_line.append(f"report_density={density_key}")
    paths_line.append(f"report_style={style_key}")
    from ml.ftir_peak_picking import resolve_peak_label_thresholds, resolve_peak_thresholds

    peak_th = resolve_peak_thresholds(
        peak_sensitivity,
        peak_min_height=peak_min_height,
        peak_min_prominence=peak_min_prominence,
    )
    label_th = resolve_peak_label_thresholds(
        audience,
        peak_min_height=peak_th["peak_min_height"],
        peak_min_prominence=peak_th["peak_min_prominence"],
        peak_label_min_height=peak_label_min_height,
        peak_label_min_prominence=peak_label_min_prominence,
        peak_label_preset=peak_label_preset,
    )
    paths_line.append(f"peak_sensitivity={peak_sensitivity}")
    paths_line.append(f"peak_min_height={peak_th['peak_min_height']:.3f}")
    paths_line.append(f"peak_min_prominence={peak_th['peak_min_prominence']:.3f}")
    paths_line.append(f"peak_label_min_height={label_th['peak_label_min_height']:.3f}")
    paths_line.append(f"peak_label_min_prominence={label_th['peak_label_min_prominence']:.3f}")
    if label_th.get("peak_label_preset"):
        paths_line.append(f"peak_label_preset={label_th['peak_label_preset']}")
    if show_weak_peaks:
        paths_line.append("show_weak_peaks=on")
    if show_deconvolution:
        paths_line.append("show_deconvolution=on")
        paths_line.append(f"deconv_regions={deconv_cfg.region_preset}")
    paths_line.append(f"max_peak_labels={effective_max_peak_labels}")
    paths_line.append(f"peak_label_layout={peak_label_layout}")
    paths_line.append(f"auto_layout={'on' if auto_layout else 'off'}")
    if fingerprint_cluster_distance is not None:
        paths_line.append(f"fingerprint_cluster_distance={float(fingerprint_cluster_distance):.1f}")
    if presentation_mode:
        paths_line.append("presentation_mode=on")
    paths_line.append(f"report_audience={audience}")
    paths_line.append(f"shade_min_activity={float(shade_min_activity):.2f}")
    paths_line.append(f"shade_faint_min={float(shade_faint_min):.2f}")
    if shade_sensitive:
        paths_line.append("shade_sensitive=on")
    from reports.matlab_visual_theme import normalize_visual_theme

    theme_key = normalize_visual_theme(visual_theme)
    paths_line.append(f"visual_theme={theme_key}")
    if export_static_figures:
        paths_line.append(f"static_format={static_format}")
        paths_line.append(f"static_dpi={int(static_dpi)}")
    if export_paper_figures:
        paths_line.append("export_paper_figures=on")
        paths_line.append(f"paper_figure_modes={','.join(paper_figure_modes)}")
        paths_line.append(f"max_paper_peak_labels={int(max_paper_peak_labels)}")
        paths_line.append(f"paper_formats={','.join(paper_formats)}")
        paths_line.append(f"min_peak_prominence={float(min_peak_prominence):.3f}")
        paths_line.append(f"min_peak_height={float(min_peak_height):.3f}")
        paths_line.append(f"min_peak_distance_cm1={float(min_peak_distance_cm1):.1f}")
        if ignore_label_ranges:
            paths_line.append(f"ignore_label_ranges={','.join(ignore_label_ranges)}")
        if use_shoulder_detection:
            paths_line.append("use_shoulder_detection=on")
        if override_file:
            paths_line.append(f"override_file={_path_for_publish(override_file, anonymize=anonymize_metadata)}")
        if export_spectrum_feedback:
            paths_line.append("export_spectrum_feedback=on")
    if export_interactive_curation:
        paths_line.append("export_interactive_curation=on")
    if export_region_stacks:
        paths_line.append("export_region_stacks=on")
        paths_line.append(f"offset_gap={float(offset_gap):.2f}")
    if apply_label_overrides:
        paths_line.append("apply_label_overrides=on")
    if show_region_ruler is False:
        paths_line.append("region_ruler=off")
    elif show_region_ruler is True or (show_region_ruler is None and style_key == "product_v1"):
        paths_line.append("region_ruler=on")
    if effective_label_all:
        paths_line.append("label_all_diagnostic_peaks=on")
    if label_all_above_height is not None:
        paths_line.append(f"label_all_above_height={float(label_all_above_height):.3f}")
    if label_band_shading:
        paths_line.append("label_band_shading=on")
    paths_line.append(f"top_n_summary={int(top_n_summary)}")
    if measurement_mode:
        paths_line.append(f"measurement_mode={measurement_mode}")
    if atr_crystal:
        paths_line.append(f"atr_crystal={atr_crystal}")
    if atr_aware is not None:
        paths_line.append(f"atr_aware={atr_aware}")

    from reports.reproducibility_meta import build_reproducibility_html, build_run_context

    repro_ctx = build_run_context(
        paths_line=paths_line,
        family_model=Path(basic_model_path) if basic_model_path else None,
        specific_model=Path(subtle_model_path) if subtle_model_path else None,
        legacy_model=Path(model_path) if model_path else None,
        extra={
            "ontology": "v4",
            "guardrails_mode": guardrails_mode,
            "ml_guardrails": ml_guardrails,
            "rules_preset": "conservative",
            "report_style": style_key,
            "report_audience": audience,
            "visual_theme": theme_key,
            "feature_set": "spectral+evidence_v2",
        },
    )
    reproducibility_html = build_reproducibility_html(repro_ctx)

    ml_artifacts: dict[str, Any] = {}
    if legacy_art is not None:
        ml_artifacts["legacy"] = legacy_art
    if basic_art is not None:
        ml_artifacts["basic"] = basic_art
    if subtle_art is not None:
        ml_artifacts["subtle"] = subtle_art

    sections: list[dict[str, Any]] = []
    pipeline_batch: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matlab_stems: list[str] = []
    presentation_files: list[str] = []
    paper_manifests: list[dict[str, Any]] = []
    paper_out_dir = paper_out or (out_path.parent / "presentation" / "paper_figures")
    paper_override_store: dict[str, Any] | None = None
    if export_paper_figures and override_file:
        from reports.paper_peak_overrides import load_override_store

        paper_override_store = load_override_store(Path(override_file))
    from reports.paper_peak_selection import parse_ignore_label_ranges

    paper_ignore_ranges = parse_ignore_label_ranges(
        list(ignore_label_ranges) if ignore_label_ranges else None
    )
    curation_html_global = ""
    region_stacks_manifest: dict[str, Any] | None = None
    label_ov_dir = (
        label_overrides_dir.resolve()
        if label_overrides_dir
        else out_path.parent
    )
    static_out_dir = (
        static_out.resolve()
        if static_out
        else (out_path.parent / "presentation" / "figures" if export_static_figures else out_path.parent / "figures")
    )
    matlab_dir = (
        matlab_export_dir.resolve()
        if matlab_export_dir
        else out_path.parent / "matlab_export"
    )
    from reports.matlab_visual_theme import (
        _safe_stem,
        export_matlab_spectrum_csvs,
        write_make_figures_m,
        write_presentation_figures_index,
    )

    for i, p in enumerate(input_paths):
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        ps = load_processed_spectrum(p)
        md = _metadata_for_path(p)
        gr = str(guardrails_mode or "v3").lower()
        ev_cfg: dict[str, Any] = {
            "peak_sensitivity": peak_sensitivity,
            "max_peaks": int(max_peaks),
        }
        if peak_min_height is not None:
            ev_cfg["peak_min_height"] = float(peak_min_height)
        if peak_min_prominence is not None:
            ev_cfg["peak_min_prominence"] = float(peak_min_prominence)
        if density_key == "audit" or show_deconvolution:
            ev_cfg["compute_deconv"] = True
        pipeline = run_evidence_first_pipeline(
            ps.wn,
            ps.y,
            md=md,
            peaks=None,
            ml_mode=resolved_mode,  # type: ignore[arg-type]
            fusion_mode=fusion_mode,  # type: ignore[arg-type]
            basic_model=basic_art,
            subtle_model=subtle_art,
            legacy_model=legacy_art,
            rules_config=rules_config,
            evidence_config=ev_cfg,
            guardrails_mode=gr,
            ml_guardrails=str(ml_guardrails or "strict").lower(),  # type: ignore[arg-type]
            measurement_mode=measurement_mode or None,
            atr_crystal=atr_crystal or None,
            atr_aware=atr_aware,
            spectrum_path=p,
        )
        pipeline_batch.append(
            {"spectrum": ps.name, "path": _path_for_publish(p, anonymize=anonymize_metadata), "pipeline": pipeline}
        )

        legacy_probs = None
        if legacy_art is not None and resolved_mode in ("legacy", "basic", "both"):
            try:
                legacy_probs = predict_proba_row(legacy_art, wn=ps.wn, y=ps.y, md=md)
            except Exception:
                legacy_probs = None

        peaks_all = list((pipeline.get("evidence") or {}).get("peaks") or [])
        from ml.canonical_peaks import build_canonical_peak_table, peaks_as_legacy_dicts
        from ml.ftir_peak_picking import key_evidence_peak_wavenumbers, summarize_peak_display
        from reports.report_consistency import write_report_consistency_audit

        weak_on = show_weak_peaks or density_key == "audit"
        if label_all_above_height is not None:
            weak_on = True
        key_wn = key_evidence_peak_wavenumbers(pipeline)
        label_floor = float(label_all_above_height if label_all_above_height is not None else 0.05)
        canon = build_canonical_peak_table(
            pipeline,
            label_min_height=label_floor,
            label_all_above_height=label_all_above_height,
            report_audience=audience,
        )
        write_report_consistency_audit(
            out_path.parent,
            pipeline,
            strict=(str(audience or "").lower() == "debug" and density_key == "audit"),
        )
        peaks_disp = peaks_as_legacy_dicts(canon, plotted=True)
        peaks_lab = peaks_as_legacy_dicts(canon, labeled=True)
        peaks_kron = list(peaks_disp)
        fig_cluster_dist = 0.0 if label_all_above_height is not None else fingerprint_cluster_distance

        peak_counts = summarize_peak_display(
            peaks_all,
            show_weak_peaks=weak_on,
            max_peak_labels=max(effective_max_peak_labels, len(peaks_lab)),
            label_all_diagnostic=effective_label_all,
            report_density=density_key,
            report_audience=audience,
            peak_label_min_height=label_th["peak_label_min_height"],
            peak_label_min_prominence=label_th["peak_label_min_prominence"],
            peak_label_preset=peak_label_preset,
            key_evidence_wn=key_wn,
            peak_min_height=peak_th["peak_min_height"],
            peak_min_prominence=peak_th["peak_min_prominence"],
            peak_sensitivity=peak_sensitivity,
            label_all_above_height=label_all_above_height,
        )
        peak_counts["n_plotted_peaks"] = canon["stats"].get("plotted_peak_count", len(peaks_disp))
        peak_counts["n_labeled_peaks"] = canon["stats"].get("labeled_peak_count", len(peaks_lab))
        peak_counts["canonical_peak_count"] = canon["stats"].get("canonical_peak_count", len(peaks_all))
        pipeline["peak_display"] = peak_counts
        anchor = f"spec-{i}-{p.stem.replace(' ', '_')}"
        deconv_pack: dict[str, Any] | None = None
        deconv_table_html = ""
        if show_deconvolution:
            from ml.ftir_deconv_assignment import run_report_deconvolution
            from reports.deconv_report import build_deconv_candidates_table_html

            deconv_pack = run_report_deconvolution(ps.wn, ps.y, pipeline, config=deconv_cfg)
            pipeline["deconv_report"] = deconv_pack
            drows = (
                deconv_pack.get("table_rows_audit")
                if density_key == "audit"
                else deconv_pack.get("table_rows")
            ) or []
            deconv_table_html = build_deconv_candidates_table_html(
                drows,
                anchor=anchor,
                audit=density_key == "audit",
                max_rows=deconv_cfg.front_max_table_rows,
            )

        pwn = [float(p["wn_cm1"]) for p in peaks_disp]
        ph = [float(p["height"]) for p in peaks_disp]
        fig, fig_meta = _build_stacked_interactive_figure(
            name=ps.name,
            wn=ps.wn,
            y=ps.y,
            peak_wn=pwn,
            peak_h=ph,
            pipeline=pipeline,
            peaks_dicts=peaks_disp,
            show_band_shading=show_band_shading,
            show_region_ruler=show_region_ruler,
            include_ml=include_ml and resolved_mode != "none",
            hover_top_fg=hover_top_fg,
            report_density=str(density_key),
            report_style=style_key,
            ontology=str(pipeline.get("ontology") or "v3"),
            label_band_shading=label_band_shading,
            show_weak_peaks=weak_on,
            max_peak_labels=effective_max_peak_labels,
            label_all_diagnostic_peaks=effective_label_all,
            peaks_plotted=peaks_disp,
            peaks_labeled=peaks_lab,
            shade_min_activity=float(shade_min_activity),
            shade_faint_min=float(shade_faint_min),
            shade_sensitive=shade_sensitive,
            report_audience=audience,
            visual_theme=theme_key,
            panel_mode="full",
            label_all_above_height=label_all_above_height,
            presentation_mode=presentation_mode,
            show_deconvolution=show_deconvolution,
            deconv_pack=deconv_pack,
            peak_label_layout=peak_label_layout,
            auto_layout=auto_layout,
            fingerprint_cluster_distance=fig_cluster_dist,
        )
        if fig_meta.get("unlabeled_reason_counts"):
            for k, v in fig_meta["unlabeled_reason_counts"].items():
                peak_counts.setdefault("unlabeled_reason_counts", {})[k] = (
                    peak_counts.get("unlabeled_reason_counts", {}).get(k, 0) + int(v)
                )
        if label_th.get("peak_label_preset"):
            peak_counts["peak_label_preset"] = label_th["peak_label_preset"]
        if export_static_figures:
            fig_spec, _ = _build_stacked_interactive_figure(
                name=ps.name,
                wn=ps.wn,
                y=ps.y,
                peak_wn=pwn,
                peak_h=ph,
                pipeline=pipeline,
                peaks_dicts=peaks_disp,
                show_band_shading=show_band_shading,
                show_region_ruler=show_region_ruler,
                include_ml=include_ml and resolved_mode != "none",
                hover_top_fg=hover_top_fg,
                report_density=str(density_key),
                report_style=style_key,
                ontology=str(pipeline.get("ontology") or "v3"),
                label_band_shading=label_band_shading,
                show_weak_peaks=weak_on,
                max_peak_labels=effective_max_peak_labels,
                label_all_diagnostic_peaks=effective_label_all,
                peaks_plotted=peaks_disp,
                peaks_labeled=peaks_lab,
                shade_min_activity=float(shade_min_activity),
                shade_faint_min=float(shade_faint_min),
                shade_sensitive=shade_sensitive,
                report_audience=audience,
                visual_theme=theme_key,
                panel_mode="spectrum",
                label_all_above_height=label_all_above_height,
                presentation_mode=presentation_mode,
                peak_label_layout=peak_label_layout,
                auto_layout=auto_layout,
                fingerprint_cluster_distance=fig_cluster_dist,
            )
            fig_kron, _ = _build_stacked_interactive_figure(
                name=ps.name,
                wn=ps.wn,
                y=ps.y,
                peak_wn=pwn,
                peak_h=ph,
                pipeline=pipeline,
                peaks_dicts=peaks_disp,
                show_band_shading=False,
                show_region_ruler=False,
                include_ml=include_ml and resolved_mode != "none",
                hover_top_fg=hover_top_fg,
                report_density=str(density_key),
                report_style=style_key,
                ontology=str(pipeline.get("ontology") or "v3"),
                show_weak_peaks=weak_on,
                max_peak_labels=effective_max_peak_labels,
                peaks_plotted=peaks_disp,
                peaks_labeled=peaks_lab,
                report_audience=audience,
                visual_theme=theme_key,
                panel_mode="kronecker",
                label_all_above_height=label_all_above_height,
                presentation_mode=presentation_mode,
                peak_label_layout=peak_label_layout,
                auto_layout=auto_layout,
                fingerprint_cluster_distance=fig_cluster_dist,
            )
            from reports.static_figure_export import export_static_matplotlib_bundle

            _spec_static_policy = (
                "all"
                if label_all_above_height is not None
                else str(static_peak_label_policy)  # type: ignore[assignment]
            )
            static_result = export_static_matplotlib_bundle(
                spectrum_name=ps.name,
                wn=ps.wn,
                y=ps.y,
                pipeline=pipeline,
                canonical_pack=canon,
                out_dir=static_out_dir,
                fmt=str(static_format),
                dpi=int(static_dpi),
                static_label_policy=str(static_peak_label_policy),  # type: ignore[arg-type]
                spectrum_label_policy=_spec_static_policy,  # type: ignore[arg-type]
                max_static_labels=int(max_static_peak_labels),
                show_ruler=bool(show_region_ruler if show_region_ruler is not None else style_key == "product_v1"),
                export_separate_panels=True,
            )
            fig_meta["static_export"] = static_result
            presentation_files.extend(static_result.get("files") or [])
        paper_figures_html = ""
        curation_html = ""
        paper_manifest: dict[str, Any] | None = None
        if export_paper_figures:
            from reports.paper_ftir_figures import (
                PaperFigureConfig,
                build_paper_figures_section_html,
                export_paper_figures_for_spectrum,
            )

            _modes = tuple(
                m
                for m in paper_figure_modes
                if m in ("transmittance", "normalized_absorbance")
            ) or ("transmittance", "normalized_absorbance")
            _fmts = tuple(f for f in paper_formats if f in ("png", "svg", "pdf")) or (
                "png",
                "svg",
                "pdf",
            )
            pcfg = PaperFigureConfig(
                modes=_modes,  # type: ignore[arg-type]
                formats=_fmts,
                max_peak_labels=int(max_paper_peak_labels),
                label_style=(
                    paper_label_style
                    if paper_label_style in ("horizontal-leader",)
                    else "horizontal-leader"
                ),  # type: ignore[arg-type]
                dpi=int(static_dpi or 300),
                peak_sensitivity=str(peak_sensitivity),
                min_peak_prominence=float(min_peak_prominence),
                min_peak_height=float(min_peak_height),
                min_peak_distance_cm1=float(min_peak_distance_cm1),
                ignore_label_ranges=list(paper_ignore_ranges),
                use_shoulder_detection=bool(use_shoulder_detection),
                override_file=Path(override_file) if override_file else None,
                export_spectrum_feedback=bool(export_spectrum_feedback),
                label_overrides_dir=label_ov_dir,
                apply_label_overrides=bool(apply_label_overrides),
                save_label_overrides=bool(save_label_overrides),
                show_peak_markers=bool(show_peak_markers),
                allow_apparent_transmittance=bool(allow_apparent_transmittance),
                force_intensity_mode=force_intensity_mode,  # type: ignore[arg-type]
                apparent_transmittance_label=str(apparent_transmittance_label),
            )
            paper_manifest = export_paper_figures_for_spectrum(
                p,
                paper_out_dir,
                config=pcfg,
                pipeline=pipeline,
                override_store=paper_override_store,
            )
            paper_manifests.append(paper_manifest)
            for flist in (paper_manifest.get("figures") or {}).values():
                presentation_files.extend(flist)
            if str(paper_report_style).lower() in ("both", "product", "full"):
                paper_figures_html = build_paper_figures_section_html(
                    paper_manifest,
                    report_dir=out_path.parent,
                )
        if export_interactive_curation:
            from reports.interactive_curation import export_interactive_curation_for_spectrum
            from reports.paper_peak_selection import PaperPeakSelectionConfig

            sel_cfg = PaperPeakSelectionConfig(
                min_prominence=float(min_peak_prominence),
                min_height=float(min_peak_height),
                min_distance_cm1=float(min_peak_distance_cm1),
                max_labels=int(max_paper_peak_labels),
                ignore_label_ranges=list(paper_ignore_ranges),
            )
            cur = export_interactive_curation_for_spectrum(
                p,
                report_dir=out_path.parent,
                overrides_dir=label_ov_dir,
                selection_config=sel_cfg,
                apply_saved_overrides=True,
                save_auto_overrides=bool(save_label_overrides),
                allow_apparent_transmittance=bool(allow_apparent_transmittance),
                force_intensity_mode=force_intensity_mode,  # type: ignore[arg-type]
                apparent_transmittance_label=str(apparent_transmittance_label),
            )
            curation_html = cur.get("html", "")
            if paper_manifest is None:
                paper_manifest = {"stem": cur.get("stem"), "curation": cur}
            else:
                paper_manifest["curation"] = cur
        if export_static_figures or theme_key == "matlab":
            stem = _safe_stem(ps.name)
            matlab_stems.append(stem)
            export_matlab_spectrum_csvs(
                matlab_dir=matlab_dir,
                spectrum_name=ps.name,
                wn=ps.wn,
                y=ps.y,
                peaks_labeled=peaks_lab,
                peaks_plotted=peaks_disp,
                pipeline=pipeline,
                fig_meta=fig_meta,
                include_evidence=include_evidence,
            )
        use_ruler = bool(fig_meta.get("show_region_ruler"))
        shade_note = (
            " Optional background shading shows regional activity only."
            if show_band_shading
            else ""
        )
        if style_key == "product_v1":
            if is_front:
                fig_note = ""
            else:
                fig_note = (
                    "<p class='product-hint'>FTIR region ruler (top) shows traditional tentative ranges; "
                    "hover the spectrum for local band context and labeled diagnostic peaks."
                    f"{shade_note}</p>"
                )
        else:
            fig_note = (
                "<p class='hint'>Hover: local band evidence at each wavenumber (not global ML probabilities). "
                "Diamond markers label strongest diagnostic peaks by category."
                f"{shade_note}</p>"
            )
        ev_sum = (pipeline.get("evidence") or {}).get("summary") or {}
        peaks_all = list((pipeline.get("evidence") or {}).get("peaks") or [])
        peak_picking_meta = {
            "peak_sensitivity": ev_sum.get("peak_sensitivity", peak_sensitivity),
            "peak_min_height": ev_sum.get("peak_min_height", peak_th["peak_min_height"]),
            "peak_min_prominence": ev_sum.get(
                "peak_min_prominence", peak_th["peak_min_prominence"]
            ),
            "peak_label_min_height": label_th["peak_label_min_height"],
            "peak_label_min_prominence": label_th["peak_label_min_prominence"],
            "peak_label_preset": peak_counts.get("peak_label_preset") or label_th.get("peak_label_preset"),
            "n_detected_peaks": peak_counts.get("n_detected_peaks", len(peaks_all)),
            "n_plotted_peaks": peak_counts.get("n_plotted_peaks", len(peaks_disp)),
            "n_labeled_peaks": peak_counts.get(
                "n_labeled_peaks", fig_meta.get("n_labeled_peaks", 0)
            ),
            "detected_peaks_count": peak_counts.get("detected_peaks_count", len(peaks_all)),
            "displayed_peaks_count": peak_counts.get("displayed_peaks_count", len(peaks_disp)),
            "labeled_peaks_count": peak_counts.get(
                "labeled_peaks_count", fig_meta.get("n_labeled_peaks", 0)
            ),
            "label_reason_counts": peak_counts.get("label_reason_counts"),
            "unlabeled_reason_counts": peak_counts.get("unlabeled_reason_counts"),
            "n_diagnostic_peaks": ev_sum.get("n_diagnostic_peaks"),
            "n_weak_peaks": ev_sum.get("n_weak_peaks"),
        }
        meta_payload: dict[str, Any] = {
            **{k: md.get(k) for k in ("title", "name", "cas", "formula", "xunits") if md.get(k)},
            "guardrails": gr,
            "ml_guardrails": str(ml_guardrails),
            "ontology": pipeline.get("ontology"),
            "peak_picking": peak_picking_meta,
        }
        if str(audience or "").lower() == "debug" or density_key == "audit":
            try:
                from ml.external.provenance import provenance_summary

                prov = provenance_summary(md)
                if prov:
                    meta_payload["provenance"] = prov
            except ImportError:
                pass
        meta_line = json.dumps(meta_payload, sort_keys=True)

        ml_enabled = bool(include_ml and resolved_mode != "none")
        row = spectrum_summary_row(name=ps.name, anchor=anchor, pipeline=pipeline, ml_enabled=ml_enabled)
        summary_rows.append(row)

        top_asg = ", ".join(f"{a[0]}" for a in (row.get("top3") or [])[:3]) or "—"
        caut_hdr = row.get("caut_major") or row.get("caut_short") or []
        caut_top = "; ".join(_truncate(str(c), 120) for c in caut_hdr[:2]) or "—"
        title_row_html = spectrum_card_title_row_html(
            name=ps.name,
            anchor=anchor,
            status=row["status"],
            top_assignments=top_asg,
            top_cautions=caut_top,
            ml_status=str(row.get("ml_agree", "")),
        )
        meta_open = density_key == "audit" and style_key != "product_v1"
        meta_html = (
            f"<details class='metadata-details'{' open' if meta_open else ''}>"
            "<summary>Metadata</summary>"
            f"<pre class='mono'>{html.escape(meta_line)}</pre></details>"
        )

        explain_html = ""
        band_map_html = ""
        justify_panels_html = ""
        if include_evidence:
            from reports.report_render import (
                MARKER_AMBIGUITY,
                MARKER_ARTIFACT_FLAGS,
                render_band_evidence_map,
                render_evidence_table,
            )

            explain_html = render_evidence_table(pipeline, anchor=anchor)
            if str(pipeline.get("ontology") or "").lower() == "v4":
                from reports.v4_evidence_report import build_fg_justification_panels_html

                band_map_html = render_band_evidence_map(
                    pipeline, anchor=anchor, audience=audience
                )
                justify_panels_html = build_fg_justification_panels_html(pipeline, anchor=anchor)
                if show_deconvolution and deconv_pack:
                    from reports.deconv_report import append_deconv_to_justify_panels

                    justify_panels_html = append_deconv_to_justify_panels(
                        justify_panels_html, pipeline, deconv_pack
                    )
            if show_ambiguity_labels and (pipeline.get("rule_assignments") or {}).get("ambiguity_labels"):
                explain_html = MARKER_AMBIGUITY + explain_html
            if show_artifact_flags:
                art = (pipeline.get("evidence") or {}).get("artifacts") or {}
                if art.get("flags") or art.get("cautions"):
                    explain_html = MARKER_ARTIFACT_FLAGS + explain_html

        fg_block = ""
        just_block = ""
        summary_details = ""
        body_rest = ""
        if include_evidence:
            from reports.report_render import render_consensus_table

            fg_block = render_consensus_table(pipeline, top_n=12) if include_consensus else ""
            just_block = build_fg_justification_intro_html(pipeline, top_n=4)
            if density_key in ("summary", "balanced"):
                fg_block = f"<details class='fold-consensus'><summary>Consensus table</summary>{fg_block}</details>"
                just_block = f"<details><summary>Evidence notes</summary>{just_block}</details>"
            summary_details = spectrum_summary_quick_details_html(
                anchor=anchor,
                row=row,
                pipeline=pipeline,
                density=density_key,
                top_n_summary=int(top_n_summary),
                include_evidence=True,
            )
            body_rest = build_spectrum_body_html(
                pipeline=pipeline,
                anchor=anchor,
                gr=gr,
                density=density_key,
                top_n_summary=int(top_n_summary),
                include_evidence=include_evidence,
                include_ml=include_ml,
                include_consensus=include_consensus,
                resolved_mode=resolved_mode,
                legacy_probs=legacy_probs,
                show_ambiguity_labels=show_ambiguity_labels,
                show_artifact_flags=show_artifact_flags,
                omit_quick_interpretation=True,
                plot_first_layout=True,
            )
        else:
            summary_details = spectrum_summary_quick_details_html(
                anchor=anchor,
                row=row,
                pipeline=pipeline,
                density=density_key,
                top_n_summary=int(top_n_summary),
                include_evidence=False,
            )
            body_rest = "<p class='muted'>Evidence sections omitted for this export.</p>"

        if style_key == "product_v1":
            from reports.product_v1_report import (
                MARKER_SPECTRUM_ANNOTATIONS,
                build_peak_labeling_summary_html,
                build_peak_picking_summary_html,
                build_product_tables_stack,
            )

            peak_pick_html = build_peak_picking_summary_html(peak_picking_meta)
            peak_label_html = (
                build_peak_labeling_summary_html(peak_picking_meta)
                if not is_front
                else ""
            )
            audit_body = "<!-- report-feature:product-audit -->" + body_rest
            if is_front:
                from reports.front_facing_report import sanitize_run_settings_line

                settings_for_card = sanitize_run_settings_line(" | ".join(paths_line))
            else:
                settings_for_card = ""
            if is_front:
                stem = _safe_stem(ps.name)
                tables_stack = MARKER_SPECTRUM_ANNOTATIONS + build_front_card_stack(
                    pipeline=pipeline,
                    anchor=anchor,
                    ml_enabled=ml_enabled,
                    include_evidence=include_evidence,
                    audit_html=audit_body,
                    band_map_html=band_map_html,
                    justify_html=justify_panels_html,
                    explain_html=explain_html,
                    fg_block=fg_block,
                    just_block=just_block,
                    meta_html=meta_html,
                    peak_picking_html="",
                    peak_picking_meta=peak_picking_meta,
                    show_metadata=show_metadata or density_key == "audit",
                    run_settings_line=settings_for_card,
                    reproducibility_html=reproducibility_html,
                    editable_text=use_editable_text,
                    summary_override=interpretation_override_for_spectrum(
                        interpretation_notes_map, ps.name
                    ),
                    presentation_figures_html=(paper_figures_html or "") + (curation_html or ""),
                )
                if deconv_table_html:
                    from reports.front_facing_report import MARKER_FRONT_TECHNICAL

                    tables_stack = tables_stack.replace(
                        MARKER_FRONT_TECHNICAL,
                        deconv_table_html + MARKER_FRONT_TECHNICAL,
                        1,
                    )
            else:
                tables_stack = (
                    MARKER_SPECTRUM_ANNOTATIONS
                    + build_product_tables_stack(
                        pipeline=pipeline,
                        anchor=anchor,
                        ml_enabled=ml_enabled,
                        include_evidence=include_evidence,
                        audit_html=audit_body,
                        band_map_html=band_map_html,
                        justify_html=justify_panels_html,
                        explain_html=explain_html,
                        fg_block=fg_block,
                        just_block=just_block,
                        meta_html=meta_html,
                        density=density_key,
                        peak_picking_html=peak_pick_html + deconv_table_html,
                        peak_labeling_html=peak_label_html,
                        reproducibility_html=reproducibility_html,
                    )
                )
                if paper_figures_html:
                    tables_stack += paper_figures_html
                if curation_html:
                    tables_stack += curation_html
        else:
            tables_stack = (
                title_row_html
                + band_map_html
                + justify_panels_html
                + explain_html
                + fg_block
                + just_block
                + "<div class='expand-stack'>"
                + summary_details
                + meta_html
                + f"<div class='post-figure-blocks'>{body_rest}</div>"
                + "</div>"
            )

        robustness_badge = ""
        if include_robustness and ml_artifacts:
            _, summ = evaluate_robustness_one_spectrum(ps.wn, ps.y, md, ml_artifacts)
            robustness_badge = _robustness_badge_html(float(summ.get("overall_robustness_score", 0)))

        spec_title_html = build_front_spec_title(ps.name) if is_front and style_key == "product_v1" else ""
        sections.append(
            {
                "name": ps.name,
                "anchor": anchor,
                "meta_line": meta_line,
                "title_html": spec_title_html,
                "figure": fig,
                "figure_note_html": fig_note,
                "table_html": "",
                "tables_html": tables_stack,
                "interpret_html": "",
                "robustness_badge": robustness_badge if not is_front else "",
                "sidebar_badge_html": status_badge_html(row["status"], front_mode=is_front),
                "band_shading": show_band_shading,
                "region_ruler": use_ruler,
                "peak_labels": bool(fig_meta.get("has_peak_labels")),
                "paper_manifest": paper_manifest,
            }
        )

    ml_enabled_batch = bool(include_ml and resolved_mode != "none")
    include_summary_for_html = include_summary_table and not is_front
    summary_html = ""
    if include_summary_for_html:
        if style_key == "product_v1":
            from reports.product_v1_report import build_product_summary_table_html, enrich_summary_rows

            pipelines_only = [x["pipeline"] for x in pipeline_batch]
            enriched = enrich_summary_rows(summary_rows, pipelines_only)
            if is_front:
                summary_html = build_front_consensus_table_html(
                    rows=enriched,
                    ml_enabled=ml_enabled_batch,
                )
            else:
                summary_html = build_product_summary_table_html(
                    rows=enriched,
                    ml_enabled=ml_enabled_batch,
                )
        else:
            from reports.report_render import render_summary_table

            summary_html = render_summary_table(summary_rows, ml_enabled=ml_enabled_batch)

    if use_editable_text and is_front and not notes_template_path.is_file():
        note_sections = []
        for inp in input_paths:
            stem = _safe_stem(Path(inp).name)
            note_sections.append(f"## {stem}\n\nEdit your interpretation for {Path(inp).name} here.\n")
        notes_template_path.write_text(
            "\n".join(note_sections)
            + "\nRe-run with:\n"
            f'  --interpretation-notes "{notes_template_path}"\n',
            encoding="utf-8",
        )
    if export_paper_figures and paper_manifests:
        from reports.paper_ftir_figures import write_paper_figures_index

        write_paper_figures_index(paper_out_dir, paper_manifests)
    if export_static_figures or presentation_files:
        write_presentation_figures_index(
            out_path.parent / "presentation",
            figure_files=presentation_files,
            report_html=out_path,
            notes_template=notes_template_path if notes_template_path.is_file() else None,
        )

    region_stacks_html = ""
    if export_region_stacks and input_paths:
        from reports.region_stack_export import (
            build_region_stacks_section_html,
            export_region_stacks,
            spectra_from_batch,
        )

        _smodes = tuple(
            m for m in stack_modes if m in ("normalized_absorbance", "transmittance")
        ) or ("normalized_absorbance", "transmittance")
        stack_specs = spectra_from_batch(
            [Path(x) for x in input_paths],
            paper_out_dir=paper_out_dir if export_paper_figures else None,
        )
        region_stacks_manifest = export_region_stacks(
            spectra=stack_specs,
            out_dir=out_path.parent,
            regions_file=regions_file,
            stack_modes=_smodes,  # type: ignore[arg-type]
            formats=paper_formats if export_paper_figures else ("png", "svg", "pdf"),
            offset_gap=float(offset_gap),
            region_labels=str(region_labels),
            dpi=int(static_dpi or 300),
            ignore_label_ranges=list(paper_ignore_ranges),
            show_peak_markers=bool(show_peak_markers),
            export_chunk_data=bool(export_chunk_data),
            export_collage=bool(export_chunk_collage),
            allow_apparent_transmittance=bool(allow_apparent_transmittance),
            force_intensity_mode=force_intensity_mode,  # type: ignore[arg-type]
            apparent_transmittance_label=str(apparent_transmittance_label),
        )
        region_stacks_html = build_region_stacks_section_html(
            region_stacks_manifest, out_path.parent
        )
        presentation_files.extend(
            p for paths in region_stacks_manifest.get("outputs", {}).values() for p in paths
        )

    extra_css = ""
    extra_script = ""
    extra_body_html = region_stacks_html
    if export_region_stacks:
        from reports.range_editor import range_editor_css, range_editor_js

        extra_css += f"<style>{range_editor_css()}</style>"
        extra_script += range_editor_js()
    if export_interactive_curation:
        from reports.interactive_curation import curation_css, curation_js

        extra_css += f"<style>{curation_css()}</style>"
        extra_script += curation_js()

    html_path = write_interactive_report_html(
        out_path=out_path,
        page_title=page_title,
        subtitle=subtitle,
        model_path=None if anonymize_metadata else model_path,
        model_paths_line=" | ".join(paths_line),
        sections=sections,
        summary_table_html=summary_html,
        report_density_label=str(density_key),
        redacted_paths_notice=anonymize_metadata,
        include_summary_table=include_summary_for_html,
        report_style=style_key,
        report_audience=audience,
        visual_theme=theme_key,
        export_static_figures=export_static_figures,
        editable_text=use_editable_text,
        extra_css=extra_css,
        extra_body_html=extra_body_html,
        extra_script=extra_script,
    )
    manuscript_path: Path | None = None
    if export_paper_figures and str(paper_report_style).lower() in ("both", "manuscript"):
        from reports.manuscript_report import write_manuscript_report_html

        manuscript_path = write_manuscript_report_html(
            out_path=out_path.parent / "MANUSCRIPT_REPORT.html",
            page_title=page_title,
            sections=sections,
            paper_manifests=paper_manifests,
            report_dir=out_path.parent,
            region_stacks_html=region_stacks_html,
        )
    if matlab_stems:
        write_make_figures_m(matlab_dir, matlab_stems)
    if export_csv_dir is not None:
        export_pipeline_batch_csv(pipeline_batch, export_csv_dir)
    return html_path, pipeline_batch


def cmd_batch(args: argparse.Namespace) -> int:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise SystemExit("Plotly required: pip install plotly") from None

    paths_src = args.inputs or getattr(args, "input_alt", None)
    if not paths_src:
        raise SystemExit("Provide --inputs PATH [PATH...] or --input PATH [PATH...]")
    paths = [_resolve_under_chunks(Path(x)) for x in paths_src]
    out = _resolve_under_chunks(Path(args.out))
    if out.suffix.lower() != ".html":
        out = out / "REPORT.html"
    basic_p = Path(args.basic_model) if getattr(args, "basic_model", "") else None
    subtle_p = Path(args.subtle_model) if getattr(args, "subtle_model", "") else None
    if getattr(args, "family_model", ""):
        basic_p = Path(args.family_model)
    if getattr(args, "specific_model", ""):
        subtle_p = Path(args.specific_model)
    legacy_p = Path(args.model) if getattr(args, "model", "") else None
    ml_mode = getattr(args, "ml_mode", "auto")
    if ml_mode == "legacy" and legacy_p is None and basic_p is None:
        legacy_p = Path("models/struct_fg_v7_pubchem_mordred.joblib")

    rules_config = None
    preset = (getattr(args, "rules_preset", "") or "").strip()
    rc_path = (getattr(args, "rules_config", "") or "").strip()
    ont_arg = (getattr(args, "ontology", "") or "").strip().lower()
    if preset or rc_path:
        rules_config = load_rules_config(
            preset=preset or None,
            config_path=rc_path or None,
        )
    if ont_arg in ("v3", "v4"):
        rules_config = dict(rules_config or {})
        rules_config["ontology"] = ont_arg
    if getattr(args, "suppress_nitro_reporting", False):
        from ml.report_suppression import nitro_suppression_rules_patch

        rules_config = dict(rules_config or {})
        rules_config = {**rules_config, **nitro_suppression_rules_patch()}

    export_csv_dir = None
    if getattr(args, "export_csv", None):
        export_csv_dir = _resolve_under_chunks(Path(args.export_csv))

    from reports.front_facing_report import front_page_header, is_front_audience, resolve_report_audience

    report_style = str(getattr(args, "report_style", "product_v1"))
    report_density = str(getattr(args, "report_density", "balanced"))
    audience = resolve_report_audience(
        getattr(args, "report_audience", None),
        report_style=report_style,
        report_density=report_density,
        front_facing_flag=bool(getattr(args, "front_facing", False)),
    )
    default_title = "Structural FG SVM — interactive Kronecker report"
    page_title = args.title
    subtitle = args.subtitle
    if is_front_audience(audience):
        ft, fs = front_page_header(n_spectra=len(paths), audience=audience)
        if page_title == default_title:
            page_title = ft
        if not subtitle:
            subtitle = fs

    rp, _batch = run_batch(
        input_paths=paths,
        model_path=_resolve_under_chunks(legacy_p) if legacy_p else None,
        basic_model_path=_resolve_under_chunks(basic_p) if basic_p else None,
        subtle_model_path=_resolve_under_chunks(subtle_p) if subtle_p else None,
        out_path=out,
        page_title=page_title,
        subtitle=subtitle or f"{len(paths)} spectra | evidence-first FTIR + Kronecker",
        max_peaks=int(args.max_peaks),
        hover_top_fg=int(args.hover_top_fg),
        ml_mode=ml_mode,
        fusion_mode=getattr(args, "fusion_mode", "annotate"),
        include_evidence=bool(getattr(args, "include_evidence", True)),
        include_ml=bool(getattr(args, "include_ml", True)),
        include_consensus=bool(getattr(args, "include_consensus", True)),
        include_robustness=bool(getattr(args, "include_robustness", False)),
        rules_config=rules_config,
        export_csv_dir=export_csv_dir,
        show_band_shading=bool(getattr(args, "show_band_shading", False)),
        show_region_ruler=(
            None
            if getattr(args, "show_region_ruler", None) is None
            else bool(args.show_region_ruler)
        ),
        label_all_diagnostic_peaks=bool(getattr(args, "label_all_diagnostic_peaks", False)),
        label_all_above_height=getattr(args, "label_all_above_height", None),
        guardrails_mode=str(getattr(args, "guardrails", "v3")),
        ml_guardrails=str(getattr(args, "ml_guardrails", "strict")),
        show_ambiguity_labels=bool(getattr(args, "show_ambiguity_labels", True)),
        show_artifact_flags=bool(getattr(args, "show_artifact_flags", True)),
        report_density=str(getattr(args, "report_density", "balanced")),
        top_n_summary=int(getattr(args, "top_n_summary", 5)),
        rules_preset_label=preset or "",
        rules_config_path_label=rc_path or "",
        anonymize_metadata=bool(getattr(args, "anonymize_metadata", False)),
        include_summary_table=not bool(getattr(args, "no_summary_table", False)),
        measurement_mode=str(getattr(args, "mode", "") or ""),
        atr_crystal=str(getattr(args, "atr_crystal", "") or ""),
        atr_aware=(
            None
            if getattr(args, "atr_aware", None) is None
            else bool(getattr(args, "atr_aware"))
        ),
        report_style=str(getattr(args, "report_style", "product_v1")),
        label_band_shading=bool(getattr(args, "label_band_shading", False)),
        peak_sensitivity=str(getattr(args, "peak_sensitivity", "balanced")),
        show_weak_peaks=bool(getattr(args, "show_weak_peaks", False)),
        max_peak_labels=int(getattr(args, "max_peak_labels", 24)),
        peak_min_height=getattr(args, "peak_min_height", None),
        peak_min_prominence=getattr(args, "peak_min_prominence", None),
        peak_label_min_height=getattr(args, "peak_label_min_height", None),
        peak_label_min_prominence=getattr(args, "peak_label_min_prominence", None),
        peak_label_preset=getattr(args, "peak_label_preset", None),
        presentation_mode=bool(getattr(args, "presentation_mode", False)),
        peak_label_layout=str(getattr(args, "peak_label_layout", "smart")),
        auto_layout=(
            None
            if getattr(args, "auto_layout", None) is None
            else bool(args.auto_layout)
        ),
        fingerprint_cluster_distance=getattr(args, "fingerprint_cluster_distance", 18.0),
        shade_min_activity=float(getattr(args, "shade_min_activity", 0.10)),
        shade_faint_min=float(getattr(args, "shade_faint_min", 0.05)),
        shade_sensitive=bool(getattr(args, "shade_sensitive", False)),
        report_audience=audience,
        front_max_peak_labels=int(getattr(args, "front_max_peak_labels", 30)),
        show_metadata=bool(getattr(args, "show_metadata", False)),
        front_facing=bool(getattr(args, "front_facing", False)),
        visual_theme=str(getattr(args, "visual_theme", "default")),
        export_static_figures=bool(getattr(args, "export_static_figures", False)),
        static_format=str(getattr(args, "static_format", "png")),
        static_dpi=int(getattr(args, "static_dpi", 300)),
        static_out=(
            _resolve_under_chunks(Path(args.static_out))
            if getattr(args, "static_out", "")
            else None
        ),
        editable_text=(
            None
            if getattr(args, "editable_text", None) is None
            else bool(args.editable_text)
        ),
        interpretation_notes=(
            _resolve_under_chunks(Path(args.interpretation_notes))
            if getattr(args, "interpretation_notes", "")
            else None
        ),
        show_deconvolution=bool(getattr(args, "show_deconvolution", False)),
        deconv_max_components_per_region=int(
            getattr(args, "deconv_max_components_per_region", 6)
        ),
        deconv_min_component_height=float(
            getattr(args, "deconv_min_component_height", 0.03)
        ),
        deconv_model=str(getattr(args, "deconv_model", "pseudo_voigt")),
        deconv_regions=str(getattr(args, "deconv_regions", "auto")),
        static_peak_label_policy=str(getattr(args, "static_peak_label_policy", "key")),
        max_static_peak_labels=int(getattr(args, "max_static_peak_labels", 12)),
        export_paper_figures=bool(getattr(args, "export_paper_figures", False)),
        paper_figure_modes=tuple(getattr(args, "paper_figure_modes", None) or ("transmittance", "normalized_absorbance")),
        paper_label_style=str(getattr(args, "paper_label_style", "horizontal-leader")),
        paper_report_style=str(getattr(args, "paper_report_style", "both")),
        paper_formats=tuple(getattr(args, "paper_formats", None) or ("png", "svg", "pdf")),
        max_paper_peak_labels=int(getattr(args, "max_paper_peak_labels", 10)),
        paper_out=(
            _resolve_under_chunks(Path(args.paper_out))
            if getattr(args, "paper_out", "")
            else None
        ),
        min_peak_prominence=float(getattr(args, "min_peak_prominence", 0.04)),
        min_peak_height=float(getattr(args, "min_peak_height", 0.05)),
        min_peak_distance_cm1=float(getattr(args, "min_peak_distance_cm1", 20.0)),
        ignore_label_ranges=tuple(getattr(args, "ignore_label_ranges", None) or ("900:400",)),
        use_shoulder_detection=bool(getattr(args, "use_shoulder_detection", False)),
        override_file=(
            _resolve_under_chunks(Path(args.override_file))
            if getattr(args, "override_file", "")
            else None
        ),
        export_spectrum_feedback=bool(getattr(args, "export_spectrum_feedback", True)),
        export_interactive_curation=bool(getattr(args, "export_interactive_curation", False)),
        export_region_stacks=bool(getattr(args, "export_region_stacks", False)),
        regions_file=(
            _resolve_under_chunks(Path(args.regions_file))
            if getattr(args, "regions_file", "")
            else None
        ),
        export_chunk_data=bool(
            getattr(args, "export_chunk_data", True)
            if getattr(args, "export_region_stacks", False)
            else False
        ),
        export_chunk_collage=bool(getattr(args, "chunk_collage", True)),
        label_overrides_dir=(
            _resolve_under_chunks(Path(args.label_overrides))
            if getattr(args, "label_overrides", "")
            else None
        ),
        save_label_overrides=bool(getattr(args, "save_label_overrides", True)),
        apply_label_overrides=bool(getattr(args, "apply_label_overrides", False)),
        interactive_png_export=bool(getattr(args, "interactive_png_export", True)),
        show_peak_markers=bool(getattr(args, "show_peak_markers", False)),
        allow_apparent_transmittance=bool(getattr(args, "allow_apparent_transmittance", False)),
        force_intensity_mode=(
            str(getattr(args, "force_intensity_mode", "") or "") or None
        ),
        apparent_transmittance_label=str(
            getattr(args, "apparent_transmittance_label", "Apparent Transmittance (%)")
        ),
        offset_gap=float(getattr(args, "offset_gap", 0.15)),
        stack_modes=tuple(
            getattr(args, "chunk_modes", None)
            or getattr(args, "stack_modes", None)
            or ("normalized_absorbance", "transmittance")
        ),
        region_labels=str(getattr(args, "region_labels", "selected")),
    )
    out_json: dict[str, Any] = {"report": str(rp), "report_audience": audience}
    if getattr(args, "export_paper_figures", False):
        paper_dir = (
            _resolve_under_chunks(Path(args.paper_out))
            if getattr(args, "paper_out", "")
            else out.parent / "presentation" / "paper_figures"
        )
        out_json["paper_figures_dir"] = str(paper_dir.resolve())
        out_json["paper_figures_index"] = str((paper_dir / "PAPER_FIGURES_INDEX.md").resolve())
        ms = out.parent / "MANUSCRIPT_REPORT.html"
        if ms.is_file():
            out_json["manuscript_report"] = str(ms.resolve())
    if getattr(args, "export_static_figures", False):
        static_dir = (
            _resolve_under_chunks(Path(args.static_out))
            if getattr(args, "static_out", "")
            else out.parent / "presentation" / "figures"
        )
        out_json["static_figures_dir"] = str(static_dir.resolve())
        out_json["presentation_index"] = str((out.parent / "presentation" / "FIGURES_INDEX.md").resolve())
    if (out.parent / "INTERPRETATION_NOTES.txt").is_file():
        out_json["interpretation_notes_template"] = str(
            (out.parent / "INTERPRETATION_NOTES.txt").resolve()
        )
    if getattr(args, "visual_theme", "default") == "matlab" or getattr(
        args, "export_static_figures", False
    ):
        out_json["matlab_export_dir"] = str(out.parent / "matlab_export")
    if export_csv_dir is not None:
        out_json["export_csv_dir"] = str(export_csv_dir)
    if preset:
        out_json["rules_preset"] = preset
    print(json.dumps(out_json, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interactive structural FG SVM report with stacked Kronecker panel"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_b = sub.add_parser("batch", help="Build one interactive HTML report")
    p_b.add_argument("--inputs", nargs="+", default=None, metavar="PATH")
    p_b.add_argument("--input", nargs="+", dest="input_alt", default=None, metavar="PATH")
    p_b.add_argument("--model", default="", help="Legacy SVM .joblib (optional secondary refinement)")
    p_b.add_argument("--basic-model", default="", help="Broad-label basic SVM .joblib")
    p_b.add_argument("--subtle-model", default="", help="Subtle-label SVM .joblib")
    p_b.add_argument(
        "--family-model",
        default="",
        help="v4 family ontology SVM .joblib (alias for --basic-model)",
    )
    p_b.add_argument(
        "--specific-model",
        default="",
        help="v4 specific FG SVM .joblib (alias for --subtle-model)",
    )
    p_b.add_argument(
        "--ml-mode",
        choices=("auto", "none", "basic", "subtle", "both", "legacy"),
        default="auto",
        help="auto=infer from models; none=evidence only (no ML in hover); legacy=use legacy bundle when present",
    )
    p_b.add_argument(
        "--fusion-mode",
        choices=("annotate", "weighted", "gate", "ml_only"),
        default="annotate",
    )
    p_b.add_argument(
        "--ontology",
        choices=("v3", "v4"),
        default="",
        help="Rule + evidence ontology (v4: v4 buckets, v4-basic ML heads, richer local hover). Empty = rules default.",
    )
    p_b.add_argument("--include-evidence", action=argparse.BooleanOptionalAction, default=True)
    p_b.add_argument("--include-ml", action=argparse.BooleanOptionalAction, default=True)
    p_b.add_argument("--include-consensus", action=argparse.BooleanOptionalAction, default=True)
    p_b.add_argument("--include-interpretability", action="store_true", help="Deprecated: use --include-evidence")
    p_b.add_argument("--include-robustness", action="store_true")
    p_b.add_argument(
        "--rules-preset",
        default="",
        help="Optional: conservative | sensitive | phenol_alcohol_strict (see ml/calibrate_rules.py list-presets)",
    )
    p_b.add_argument("--rules-config", default="", help="Optional JSON file merging into rule thresholds")
    p_b.add_argument(
        "--export-csv",
        default="",
        help="Optional directory for consensus/rules CSV export (in addition to HTML)",
    )
    p_b.add_argument(
        "--out",
        "--out-dir",
        "--output",
        dest="out",
        required=True,
        help="Output HTML path or directory (directory → REPORT.html inside)",
    )
    p_b.add_argument("--title", default="Structural FG SVM — interactive Kronecker report")
    p_b.add_argument("--subtitle", default="")
    p_b.add_argument("--max-peaks", type=int, default=80)
    p_b.add_argument(
        "--peak-sensitivity",
        choices=("conservative", "balanced", "sensitive", "very_sensitive"),
        default="balanced",
        help="Peak-picking preset: conservative (fewer) → very_sensitive (audit-style, capped display)",
    )
    p_b.add_argument(
        "--show-weak-peaks",
        action="store_true",
        help="Show weak-quality peaks on spectrum (smaller/muted markers); rules still require diagnostic evidence",
    )
    p_b.add_argument(
        "--max-peak-labels",
        type=int,
        default=24,
        help="Max numeric peak labels on spectrum (diamond markers)",
    )
    p_b.add_argument(
        "--label-all-diagnostic-peaks",
        action="store_true",
        help="Label all diagnostic peaks (up to --max-peak-labels)",
    )
    p_b.add_argument(
        "--label-all-above-height",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Label every picked peak with normalized absorbance >= FLOAT (e.g. 0.1); ignores label cap",
    )
    p_b.add_argument(
        "--show-region-ruler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="FTIR tentative-range ruler above spectrum (default on for product_v1)",
    )
    p_b.add_argument(
        "--peak-min-height",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Normalized absorbance floor for peaks (default: preset from --peak-sensitivity)",
    )
    p_b.add_argument(
        "--peak-min-prominence",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Normalized prominence floor (default: preset from --peak-sensitivity)",
    )
    p_b.add_argument(
        "--peak-label-preset",
        choices=("conservative", "balanced", "sensitive", "all-visible"),
        default=None,
        help="Label height preset (default: label thresholds follow --peak-min-height unless set)",
    )
    p_b.add_argument(
        "--peak-label-min-height",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Min normalized height for peak labels (default: same as --peak-min-height)",
    )
    p_b.add_argument(
        "--peak-label-min-prominence",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Min normalized prominence for peak labels (default: same as --peak-min-prominence)",
    )
    p_b.add_argument(
        "--presentation-mode",
        action="store_true",
        help="Slides/screenshots: fewer labels, larger fonts, thicker traces, extra spacing",
    )
    p_b.add_argument(
        "--peak-label-layout",
        choices=("smart", "simple"),
        default="smart",
        help="Peak frequency label placement: smart (collision-aware) or simple",
    )
    p_b.add_argument(
        "--auto-layout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dynamic figure height/margins from label density (default on for --report-audience front)",
    )
    p_b.add_argument(
        "--fingerprint-cluster-distance",
        type=float,
        default=18.0,
        metavar="CM1",
        help="In fingerprint region, label strongest peak per cluster (cm⁻¹); 0 disables",
    )
    p_b.add_argument(
        "--static-peak-label-policy",
        choices=("key", "top", "all"),
        default="key",
        help="Static PNG/SVG: which labeled peaks to draw (interactive HTML unchanged)",
    )
    p_b.add_argument(
        "--max-static-peak-labels",
        type=int,
        default=12,
        help="Max peak labels on static export when policy is key or top",
    )
    p_b.add_argument(
        "--shade-min-activity",
        type=float,
        default=0.10,
        help="Strong region-activity shading threshold (normalized absorbance)",
    )
    p_b.add_argument(
        "--shade-faint-min",
        type=float,
        default=0.05,
        help="Faint region-activity shading threshold",
    )
    p_b.add_argument(
        "--shade-sensitive",
        action="store_true",
        help="Lower faint shading to 0.03 (weak C–H / upper-mid activity)",
    )
    p_b.add_argument(
        "--hover-top-fg",
        type=int,
        default=8,
        help="Max local assignment lines / deduped FG rows in spectrum hover",
    )
    p_b.add_argument(
        "--show-band-shading",
        action="store_true",
        help="Shade non-overlapping traditional FTIR windows (O–H, C=O, C≡C/C≡N, fingerprint, etc.) when activity is present",
    )
    p_b.add_argument(
        "--label-band-shading",
        action="store_true",
        help="Label shaded regions on the spectrum (product_v1; use with --show-band-shading)",
    )
    p_b.add_argument(
        "--report-style",
        choices=("legacy", "product_v1"),
        default="product_v1",
        help="legacy = table-heavy layout; product_v1 = spectrum-centric polished interpretation (default)",
    )
    p_b.add_argument(
        "--report-audience",
        choices=("front", "debug"),
        default=None,
        help="front = polished spectroscopist report; debug = full metadata/diagnostics (default: front for product_v1, debug for audit)",
    )
    p_b.add_argument(
        "--front-facing",
        action="store_true",
        help="Alias for --report-audience front",
    )
    p_b.add_argument(
        "--front-max-peak-labels",
        type=int,
        default=30,
        help="Max labeled peaks on spectrum in front-facing mode (default 30)",
    )
    p_b.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show metadata block in front-facing mode (default hidden)",
    )
    p_b.add_argument(
        "--guardrails",
        choices=("none", "v2", "v3"),
        default="v3",
        help="none=no extra confidence classes; v2=annotate classes only; v3=v3_guarded soft FP control",
    )
    p_b.add_argument(
        "--ml-guardrails",
        choices=("strict", "moderate", "off"),
        default="strict",
        help="How ML may combine with weak spectral evidence (strict default for production)",
    )
    p_b.add_argument(
        "--show-ambiguity-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show family fallback labels when guardrails=v3 (default: on)",
    )
    p_b.add_argument(
        "--show-artifact-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show artifact / interference block when guardrails=v3 (default: on)",
    )
    p_b.add_argument(
        "--report-density",
        choices=("summary", "balanced", "audit"),
        default="balanced",
        help="summary = very compact; balanced = default; audit = full scientific detail",
    )
    p_b.add_argument(
        "--top-n-summary",
        type=int,
        default=5,
        metavar="N",
        help="Max evidence rows shown before 'show all' expand (balanced/summary)",
    )
    p_b.add_argument(
        "--anonymize-metadata",
        action="store_true",
        help="Shareable/publishable HTML+CSV: strip machine paths (model paths → .joblib basename; CSV path column → spectrum filename only).",
    )
    p_b.add_argument(
        "--no-summary-table",
        action="store_true",
        help="Omit the batch-level summary table at the top of the report (per-spectrum sections only).",
    )
    p_b.add_argument(
        "--mode",
        choices=("ATR", "transmission", "unknown", ""),
        default="",
        help="Measurement mode (empty = infer from path/metadata).",
    )
    p_b.add_argument(
        "--atr-crystal",
        choices=("diamond", "ZnSe", "Ge", "Si", "unknown", ""),
        default="",
        help="ATR crystal type when mode=ATR (empty = infer).",
    )
    p_b.add_argument(
        "--atr-aware",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply ATR Si-O/siloxane guardrails (default: on when mode=ATR or inferred ATR).",
    )
    p_b.add_argument(
        "--visual-theme",
        choices=("default", "matlab", "dark"),
        default="default",
        help="Plotly/HTML styling: default (product), matlab (publication blue/white), dark",
    )
    p_b.add_argument(
        "--show-deconvolution",
        action="store_true",
        help="Advisory deconvolution overlay + candidate peak table (does not change consensus)",
    )
    p_b.add_argument(
        "--deconv-max-components-per-region",
        type=int,
        default=6,
        help="Max fitted components per interpretable window (with --show-deconvolution)",
    )
    p_b.add_argument(
        "--deconv-min-component-height",
        type=float,
        default=0.03,
        help="Min normalized component height to display (with --show-deconvolution)",
    )
    p_b.add_argument(
        "--deconv-model",
        choices=("lorentzian", "gaussian", "pseudo_voigt"),
        default="pseudo_voigt",
        help="Peak profile for regional deconvolution fits",
    )
    p_b.add_argument(
        "--deconv-regions",
        choices=("auto", "all", "fingerprint", "carbonyl", "oh_nh", "ch", "custom"),
        default="auto",
        help="Which interpretable windows to deconvolve",
    )
    p_b.add_argument(
        "--editable-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Front reports: click-to-edit summary (browser localStorage). Default on for --report-audience front",
    )
    p_b.add_argument(
        "--interpretation-notes",
        default="",
        metavar="PATH",
        help="Optional text file to seed/replace the spectroscopist summary (## spectrum_stem sections)",
    )
    p_b.add_argument(
        "--suppress-nitro-reporting",
        action="store_true",
        help="Omit nitro / NO₂ labels, band matches, and prose (samples known to lack R–NO₂)",
    )
    p_b.add_argument(
        "--export-static-figures",
        action="store_true",
        help="Export per-spectrum static figures (PNG/SVG/PDF) for presentations",
    )
    p_b.add_argument(
        "--static-format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Static figure format when --export-static-figures (requires kaleido)",
    )
    p_b.add_argument(
        "--static-dpi",
        type=int,
        default=300,
        help="Static export resolution scale (96 DPI baseline; 300 ≈ print quality)",
    )
    p_b.add_argument(
        "--static-out",
        default="",
        help="Directory for static figures (default: <report-dir>/figures)",
    )
    p_b.add_argument(
        "--export-paper-figures",
        action="store_true",
        help="Export manuscript-ready transmittance + normalized absorbance figures (PNG/SVG/PDF) with horizontal leader-line peak labels",
    )
    p_b.add_argument(
        "--paper-figure-modes",
        nargs="+",
        choices=("transmittance", "normalized_absorbance"),
        default=["transmittance", "normalized_absorbance"],
        help="Which paper figure modes to export (default: both)",
    )
    p_b.add_argument(
        "--paper-label-style",
        choices=("horizontal-leader",),
        default="horizontal-leader",
        help="Peak label layout for paper figures",
    )
    p_b.add_argument(
        "--paper-report-style",
        choices=("both", "product", "full", "manuscript"),
        default="both",
        help="Which HTML reports include paper figures: product/full REPORT.html, MANUSCRIPT_REPORT.html, or both",
    )
    p_b.add_argument(
        "--paper-formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png", "svg", "pdf"],
        help="Vector/raster formats for paper figure export",
    )
    p_b.add_argument(
        "--max-paper-peak-labels",
        type=int,
        default=10,
        help="Max horizontal peak labels per paper figure (8–12 recommended)",
    )
    p_b.add_argument(
        "--paper-out",
        default="",
        help="Directory for paper figures (default: <report-dir>/presentation/paper_figures)",
    )
    p_b.add_argument(
        "--min-peak-prominence",
        type=float,
        default=0.04,
        help="Minimum prominence for paper figure candidate peaks (normalized absorbance)",
    )
    p_b.add_argument(
        "--min-peak-height",
        type=float,
        default=0.05,
        help="Minimum height for paper figure candidate peaks (normalized absorbance)",
    )
    p_b.add_argument(
        "--min-peak-distance-cm1",
        type=float,
        default=20.0,
        help="Minimum peak spacing in cm⁻¹ for candidate detection",
    )
    p_b.add_argument(
        "--ignore-label-ranges",
        nargs="+",
        default=["900:400"],
        help="Wavenumber ranges excluded from automatic labeling (default: 900:400)",
    )
    p_b.add_argument(
        "--use-shoulder-detection",
        action="store_true",
        help="Flag broad shoulder-like peaks during candidate detection",
    )
    p_b.add_argument(
        "--override-file",
        default="",
        help="Optional YAML/CSV peak override file (required_peaks, suppress_ranges, etc.)",
    )
    p_b.add_argument(
        "--export-spectrum-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-spectrum feedback TXT/MD and embed in HTML reports",
    )
    p_b.add_argument(
        "--export-interactive-curation",
        action="store_true",
        help="Embed interactive Plotly curation figures with label adjustment UI in REPORT.html",
    )
    p_b.add_argument(
        "--export-region-stacks",
        action="store_true",
        help="Export spectral chunks: singles, offset stacks, collages, and chunk data under stacks/",
    )
    p_b.add_argument(
        "--export-chunk-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-range chunk_data JSON/CSV when --export-region-stacks is used (default: on)",
    )
    p_b.add_argument(
        "--chunk-collage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export multi-range collage figures when --export-region-stacks is used",
    )
    p_b.add_argument(
        "--chunk-modes",
        nargs="+",
        choices=("normalized_absorbance", "transmittance"),
        default=None,
        help="Chunk figure intensity modes (defaults to --stack-modes)",
    )
    p_b.add_argument(
        "--regions-file",
        "--ranges-file",
        dest="regions_file",
        default="",
        help="JSON/YAML ranges_config.json defining custom wavenumber chunks",
    )
    p_b.add_argument(
        "--label-overrides",
        default="",
        help="Directory for per-spectrum {stem}_label_overrides.json files",
    )
    p_b.add_argument(
        "--save-label-overrides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write auto label override JSON templates when missing",
    )
    p_b.add_argument(
        "--apply-label-overrides",
        action="store_true",
        help="Apply saved label overrides when exporting paper/manuscript figures",
    )
    p_b.add_argument(
        "--allow-apparent-transmittance",
        action="store_true",
        help="Allow T_app = 100·10^(−A) figures from absorbance-like data (clearly labeled, not native %T)",
    )
    p_b.add_argument(
        "--force-intensity-mode",
        choices=("transmittance_percent", "absorbance", "absorbance_difference"),
        default=None,
        help="Override automatic intensity classification for paper/curation/chunk export",
    )
    p_b.add_argument(
        "--apparent-transmittance-label",
        default="Apparent Transmittance (%)",
        help="Y-axis label when exporting apparent transmittance from absorbance",
    )
    p_b.add_argument(
        "--show-peak-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw orange peak-tip markers on static manuscript/chunk figures (default: off; interactive curation uses optional checkbox)",
    )
    p_b.add_argument(
        "--interactive-png-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Plotly modebar PNG export on interactive curation figures",
    )
    p_b.add_argument(
        "--offset-gap",
        type=float,
        default=0.15,
        help="Fractional gap between traces in region offset stacks (default 0.15)",
    )
    p_b.add_argument(
        "--stack-modes",
        nargs="+",
        choices=("normalized_absorbance", "transmittance"),
        default=["normalized_absorbance", "transmittance"],
        help="Stack figure intensity modes per discussion region",
    )
    p_b.add_argument(
        "--region-labels",
        choices=("selected", "none"),
        default="selected",
        help="Peak label policy on region offset stacks",
    )
    p_b.set_defaults(func=cmd_batch)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
