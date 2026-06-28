"""
Interactive Plotly FTIR peak curation for HTML reports.

Embeds self-contained plot divs + JSON data blocks; a single page-level script
initializes Plotly after DOM ready and after Plotly.js is loaded from the main
spectrum figure.
"""

from __future__ import annotations

import html
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from lib.ftir_foundation import preprocess_spectrum, read_spectrum
from reports.label_overrides import (
    build_auto_curation_labels,
    build_auto_override_payload,
    label_record,
    load_label_overrides,
    merge_overrides_with_auto,
    overrides_path,
    save_label_overrides,
)
from reports.peak_snap import DEFAULT_SNAP_WINDOW_CM1
from lib.intensity_modes import (
    ForceIntensityMode,
    TransmittancePanelPlan,
    classify_intensity,
    has_transmittance_panel,
    plan_transmittance_panel,
)
from reports.paper_ftir_figures import (
    _percent_transmittance_from_plan,
    _smooth_percent_t,
    _window_mask,
    apparent_transmittance_baseline_pct,
    prepare_transmittance_trace,
    transmittance_y_limits,
)
from reports.paper_peak_selection import (
    PaperPeakSelectionConfig,
    match_transmittance_minima,
    select_labeled_peaks,
)

PEAK_COLOR = "#d95319"
SPECTRUM_COLOR = "#0072bd"
CANDIDATE_COLOR = "#f0a070"

CURATION_CSS = """
.curation-section { margin: 20px 0; padding: 14px 16px; border: 1px solid #dbeafe; border-radius: 10px; background: #f8fafc; }
.curation-section h3 { margin: 0 0 8px; font-size: 1.05rem; }
.curation-section h4 { margin: 16px 0 6px; font-size: 0.98rem; }
.curation-hint { font-size: 0.85rem; color: #475569; margin: 0 0 12px; }
.curation-plot { height: 420px; min-height: 420px; width: 100%; margin: 8px 0 12px; }
.curation-error { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; padding: 10px 12px; border-radius: 8px; margin: 8px 0; font-size: 0.88rem; }
.curation-controls { overflow-x: auto; margin: 10px 0; }
.curation-controls table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
.curation-controls th, .curation-controls td { border: 1px solid #cbd5e1; padding: 4px 6px; text-align: left; vertical-align: middle; }
.curation-controls th { background: #e2e8f0; position: sticky; top: 0; }
.curation-controls input[type=number] { width: 48px; }
.curation-controls input[type=text].lbl-text { width: 56px; }
.curation-btn { font-size: 0.76rem; padding: 2px 6px; margin: 0 1px; cursor: pointer; }
.curation-toolbar { margin: 10px 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.curation-toolbar input.manual-wn-input { width: 72px; }
.curation-toolbar select.manual-mode-select { font-size: 0.8rem; }
.curation-click-add-active { outline: 2px solid #2563eb; outline-offset: 2px; }
.curation-row-selected { background: #eff6ff; }
.curation-warn.intensity-warn { font-size: 0.85rem; color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; padding: 10px 12px; border-radius: 8px; margin: 10px 0; }
"""


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(name).stem)


def _yaxis_with_label_padding(
    y: np.ndarray,
    *,
    mode: str,
    padding_frac: float = 0.16,
    baseline_pct: float | None = None,
) -> tuple[float, float]:
    if mode == "transmittance" and baseline_pct is not None:
        return transmittance_y_limits(y, baseline_pct=baseline_pct, padding_frac=padding_frac)
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0, 1.0
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    span = max(y_max - y_min, 1e-6)
    pad = span * padding_frac
    if mode == "transmittance":
        return y_min - pad * 1.5, y_max + pad * 0.35
    return y_min - pad * 0.2, y_max + pad * 1.3


def _px_to_data_y(yshift_px: float, y_span: float) -> float:
    return (float(yshift_px) / 280.0) * max(y_span, 1e-9)


def _label_annotation(
    lab: dict[str, Any],
    *,
    y_span: float,
    label_side: str,
) -> dict[str, Any]:
    wn = float(lab["wavenumber_cm1"])
    peak_y = float(lab.get("peak_y", 0))
    direction = 1.0 if label_side == "above" else -1.0
    base_offset = 0.08 * y_span * direction
    y_off = _px_to_data_y(float(lab.get("yshift_px", 12)), y_span) * direction + base_offset
    label_x = float(lab["label_x_cm1"]) if lab.get("label_x_cm1") is not None else wn
    label_y = float(lab["label_y_value"]) if lab.get("label_y_value") is not None else peak_y + y_off
    label_x += _px_to_data_y(float(lab.get("xshift_px", 0)), y_span) * 0.02
    return dict(
        x=label_x,
        y=label_y,
        xref="x",
        yref="y",
        text=str(lab.get("label_text", f"{wn:.0f}")),
        showarrow=True,
        arrowhead=0,
        arrowwidth=0.9,
        arrowcolor=PEAK_COLOR,
        ax=0,
        ay=peak_y - label_y,
        textangle=0,
        font=dict(size=10, color="black"),
        bgcolor="rgba(255,255,255,0.9)",
        borderpad=2,
        captureevents=True,
    )


def _build_figure_bundle(
    wn: np.ndarray,
    y: np.ndarray,
    labels: list[dict[str, Any]],
    *,
    mode: str,
    title: str,
    wn_min: float,
    wn_max: float,
    transmittance_baseline_pct: float | None = None,
) -> dict[str, Any]:
    label_side = "above" if mode == "normalized_absorbance" else "below"
    ylo, yhi = _yaxis_with_label_padding(
        y,
        mode=mode,
        baseline_pct=transmittance_baseline_pct if mode == "transmittance" else None,
    )
    y_span = yhi - ylo
    mode_labels = [l for l in labels if str(l.get("mode", "")) == mode]

    candidate_x: list[float] = []
    candidate_y: list[float] = []
    customdata: list[list[Any]] = []
    for lab in mode_labels:
        gidx = labels.index(lab)
        wn_v = float(lab["wavenumber_cm1"])
        py = float(lab.get("peak_y", 0))
        candidate_x.append(wn_v)
        candidate_y.append(py)
        customdata.append(
            [
                gidx,
                wn_v,
                py,
                float(lab.get("prominence", 0)),
                str(lab.get("region", "")),
                float(lab.get("score", 0)),
                bool(lab.get("show_label", False)),
            ]
        )

    annotations = [
        _label_annotation(lab, y_span=y_span, label_side=label_side)
        for lab in mode_labels
        if lab.get("show_label", True)
    ]

    ylabel = "Normalized Absorbance (0–1)" if mode == "normalized_absorbance" else "Transmittance (%T)"
    data = [
        go.Scatter(
            x=wn,
            y=y,
            mode="lines",
            name="spectrum",
            line=dict(color=SPECTRUM_COLOR, width=1.2),
            hovertemplate="%{x:.1f} cm⁻¹<br>%{y:.4f}<extra></extra>",
        ),
        go.Scatter(
            x=candidate_x,
            y=candidate_y,
            mode="markers",
            name="candidate_peaks",
            marker=dict(
                color="rgba(240,160,112,0.55)",
                size=7,
                line=dict(width=0.8, color="rgba(217,83,25,0.65)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "ν=%{customdata[1]:.1f} cm⁻¹<br>"
                "I=%{customdata[2]:.4f}<br>"
                "prom=%{customdata[3]:.4f}<br>"
                "region=%{customdata[4]}<br>"
                "score=%{customdata[5]:.3f}<br>"
                "selected=%{customdata[6]}<extra>click to toggle label</extra>"
            ),
        ),
    ]
    layout = go.Layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Wavenumber (cm⁻¹)", range=[wn_max, wn_min], autorange=False, gridcolor="#e6e6e6"),
        yaxis=dict(title=ylabel, range=[ylo, yhi], autorange=False, gridcolor="#e6e6e6"),
        margin=dict(l=56, r=28, t=48, b=48),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=annotations,
        dragmode="zoom",
    )
    config = {
        "displayModeBar": True,
        "responsive": True,
        "scrollZoom": True,
        "editable": True,
        "edits": {"annotationPosition": True, "annotationText": True},
        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": title.replace(" ", "_"),
            "height": 800,
            "width": 1200,
            "scale": 2,
        },
    }
    fig = go.Figure(data=data, layout=layout)
    return {"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"], "config": config}


def export_interactive_curation_for_spectrum(
    input_path: Path,
    *,
    report_dir: Path,
    overrides_dir: Path | None = None,
    selection_config: PaperPeakSelectionConfig | None = None,
    apply_saved_overrides: bool = True,
    save_auto_overrides: bool = True,
    wn_min: float = 400.0,
    wn_max: float = 4000.0,
    allow_apparent_transmittance: bool = False,
    force_intensity_mode: ForceIntensityMode | None = None,
    apparent_transmittance_label: str = "Apparent Transmittance (%)",
) -> dict[str, Any]:
    input_path = Path(input_path)
    report_dir = Path(report_dir)
    stem = _safe_stem(input_path.name)
    ov_dir = Path(overrides_dir) if overrides_dir else report_dir
    ov_path = overrides_path(ov_dir, stem)

    wn, raw, hint = read_spectrum(input_path)
    classification = classify_intensity(
        input_path, hint, raw, force_mode=force_intensity_mode
    )
    int_mode = classification.preprocess_mode
    t_plan = plan_transmittance_panel(
        raw,
        classification,
        allow_apparent=allow_apparent_transmittance,
        apparent_label=apparent_transmittance_label,
    )
    wn_p, y_norm, _info = preprocess_spectrum(wn, raw, intensity_mode=int_mode, normalize=True)
    mask_a = _window_mask(wn_p, wn_min=wn_min, wn_max=wn_max)

    sel_cfg = selection_config or PaperPeakSelectionConfig()
    selection = select_labeled_peaks(wn_p[mask_a], y_norm[mask_a], config=sel_cfg)

    t_minima: list[dict[str, Any]] = []
    wn_t = np.array([])
    y_t = np.array([])
    t_baseline_pct: float | None = None
    if has_transmittance_panel(t_plan):
        y_pct = _percent_transmittance_from_plan(t_plan.y_values)  # type: ignore[arg-type]
        mask_t = _window_mask(wn, wn_min=wn_min, wn_max=wn_max)
        t_baseline_pct = apparent_transmittance_baseline_pct(t_plan)
        y_t = prepare_transmittance_trace(
            _smooth_percent_t(wn[mask_t], y_pct[mask_t]),
            baseline_pct=t_baseline_pct,
        )
        wn_t = wn[mask_t]
        t_minima = match_transmittance_minima(
            wn_t,
            y_t,
            selection.selected,
            match_window_cm1=sel_cfg.transmittance_match_cm1,
            baseline_pct=t_baseline_pct,
        )

    auto_labels = build_auto_curation_labels(selection, t_minima)
    saved = load_label_overrides(ov_path) if apply_saved_overrides else {}
    merged = merge_overrides_with_auto(auto_labels, saved)

    if save_auto_overrides and not ov_path.is_file():
        save_label_overrides(ov_path, build_auto_override_payload(stem=stem, labels=auto_labels))

    wn_abs = wn_p[mask_a]
    y_abs = y_norm[mask_a]
    fig_abs = _build_figure_bundle(
        wn_abs,
        y_abs,
        merged,
        mode="normalized_absorbance",
        title=f"{stem} — normalized absorbance",
        wn_min=wn_min,
        wn_max=wn_max,
    )
    fig_t: dict[str, Any] | None = None
    if has_transmittance_panel(t_plan) and wn_t.size:
        t_title = f"{stem} — {t_plan.ylabel}"
        fig_t = _build_figure_bundle(
            wn_t,
            y_t,
            merged,
            mode="transmittance",
            title=t_title,
            wn_min=wn_min,
            wn_max=wn_max,
            transmittance_baseline_pct=t_baseline_pct,
        )

    curation_id = f"curation-{_safe_stem(stem)}"
    html_section = build_curation_section_html(
        stem=stem,
        curation_id=curation_id,
        fig_abs=fig_abs,
        fig_t=fig_t,
        labels=merged,
        overrides_path=ov_path,
        transmittance_plan=t_plan,
        abs_wn=wn_abs,
        abs_y=y_abs,
        t_wn=wn_t,
        t_y=y_t,
        snap_window_cm1=float(DEFAULT_SNAP_WINDOW_CM1),
        transmittance_baseline_pct=t_baseline_pct,
    )

    return {
        "stem": stem,
        "html": html_section,
        "overrides_path": str(ov_path.resolve()),
        "labels": merged,
        "transmittance_valid": has_transmittance_panel(t_plan),
        "intensity_category": classification.category,
        "transmittance_is_apparent": bool(t_plan.is_apparent and has_transmittance_panel(t_plan)),
        "transmittance_skip_reason": t_plan.skip_reason,
    }


def build_curation_section_html(
    *,
    stem: str,
    curation_id: str,
    fig_abs: dict[str, Any],
    fig_t: dict[str, Any] | None,
    labels: list[dict[str, Any]],
    overrides_path: Path,
    transmittance_plan: TransmittancePanelPlan,
    abs_wn: np.ndarray | None = None,
    abs_y: np.ndarray | None = None,
    t_wn: np.ndarray | None = None,
    t_y: np.ndarray | None = None,
    snap_window_cm1: float = DEFAULT_SNAP_WINDOW_CM1,
    transmittance_baseline_pct: float | None = None,
) -> str:
    abs_plot_id = f"{curation_id}-abs-{uuid.uuid4().hex[:8]}"
    t_plot_id = f"{curation_id}-t-{uuid.uuid4().hex[:8]}"
    abs_err_id = f"{abs_plot_id}-err"
    t_err_id = f"{t_plot_id}-err"

    parts = [
        f"<section class='curation-section card ftir-curation-root' id='{html.escape(curation_id)}' "
        f"data-stem='{html.escape(stem)}'>",
        f"<h3>Interactive Peak Curation — {html.escape(stem.replace('_', ' '))}</h3>",
        "<p class='curation-hint'>Click orange candidate markers to toggle labels, or enable "
        "<b>Add manual peak from click</b> and click the spectrum trace. Typed wavenumbers snap to the "
        "nearest local maximum (absorbance) or minimum (transmittance).</p>",
        "<h4>Normalized absorbance</h4>",
        f"<div id='{html.escape(abs_err_id)}' class='curation-error' style='display:none'></div>",
        f"<div id='{html.escape(abs_plot_id)}' class='curation-plot plot-wrap' "
        f"data-curation-plot='1' data-mode='normalized_absorbance'></div>",
        f"<script type='application/json' id='{html.escape(abs_plot_id)}-fig'>{json.dumps(fig_abs)}</script>",
        build_label_table_html(curation_id, labels, mode="normalized_absorbance"),
    ]
    if transmittance_plan.banner_html and not has_transmittance_panel(transmittance_plan):
        parts.append(transmittance_plan.banner_html)
    if fig_t is not None and has_transmittance_panel(transmittance_plan):
        if transmittance_plan.warning:
            parts.append(
                f"<p class='curation-warn intensity-warn'>{html.escape(transmittance_plan.warning)}</p>"
            )
        parts.extend(
            [
                f"<h4>{html.escape(transmittance_plan.ylabel)}</h4>",
                f"<div id='{html.escape(t_err_id)}' class='curation-error' style='display:none'></div>",
                f"<div id='{html.escape(t_plot_id)}' class='curation-plot plot-wrap' "
                f"data-curation-plot='1' data-mode='transmittance'></div>",
                f"<script type='application/json' id='{html.escape(t_plot_id)}-fig'>{json.dumps(fig_t)}</script>",
                build_label_table_html(curation_id, labels, mode="transmittance"),
            ]
        )

    meta_payload = {
        "stem": stem,
        "overrides_name": overrides_path.name,
        "abs_plot_id": abs_plot_id,
        "t_plot_id": t_plot_id if fig_t else "",
        "snap_window_cm1": float(snap_window_cm1),
        "abs_wn": np.asarray(abs_wn, dtype=float).tolist() if abs_wn is not None else [],
        "abs_y": np.asarray(abs_y, dtype=float).tolist() if abs_y is not None else [],
        "t_wn": np.asarray(t_wn, dtype=float).tolist() if t_wn is not None and np.asarray(t_wn).size else [],
        "t_y": np.asarray(t_y, dtype=float).tolist() if t_y is not None and np.asarray(t_y).size else [],
        "transmittance_baseline_pct": transmittance_baseline_pct,
    }

    parts.extend(
        [
            "<div class='curation-toolbar'>",
            f"<button type='button' class='curation-btn' data-curation='{html.escape(curation_id)}' "
            f"data-action='toggle-click-add'>Add manual peak from click</button>",
            "<label>Wavenumber <input type='number' class='manual-wn-input' step='1' placeholder='1605'/></label>",
            "<select class='manual-mode-select'>"
            "<option value='normalized_absorbance'>Absorbance</option>"
            "<option value='transmittance'>Transmittance</option>"
            "</select>",
            f"<button type='button' class='curation-btn' data-curation='{html.escape(curation_id)}' "
            f"data-action='add-typed-wn'>Add peak</button>",
            f"<button type='button' class='curation-btn' data-curation='{html.escape(curation_id)}' "
            f"data-action='download-overrides'>Download label overrides JSON</button>",
            f"<button type='button' class='curation-btn' data-curation='{html.escape(curation_id)}' "
            f"data-action='reset-layout'>Reset to auto layout</button>",
            f"<label class='curation-hint'><input type='checkbox' class='show-candidate-markers' "
            f"data-curation='{html.escape(curation_id)}' checked/> Show candidate markers</label>",
            "</div>",
            f"<script type='application/json' id='{html.escape(curation_id)}-labels'>{json.dumps(labels)}</script>",
            f"<script type='application/json' id='{html.escape(curation_id)}-meta'>"
            f"{json.dumps(meta_payload)}"
            f"</script>",
            "</section>",
        ]
    )
    return "".join(parts)


def build_label_table_html(curation_id: str, labels: list[dict[str, Any]], *, mode: str) -> str:
    mode_labels = [(i, lab) for i, lab in enumerate(labels) if str(lab.get("mode", "")) == mode]
    rows: list[str] = []
    for idx, lab in mode_labels:
        wn = float(lab.get("wavenumber_cm1", 0))
        show = "checked" if lab.get("show_label", True) else ""
        rows.append(
            f"<tr data-curation='{html.escape(curation_id)}' data-label-idx='{idx}' "
            f"data-mode='{html.escape(mode)}' data-wn='{wn:.2f}'>"
            f"<td><input type='checkbox' class='lbl-show' {show}/></td>"
            f"<td>{wn:.1f}</td>"
            f"<td>{html.escape(str(lab.get('region', '')))}</td>"
            f"<td>{float(lab.get('score', 0)):.3f}</td>"
            f"<td>{float(lab.get('prominence', 0)):.3f}</td>"
            f"<td><input type='text' class='lbl-text' value='{html.escape(str(lab.get('label_text', '')))}'/></td>"
            f"<td><button type='button' class='curation-btn lbl-x-minus'>←</button>"
            f"<input type='number' class='lbl-xshift' value='{float(lab.get('xshift_px', 0)):.0f}' step='1'/>"
            f"<button type='button' class='curation-btn lbl-x-plus'>→</button></td>"
            f"<td><button type='button' class='curation-btn lbl-y-minus'>↓</button>"
            f"<input type='number' class='lbl-yshift' value='{float(lab.get('yshift_px', 12)):.0f}' step='1'/>"
            f"<button type='button' class='curation-btn lbl-y-plus'>↑</button></td>"
            f"<td>"
            f"<button type='button' class='curation-btn lbl-reset'>Reset</button> "
            f"<button type='button' class='curation-btn lbl-delete'>Del</button>"
            f"</td></tr>"
        )
    if not rows:
        rows.append(f"<tr><td colspan='9' class='muted'>No candidate peaks for {html.escape(mode)}.</td></tr>")
    return (
        f"<div class='curation-controls' data-table-mode='{html.escape(mode)}'>"
        f"<table><thead><tr><th>Show</th><th>ν (cm⁻¹)</th><th>Region</th><th>Score</th>"
        f"<th>Prom</th><th>Text</th><th>X (px)</th><th>Y (px)</th><th>Actions</th></tr></thead>"
        f"<tbody id='{html.escape(curation_id)}-tbody-{html.escape(mode)}'>{''.join(rows)}</tbody></table></div>"
    )


CURATION_JS = r"""
<script>
(function () {
  var stateByRoot = {};

  function warn(msg) { if (typeof console !== "undefined" && console.warn) console.warn("[FTIR curation]", msg); }

  function waitForPlotly(maxMs) {
    return new Promise(function (resolve, reject) {
      var t0 = Date.now();
      (function poll() {
        if (window.Plotly && window.Plotly.newPlot) return resolve(window.Plotly);
        if (Date.now() - t0 > maxMs) return reject(new Error("Plotly.js not loaded after " + maxMs + "ms"));
        setTimeout(poll, 50);
      })();
    });
  }

  function showError(errId, msg) {
    var el = document.getElementById(errId);
    if (!el) { warn(msg); return; }
    el.style.display = "block";
    el.textContent = msg;
  }

  function parseJsonId(id) {
    var el = document.getElementById(id);
    if (!el) { warn("Missing JSON block: " + id); return null; }
    try { return JSON.parse(el.textContent || "null"); } catch (e) { warn("Bad JSON in " + id + ": " + e); return null; }
  }

  function getState(rootId) {
    if (!stateByRoot[rootId]) {
      var labels = parseJsonId(rootId + "-labels") || [];
      var meta = parseJsonId(rootId + "-meta") || {};
      stateByRoot[rootId] = { labels: labels, meta: meta, autoLabels: JSON.parse(JSON.stringify(labels)), showCandidates: true };
    }
    return stateByRoot[rootId];
  }

  function labelSide(mode) { return mode === "transmittance" ? "below" : "above"; }

  function spectrumArrays(meta, mode) {
    if (mode === "transmittance") return { wn: meta.t_wn || [], y: meta.t_y || [] };
    return { wn: meta.abs_wn || [], y: meta.abs_y || [] };
  }

  function snapPeak(meta, requested, mode) {
    var spec = spectrumArrays(meta, mode);
    var wnArr = spec.wn, yArr = spec.y;
    if (!wnArr.length) return null;
    var win = meta.snap_window_cm1 || 25;
    var baseline = meta.transmittance_baseline_pct;
    var lo = requested - win, hi = requested + win;
    var bestIdx = -1, bestVal = mode === "transmittance" ? Infinity : -Infinity;
    for (var i = 0; i < wnArr.length; i++) {
      if (wnArr[i] < lo || wnArr[i] > hi) continue;
      if (mode === "transmittance") {
        if (baseline != null && yArr[i] >= baseline - 1e-6) continue;
        if (yArr[i] < bestVal) { bestVal = yArr[i]; bestIdx = i; }
      } else if (yArr[i] > bestVal) { bestVal = yArr[i]; bestIdx = i; }
    }
    if (bestIdx < 0) {
      var nearest = 0, nd = Math.abs(wnArr[0] - requested);
      for (var j = 1; j < wnArr.length; j++) {
        var d = Math.abs(wnArr[j] - requested);
        if (d < nd) { nd = d; nearest = j; }
      }
      bestIdx = nearest;
    }
    var snapStatus = "local_extremum";
    if (requested + win < wnArr[0] || requested - win > wnArr[wnArr.length - 1]) snapStatus = "nearest_point";
    return {
      requested_wavenumber_cm1: requested,
      snapped_wavenumber_cm1: wnArr[bestIdx],
      wavenumber_cm1: wnArr[bestIdx],
      peak_y: yArr[bestIdx],
      snap_window_cm1: win,
      snap_target: mode === "transmittance" ? "min_transmittance" : "max_absorbance",
      snap_status: snapStatus
    };
  }

  function findLabelIdx(labels, mode, wn) {
    var key = Math.round(wn);
    for (var i = 0; i < labels.length; i++) {
      if (labels[i].mode !== mode) continue;
      if (Math.round(labels[i].wavenumber_cm1) === key) return i;
    }
    return -1;
  }

  function persistLabels(rootId) {
    var st = getState(rootId);
    var el = document.getElementById(rootId + "-labels");
    if (el) el.textContent = JSON.stringify(st.labels);
  }

  function rowHtml(rootId, idx, lab, mode) {
    var wn = lab.wavenumber_cm1;
    var show = lab.show_label ? "checked" : "";
    return "<tr data-curation='" + rootId + "' data-label-idx='" + idx + "' data-mode='" + mode + "' data-wn='" + wn.toFixed(2) + "'>" +
      "<td><input type='checkbox' class='lbl-show' " + show + "/></td>" +
      "<td>" + wn.toFixed(1) + "</td>" +
      "<td>" + (lab.region || "") + "</td>" +
      "<td>" + (lab.score || 0).toFixed(3) + "</td>" +
      "<td>" + (lab.prominence || 0).toFixed(3) + "</td>" +
      "<td><input type='text' class='lbl-text' value='" + String(lab.label_text || Math.round(wn)) + "'/></td>" +
      "<td><button type='button' class='curation-btn lbl-x-minus'>&larr;</button>" +
      "<input type='number' class='lbl-xshift' value='" + (lab.xshift_px || 0) + "' step='1'/>" +
      "<button type='button' class='curation-btn lbl-x-plus'>&rarr;</button></td>" +
      "<td><button type='button' class='curation-btn lbl-y-minus'>&darr;</button>" +
      "<input type='number' class='lbl-yshift' value='" + (lab.yshift_px || 12) + "' step='1'/>" +
      "<button type='button' class='curation-btn lbl-y-plus'>&uarr;</button></td>" +
      "<td><button type='button' class='curation-btn lbl-reset'>Reset</button> " +
      "<button type='button' class='curation-btn lbl-delete'>Del</button></td></tr>";
  }

  function rebuildTable(rootId, mode) {
    var st = getState(rootId);
    var tbody = document.getElementById(rootId + "-tbody-" + mode);
    if (!tbody) return;
    var html = "";
    st.labels.forEach(function (lab, idx) {
      if (lab.mode !== mode) return;
      html += rowHtml(rootId, idx, lab, mode);
    });
    tbody.innerHTML = html || "<tr><td colspan='9' class='muted'>No candidate peaks.</td></tr>";
  }

  function rebuildAllTables(rootId) {
    rebuildTable(rootId, "normalized_absorbance");
    rebuildTable(rootId, "transmittance");
  }

  function addManualPeak(rootId, mode, requested, addedBy) {
    var st = getState(rootId);
    var snap = snapPeak(st.meta, parseFloat(requested), mode);
    if (!snap) { warn("No spectrum arrays for mode " + mode); return; }
    var idx = findLabelIdx(st.labels, mode, snap.wavenumber_cm1);
    if (idx >= 0) {
      st.labels[idx].show_label = true;
      if (st.labels[idx].source !== "manual") st.labels[idx].source = "manual";
    } else {
      st.labels.push({
        mode: mode,
        wavenumber_cm1: snap.wavenumber_cm1,
        peak_y: snap.peak_y,
        label_text: String(Math.round(snap.wavenumber_cm1)),
        show_label: true,
        xshift_px: 0,
        yshift_px: mode === "transmittance" ? -12 : 12,
        label_x_cm1: null,
        label_y_value: null,
        source: "manual",
        region: "",
        score: 0,
        prominence: 0,
        comment: "manual peak",
        requested_wavenumber_cm1: snap.requested_wavenumber_cm1,
        snapped_wavenumber_cm1: snap.snapped_wavenumber_cm1,
        snap_window_cm1: snap.snap_window_cm1,
        snap_target: snap.snap_target,
        added_by: addedBy,
        snap_status: snap.snap_status || "local_extremum"
      });
    }
    persistLabels(rootId);
    rebuildAllTables(rootId);
    var plotId = mode === "transmittance" ? st.meta.t_plot_id : st.meta.abs_plot_id;
    if (plotId) refreshPlot(rootId, plotId, mode);
  }

  function buildAnnotations(gd, labels, mode) {
    var ySpan = (gd.layout.yaxis.range[1] - gd.layout.yaxis.range[0]) || 1;
    var direction = mode === "transmittance" ? -1 : 1;
    var anns = [];
    labels.forEach(function (lab) {
      if (lab.mode !== mode || !lab.show_label) return;
      var wn = lab.wavenumber_cm1;
      var peakY = lab.peak_y;
      var base = 0.08 * ySpan * direction;
      var yOff = (lab.yshift_px / 280.0) * ySpan * direction + base;
      var lx = lab.label_x_cm1 != null ? lab.label_x_cm1 : wn;
      var ly = lab.label_y_value != null ? lab.label_y_value : peakY + yOff;
      lx += (lab.xshift_px / 280.0) * ySpan * 0.02;
      anns.push({
        x: lx, y: ly, xref: "x", yref: "y",
        text: String(lab.label_text || Math.round(wn)),
        showarrow: true, arrowhead: 0, arrowwidth: 0.9, arrowcolor: "#d95319",
        ax: 0, ay: peakY - ly, textangle: 0,
        font: { size: 10, color: "black" },
        bgcolor: "rgba(255,255,255,0.9)", borderpad: 2, captureevents: true
      });
    });
    return anns;
  }

  function refreshCandidateTrace(gd, labels, mode, showCandidates) {
    if (showCandidates === false) {
      Plotly.restyle(gd, { x: [[]], y: [[]], customdata: [[]] }, [1]);
      return;
    }
    var modeLabels = labels.filter(function (l) { return l.mode === mode; });
    var xs = [], ys = [], cd = [];
    modeLabels.forEach(function (lab) {
      xs.push(lab.wavenumber_cm1);
      ys.push(lab.peak_y);
      cd.push([
        labels.indexOf(lab),
        lab.wavenumber_cm1,
        lab.peak_y,
        lab.prominence || 0,
        lab.region || "",
        lab.score || 0,
        !!lab.show_label
      ]);
    });
    Plotly.restyle(gd, { x: [xs], y: [ys], customdata: [cd] }, [1]);
  }

  function refreshPlot(rootId, plotDivId, mode) {
    var gd = document.getElementById(plotDivId);
    if (!gd || !gd.layout) return;
    var st = getState(rootId);
    var showCandidates = st.showCandidates !== false;
    refreshCandidateTrace(gd, st.labels, mode, showCandidates);
    Plotly.relayout(gd, { annotations: buildAnnotations(gd, st.labels, mode) });
  }

  function syncRow(row) {
    var rootId = row.getAttribute("data-curation");
    var idx = parseInt(row.getAttribute("data-label-idx"), 10);
    var mode = row.getAttribute("data-mode");
    var st = getState(rootId);
    var lab = st.labels[idx];
    if (!lab) return;
    lab.show_label = row.querySelector(".lbl-show").checked;
    lab.label_text = row.querySelector(".lbl-text").value;
    lab.xshift_px = parseFloat(row.querySelector(".lbl-xshift").value) || 0;
    lab.yshift_px = parseFloat(row.querySelector(".lbl-yshift").value) || 0;
    var meta = st.meta;
    var plotId = mode === "transmittance" ? meta.t_plot_id : meta.abs_plot_id;
    refreshPlot(rootId, plotId, mode);
  }

  function mountPlot(rootId, plotDivId, errId) {
    var bundle = parseJsonId(plotDivId + "-fig");
    var host = document.getElementById(plotDivId);
    if (!host) { showError(errId, "Plot container missing: " + plotDivId); return; }
    if (!bundle || !bundle.data) { showError(errId, "Figure data missing for " + plotDivId); return; }
    var cfg = bundle.config || { responsive: true, displayModeBar: true };
    Plotly.newPlot(plotDivId, bundle.data, bundle.layout, cfg).then(function (gd) {
      var mode = host.getAttribute("data-mode");
      refreshPlot(rootId, plotDivId, mode);
      gd.on("plotly_click", function (ev) {
        if (!ev || !ev.points || !ev.points.length) return;
        var pt = ev.points[0];
        if (pt.curveNumber === 0) {
          var root = document.getElementById(rootId);
          if (root && root.classList.contains("curation-click-add-active")) {
            addManualPeak(rootId, mode, pt.x, "click");
          }
          return;
        }
        if (pt.curveNumber !== 1) return;
        var gidx = pt.customdata[0];
        var st = getState(rootId);
        var lab = st.labels[gidx];
        if (!lab || lab.mode !== mode) return;
        lab.show_label = !lab.show_label;
        var row = document.querySelector(
          "tr[data-curation='" + rootId + "'][data-label-idx='" + gidx + "']"
        );
        if (row) row.querySelector(".lbl-show").checked = lab.show_label;
        refreshPlot(rootId, plotDivId, mode);
      });
      gd.on("plotly_relayout", function (ev) {
        if (!ev) return;
        var st = getState(rootId);
        Object.keys(ev).forEach(function (k) {
          if (k.indexOf("annotations[") !== 0) return;
          var m = k.match(/annotations\[(\d+)\]\.(x|y|text)/);
          if (!m) return;
          var annIdx = parseInt(m[1], 10);
          var field = m[2];
          var shown = st.labels.filter(function (l) { return l.mode === mode && l.show_label; });
          var lab = shown[annIdx];
          if (!lab) return;
          if (field === "text") lab.label_text = ev[k];
          if (field === "x") lab.label_x_cm1 = ev[k];
          if (field === "y") lab.label_y_value = ev[k];
        });
      });
    }).catch(function (err) {
      showError(errId, "Plotly failed: " + err);
      warn(String(err));
    });
  }

  function downloadOverrides(rootId) {
    var st = getState(rootId);
    var payload = { spectrum_stem: st.meta.stem, labels: st.labels, source: "interactive" };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = st.meta.overrides_name || (st.meta.stem + "_label_overrides.json");
    a.click();
  }

  function resetLayout(rootId) {
    var st = getState(rootId);
    st.labels = JSON.parse(JSON.stringify(st.autoLabels));
    persistLabels(rootId);
    rebuildAllTables(rootId);
    if (st.meta.abs_plot_id) refreshPlot(rootId, st.meta.abs_plot_id, "normalized_absorbance");
    if (st.meta.t_plot_id) refreshPlot(rootId, st.meta.t_plot_id, "transmittance");
  }

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest("button");
    if (!btn) return;
    var rootId = btn.getAttribute("data-curation");
    var action = btn.getAttribute("data-action");
    if (action === "download-overrides" && rootId) return downloadOverrides(rootId);
    if (action === "reset-layout" && rootId) return resetLayout(rootId);
    if (action === "toggle-click-add" && rootId) {
      var root = document.getElementById(rootId);
      if (!root) return;
      root.classList.toggle("curation-click-add-active");
      btn.textContent = root.classList.contains("curation-click-add-active")
        ? "Click-add ON (click spectrum)"
        : "Add manual peak from click";
      return;
    }
    if (action === "add-typed-wn" && rootId) {
      var toolbar = btn.closest(".curation-toolbar");
      var wnInput = toolbar ? toolbar.querySelector(".manual-wn-input") : null;
      var modeSel = toolbar ? toolbar.querySelector(".manual-mode-select") : null;
      var wnVal = wnInput ? parseFloat(wnInput.value) : NaN;
      var modeVal = modeSel ? modeSel.value : "normalized_absorbance";
      if (!isFinite(wnVal)) { warn("Enter a valid wavenumber"); return; }
      addManualPeak(rootId, modeVal, wnVal, "typed");
      return;
    }
    var row = btn.closest("tr[data-label-idx]");
    if (!row) return;
    if (btn.classList.contains("lbl-x-minus")) row.querySelector(".lbl-xshift").value = (parseFloat(row.querySelector(".lbl-xshift").value)||0) - 2;
    if (btn.classList.contains("lbl-x-plus")) row.querySelector(".lbl-xshift").value = (parseFloat(row.querySelector(".lbl-xshift").value)||0) + 2;
    if (btn.classList.contains("lbl-y-minus")) row.querySelector(".lbl-yshift").value = (parseFloat(row.querySelector(".lbl-yshift").value)||0) - 2;
    if (btn.classList.contains("lbl-y-plus")) row.querySelector(".lbl-yshift").value = (parseFloat(row.querySelector(".lbl-yshift").value)||0) + 2;
    if (btn.classList.contains("lbl-reset")) {
      var st = getState(row.getAttribute("data-curation"));
      var idx = parseInt(row.getAttribute("data-label-idx"), 10);
      var auto = st.autoLabels[idx];
      if (auto) {
        ["show_label","label_text","xshift_px","yshift_px","label_x_cm1","label_y_value"].forEach(function (k) {
          st.labels[idx][k] = auto[k];
        });
        row.querySelector(".lbl-show").checked = st.labels[idx].show_label;
        row.querySelector(".lbl-text").value = st.labels[idx].label_text;
        row.querySelector(".lbl-xshift").value = st.labels[idx].xshift_px;
        row.querySelector(".lbl-yshift").value = st.labels[idx].yshift_px;
      }
    }
    if (btn.classList.contains("lbl-delete")) {
      var st2 = getState(row.getAttribute("data-curation"));
      var idx2 = parseInt(row.getAttribute("data-label-idx"), 10);
      var mode2 = row.getAttribute("data-mode");
      if (st2.labels[idx2] && st2.labels[idx2].source === "manual") {
        st2.labels.splice(idx2, 1);
        persistLabels(row.getAttribute("data-curation"));
        rebuildAllTables(row.getAttribute("data-curation"));
        var plotId2 = mode2 === "transmittance" ? st2.meta.t_plot_id : st2.meta.abs_plot_id;
        if (plotId2) refreshPlot(row.getAttribute("data-curation"), plotId2, mode2);
        return;
      }
    }
    syncRow(row);
  });

  document.addEventListener("change", function (ev) {
    if (ev.target.classList && ev.target.classList.contains("show-candidate-markers")) {
      var rootId = ev.target.getAttribute("data-curation");
      var st = getState(rootId);
      st.showCandidates = ev.target.checked;
      if (st.meta.abs_plot_id) refreshPlot(rootId, st.meta.abs_plot_id, "normalized_absorbance");
      if (st.meta.t_plot_id) refreshPlot(rootId, st.meta.t_plot_id, "transmittance");
      return;
    }
    var row = ev.target.closest("tr[data-label-idx]");
    if (row) syncRow(row);
  });
  document.addEventListener("input", function (ev) {
    var row = ev.target.closest("tr[data-label-idx]");
    if (row && (ev.target.classList.contains("lbl-text") || ev.target.classList.contains("lbl-xshift") || ev.target.classList.contains("lbl-yshift"))) syncRow(row);
  });

  function initAll() {
    waitForPlotly(15000).then(function () {
      document.querySelectorAll(".ftir-curation-root").forEach(function (root) {
        var rootId = root.id;
        var meta = parseJsonId(rootId + "-meta") || {};
        if (meta.abs_plot_id) mountPlot(rootId, meta.abs_plot_id, meta.abs_plot_id + "-err");
        if (meta.t_plot_id) mountPlot(rootId, meta.t_plot_id, meta.t_plot_id + "-err");
      });
    }).catch(function (err) {
      document.querySelectorAll(".curation-error").forEach(function (el) {
        el.style.display = "block";
        el.textContent = String(err);
      });
      warn(String(err));
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initAll);
  else initAll();
})();
</script>
"""


def curation_css() -> str:
    return CURATION_CSS


def curation_js() -> str:
    return CURATION_JS


def build_curation_plotly_figure(*args: Any, **kwargs: Any) -> go.Figure:
    """Backward-compatible helper for tests."""
    if "wn_min" not in kwargs and args:
        wn = np.asarray(args[0], dtype=float)
        kwargs.setdefault("wn_min", float(np.nanmin(wn)))
        kwargs.setdefault("wn_max", float(np.nanmax(wn)))
    elif "wn_min" not in kwargs:
        kwargs.setdefault("wn_min", 400.0)
        kwargs.setdefault("wn_max", 4000.0)
    bundle = _build_figure_bundle(*args, **kwargs)
    return go.Figure(data=bundle["data"], layout=bundle["layout"])
