"""
Plotly overlay and HTML tables for advisory deconvolution on FTIR reports.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

from ml.ftir_deconv_assignment import candidate_summary_line

MARKER_DECONV_OVERLAY = "<!-- report-feature:deconv-overlay -->"


def _esc(s: Any) -> str:
    return html.escape(str(s))


def add_deconv_overlay_traces(
    fig: Any,
    deconv_pack: dict[str, Any],
    *,
    spectrum_row: int = 1,
    col: int = 1,
    audit: bool = False,
    theme_line: str = "#94a3b8",
) -> list[str]:
    """
  Add measured-segment fit, component curves, and markers. Returns trace names for legend toggle.
    """
    import plotly.graph_objects as go

    curves = deconv_pack.get("curves") or {}
    segments = curves.get("segments") or []
    trace_names: list[str] = []
    comp_idx = 0
    for seg in segments:
        wn = np.array(seg.get("wn") or [], dtype=float)
        if wn.size == 0:
            continue
        y_fit = np.array(seg.get("total_fit") or [], dtype=float)
        if y_fit.size == wn.size:
            name = f"Deconv fit ({seg.get('region_id', '')})"
            fig.add_trace(
                go.Scatter(
                    x=wn,
                    y=y_fit,
                    mode="lines",
                    name=name,
                    line=dict(color=theme_line, width=1, dash="dash"),
                    opacity=0.75,
                    legendgroup="deconv-fit",
                    showlegend=audit,
                    hovertemplate="Fitted sum %{x:.0f} cm⁻¹<br>A=%{y:.3f}<extra></extra>",
                ),
                row=spectrum_row,
                col=col,
            )
            trace_names.append(name)
        for comp_curve in seg.get("components") or []:
            yc = np.array(comp_curve.get("y") or [], dtype=float)
            if yc.size != wn.size:
                continue
            center = float(comp_curve.get("center", wn[int(len(wn) / 2)]))
            comp_idx += 1
            name = f"Component {comp_idx} ({center:.0f})"
            fig.add_trace(
                go.Scatter(
                    x=wn,
                    y=yc,
                    mode="lines",
                    name=name,
                    line=dict(width=1),
                    opacity=0.28,
                    legendgroup="deconv-comp",
                    showlegend=audit,
                    visible=True,
                    hoverinfo="skip",
                ),
                row=spectrum_row,
                col=col,
            )
            trace_names.append(name)
            y_mark = float(np.interp(center, wn, yc))
            hover_lines = [f"<b>{center:.0f} cm⁻¹</b>"]
            for comp in deconv_pack.get("components") or []:
                if abs(float(comp.get("center", 0)) - center) < 2.0:
                    cands = comp.get("candidates") or []
                    hover_lines.append(candidate_summary_line(center, cands))
                    hover_lines.append(
                        f"Height {float(comp.get('height', 0)):.3f} · "
                        f"FWHM {float(comp.get('fwhm', 0)):.1f} · "
                        f"rel. area {float(comp.get('rel_area', 0)):.2f}"
                    )
                    if comp.get("fit_r2") is not None:
                        hover_lines.append(f"Regional R² {float(comp['fit_r2']):.2f}")
                    break
            fig.add_trace(
                go.Scatter(
                    x=[center],
                    y=[y_mark],
                    mode="markers+text" if audit else "markers",
                    text=[f"{center:.0f}"] if audit else None,
                    textposition="top center",
                    name=f"Deconv center {center:.0f}",
                    marker=dict(size=7, symbol="line-ns-open", color="#6366f1", line=dict(width=1.5)),
                    legendgroup="deconv-markers",
                    showlegend=False,
                    customdata=["<br>".join(hover_lines)],
                    hovertemplate="%{customdata}<extra>Deconv component (candidate)</extra>",
                ),
                row=spectrum_row,
                col=col,
            )
    return trace_names


def deconv_legend_buttons(trace_names: list[str]) -> list[dict[str, Any]]:
    """Plotly updatemenu: hide/show component traces."""
    if not trace_names:
        return []
    return [
        {
            "label": "Hide components",
            "method": "restyle",
            "args": [{"visible": "legendonly"}, trace_names],
        },
        {
            "label": "Show components",
            "method": "restyle",
            "args": [{"visible": True}, trace_names],
        },
    ]


def build_deconv_candidates_table_html(
    rows: list[dict[str, Any]],
    *,
    anchor: str = "",
    audit: bool = False,
    max_rows: int = 12,
) -> str:
    if not rows:
        return ""
    display = rows if audit else rows[:max_rows]
    trs = []
    for r in display:
        trs.append(
            "<tr>"
            f"<td>{r['center']}</td>"
            f"<td>{r['width']}</td>"
            f"<td>{r['rel_area']}</td>"
            f"<td>{_esc(r['top_candidates'])}</td>"
            f"<td>{_esc(r['evidence_role'])}</td>"
            f"<td>{_esc(_truncate(r.get('caution', ''), 100))}</td>"
            f"<td>{_esc(r['linked_assignment'])}</td>"
            "</tr>"
        )
    extra = ""
    if not audit and len(rows) > len(display):
        extra = f"<p class='muted small'>Showing top {len(display)} of {len(rows)} components.</p>"
    resid_col = ""
    if audit:
        resid_col = "<th>R²</th>"
        trs = []
        for r in display:
            trs.append(
                "<tr>"
                f"<td>{r['center']}</td>"
                f"<td>{r['width']}</td>"
                f"<td>{r['rel_area']}</td>"
                f"<td>{r.get('height', '')}</td>"
                f"<td>{_esc(r['top_candidates'])}</td>"
                f"<td>{_esc(r['evidence_role'])}</td>"
                f"<td>{float(r.get('fit_r2', 0)):.2f}</td>"
                f"<td>{_esc(_truncate(r.get('caution', ''), 120))}</td>"
                f"<td>{_esc(r['linked_assignment'])}</td>"
                "</tr>"
            )
    return (
        MARKER_DECONV_OVERLAY
        + f"<section class='deconv-candidates-section' id='{_esc(anchor)}-deconv'>"
        "<h3>Deconvoluted peak candidates</h3>"
        "<p class='muted small'>Advisory fitted components — possible overlap; not confirmed assignments.</p>"
        + extra
        + "<div class='table-scroll'><table class='tbl tbl-zebra deconv-candidates-tbl'><thead><tr>"
        "<th>Center (cm⁻¹)</th><th>Width</th><th>Rel. area</th>"
        + ("<th>Height</th>" if audit else "")
        + "<th>Top candidates</th><th>Evidence role</th>"
        + resid_col
        + "<th>Caution</th><th>Linked assignment</th>"
        "</tr></thead><tbody>"
        + "".join(trs)
        + "</tbody></table></div></section>"
    )


def _truncate(s: str, n: int) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def append_deconv_to_justify_panels(
    justify_html: str,
    pipeline: dict[str, Any],
    deconv_pack: dict[str, Any] | None,
) -> str:
    """Inject supporting deconv lines into fg-justify cards when present."""
    if not deconv_pack or not justify_html:
        return justify_html
    from ml.ftir_deconv_assignment import fg_deconv_support_snippet

    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    out = justify_html
    for lab in assigns:
        snip = fg_deconv_support_snippet(lab, deconv_pack, pipeline=pipeline)
        if not snip:
            continue
        needle = f"<h4>{_esc(lab)}"
        if needle in out:
            out = out.replace(
                needle,
                needle,
                1,
            )
            idx = out.find(needle)
            if idx >= 0:
                ul = out.find("<ul", idx)
                if ul >= 0:
                    out = (
                        out[:ul]
                        + f"<p class='muted small deconv-fg-support'>{_esc(snip)}</p>"
                        + out[ul:]
                    )
    return out
