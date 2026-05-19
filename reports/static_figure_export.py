"""
Matplotlib static figures for presentations (separate from interactive Plotly HTML).

Exports combined stack plus standalone panels (MATLAB-style):
- ``{stem}_spectrum_peaks`` — absorbance + all selected peak wavenumber labels
- ``{stem}_region_guide`` — FTIR region ruler only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml.canonical_peaks import StaticLabelPolicy, select_static_label_peak_ids
from ml.ftir_region_ruler import FTIR_RULER_REGIONS, ruler_activity_tier, ruler_region_activity
from reports.annotation_layout import (
    apply_peak_label_layout,
    plan_ruler_row_layouts,
    ruler_subplot_height_fraction,
)


def _short_title(name: str, max_len: int = 48) -> str:
    stem = Path(name).stem if name else "spectrum"
    if len(stem) <= max_len:
        return stem
    return stem[: max_len - 1] + "…"


def _peak_annotation_specs(
    canonical_pack: dict[str, Any],
    label_ids: set[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in canonical_pack.get("peaks") or []:
        if not row.get("labeled"):
            continue
        if str(row["peak_id"]) not in label_ids:
            continue
        specs.append(
            {
                "wn": float(row["center_cm1"]),
                "y": float(row["height"]),
                "text": row.get("label_text") or f"{float(row['center_cm1']):.0f}",
                "_peak": row.get("_raw") or {},
            }
        )
    return specs


def _layout_peak_labels(
    ann_specs: list[dict[str, Any]],
    *,
    wn: np.ndarray,
    y: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    y_max = float(np.nanmax(y)) if y.size else 1.0
    laid, layout_stats = apply_peak_label_layout(
        ann_specs,
        mode="smart",
        y_max=y_max,
        wn_min=float(np.nanmin(wn)) if wn.size else None,
        wn_max=float(np.nanmax(wn)) if wn.size else None,
    )
    y_span = max(y_max - float(np.nanmin(y)), 1e-9)
    return laid, layout_stats, y_span


def _annotate_spectrum_ax(
    ax: Any,
    *,
    laid: list[dict[str, Any]],
    y_span: float,
    label_fontsize: float = 8.0,
) -> None:
    for a in laid:
        wn_p = float(a["wn"])
        y_p = float(a["y"])
        ysh = int(a.get("yshift", 10) or 10)
        y_txt = y_p + (ysh / 280.0) * y_span
        angle = float(a.get("textangle", -90) or -90)
        ax.text(
            wn_p,
            y_txt,
            a.get("text", ""),
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color="#333333",
            rotation=angle,
            rotation_mode="anchor",
            clip_on=True,
        )
        if a.get("showarrow"):
            ax.annotate(
                "",
                xy=(wn_p, y_p),
                xytext=(wn_p, y_txt),
                arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.6),
            )


def _spectrum_figure_height(n_labels: int) -> float:
    """Extra vertical room when many peak labels are drawn."""
    return 6.0 + min(5.5, max(0, n_labels - 12) * 0.14)


def _ruler_figure_height_inches(total_line_weight: float) -> float:
    return max(3.2, 1.4 + 0.26 * float(total_line_weight))


def _draw_ruler_panel(
    ax: Any,
    *,
    evidence: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    front: bool = True,
    show_wavenumber_axis: bool = False,
) -> None:
    ax.set_xlim(float(np.nanmax(wn)), float(np.nanmin(wn)))
    ax.set_ylim(0, 1)
    row_layouts, _ = plan_ruler_row_layouts(FTIR_RULER_REGIONS, front=front)
    for layout in row_layouts:
        spec = layout["spec"]
        rel = ruler_region_activity(evidence, wn, y, spec)
        tier = ruler_activity_tier(rel)
        y0 = float(layout["y0"])
        y1 = float(layout["y1"])
        x0, x1 = float(spec.lo), float(spec.hi)
        fill = "#e2e8f0" if tier == "strong" else ("#f1f5f9" if tier == "active" else "#f8fafc")
        ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=fill, edgecolor="#94a3b8", linewidth=0.6)
        label = layout["label_text"].replace("<br>", "\n")
        from reports.annotation_layout import ruler_font_size_for_band

        fs = ruler_font_size_for_band(
            x0,
            x1,
            n_lines=int(layout["n_lines"]),
            band_height=float(layout["band_height"]),
            front=front,
            tier=tier,
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color="#334155" if tier != "muted" else "#94a3b8",
            clip_on=True,
        )
    if show_wavenumber_axis:
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    else:
        ax.axis("off")


def export_standalone_region_guide(
    *,
    spectrum_name: str,
    wn: np.ndarray,
    y: np.ndarray,
    pipeline: dict[str, Any],
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> str:
    """Region ruler / FTIR guide only (no spectrum stack)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence = pipeline.get("evidence") or {}
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(spectrum_name).stem)
    _, total_w = plan_ruler_row_layouts(FTIR_RULER_REGIONS, front=True)
    fig_h = _ruler_figure_height_inches(total_w)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    _draw_ruler_panel(
        ax,
        evidence=evidence,
        wn=wn,
        y=y,
        front=True,
        show_wavenumber_axis=True,
    )
    ax.set_title("FTIR region guide (tentative ranges)", fontsize=11, fontweight="semibold", pad=10)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12)
    path = out_dir / f"{stem}_region_guide.{fmt.lstrip('.')}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def export_standalone_spectrum_peaks(
    *,
    spectrum_name: str,
    wn: np.ndarray,
    y: np.ndarray,
    canonical_pack: dict[str, Any],
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
    label_policy: StaticLabelPolicy = "all",
    max_labels: int = 999,
    pipeline: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Spectrum + peak wavenumber labels only (no ruler, no Kronecker)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(spectrum_name).stem)
    pipe = pipeline or {}
    label_ids = select_static_label_peak_ids(
        canonical_pack,
        pipe,
        policy=label_policy,
        max_labels=max_labels,
    )
    ann_specs = _peak_annotation_specs(canonical_pack, label_ids)
    laid, layout_stats, y_span = _layout_peak_labels(ann_specs, wn=wn, y=y)
    n_labels = len(laid)
    label_fs = 7.5 if n_labels > 28 else (8.0 if n_labels > 18 else 8.5)

    fig_h = _spectrum_figure_height(n_labels)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.plot(wn, y, color="#0072bd", linewidth=1.0)
    plotted = [r for r in (canonical_pack.get("peaks") or []) if r.get("plotted")]
    ax.scatter(
        [float(r["center_cm1"]) for r in plotted],
        [float(r["height"]) for r in plotted],
        s=20,
        c="#d95319",
        zorder=3,
        alpha=0.9,
    )
    _annotate_spectrum_ax(ax, laid=laid, y_span=y_span, label_fontsize=label_fs)
    ax.set_xlim(float(np.nanmax(wn)), float(np.nanmin(wn)))
    ax.set_ylabel("Normalized absorbance", fontsize=9)
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    ax.set_title(_short_title(spectrum_name), fontsize=11, fontweight="semibold", pad=10)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    top_pad = 0.90 if n_labels <= 20 else 0.88
    fig.subplots_adjust(left=0.08, right=0.98, top=top_pad, bottom=0.10)
    path = out_dir / f"{stem}_spectrum_peaks.{fmt.lstrip('.')}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {
        "static_labels_requested": len(ann_specs),
        "static_labels_drawn": n_labels,
        "static_labels_hidden": len(ann_specs) - n_labels,
        "layout_stats": layout_stats,
        "spectrum_label_policy": label_policy,
    }
    return str(path), meta


def export_static_matplotlib_bundle(
    *,
    spectrum_name: str,
    wn: np.ndarray,
    y: np.ndarray,
    pipeline: dict[str, Any],
    canonical_pack: dict[str, Any],
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
    static_label_policy: StaticLabelPolicy = "key",
    spectrum_label_policy: StaticLabelPolicy | None = None,
    max_static_labels: int = 12,
    show_ruler: bool = True,
    export_separate_panels: bool = True,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(spectrum_name).stem)
    evidence = pipeline.get("evidence") or {}
    spec_policy: StaticLabelPolicy = spectrum_label_policy or static_label_policy
    spec_max = max_static_labels if spec_policy != "all" else 999

    combined_ids = select_static_label_peak_ids(
        canonical_pack,
        pipeline,
        policy=static_label_policy,
        max_labels=max_static_labels,
    )
    ann_specs = _peak_annotation_specs(canonical_pack, combined_ids)
    laid, layout_stats, y_span = _layout_peak_labels(ann_specs, wn=wn, y=y)
    hidden = len(ann_specs) - len(laid)

    files: list[str] = []
    spectrum_peaks_meta: dict[str, Any] = {}

    if export_separate_panels:
        if show_ruler:
            files.append(
                export_standalone_region_guide(
                    spectrum_name=spectrum_name,
                    wn=wn,
                    y=y,
                    pipeline=pipeline,
                    out_dir=out_dir,
                    fmt=fmt,
                    dpi=dpi,
                )
            )
        spec_path, spectrum_peaks_meta = export_standalone_spectrum_peaks(
            spectrum_name=spectrum_name,
            wn=wn,
            y=y,
            canonical_pack=canonical_pack,
            out_dir=out_dir,
            fmt=fmt,
            dpi=dpi,
            label_policy=spec_policy,
            max_labels=spec_max,
            pipeline=pipeline,
        )
        files.append(spec_path)

    nrows = 3 if show_ruler else 2
    if show_ruler:
        _, total_w = plan_ruler_row_layouts(FTIR_RULER_REGIONS, front=True)
        ruler_h = ruler_subplot_height_fraction(
            n_regions=len(FTIR_RULER_REGIONS),
            total_line_weight=total_w,
            front=True,
        )
        rest = 1.0 - ruler_h
        heights = [ruler_h, rest * 0.72, rest * 0.22]
        fig_h = 9.5 + max(0.0, ruler_h - 0.12) * 12.0
    else:
        heights = [0.72, 0.22]
        fig_h = 7.5
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(11, fig_h),
        gridspec_kw={"height_ratios": heights, "hspace": 0.08},
    )
    if nrows == 1:
        axes = [axes]
    ax_i = 0
    if show_ruler:
        _draw_ruler_panel(axes[ax_i], evidence=evidence, wn=wn, y=y)
        ax_i += 1
    ax_spec = axes[ax_i]
    ax_kron = axes[ax_i + 1]

    ax_spec.plot(wn, y, color="#0072bd", linewidth=1.0)
    ax_spec.set_ylabel("Normalized absorbance")
    ax_spec.set_xlim(float(np.nanmax(wn)), float(np.nanmin(wn)))
    ax_spec.grid(True, alpha=0.25, linewidth=0.4)

    plotted = [r for r in (canonical_pack.get("peaks") or []) if r.get("plotted")]
    ax_spec.scatter(
        [float(r["center_cm1"]) for r in plotted],
        [float(r["height"]) for r in plotted],
        s=18,
        c="#d95319",
        zorder=3,
        alpha=0.85,
    )
    _annotate_spectrum_ax(ax_spec, laid=laid, y_span=y_span)
    ax_spec.set_title(_short_title(spectrum_name), fontsize=11, fontweight="semibold", pad=8)

    kx = [float(r["center_cm1"]) for r in plotted]
    kh = [float(r["height"]) for r in plotted]
    ax_kron.stem(kx, kh, linefmt="#4a7fa5", markerfmt=" ", basefmt=" ")
    ax_kron.set_xlabel("Wavenumber (cm⁻¹)")
    ax_kron.set_ylabel("Peak height")
    ax_kron.set_xlim(float(np.nanmax(wn)), float(np.nanmin(wn)))
    ax_kron.grid(True, alpha=0.2, linewidth=0.35)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.12)
    combined = out_dir / f"{stem}_combined.{fmt.lstrip('.')}"
    fig.savefig(combined, dpi=dpi, bbox_inches="tight", facecolor="white")
    files.append(str(combined))
    plt.close(fig)

    kron_path = out_dir / f"{stem}_kronecker.{fmt.lstrip('.')}"
    fig3, ax3 = plt.subplots(figsize=(11, 2.8))
    ax3.stem(kx, kh, linefmt="#4a7fa5", markerfmt=" ", basefmt=" ")
    ax3.set_xlim(float(np.nanmax(wn)), float(np.nanmin(wn)))
    ax3.set_xlabel("Wavenumber (cm⁻¹)")
    fig3.savefig(kron_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    files.append(str(kron_path))

    return {
        "files": files,
        "static_labels_requested": len(ann_specs),
        "static_labels_drawn": len(laid),
        "static_labels_hidden": hidden,
        "layout_stats": layout_stats,
        "static_label_policy": static_label_policy,
        "spectrum_label_policy": spec_policy,
        "spectrum_peaks": spectrum_peaks_meta,
        "region_guide": show_ruler and export_separate_panels,
    }
