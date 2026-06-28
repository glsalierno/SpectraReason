"""
Publication-ready FTIR figures: transmittance (%T) and normalized absorbance with
horizontal leader-line peak labels.

Uses ``lib.ftir_foundation.preprocess_spectrum`` for absorbance preprocessing.
Two-stage peak picking via ``reports.paper_peak_selection``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.signal import savgol_filter

from lib.ftir_foundation import infer_intensity_mode, preprocess_spectrum, read_spectrum
from lib.intensity_modes import (
    ForceIntensityMode,
    classify_intensity,
    has_transmittance_panel,
    plan_transmittance_panel,
    resolve_intensity_mode as classify_preprocess_mode,
)
from reports.annotation_layout import apply_horizontal_leader_label_layout
from reports.label_overrides import (
    build_auto_curation_labels,
    build_auto_override_payload,
    load_label_overrides,
    merge_overrides_with_auto,
    overrides_path,
    overrides_to_laid_peaks,
    save_label_overrides,
    selected_absorbance_peaks_from_labels,
)
from reports.paper_peak_overrides import load_override_store, resolve_overrides_for_stem
from reports.paper_peak_selection import (
    PaperPeakSelectionConfig,
    match_transmittance_minima,
    parse_ignore_label_ranges,
    select_labeled_peaks,
)
from reports.spectrum_feedback import (
    build_feedback_section_html,
    build_spectrum_feedback,
    write_spectrum_feedback_files,
)

MATLAB_BLUE = "#0072bd"
PEAK_COLOR = "#d95319"
WN_MIN, WN_MAX = 400.0, 4000.0
FIG_W, FIG_H = 6.5, 3.4
DEFAULT_DPI = 300
DEFAULT_APPARENT_TRANSMITTANCE_BASELINE_PCT = 95.0
PaperFigureMode = Literal["transmittance", "normalized_absorbance"]
PaperLabelStyle = Literal["horizontal-leader"]


@dataclass
class PaperFigureConfig:
    modes: tuple[PaperFigureMode, ...] = ("transmittance", "normalized_absorbance")
    formats: tuple[str, ...] = ("png", "svg", "pdf")
    max_peak_labels: int = 10
    label_style: PaperLabelStyle = "horizontal-leader"
    dpi: int = DEFAULT_DPI
    wn_min: float = WN_MIN
    wn_max: float = WN_MAX
    peak_sensitivity: str = "sensitive"
    spectrum_line_color: str = MATLAB_BLUE
    peak_color: str = PEAK_COLOR
    label_fontsize: float = 9.0
    min_peak_prominence: float = 0.04
    min_peak_height: float = 0.05
    min_peak_distance_cm1: float = 20.0
    ignore_label_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: [(400.0, 900.0)]
    )
    use_shoulder_detection: bool = False
    override_file: Path | None = None
    export_spectrum_feedback: bool = True
    transmittance_match_cm1: float = 25.0
    label_overrides_dir: Path | None = None
    apply_label_overrides: bool = False
    save_label_overrides: bool = True
    show_peak_markers: bool = False
    allow_apparent_transmittance: bool = False
    force_intensity_mode: ForceIntensityMode | None = None
    apparent_transmittance_label: str = "Apparent Transmittance (%)"


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(name).stem)


def resolve_intensity_mode(
    path: Path,
    hint: str,
    raw: np.ndarray,
    *,
    force_mode: ForceIntensityMode | None = None,
) -> str:
    return classify_preprocess_mode(path, hint, raw, force_mode=force_mode)


def transmittance_valid(mode: str) -> bool:
    from lib.intensity_modes import is_native_transmittance_category

    return is_native_transmittance_category(mode)


def _percent_transmittance_from_plan(y_plan: np.ndarray) -> np.ndarray:
    return np.asarray(y_plan, dtype=float)


def _smooth_percent_t(wn: np.ndarray, y_pct: np.ndarray, *, sg_window: int = 11, sg_poly: int = 2) -> np.ndarray:
    if y_pct.size < max(sg_window, 5):
        return y_pct
    win = min(sg_window, y_pct.size if y_pct.size % 2 else y_pct.size - 1)
    win = max(win | 1, 5)
    return savgol_filter(y_pct, window_length=win, polyorder=min(sg_poly, win - 1))


def apparent_transmittance_baseline_pct(t_plan: Any) -> float | None:
    """Flat baseline for apparent %T: suppress upward artifacts above ~95 %T."""
    if bool(getattr(t_plan, "is_apparent", False)) and has_transmittance_panel(t_plan):
        return DEFAULT_APPARENT_TRANSMITTANCE_BASELINE_PCT
    return None


def prepare_transmittance_trace(y_pct: np.ndarray, *, baseline_pct: float | None = None) -> np.ndarray:
    """Clip apparent %T at baseline so positive peaks above the floor are flattened."""
    y = np.asarray(y_pct, dtype=float)
    if baseline_pct is not None:
        y = np.minimum(y, float(baseline_pct))
    return y


def transmittance_y_limits(
    y: np.ndarray,
    *,
    baseline_pct: float | None = None,
    padding_frac: float = 0.18,
) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0, 100.0
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if baseline_pct is not None:
        y_max = min(y_max, float(baseline_pct))
    span = max(y_max - y_min, 1e-6)
    pad = span * padding_frac
    return y_min - pad * 1.5, y_max + pad * 0.35


def _window_mask(wn: np.ndarray, *, wn_min: float, wn_max: float) -> np.ndarray:
    return (wn >= wn_min) & (wn <= wn_max)


def _assignment_for_wn(wn: float, pipeline: dict[str, Any] | None) -> str:
    if not pipeline:
        return ""
    ev = pipeline.get("evidence") or {}
    best = ""
    best_d = 1e9
    for p in ev.get("peaks") or []:
        if not isinstance(p, dict):
            continue
        pw = float(p.get("wn_cm1", p.get("center_cm1", 0)) or 0)
        d = abs(pw - wn)
        if d < best_d and d <= 12.0:
            best_d = d
            labs = p.get("labels") or p.get("fg_labels") or []
            if isinstance(labs, (list, tuple)) and labs:
                best = str(labs[0])
            elif p.get("assignment"):
                best = str(p["assignment"])
    return best


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _style_axes(
    ax: Any,
    *,
    ylabel: str,
    show_grid: bool,
    title: str = "",
    wn_min: float = WN_MIN,
    wn_max: float = WN_MAX,
) -> None:
    ax.set_xlim(wn_max, wn_min)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    if show_grid:
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
    else:
        ax.grid(False)


def _draw_peak_labels(
    ax: Any,
    laid: list[dict[str, Any]],
    *,
    label_side: str,
    peak_color: str,
    label_fontsize: float,
    show_peak_markers: bool = False,
) -> None:
    above = label_side == "above"
    for item in laid:
        if not item.get("show_label", True):
            continue
        wn = float(item["wn"])
        y = float(item["y"])
        label_y = float(item.get("label_y", y))
        if show_peak_markers:
            ax.plot(wn, y, "o", color=peak_color, markersize=4.5, zorder=4)
        ax.annotate(
            str(item.get("text", "")),
            xy=(wn, y),
            xytext=(wn, label_y),
            textcoords="data",
            ha="center",
            va="bottom" if above else "top",
            fontsize=label_fontsize,
            color="black",
            rotation=0,
            arrowprops=dict(arrowstyle="-", color=peak_color, lw=0.9, shrinkA=0, shrinkB=2),
            clip_on=True,
            zorder=5,
        )


def _save_figure(fig: Any, base: Path, formats: tuple[str, ...], dpi: int) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        f = str(fmt).lower()
        out = base.with_suffix(f".{f}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if f == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out, format=f, **kwargs)
        paths.append(str(out.resolve()))
    return paths


def format_download_links(
    paths: list[str],
    report_dir: Path,
    *,
    extra: list[tuple[str, str]] | None = None,
) -> str:
    """Compact download row (PNG/SVG/PDF/CSV/JSON) without embedded images."""
    import html as html_mod

    base = Path(report_dir).resolve()

    def rel(path: str) -> str:
        try:
            return Path(path).resolve().relative_to(base).as_posix()
        except ValueError:
            return Path(path).name

    links: list[str] = []
    for p in paths:
        suffix = Path(p).suffix.lower().lstrip(".")
        if suffix:
            links.append(
                f"<a class='btn-dl' href='{html_mod.escape(rel(p), quote=True)}' download>"
                f"{suffix.upper()}</a>"
            )
    for label, path in extra or []:
        if path:
            links.append(
                f"<a class='btn-dl' href='{html_mod.escape(rel(path), quote=True)}' download>"
                f"{html_mod.escape(label)}</a>"
            )
    return " ".join(links)


def _write_peak_csv_tables(
    out_dir: Path,
    stem: str,
    selection: Any,
    t_minima_rows: list[dict[str, Any]],
    pipeline: dict[str, Any] | None,
    merged_labels: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    all_fields = [
        "wavenumber_cm1",
        "intensity",
        "prominence",
        "height",
        "region",
        "score",
        "detected",
        "label_selected",
        "reason_not_selected",
    ]
    all_rows = []
    for c in selection.candidates:
        wn_v = float(c["wavenumber_cm1"])
        all_rows.append(
            {
                "wavenumber_cm1": f"{wn_v:.1f}",
                "intensity": f"{float(c['intensity']):.4f}",
                "prominence": f"{float(c['prominence']):.4f}",
                "height": f"{float(c.get('height', c['intensity'])):.4f}",
                "region": c.get("region", ""),
                "score": f"{float(c['score']):.4f}",
                "detected": "yes" if c.get("detected") else "no",
                "label_selected": "yes" if c.get("label_selected") else "no",
                "reason_not_selected": c.get("reason_not_selected", ""),
            }
        )

    selected_fields = [
        "wavenumber_cm1",
        "intensity",
        "prominence",
        "height",
        "region",
        "score",
        "label_shown",
        "assignment_if_available",
    ]
    selected_rows = []
    manual_rows: list[dict[str, Any]] = []
    manual_fields = [
        "requested_wavenumber_cm1",
        "snapped_wavenumber_cm1",
        "mode",
        "source",
        "snap_target",
        "snap_window_cm1",
        "intensity",
        "region",
        "label_text",
        "show_label",
    ]
    if merged_labels:
        for lab in merged_labels:
            if str(lab.get("source", "")).lower() != "manual":
                continue
            wn_v = float(lab.get("snapped_wavenumber_cm1", lab.get("wavenumber_cm1", 0)))
            manual_rows.append(
                {
                    "requested_wavenumber_cm1": f"{float(lab.get('requested_wavenumber_cm1', wn_v)):.1f}",
                    "snapped_wavenumber_cm1": f"{wn_v:.1f}",
                    "mode": lab.get("mode", ""),
                    "source": lab.get("source", "manual"),
                    "snap_target": lab.get("snap_target", ""),
                    "snap_window_cm1": lab.get("snap_window_cm1", ""),
                    "intensity": f"{float(lab.get('peak_y', 0)):.4f}",
                    "region": lab.get("region", ""),
                    "label_text": lab.get("label_text", ""),
                    "show_label": "yes" if lab.get("show_label", True) else "no",
                }
            )
        for lab in merged_labels:
            if not lab.get("show_label", True):
                continue
            if str(lab.get("mode", "")) != "normalized_absorbance":
                continue
            wn_v = float(lab["wavenumber_cm1"])
            selected_rows.append(
                {
                    "wavenumber_cm1": f"{wn_v:.1f}",
                    "intensity": f"{float(lab.get('peak_y', 0)):.4f}",
                    "prominence": f"{float(lab.get('prominence', 0)):.4f}",
                    "height": f"{float(lab.get('peak_y', 0)):.4f}",
                    "region": lab.get("region", ""),
                    "score": f"{float(lab.get('score', 0)):.4f}",
                    "label_shown": "yes",
                    "assignment_if_available": _assignment_for_wn(wn_v, pipeline),
                }
            )
        for lab in merged_labels:
            if str(lab.get("source", "")).lower() != "manual":
                continue
            wn_v = float(lab.get("snapped_wavenumber_cm1", lab.get("wavenumber_cm1", 0)))
            all_rows.append(
                {
                    "wavenumber_cm1": f"{wn_v:.1f}",
                    "intensity": f"{float(lab.get('peak_y', 0)):.4f}",
                    "prominence": f"{float(lab.get('prominence', 0)):.4f}",
                    "height": f"{float(lab.get('peak_y', 0)):.4f}",
                    "region": lab.get("region", ""),
                    "score": f"{float(lab.get('score', 0)):.4f}",
                    "detected": "manual",
                    "label_selected": "yes" if lab.get("show_label", True) else "no",
                    "reason_not_selected": "manual",
                }
            )
    else:
        for c in selection.selected:
            wn_v = float(c["wavenumber_cm1"])
            selected_rows.append(
                {
                    "wavenumber_cm1": f"{wn_v:.1f}",
                    "intensity": f"{float(c['intensity']):.4f}",
                    "prominence": f"{float(c['prominence']):.4f}",
                    "height": f"{float(c.get('height', c['intensity'])):.4f}",
                    "region": c.get("region", ""),
                    "score": f"{float(c['score']):.4f}",
                    "label_shown": "yes",
                    "assignment_if_available": _assignment_for_wn(wn_v, pipeline),
                }
            )

    suppressed_fields = [
        "wavenumber_cm1",
        "intensity",
        "prominence",
        "height",
        "region",
        "score",
        "reason_suppressed",
    ]
    suppressed_rows = []
    for c in selection.suppressed:
        wn_v = float(c["wavenumber_cm1"])
        suppressed_rows.append(
            {
                "wavenumber_cm1": f"{wn_v:.1f}",
                "intensity": f"{float(c['intensity']):.4f}",
                "prominence": f"{float(c['prominence']):.4f}",
                "height": f"{float(c.get('height', c['intensity'])):.4f}",
                "region": c.get("region", ""),
                "score": f"{float(c['score']):.4f}",
                "reason_suppressed": c.get("reason_suppressed", c.get("reason_not_selected", "")),
            }
        )

    t_fields = [
        "absorbance_peak_cm1",
        "transmittance_label_cm1",
        "transmittance_value",
        "matched_within_cm1",
        "label_shown",
    ]
    t_rows = [{k: r.get(k, "") for k in t_fields} for r in t_minima_rows]

    paths = {
        "all_candidates": out_dir / f"{stem}_peaks_all_candidates.csv",
        "selected": out_dir / f"{stem}_peaks_selected.csv",
        "suppressed": out_dir / f"{stem}_peaks_suppressed.csv",
        "transmittance_minima": out_dir / f"{stem}_peaks_transmittance_minima.csv",
        "manual": out_dir / f"{stem}_peaks_manual.csv",
    }
    _write_csv(paths["all_candidates"], all_rows, all_fields)
    _write_csv(paths["selected"], selected_rows, selected_fields)
    _write_csv(paths["suppressed"], suppressed_rows, suppressed_fields)
    _write_csv(paths["transmittance_minima"], t_rows, t_fields)
    _write_csv(paths["manual"], manual_rows, manual_fields)
    return {k: str(v.resolve()) for k, v in paths.items()}


def export_paper_figures_for_spectrum(
    input_path: Path,
    out_dir: Path,
    *,
    config: PaperFigureConfig | None = None,
    pipeline: dict[str, Any] | None = None,
    title: str | None = None,
    override_store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export labeled paper figures, peak CSV tables, and feedback for one spectrum."""
    cfg = config or PaperFigureConfig()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(input_path.name)
    display_title = title or stem.replace("_", " ")
    wn, raw, hint = read_spectrum(input_path)
    classification = classify_intensity(
        input_path,
        hint,
        raw,
        force_mode=cfg.force_intensity_mode,
    )
    mode = classification.preprocess_mode
    t_plan = plan_transmittance_panel(
        raw,
        classification,
        allow_apparent=bool(cfg.allow_apparent_transmittance),
        apparent_label=str(cfg.apparent_transmittance_label),
    )

    wn_p, y_norm, info = preprocess_spectrum(
        wn,
        raw,
        intensity_mode=mode,
        normalize=True,
    )
    mask_a = _window_mask(wn_p, wn_min=cfg.wn_min, wn_max=cfg.wn_max)

    store = override_store
    if store is None and cfg.override_file:
        store = load_override_store(cfg.override_file)
    overrides = resolve_overrides_for_stem(store or {}, stem)
    sel_cfg = PaperPeakSelectionConfig(
        min_prominence=float(cfg.min_peak_prominence),
        min_height=float(cfg.min_peak_height),
        min_distance_cm1=float(cfg.min_peak_distance_cm1),
        max_labels=int(cfg.max_peak_labels),
        ignore_label_ranges=list(cfg.ignore_label_ranges),
        transmittance_match_cm1=float(cfg.transmittance_match_cm1),
        use_shoulder_detection=bool(cfg.use_shoulder_detection),
    )
    selection = select_labeled_peaks(
        wn_p[mask_a],
        y_norm[mask_a],
        config=sel_cfg,
        overrides=overrides,
    )

    t_minima_rows: list[dict[str, Any]] = []
    wn_t = np.array([])
    y_t = np.array([])
    t_baseline_pct: float | None = None
    if has_transmittance_panel(t_plan):
        y_pct = _percent_transmittance_from_plan(t_plan.y_values)  # type: ignore[arg-type]
        mask_t = _window_mask(wn, wn_min=cfg.wn_min, wn_max=cfg.wn_max)
        t_baseline_pct = apparent_transmittance_baseline_pct(t_plan)
        y_t = prepare_transmittance_trace(
            _smooth_percent_t(wn[mask_t], y_pct[mask_t]),
            baseline_pct=t_baseline_pct,
        )
        wn_t = wn[mask_t]
        t_minima_rows = match_transmittance_minima(
            wn_t,
            y_t,
            selection.selected,
            match_window_cm1=float(cfg.transmittance_match_cm1),
            baseline_pct=t_baseline_pct,
        )

    auto_label_records = build_auto_curation_labels(selection, t_minima_rows)

    ov_dir = Path(cfg.label_overrides_dir) if cfg.label_overrides_dir else out_dir.parent
    ov_path = overrides_path(ov_dir, stem)
    saved_ov = load_label_overrides(ov_path) if cfg.apply_label_overrides else {}
    merged_labels = merge_overrides_with_auto(
        auto_label_records,
        saved_ov if cfg.apply_label_overrides else None,
    )
    if cfg.save_label_overrides and not ov_path.is_file():
        save_label_overrides(ov_path, build_auto_override_payload(stem=stem, labels=auto_label_records))

    y_abs_min = float(np.nanmin(y_norm[mask_a]))
    y_abs_max = float(np.nanmax(y_norm[mask_a]))
    y_abs_pad = (y_abs_max - y_abs_min) * 0.18
    if cfg.apply_label_overrides and merged_labels:
        laid_abs = overrides_to_laid_peaks(
            merged_labels,
            mode="normalized_absorbance",
            y_min=y_abs_min - y_abs_pad * 0.2,
            y_max=y_abs_max + y_abs_pad,
        )
        layout_stats = {"labels_placed": len(laid_abs), "from_overrides": True}
    else:
        laid_abs, layout_stats = apply_horizontal_leader_label_layout(
            selection.selected_for_plot,
            y_min=y_abs_min,
            y_max=y_abs_max + y_abs_pad,
            wn_min=cfg.wn_min,
            wn_max=cfg.wn_max,
            label_side="above",
            max_labels=int(cfg.max_peak_labels),
        )
    labeled_abs_wn = {round(float(x["wn"]), 1) for x in laid_abs if x.get("show_label")}
    result: dict[str, Any] = {
        "input": str(input_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "stem": stem,
        "display_title": display_title,
        "intensity_mode": classification.category,
        "preprocess_mode": mode,
        "transmittance_valid": has_transmittance_panel(t_plan),
        "transmittance_is_apparent": bool(t_plan.is_apparent and has_transmittance_panel(t_plan)),
        "transmittance_note": t_plan.skip_reason or t_plan.warning or "",
        "transmittance_skip_banner": t_plan.banner_html or "",
        "figures": {},
        "peak_tables": {},
        "layout_stats": layout_stats,
        "peak_selection": {
            "n_candidates": len(selection.candidates),
            "n_selected": len(selection.selected),
            "n_suppressed": len(selection.suppressed),
            "n_labeled_absorbance": len(labeled_abs_wn),
        },
        "preprocess": {k: info[k] for k in ("intensity_mode", "baseline", "normalized") if k in info},
        "feedback": {},
        "label_overrides_path": str(ov_path.resolve()),
    }

    if "normalized_absorbance" in cfg.modes:
        for variant, show_labels, show_grid in (
            ("normalized_absorbance_peaks", True, True),
            ("manuscript_normalized_absorbance", False, False),
        ):
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.plot(
                wn_p[mask_a],
                y_norm[mask_a],
                color=cfg.spectrum_line_color,
                linewidth=1.0,
            )
            ax.set_ylim(y_abs_min - y_abs_pad * 0.2, y_abs_max + y_abs_pad)
            if show_labels:
                _draw_peak_labels(
                    ax,
                    laid_abs,
                    label_side="above",
                    peak_color=cfg.peak_color,
                    label_fontsize=cfg.label_fontsize,
                    show_peak_markers=cfg.show_peak_markers,
                )
            _style_axes(
                ax,
                ylabel="Normalized Absorbance (0–1)",
                show_grid=show_grid,
                wn_min=cfg.wn_min,
                wn_max=cfg.wn_max,
            )
            fig.tight_layout()
            paths = _save_figure(fig, out_dir / f"{stem}_{variant}", cfg.formats, cfg.dpi)
            plt.close(fig)
            result["figures"][variant] = paths

    if "transmittance" in cfg.modes:
        if not has_transmittance_panel(t_plan):
            result["transmittance_note"] = t_plan.skip_reason or (
                "Transmittance export skipped: native %T not available."
            )
        else:
            y_t_min, y_t_max = transmittance_y_limits(y_t, baseline_pct=t_baseline_pct)
            y_t_pad = max(y_t_max - y_t_min, 1e-6) * 0.18
            t_ylabel = t_plan.ylabel
            if cfg.apply_label_overrides and merged_labels:
                laid_t = overrides_to_laid_peaks(
                    merged_labels,
                    mode="transmittance",
                    y_min=y_t_min - y_t_pad,
                    y_max=y_t_max + y_t_pad * 0.3,
                )
                t_stats = {"labels_placed": len(laid_t), "from_overrides": True}
            else:
                dip_plot = [
                    {
                        "wn": float(r["wn"]),
                        "y": float(r["y"]),
                        "text": str(r.get("text", "")),
                        "prominence": float(r.get("prominence", 0)),
                    }
                    for r in t_minima_rows
                    if str(r.get("label_shown", "")).lower() == "yes"
                ]
                laid_t, t_stats = apply_horizontal_leader_label_layout(
                    dip_plot,
                    y_min=y_t_min - y_t_pad,
                    y_max=y_t_max + y_t_pad * 0.3,
                    wn_min=cfg.wn_min,
                    wn_max=cfg.wn_max,
                    label_side="below",
                    max_labels=int(cfg.max_peak_labels),
                )
            result["layout_stats"]["transmittance"] = t_stats
            if t_plan.warning:
                result["transmittance_note"] = t_plan.warning

            for variant, show_labels, show_grid in (
                ("transmittance_peaks", True, True),
                ("manuscript_transmittance", False, False),
            ):
                fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                ax.plot(wn_t, y_t, color=cfg.spectrum_line_color, linewidth=1.0)
                ax.set_ylim(y_t_min - y_t_pad, y_t_max + y_t_pad * 0.3)
                if show_labels:
                    _draw_peak_labels(
                        ax,
                        laid_t,
                        label_side="below",
                        peak_color=cfg.peak_color,
                        label_fontsize=cfg.label_fontsize,
                        show_peak_markers=cfg.show_peak_markers,
                    )
                _style_axes(
                    ax,
                    ylabel=t_ylabel,
                    show_grid=show_grid,
                    wn_min=cfg.wn_min,
                    wn_max=cfg.wn_max,
                )
                fig.tight_layout()
                paths = _save_figure(fig, out_dir / f"{stem}_{variant}", cfg.formats, cfg.dpi)
                plt.close(fig)
                result["figures"][variant] = paths

    result["peak_tables"] = _write_peak_csv_tables(
        out_dir,
        stem,
        selection,
        t_minima_rows,
        pipeline,
        merged_labels=merged_labels if cfg.apply_label_overrides else None,
    )

    if cfg.export_spectrum_feedback:
        feedback_selected = selection.selected
        if cfg.apply_label_overrides and merged_labels:
            feedback_selected = selected_absorbance_peaks_from_labels(merged_labels)
        feedback = build_spectrum_feedback(
            stem=stem,
            selection_result=selection,
            pipeline=pipeline,
            preprocess_info=result.get("preprocess"),
            transmittance_valid=bool(result["transmittance_valid"]),
            transmittance_note=str(result.get("transmittance_note") or ""),
            y_norm=y_norm[mask_a],
            selected_override=feedback_selected if cfg.apply_label_overrides else None,
        )
        feedback_paths = write_spectrum_feedback_files(out_dir.parent.parent, stem, feedback)
        result["feedback"] = {**feedback, "files": feedback_paths}
        result["feedback_html"] = build_feedback_section_html(feedback, out_dir.parent.parent)

    return result


def write_paper_figures_index(out_dir: Path, manifests: list[dict[str, Any]]) -> Path:
    out_dir = Path(out_dir)
    lines = [
        "# Paper / manuscript FTIR figures",
        "",
        "Two-stage peak picking on normalized absorbance; transmittance labels reuse selected wavenumbers.",
        "900–400 cm⁻¹ is excluded from automatic labeling unless overridden.",
        "",
        "| Spectrum | Transmittance | Normalized absorbance | Selected peaks | Feedback |",
        "|----------|---------------|------------------------|----------------|----------|",
    ]
    for m in manifests:
        stem = m.get("stem", "?")
        t_figs = m.get("figures", {}).get("transmittance_peaks") or m.get("figures", {}).get(
            "manuscript_transmittance"
        )
        a_figs = m.get("figures", {}).get("normalized_absorbance_peaks") or m.get(
            "figures", {}
        ).get("manuscript_normalized_absorbance")
        t_cell = Path(t_figs[0]).name if t_figs else ("—" if not m.get("transmittance_valid") else "missing")
        if t_figs and m.get("transmittance_is_apparent"):
            t_cell = f"{t_cell} (apparent)"
        a_cell = Path(a_figs[0]).name if a_figs else "missing"
        sel = Path((m.get("peak_tables") or {}).get("selected", "")).name
        fb = Path((m.get("feedback") or {}).get("files", {}).get("txt", "")).name
        lines.append(f"| {stem} | `{t_cell}` | `{a_cell}` | `{sel}` | `{fb or '—'}` |")
    lines.extend(["", "## Notes", ""])
    for m in manifests:
        note = m.get("transmittance_note") or ""
        if note:
            lines.append(f"- **{m.get('stem')}**: {note}")
    idx = out_dir / "PAPER_FIGURES_INDEX.md"
    idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "paper_figures_manifest.json").write_text(
        json.dumps(manifests, indent=2, default=str), encoding="utf-8"
    )
    return idx


def build_paper_figures_section_html(
    manifest: dict[str, Any],
    *,
    report_dir: Path,
) -> str:
    """Embed paper figures, peak tables, feedback, and download links for the product HTML report."""
    import html as html_mod

    if not manifest or not manifest.get("figures"):
        return ""
    stem = manifest.get("stem", "spectrum")
    base = Path(report_dir).resolve()

    def _rel(path: str) -> str:
        if not path:
            return ""
        try:
            return Path(path).resolve().relative_to(base).as_posix()
        except ValueError:
            return Path(path).name

    def _esc(s: str) -> str:
        return html_mod.escape(str(s), quote=True)

    blocks: list[str] = [
        "<section class='paper-figures-section card'>",
        "<h3>Publication figures</h3>",
        f"<p class='hint small'>Two-stage peak picking on normalized absorbance; "
        f"900–400 cm⁻¹ excluded from automatic labels. PNG ({DEFAULT_DPI} dpi), SVG, PDF.</p>",
    ]
    if manifest.get("transmittance_note"):
        blocks.append(
            f"<p class='hint small'><em>{_esc(manifest['transmittance_note'])}</em></p>"
        )

    for label, key in (
        ("Transmittance (%T)", "transmittance_peaks"),
        ("Normalized absorbance (0–1)", "normalized_absorbance_peaks"),
    ):
        paths = manifest.get("figures", {}).get(key) or []
        if not paths:
            continue
        blocks.append(f"<h4>{_esc(label)}</h4>")
        blocks.append(
            f"<p class='paper-dl'>{format_download_links(paths, base)}</p>"
        )

    sel_csv = (manifest.get("peak_tables") or {}).get("selected")
    if sel_csv:
        blocks.append("<h4>Selected peaks</h4>")
        blocks.append(_selected_peak_table_html(sel_csv, base))

    dl_labels = {
        "all_candidates": "All candidates CSV",
        "selected": "Selected peaks CSV",
        "suppressed": "Suppressed peaks CSV",
        "transmittance_minima": "Transmittance minima CSV",
        "manual": "Manual peaks CSV",
    }
    extra_links: list[tuple[str, str]] = []
    for key, label in dl_labels.items():
        path = (manifest.get("peak_tables") or {}).get(key)
        if path:
            extra_links.append((label, str(path)))
    ov_path = manifest.get("label_overrides_path") or ""
    if ov_path:
        extra_links.append(("Label overrides JSON", str(ov_path)))
    fb_files = (manifest.get("feedback") or {}).get("files") or {}
    for key, label in (("txt", "Feedback TXT"), ("md", "Feedback MD")):
        path = fb_files.get(key)
        if path:
            extra_links.append((label, str(path)))
    blocks.append(
        f"<p class='paper-dl'>{format_download_links([], base, extra=extra_links)}</p>"
    )

    feedback_html = manifest.get("feedback_html") or build_feedback_section_html(
        manifest.get("feedback") or {}, base
    )
    if feedback_html:
        blocks.append(feedback_html)
    blocks.append("</section>")
    return "".join(blocks)


def _selected_peak_table_html(csv_path: str, report_dir: Path) -> str:
    import csv as csv_mod
    import html as html_mod

    rows: list[dict[str, str]] = []
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        for row in csv_mod.DictReader(fh):
            rows.append(row)
    if not rows:
        return "<p class='muted'>No selected peaks.</p>"

    def esc(s: str) -> str:
        return html_mod.escape(str(s or ""), quote=True)

    try:
        rel = Path(csv_path).resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        rel = Path(csv_path).name

    hdr = (
        "<table class='tbl tbl-zebra small'><thead><tr>"
        "<th>Wavenumber (cm⁻¹)</th><th>Intensity</th><th>Prominence</th>"
        "<th>Region</th><th>Assignment</th></tr></thead><tbody>"
    )
    body = "".join(
        f"<tr><td>{esc(r.get('wavenumber_cm1',''))}</td>"
        f"<td>{esc(r.get('intensity',''))}</td>"
        f"<td>{esc(r.get('prominence',''))}</td>"
        f"<td>{esc(r.get('region',''))}</td>"
        f"<td>{esc(r.get('assignment_if_available','')) or '—'}</td></tr>"
        for r in rows[:12]
    )
    return hdr + body + f"</tbody></table><p><a href='{esc(rel)}' download>Full selected peaks CSV</a></p>"
