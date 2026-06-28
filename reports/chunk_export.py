"""Spectral chunk exports: singles, stacks, collages, and machine-readable chunk data."""

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

from lib.intensity_modes import ForceIntensityMode
from reports.discussion_regions import (
    DiscussionRegion,
    default_ranges_config,
    load_ranges_config,
    ranges_to_discussion_regions,
    save_ranges_config,
)
from reports.region_stack_export import (
    StackMode,
    StackSpectrum,
    _prepare_region_traces,
    _robust_span,
    _save_figure,
    spectra_from_batch,
)

ChunkMode = Literal["normalized_absorbance", "transmittance"]

# Pre-2026-06 region stack filenames (superseded by named ranges in discussion_regions.py).
_LEGACY_STACK_GLOBS = (
    "oh_nh_*",
    "ch_*",
    "co_aromatic_*",
    "ring_cn_*",
    "co_fp_*",
)
_LEGACY_STACK_FILES = (
    "region_stacks_manifest.json",
    "REGION_STACKS_INDEX.md",
)


def _cleanup_legacy_stack_artifacts(stacks_dir: Path) -> None:
    """Remove obsolete stack outputs from the pre-chunk naming scheme."""
    if not stacks_dir.is_dir():
        return
    for pattern in _LEGACY_STACK_GLOBS:
        for path in stacks_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)
    for name in _LEGACY_STACK_FILES:
        path = stacks_dir / name
        if path.is_file():
            path.unlink(missing_ok=True)


@dataclass
class ChunkExportConfig:
    stack_modes: tuple[ChunkMode, ...] = ("normalized_absorbance", "transmittance")
    formats: tuple[str, ...] = ("png", "svg", "pdf")
    offset_gap: float = 0.15
    region_labels: str = "selected"
    dpi: int = 300
    ignore_label_ranges: list[tuple[float, float]] = field(default_factory=lambda: [(400.0, 900.0)])
    show_peak_markers: bool = False
    export_chunk_data: bool = True
    export_collage: bool = True
    regions_file: Path | None = None
    regions: list[DiscussionRegion] | None = None
    allow_apparent_transmittance: bool = False
    force_intensity_mode: ForceIntensityMode | None = None
    apparent_transmittance_label: str = "Apparent Transmittance (%)"


def _ignore_bounds(ignore: list[tuple[float, float]] | None) -> tuple[float, float]:
    if ignore:
        lo, hi = ignore[0]
        return min(lo, hi), max(lo, hi)
    return 400.0, 900.0


def _annotate_peaks(
    ax: Any,
    wn: np.ndarray,
    y: np.ndarray,
    peak_wns: list[float],
    *,
    y_offset: float = 0.0,
    y_span: float,
    show_peak_markers: bool,
    label_side: str = "above",
) -> None:
    direction = 1.0 if label_side == "above" else -1.0
    for wn_p in peak_wns:
        if wn_p < float(np.min(wn)) or wn_p > float(np.max(wn)):
            continue
        y_at = float(np.interp(wn_p, wn, y)) + y_offset
        label_y = y_at + 0.08 * y_span * direction
        if show_peak_markers:
            ax.plot(wn_p, y_at, "o", color="#d95319", markersize=3.5, alpha=0.85, zorder=4)
        ax.annotate(
            f"{wn_p:.0f}",
            xy=(wn_p, y_at),
            xytext=(wn_p, label_y),
            textcoords="data",
            ha="center",
            va="bottom" if label_side == "above" else "top",
            fontsize=8,
            color="black",
            rotation=0,
            arrowprops=dict(arrowstyle="-", color="#d95319", lw=0.8),
            clip_on=True,
        )


def _draw_single_chunk(
    trace: Any,
    *,
    region: DiscussionRegion,
    mode: ChunkMode,
    show_labels: bool,
    show_peak_markers: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    color = trace.spec.color or "#0072bd"
    ax.plot(trace.wn, trace.y, color=color, linewidth=1.0)
    if show_labels:
        side = "below" if mode == "transmittance" else "above"
        _annotate_peaks(
            ax,
            trace.wn,
            trace.y,
            trace.peak_wns,
            y_span=trace.span,
            show_peak_markers=show_peak_markers,
            label_side=side,
        )
    ax.set_xlim(region.hi, region.lo)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=10)
    ylabel = "Transmittance (%T)" if mode == "transmittance" else "Normalized absorbance (0–1)"
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{trace.spec.label} — {region.title or region.name}", fontsize=10, pad=5)
    ax.grid(True, color="#e6e6e6", linewidth=0.5)
    fig.tight_layout()
    return fig


def _draw_region_stack(
    traces: list[Any],
    *,
    region: DiscussionRegion,
    mode: ChunkMode,
    show_peak_labels: bool,
    show_peak_markers: bool,
) -> plt.Figure:
    n = len(traces)
    fig_h = 1.6 + 0.5 * n + (0.3 if show_peak_labels else 0)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    wn_span = region.hi - region.lo
    label_x = region.hi - 0.02 * wn_span

    for i, tr in enumerate(traces):
        color = tr.spec.color or "#0072bd"
        y_plot = tr.y + tr.offset
        ax.plot(tr.wn, y_plot, color=color, linewidth=1.0)
        ax.text(
            label_x,
            tr.offset + 0.5 * tr.span,
            tr.spec.label,
            fontsize=9,
            va="center",
            ha="right",
            color=color,
            fontweight="semibold",
        )
        if show_peak_labels:
            side = "below" if mode == "transmittance" else "above"
            _annotate_peaks(
                ax,
                tr.wn,
                tr.y,
                tr.peak_wns,
                y_offset=tr.offset,
                y_span=tr.span,
                show_peak_markers=show_peak_markers,
                label_side=side,
            )

    ax.set_xlim(region.hi, region.lo)
    ax.set_yticks([])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=10)
    ax.set_title(region.title or region.name, fontsize=11, pad=6)
    if mode == "transmittance":
        ax.set_ylabel("Transmittance (%T, offset)", fontsize=10)
    else:
        ax.set_ylabel("Normalized absorbance (offset)", fontsize=10)
    ax.grid(False)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.14)
    return fig


def _draw_collage(
    traces_by_region: list[tuple[DiscussionRegion, list[Any]]],
    *,
    mode: ChunkMode,
    show_peak_labels: bool,
    show_peak_markers: bool,
) -> plt.Figure | None:
    panels = [(r, t) for r, t in traces_by_region if t]
    if not panels:
        return None
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    for col, (region, traces) in enumerate(panels):
        ax = axes[0, col]
        if len(traces) == 1:
            tr = traces[0]
            ax.plot(tr.wn, tr.y, color=tr.spec.color or "#0072bd", linewidth=1.0)
            if show_peak_labels:
                _annotate_peaks(
                    ax,
                    tr.wn,
                    tr.y,
                    tr.peak_wns,
                    y_span=tr.span,
                    show_peak_markers=show_peak_markers,
                    label_side="below" if mode == "transmittance" else "above",
                )
        else:
            for i, tr in enumerate(traces):
                color = tr.spec.color or "#0072bd"
                y_plot = tr.y + tr.offset
                ax.plot(tr.wn, y_plot, color=color, linewidth=0.9)
                if show_peak_labels:
                    _annotate_peaks(
                        ax,
                        tr.wn,
                        tr.y,
                        tr.peak_wns,
                        y_offset=tr.offset,
                        y_span=tr.span,
                        show_peak_markers=show_peak_markers,
                        label_side="below" if mode == "transmittance" else "above",
                    )
        ax.set_xlim(region.hi, region.lo)
        ax.set_title(region.title or region.name, fontsize=9)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel("%T" if mode == "transmittance" else "Abs (0–1)", fontsize=9)
    fig.suptitle(f"Spectral chunks — {mode.replace('_', ' ')}", fontsize=11)
    fig.tight_layout()
    return fig


def _chunk_data_record(
    region: DiscussionRegion,
    traces: list[Any],
    *,
    mode: ChunkMode,
) -> dict[str, Any]:
    spectra: list[dict[str, Any]] = []
    for tr in traces:
        peaks = [
            {"wavenumber_cm1": wn, "label_text": f"{wn:.0f}"}
            for wn in tr.peak_wns
        ]
        spectra.append(
            {
                "label": tr.spec.label,
                "stem": tr.spec.stem,
                "wavenumber_cm1": [float(x) for x in tr.wn.tolist()],
                "intensity": [float(x) for x in tr.y.tolist()],
                "offset": float(tr.offset),
                "color": tr.spec.color or "",
                "selected_peaks": peaks,
            }
        )
    return {
        "range_name": region.name,
        "wn_min": region.lo,
        "wn_max": region.hi,
        "mode": mode,
        "spectra": spectra,
    }


def _write_chunk_data_files(
    stacks_dir: Path,
    region: DiscussionRegion,
    traces: list[Any],
    *,
    mode: ChunkMode,
) -> dict[str, str]:
    record = _chunk_data_record(region, traces, mode=mode)
    json_path = stacks_dir / f"{region.name}_chunk_data.json"
    csv_path = stacks_dir / f"{region.name}_chunk_data.csv"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "label", "wavenumber_cm1", "intensity", "offset", "peak_wn", "peak_label"])
        for spec in record["spectra"]:
            peak_map = {p["wavenumber_cm1"]: p["label_text"] for p in spec["selected_peaks"]}
            for wn_v, int_v in zip(spec["wavenumber_cm1"], spec["intensity"]):
                pk = ""
                for pwn in peak_map:
                    if abs(pwn - wn_v) < 0.05:
                        pk = peak_map[pwn]
                        break
                w.writerow(
                    [
                        spec["stem"],
                        spec["label"],
                        f"{wn_v:.4f}",
                        f"{int_v:.6f}",
                        spec["offset"],
                        "",
                        pk,
                    ]
                )
            for p in spec["selected_peaks"]:
                w.writerow(
                    [
                        spec["stem"],
                        spec["label"],
                        f"{p['wavenumber_cm1']:.4f}",
                        "",
                        spec["offset"],
                        f"{p['wavenumber_cm1']:.1f}",
                        p["label_text"],
                    ]
                )
    return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve())}


def export_spectral_chunks(
    *,
    spectra: list[StackSpectrum],
    out_dir: Path,
    config: ChunkExportConfig | None = None,
) -> dict[str, Any]:
    """Export singles, stacks, collages, chunk data, and ranges_config.json."""
    cfg = config or ChunkExportConfig()
    out_dir = Path(out_dir)
    stacks_dir = out_dir / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_stack_artifacts(stacks_dir)

    ranges_payload = load_ranges_config(cfg.regions_file)
    if not cfg.regions_file or not Path(cfg.regions_file).is_file():
        ranges_payload = default_ranges_config()
    regions = cfg.regions or ranges_to_discussion_regions(ranges_payload)
    ignore_lo, ignore_hi = _ignore_bounds(cfg.ignore_label_ranges)

    ranges_path = stacks_dir / "ranges_config.json"
    save_ranges_config(ranges_path, ranges_payload)

    show_labels = str(cfg.region_labels).lower() == "selected"
    manifest: dict[str, Any] = {
        "stacks_dir": str(stacks_dir.resolve()),
        "ranges_config_path": str(ranges_path.resolve()),
        "outputs": {},
        "chunk_data": {},
        "collage": {},
    }

    collage_inputs: dict[str, list[tuple[DiscussionRegion, list[Any]]]] = {
        m: [] for m in cfg.stack_modes
    }

    for region in regions:
        if not getattr(region, "show_in_stacks", True):
            continue
        for mode in cfg.stack_modes:
            traces = _prepare_region_traces(
                spectra,
                region=region,
                mode=mode,  # type: ignore[arg-type]
                offset_gap=cfg.offset_gap,
                ignore_lo=ignore_lo,
                ignore_hi=ignore_hi,
                allow_apparent_transmittance=cfg.allow_apparent_transmittance,
                force_intensity_mode=cfg.force_intensity_mode,
            )
            if not traces:
                continue

            stack_fig = _draw_region_stack(
                traces,
                region=region,
                mode=mode,  # type: ignore[arg-type]
                show_peak_labels=show_labels and mode == "normalized_absorbance",
                show_peak_markers=cfg.show_peak_markers,
            )
            stack_base = stacks_dir / f"{region.name}_{mode}_stack"
            stack_paths = _save_figure(stack_fig, stack_base, cfg.formats, cfg.dpi)
            manifest["outputs"][f"{region.name}_{mode}_stack"] = stack_paths

            if len(spectra) == 1:
                single_fig = _draw_single_chunk(
                    traces[0],
                    region=region,
                    mode=mode,  # type: ignore[arg-type]
                    show_labels=show_labels,
                    show_peak_markers=cfg.show_peak_markers,
                )
                single_base = stacks_dir / f"{region.name}_single_{mode}"
                single_paths = _save_figure(single_fig, single_base, cfg.formats, cfg.dpi)
                manifest["outputs"][f"{region.name}_single_{mode}"] = single_paths

            if cfg.export_chunk_data:
                manifest["chunk_data"][f"{region.name}_{mode}"] = _write_chunk_data_files(
                    stacks_dir, region, traces, mode=mode  # type: ignore[arg-type]
                )

            collage_inputs[mode].append((region, traces))

    if cfg.export_collage:
        for mode, panels in collage_inputs.items():
            fig = _draw_collage(
                panels,
                mode=mode,  # type: ignore[arg-type]
                show_peak_labels=show_labels,
                show_peak_markers=cfg.show_peak_markers,
            )
            if fig is None:
                continue
            base = stacks_dir / f"ranges_collage_{mode}"
            paths = _save_figure(fig, base, cfg.formats, cfg.dpi)
            manifest["collage"][mode] = paths
            manifest["outputs"][f"ranges_collage_{mode}"] = paths

    index_path = stacks_dir / "CHUNKS_INDEX.md"
    lines = ["# Spectral chunk exports", "", f"Ranges: `{ranges_path}`", ""]
    for key, paths in manifest.get("outputs", {}).items():
        lines.append(f"## {key}")
        for p in paths:
            lines.append(f"- `{p}`")
        lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    manifest["index"] = str(index_path.resolve())
    (stacks_dir / "chunks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "ChunkExportConfig",
    "export_spectral_chunks",
    "spectra_from_batch",
]
