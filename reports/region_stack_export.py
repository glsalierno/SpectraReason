"""Region-specific offset stack exports for manuscript discussion windows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from lib.ftir_foundation import preprocess_spectrum, read_spectrum
from reports.discussion_regions import DiscussionRegion, load_discussion_regions
from lib.intensity_modes import (
    ForceIntensityMode,
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
)

StackMode = Literal["normalized_absorbance", "transmittance"]
TRACE_COLORS = ("#0072bd", "#d95319", "#77ac30", "#7e2f8e", "#a2142f", "#4dbeee")


@dataclass
class StackSpectrum:
    stem: str
    label: str
    path: Path
    color: str = ""
    selected_peaks_csv: Path | None = None


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(name).stem)


def _robust_span(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 1.0
    p1, p99 = np.percentile(y, [1, 99])
    return max(float(p99 - p1), 1e-6)


def _load_selected_peaks_in_region(
    csv_path: Path | None,
    *,
    wn_min: float,
    wn_max: float,
) -> list[float]:
    if not csv_path or not Path(csv_path).is_file():
        return []
    peaks: list[float] = []
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                wn = float(row.get("wavenumber_cm1", 0))
            except (TypeError, ValueError):
                continue
            if wn_min <= wn <= wn_max:
                peaks.append(wn)
    return peaks


@dataclass
class PreparedRegionTrace:
    spec: StackSpectrum
    wn: np.ndarray
    y: np.ndarray
    offset: float
    span: float
    peak_wns: list[float]


def _prepare_region_traces(
    spectra: list[StackSpectrum],
    *,
    region: DiscussionRegion,
    mode: StackMode,
    offset_gap: float,
    ignore_lo: float = 400.0,
    ignore_hi: float = 900.0,
    allow_apparent_transmittance: bool = False,
    force_intensity_mode: ForceIntensityMode | None = None,
) -> list[PreparedRegionTrace]:
    out: list[PreparedRegionTrace] = []
    cumulative = 0.0
    for i, spec in enumerate(spectra):
        wn, raw, hint = read_spectrum(spec.path)
        classification = classify_intensity(
            spec.path, hint, raw, force_mode=force_intensity_mode
        )
        wn_min, wn_max = region.lo, region.hi

        if mode == "transmittance":
            t_plan = plan_transmittance_panel(
                raw,
                classification,
                allow_apparent=allow_apparent_transmittance,
            )
            if not has_transmittance_panel(t_plan):
                continue
            y_full = _percent_transmittance_from_plan(t_plan.y_values)  # type: ignore[arg-type]
            m = _window_mask(wn, wn_min=wn_min, wn_max=wn_max)
            baseline = apparent_transmittance_baseline_pct(t_plan)
            y_full = prepare_transmittance_trace(
                _smooth_percent_t(wn[m], y_full[m]),
                baseline_pct=baseline,
            )
            wn_v, y_v = wn[m], y_full
        else:
            wn_v, y_v, _ = preprocess_spectrum(
                wn,
                raw,
                intensity_mode=classification.preprocess_mode,
                normalize=True,
            )
            m = _window_mask(wn_v, wn_min=wn_min, wn_max=wn_max)
            wn_v, y_v = wn_v[m], y_v[m]

        if wn_v.size < 3:
            continue
        span = _robust_span(y_v)
        offset = cumulative
        cumulative += span * (1.0 + offset_gap)
        peaks = _load_selected_peaks_in_region(
            spec.selected_peaks_csv, wn_min=wn_min, wn_max=wn_max
        )
        peaks = [p for p in peaks if not (ignore_lo <= p <= ignore_hi)]
        out.append(
            PreparedRegionTrace(
                spec=spec,
                wn=wn_v,
                y=y_v,
                offset=offset,
                span=span,
                peak_wns=peaks,
            )
        )
    return out


def _save_figure(fig: plt.Figure, base: Path, formats: tuple[str, ...], dpi: int) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        f = fmt.lower().lstrip(".")
        out = base.with_suffix(f".{f}")
        kw: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if f == "png":
            kw["dpi"] = dpi
        fig.savefig(out, format=f, **kw)
        paths.append(str(out.resolve()))
    plt.close(fig)
    return paths


def export_region_stacks(
    *,
    spectra: list[StackSpectrum],
    out_dir: Path,
    regions: list[DiscussionRegion] | None = None,
    regions_file: Path | None = None,
    stack_modes: tuple[StackMode, ...] = ("normalized_absorbance", "transmittance"),
    formats: tuple[str, ...] = ("png", "svg", "pdf"),
    offset_gap: float = 0.15,
    region_labels: str = "selected",
    dpi: int = 300,
    ignore_label_ranges: list[tuple[float, float]] | None = None,
    show_peak_markers: bool = False,
    export_chunk_data: bool = True,
    export_collage: bool = True,
    allow_apparent_transmittance: bool = False,
    force_intensity_mode: ForceIntensityMode | None = None,
    apparent_transmittance_label: str = "Apparent Transmittance (%)",
) -> dict[str, Any]:
    """Backward-compatible wrapper around full spectral chunk export."""
    from reports.chunk_export import ChunkExportConfig, export_spectral_chunks

    cfg = ChunkExportConfig(
        stack_modes=stack_modes,  # type: ignore[arg-type]
        formats=formats,
        offset_gap=offset_gap,
        region_labels=region_labels,
        dpi=dpi,
        ignore_label_ranges=list(ignore_label_ranges or [(400.0, 900.0)]),
        show_peak_markers=show_peak_markers,
        export_chunk_data=export_chunk_data,
        export_collage=export_collage,
        regions_file=regions_file,
        regions=regions,
        allow_apparent_transmittance=allow_apparent_transmittance,
        force_intensity_mode=force_intensity_mode,
        apparent_transmittance_label=apparent_transmittance_label,
    )
    return export_spectral_chunks(spectra=spectra, out_dir=out_dir, config=cfg)


def build_region_stacks_section_html(manifest: dict[str, Any], report_dir: Path) -> str:
    from reports.range_editor import build_range_editor_section_html

    stacks_dir = Path(manifest.get("stacks_dir", ""))
    if not stacks_dir.is_dir():
        return ""

    ranges_path = manifest.get("ranges_config_path")
    ranges_payload = None
    if ranges_path and Path(ranges_path).is_file():
        import json as _json

        ranges_payload = _json.loads(Path(ranges_path).read_text(encoding="utf-8"))

    return build_range_editor_section_html(
        editor_id="ftir-range-editor",
        ranges_payload=ranges_payload,
        chunks_manifest=manifest,
        report_dir=report_dir,
    )


def spectra_from_batch(
    input_paths: list[Path],
    *,
    paper_out_dir: Path | None = None,
) -> list[StackSpectrum]:
    out: list[StackSpectrum] = []
    for i, p in enumerate(input_paths):
        stem = _safe_stem(p.name)
        sel_csv = None
        if paper_out_dir:
            candidate = Path(paper_out_dir) / f"{stem}_peaks_selected.csv"
            if candidate.is_file():
                sel_csv = candidate
        out.append(
            StackSpectrum(
                stem=stem,
                label=stem.replace("_", " "),
                path=Path(p),
                color=TRACE_COLORS[i % len(TRACE_COLORS)],
                selected_peaks_csv=sel_csv,
            )
        )
    return out
