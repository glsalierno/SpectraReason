"""
MATLAB-style visual themes, static figure export, and MATLAB postprocessing bundles.

Visualization-only: does not alter evidence, rules, or assignments.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

MARKER_MATLAB_THEME = "<!-- report-feature:matlab-visual-theme -->"

VISUAL_THEMES = ("default", "matlab", "dark")

MATLAB_EXTRA_CSS = """
body.visual-matlab { background: #fff; color: #222; }
body.visual-matlab #sidebar { background: #fafafa; border-color: #ddd; }
body.visual-matlab .card { border-color: #ddd; border-radius: 4px; box-shadow: none; padding: 12px 14px 16px; margin: 12px 0 18px; }
body.visual-matlab .product-interpretation { padding: 10px 12px; margin: 8px 0; border-radius: 4px; box-shadow: none; }
body.visual-matlab .key-evidence-section h3 { font-size: 0.95rem; font-weight: 600; }
body.visual-matlab .plot-wrap { margin-top: 4px; }
body.visual-matlab .product-hint, body.visual-matlab .hint { font-size: 0.82rem; margin: 4px 0 8px; }
body.visual-matlab .tbl th { background: #f5f5f5; }
body.visual-matlab h1 { font-size: 1.2rem; font-weight: 600; }
body.visual-matlab .muted { color: #555; }
body.visual-matlab.front-audience .card { border: 1px solid #e0e0e0; }
"""

DARK_EXTRA_CSS = """
body.visual-dark { background: #1a1a1a; color: #e8e8e8; }
body.visual-dark #sidebar { background: #252525; border-color: #404040; }
body.visual-dark .card { background: #222; border-color: #404040; box-shadow: none; }
body.visual-dark .muted, body.visual-dark .hint { color: #aaa; }
body.visual-dark .tbl th { background: #333; }
body.visual-dark .tbl th, body.visual-dark .tbl td { border-color: #444; }
"""


def normalize_visual_theme(theme: str | None) -> str:
    t = str(theme or "default").lower()
    return t if t in VISUAL_THEMES else "default"


def get_plotly_theme(visual_theme: str) -> dict[str, Any]:
    """Plotly trace/layout styling per visual theme."""
    t = normalize_visual_theme(visual_theme)
    if t == "matlab":
        return {
            "spectrum_line": "#0072bd",
            "spectrum_width": 1.2,
            "peak_strong": "#d95319",
            "peak_moderate": "#77ac30",
            "peak_weak": "#a2a2a2",
            "peak_label_color": "#333333",
            "peak_label_font": 9,
            "peak_marker_line": "#ffffff",
            "kron_fill": "#7eb3d4",
            "kron_line": "#4a7fa5",
            "kron_line_width": 0.2,
            "shade_colors": (
                "rgba(0,114,189,0.10)",
                "rgba(119,172,48,0.10)",
                "rgba(217,83,25,0.08)",
                "rgba(148,148,148,0.08)",
                "rgba(0,114,189,0.06)",
            ),
            "paper_bg": "#ffffff",
            "plot_bg": "#ffffff",
            "grid_color": "#e6e6e6",
            "axis_color": "#333333",
            "spike_color": "#999999",
            "font_family": "Arial, Helvetica, sans-serif",
            "hover_bg": "#ffffff",
            "hover_border": "#cccccc",
            "legend_y": 1.01,
            "title_font_size": 13,
        }
    if t == "dark":
        return {
            "spectrum_line": "#5dade2",
            "spectrum_width": 1.2,
            "peak_strong": "#f39c12",
            "peak_moderate": "#2ecc71",
            "peak_weak": "#7f8c8d",
            "peak_label_color": "#ecf0f1",
            "peak_label_font": 9,
            "peak_marker_line": "#1a1a1a",
            "kron_fill": "#3498db",
            "kron_line": "#2980b9",
            "kron_line_width": 0.25,
            "shade_colors": (
                "rgba(93,173,226,0.15)",
                "rgba(46,204,113,0.12)",
                "rgba(243,156,18,0.10)",
                "rgba(127,140,141,0.10)",
                "rgba(93,173,226,0.08)",
            ),
            "paper_bg": "#1a1a1a",
            "plot_bg": "#252525",
            "grid_color": "#404040",
            "axis_color": "#cccccc",
            "spike_color": "#666666",
            "font_family": "Arial, Helvetica, sans-serif",
            "hover_bg": "#2a2a2a",
            "hover_border": "#555555",
            "legend_y": 1.01,
            "title_font_size": 13,
        }
    return {
        "spectrum_line": "#1d4ed8",
        "spectrum_width": 1.1,
        "peak_strong": None,
        "peak_moderate": None,
        "peak_weak": None,
        "peak_label_color": "#334155",
        "peak_label_font": 9,
        "peak_marker_line": "white",
        "kron_fill": None,
        "kron_line": None,
        "kron_line_width": None,
        "shade_colors": None,
        "paper_bg": None,
        "plot_bg": None,
        "grid_color": None,
        "axis_color": None,
        "spike_color": "#94a3b8",
        "font_family": None,
        "hover_bg": "#ffffff",
        "hover_border": "#cbd5e1",
        "legend_y": 1.02,
        "title_font_size": None,
    }


def peak_color_for_theme(visual_theme: str, quality: str) -> str | None:
    cfg = get_plotly_theme(visual_theme)
    key = f"peak_{quality}"
    return cfg.get(key)


def body_class_for_theme(visual_theme: str) -> str:
    t = normalize_visual_theme(visual_theme)
    if t == "matlab":
        return "visual-matlab"
    if t == "dark":
        return "visual-dark"
    return ""


def extra_css_for_theme(visual_theme: str) -> str:
    t = normalize_visual_theme(visual_theme)
    if t == "matlab":
        return MATLAB_EXTRA_CSS
    if t == "dark":
        return DARK_EXTRA_CSS
    return ""


def apply_plotly_theme_to_figure(
    fig: Any,
    visual_theme: str,
    *,
    n_rows: int = 2,
    spectrum_row: int = 1,
    kron_row: int = 2,
) -> None:
    """Apply publication-style layout after traces are added."""
    t = normalize_visual_theme(visual_theme)
    if t == "default":
        return
    cfg = get_plotly_theme(t)
    font = dict(family=cfg["font_family"], size=11, color=cfg["axis_color"])
    title_font = dict(
        family=cfg["font_family"],
        size=cfg.get("title_font_size") or 12,
        color=cfg["axis_color"],
    )
    fig.update_layout(
        paper_bgcolor=cfg["paper_bg"],
        plot_bgcolor=cfg["plot_bg"],
        font=font,
        hoverlabel=dict(
            bgcolor=cfg["hover_bg"],
            font_size=11,
            font_family=cfg["font_family"],
            bordercolor=cfg["hover_border"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=cfg["legend_y"],
            x=0,
            font=dict(size=10, color=cfg["axis_color"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    for r in range(1, n_rows + 1):
        fig.update_xaxes(
            gridcolor=cfg["grid_color"],
            gridwidth=0.5,
            linecolor=cfg["axis_color"],
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickfont=font,
            title_font=title_font,
            row=r,
            col=1,
        )
        fig.update_yaxes(
            gridcolor=cfg["grid_color"],
            gridwidth=0.5,
            linecolor=cfg["axis_color"],
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickfont=font,
            title_font=title_font,
            row=r,
            col=1,
        )
    if cfg.get("spike_color"):
        fig.update_xaxes(
            spikecolor=cfg["spike_color"],
            row=spectrum_row,
            col=1,
        )
        fig.update_xaxes(
            spikecolor=cfg["spike_color"],
            row=kron_row,
            col=1,
        )


def kaleido_available() -> bool:
    try:
        import kaleido  # noqa: F401

        return True
    except ImportError:
        return False


def _safe_stem(name: str) -> str:
    base = Path(name).stem if name else "spectrum"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in base)


def export_plotly_image(
    fig: Any,
    path: Path,
    *,
    fmt: str,
    dpi: int,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), format=fmt, scale=max(1, dpi) / 96.0)
        return True
    except Exception:
        return False


def write_presentation_figures_index(
    presentation_dir: Path,
    *,
    figure_files: list[str],
    report_html: Path,
    notes_template: Path | None = None,
) -> Path:
    """Write FIGURES_INDEX.md with absolute paths for slides/manuscripts."""
    presentation_dir = presentation_dir.resolve()
    presentation_dir.mkdir(parents=True, exist_ok=True)
    index_path = presentation_dir / "FIGURES_INDEX.md"
    lines = [
        "# Presentation figures\n",
        f"Interactive report: `{report_html.resolve()}`\n",
        "\n## Exported files\n",
    ]
    if figure_files:
        for f in sorted(figure_files):
            lines.append(f"- `{Path(f).resolve()}`\n")
    else:
        lines.append(
            "- _(No raster/vector files — install Kaleido: `pip install kaleido`, "
            "or use Plotly JSON under `figures/plotly_json/`.)_\n"
        )
    lines.append(
        "\n## Edit interpretation text\n"
        "- In the HTML report: click the blue **summary** paragraph (saved in your browser).\n"
    )
    if notes_template and notes_template.is_file():
        lines.append(
            f"- Or edit `{notes_template.resolve()}` and re-run with `--interpretation-notes`.\n"
        )
    index_path.write_text("".join(lines), encoding="utf-8")
    return index_path


def export_static_figure_bundle(
    *,
    spectrum_name: str,
    fig_combined: Any,
    fig_spectrum: Any | None,
    fig_kronecker: Any | None,
    static_out_dir: Path,
    static_format: str,
    static_dpi: int,
) -> dict[str, Any]:
    """Export PNG/SVG/PDF via Kaleido; fallback to JSON + note."""
    stem = _safe_stem(spectrum_name)
    fmt = str(static_format or "png").lower().lstrip(".")
    out_dir = static_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "spectrum": stem,
        "format": fmt,
        "dpi": static_dpi,
        "kaleido": kaleido_available(),
        "files": [],
        "note": "",
    }
    targets: list[tuple[str, Any | None]] = [
        (f"{stem}_combined", fig_combined),
        (f"{stem}_spectrum", fig_spectrum),
        (f"{stem}_kronecker", fig_kronecker),
    ]
    if result["kaleido"]:
        for base, ffig in targets:
            if ffig is None:
                continue
            path = out_dir / f"{base}.{fmt}"
            if export_plotly_image(ffig, path, fmt=fmt, dpi=static_dpi):
                result["files"].append(str(path))
        if not result["files"]:
            result["note"] = "Kaleido present but image export failed."
    else:
        note_path = out_dir / "STATIC_EXPORT_README.txt"
        json_dir = out_dir / "plotly_json"
        json_dir.mkdir(parents=True, exist_ok=True)
        for base, ffig in targets:
            if ffig is None:
                continue
            jp = json_dir / f"{base}.json"
            try:
                ffig.write_json(str(jp))
                result["files"].append(str(jp))
            except Exception:
                pass
        result["note"] = (
            "Static raster/vector export requires Kaleido (pip install kaleido). "
            "Plotly JSON and MATLAB CSV exports were written for MATLAB postprocessing."
        )
        note_path.write_text(result["note"] + "\n", encoding="utf-8")
        result["files"].append(str(note_path))
    return result


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _status_display(ent: dict[str, Any], pipeline: dict[str, Any], lab: str) -> str:
    from reports.product_v1_report import _status_display as _sd

    return _sd(ent, pipeline, lab)


def export_matlab_spectrum_csvs(
    *,
    matlab_dir: Path,
    spectrum_name: str,
    wn: np.ndarray,
    y: np.ndarray,
    peaks_labeled: list[dict[str, Any]],
    peaks_plotted: list[dict[str, Any]],
    pipeline: dict[str, Any],
    fig_meta: dict[str, Any],
    include_evidence: bool = True,
) -> list[Path]:
    """Write per-spectrum MATLAB-ready CSV bundle."""
    stem = _safe_stem(spectrum_name)
    matlab_dir = matlab_dir.resolve()
    matlab_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    spec_path = matlab_dir / f"{stem}_spectrum.csv"
    _write_csv(
        spec_path,
        ["wavenumber_cm1", "absorbance"],
        [
            {"wavenumber_cm1": f"{float(x):.4f}", "absorbance": f"{float(v):.6f}"}
            for x, v in zip(wn, y)
        ],
    )
    written.append(spec_path)

    peaks_rows: list[dict[str, Any]] = []
    for p in peaks_plotted or peaks_labeled:
        peaks_rows.append(
            {
                "peak_position_cm1": f"{float(p.get('wn_cm1', 0)):.2f}",
                "peak_height": f"{float(p.get('height', 0)):.6f}",
                "peak_quality": str(p.get("peak_quality", "")),
                "peak_prominence": f"{float(p.get('prominence', 0) or 0):.6f}",
                "is_labeled": int(
                    any(
                        abs(float(p.get("wn_cm1", 0)) - float(q.get("wn_cm1", -1))) < 0.5
                        for q in (peaks_labeled or [])
                    )
                ),
            }
        )
    peaks_path = matlab_dir / f"{stem}_peaks.csv"
    _write_csv(
        peaks_path,
        [
            "peak_position_cm1",
            "peak_height",
            "peak_quality",
            "peak_prominence",
            "is_labeled",
        ],
        peaks_rows,
    )
    written.append(peaks_path)

    from ml.ftir_region_ruler import FTIR_RULER_REGIONS
    from ml.report_suppression import nitro_reporting_suppressed, ruler_hover_note

    suppress_nitro = nitro_reporting_suppressed(pipeline)
    ruler_rows: list[dict[str, Any]] = []
    act_map = {a["id"]: a for a in (fig_meta.get("ruler_activities") or [])}
    for spec in FTIR_RULER_REGIONS:
        act = act_map.get(spec.id, {})
        ruler_rows.append(
            {
                "region_id": spec.id,
                "region_label": spec.short_label,
                "lo_cm1": spec.lo,
                "hi_cm1": spec.hi,
                "rel_activity": act.get("rel_activity", ""),
                "tier": act.get("tier", ""),
                "hover_note": ruler_hover_note(spec, suppress_nitro=suppress_nitro),
            }
        )
    ruler_path = matlab_dir / f"{stem}_ruler_regions.csv"
    _write_csv(
        ruler_path,
        ["region_id", "region_label", "lo_cm1", "hi_cm1", "rel_activity", "tier", "hover_note"],
        ruler_rows,
    )
    written.append(ruler_path)

    key_rows: list[dict[str, Any]] = []
    if include_evidence:
        from reports.v4_evidence_report import evidence_ranked_assignments
        from reports.product_v1_report import chemistry_label

        ranked = evidence_ranked_assignments(pipeline, top_n=12)
        for lab, ent in ranked:
            bands = ent.get("supporting_bands") or ent.get("matched_bands") or []
            band_parts: list[str] = []
            for b in bands[:6]:
                if isinstance(b, dict) and b.get("wn_cm1"):
                    band_parts.append(f"{float(b['wn_cm1']):.0f}")
                elif isinstance(b, (int, float)):
                    band_parts.append(f"{float(b):.0f}")
            key_rows.append(
                {
                    "assignment_label": chemistry_label(lab, pipeline, ent),
                    "key_bands_cm1": ", ".join(band_parts),
                    "confidence_status": _status_display(ent, pipeline, lab),
                    "ontology_category": str(ent.get("ontology_category") or ""),
                    "score": f"{float(ent.get('score', 0) or 0):.4f}",
                    "notes": str(ent.get("competing_explanation") or ent.get("rationale") or "")[:300],
                }
            )
    key_path = matlab_dir / f"{stem}_key_evidence.csv"
    _write_csv(
        key_path,
        [
            "assignment_label",
            "key_bands_cm1",
            "confidence_status",
            "ontology_category",
            "score",
            "notes",
        ],
        key_rows,
    )
    written.append(key_path)
    if key_rows and include_evidence:
        from reports.product_v1_report import build_key_evidence_table_html

        key_html_path = matlab_dir / f"{stem}_key_evidence.html"
        key_html_path.write_text(
            "<!doctype html><html><body>"
            + build_key_evidence_table_html(pipeline, anchor=stem)
            + "</body></html>",
            encoding="utf-8",
        )
        written.append(key_html_path)

    ann_rows: list[dict[str, Any]] = []
    evidence = pipeline.get("evidence") or {}
    if str(pipeline.get("ontology") or "").lower() == "v4" and peaks_labeled and evidence:
        from reports.v4_evidence_report import peak_annotation_specs

        ann, _layout = peak_annotation_specs(
            peaks_labeled, evidence, max_peaks=48, include_weak=True
        )
        for a in ann:
            ann_rows.append(
                {
                    "peak_position_cm1": f"{float(a.get('wn', 0)):.2f}",
                    "peak_height": f"{float(a.get('y', 0)):.6f}",
                    "annotation_text": f"{float(a.get('wn', 0)):.0f}",
                    "category": str(a.get("category", "")),
                    "hover_text": str(a.get("hover", ""))[:500],
                }
            )
    ann_path = matlab_dir / f"{stem}_annotations.csv"
    _write_csv(
        ann_path,
        ["peak_position_cm1", "peak_height", "annotation_text", "category", "hover_text"],
        ann_rows,
    )
    written.append(ann_path)

    return written


def write_make_figures_m(matlab_dir: Path, spectrum_stems: list[str]) -> Path:
    """Generate make_figures.m for batch MATLAB postprocessing."""
    matlab_dir = matlab_dir.resolve()
    matlab_dir.mkdir(parents=True, exist_ok=True)
    # Use char literals ('stem') — double-quoted strings break warning/fullfile formatting.
    stems_literal = ", ".join(f"'{s}'" for s in spectrum_stems)
    script = textwrap.dedent(
        f"""\
        function make_figures()
        % Auto-generated FTIR report figures (visualization only; assignments unchanged).
        % Requires CSV exports in the same folder as this file.
        % Run: cd('.../matlab_export'); make_figures

        showPeakLabels = true;
        labelAllLabeledPeaks = true;  % all is_labeled==1 rows in peaks CSV (matches HTML report)
        maxPeakLabels = 48;           % cap when labelAllLabeledPeaks is false
        showSeparatePanels = true;    % region guide + spectrum peaks as separate PNGs (recommended)
        showStackedFigure = false;    % optional legacy stacked spectrum+Kronecker
        showRulerOverlay = false;     % ruler bands on spectrum (off when using separate region guide)
        showKronecker = false;
        reverseX = true;
        outputFormat = 'png';
        exportDpi = 300;
        closeFiguresAfterExport = false;  % false = keep figures open for resize/review in MATLAB
        bringFiguresToFront = true;       % focus each new figure when kept open

        % --- User-tuned typography (edit here; preserved across report regen) ---
        fontRegionBand = 8;
        fontRegionAxis = 10;
        fontSpectrumPeakLabel = 14;
        fontSpectrumAxis = 18;
        fontCombinedPeakLabel = 8;
        fontCombinedAxis = 10;
        fontCombinedRulerOverlay = 18;
        peakLabelRotation = -90;          % vertical wavenumber labels (degrees)
        peakLabelCollisionAware = true;   % stagger vertically when labels overlap

        dataDir = char(fileparts(mfilename('fullpath')));
        outDir = char(fullfile(dataDir, '..', 'presentation', 'figures'));
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end

        spectrumList = {{{stems_literal}}};

        for si = 1:numel(spectrumList)
            stem = normalizeStem(spectrumList{{si}});
            if showSeparatePanels
                plotRegionGuide(stem, dataDir, outDir);
                plotSpectrumPeaks(stem, dataDir, outDir);
            end
            if showStackedFigure
                plotOneSpectrum(stem, dataDir, outDir);
            end
        end

        function stemChar = normalizeStem(stem)
            if iscell(stem)
                stem = stem{{1}};
            end
            stemChar = char(strtrim(string(stem)));
            if isempty(stemChar)
                stemChar = 'spectrum';
            end
        end

        function csvPath = csvPathFor(dataDir, stem, suffix)
            csvPath = char(fullfile(dataDir, [stem suffix]));
        end

        function warnMissing(msg, pathChar)
            warning('FTIR:make_figures:MissingFile', '%s: %s', msg, pathChar);
        end

        function setWavenumberXLim(ax, wn)
            % xlim requires increasing limits; use XDir reverse for IR high-to-low display.
            xlim(ax, [min(wn) max(wn)]);
            if reverseX
                set(ax, 'XDir', 'reverse');
            end
        end

        function exportFigure(figOrLayout, outFile)
            outFile = char(outFile);
            outFolder = fileparts(outFile);
            if ~isempty(outFolder) && ~exist(outFolder, 'dir')
                mkdir(outFolder);
            end
            figHandle = ancestor(figOrLayout, 'figure');
            if isempty(figHandle) || ~ishandle(figHandle)
                figHandle = gcf;
            end
            try
                exportgraphics(figOrLayout, outFile, 'Resolution', exportDpi);
            catch
                print(figOrLayout, outFile, ['-d' char(outputFormat)], sprintf('-r%d', exportDpi));
            end
            fprintf('Wrote %s\\n', outFile);
            if ~closeFiguresAfterExport
                if bringFiguresToFront
                    figure(figHandle);
                end
            else
                close(figHandle);
            end
        end

        function plotRegionGuide(stem, dataDir, outDir)
            rulerFile = csvPathFor(dataDir, stem, '_ruler_regions.csv');
            specFile = csvPathFor(dataDir, stem, '_spectrum.csv');
            if ~isfile(rulerFile)
                warnMissing('Missing ruler CSV', rulerFile);
                return;
            end
            if ~isfile(specFile)
                warnMissing('Missing spectrum CSV (for wavenumber axis)', specFile);
                return;
            end
            T = readtable(specFile);
            wn = T.wavenumber_cm1;
            R = readtable(rulerFile);
            nR = height(R);
            figH = max(3.5, 1.2 + 0.28 * nR);
            fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 16 figH]);
            ax = axes(fig);
            hold(ax, 'on');
            yLo = 0.02;
            yHi = 0.98;
            rowH = (yHi - yLo) / max(nR, 1);
            for ri = 1:nR
                y0 = yHi - ri * rowH + 0.12 * rowH;
                y1 = yHi - (ri - 1) * rowH - 0.12 * rowH;
                lo = R.lo_cm1(ri);
                hi = R.hi_cm1(ri);
                fill(ax, [lo hi hi lo], [y0 y0 y1 y1], [0.89 0.91 0.94], ...
                    'EdgeColor', [0.58 0.64 0.72], 'LineWidth', 0.6);
                lbl = char(string(R.region_label(ri)));
                text(ax, (lo + hi) / 2, (y0 + y1) / 2, lbl, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', fontRegionBand, 'Color', [0.2 0.25 0.33], 'Interpreter', 'none');
            end
            hold(ax, 'off');
            ylim(ax, [0 1]);
            set(ax, 'YTick', []);
            xlabel(ax, 'Wavenumber (cm^{{-1}})');
            title(ax, 'FTIR region guide (tentative ranges)', 'FontWeight', 'normal');
            set(ax, 'Box', 'on', 'FontName', 'Arial', 'FontSize', fontRegionAxis);
            setWavenumberXLim(ax, wn);
            outFile = fullfile(outDir, [stem '_region_guide_matlab.' char(outputFormat)]);
            exportFigure(fig, outFile);
        end

        function drawSpectrumPeakLabels(ax, wnPk, yPk, txtLabels, ySpan, labelFontSize)
            if nargin < 6 || isempty(labelFontSize)
                labelFontSize = fontSpectrumPeakLabel;
            end
            if isempty(wnPk)
                return;
            end
            laid = layoutPeakLabels(wnPk, yPk, txtLabels, ySpan);
            for li = 1:numel(laid)
                text(ax, laid(li).x, laid(li).yText, laid(li).text, ...
                    'HorizontalAlignment', laid(li).hAlign, ...
                    'VerticalAlignment', laid(li).vAlign, ...
                    'FontSize', labelFontSize, ...
                    'Color', [0.2 0.2 0.2], ...
                    'Rotation', laid(li).angle);
            end
        end

        function laid = layoutPeakLabels(wnPk, yPk, txtLabels, ySpan)
            n = numel(wnPk);
            laid = repmat(struct('x', 0, 'yText', 0, 'text', '', 'hAlign', 'center', ...
                'vAlign', 'bottom', 'angle', peakLabelRotation), n, 1);
            yMax = max(yPk);
            yMin = min(yPk);
            ySpan = max(max(ySpan, yMax - yMin), 1e-9);
            wnSpan = max(max(wnPk) - min(wnPk), 400);
            yCeil = yMax + 0.18 * ySpan;
            baseShift = 10;
            for i = 1:n
                laid(i).x = wnPk(i);
                laid(i).text = txtLabels{{i}};
                laid(i).yText = yPk(i) + (baseShift / 280) * ySpan;
                laid(i).angle = peakLabelRotation;
            end
            if ~peakLabelCollisionAware || n <= 1
                return;
            end
            [~, ord] = sort(yPk, 'descend');
            shiftsPx = [10 20 32 44 58 72 88 104];
            placed = zeros(0, 4);
            for oi = 1:numel(ord)
                i = ord(oi);
                wn = wnPk(i);
                y = yPk(i);
                txt = txtLabels{{i}};
                ok = false;
                for si = 1:numel(shiftsPx)
                    ysh = shiftsPx(si);
                    box = estimateLabelBox(wn, y, txt, ySpan, wnSpan, peakLabelRotation, ysh, yCeil);
                    if isempty(box) || anyBoxOverlaps(box, placed)
                        continue;
                    end
                    placed(end+1, :) = box; %#ok<AGROW>
                    laid(i).angle = peakLabelRotation;
                    laid(i).yText = y + (ysh / 280) * ySpan;
                    laid(i).hAlign = 'center';
                    laid(i).vAlign = 'bottom';
                    ok = true;
                    break;
                end
            end
        end

        function box = estimateLabelBox(wn, y, txt, ySpan, wnSpan, angle, yshiftPx, yCeil)
            yOff = (yshiftPx / 280) * ySpan;
            if abs(angle) >= 89
                xHalf = max(5, 7) * (wnSpan / max(ySpan * 50, 1)) * 0.06;
                charH = 0.026 * ySpan * max(numel(txt), 3);
                yTop = y + yOff + charH + 0.02 * ySpan;
                yBot = y + yOff - 0.01 * ySpan;
                if yTop > yCeil
                    box = [];
                    return;
                end
                box = [wn - xHalf, wn + xHalf, yBot, yTop];
                return;
            end
            wHalf = max(12, 10 + numel(txt) * 1.6);
            if abs(angle) >= 45
                wHalf = max(10, 8 + numel(txt) * 0.9);
            end
            xHalf = wHalf * (wnSpan / max(ySpan * 50, 1)) * 0.15;
            yTop = y + yOff + 0.045 * ySpan;
            yBot = y + yOff - 0.012 * ySpan;
            if abs(angle) < 89 && yTop > yCeil
                box = [];
                return;
            end
            box = [wn - xHalf, wn + xHalf, yBot, yTop];
        end

        function tf = anyBoxOverlaps(box, placed)
            tf = false;
            if isempty(placed)
                return;
            end
            for r = 1:size(placed, 1)
                b = placed(r, :);
                if box(1) < b(2) && b(1) < box(2) && box(3) < b(4) && b(3) < box(4)
                    tf = true;
                    return;
                end
            end
        end

        function plotSpectrumPeaks(stem, dataDir, outDir)
            specFile = csvPathFor(dataDir, stem, '_spectrum.csv');
            peaksFile = csvPathFor(dataDir, stem, '_peaks.csv');
            if ~isfile(specFile)
                warnMissing('Missing spectrum CSV', specFile);
                return;
            end
            T = readtable(specFile);
            wn = T.wavenumber_cm1;
            y = T.absorbance;
            nLabels = 0;
            if showPeakLabels && isfile(peaksFile)
                P = readtable(peaksFile);
                if ismember('is_labeled', P.Properties.VariableNames)
                    idx = find(P.is_labeled == 1);
                else
                    idx = (1:min(height(P), maxPeakLabels))';
                end
                if ~labelAllLabeledPeaks
                    idx = idx(1:min(numel(idx), maxPeakLabels));
                end
                nLabels = numel(idx);
            end
            figH = 12 + min(6, max(0, nLabels - 18) * 0.12);
            fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 16 figH]);
            ax = axes(fig);
            hold(ax, 'on');
            plot(ax, wn, y, 'Color', [0 0.447 0.741], 'LineWidth', 1.1);
            if showPeakLabels && isfile(peaksFile)
                P = readtable(peaksFile);
                if ismember('is_labeled', P.Properties.VariableNames)
                    idx = find(P.is_labeled == 1);
                else
                    idx = (1:min(height(P), maxPeakLabels))';
                end
                if ~labelAllLabeledPeaks
                    idx = idx(1:min(numel(idx), maxPeakLabels));
                end
                ySpan = max(y) - min(y);
                wnPk = P.peak_position_cm1(idx);
                yPk = P.peak_height(idx);
                txtLabels = arrayfun(@(x) sprintf('%.0f', x), wnPk, 'UniformOutput', false);
                plot(ax, wnPk, yPk, 'o', 'MarkerSize', 4, 'Color', [0.85 0.33 0.1]);
                drawSpectrumPeakLabels(ax, wnPk, yPk, txtLabels, ySpan);
            end
            hold(ax, 'off');
            xlabel(ax, 'Wavenumber (cm^{{-1}})');
            ylabel(ax, 'Normalized absorbance');
            title(ax, strrep(stem, '_', ' '), 'FontWeight', 'normal');
            set(ax, 'Box', 'on', 'FontName', 'Arial', 'FontSize', fontSpectrumAxis);
            grid(ax, 'on');
            setWavenumberXLim(ax, wn);
            outFile = fullfile(outDir, [stem '_spectrum_peaks_matlab.' char(outputFormat)]);
            exportFigure(fig, outFile);
        end

        function plotOneSpectrum(stem, dataDir, outDir)
            specFile = csvPathFor(dataDir, stem, '_spectrum.csv');
            peaksFile = csvPathFor(dataDir, stem, '_peaks.csv');
            rulerFile = csvPathFor(dataDir, stem, '_ruler_regions.csv');
            if ~isfile(specFile)
                warnMissing('Missing spectrum CSV', specFile);
                return;
            end
            T = readtable(specFile);
            wn = T.wavenumber_cm1;
            y = T.absorbance;
            if showKronecker
                tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
            else
                tl = tiledlayout(1, 1, 'Padding', 'compact');
            end
            ax1 = nexttile(tl, 1);
            hold(ax1, 'on');
            plot(ax1, wn, y, 'Color', [0 0.447 0.741], 'LineWidth', 1.2);
            if showRulerOverlay && isfile(rulerFile)
                R = readtable(rulerFile);
                yR = max(y) * 1.02;
                for ri = 1:height(R)
                    lo = R.lo_cm1(ri);
                    hi = R.hi_cm1(ri);
                    patch(ax1, [lo hi hi lo], [yR*0.98 yR*0.98 yR yR], ...
                        [0.85 0.85 0.85], 'EdgeColor', [0.4 0.4 0.4], 'FaceAlpha', 0.35);
                    text(ax1, (lo+hi)/2, yR*1.01, char(string(R.region_label(ri))), ...
                        'HorizontalAlignment', 'center', 'FontSize', fontCombinedRulerOverlay, 'Color', [0.2 0.2 0.2]);
                end
            end
            if showPeakLabels && isfile(peaksFile)
                P = readtable(peaksFile);
                if ismember('is_labeled', P.Properties.VariableNames)
                    idx = find(P.is_labeled == 1);
                else
                    idx = (1:min(height(P), maxPeakLabels))';
                end
                if ~labelAllLabeledPeaks
                    idx = idx(1:min(numel(idx), maxPeakLabels));
                end
                ySpan = max(y) - min(y);
                wnPk = P.peak_position_cm1(idx);
                yPk = P.peak_height(idx);
                txtLabels = arrayfun(@(x) sprintf('%.0f', x), wnPk, 'UniformOutput', false);
                plot(ax1, wnPk, yPk, 'o', 'MarkerSize', 5, 'Color', [0.85 0.33 0.1]);
                drawSpectrumPeakLabels(ax1, wnPk, yPk, txtLabels, ySpan, fontCombinedPeakLabel);
            end
            hold(ax1, 'off');
            xlabel(ax1, 'Wavenumber (cm^{{-1}})');
            ylabel(ax1, 'Normalized absorbance');
            title(ax1, strrep(stem, '_', ' '), 'FontWeight', 'normal');
            set(ax1, 'Box', 'on', 'FontName', 'Arial', 'FontSize', fontCombinedAxis);
            grid(ax1, 'on');
            setWavenumberXLim(ax1, wn);
            if showKronecker && isfile(peaksFile)
                ax2 = nexttile(tl, 2);
                P = readtable(peaksFile);
                stem(ax2, P.peak_position_cm1, P.peak_height, 'Color', [0.3 0.5 0.7], 'LineWidth', 0.8);
                xlabel(ax2, 'Wavenumber (cm^{{-1}})');
                ylabel(ax2, 'Peak height');
                set(ax2, 'Box', 'on', 'FontName', 'Arial', 'FontSize', 10);
                grid(ax2, 'on');
                setWavenumberXLim(ax2, wn);
            end
            outFile = fullfile(outDir, [stem '_combined_matlab.' char(outputFormat)]);
            exportFigure(tl, outFile);
        end

        end
        """
    )
    path = matlab_dir / "make_figures.m"
    path.write_text(script, encoding="utf-8")
    return path


def export_key_evidence_table_csv(
    pipeline: dict[str, Any],
    path: Path,
) -> None:
    """Optional standalone key-evidence CSV (same columns as matlab bundle)."""
    from reports.v4_evidence_report import evidence_ranked_assignments
    from reports.product_v1_report import chemistry_label

    rows: list[dict[str, Any]] = []
    for lab, ent in evidence_ranked_assignments(pipeline, top_n=16):
        bands = ent.get("supporting_bands") or ent.get("matched_bands") or []
        band_parts: list[str] = []
        for b in bands[:6]:
            if isinstance(b, dict) and b.get("wn_cm1"):
                band_parts.append(f"{float(b['wn_cm1']):.0f}")
            elif isinstance(b, (int, float)):
                band_parts.append(f"{float(b):.0f}")
        rows.append(
            {
                "assignment_label": chemistry_label(lab, pipeline, ent),
                "key_bands_cm1": ", ".join(band_parts),
                "confidence_status": _status_display(ent, pipeline, lab),
                "score": f"{float(ent.get('score', 0) or 0):.4f}",
            }
        )
    _write_csv(
        path,
        ["assignment_label", "key_bands_cm1", "confidence_status", "score"],
        rows,
    )
