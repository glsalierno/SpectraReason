#!/usr/bin/env python3
"""
Article FTIR multi-spectrum stack exporter (standalone; not production report code).

Default run writes normalized absorbance stack, %T stack (transmittance inputs only),
and fingerprint-zoom normalized stack under --out-root.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from lib.ftir_foundation import infer_intensity_mode, preprocess_spectrum, read_spectrum
from reports.annotation_layout import apply_peak_label_layout, cluster_peaks_for_labeling

StackKind = Literal["normalized_absorbance", "transmittance"]

TRACE_COLORS = ("#0072bd", "#d95319", "#77ac30", "#7e2f8e")
ARTICLE_ROOT = Path(
    r"C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\FTIR"
)
FTIR_POWDER = Path(
    r"C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\FTIR_POWDER"
)
PDA_ODA = Path(r"C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\PDA_ODA")


@dataclass
class StackCase:
    id: str
    label: str
    path: Path
    force_mode: str | None = None
    include_transmittance: bool = True
    peaks_csv: Path | None = None


def default_manifest(report_root: Path | None = None) -> list[StackCase]:
    root = report_root or ARTICLE_ROOT
    scaled = FTIR_POWDER / "pda_eg_con_new_minus_air_scaled.CSV"
    if not scaled.is_file():
        scaled = root / "pda_eg_con_new_minus_air_scaled" / "pda_eg_con_new_minus_air_scaled.CSV"

    def peaks(case_folder: str, stem: str) -> Path | None:
        p = root / case_folder / "matlab_export" / f"{stem}_peaks.csv"
        return p if p.is_file() else None

    return [
        StackCase(
            id="dopamine_powder",
            label="Dopamine powder",
            path=FTIR_POWDER / "Dopamine_Powder.CSV",
            peaks_csv=peaks("Dopamine_Powder", "Dopamine_Powder"),
        ),
        StackCase(
            id="pda_eg_air_corrected",
            label="PDA/EG (air corrected)",
            path=scaled,
            force_mode="absorbance",
            include_transmittance=False,
            peaks_csv=peaks("pda_eg_con_new_minus_air_scaled", "pda_eg_con_new_minus_air_scaled"),
        ),
        StackCase(
            id="oda_ethanol",
            label="ODA in ethanol",
            path=PDA_ODA / "ODA_in_Ethanol_blank_subtracted.CSV",
            force_mode="absorbance",
            include_transmittance=False,
            peaks_csv=peaks("ODA_in_Ethanol_blank_subtracted", "ODA_in_Ethanol_blank_subtracted"),
        ),
        StackCase(
            id="pda_oda",
            label="PDA–ODA",
            path=PDA_ODA / "pda_oda.CSV",
            peaks_csv=peaks("pda_oda", "pda_oda"),
        ),
    ]


def _resolve_mode(case: StackCase, raw: np.ndarray, hint: str) -> str:
    if case.force_mode:
        return case.force_mode
    if hint != "auto":
        return hint
    return infer_intensity_mode(raw)


def _transmittance_percent(raw: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(raw, dtype=float)
    if mode == "transmittance_fraction":
        return y * 100.0
    if mode == "transmittance_percent":
        return y
    raise ValueError(f"Not a transmittance spectrum (mode={mode!r})")


def _mask(wn: np.ndarray, wn_min: float, wn_max: float) -> np.ndarray:
    return (wn >= wn_min) & (wn <= wn_max)


def _load_peak_candidates(
    case: StackCase,
    wn: np.ndarray,
    y: np.ndarray,
    *,
    wn_min: float,
    wn_max: float,
    max_labels: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if case.peaks_csv and case.peaks_csv.is_file():
        seen: set[int] = set()
        with case.peaks_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wn_p = float(row["peak_position_cm1"])
                key = int(round(wn_p))
                if key in seen:
                    continue
                seen.add(key)
                h = float(row.get("peak_height") or 0)
                labeled = str(row.get("is_labeled", "0")).strip() in ("1", "true", "True")
                rows.append(
                    {
                        "wn_cm1": wn_p,
                        "height": h,
                        "peak_quality": row.get("peak_quality") or "moderate",
                        "label_reason": "key_evidence" if labeled else "height_prominence",
                    }
                )
        rows.sort(key=lambda r: -float(r["height"]))
    else:
        # Fallback: local maxima proxy from processed trace (coarse grid)
        m = _mask(wn, wn_min, wn_max)
        wn_m, y_m = wn[m], y[m]
        if wn_m.size < 5:
            return []
        step = max(1, wn_m.size // 200)
        for i in range(1, len(y_m) - 1, step):
            if y_m[i] >= y_m[i - 1] and y_m[i] >= y_m[i + 1] and y_m[i] > 0.08:
                rows.append(
                    {
                        "wn_cm1": float(wn_m[i]),
                        "height": float(y_m[i]),
                        "peak_quality": "moderate",
                        "label_reason": "height_prominence",
                    }
                )
        rows.sort(key=lambda r: -float(r["height"]))

    clustered, _ = cluster_peaks_for_labeling(rows, cluster_distance_cm1=18.0)
    picked = clustered[: max(max_labels * 3, max_labels)]
    ann: list[dict[str, Any]] = []
    for p in picked:
        wn_p = float(p["wn_cm1"])
        if wn_p < wn_min or wn_p > wn_max:
            continue
        y_at = float(np.interp(wn_p, wn, y))
        ann.append(
            {
                "wn": wn_p,
                "y": y_at,
                "text": f"{wn_p:.0f}",
                "_peak": p,
            }
        )
    laid, _stats = apply_peak_label_layout(
        ann,
        mode="smart",
        y_max=float(np.nanmax(y)) if y.size else 1.0,
        y_min=float(np.nanmin(y)) if y.size else 0.0,
        wn_min=wn_min,
        wn_max=wn_max,
        presentation=True,
    )
    return laid[:max_labels]


@dataclass
class PreparedTrace:
    case: StackCase
    wn: np.ndarray
    y: np.ndarray
    offset: float
    span: float
    color: str
    intensity_mode: str


def _prepare_traces(
    cases: list[StackCase],
    *,
    kind: StackKind,
    wn_min: float,
    wn_max: float,
    offset_step: float,
) -> list[PreparedTrace]:
    out: list[PreparedTrace] = []
    cumulative = 0.0
    color_i = 0
    for case in cases:
        wn, raw, hint = read_spectrum(case.path)
        mode = _resolve_mode(case, raw, hint)

        if kind == "transmittance":
            if not case.include_transmittance or mode not in (
                "transmittance_percent",
                "transmittance_fraction",
            ):
                continue
            y_full = _transmittance_percent(raw, mode)
            m = _mask(wn, wn_min, wn_max)
            wn_v, y_v = wn[m], y_full[m]
            span = float(np.nanmax(y_v) - np.nanmin(y_v)) if y_v.size else 1.0
            span = max(span, 1e-6)
        else:
            wn_v, y_v, _info = preprocess_spectrum(
                wn, raw, intensity_mode=mode, normalize=True
            )
            m = _mask(wn_v, wn_min, wn_max)
            wn_v, y_v = wn_v[m], y_v[m]
            span = 1.0

        offset = cumulative
        cumulative += span * offset_step
        out.append(
            PreparedTrace(
                case=case,
                wn=wn_v,
                y=y_v,
                offset=offset,
                span=span,
                color=TRACE_COLORS[color_i % len(TRACE_COLORS)],
                intensity_mode=mode,
            )
        )
        color_i += 1
    return out


def _figure_height(n_traces: int, *, labeled: bool) -> float:
    base = 1.8 + 0.55 * n_traces
    return base + (0.8 if labeled else 0.0)


def _draw_stack(
    traces: list[PreparedTrace],
    *,
    kind: StackKind,
    wn_min: float,
    wn_max: float,
    label_peaks: bool,
    max_labels_per_trace: int,
) -> plt.Figure:
    labeled = label_peaks and kind == "normalized_absorbance"
    fig, ax = plt.subplots(figsize=(7.0, _figure_height(len(traces), labeled=labeled)))
    wn_span = wn_max - wn_min
    label_x = wn_max - 0.02 * wn_span

    for tr in traces:
        y_plot = tr.y + tr.offset
        ax.plot(tr.wn, y_plot, color=tr.color, linewidth=1.0, solid_capstyle="round")
        ax.text(
            label_x,
            tr.offset + 0.5 * tr.span,
            tr.case.label,
            fontsize=9,
            va="center",
            ha="right",
            color=tr.color,
            fontweight="semibold",
        )
        if label_peaks and kind == "normalized_absorbance":
            peaks = _load_peak_candidates(
                tr.case,
                tr.wn,
                tr.y,
                wn_min=wn_min,
                wn_max=wn_max,
                max_labels=max_labels_per_trace,
            )
            label_y_span = max(tr.span * 1.15, 1.0)
            for pk in peaks:
                wn_p = float(pk["wn"])
                y_p = float(pk["y"]) + tr.offset
                ysh = int(pk.get("yshift", 12) or 12)
                y_txt = y_p + (ysh / 280.0) * label_y_span
                angle = float(pk.get("textangle", -90) or -90)
                ax.text(
                    wn_p,
                    y_txt,
                    pk.get("text", ""),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                    rotation=angle,
                    rotation_mode="anchor",
                    clip_on=True,
                )

    ax.set_xlim(wn_max, wn_min)
    ax.set_yticks([])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
    if kind == "transmittance":
        ax.set_ylabel("Transmittance (%)", fontsize=11)
    else:
        ax.set_ylabel("Normalized absorbance (offset)", fontsize=11)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.96, bottom=0.12)
    return fig


def _save_figure(fig: plt.Figure, base_path: Path, formats: list[str], dpi: int) -> list[str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    stem = base_path.stem
    parent = base_path.parent
    for fmt in formats:
        fmt = fmt.lstrip(".").lower()
        out = parent / f"{stem}.{fmt}"
        save_kw: dict[str, Any] = {"facecolor": "white", "bbox_inches": "tight"}
        if fmt == "png":
            save_kw["dpi"] = dpi
        fig.savefig(out, format=fmt, **save_kw)
        written.append(str(out.resolve()))
    plt.close(fig)
    return written


def _load_manifest(path: Path) -> list[StackCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[StackCase] = []
    for row in data:
        cases.append(
            StackCase(
                id=str(row["id"]),
                label=str(row["label"]),
                path=Path(row["path"]),
                force_mode=row.get("force_mode"),
                include_transmittance=bool(row.get("include_transmittance", True)),
                peaks_csv=Path(row["peaks_csv"]) if row.get("peaks_csv") else None,
            )
        )
    return cases


def export_stacks(
    *,
    out_root: Path,
    cases: list[StackCase],
    wn_min: float,
    wn_max: float,
    fp_wn_min: float,
    fp_wn_max: float,
    offset_step: float,
    dpi: int,
    formats: list[str],
    label_peaks: bool,
    max_labels_per_trace: int,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "out_root": str(out_root.resolve()),
        "wn_range": [wn_min, wn_max],
        "fingerprint_range": [fp_wn_min, fp_wn_max],
        "offset_step": offset_step,
        "cases": [{"id": c.id, "label": c.label, "path": str(c.path.resolve())} for c in cases],
        "outputs": {},
    }

    norm_traces = _prepare_traces(
        cases,
        kind="normalized_absorbance",
        wn_min=wn_min,
        wn_max=wn_max,
        offset_step=offset_step,
    )
    if norm_traces:
        fig = _draw_stack(
            norm_traces,
            kind="normalized_absorbance",
            wn_min=wn_min,
            wn_max=wn_max,
            label_peaks=False,
            max_labels_per_trace=max_labels_per_trace,
        )
        manifest["outputs"]["normalized_absorbance"] = _save_figure(
            fig,
            out_root / "article_ftir_stack_normalized_absorbance",
            formats,
            dpi,
        )

    t_traces = _prepare_traces(
        cases,
        kind="transmittance",
        wn_min=wn_min,
        wn_max=wn_max,
        offset_step=offset_step,
    )
    if t_traces:
        fig = _draw_stack(
            t_traces,
            kind="transmittance",
            wn_min=wn_min,
            wn_max=wn_max,
            label_peaks=False,
            max_labels_per_trace=max_labels_per_trace,
        )
        manifest["outputs"]["transmittance"] = _save_figure(
            fig,
            out_root / "article_ftir_stack_transmittance",
            formats,
            dpi,
        )

    fp_traces = _prepare_traces(
        cases,
        kind="normalized_absorbance",
        wn_min=fp_wn_min,
        wn_max=fp_wn_max,
        offset_step=offset_step,
    )
    if fp_traces:
        fig = _draw_stack(
            fp_traces,
            kind="normalized_absorbance",
            wn_min=fp_wn_min,
            wn_max=fp_wn_max,
            label_peaks=False,
            max_labels_per_trace=max_labels_per_trace,
        )
        fp_formats = formats if "png" in formats else ["png"]
        manifest["outputs"]["fingerprint_normalized_absorbance"] = _save_figure(
            fig,
            out_root / "article_ftir_stack_fingerprint_normalized_absorbance",
            fp_formats,
            dpi,
        )

        if label_peaks:
            fig_l = _draw_stack(
                fp_traces,
                kind="normalized_absorbance",
                wn_min=fp_wn_min,
                wn_max=fp_wn_max,
                label_peaks=True,
                max_labels_per_trace=max_labels_per_trace,
            )
            manifest["outputs"]["fingerprint_normalized_absorbance_labeled"] = _save_figure(
                fig_l,
                out_root / "article_ftir_stack_fingerprint_normalized_absorbance_labeled",
                fp_formats,
                dpi,
            )

    (out_root / "stack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_index(out_root, manifest)
    return manifest


def _write_index(out_root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Article FTIR stack figures",
        "",
        f"Output root: `{out_root.resolve()}`",
        "",
        "## Files",
        "",
    ]
    for key, paths in manifest.get("outputs", {}).items():
        lines.append(f"### {key}")
        for p in paths:
            lines.append(f"- `{p}`")
        lines.append("")
    (out_root / "STACK_FIGURES_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export article FTIR spectrum stacks.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ARTICLE_ROOT / "stacks",
        help="Directory for stack figure outputs",
    )
    ap.add_argument("--wn-min", type=float, default=400.0)
    ap.add_argument("--wn-max", type=float, default=4000.0)
    ap.add_argument("--fingerprint-wn-min", type=float, default=400.0)
    ap.add_argument("--fingerprint-wn-max", type=float, default=1500.0)
    ap.add_argument("--offset-step", type=float, default=1.15)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg", "pdf"],
        help="Output formats (png svg pdf)",
    )
    ap.add_argument("--label-peaks", action="store_true", help="Emit labeled fingerprint stack")
    ap.add_argument("--max-labels-per-trace", type=int, default=5)
    ap.add_argument("--manifest", type=Path, default=None, help="JSON manifest overriding defaults")
    ap.add_argument(
        "--report-root",
        type=Path,
        default=ARTICLE_ROOT,
        help="Base folder for per-case report peaks CSV resolution",
    )
    args = ap.parse_args()

    cases = _load_manifest(args.manifest) if args.manifest else default_manifest(args.report_root)
    manifest = export_stacks(
        out_root=args.out_root,
        cases=cases,
        wn_min=float(args.wn_min),
        wn_max=float(args.wn_max),
        fp_wn_min=float(args.fingerprint_wn_min),
        fp_wn_max=float(args.fingerprint_wn_max),
        offset_step=float(args.offset_step),
        dpi=int(args.dpi),
        formats=[f.lstrip(".").lower() for f in args.formats],
        label_peaks=bool(args.label_peaks),
        max_labels_per_trace=int(args.max_labels_per_trace),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
