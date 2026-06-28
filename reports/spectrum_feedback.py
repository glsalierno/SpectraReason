"""Per-spectrum manuscript feedback summaries from paper peak selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reports.paper_peak_selection import PAPER_LABEL_REGIONS, region_title


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


def _peak_list_text(peaks: list[dict[str, Any]], pipeline: dict[str, Any] | None) -> str:
    if not peaks:
        return "none selected"
    parts: list[str] = []
    for p in peaks:
        wn = float(p.get("wavenumber_cm1", p.get("wn", 0)))
        assign = _assignment_for_wn(wn, pipeline)
        if assign:
            parts.append(f"{wn:.0f} cm⁻¹ ({assign})")
        else:
            parts.append(f"{wn:.0f} cm⁻¹")
    return ", ".join(parts)


def _detect_artifacts(selected: list[dict[str, Any]], y_norm: Any) -> list[str]:
    notes: list[str] = []
    import numpy as np

    y = np.asarray(y_norm, dtype=float)
    for p in selected:
        wn = float(p.get("wavenumber_cm1", 0))
        if 2300 <= wn <= 2400:
            notes.append("Possible CO₂ band near 2340 cm⁻¹ — verify sample environment.")
        if 1600 <= wn <= 1640 and float(p.get("prominence", 0)) > 0.15:
            notes.append("Strong feature near 1600 cm⁻¹ may include amide / aromatic overlap.")
    if y.size > 10:
        noise = float(np.std(np.diff(y)))
        if noise > 0.04:
            notes.append("Moderate high-frequency noise — interpret weak shoulders cautiously.")
    return list(dict.fromkeys(notes))


def build_spectrum_feedback(
    *,
    stem: str,
    selection_result: Any,
    pipeline: dict[str, Any] | None,
    preprocess_info: dict[str, Any] | None,
    transmittance_valid: bool,
    transmittance_note: str,
    y_norm: Any,
    selected_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = list(selected_override if selected_override is not None else selection_result.selected)
    candidates = list(selection_result.candidates)
    suppressed = list(selection_result.suppressed)

    broad = [p for p in selected if float(p.get("prominence", 0)) >= 0.12]
    sharp = [p for p in selected if float(p.get("prominence", 0)) < 0.08 and float(p.get("height", 0)) >= 0.06]

    region_map: dict[str, list[float]] = {str(r["id"]): [] for r in PAPER_LABEL_REGIONS}
    for p in selected:
        rid = str(p.get("region") or "")
        if rid in region_map:
            region_map[rid].append(float(p["wavenumber_cm1"]))

    char_parts: list[str] = []
    if region_map.get("oh_nh"):
        char_parts.append("O–H / N–H stretch activity")
    if region_map.get("ch"):
        char_parts.append("C–H stretch bands")
    if region_map.get("co_aromatic"):
        char_parts.append("mid-IR carbonyl / aromatic-region features")
    if region_map.get("ring_cn"):
        char_parts.append("ring / C–N region bands")
    if region_map.get("co_fp"):
        char_parts.append("C–O / fingerprint-region features")
    overall = (
        f"The spectrum shows {', '.join(char_parts)}."
        if char_parts
        else "The spectrum shows limited labeled features in the standard manuscript regions."
    )

    highlights: list[str] = []
    if broad:
        highlights.append(
            f"Broad or strong bands near {', '.join(f'{float(p['wavenumber_cm1']):.0f}' for p in broad[:3])} cm⁻¹."
        )
    if sharp:
        highlights.append(
            f"Sharper features near {', '.join(f'{float(p['wavenumber_cm1']):.0f}' for p in sharp[:3])} cm⁻¹."
        )
    if len(candidates) > len(selected) + 3:
        highlights.append(
            f"{len(candidates) - len(selected)} weaker or crowded peaks were detected but not labeled."
        )
    highlights.extend(_detect_artifacts(selected, y_norm))

    baseline = str((preprocess_info or {}).get("baseline", "unknown"))
    quality_notes = [
        f"Baseline handling: {baseline}.",
        f"Selected labels: {len(selected)} of {len(candidates)} detected candidates.",
    ]
    if not transmittance_valid:
        quality_notes.append(transmittance_note or "Transmittance (%T) not exported for this input.")
    else:
        quality_notes.append("Transmittance labels reuse absorbance-selected wavenumbers.")

    region_lines: list[str] = []
    for spec in PAPER_LABEL_REGIONS:
        rid = str(spec["id"])
        peaks = region_map.get(rid) or []
        if peaks:
            region_lines.append(
                f"- {spec['title']}: {_peak_list_text([{'wavenumber_cm1': w} for w in peaks], pipeline)}"
            )
        else:
            region_lines.append(f"- {spec['title']}: no peaks selected for labeling.")
    region_lines.append(
        "- 900–400 cm⁻¹: ignored for automatic labeling and interpretation (fingerprint tail)."
    )

    draft_lines: list[str] = []
    for p in selected[:6]:
        wn = float(p["wavenumber_cm1"])
        assign = _assignment_for_wn(wn, pipeline)
        if assign:
            draft_lines.append(
                f"A band near {wn:.0f} cm⁻¹ is consistent with {assign.lower()} (tentative; confirm with reference standards)."
            )
        else:
            draft_lines.append(
                f"A feature near {wn:.0f} cm⁻¹ may contribute to the overall functional-group profile; assign cautiously."
            )
    if not draft_lines:
        draft_lines.append(
            "Few strong labeled peaks were found in the manuscript regions; emphasize baseline quality and comparison spectra."
        )

    return {
        "stem": stem,
        "overall_character": overall,
        "key_regions": [
            {
                "region": region_title(rid),
                "peaks_cm1": region_map.get(rid, []),
            }
            for rid in region_map
            if region_map.get(rid)
        ],
        "highlights": highlights,
        "region_summary_lines": region_lines,
        "quality_notes": quality_notes,
        "draft_guidance": draft_lines,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "n_suppressed": len(suppressed),
    }


def _render_feedback_text(feedback: dict[str, Any]) -> str:
    lines = [
        f"# FTIR feedback — {feedback.get('stem', 'spectrum')}",
        "",
        "## Overall spectral character",
        str(feedback.get("overall_character", "")),
        "",
        "## Key functional group regions detected",
    ]
    for item in feedback.get("key_regions") or []:
        peaks = ", ".join(f"{p:.0f}" for p in item.get("peaks_cm1") or [])
        lines.append(f"- {item.get('region')}: {peaks or '—'}")
    lines.extend(["", "## Notable differences or highlights"])
    for h in feedback.get("highlights") or []:
        lines.append(f"- {h}")
    if not feedback.get("highlights"):
        lines.append("- No unusual highlights flagged automatically.")
    lines.extend(["", "## Region-by-region summary"])
    lines.extend(feedback.get("region_summary_lines") or [])
    lines.extend(["", "## Data quality notes"])
    for q in feedback.get("quality_notes") or []:
        lines.append(f"- {q}")
    lines.extend(["", "## Draft interpretation guidance"])
    for d in feedback.get("draft_guidance") or []:
        lines.append(f"- {d}")
    lines.append("")
    return "\n".join(lines)


def _render_feedback_md(feedback: dict[str, Any]) -> str:
    return _render_feedback_text(feedback).replace("# FTIR feedback", "# FTIR feedback (markdown)")


def write_spectrum_feedback_files(
    out_dir: Path,
    stem: str,
    feedback: dict[str, Any],
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{stem}_feedback.txt"
    md_path = out_dir / f"{stem}_feedback.md"
    txt_path.write_text(_render_feedback_text(feedback), encoding="utf-8")
    md_path.write_text(_render_feedback_md(feedback), encoding="utf-8")
    return {
        "txt": str(txt_path.resolve()),
        "md": str(md_path.resolve()),
    }


def build_feedback_section_html(feedback: dict[str, Any], report_dir: Path) -> str:
    import html as html_mod

    def esc(s: str) -> str:
        return html_mod.escape(str(s or ""), quote=True)

    parts = [
        "<section class='paper-feedback-section card'>",
        "<h3>Spectrum feedback (manuscript draft)</h3>",
        f"<p>{esc(feedback.get('overall_character', ''))}</p>",
        "<h4>Highlights</h4><ul>",
    ]
    for h in feedback.get("highlights") or []:
        parts.append(f"<li>{esc(h)}</li>")
    if not feedback.get("highlights"):
        parts.append("<li class='muted'>No unusual highlights flagged automatically.</li>")
    parts.append("</ul><h4>Region summary</h4><ul>")
    for line in feedback.get("region_summary_lines") or []:
        parts.append(f"<li>{esc(line.lstrip('- '))}</li>")
    parts.append("</ul><h4>Draft interpretation guidance</h4><ul>")
    for d in feedback.get("draft_guidance") or []:
        parts.append(f"<li>{esc(d)}</li>")
    parts.append("</ul></section>")
    return "".join(parts)
