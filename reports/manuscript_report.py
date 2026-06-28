"""
Concise manuscript-style HTML report for FTIR paper figures.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from reports.paper_ftir_figures import format_download_links


def _esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def _rel_href(path: str, report_dir: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return p.name


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(name).stem)


def build_manuscript_spectrum_section(
    *,
    name: str,
    anchor: str,
    paper_manifest: dict[str, Any],
    report_dir: Path,
) -> str:
    stem = paper_manifest.get("stem") or Path(name).stem
    figures = paper_manifest.get("figures") or {}
    parts: list[str] = [
        f"<section id='{_esc(anchor)}' class='card manuscript-spec'>",
        f"<h2>{_esc(stem.replace('_', ' '))}</h2>",
        "<p class='muted small'>Selected peaks from two-stage absorbance picking; "
        "900–400 cm⁻¹ excluded unless overridden.</p>",
    ]
    if paper_manifest.get("transmittance_note"):
        parts.append(f"<p class='hint'><em>{_esc(paper_manifest['transmittance_note'])}</em></p>")

    for heading, peak_key, clean_key in (
        ("Transmittance (%T)", "transmittance_peaks", "manuscript_transmittance"),
        ("Normalized absorbance (0–1)", "normalized_absorbance_peaks", "manuscript_normalized_absorbance"),
    ):
        paths = figures.get(clean_key) or figures.get(peak_key) or []
        if not paths:
            continue
        parts.append(f"<h3>{heading}</h3>")
        parts.append(f"<p class='dl-row'>{format_download_links(paths, report_dir)}</p>")

    sel_table = (paper_manifest.get("peak_tables") or {}).get("selected")
    if sel_table:
        parts.append("<h3>Selected peaks</h3>")
        parts.append(_peak_table_from_csv(sel_table, report_dir))

    feedback = paper_manifest.get("feedback") or {}
    if feedback:
        parts.append("<h3>Spectrum feedback</h3>")
        parts.append(f"<p>{_esc(feedback.get('overall_character', ''))}</p>")
        if feedback.get("highlights"):
            parts.append("<ul>")
            for h in feedback.get("highlights") or []:
                parts.append(f"<li>{_esc(h)}</li>")
            parts.append("</ul>")
        if feedback.get("draft_guidance"):
            parts.append("<p><strong>Draft guidance</strong></p><ul>")
            for d in feedback.get("draft_guidance") or []:
                parts.append(f"<li>{_esc(d)}</li>")
            parts.append("</ul>")

    parts.append("<p class='dl-row'>")
    extra: list[tuple[str, str]] = []
    for key, label in (
        ("selected", "Selected peaks CSV"),
        ("all_candidates", "All candidates CSV"),
        ("suppressed", "Suppressed CSV"),
        ("transmittance_minima", "Transmittance minima CSV"),
        ("manual", "Manual peaks CSV"),
    ):
        path = (paper_manifest.get("peak_tables") or {}).get(key)
        if path:
            extra.append((label, str(path)))
    ov_path = paper_manifest.get("label_overrides_path") or ""
    if ov_path:
        extra.append(("Label overrides JSON", str(ov_path)))
    fb_files = feedback.get("files") or {}
    for key, label in (("txt", "Feedback TXT"), ("md", "Feedback MD")):
        path = fb_files.get(key)
        if path:
            extra.append((label, str(path)))
    parts.append(format_download_links([], report_dir, extra=extra))
    parts.append("</p>")

    parts.append("</section>")
    return "".join(parts)


def _peak_table_from_csv(csv_path: str, report_dir: Path) -> str:
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        return "<p class='muted'>No selected peaks exported.</p>"
    hdr = (
        "<table class='tbl tbl-zebra small'><thead><tr>"
        "<th>Wavenumber (cm⁻¹)</th><th>Intensity</th><th>Prominence</th>"
        "<th>Region</th><th>Assignment</th></tr></thead><tbody>"
    )
    body = "".join(
        f"<tr><td>{_esc(r.get('wavenumber_cm1',''))}</td>"
        f"<td>{_esc(r.get('intensity',''))}</td>"
        f"<td>{_esc(r.get('prominence',''))}</td>"
        f"<td>{_esc(r.get('region',''))}</td>"
        f"<td>{_esc(r.get('assignment_if_available','')) or '—'}</td></tr>"
        for r in rows[:12]
    )
    dl = _rel_href(csv_path, report_dir)
    return hdr + body + f"</tbody></table><p><a href='{_esc(dl)}' download>Full selected peaks CSV</a></p>"


def write_manuscript_report_html(
    *,
    out_path: Path,
    page_title: str,
    sections: list[dict[str, Any]],
    paper_manifests: list[dict[str, Any]],
    report_dir: Path | None = None,
    region_stacks_html: str = "",
) -> Path:
    """Write concise MANUSCRIPT_REPORT.html beside the main product report."""
    report_dir = (report_dir or out_path.parent).resolve()
    out_path = out_path.resolve()
    if out_path.name.upper() != "MANUSCRIPT_REPORT.HTML":
        out_path = out_path.parent / "MANUSCRIPT_REPORT.html"
    manifest_by_stem = {m.get("stem"): m for m in paper_manifests}

    body_parts: list[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>",
        f"<title>{_esc(page_title)} — manuscript figures</title>",
        "<style>",
        "body{font-family:Georgia,'Times New Roman',serif;margin:0;padding:24px 32px 48px;color:#111;max-width:920px;}",
        "h1{font-size:1.4rem;margin:0 0 8px;}",
        ".muted{color:#555;font-size:0.92rem;}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:16px 18px;margin:24px 0;}",
        "h2{font-size:1.15rem;margin:0 0 10px;}",
        "h3{font-size:1rem;margin:16px 0 8px;}",
        ".tbl{border-collapse:collapse;width:100%;font-size:0.88rem;}",
        ".tbl th,.tbl td{border:1px solid #ccc;padding:5px 8px;text-align:left;}",
        ".tbl th{background:#f5f5f5;}",
        "a{color:#0072bd;}",
        ".dl-row{font-size:0.85rem;}",
        "</style></head><body>",
        f"<h1>{_esc(page_title)}</h1>",
        "<p class='muted'>Manuscript-ready FTIR figures with improved peak picking and per-spectrum feedback.</p>",
    ]

    for sec in sections:
        name = sec.get("name", "")
        anchor = sec.get("anchor", "spec")
        m = sec.get("paper_manifest") or manifest_by_stem.get(_safe_stem(name))
        if not m:
            continue
        body_parts.append(
            build_manuscript_spectrum_section(
                name=name,
                anchor=anchor,
                paper_manifest=m,
                report_dir=report_dir,
            )
        )

    if region_stacks_html:
        body_parts.append(region_stacks_html)

    body_parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(body_parts), encoding="utf-8")
    return out_path
