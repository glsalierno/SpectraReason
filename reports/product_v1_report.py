"""
product_v1 — polished, spectrum-centric FTIR interpretation presentation layer.

Chemistry engine unchanged; this module curates hierarchy, density, and visual UX.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from reports.kronecker_pi_layout import (
    _esc,
    _truncate,
    collect_pipeline_cautions,
    details_block,
    families_and_specifics_from_assignments,
    ml_agreement_summary,
    spectrum_summary_row,
    split_major_minor_cautions,
    top_supported_assignments,
)
from reports.report_render import region_activity

# Stable HTML markers (see reports/REPORT_PRODUCT_CONTRACT.md)
MARKER_PRODUCT_INTERPRETATION = "<!-- report-feature:product-interpretation -->"
MARKER_KEY_EVIDENCE = "<!-- report-feature:key-evidence -->"
MARKER_SPECTRUM_ANNOTATIONS = "<!-- report-feature:spectrum-annotations -->"
MARKER_PRODUCT_DETAILS = "<!-- report-feature:product-details -->"
MARKER_PRODUCT_AUDIT = "<!-- report-feature:product-audit -->"
MARKER_METADATA_HIDDEN = "<!-- report-feature:metadata-hidden -->"

PRODUCT_V1_CSS = """
body.product-v1 { background: #f4f6f9; }
body.product-v1 main { max-width: 1280px; }
body.product-v1 .card.spec-card { border: none; box-shadow: 0 2px 12px rgba(15,23,42,0.06); padding: 20px 22px 28px; margin: 28px 0 40px; }
body.product-v1 .plot-wrap { margin: 0 0 8px; }
body.product-v1 .plot-wrap .plotly-graph-div { min-height: 520px !important; }
body.product-v1 .product-hint { font-size: 0.82rem; color: #64748b; margin: 0 0 14px; }
body.product-v1 .interp-panel { background: linear-gradient(180deg, #fafbfc 0%, #fff 100%); border: 1px solid #e8edf3; border-radius: 12px; padding: 16px 18px; margin: 12px 0 18px; }
body.product-v1 .interp-panel h3 { margin: 0 0 12px; font-size: 1rem; font-weight: 650; color: #1e293b; }
body.product-v1 .interp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
body.product-v1 .interp-block label { display: block; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; font-weight: 600; margin-bottom: 6px; }
body.product-v1 .chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
body.product-v1 .chip { display: inline-block; font-size: 0.78rem; font-weight: 500; padding: 4px 10px; border-radius: 999px; background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; }
body.product-v1 .chip-strong { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
body.product-v1 .chip-supported { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
body.product-v1 .chip-tentative { background: #fffbeb; color: #b45309; border-color: #fde68a; }
body.product-v1 .chip-caution { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
body.product-v1 .chip-overlap { background: #f8fafc; color: #475569; border-color: #cbd5e1; }
body.product-v1 .chip-group-hydroxy { border-left: 3px solid #0d9488; }
body.product-v1 .chip-group-carbonyl { border-left: 3px solid #b45309; }
body.product-v1 .chip-group-aromatic { border-left: 3px solid #7c3aed; }
body.product-v1 .chip-group-nitrogen { border-left: 3px solid #059669; }
body.product-v1 .chip-group-silicon { border-left: 3px solid #64748b; }
body.product-v1 .caution-compact { margin: 0; padding: 0; list-style: none; font-size: 0.82rem; color: #57534e; }
body.product-v1 .caution-compact li { margin: 4px 0; }
body.product-v1 .key-evidence-section { margin: 16px 0 20px; }
body.product-v1 .key-evidence-section h3 { font-size: 0.95rem; font-weight: 650; margin: 0 0 8px; }
body.product-v1 .tbl-key-evidence { font-size: 11.5px; }
body.product-v1 .tbl-key-evidence th { background: #f1f5f9; font-weight: 600; }
body.product-v1 .status-supported { color: #047857; font-weight: 600; }
body.product-v1 .status-tentative { color: #b45309; font-weight: 600; }
body.product-v1 .status-overlap { color: #64748b; font-weight: 600; }
body.product-v1 .product-details { margin-top: 16px; border-top: 1px solid #eef2f7; padding-top: 12px; }
body.product-v1 .product-details > summary { font-size: 0.9rem; font-weight: 600; color: #475569; cursor: pointer; }
body.product-v1 .product-audit { margin-top: 8px; }
body.product-v1 .product-audit > summary { font-size: 0.88rem; color: #64748b; }
body.product-v1 .summary-table-section.product-summary .tbl-summary th,
body.product-v1 .summary-table-section.product-summary .tbl-summary td { font-size: 11.5px; padding: 9px 10px; }
body.product-v1 .ml-status-line { font-size: 0.82rem; color: #64748b; margin: 8px 0 12px; }
body.product-v1 .confidence-banner { display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 6px; margin-bottom: 10px; }
body.product-v1 .conf-strong { background: #d1fae5; color: #065f46; }
body.product-v1 .conf-supported { background: #dbeafe; color: #1e40af; }
body.product-v1 .conf-tentative { background: #fef3c7; color: #92400e; }
body.product-v1 .conf-overlap { background: #f1f5f9; color: #475569; }
"""

_CHEMISTRY_LABELS: dict[str, str] = {
    "aromatic_system": "Aromatic system",
    "aromatic_family": "Aromatic family",
    "hydroxy_containing": "Hydroxy-containing",
    "hydroxy_family": "Hydroxy family",
    "carbonyl_containing": "Carbonyl-containing",
    "carbonyl_family": "Carbonyl family",
    "nitrogen_containing": "Nitrogen-containing",
    "nitrogen_family": "Nitrogen family",
    "C_O_containing": "C–O-containing",
    "ether_C_O_family": "Ether / C–O family",
    "unsaturation_possible": "Unsaturation",
    "unsaturation_family": "Unsaturation family",
    "fingerprint_C_O_or_Si_O_overlap": "C–O / Si–O overlap",
    "triple_bond_region_possible": "Triple-bond region",
    "aliphatic_CH_present": "C–H stretch observed",
    "aliphatic_CH_region": "Aliphatic C–H",
    "aromatic_CH_region": "Aromatic C–H",
    "upper_mid_activity_region": "Upper mid-IR activity",
    "phenol": "Phenol",
    "alcohol": "Alcohol",
    "ester": "Ester",
    "amide": "Amide",
    "ketone": "Ketone",
    "aldehyde": "Aldehyde",
    "carboxylic_acid": "Carboxylic acid",
    "ether": "Ether",
    "aryl_ether": "Aryl ether",
    "nitrile": "Nitrile",
    "nitro": "Nitro",
    "amine": "Amine",
    "siloxane": "Siloxane",
    "silicone_or_silane": "Silicone / silane",
}

_CHIP_GROUP: dict[str, str] = {
    "hydroxy": "hydroxy",
    "alcohol": "hydroxy",
    "phenol": "hydroxy",
    "carbonyl": "carbonyl",
    "ester": "carbonyl",
    "amide": "carbonyl",
    "ketone": "carbonyl",
    "aldehyde": "carbonyl",
    "carboxylic": "carbonyl",
    "aromatic": "aromatic",
    "indole": "aromatic",
    "pyrrole": "aromatic",
    "benzene": "aromatic",
    "nitrile": "nitrogen",
    "nitro": "nitrogen",
    "amine": "nitrogen",
    "nitrogen": "nitrogen",
    "siloxane": "silicon",
    "silicone": "silicon",
    "si-o": "silicon",
    "si–o": "silicon",
}

_CAUTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"moisture[- ]like.*o.?h", re.I), "⚠ O–H region may include moisture overlap."),
    (re.compile(r"dense fingerprint.*c.?o.*overlap", re.I), "⚠ C–O / Si–O overlap region."),
    (re.compile(r"siloxane.*ether.*ester", re.I), "⚠ C–O / Si–O overlap region."),
    (re.compile(r"atr[- ]sensitive", re.I), "⚠ ATR-sensitive overlap region."),
    (re.compile(r"co2", re.I), "⚠ CO₂ region elevated."),
    (re.compile(r"water vapor|moisture", re.I), "⚠ Possible moisture / vapor bands."),
    (re.compile(r"fingerprint crowd", re.I), "⚠ Fingerprint crowding."),
    (re.compile(r"ml-only", re.I), "⚠ ML advisory only (weak band evidence)."),
]

_REGION_ANNOTATION_LABELS: dict[str, str] = {
    "oh_nh": "O–H / N–H",
    "nh_ch_transition": "N–H / Ar C–H",
    "ch_stretch": "C–H",
    "aliphatic_ch": "aliphatic C–H",
    "aromatic_ch": "aromatic C–H",
    "upper_mid_activity": "upper mid-IR",
    "triple_bond": "C≡C / C≡N",
    "carbonyl_overtone_gap": "overtone / combo",
    "carbonyl": "C=O",
    "unsat_mid": "Aromatic / C=C",
    "co_fingerprint": "C–O / fingerprint",
    "fingerprint_low": "Fingerprint",
}

_CONFIDENCE_SUMMARY: dict[str, tuple[str, str]] = {
    "strong": ("Strong evidence", "conf-strong"),
    "supported": ("Supported with overlap", "conf-supported"),
    "tentative": ("Tentative subclass", "conf-tentative"),
    "local_possible": ("Local overlap only", "conf-overlap"),
    "local_motif_only": ("Local overlap only", "conf-overlap"),
    "not_supported": ("Weak / not supported", "conf-overlap"),
    "artifact_limited": ("Artifact-limited", "conf-overlap"),
}


def chemistry_label(lab: str, pipeline: dict[str, Any], ent: dict[str, Any] | None = None) -> str:
    from ml.ftir_guardrails import SILICONE_FG_LABELS, _match_map, _silicon_evidence_region_count
    from reports.v4_evidence_report import format_summary_assignment_label

    lb = str(lab)
    ent = ent or ((pipeline.get("rule_assignments") or {}).get("assignments") or {}).get(lb) or {}
    if lb in SILICONE_FG_LABELS:
        n_si = _silicon_evidence_region_count(_match_map(pipeline.get("evidence") or {}), 0.08)
        if n_si < 2:
            return format_summary_assignment_label(lb, ent, pipeline)
    return _CHEMISTRY_LABELS.get(lb, lb.replace("_", " ").title())


def compress_caution(text: str, *, max_len: int = 72) -> str:
    raw = str(text).strip()
    if ":" in raw:
        raw = raw.split(":", 1)[1].strip()
    for pat, repl in _CAUTION_PATTERNS:
        if pat.search(raw):
            return _truncate(repl, max_len)
    return _truncate(raw, max_len)


def _chip_class_for_label(lab: str, cc: str) -> str:
    low = lab.lower()
    grp = "chip"
    for key, g in _CHIP_GROUP.items():
        if key in low:
            grp = f"chip chip-group-{g}"
            break
    cc_l = str(cc or "").lower()
    if cc_l == "strong":
        return f"{grp} chip-strong"
    if cc_l == "supported":
        return f"{grp} chip-supported"
    if cc_l in ("tentative", "local_possible", "local_motif_only"):
        return f"{grp} chip-tentative"
    return grp


def _status_display(ent: dict[str, Any], pipeline: dict[str, Any], lab: str) -> str:
    from ml.ftir_guardrails import SILICONE_FG_LABELS, _match_map, _silicon_evidence_region_count

    cc = str(ent.get("confidence_class") or "").lower()
    if lab in SILICONE_FG_LABELS:
        n_si = _silicon_evidence_region_count(_match_map(pipeline.get("evidence") or {}), 0.08)
        if n_si < 2:
            return "Local overlap"
    mapping = {
        "strong": "Supported",
        "supported": "Supported",
        "tentative": "Tentative",
        "local_possible": "Local overlap",
        "local_motif_only": "Local overlap",
        "artifact_limited": "Artifact-limited",
    }
    return mapping.get(cc, "—")


def extract_product_interpretation(pipeline: dict[str, Any], *, ml_enabled: bool) -> dict[str, Any]:
    from reports.v4_evidence_report import evidence_ranked_assignments

    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    ranked = evidence_ranked_assignments(pipeline, top_n=16)
    fam_text, spec_text = families_and_specifics_from_assignments(pipeline)
    main_chem: list[str] = []
    for part in fam_text.split(","):
        p = part.strip()
        if p and p != "—" and p not in main_chem:
            main_chem.append(chemistry_label(p, pipeline))
    likely: list[tuple[str, str]] = []
    for lab, ent in ranked[:8]:
        oc = str(ent.get("ontology_category") or ent.get("assignment_type") or "")
        if oc in ("local_motif", "artifact"):
            continue
        if oc in ("fallback", "family") or lab in main_chem:
            continue
        from ml.ftir_guardrails import SILICONE_FG_LABELS, _match_map, _silicon_evidence_region_count

        if lab in SILICONE_FG_LABELS and _silicon_evidence_region_count(_match_map(pipeline.get("evidence") or {}), 0.08) < 2:
            continue
        disp = chemistry_label(lab, pipeline, ent)
        cc = str(ent.get("confidence_class") or "")
        suffix = "tentative" if cc == "tentative" else ("supported" if cc in ("strong", "supported") else "partial")
        likely.append((disp, suffix))
    caut_raw = collect_pipeline_cautions(pipeline, max_total=12)
    maj, _ = split_major_minor_cautions(caut_raw)
    cautions: list[str] = []
    seen: set[str] = set()
    for c in maj[:6]:
        short = compress_caution(c)
        if short not in seen:
            seen.add(short)
            cautions.append(short)
    best_cc = "not_supported"
    for _lab, ent in ranked[:5]:
        cc = str(ent.get("confidence_class") or "")
        if cc == "strong":
            best_cc = "strong"
            break
        if cc == "supported" and best_cc not in ("strong",):
            best_cc = "supported"
        elif cc == "tentative" and best_cc not in ("strong", "supported"):
            best_cc = "tentative"
    conf_title, conf_cls = _CONFIDENCE_SUMMARY.get(best_cc, ("Interpretation", "conf-overlap"))
    return {
        "main_chemistry": main_chem[:5],
        "likely_specific": likely[:5],
        "cautions": cautions[:4],
        "ml_status": ml_agreement_summary(pipeline, ml_enabled=ml_enabled),
        "confidence_title": conf_title,
        "confidence_class": conf_cls,
    }


def build_interpretation_panel_html(
    pipeline: dict[str, Any],
    *,
    anchor: str,
    ml_enabled: bool,
) -> str:
    data = extract_product_interpretation(pipeline, ml_enabled=ml_enabled)
    main_chips = "".join(f"<span class='chip chip-supported'>{_esc(c)}</span>" for c in data["main_chemistry"]) or "<span class='muted'>—</span>"
    spec_chips = "".join(
        f"<span class='{_chip_class_for_label(l, s)}'>{_esc(l)} ({_esc(s)})</span>" for l, s in data["likely_specific"]
    ) or "<span class='muted'>—</span>"
    caut_html = ""
    if data["cautions"]:
        items = "".join(f"<li>{_esc(c)}</li>" for c in data["cautions"])
        caut_html = f"<ul class='caution-compact'>{items}</ul>"
    else:
        caut_html = "<span class='muted'>No major cautions.</span>"
    ml = _esc(data["ml_status"]) if data["ml_status"] != "N/A" else ""
    ml_block = f"<p class='ml-status-line'><b>ML</b> {ml}</p>" if ml else ""
    conf = (
        f"<span class='confidence-banner {data['confidence_class']}'>"
        f"{_esc(data['confidence_title'])}</span>"
    )
    return (
        MARKER_PRODUCT_INTERPRETATION
        + f"<div class='interp-panel' id='{_esc(anchor)}-interpretation'>"
        + f"<h3>Interpretation</h3>{conf}"
        + "<div class='interp-grid'>"
        + f"<div class='interp-block'><label>Main supported chemistry</label><div class='chip-row'>{main_chips}</div></div>"
        + f"<div class='interp-block'><label>Likely specific assignments</label><div class='chip-row'>{spec_chips}</div></div>"
        + f"<div class='interp-block'><label>Main cautions</label>{caut_html}</div>"
        + "</div></div>"
        + ml_block
    )


def _local_motif_key_rows(pipeline: dict[str, Any]) -> list[str]:
    """Concise key-evidence rows for C–H / upper-mid local motifs (not strong FG)."""
    evidence = pipeline.get("evidence") or {}
    lm = evidence.get("local_motifs") or {}
    rows: list[str] = []
    specs = (
        ("aliphatic_CH_region", "C–H stretch observed (aliphatic)", "2920–2965", 0.10),
        ("aromatic_CH_region", "Aromatic C–H stretch observed", "3000–3100", 0.10),
        ("upper_mid_activity_region", "Upper mid-IR activity", "2260–2800", 0.12),
    )
    for key, label, bands, thr in specs:
        block = lm.get(key) or {}
        sc = float(block.get("support_score", 0) or 0)
        if sc < thr:
            continue
        rows.append(
            f"<tr><td>{_esc(label)}</td><td>{_esc(bands)} cm⁻¹</td>"
            f"<td class='status-overlap'>Local motif</td>"
            f"<td class='muted'>Region activity; not specific alone</td></tr>"
        )
    return rows[:3]


def build_key_evidence_table_html(pipeline: dict[str, Any], *, anchor: str = "", top_n: int = 8) -> str:
    from reports.v4_evidence_report import evidence_ranked_assignments

    ranked = evidence_ranked_assignments(pipeline, top_n=top_n)
    rows: list[str] = _local_motif_key_rows(pipeline)
    for lab, ent in ranked:
        oc = str(ent.get("ontology_category") or "")
        if oc in ("local_motif", "artifact"):
            continue
        disp = chemistry_label(lab, pipeline, ent)
        bands = ent.get("supporting_bands") or ent.get("matched_bands") or []
        band_str = "—"
        if bands:
            parts = []
            for b in bands[:4]:
                if isinstance(b, dict) and b.get("wn_cm1"):
                    parts.append(f"{float(b['wn_cm1']):.0f}")
                elif isinstance(b, (int, float)):
                    parts.append(f"{float(b):.0f}")
                else:
                    parts.append(str(b)[:20])
            band_str = ", ".join(parts) + " cm⁻¹"
        status = _status_display(ent, pipeline, lab)
        st_cls = "status-supported" if status == "Supported" else ("status-tentative" if status == "Tentative" else "status-overlap")
        notes = compress_caution((ent.get("competing_explanation") or ent.get("rationale") or "")[:200] or "—", max_len=80)
        if notes == "—" and ent.get("missing_expected_bands"):
            notes = "Incomplete pairing"
        rows.append(
            f"<tr><td>{_esc(disp)}</td><td>{_esc(band_str)}</td>"
            f"<td class='{st_cls}'>{_esc(status)}</td><td class='muted'>{_esc(notes)}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4' class='muted'>No key evidence rows above threshold.</td></tr>")
    eid = f"{_esc(anchor)}-key-evidence" if anchor else "key-evidence"
    return (
        MARKER_KEY_EVIDENCE
        + f"<section class='key-evidence-section' id='{eid}'>"
        + "<h3>Key evidence</h3>"
        + "<div class='table-scroll'><table class='tbl tbl-zebra tbl-key-evidence'><thead><tr>"
        + "<th>Functional group</th><th>Key bands</th><th>Status</th><th>Notes</th>"
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def build_product_summary_table_html(*, rows: list[dict[str, Any]], ml_enabled: bool) -> str:
    from reports.report_render import MARKER_SUMMARY_TABLE

    parts = [
        MARKER_SUMMARY_TABLE,
        "<section id='summary-table' class='summary-table-section product-summary card'>",
        "<h2 class='summary-table-heading'>Summary</h2>",
        "<div class='table-scroll summary-table-wrap'>",
        "<table class='tbl tbl-zebra tbl-sticky tbl-summary'><thead><tr>",
        "<th>Spectrum</th><th>Main chemistry</th><th>Specific assignments</th>",
        "<th>Main caution</th><th>Confidence</th>",
    ]
    if ml_enabled:
        parts.append("<th>ML</th>")
    parts.append("<th></th></tr></thead><tbody>")
    for r in rows:
        pipeline = r.get("_pipeline") or {}
        interp = extract_product_interpretation(pipeline, ml_enabled=ml_enabled) if pipeline else {}
        main = _esc(", ".join(interp.get("main_chemistry") or []) or str(r.get("families_text") or "—")[:80])
        spec = _esc(
            ", ".join(f"{a} ({b})" for a, b in (interp.get("likely_specific") or [])[:3])
            or str(r.get("specifics_text") or "—")[:80]
        )
        caut = _esc((interp.get("cautions") or [""])[0] if interp.get("cautions") else "—")
        conf = _esc(interp.get("confidence_title", "—"))
        parts.append("<tr>")
        parts.append(f"<td class='td-spectrum'>{_esc(r['name'])}</td>")
        parts.append(f"<td>{main}</td><td>{spec}</td><td>{caut}</td><td>{conf}</td>")
        if ml_enabled:
            parts.append(f"<td>{_esc(str(r.get('ml_agree', '—')))}</td>")
        parts.append(f"<td><a href='#{_esc(r['anchor'])}'>View</a></td></tr>")
    parts.append("</tbody></table></div></section>")
    return "".join(parts)


_CH_ANNOTATION_REGIONS = frozenset({"aliphatic_ch", "aromatic_ch", "ch_stretch", "nh_ch_transition"})
_FAINT_ANNOTATION_REGIONS = frozenset({"upper_mid_activity", "carbonyl_overtone_gap"})


def region_annotation_specs(
    pipeline: dict[str, Any],
    wn: np.ndarray,
    y: np.ndarray,
    *,
    max_annotations: int = 6,
    min_rel: float = 0.10,
) -> list[dict[str, Any]]:
    from ml.ftir_interpretable_features import SPECTRUM_SHADE_EVIDENCE_KEYS, SPECTRUM_SHADE_REGIONS

    evidence = pipeline.get("evidence") or {}
    y_max = float(np.nanmax(y)) if y.size else 1.0
    specs: list[tuple[float, dict[str, Any]]] = []
    for rid, lo, hi, _label in SPECTRUM_SHADE_REGIONS:
        keys = SPECTRUM_SHADE_EVIDENCE_KEYS.get(rid, ())
        rel = region_activity(evidence, wn, y, float(lo), float(hi), evidence_keys=keys)
        if rid in _CH_ANNOTATION_REGIONS:
            thr = 0.04
        elif rid in _FAINT_ANNOTATION_REGIONS:
            thr = 0.05
        else:
            thr = min_rel
        if rel < thr:
            continue
        cx = (float(lo) + float(hi)) / 2.0
        mask = (wn >= lo) & (wn <= hi)
        y_peak = float(np.nanmax(y[mask])) if mask.any() else y_max * 0.5
        text = _REGION_ANNOTATION_LABELS.get(rid, rid)
        specs.append((rel, {"x": cx, "y": y_peak, "text": text, "rid": rid}))
    specs.sort(key=lambda t: -t[0])
    return [s[1] for s in specs[:max_annotations]]


def add_spectrum_annotations(
    fig: Any,
    annotations: list[dict[str, Any]],
    *,
    y_max: float,
    row: int = 1,
    col: int = 1,
) -> None:
    if not annotations:
        return
    n = len(annotations)
    for i, ann in enumerate(annotations):
        y_off = 28 + (i % 3) * 22
        fig.add_annotation(
            x=ann["x"],
            y=ann["y"] + 0.04 * y_max,
            xref="x",
            yref="y",
            text=ann["text"],
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor="#94a3b8",
            ax=0,
            ay=-y_off,
            font=dict(size=10, color="#334155"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            row=row,
            col=col,
        )


def add_band_shading_labels(
    fig: Any,
    shade_names: list[str],
    shapes: list[dict[str, Any]],
    *,
    row: int = 1,
    col: int = 1,
) -> None:
    for i, sh in enumerate(shapes[:8]):
        if i >= len(shade_names):
            break
        x0, x1 = float(sh.get("x0", 0)), float(sh.get("x1", 0))
        y1 = float(sh.get("y1", 0))
        fig.add_annotation(
            x=(x0 + x1) / 2.0,
            y=y1,
            xref="x",
            yref="y",
            text=shade_names[i].split(" (")[0],
            showarrow=False,
            font=dict(size=8, color="#64748b"),
            bgcolor="rgba(255,255,255,0.6)",
            borderpad=2,
            row=row,
            col=col,
        )


def build_peak_labeling_summary_html(peak_picking: dict[str, Any] | None) -> str:
    """Debug/audit: detected vs displayed vs labeled and unlabeled reason counts."""
    pp = peak_picking or {}
    if not pp:
        return ""
    unlabeled = pp.get("unlabeled_reason_counts") or {}
    unlab_txt = ""
    if unlabeled:
        unlab_txt = "<br>Unlabeled: " + ", ".join(
            f"{k}={v}" for k, v in sorted(unlabeled.items())
        )
    preset = pp.get("peak_label_preset") or "—"
    return (
        "<p class='muted peak-labeling-summary'>"
        "<b>Peak labeling</b> "
        f"detected={pp.get('detected_peaks_count', pp.get('n_detected_peaks', '—'))} · "
        f"displayed={pp.get('displayed_peaks_count', pp.get('n_plotted_peaks', '—'))} · "
        f"labeled={pp.get('labeled_peaks_count', pp.get('n_labeled_peaks', '—'))}"
        f"<br>Label preset: {preset}; "
        f"label height ≥ {pp.get('peak_label_min_height', '—')}, "
        f"prominence ≥ {pp.get('peak_label_min_prominence', '—')}"
        f"{unlab_txt}"
        "</p>"
    )


def build_peak_picking_summary_html(peak_picking: dict[str, Any] | None) -> str:
    """Visible peak detect / plot / label counts for Report Details."""
    pp = peak_picking or {}
    if not pp:
        return ""
    reasons = pp.get("label_reason_counts") or {}
    reason_txt = ""
    if reasons:
        reason_txt = " · label reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    return (
        "<p class='muted peak-pick-summary'>"
        f"Peaks: <b>{pp.get('n_detected_peaks', '—')}</b> detected · "
        f"<b>{pp.get('n_plotted_peaks', '—')}</b> plotted · "
        f"<b>{pp.get('n_labeled_peaks', '—')}</b> labeled<br>"
        f"Detection: height ≥ {pp.get('peak_min_height', '—')}, "
        f"prominence ≥ {pp.get('peak_min_prominence', '—')} · "
        f"Label: height ≥ {pp.get('peak_label_min_height', '—')}, "
        f"prominence ≥ {pp.get('peak_label_min_prominence', '—')} "
        f"({pp.get('peak_sensitivity', 'balanced')} sensitivity)"
        f"{reason_txt}"
        "</p>"
    )


def build_product_tables_stack(
    *,
    pipeline: dict[str, Any],
    anchor: str,
    ml_enabled: bool,
    include_evidence: bool,
    audit_html: str,
    band_map_html: str,
    justify_html: str,
    explain_html: str,
    fg_block: str,
    just_block: str,
    meta_html: str,
    density: str,
    peak_picking_html: str = "",
    peak_labeling_html: str = "",
    reproducibility_html: str = "",
) -> str:
    interp = build_interpretation_panel_html(pipeline, anchor=anchor, ml_enabled=ml_enabled)
    key_ev = build_key_evidence_table_html(pipeline, anchor=anchor) if include_evidence else ""
    audit_inner = (
        reproducibility_html
        + peak_labeling_html
        + peak_picking_html
        + band_map_html
        + justify_html
        + explain_html
        + fg_block
        + just_block
        + audit_html
    )
    details = (
        MARKER_PRODUCT_DETAILS
        + f"<details class='product-details' id='{_esc(anchor)}-details'>"
        + "<summary>Details — band map, full assignments, diagnostics</summary>"
        + f"<div class='product-details-body'>{audit_inner}</div>"
        + "</details>"
    )
    meta_hidden = MARKER_METADATA_HIDDEN + meta_html
    return interp + peak_picking_html + key_ev + details + meta_hidden


def enrich_summary_rows(rows: list[dict[str, Any]], pipelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r, pl in zip(rows, pipelines):
        r2 = dict(r)
        r2["_pipeline"] = pl
        out.append(r2)
    return out
