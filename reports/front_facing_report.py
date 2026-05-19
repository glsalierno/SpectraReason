"""
Front-facing spectroscopist report presentation (concise, spectrum-first).

Debug/audit content remains available via ``report_audience=debug`` or Technical details.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reports.kronecker_pi_layout import _esc, _truncate, collect_pipeline_cautions
from reports.product_v1_report import (
    MARKER_KEY_EVIDENCE,
    MARKER_METADATA_HIDDEN,
    MARKER_PRODUCT_DETAILS,
    chemistry_label,
    compress_caution,
    extract_product_interpretation,
)

MARKER_SPECTROSCOPIST_SUMMARY = "<!-- report-feature:spectroscopist-summary -->"
MARKER_EDITABLE_TEXT = "<!-- report-feature:editable-text -->"
MARKER_FRONT_TECHNICAL = "<!-- report-feature:technical-details -->"
MARKER_QUALITY_LIMITED = "<!-- report-feature:quality-limited -->"

FRONT_EXTRA_CSS = """
body.product-v1.front-audience .plot-wrap .plotly-graph-div { min-height: 560px !important; }
body.product-v1.front-audience .spectroscopist-summary {
  font-size: 1.02rem; line-height: 1.55; color: #1e293b; margin: 14px 0 18px;
  padding: 14px 16px; background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0;
}
body.product-v1.front-audience .caution-front { margin: 0 0 14px; padding: 0; list-style: none; }
body.product-v1.front-audience .caution-front li {
  margin: 6px 0; padding: 8px 12px; background: #fffbeb; border: 1px solid #fde68a;
  border-radius: 8px; font-size: 0.88rem; color: #78350f;
}
body.product-v1.front-audience .ml-check-line { font-size: 0.86rem; color: #64748b; margin: 0 0 16px; }
body.product-v1.front-audience .quality-card {
  padding: 14px 16px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; margin: 12px 0 16px;
}
body.product-v1.front-audience .spec-title-front { font-size: 1.15rem; font-weight: 650; margin: 0 0 4px; color: #0f172a; }
body.product-v1.front-audience .product-details > summary { font-size: 0.92rem; }
body.product-v1.front-audience .interp-panel { display: none; }
body.product-v1.front-audience .peak-pick-summary { display: none; }
body.product-v1.front-audience .peak-labeling-summary { display: none; }
body.product-v1.front-audience .settings-appendix { display: none; }
body.product-v1.front-audience .badge-overlap {
  background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; font-weight: 500;
}
body.product-v1.front-audience.visual-matlab .plot-wrap .plotly-graph-div {
  max-height: 1100px !important;
}
body.product-v1.front-audience .editable-hint {
  font-size: 0.82rem; color: #64748b; margin: 0 0 6px;
}
body.product-v1.front-audience .spectroscopist-summary.editable-report-text {
  cursor: text; outline: none;
}
body.product-v1.front-audience .spectroscopist-summary.editable-report-text:focus {
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35);
  background: #fff;
}
body.product-v1.front-audience .presentation-figures-list {
  font-size: 0.88rem; margin: 8px 0 0; padding-left: 1.2rem;
}
body.product-v1.front-audience .presentation-figures-list code { font-size: 0.8rem; word-break: break-all; }
body.product-v1.front-audience .deconv-candidates-section { margin: 12px 0 18px; }
body.product-v1.front-audience .deconv-candidates-section h3 { font-size: 0.98rem; margin: 0 0 8px; }
body.product-v1.front-audience .deconv-fg-support { margin: 4px 0 8px; }
"""

EDITABLE_TEXT_SCRIPT = """
<script>
(function () {
  var keyPrefix = "ftir-report-edit-";
  document.querySelectorAll(".editable-report-text").forEach(function (el) {
    var key = keyPrefix + (el.id || el.getAttribute("data-edit-key") || "");
    if (!key || key === keyPrefix) return;
    try {
      var saved = localStorage.getItem(key);
      if (saved) el.textContent = saved;
    } catch (e) {}
    el.addEventListener("blur", function () {
      try { localStorage.setItem(key, el.textContent || ""); } catch (e) {}
    });
  });
})();
</script>
"""

_LOCAL_MOTIF_SKIP = frozenset(
    {
        "aliphatic_CH_region",
        "aromatic_CH_region",
        "upper_mid_activity_region",
        "CH_stretch_region",
        "nh_ch_transition_region",
        "Si_O_overlap_region",
        "fingerprint_C_O_or_Si_O_overlap",
        "NO2_asym_region",
        "NO2_sym_region",
        "C_O_fingerprint_region",
        "carbonyl_region",
        "broad_OH_NH_region",
        "nitrile_alkyne_region",
        "amide_II_region",
        "enamine_region",
        "heterocyclic_N_O_region",
        "n_oxide_confounded_region",
        "aromatic_CC_region",
    }
)

_EVIDENCE_PHRASES: dict[str, str] = {
    "oh_nh_broad": "broad O–H/N–H region",
    "aromatic_cc": "aromatic ring modes (1450–1600 cm⁻¹)",
    "aromatic_ch_stretch": "aromatic C–H stretch",
    "aliphatic_ch": "aliphatic C–H stretch",
    "aldehydic_ch": "aldehydic C–H region",
    "carbonyl": "C=O stretch",
    "amide_i": "amide I band",
    "amide_ii": "amide II band",
    "ester_co": "ester C=O",
    "ether_co": "C–O stretch",
    "c_o_stretch": "C–O / fingerprint activity",
    "nitrile_cn": "nitrile stretch",
    "nitro_asym": "nitro asymmetric stretch",
    "siloxane_sio": "Si–O-like band",
}


def resolve_report_audience(
    audience: str | None,
    *,
    report_style: str = "legacy",
    report_density: str = "balanced",
    front_facing_flag: bool = False,
) -> str:
    if audience in ("front", "debug"):
        return str(audience)
    if front_facing_flag:
        return "front"
    if str(report_density or "").lower() == "audit":
        return "debug"
    if str(report_style or "").lower() == "product_v1":
        return "front"
    return "debug"


def is_front_audience(audience: str) -> bool:
    return str(audience or "").lower() == "front"


def front_page_header(*, n_spectra: int, audience: str) -> tuple[str, str]:
    if is_front_audience(audience):
        title = "FTIR spectroscopic interpretation report"
        subtitle = (
            f"{n_spectra} {'spectrum' if n_spectra == 1 else 'spectra'} · "
            "evidence-first interpretation with optional calibrated SVM advisory layer"
        )
        return title, subtitle
    return "Structural FG SVM — interactive Kronecker report", f"{n_spectra} spectra | evidence-first FTIR + Kronecker"


def front_ml_check_line(pipeline: dict[str, Any], *, ml_enabled: bool) -> str:
    if not ml_enabled:
        return "ML check: ML not used (rules and spectral evidence only)."
    consensus = pipeline.get("consensus") or {}
    top = consensus.get("top_labels") or []
    statuses = [str(ent.get("agreement_status") or "") for _lab, ent in top[:12] if ent]
    if not statuses:
        return "ML check: ML advisory only; rules carry the interpretation."
    if any(s == "ml_only_warning" for s in statuses):
        return "ML check: ML advisory only; band evidence should lead subclass calls."
    if any(s == "conflict" for s in statuses):
        return "ML check: Mixed evidence; some rule and ML calls disagree."
    n_agree = sum(1 for s in statuses if s == "rule_and_ml_agree")
    if n_agree >= max(1, len(statuses) // 2):
        return "ML check: Rules + ML agree on several leading assignments."
    if all(s == "rule_only" for s in statuses):
        return "ML check: Rules dominant; ML did not change interpretation."
    return "ML check: Family/specific ML is advisory; confirm with band evidence."


def _artifact_quality_flags(evidence: dict[str, Any]) -> list[str]:
    flags = (evidence.get("artifacts") or {}).get("flags") or {}
    out: list[str] = []
    if flags.get("saturation_or_clipping"):
        out.append("Possible saturation/clipping; peak intensities may be unreliable.")
    if flags.get("fingerprint_crowding"):
        out.append("Fingerprint region crowded; subclass calls are tentative.")
    if flags.get("water_vapor_or_moisture_like"):
        out.append("O–H region may include moisture contribution.")
    if flags.get("co2_region_elevated"):
        out.append("CO₂ region elevated; check nitrile/alkyne region calls.")
    if flags.get("atr_crystal_fingerprint_overlap"):
        out.append("ATR/crystal fingerprint overlap may affect low-wavenumber bands.")
    return out


def spectrum_quality_limited(pipeline: dict[str, Any]) -> tuple[bool, str]:
    from reports.v4_evidence_report import evidence_ranked_assignments

    evidence = pipeline.get("evidence") or {}
    ranked = [
        (lab, ent)
        for lab, ent in evidence_ranked_assignments(pipeline, top_n=12)
        if str(ent.get("ontology_category") or "") not in ("local_motif", "artifact")
        and float(ent.get("score", 0) or 0) >= 0.22
    ]
    arts = _artifact_quality_flags(evidence)
    if arts and not ranked:
        return True, arts[0]
    if arts and len(ranked) <= 1:
        return True, arts[0] + " Leading assignments are limited."
    return False, ""


def format_key_spectral_evidence(ent: dict[str, Any], pipeline: dict[str, Any], lab: str) -> str:
    """Human-readable evidence phrase (no raw tuple dumps)."""
    phrases: list[str] = []
    for line in ent.get("evidence") or []:
        if isinstance(line, str) and line.strip():
            t = line.strip()
            if "(" in t and "support" in t.lower():
                t = re.sub(r"\s*\(support[^)]*\)", "", t, flags=re.I).strip()
            if 20 < len(t) < 120:
                phrases.append(t)
    for raw in ent.get("supporting_bands") or []:
        s = str(raw)
        m = re.search(r"(\d{3,4}(?:\.\d+)?)\s*[-–]\s*(\d{3,4}(?:\.\d+)?)\s*cm", s)
        mode_m = re.search(r"([A-Za-z0-9≡–/ ]+stretch|[A-Za-z0-9≡–/ ]+band|C=O|N–H|O–H)", s, re.I)
        if m:
            lo, hi = int(float(m.group(1))), int(float(m.group(2)))
            mode = mode_m.group(1).strip() if mode_m else ""
            if mode:
                phrases.append(f"{mode} ({lo}–{hi} cm⁻¹)")
            else:
                phrases.append(f"{lo}–{hi} cm⁻¹ activity")
        elif len(s) < 80 and "(" not in s[:5]:
            phrases.append(s)
    if not phrases:
        match_map = {
            str(m.get("band_id")): m
            for m in (pipeline.get("evidence") or {}).get("band_matches") or []
            if m.get("matched")
        }
        for bid, phrase in _EVIDENCE_PHRASES.items():
            if bid in match_map and lab.lower() in str(match_map[bid].get("label", "")).lower():
                phrases.append(phrase)
                break
            if bid in match_map and any(bid in str(lab) for _ in [1]):
                phrases.append(phrase)
    if not phrases:
        return "spectral activity in diagnostic regions"
    seen: set[str] = set()
    uniq: list[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return "; ".join(uniq[:3])


def _interpretation_phrase(ent: dict[str, Any], pipeline: dict[str, Any], lab: str) -> str:
    from reports.front_consensus import front_display_name, nitro_is_supported, professional_confidence_label

    if lab in ("NO2_asym_region", "NO2_sym_region") or (lab == "nitro" and not nitro_is_supported(pipeline)):
        return "N–O / NO₂ overlap (paired nitro not confirmed)"
    prof = professional_confidence_label(ent, pipeline, lab)
    if prof == "Insufficient evidence":
        return "insufficient evidence for standalone assignment"
    cc = str(ent.get("confidence_class") or "").lower()
    oc = str(ent.get("ontology_category") or "").lower()
    disp = front_display_name(lab, pipeline, ent).lower()
    if oc == "local_motif" or cc in ("local_possible", "local_motif_only"):
        return "local overlap / supporting activity only"
    if cc == "strong":
        return f"{disp} supported"
    if cc == "supported":
        return f"{disp} supported with overlap context"
    if cc == "tentative":
        return f"tentative {disp} signal"
    return "weak or ambiguous"


def _confidence_front(ent: dict[str, Any], pipeline: dict[str, Any], lab: str) -> str:
    from reports.front_consensus import professional_confidence_label

    return professional_confidence_label(ent, pipeline, lab)


def build_spectroscopist_summary(pipeline: dict[str, Any], *, ml_enabled: bool) -> str:
    """1–3 sentence prose summary for front-facing readers."""
    limited, lim_msg = spectrum_quality_limited(pipeline)
    if limited:
        return (
            f"This spectrum has limited reliable assignment. {lim_msg} "
            "Use the plot and technical details for band-level review."
        )
    data = extract_product_interpretation(pipeline, ml_enabled=ml_enabled)
    main = data.get("main_chemistry") or []
    cautions = data.get("cautions") or []
    parts: list[str] = []
    if main:
        lead = ", ".join(main[:3])
        parts.append(f"This spectrum is most consistent with {lead}.")
    else:
        parts.append("This spectrum shows limited definitive functional-group support.")
    if data.get("likely_specific"):
        spec_names = [n for n, _s in data["likely_specific"][:2]]
        if spec_names:
            parts.append(f"More specific calls such as {', '.join(spec_names)} remain tentative.")
    if cautions:
        c0 = cautions[0].replace("⚠ ", "").strip()
        parts.append(c0[0].upper() + c0[1:] if c0 else "")
    text = " ".join(parts[:3])
    if len(text) > 420:
        text = text[:417] + "…"
    return text


def build_front_cautions_html(pipeline: dict[str, Any], *, max_items: int = 4) -> str:
    evidence = pipeline.get("evidence") or {}
    seen: set[str] = set()
    items: list[str] = []
    for c in _artifact_quality_flags(evidence):
        short = compress_caution(c, max_len=100).replace("⚠ ", "")
        if short not in seen:
            seen.add(short)
            items.append(short)
    for c in collect_pipeline_cautions(pipeline, max_total=10):
        short = compress_caution(c, max_len=100).replace("⚠ ", "")
        if short not in seen:
            seen.add(short)
            items.append(short)
        if len(items) >= max_items:
            break
    if not items:
        return ""
    lis = "".join(f"<li>{_esc(c)}</li>" for c in items[:max_items])
    return f"<ul class='caution-front'>{lis}</ul>"


def build_front_key_evidence_table_html(
    pipeline: dict[str, Any],
    *,
    anchor: str = "",
    top_n: int = 6,
) -> str:
    from reports.front_consensus import front_display_name, should_show_front_consensus_row
    from reports.v4_evidence_report import evidence_ranked_assignments

    ranked = evidence_ranked_assignments(pipeline, top_n=top_n + 8)
    rows: list[str] = []
    for lab, ent in ranked:
        if lab in _LOCAL_MOTIF_SKIP:
            continue
        if not should_show_front_consensus_row(lab, ent, pipeline):
            continue
        oc = str(ent.get("ontology_category") or "")
        if oc in ("local_motif", "artifact"):
            continue
        cc = str(ent.get("confidence_class") or "").lower()
        if cc == "not_supported":
            continue
        disp = front_display_name(lab, pipeline, ent)
        ev_phrase = format_key_spectral_evidence(ent, pipeline, lab)
        interp = _interpretation_phrase(ent, pipeline, lab)
        conf = _confidence_front(ent, pipeline, lab)
        st_cls = "status-supported" if conf in ("Strong match", "Supported") else (
            "status-tentative" if conf == "Tentative subclass" else "status-overlap"
        )
        rows.append(
            f"<tr><td>{_esc(disp)}</td><td>{_esc(ev_phrase)}</td>"
            f"<td>{_esc(interp)}</td><td class='{st_cls}'>{_esc(conf)}</td></tr>"
        )
        if len(rows) >= top_n:
            break
    if not rows:
        return ""
    eid = f"{_esc(anchor)}-key-evidence" if anchor else "key-evidence"
    return (
        MARKER_KEY_EVIDENCE
        + f"<section class='key-evidence-section' id='{eid}'>"
        + "<h3>Key evidence</h3>"
        + "<div class='table-scroll'><table class='tbl tbl-zebra tbl-key-evidence tbl-front-evidence'>"
        + "<thead><tr><th>Assignment</th><th>Key spectral evidence</th>"
        + "<th>Interpretation</th><th>Confidence</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def build_quality_limited_card(pipeline: dict[str, Any], *, peak_picking: dict[str, Any] | None = None) -> str:
    limited, msg = spectrum_quality_limited(pipeline)
    if not limited and (peak_picking or {}).get("n_detected_peaks", 1) > 0:
        return ""
    pp = peak_picking or {}
    flags = _artifact_quality_flags(pipeline.get("evidence") or {})
    flag_txt = "; ".join(flags[:2]) if flags else "insufficient diagnostic evidence"
    return (
        MARKER_QUALITY_LIMITED
        + "<div class='quality-card'>"
        + "<strong>Limited reliable assignment.</strong> "
        + f"{_esc(msg or flag_txt)} "
        f"Peaks detected: {pp.get('n_detected_peaks', '—')}; "
        f"labeled: {pp.get('n_labeled_peaks', '—')}. "
        "Expand technical details for full diagnostics."
        + "</div>"
    )


def build_front_spec_title(name: str) -> str:
    return f"<h2 class='spec-title-front'>{_esc(name)}</h2>"


def load_interpretation_notes(path: Path | None) -> dict[str, str]:
    """Load optional per-spectrum or global interpretation overrides from a text file."""
    p = Path(path) if path else None
    if not p or not p.is_file():
        return {}
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    if "---" not in raw:
        return {"*": raw}
    out: dict[str, str] = {}
    block: list[str] = []
    current = "*"
    for line in raw.splitlines():
        if line.strip().startswith("## "):
            if block:
                out[current] = "\n".join(block).strip()
            current = line.strip()[3:].strip()
            block = []
        else:
            block.append(line)
    if block:
        out[current] = "\n".join(block).strip()
    return out


def interpretation_override_for_spectrum(
    notes: dict[str, str], spectrum_name: str
) -> str | None:
    if not notes:
        return None
    stem = Path(spectrum_name).stem
    for key in (stem, spectrum_name, "*"):
        if key in notes and notes[key]:
            return notes[key]
    return None


def build_presentation_figures_html(
    files: list[str],
    *,
    report_dir: Path | None = None,
) -> str:
    if not files:
        return ""
    items: list[str] = []
    base = Path(report_dir) if report_dir else None
    for f in sorted(files):
        p = Path(f)
        href = p.name
        if base is not None:
            try:
                href = p.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                href = p.name
        items.append(
            f"<li><a class='btn-dl' href='{_esc(href)}' download>{_esc(p.name)}</a></li>"
        )
    return (
        "<details class='presentation-figures-details'>"
        "<summary>Presentation figures (PNG/SVG/PDF)</summary>"
        f"<ul class='presentation-figures-list'>{''.join(items)}</ul>"
        "</details>"
    )


def sanitize_run_settings_line(line: str) -> str:
    """Strip absolute paths from run settings for front-facing display."""
    from pathlib import Path

    out: list[str] = []
    for token in (line or "").split(" | "):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, val = token.split("=", 1)
            if ":\\" in val or ":/" in val or val.startswith("\\\\"):
                val = Path(val).name
            out.append(f"{key}={val}")
        else:
            out.append(token)
    return " | ".join(out)


def build_front_card_stack(
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
    peak_picking_html: str = "",
    peak_picking_meta: dict[str, Any] | None = None,
    show_metadata: bool = False,
    run_settings_line: str = "",
    reproducibility_html: str = "",
    editable_text: bool = False,
    summary_override: str | None = None,
    presentation_figures_html: str = "",
) -> str:
    if (summary_override or "").strip():
        summary = summary_override.strip()
    else:
        from reports.front_consensus import build_consensus_interpretation_text

        summary = build_consensus_interpretation_text(pipeline, ml_enabled=ml_enabled)
    if editable_text:
        hint = (
            "<p class='editable-hint'>Click the summary to edit. "
            "Changes are saved in this browser (localStorage) until you clear site data.</p>"
        )
        summary_block = (
            MARKER_SPECTROSCOPIST_SUMMARY
            + MARKER_EDITABLE_TEXT
            + hint
            + f"<div class='spectroscopist-summary editable-report-text' "
            f"id='{_esc(anchor)}-summary' contenteditable='true' spellcheck='true'>"
            f"{_esc(summary)}</div>"
        )
    else:
        summary_block = (
            MARKER_SPECTROSCOPIST_SUMMARY
            + f"<div class='spectroscopist-summary' id='{_esc(anchor)}-summary'>"
            f"{_esc(summary)}</div>"
        )
    key_ev = build_front_key_evidence_table_html(pipeline, anchor=anchor) if include_evidence else ""
    if not key_ev and include_evidence:
        key_ev = build_quality_limited_card(pipeline, peak_picking=peak_picking_meta)
        if not key_ev:
            key_ev = (
                MARKER_QUALITY_LIMITED
                + "<div class='quality-card'><strong>Limited key evidence.</strong> "
                "Reliable assignment was limited by weak diagnostic support. "
                "Expand technical details for full band map and diagnostics.</div>"
            )
    from reports.front_consensus import build_front_ambiguity_cards_html

    cautions = build_front_ambiguity_cards_html(pipeline, anchor=anchor) or build_front_cautions_html(pipeline)
    ml_line = f"<p class='ml-check-line'>{_esc(front_ml_check_line(pipeline, ml_enabled=ml_enabled))}</p>"
    settings_block = ""
    if run_settings_line:
        safe = sanitize_run_settings_line(run_settings_line)
        settings_block = (
            "<p class='muted run-settings-inline'><b>Run settings</b> "
            f"<code>{_esc(safe)}</code></p>"
        )
    audit_inner = (
        settings_block
        + reproducibility_html
        + band_map_html
        + justify_html
        + explain_html
        + fg_block
        + just_block
        + audit_html
    )
    if peak_picking_html and show_metadata:
        audit_inner = peak_picking_html + audit_inner
    details = (
        MARKER_FRONT_TECHNICAL
        + MARKER_PRODUCT_DETAILS
        + f"<details class='product-details' id='{_esc(anchor)}-details'>"
        + "<summary>Technical details</summary>"
        + f"<div class='product-details-body'>{audit_inner}</div>"
        + "</details>"
    )
    meta_block = (MARKER_METADATA_HIDDEN + meta_html) if show_metadata else ""
    return (
        summary_block
        + key_ev
        + cautions
        + ml_line
        + presentation_figures_html
        + details
        + meta_block
    )


def build_front_summary_table_html(*, rows: list[dict[str, Any]], ml_enabled: bool) -> str:
    from reports.report_render import MARKER_SUMMARY_TABLE

    parts = [
        MARKER_SUMMARY_TABLE,
        "<section id='summary-table' class='summary-table-section product-summary card front-summary'>",
        "<h2 class='summary-table-heading'>Summary</h2>",
        "<div class='table-scroll summary-table-wrap'>",
        "<table class='tbl tbl-zebra tbl-sticky tbl-summary tbl-front-summary'><thead><tr>",
        "<th>Spectrum</th><th>Main interpretation</th><th>Key evidence</th>",
        "<th>Main caution</th><th>Confidence</th><th></th>",
        "</tr></thead><tbody>",
    ]
    for r in rows:
        pipeline = r.get("_pipeline") or {}
        summary = build_spectroscopist_summary(pipeline, ml_enabled=ml_enabled)
        interp = extract_product_interpretation(pipeline, ml_enabled=ml_enabled) if pipeline else {}
        main = _truncate(summary, 120)
        ev_bits: list[str] = []
        for lab, ent in (pipeline.get("rule_assignments") or {}).get("assignments", {}).items():
            if not isinstance(ent, dict):
                continue
            if str(ent.get("ontology_category") or "") in ("local_motif",):
                continue
            if float(ent.get("score", 0) or 0) < 0.25:
                continue
            ev_bits.append(chemistry_label(lab, pipeline, ent))
            if len(ev_bits) >= 2:
                break
        ev_s = _esc(", ".join(ev_bits) if ev_bits else "—")
        caut = _esc((interp.get("cautions") or ["—"])[0].replace("⚠ ", ""))
        conf = _esc(interp.get("confidence_title", "—"))
        parts.append(
            f"<tr><td class='td-spectrum'>{_esc(r['name'])}</td>"
            f"<td>{_esc(main)}</td><td>{ev_s}</td><td>{caut}</td><td>{conf}</td>"
            f"<td><a href='#{_esc(r['anchor'])}'>View</a></td></tr>"
        )
    parts.append("</tbody></table></div></section>")
    return "".join(parts)
