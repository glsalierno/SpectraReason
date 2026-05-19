"""
v4 evidence-first report helpers: band maps, justification panels, evidence-based ranking.
"""

from __future__ import annotations

from typing import Any

from reports.kronecker_pi_layout import _esc, _truncate

SCORE_TOOLTIP = (
    "Evidence scores are capped rule-support values, not probabilities. "
    "Report ranking score is for table ordering only."
)

_CONF_RANK = {
    "strong": 0,
    "supported": 1,
    "tentative": 2,
    "local_possible": 3,
    "not_supported": 4,
    "local_motif_only": 5,
}
_EVID_RANK = {
    "complete": 0,
    "partial": 1,
    "single_band": 2,
    "conflicting": 3,
    "artifact_limited": 4,
}
_TYPE_RANK = {
    "specific": 0,
    "family": 1,
    "fallback": 2,
    "local_band_only": 3,
    "local_motif": 4,
    "artifact": 5,
}

_PEAK_CATEGORY_COLORS = {
    "hydroxy": "#0d9488",
    "carbonyl": "#b45309",
    "aromatic": "#7c3aed",
    "c_o": "#2563eb",
    "nitro": "#dc2626",
    "nitrile": "#be123c",
    "si_o": "#64748b",
    "amine": "#059669",
    "other": "#475569",
}

_BAND_CATEGORY: dict[str, str] = {
    "broad_oh": "hydroxy",
    "alcohol_oh": "hydroxy",
    "phenol_oh": "hydroxy",
    "acid_oh_broad": "hydroxy",
    "ketone_co": "carbonyl",
    "ester_co": "carbonyl",
    "amide_co": "carbonyl",
    "aldehyde_co": "carbonyl",
    "carboxylic_co": "carbonyl",
    "aromatic_cc": "aromatic",
    "heteroaromatic": "aromatic",
    "ether_co": "c_o",
    "aryl_ether_co": "c_o",
    "phenolic_co": "c_o",
    "ester_co_o": "c_o",
    "siloxane_sio": "si_o",
    "silicone_sic": "si_o",
    "nitro_asym": "nitro",
    "nitro_sym": "nitro",
    "nitrile_cn": "nitrile",
    "amine_nh": "amine",
    "amine_nh2": "amine",
    "amide_nh": "amine",
}


def _th(title: str, *, tip: str = "") -> str:
    if tip:
        return f"<th title='{_esc(tip)}'>{_esc(title)}</th>"
    return f"<th>{_esc(title)}</th>"


def assignment_evidence_rank(ent: dict[str, Any]) -> tuple[int, int, int, float]:
    cc = str(ent.get("confidence_class") or "not_supported").lower()
    ec = str(ent.get("evidence_completeness") or "artifact_limited").lower()
    at = str(ent.get("assignment_type") or ent.get("ontology_category") or "specific").lower()
    if at == "local_motif":
        at = "local_band_only"
    return (
        _CONF_RANK.get(cc, 9),
        _EVID_RANK.get(ec, 9),
        _TYPE_RANK.get(at, 9),
        -float(ent.get("score", 0) or 0),
    )


def evidence_ranked_assignments(
    pipeline: dict[str, Any],
    *,
    min_score: float = 0.12,
    top_n: int = 24,
) -> list[tuple[str, dict[str, Any]]]:
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    rows: list[tuple[str, dict[str, Any]]] = []
    for lab, ent in assigns.items():
        if not isinstance(ent, dict):
            continue
        if float(ent.get("score", 0) or 0) < min_score:
            continue
        rows.append((str(lab), ent))
    rows.sort(key=lambda kv: assignment_evidence_rank(kv[1]))
    return rows[:top_n]


def peak_category(band_id: str) -> str:
    return _BAND_CATEGORY.get(str(band_id), "other")


def peak_annotation_specs(
    peaks_dicts: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    max_peaks: int = 14,
    report_density: str = "balanced",
    include_weak: bool = False,
    y_max: float | None = None,
    y_min: float = 0.0,
    wn_min: float | None = None,
    wn_max: float | None = None,
    peak_label_layout: str = "smart",
    fingerprint_cluster_distance: float | None = 18.0,
    presentation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Numeric peak labels for Plotly (input list should already be label-selected)."""
    from reports.annotation_layout import apply_peak_label_layout, cluster_peaks_for_labeling

    preselected = bool(peaks_dicts) and all(p.get("peak_id") for p in peaks_dicts)
    peaks_input = list(peaks_dicts)
    cluster_stats: dict[str, int] = {}
    if fingerprint_cluster_distance and fingerprint_cluster_distance > 0 and not preselected:
        peaks_input, cluster_stats = cluster_peaks_for_labeling(
            peaks_dicts,
            cluster_distance_cm1=float(fingerprint_cluster_distance),
        )

    match_map = {str(m.get("band_id")): m for m in (evidence.get("band_matches") or [])}
    audit = str(report_density or "").lower() == "audit"
    scored: list[tuple[float, dict[str, Any], str]] = []
    for p in peaks_input:
        q = str(p.get("peak_quality") or "moderate")
        role = str(p.get("peak_role") or "")
        if q == "noise_like":
            continue
        if preselected:
            pass
        elif p.get("label_reason"):
            pass
        elif not audit and not include_weak and q == "weak":
            continue
        elif role not in ("diagnostic_peak", "weak_peak") and not audit and not p.get("label_reason"):
            continue
        wn = float(p.get("wn_cm1", 0) or 0)
        rh = float(p.get("rel_height", p.get("height", 0)) or 0)
        rank = rh * float(p.get("rule_support_weight", 1.0) or 0.0)
        if q == "strong":
            rank *= 1.4
        elif q == "moderate":
            rank *= 1.1
        best_bid = ""
        best_sup = 0.0
        for bid, m in match_map.items():
            if not m.get("matched"):
                continue
            for npk in m.get("peaks_near") or []:
                if isinstance(npk, dict) and abs(float(npk.get("wn_cm1", 0)) - wn) < 14.0:
                    sup = float(m.get("support_score", 0) or 0)
                    if sup > best_sup:
                        best_sup = sup
                        best_bid = bid
        cat = peak_category(best_bid) if best_bid else "other"
        scored.append((rank, p, cat))
    scored.sort(key=lambda t: -t[0])
    raw: list[dict[str, Any]] = []
    for _rh, p, cat in scored[: max_peaks * 2]:
        wn = float(p.get("wn_cm1", 0))
        q = str(p.get("peak_quality") or "moderate")
        lines = [f"<b>{wn:.0f} cm⁻¹</b>"]
        if best_bid := next(
            (bid for bid, m in match_map.items() if m.get("matched") and any(
                isinstance(npk, dict) and abs(float(npk.get("wn_cm1", 0)) - wn) < 14
                for npk in (m.get("peaks_near") or [])
            )),
            "",
        ):
            m = match_map.get(best_bid) or {}
            lines.append(f"Band: {_esc(str(m.get('label') or best_bid))}")
            lines.append(f"Category: {cat}")
        else:
            lines.append("Local peak (no strong band match)")
        raw.append(
            {
                "wn": wn,
                "y": float(p.get("height", p.get("rel_height", 0)) or 0),
                "text": f"{wn:.0f}",
                "color": _PEAK_CATEGORY_COLORS.get(cat, _PEAK_CATEGORY_COLORS["other"]),
                "hover": "<br>".join(lines),
                "category": cat,
                "peak_quality": q,
                "_peak": p,
            }
        )
    ymax = float(y_max) if y_max is not None else max((a["y"] for a in raw), default=1.0)
    cap = max_peaks * 2 if str(peak_label_layout or "smart").lower() == "smart" else max_peaks
    raw_capped = raw[:cap]

    laid, layout_stats = apply_peak_label_layout(
        raw_capped,
        mode=peak_label_layout,
        y_max=ymax,
        y_min=y_min,
        wn_min=wn_min,
        wn_max=wn_max,
        presentation=presentation,
    )
    if cluster_stats:
        layout_stats = {**cluster_stats, **layout_stats}
    layout_stats.setdefault("n_labels", len(raw_capped))
    return laid[:max_peaks], layout_stats


def _labels_for_band(band_id: str, pipeline: dict[str, Any]) -> list[str]:
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    evidence = pipeline.get("evidence") or {}
    out: list[str] = []
    bid = str(band_id)
    for lab, ent in assigns.items():
        if not isinstance(ent, dict):
            continue
        sup = ent.get("supporting_bands") or []
        if any(bid in str(b) for b in sup):
            out.append(str(lab))
    for lab, body in (evidence.get("fg_evidence") or {}).items():
        if bid in (body.get("band_ids") or []) and str(lab) not in out:
            out.append(str(lab))
    return out[:6]


def format_summary_assignment_label(
    lab: str,
    ent: dict[str, Any],
    pipeline: dict[str, Any],
) -> str:
    """Summary-table display name (ATR siloxane wording)."""
    from ml.ftir_atr import atr_sensitive_interpretation
    from ml.ftir_guardrails import SILICONE_FG_LABELS, _silicon_evidence_region_count, _match_map

    if lab not in SILICONE_FG_LABELS:
        return lab
    evidence = pipeline.get("evidence") or {}
    n_si = _silicon_evidence_region_count(_match_map(evidence), 0.08)
    cc = str(ent.get("confidence_class") or "")
    if atr_sensitive_interpretation(evidence) and n_si < 2:
        return "Si-O overlap local only; ATR-sensitive"
    if n_si < 2 and cc in ("local_possible", "local_motif_only", "tentative"):
        return "Si-O overlap local only"
    return lab


def _atr_sensitive_band(band_id: str, evidence: dict[str, Any]) -> bool:
    from ml.ftir_atr import atr_sensitive_interpretation

    if str(band_id) not in ("siloxane_sio", "ether_co", "ester_co_o", "aryl_ether_co", "phenolic_co", "silicone_sic"):
        return False
    return atr_sensitive_interpretation(evidence)


def _support_status_for_labels(labels: list[str], assigns: dict[str, Any], evidence: dict[str, Any]) -> str:
    if not labels:
        return "local only"
    from ml.ftir_atr import atr_sensitive_interpretation
    from ml.ftir_guardrails import _match_map, _silicon_evidence_region_count

    n_si = _silicon_evidence_region_count(_match_map(evidence), 0.08)
    if any(l in ("siloxane", "silicone_or_silane") for l in labels) and n_si < 2:
        if atr_sensitive_interpretation(evidence):
            return "ATR-sensitive overlap"
        return "siloxane local only"
    best = "local only"
    for lab in labels:
        cc = str((assigns.get(lab) or {}).get("confidence_class") or "")
        if cc == "strong":
            return "strong"
        if cc == "supported":
            best = "supported"
        elif cc == "tentative" and best not in ("strong", "supported"):
            best = "tentative"
    return best


def build_band_evidence_map_html(
    pipeline: dict[str, Any],
    *,
    anchor: str = "",
    audience: str = "front",
) -> str:
    evidence = pipeline.get("evidence") or {}
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    pack = pipeline.get("canonical_peaks") or {}
    evidence_rows = list(pack.get("evidence_rows") or [])
    eid = f"{_esc(anchor)}-band-evidence-map" if anchor else "band-evidence-map"
    rows_html: list[str] = []
    is_front = str(audience or "front").lower() == "front"
    for er in evidence_rows[:32]:
        labels = [x for x in (er.get("possible_functional_groups") or []) if x]
        if is_front and not labels:
            continue
        peak_cell = "—"
        if er.get("peak_cm1") is not None:
            peak_cell = f"{int(er['peak_cm1'])}"
        region = _esc(str(er.get("band_region") or "—"))
        fg_s = _esc(", ".join(labels[:5]) if labels else "—")
        status = _support_status_for_labels(labels, assigns, evidence)
        miss_s = "—"
        comp = "—"
        bid = str(er.get("band_id", ""))
        if bid in ("siloxane_sio", "ether_co", "ester_co_o", "aryl_ether_co", "phenolic_co", "silicone_sic"):
            if _atr_sensitive_band(bid, evidence):
                miss_s = _esc("ATR-sensitive overlap; Si–O / C–O competitors")
            else:
                miss_s = _esc("overlaps Si–O / C–O fingerprint")
        pq = "—"
        if er.get("peak_id") and pack.get("peak_by_id"):
            crow = (pack.get("peak_by_id") or {}).get(er["peak_id"]) or {}
            pq = _esc(str(crow.get("quality_class") or "—"))
        debug_cols = ""
        if not is_front:
            debug_cols = (
                f"<td class='mono'>{_esc(str(er.get('peak_id') or '—'))}</td>"
                f"<td class='mono'>{_esc(str(er.get('band_id') or '—'))}</td>"
                f"<td>{_esc(str(er.get('source') or '—'))}</td>"
            )
        rows_html.append(
            f"<tr><td>{peak_cell}</td><td>{region}</td><td>{fg_s}</td>"
            f"<td><span class='support-pill'>{_esc(status)}</span></td>"
            f"<td>{pq}</td><td>{miss_s}</td><td>{comp}</td>{debug_cols}</tr>"
        )
    if not rows_html:
        rows_html.append("<tr><td colspan='7' class='muted'>No matched bands above threshold.</td></tr>")
    return (
        f"<div class='band-evidence-map' id='{eid}'>"
        + "<h3>Band evidence map</h3>"
        + f"<p class='hint small'>{_esc(SCORE_TOOLTIP)}</p>"
        + "<div class='table-scroll'><table class='tbl tbl-zebra tbl-band-map small'><thead><tr>"
        + _th("Peak (cm⁻¹)")
        + _th("Band / region")
        + _th("Possible functional groups")
        + _th("Support")
        + _th("Peak quality")
        + _th("Missing / overlap")
        + _th("Competing explanation")
        + ("" if is_front else _th("peak_id") + _th("band_id") + _th("source"))
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div></div>"
    )


def _req_met_icon(ent: dict[str, Any]) -> str:
    ec = str(ent.get("evidence_completeness") or "")
    if ec == "complete":
        return "<span class='ev-icon ev-ok' title='required bands met'>✓</span>"
    if ec in ("partial", "single_band"):
        return "<span class='ev-icon ev-part' title='partial evidence'>△</span>"
    return "<span class='ev-icon ev-miss' title='missing required evidence'>✕</span>"


def _ml_agreement_short(pipeline: dict[str, Any], lab: str) -> str:
    cent = ((pipeline.get("consensus") or {}).get("per_label") or {}).get(lab) or {}
    st = str(cent.get("agreement_status") or "")
    if st == "rule_and_ml_agree":
        return "✓ agree"
    if st == "ml_only_warning":
        return "⚠ ML warning"
    if st == "conflict":
        return "⚠ conflict"
    if st == "rule_only":
        return "rules only"
    return "—"


def why_here_line(lab: str, ent: dict[str, Any], pipeline: dict[str, Any]) -> str:
    peaks = ent.get("supporting_peaks") or []
    peak_bits: list[str] = []
    for p in peaks[:3]:
        if isinstance(p, dict):
            peak_bits.append(f"{float(p.get('wn_cm1', p.get('wn', 0))):.0f} cm⁻¹")
        else:
            peak_bits.append(str(p)[:20])
    sup_b = ent.get("supporting_bands") or []
    band_bits = [str(b) for b in sup_b[:2]]
    parts: list[str] = []
    if peak_bits:
        parts.append("peaks " + ", ".join(peak_bits))
    if band_bits:
        parts.append("bands " + ", ".join(band_bits))
    miss = ent.get("missing_expected_bands") or []
    if miss:
        parts.append("missing " + ", ".join(str(m) for m in miss[:2]))
    comp = ent.get("competing_explanation")
    if comp:
        parts.append(f"vs {_truncate(str(comp), 60)}")
    if not parts:
        parts.append(str(ent.get("human_readable_summary") or "")[:80] or "—")
    return _truncate("; ".join(parts), 160)


def build_fg_justification_panels_html(
    pipeline: dict[str, Any],
    *,
    anchor: str = "",
    top_n: int = 8,
) -> str:
    ranked = evidence_ranked_assignments(pipeline, min_score=0.15, top_n=top_n)
    evidence = pipeline.get("evidence") or {}
    art = (evidence.get("artifacts") or {}).get("flags") or {}
    art_on = [k for k, v in art.items() if v][:4]
    eid = f"{_esc(anchor)}-fg-justify-panels" if anchor else "fg-justify-panels"
    cards: list[str] = []
    for lab, ent in ranked:
        if str(ent.get("ontology_category") or "") in ("artifact",):
            continue
        peaks = ent.get("supporting_peaks") or []
        pk_line = ", ".join(
            f"{float(p.get('wn_cm1', 0)):.0f}" for p in peaks[:4] if isinstance(p, dict)
        ) or "—"
        miss = ent.get("missing_expected_bands") or []
        miss_line = ", ".join(str(m) for m in miss[:4]) if miss else "none"
        comp = ent.get("competing_explanation") or "—"
        caut = "; ".join(str(c) for c in (ent.get("caution_flags") or [])[:3]) or "—"
        if art_on:
            caut = (caut + "; " if caut != "—" else "") + "⚠ " + ", ".join(art_on[:2])
        cards.append(
            f"<div class='fg-justify-card'>"
            f"<h4>{_esc(lab)} {_req_met_icon(ent)} "
            f"<span class='muted small'>{_esc(str(ent.get('confidence_class') or ''))} · "
            f"{_esc(str(ent.get('evidence_completeness') or ''))}</span></h4>"
            f"<ul class='fg-justify-list small'>"
            f"<li><b>Required bands:</b> {_req_met_icon(ent)} {miss_line if miss else 'met'}</li>"
            f"<li><b>Supporting peaks:</b> {_esc(pk_line)}</li>"
            f"<li><b>Competing:</b> {_esc(_truncate(str(comp), 120))}</li>"
            f"<li><b>Artifacts / overlap:</b> {_esc(_truncate(caut, 140))}</li>"
            f"<li><b>ML:</b> {_esc(_ml_agreement_short(pipeline, lab))}</li>"
            f"<li><b>Why here:</b> {_esc(why_here_line(lab, ent, pipeline))}</li>"
            f"</ul></div>"
        )
    if not cards:
        cards.append("<p class='muted'>No assignments above threshold.</p>")
    return (
        f"<div class='fg-justify-panels' id='{eid}'>"
        "<h3>Assignment justification</h3>"
        f"<p class='hint small'>{_esc(SCORE_TOOLTIP)}</p>"
        + "".join(cards)
        + "</div>"
    )


def _format_ml_cell(ml_ent: dict[str, Any], head_labels: set[str], lab_key: str) -> str:
    if not head_labels:
        return "—"
    if lab_key not in head_labels and not ml_ent:
        return "no ML head"
    if ml_ent.get("ml_probability") is not None:
        lbl = str(ml_ent.get("ml_score_label") or "").lower()
        if "calibrated" in lbl and "probability" in lbl:
            return f"{float(ml_ent['ml_probability']):.3f} (calibrated ML probability)"
        return f"{float(ml_ent['ml_probability']):.3f}"
    if ml_ent.get("ml_score") is not None:
        return f"{float(ml_ent['ml_score']):.4f} (SVM decision score)"
    return "—"


def build_evidence_first_assignments_table_html(
    pipeline: dict[str, Any],
    *,
    anchor: str = "",
    top_n: int = 16,
) -> str:
    """Explainable table: evidence-first columns, de-emphasized scores."""
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    ml_blocks = pipeline.get("ml_refinement") or {}
    ml_family = ml_blocks.get("basic") or {}
    ml_specific = ml_blocks.get("subtle") or {}
    ml_family_per = ml_family.get("per_label") or {}
    ml_specific_per = ml_specific.get("per_label") or {}
    fam_heads = set((ml_family.get("ml_probabilities") or {}).keys())
    spec_heads = set((ml_specific.get("ml_probabilities") or {}).keys())
    ranked = evidence_ranked_assignments(pipeline, min_score=0.1, top_n=top_n)
    eid = f"{_esc(anchor)}-explain-assign" if anchor else "explain-assign"
    tip = SCORE_TOOLTIP
    parts = [
        f"<div class='explain-assign-wrap' id='{eid}'>",
        "<h3>Functional group assignments (evidence-first)</h3>",
        f"<p class='hint small'>{_esc(tip)}</p>",
        "<div class='table-scroll explain-assign-scroll'>",
        "<table class='tbl tbl-zebra tbl-explain-assign small'><thead><tr>",
        _th("Functional group"),
        _th("Confidence"),
        _th("Evidence completeness"),
        _th("Evidence score", tip=tip),
        _th("Report ranking score", tip="Ordering only; not a probability"),
        _th("Why this is here"),
        _th("Supporting bands"),
        _th("Calibrated ML probability", tip="Shown only when model provides calibrated probabilities"),
        _th("ML agreement"),
        "</tr></thead><tbody>",
    ]
    cons_per = (pipeline.get("consensus") or {}).get("per_label") or {}
    for lab, ent in ranked[:top_n]:
        cent = cons_per.get(lab) or {}
        ev_sc = float(ent.get("score", 0) or 0)
        rank_sc = float(cent.get("final_score", ev_sc) or ev_sc)
        cc = str(ent.get("confidence_class") or "—")
        ec = str(ent.get("evidence_completeness") or "—")
        sup_b = "; ".join(_esc(str(x)) for x in (ent.get("supporting_bands") or [])[:4]) or "—"
        why = _esc(why_here_line(lab, ent, pipeline))
        ml_spec = _format_ml_cell(ml_specific_per.get(lab) or {}, spec_heads, lab)
        agr = _esc(str(cent.get("agreement_status") or "—"))
        parts.append(
            f"<tr><td>{_esc(lab)}</td><td>{_esc(cc)}</td><td>{_esc(ec)}</td>"
            f"<td>{ev_sc:.3f}</td><td class='muted'>{rank_sc:.3f}</td>"
            f"<td>{why}</td><td>{sup_b}</td><td>{_esc(ml_spec)}</td><td>{agr}</td></tr>"
        )
    if len(ranked) == 0:
        parts.append("<tr><td colspan='9' class='muted'>No assignments above threshold.</td></tr>")
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def build_evidence_first_highlights_table_html(
    pipeline: dict[str, Any],
    *,
    top_n: int = 12,
) -> str:
    """Consensus highlights ordered by evidence rank, not raw score."""
    ranked = evidence_ranked_assignments(pipeline, min_score=0.12, top_n=top_n)
    cons_per = (pipeline.get("consensus") or {}).get("per_label") or {}
    parts = [
        "<div class='fg-highlights'>",
        "<h3>Leading assignments</h3>",
        f"<p class='hint small'>{_esc(SCORE_TOOLTIP)} Ordered by confidence and evidence completeness.</p>",
        "<div class='table-scroll'><table class='tbl tbl-zebra small'><thead><tr>",
        _th("Functional group"),
        _th("Confidence"),
        _th("Evidence"),
        _th("Evidence score", tip=SCORE_TOOLTIP),
        _th("Why this is here"),
        _th("ML agreement"),
        "</tr></thead><tbody>",
    ]
    for lab, ent in ranked:
        cent = cons_per.get(lab) or {}
        parts.append(
            f"<tr><td>{_esc(lab)}</td>"
            f"<td>{_esc(str(ent.get('confidence_class') or '—'))}</td>"
            f"<td>{_esc(str(ent.get('evidence_completeness') or '—'))}</td>"
            f"<td>{float(ent.get('score', 0)):.3f}</td>"
            f"<td>{_esc(why_here_line(lab, ent, pipeline))}</td>"
            f"<td>{_esc(str(cent.get('agreement_status') or '—'))}</td></tr>"
        )
    if not ranked:
        parts.append("<tr><td colspan='6' class='muted'>No assignments above threshold.</td></tr>")
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def descriptive_ambiguity_html(pipeline: dict[str, Any]) -> str:
    """Richer ambiguity text than generic 'ambiguous' labels."""
    amb = (pipeline.get("rule_assignments") or {}).get("ambiguity_labels") or []
    if not amb:
        return ""
    items: list[str] = []
    for a in amb:
        title = str(a.get("title") or a.get("id") or "")
        reason = str(a.get("reason") or "")
        related = a.get("related_labels") or []
        rel_s = ", ".join(str(x) for x in related[:5])
        if reason:
            line = f"<b>{_esc(title)}</b>: {_esc(reason)}"
        else:
            line = f"<b>{_esc(title)}</b>"
        if rel_s:
            line += f" <span class='muted'>({_esc(rel_s)})</span>"
        items.append(f"<li>{line}</li>")
    return (
        "<div class='ambiguity-rich'><h4>Unresolved families</h4>"
        "<ul class='ambiguity-rich-list'>" + "".join(items) + "</ul></div>"
    )
