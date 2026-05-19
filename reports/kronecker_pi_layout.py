"""
Kronecker batch HTML layout: summary table, spectrum cards, density modes, badges.
"""

from __future__ import annotations

import html
from typing import Any, Literal

ReportDensity = Literal["summary", "balanced", "audit"]


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _probs_table_html(probs: dict[str, float]) -> str:
    rows = sorted(probs.items(), key=lambda kv: -kv[1])
    parts = [
        "<table class='tbl'><tr><th>Functional group</th><th>P(label)</th></tr>",
    ]
    for lab, p in rows:
        parts.append(f"<tr><td>{_esc(lab)}</td><td>{float(p):.4f}</td></tr>")
    parts.append("</table>")
    return "".join(parts)


def _truncate(s: str, max_len: int) -> str:
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


# Fallback / broad family labels (v3 + v4) for summary-table columns.
_FALLBACK_AND_FAMILY_LABELS: frozenset[str] = frozenset(
    {
        "hydroxy_containing",
        "carbonyl_containing",
        "nitrogen_containing",
        "C_O_containing",
        "aromatic_system",
        "unsaturation_possible",
        "fingerprint_C_O_or_Si_O_overlap",
        "triple_bond_region_possible",
        "hydroxy_family",
        "carbonyl_family",
        "nitrogen_family",
        "aromatic_family",
        "ether_C_O_family",
        "unsaturation_family",
        "silicon_oxygen_family",
        "nitro_family",
        "sulfur_family",
        "halogenated_family",
    }
)


def families_and_specifics_from_assignments(
    pipeline: dict[str, Any],
    *,
    n_each: int = 3,
    min_score: float = 0.12,
) -> tuple[str, str]:
    """Split rule assignments into broad family vs specific FG lines for the summary table."""
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    fams: list[tuple[str, float]] = []
    specs: list[tuple[str, float]] = []
    for lab, ent in assigns.items():
        if not isinstance(ent, dict):
            continue
        sc = float(ent.get("score", 0) or 0)
        if sc < min_score:
            continue
        lb = str(lab)
        oc = str(ent.get("ontology_category") or "")
        if oc in ("local_motif", "artifact"):
            continue
        if oc in ("fallback", "family") or lb in _FALLBACK_AND_FAMILY_LABELS:
            fams.append((lb, sc))
        else:
            specs.append((lb, sc))
    from reports.v4_evidence_report import assignment_evidence_rank, format_summary_assignment_label

    fams.sort(key=lambda t: assignment_evidence_rank(assigns.get(t[0]) or {}))
    specs.sort(key=lambda t: assignment_evidence_rank(assigns.get(t[0]) or {}))
    fam_s = ", ".join(x[0] for x in fams[:n_each]) or "—"
    spec_s = (
        ", ".join(
            format_summary_assignment_label(x[0], assigns.get(x[0]) or {}, pipeline) for x in specs[:n_each]
        )
        or "—"
    )
    return fam_s, spec_s


def split_major_minor_cautions(cautions: list[str]) -> tuple[list[str], list[str]]:
    """Split caution strings: higher-signal first, overlap / routine notes collapsed."""
    major_kw = (
        "ml-only",
        "conflict",
        "artifact",
        "atr",
        "siloxane",
        "si-o",
        "saturation",
        "moisture",
        "water",
        "co2",
        "spike",
        "unknown model",
        "false positive",
        "interference",
        "detector",
    )
    major: list[str] = []
    minor: list[str] = []
    for c in cautions:
        low = str(c).lower()
        if any(k in low for k in major_kw):
            major.append(str(c))
        else:
            minor.append(str(c))
    if not major and cautions:
        return cautions[:2], cautions[2:]
    return major, minor


def collect_pipeline_cautions(pipeline: dict[str, Any], *, max_total: int = 40) -> list[str]:
    out: list[str] = []
    for w in pipeline.get("warnings") or []:
        if w and w not in out:
            out.append(str(w))
    for lab, ent in (pipeline.get("rule_assignments", {}).get("assignments") or {}).items():
        for c in ent.get("caution_flags") or []:
            t = f"{lab}: {c}"
            if t not in out:
                out.append(t)
    ml = pipeline.get("ml_refinement") or {}
    for key in ("basic", "subtle", "legacy"):
        block = ml.get(key)
        if not block:
            continue
        for lab, ent in (block.get("per_label") or {}).items():
            if ent.get("agreement_status") == "ml_only_warning":
                t = f"{lab}: ML-only warning (weak band evidence)"
                if t not in out:
                    out.append(t)
    return out[:max_total]


def _ml_refinement_flat_html(
    pipeline: dict[str, Any],
    *,
    ml_on: bool,
    legacy_probs: dict[str, float] | None,
) -> str:
    from ml.ftir_report_sections import ml_refinement_html

    if not ml_on:
        return ""
    ml = pipeline.get("ml_refinement") or {}
    chunks: list[str] = []
    if ml.get("basic"):
        bk = str((ml["basic"] or {}).get("model_kind") or "basic")
        fam_title = "ML refinement (family)" if bk in ("family", "basic") else f"ML refinement ({bk})"
        chunks.append(ml_refinement_html(ml["basic"], title=fam_title))
    if ml.get("subtle"):
        sk = str((ml["subtle"] or {}).get("model_kind") or "subtle")
        spec_title = "ML refinement (specific)" if sk in ("specific", "subtle") else f"ML refinement ({sk})"
        chunks.append(ml_refinement_html(ml["subtle"], title=spec_title))
    if ml.get("legacy"):
        legacy_tbl = _probs_table_html(legacy_probs) if legacy_probs else ""
        chunks.append(ml_refinement_html(ml["legacy"], title="ML refinement (legacy)") + legacy_tbl)
    return "".join(chunks)


def top_supported_assignments(
    pipeline: dict[str, Any],
    *,
    n: int = 3,
    min_score: float = 0.12,
) -> list[tuple[str, float, str | None]]:
    """Return (label, score, confidence_class) sorted by evidence rank (not raw score)."""
    from reports.v4_evidence_report import evidence_ranked_assignments

    ranked = evidence_ranked_assignments(pipeline, min_score=min_score, top_n=max(n, 12))
    return [
        (lab, float(ent.get("score", 0) or 0), str(ent.get("confidence_class") or None))
        for lab, ent in ranked[:n]
    ]


def ml_agreement_summary(pipeline: dict[str, Any], *, ml_enabled: bool) -> str:
    if not ml_enabled:
        return "N/A"
    consensus = pipeline.get("consensus") or {}
    top = consensus.get("top_labels") or []
    statuses: list[str] = []
    for _lab, ent in top[:10]:
        st = ent.get("agreement_status")
        if st:
            statuses.append(str(st))
    if not statuses:
        return "—"
    if any(s == "ml_only_warning" for s in statuses):
        return "ML-only warnings"
    if any(s == "conflict" for s in statuses):
        return "Some conflict"
    if any(s == "rule_and_ml_agree" for s in statuses):
        n_agree = sum(1 for s in statuses if s == "rule_and_ml_agree")
        if n_agree >= len(statuses) * 0.6:
            return "Mostly agree"
    return "Mixed"


def spectrum_needs_review(pipeline: dict[str, Any], *, ml_enabled: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    high_risk = {
        "phenol",
        "nitrile",
        "nitro",
        "siloxane",
        "amide",
        "ester",
        "carboxylic_acid",
    }
    for lab in high_risk:
        ent = assigns.get(lab) or {}
        sc = float(ent.get("score", 0) or 0)
        cc = str(ent.get("confidence_class") or "")
        if sc >= 0.52 and cc in ("tentative", "local_possible"):
            reasons.append(f"{lab}: high score but {cc}")

    art = (pipeline.get("evidence") or {}).get("artifacts") or {}
    flags = art.get("flags") or {}
    major_keys = (
        "water_vapor_or_moisture_like",
        "co2_region_elevated",
        "fingerprint_crowding",
        "possible_saturation",
        "weak_nitrile_region_spike",
    )
    if any(flags.get(k) for k in major_keys):
        reasons.append("artifact / interference flags")

    amb = (pipeline.get("rule_assignments") or {}).get("ambiguity_labels") or []
    if amb:
        reasons.append("ambiguity / fallback labels")

    consensus = pipeline.get("consensus") or {}
    for _lab, ent in (consensus.get("top_labels") or [])[:12]:
        st = ent.get("agreement_status")
        if ml_enabled and st in ("conflict", "ml_only_warning"):
            reasons.append(f"consensus: {st}")
            break

    strongish = 0
    for _lab, ent in assigns.items():
        cc = str(ent.get("confidence_class") or "")
        sc = float(ent.get("score", 0) or 0)
        if sc >= 0.45 and cc in ("strong", "supported"):
            strongish += 1
    if strongish == 0 and len([x for x in assigns.values() if float(x.get("score", 0)) >= 0.25]) > 0:
        reasons.append("no strong/supported anchor")

    return bool(reasons), reasons[:6]


def compute_card_status(pipeline: dict[str, Any], *, ml_enabled: bool) -> str:
    needs, _reasons = spectrum_needs_review(pipeline, ml_enabled=ml_enabled)
    if needs:
        return "attention"
    amb = (pipeline.get("rule_assignments") or {}).get("ambiguity_labels") or []
    if amb:
        return "ambiguous"
    cautions = collect_pipeline_cautions(pipeline, max_total=8)
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    has_strong = any(
        float(e.get("score", 0) or 0) >= 0.55 and str(e.get("confidence_class") or "") == "strong"
        for e in assigns.values()
    )
    if has_strong and len(cautions) <= 2:
        return "strong_support"
    if cautions:
        return "supported_with_cautions"
    return "strong_support" if has_strong else "supported_with_cautions"


def status_badge_html(status: str, *, front_mode: bool = False) -> str:
    if front_mode:
        if status not in ("attention", "ambiguous", "needs_review"):
            return ""
        return (
            "<span class='badge-sci badge-overlap' title='spectral overlap or competing bands'>"
            "Review overlap</span>"
        )
    labels = {
        "strong_support": ("Strong", "badge-strong"),
        "supported_with_cautions": ("Supported", "badge-supported"),
        "ambiguous": ("Ambiguous", "badge-ambiguous"),
        "attention": ("Check bands", "badge-attn"),
        "needs_review": ("Check bands", "badge-attn"),
    }
    text, cls = labels.get(status, ("—", "badge-muted"))
    return f"<span class='badge-sci {cls}' title='{_esc(status)}'>{_esc(text)}</span>"


def build_summary_table_html(*, rows: list[dict[str, Any]], ml_enabled: bool) -> str:
    """High-signal batch table: families, specifics, ambiguity, cautions, ML, jump links."""
    parts = [
        "<section id='summary-table' class='summary-table-section card'>",
        "<h2 class='summary-table-heading'>Summary Table</h2>",
        "<div class='table-scroll summary-table-wrap'>",
        "<table class='tbl tbl-zebra tbl-sticky tbl-summary'><thead><tr>",
        "<th>Spectrum</th>",
        "<th>Supported families</th>",
        "<th>Specific assignments</th>",
        "<th>Major ambiguity</th>",
        "<th>Major cautions</th>",
    ]
    if ml_enabled:
        parts.append("<th>ML status</th>")
    parts.append("<th>Link</th></tr></thead><tbody>")
    for r in rows:
        fam = _esc(_truncate(str(r.get("families_text") or "—"), 140))
        spec = _esc(_truncate(str(r.get("specifics_text") or "—"), 140))
        amb_raw = ", ".join(str(x) for x in (r.get("amb_titles") or [])[:4])
        amb = _esc(_truncate(amb_raw or "—", 120))
        maj = r.get("caut_major") or r.get("caut_short") or []
        mino = r.get("caut_minor") or []
        maj_join = "; ".join(str(x) for x in maj[:4])
        caut = _esc(_truncate(maj_join, 160))
        caut_cell: str
        if mino:
            rest = _esc(_truncate("; ".join(str(x) for x in mino[:10]), 220))
            caut_cell = (
                f"<span>{caut}</span> <details class='inline-details'><summary>+</summary>"
                f"<p class='muted small caution-minor-list'>{rest}</p></details>"
            )
        else:
            caut_cell = caut if caut.strip() else "—"
        parts.append("<tr>")
        parts.append(
            f"<td class='td-spectrum'>{_esc(r['name'])} {status_badge_html(str(r.get('status', '')))}</td>"
            f"<td>{fam}</td><td>{spec}</td><td>{amb}</td><td>{caut_cell}</td>"
        )
        if ml_enabled:
            parts.append(f"<td>{_esc(str(r.get('ml_agree', '—')))}</td>")
        parts.append(f"<td><a href='#{_esc(r['anchor'])}'>open</a></td></tr>")
    parts.append("</tbody></table></div></section>")
    return "".join(parts)


def spectrum_card_title_row_html(
    *,
    name: str,
    anchor: str,
    status: str,
    top_assignments: str = "",
    top_cautions: str = "",
    ml_status: str = "",
) -> str:
    """Compact header after the spectrum figure: name, badges, one-line summary, short links."""
    sub: list[str] = []
    if top_assignments.strip():
        sub.append(f"<p class='spec-card-compact'><span class='lbl'>Assignments</span> {_esc(_truncate(top_assignments, 200))}</p>")
    if top_cautions.strip():
        sub.append(f"<p class='spec-card-compact muted'><span class='lbl'>Cautions</span> {_esc(_truncate(top_cautions, 180))}</p>")
    if ml_status.strip() and ml_status != "N/A":
        sub.append(f"<p class='spec-card-compact muted'><span class='lbl'>ML</span> {_esc(_truncate(ml_status, 120))}</p>")
    links = (
        f"<p class='spec-inline-links'><a href='#{_esc(anchor)}-figure'>Spectrum</a> · "
        f"<a href='#{_esc(anchor)}-explain-assign'>Assignments table</a> · "
        f"<a href='#{_esc(anchor)}-evidence'>Evidence</a> · "
        f"<a href='#{_esc(anchor)}-audit'>Details</a></p>"
    )
    body = "".join(sub) + links
    return (
        f"<div class='spec-title-after-plot' id='{_esc(anchor)}-header'>"
        f"<div class='spec-title-row'><h2 class='spec-card-title'>{_esc(name)}</h2>{status_badge_html(status)}</div>"
        f"{body}</div>"
    )


def spectrum_summary_quick_details_html(
    *,
    anchor: str,
    row: dict[str, Any],
    pipeline: dict[str, Any],
    density: str,
    top_n_summary: int,
    include_evidence: bool,
) -> str:
    """Expandable quick scan: optional interpretation; major cautions visible, minor collapsed."""
    top3 = row.get("top3") or []
    top_line = ", ".join(f"{lab}" + (f" ({cc})" if cc else "") for lab, _s, cc in top3) or "—"
    maj = row.get("caut_major") or row.get("caut_short") or []
    mino = row.get("caut_minor") or []
    caut_major_line = "; ".join(_truncate(c, 100) for c in maj[:2]) or "—"
    caut_minor_block = ""
    if mino:
        items = "".join(f"<li>{_esc(_truncate(c, 140))}</li>" for c in mino[:12])
        caut_minor_block = (
            f"<details class='caution-minor'><summary>Additional notes ({len(mino)})</summary><ul class='caution-soft'>{items}</ul></details>"
        )
    links = (
        f"<p class='spec-inline-links'><a href='#{_esc(anchor)}-figure'>Spectrum</a> · "
        f"<a href='#{_esc(anchor)}-explain-assign'>Assignments table</a> · "
        f"<a href='#{_esc(anchor)}-evidence'>Evidence</a> · "
        f"<a href='#{_esc(anchor)}-audit'>Details</a></p>"
    )
    inner_parts = [
        f"<p class='spec-card-compact'><span class='lbl'>Assignments</span> {_esc(_truncate(top_line, 220))}</p>",
        f"<p class='spec-card-compact muted'><span class='lbl'>Cautions</span> {_esc(caut_major_line)}</p>",
        caut_minor_block,
        f"<p class='spec-card-compact muted'><span class='lbl'>ML</span> {_esc(str(row.get('ml_agree', '—')))}</p>",
        links,
    ]
    if include_evidence:
        inner_parts.append(_quick_interp_n(pipeline, str(density), int(top_n_summary)))
    open_default = str(density).lower() == "audit"
    return details_block(
        f"{anchor}-summary-quick",
        "Summary",
        "".join(inner_parts),
        open_default=open_default,
    )


def build_most_likely_fg_table_html(
    pipeline: dict[str, Any],
    *,
    top_n: int = 12,
    include_consensus: bool,
) -> str:
    """Evidence-first leading assignments (after the figure)."""
    if str(pipeline.get("ontology") or "").lower() == "v4" or include_consensus:
        from reports.v4_evidence_report import build_evidence_first_highlights_table_html

        return build_evidence_first_highlights_table_html(pipeline, top_n=top_n)
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    order: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    if include_consensus:
        for lab, cent in (pipeline.get("consensus") or {}).get("top_labels") or []:
            if float(cent.get("final_score", 0)) < 0.12:
                continue
            lab_s = str(lab)
            rule = assigns.get(lab_s) if isinstance(assigns.get(lab_s), dict) else {}
            order.append((lab_s, cent, rule or {}))
            if len(order) >= int(top_n):
                break
    if not order and assigns:
        for lab, ent in sorted(assigns.items(), key=lambda kv: -float(kv[1].get("score", 0))):
            if float(ent.get("score", 0)) < 0.12:
                continue
            sc = float(ent.get("score", 0))
            pseudo: dict[str, Any] = {
                "final_score": sc,
                "rule_score": sc,
                "agreement_status": "—",
            }
            order.append((str(lab), pseudo, ent if isinstance(ent, dict) else {}))
            if len(order) >= int(top_n):
                break
    if not order:
        return (
            "<div class='fg-highlights'><h3>Consensus</h3>"
            "<p class='muted small'>Rule scores and optional ML fusion; expand sections below for evidence tables.</p></div>"
        )
    parts = [
        "<div class='fg-highlights'>",
        "<h3>Consensus</h3>",
        "<p class='hint small'>Ordered by fused score when available; otherwise by rule score.</p>",
        "<div class='table-scroll'>",
        "<table class='tbl tbl-zebra small'>",
        "<thead><tr><th>Functional group</th><th>Rule score</th><th>Class</th>"
        "<th>Final</th><th>Agreement</th><th>Rule summary</th></tr></thead><tbody>",
    ]
    for lab, cent, rule in order:
        rs = float(cent.get("rule_score", rule.get("score", 0)))
        cc = str(rule.get("confidence_class") or "—")
        summ = _truncate(str(rule.get("human_readable_summary") or ""), 140)
        parts.append(
            f"<tr><td>{_esc(lab)}</td><td>{rs:.3f}</td><td>{_esc(cc)}</td>"
            f"<td>{float(cent.get('final_score', 0)):.3f}</td>"
            f"<td>{_esc(cent.get('agreement_status', '—'))}</td>"
            f"<td>{_esc(summ)}</td></tr>"
        )
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def build_explainable_assignments_table_html(
    pipeline: dict[str, Any],
    *,
    anchor: str = "",
    top_n: int = 16,
    min_final: float = 0.1,
) -> str:
    """
    Per-spectrum explainable table: evidence-first columns (v4) or legacy layout.
    """
    if str(pipeline.get("ontology") or "").lower() == "v4":
        from reports.v4_evidence_report import build_evidence_first_assignments_table_html

        return build_evidence_first_assignments_table_html(pipeline, anchor=anchor, top_n=top_n)
    # Legacy explainable table (v3 / non-v4)
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    ml_blocks = pipeline.get("ml_refinement") or {}
    ml_family = ml_blocks.get("basic") or {}
    ml_specific = ml_blocks.get("subtle") or {}
    ml_family_per = ml_family.get("per_label") or {}
    ml_specific_per = ml_specific.get("per_label") or {}
    model_family_labels = set((ml_family.get("ml_probabilities") or {}).keys())
    model_specific_labels = set((ml_specific.get("ml_probabilities") or {}).keys())
    cons = pipeline.get("consensus") or {}
    order: list[tuple[str, dict[str, Any]]] = list(cons.get("top_labels") or [])
    if not order and assigns:
        for lab, ent in sorted(assigns.items(), key=lambda kv: -float(kv[1].get("score", 0))):
            if float(ent.get("score", 0)) < 0.12:
                continue
            order.append(
                (
                    str(lab),
                    {
                        "final_score": float(ent.get("score", 0)),
                        "rule_score": float(ent.get("score", 0)),
                        "agreement_status": "—",
                    },
                )
            )
            if len(order) >= int(top_n):
                break

    eid = f"{_esc(anchor)}-explain-assign" if anchor else "explainable-assignments"
    parts = [
        f"<div class='explain-assign-wrap' id='{eid}'>",
        "<h3 class='explain-assign-heading'>Explainable assignment table</h3>",
        "<p class='hint small'>Every row ties to spectral rule evidence where present; "
        "<span class='muted'>ML-only</span> rows are explicitly flagged when band support is weak.</p>",
        "<div class='table-scroll explain-assign-scroll'>",
        "<table class='tbl tbl-zebra tbl-explain-assign small'>",
        "<thead><tr>",
        "<th>Functional group</th><th>Category</th><th>Score (final)</th><th>Confidence class</th>",
        "<th>Supporting bands</th><th>Supporting peaks</th>",
        "<th>Missing expected evidence</th><th>Competing explanation</th>",
        "<th>Artifact / caution</th><th>ML family</th><th>ML specific</th><th>Consensus</th><th>Justification</th>",
        "</tr></thead><tbody>",
    ]
    n = 0
    for lab, cent in order:
        fin = float(cent.get("final_score", 0))
        if fin < min_final and n > 0:
            continue
        rule = assigns.get(str(lab)) if isinstance(assigns.get(str(lab)), dict) else {}
        ml_fam_ent = ml_family_per.get(str(lab)) or {}
        ml_spec_ent = ml_specific_per.get(str(lab)) or {}
        cat = str(rule.get("ontology_category") or rule.get("assignment_type") or "—")
        cc = str(rule.get("confidence_class") or "—")
        sup_b = rule.get("supporting_bands") or []
        sup_b_s = "; ".join(_esc(str(x)) for x in sup_b[:4]) if sup_b else "—"
        peaks = rule.get("supporting_peaks") or []
        if peaks:
            bits = []
            for p in peaks[:5]:
                if isinstance(p, dict):
                    bits.append(f"{float(p.get('wn_cm1', p.get('wn', 0))):.0f}")
                else:
                    bits.append(str(p)[:24])
            pk_s = ", ".join(bits)
        else:
            pk_s = "—"
        miss = rule.get("missing_expected_bands") or []
        miss_s = "; ".join(_esc(str(x)) for x in miss[:4]) if miss else "—"
        comp = rule.get("competing_explanation")
        comp_s = _esc(comp) if comp else "—"
        caut = list(rule.get("caution_flags") or []) + list(ml_fam_ent.get("caution_flags") or [])
        caut += list(ml_spec_ent.get("caution_flags") or [])
        caut_s = "; ".join(_esc(str(c)[:160]) for c in caut[:4]) if caut else "—"
        agr = str(cent.get("agreement_status") or "—")

        def _fmt_ml_cell(ml_ent: dict[str, Any], head_labels: set[str], lab_key: str) -> str:
            if not head_labels:
                return "—"
            if lab_key not in head_labels and not ml_ent:
                mapped = False
                for hk in head_labels:
                    from ml.ftir_ml_refinement import _map_ml_label_to_rule

                    if lab_key in _map_ml_label_to_rule(hk):
                        mapped = True
                        break
                if not mapped:
                    return "no corresponding ML head"
            if ml_ent.get("ml_probability") is not None:
                ml_s = f"{float(ml_ent['ml_probability']):.3f}"
                lbl = str(ml_ent.get("ml_score_label") or "").lower()
                if "calibrated svm probability" in lbl or ("calibrated" in lbl and "probability" in lbl):
                    return f"{ml_s} (calibrated SVM probability)"
                return f"{ml_s} ({_esc(ml_ent.get('ml_score_label', 'ML'))})"
            if ml_ent.get("ml_score") is not None:
                return f"{float(ml_ent['ml_score']):.4f} ({_esc(ml_ent.get('ml_score_label', 'SVM score'))})"
            return "—"

        ml_fam_cell = _fmt_ml_cell(ml_fam_ent, model_family_labels, str(lab))
        ml_spec_cell = _fmt_ml_cell(ml_spec_ent, model_specific_labels, str(lab))
        just = _truncate(
            str(
                rule.get("human_readable_summary")
                or ml_spec_ent.get("human_readable_summary")
                or ml_fam_ent.get("human_readable_summary")
                or ""
            ),
            220,
        )
        if agr == "ml_only_warning":
            just = (
                _truncate("ML-only advisory: weak spectral rule score relative to model output. " + just, 280)
                if just
                else "ML-only advisory: weak spectral rule score relative to model output."
            )
        parts.append(
            f"<tr><td>{_esc(lab)}</td><td>{_esc(cat)}</td><td>{fin:.3f}</td><td>{_esc(cc)}</td>"
            f"<td>{sup_b_s}</td><td>{_esc(pk_s)}</td><td>{miss_s}</td><td>{comp_s}</td>"
            f"<td>{caut_s}</td><td>{_esc(ml_fam_cell)}</td><td>{_esc(ml_spec_cell)}</td>"
            f"<td>{_esc(agr)}</td><td>{_esc(just)}</td></tr>"
        )
        n += 1
        if n >= int(top_n):
            break
    if n == 0:
        parts.append(
            "<tr><td colspan='13' class='muted'>No assignments above threshold for this spectrum.</td></tr>"
        )
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def build_fg_justification_intro_html(pipeline: dict[str, Any], *, top_n: int = 4) -> str:
    """Short justification prose for leading consensus labels (below the highlights table)."""
    consensus = pipeline.get("consensus") or {}
    top = consensus.get("top_labels") or []
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    ml_blocks = pipeline.get("ml_refinement") or {}
    parts = ["<div class='fg-justify-below'>", "<h3>Evidence notes</h3>"]
    n = 0
    for lab, ent in top:
        if float(ent.get("final_score", 0)) < 0.1:
            continue
        rule = assigns.get(str(lab), {}) if isinstance(assigns.get(str(lab)), dict) else {}
        ml_ent = None
        for key in ("basic", "subtle", "legacy"):
            block = ml_blocks.get(key) or {}
            pl = block.get("per_label") or {}
            if lab in pl:
                ml_ent = pl[lab]
                break
        expl = (ml_ent or {}).get("human_readable_summary") or rule.get("human_readable_summary") or ""
        expl = _truncate(str(expl), 360)
        status = (
            f"Final {float(ent.get('final_score', 0)):.2f}; "
            f"{_esc(str(ent.get('agreement_status', '—')))}."
        )
        parts.append(
            f"<p class='fg-just-p'><b>{_esc(str(lab))}.</b> {_esc(expl)} "
            f"<span class='muted'>{status}</span></p>"
        )
        n += 1
        if n >= int(top_n):
            break
    if n == 0:
        parts.append("<p class='muted'>No consensus labels above threshold for a narrative summary.</p>")
    parts.append("</div>")
    return "".join(parts)


def spectrum_card_header_html(
    *,
    name: str,
    anchor: str,
    status: str,
    top3: list[tuple[str, float, str | None]],
    top_cautions: list[str],
    ml_agree: str,
    density: str,
) -> str:
    top_line = ", ".join(f"{lab}" + (f" ({cc})" if cc else "") for lab, _s, cc in top3) or "—"
    maj, mino = split_major_minor_cautions([str(c) for c in (top_cautions or [])])
    caut_line = "; ".join(_truncate(c, 100) for c in maj[:2]) or "—"
    minor = ""
    if mino:
        minor = (
            f"<details class='caution-minor'><summary>Additional notes</summary>"
            f"<p class='muted small'>{_esc(_truncate('; '.join(mino[:6]), 200))}</p></details>"
        )
    links = (
        f"<p class='spec-inline-links'><a href='#{_esc(anchor)}-figure'>Spectrum</a> · "
        f"<a href='#{_esc(anchor)}-explain-assign'>Assignments table</a> · "
        f"<a href='#{_esc(anchor)}-evidence'>Evidence</a> · "
        f"<a href='#{_esc(anchor)}-audit'>Details</a></p>"
    )
    return (
        f"<header class='spec-card-header' id='{_esc(anchor)}-header'>"
        f"<div class='spec-card-title-row'><h2 class='spec-card-title'>{_esc(name)}</h2>"
        f"{status_badge_html(status)}</div>"
        f"<p class='spec-card-compact'><span class='lbl'>Assignments</span> {_esc(_truncate(top_line, 220))}</p>"
        f"<p class='spec-card-compact muted'><span class='lbl'>Cautions</span> {_esc(caut_line)}</p>"
        f"{minor}"
        f"<p class='spec-card-compact muted'><span class='lbl'>ML</span> {_esc(ml_agree)}</p>"
        f"{links}"
        "</header>"
    )


def quick_interpretation_html(pipeline: dict[str, Any], *, top_n: int = 3) -> str:
    parts = ["<div class='quick-interp'>", "<h3>Quick scan</h3>", "<ul>"]
    for lab, sc, cc in top_supported_assignments(pipeline, n=top_n):
        parts.append(f"<li><b>{_esc(lab)}</b> — rule score {sc:.2f}" + (f", <span class='muted'>{_esc(cc)}</span>" if cc else "") + "</li>")
    cons = pipeline.get("consensus") or {}
    top = cons.get("top_labels") or []
    if top:
        lab0, e0 = top[0]
        parts.append(
            "<li class='muted'><b>Leading consensus:</b> "
            f"{_esc(lab0)} (final {float(e0.get('final_score', 0)):.2f}, "
            f"{_esc(e0.get('agreement_status', '—'))})</li>"
        )
    parts.append("</ul></div>")
    return "".join(parts)


def details_block(
    html_id: str,
    summary: str,
    inner: str,
    *,
    open_default: bool,
    extra_class: str = "",
) -> str:
    op = " open" if open_default else ""
    return (
        f"<details{op} class='audit-details {extra_class}' id='{_esc(html_id)}'>"
        f"<summary>{_esc(summary)}</summary><div class='audit-details-body'>{inner}</div></details>"
    )


def _quick_interp_n(pipeline: dict[str, Any], density: str, top_n_summary: int) -> str:
    d = str(density).lower()
    if d == "audit":
        n = max(3, int(top_n_summary))
    elif d == "balanced":
        n = min(3, int(top_n_summary))
    else:
        n = 3
    return quick_interpretation_html(pipeline, top_n=n)


def build_spectrum_body_html(
    *,
    pipeline: dict[str, Any],
    anchor: str,
    gr: str,
    density: str,
    top_n_summary: int,
    include_evidence: bool,
    include_ml: bool,
    include_consensus: bool,
    resolved_mode: str,
    legacy_probs: dict[str, float] | None,
    show_ambiguity_labels: bool,
    show_artifact_flags: bool,
    omit_quick_interpretation: bool = False,
    plot_first_layout: bool = False,
) -> str:
    """Numbered sections 1–8 with progressive disclosure by density."""
    from ml.ftir_report_sections import (
        ambiguity_labels_html,
        artifacts_block_html,
        caution_block_html,
        consensus_html,
        evidence_table_html,
        guardrails_diagnostics_html,
        justification_cards_html,
        rule_assignments_html,
    )

    if not include_evidence:
        return "<p class='muted'>Evidence tables were omitted for this export.</p>"

    is_v3 = gr == "v3"
    ml_on = include_ml and resolved_mode != "none"

    # Full audit bundle (same scientific content as legacy A–G)
    full_evidence_table = rule_assignments_html(
        pipeline["rule_assignments"],
        top_n=24,
        v3_columns=is_v3,
    )
    narrow_evidence = rule_assignments_html(
        pipeline["rule_assignments"],
        top_n=top_n_summary,
        v3_columns=is_v3,
    )
    band_evidence = evidence_table_html(pipeline["evidence"])
    guard_html = guardrails_diagnostics_html(pipeline["rule_assignments"]) if is_v3 else ""
    art_html = (
        ("<h4 class='section-sub'>Artifacts</h4>" + artifacts_block_html(pipeline["evidence"]))
        if show_artifact_flags
        else ""
    )
    amb_html = ""
    if show_ambiguity_labels:
        if is_v3:
            amb_html = ambiguity_labels_html(pipeline["rule_assignments"])
        else:
            from reports.v4_evidence_report import descriptive_ambiguity_html

            amb_html = descriptive_ambiguity_html(pipeline)

    ml_flat = _ml_refinement_flat_html(pipeline, ml_on=ml_on, legacy_probs=legacy_probs)

    cons_full = (
        consensus_html(pipeline["consensus"], top_n=12, with_heading=True) if include_consensus else ""
    )
    cons_full_body = (
        consensus_html(pipeline["consensus"], top_n=12, with_heading=False) if include_consensus else ""
    )
    cons_narrow = (
        consensus_html(
            pipeline["consensus"],
            top_n=min(5, top_n_summary),
            with_heading=False,
        )
        if include_consensus
        else ""
    )

    caut_full = caution_block_html(pipeline)
    cautions_list = collect_pipeline_cautions(pipeline)
    caut_bullets = ""
    if cautions_list:
        maj_b, min_b = split_major_minor_cautions([str(c) for c in cautions_list])
        show_maj = maj_b[:4] if maj_b else cautions_list[:2]
        items = "".join(f"<li>{_esc(_truncate(c, 160))}</li>" for c in show_maj)
        rest_items = "".join(f"<li>{_esc(_truncate(c, 140))}</li>" for c in min_b[:14])
        more = ""
        if min_b:
            more = (
                f"<details class='caution-minor'><summary>Additional notes ({len(min_b)})</summary>"
                f"<ul class='caution-soft'>{rest_items}</ul></details>"
            )
        elif len(cautions_list) > len(show_maj):
            tail = "".join(f"<li>{_esc(_truncate(c, 140))}</li>" for c in cautions_list[len(show_maj) : len(show_maj) + 12])
            more = (
                f"<details class='caution-minor'><summary>More ({len(cautions_list) - len(show_maj)})</summary>"
                f"<ul class='caution-soft'>{tail}</ul></details>"
            )
        caut_bullets = "<ul class='caution caution-soft'>" + items + "</ul>" + more
    else:
        caut_bullets = "<p class='muted'>No pipeline-level cautions.</p>"

    just_html = justification_cards_html(pipeline)

    warn_html = ""
    if pipeline.get("warnings"):
        warn_html = (
            "<h3>Notes</h3><p class='muted'>"
            + _esc("; ".join(pipeline["warnings"]))
            + "</p>"
        )

    if density == "audit":
        inner_parts: list[str] = []
        if not omit_quick_interpretation and not plot_first_layout:
            inner_parts.append(_quick_interp_n(pipeline, "audit", top_n_summary))
        if not plot_first_layout:
            inner_parts.append(
                f"<p class='muted small' id='{_esc(anchor)}-spectrum-hint'>Spectrum (above): hover is local to wavenumber (bands + evidence).</p>"
            )
        if plot_first_layout:
            inner_parts.append(
                details_block(
                    f"{anchor}-evidence-full",
                    "Evidence (table)",
                    f"<div id='{_esc(anchor)}-evidence'>{full_evidence_table}</div>",
                    open_default=True,
                )
            )
        else:
            inner_parts.append(f"<h3>Evidence</h3>{full_evidence_table}")
        inner_parts.append(details_block(f"{anchor}-bands", "Evidence (bands)", band_evidence, open_default=True))
        deconv_html = ""
        try:
            from ml.ftir_deconvolution import deconv_audit_table_html

            deconv_html = deconv_audit_table_html(pipeline.get("evidence") or {})
        except Exception:
            deconv_html = ""
        inner_parts.append(
            details_block(
                f"{anchor}-diagnostics",
                "Diagnostics",
                guard_html + art_html + deconv_html,
                open_default=bool(guard_html or art_html or deconv_html),
            )
        )
        inner_parts.append(details_block(f"{anchor}-amb", "Ambiguity", amb_html, open_default=bool(amb_html)))
        if ml_on:
            inner_parts.append(f"<h3>ML refinement</h3>{ml_flat}")
        else:
            inner_parts.append("<h3>ML refinement</h3><p class='muted'>ML disabled for this run.</p>")
        if plot_first_layout:
            inner_parts.append(
                details_block(
                    f"{anchor}-just",
                    "Details",
                    just_html,
                    open_default=False,
                )
            )
        else:
            inner_parts.append(f"<h3>Details</h3>{just_html}")
        if include_consensus:
            if plot_first_layout:
                inner_parts.append(
                    details_block(
                        f"{anchor}-cons-interpret",
                        "Consensus (full table)",
                        cons_full_body,
                        open_default=False,
                    )
                )
            else:
                inner_parts.append("<h3>Consensus</h3>")
                inner_parts.append(cons_full_body)
        if plot_first_layout:
            if caut_full.strip():
                inner_parts.append(
                    details_block(f"{anchor}-caut-full", "Cautions (full)", caut_full, open_default=False)
                )
        else:
            inner_parts.append(caut_full)
        inner_parts.append(warn_html)
        return "".join(inner_parts)

    if density == "balanced":
        inner_parts: list[str] = []
        if not omit_quick_interpretation and not plot_first_layout:
            inner_parts.append(_quick_interp_n(pipeline, "balanced", top_n_summary))
        if not plot_first_layout:
            inner_parts.append(
                f"<p class='muted small' id='{_esc(anchor)}-spectrum-hint'>Spectrum (above): local hover; tables below.</p>"
            )
        if plot_first_layout:
            inner_parts.append(
                details_block(
                    f"{anchor}-evidence-pack",
                    "Evidence (top rows)",
                    f"<div id='{_esc(anchor)}-evidence'>{narrow_evidence}</div>",
                    open_default=False,
                )
            )
        else:
            inner_parts.append(f"<h3 id='{_esc(anchor)}-evidence'>Evidence</h3>{narrow_evidence}")
        inner_parts.append(
            details_block(f"{anchor}-evidence-all", "Evidence (full table)", full_evidence_table, open_default=False)
        )
        inner_parts.append(details_block(f"{anchor}-bands", "Evidence (bands)", band_evidence, open_default=False))
        inner_parts.append(
            details_block(
                f"{anchor}-diagnostics",
                "Diagnostics",
                guard_html + art_html,
                open_default=False,
            )
        )
        inner_parts.append(details_block(f"{anchor}-amb", "Ambiguity", amb_html, open_default=False))
        if ml_on:
            inner_parts.append(
                details_block(f"{anchor}-ml", "ML refinement", ml_flat, open_default=False)
            )
        else:
            inner_parts.append("<h3>ML refinement</h3><p class='muted'>ML disabled for this run.</p>")
        inner_parts.append(details_block(f"{anchor}-just", "Details", just_html, open_default=False))
        cons_caut_inner = "".join(
            [
                *( [cons_narrow] if include_consensus else []),
                caut_bullets,
            ]
        )
        if plot_first_layout:
            inner_parts.append(
                details_block(
                    f"{anchor}-cons-caut",
                    "Consensus",
                    cons_caut_inner,
                    open_default=False,
                )
            )
        else:
            inner_parts.append("<h3>Consensus</h3>")
            if include_consensus:
                inner_parts.append(cons_narrow)
            inner_parts.append(caut_bullets)
        if include_consensus:
            inner_parts.append(
                details_block(f"{anchor}-cons-full", "Consensus (full table)", cons_full, open_default=False)
            )
        inner_parts.append(details_block(f"{anchor}-caut-full", "Cautions (full)", caut_full, open_default=False))
        inner_parts.append(warn_html)
        return "".join(inner_parts)

    # summary
    inner_parts: list[str] = []
    if not omit_quick_interpretation and not plot_first_layout:
        inner_parts.append(_quick_interp_n(pipeline, "summary", top_n_summary))
    if not plot_first_layout:
        inner_parts.append(
            f"<p class='muted small' id='{_esc(anchor)}-spectrum-hint'>Spectrum (above): compact layout.</p>"
        )
    if plot_first_layout:
        inner_parts.append(
            details_block(
                f"{anchor}-evidence-pack",
                "Evidence (top rows)",
                f"<div id='{_esc(anchor)}-evidence'>{narrow_evidence}</div>",
                open_default=False,
            )
        )
    else:
        inner_parts.append(f"<h3 id='{_esc(anchor)}-evidence'>Evidence</h3>{narrow_evidence}")
    cons_caut_inner2 = "".join(
        [
            *( [cons_narrow] if include_consensus else []),
            caut_bullets,
        ]
    )
    if plot_first_layout:
        inner_parts.append(
            details_block(
                f"{anchor}-cons-caut",
                "Consensus",
                cons_caut_inner2,
                open_default=False,
            )
        )
    else:
        inner_parts.append("<h3>Consensus</h3>")
        if include_consensus:
            inner_parts.append(cons_narrow)
        inner_parts.append(caut_bullets)
    audit_inner = "".join(
        [
            f"<h3>Evidence</h3>{full_evidence_table}",
            f"<h3>Evidence (bands)</h3>{band_evidence}",
            f"<h3>Diagnostics</h3>{guard_html}{art_html}",
            f"<h3>Ambiguity</h3>{amb_html}",
            f"<h3>ML refinement</h3>{ml_flat or '<p class=\"muted\">ML disabled.</p>'}",
            f"<h3>Details</h3>{just_html}",
            cons_full,
            caut_full,
            warn_html,
        ]
    )
    inner_parts.append(
        details_block(
            f"{anchor}-audit",
            "Details (full audit)",
            audit_inner,
            open_default=False,
        )
    )
    return "".join(inner_parts)


def spectrum_summary_row(
    *,
    name: str,
    anchor: str,
    pipeline: dict[str, Any],
    ml_enabled: bool,
) -> dict[str, Any]:
    """One row of data for the batch Summary Table."""
    top3 = top_supported_assignments(pipeline, n=3)
    needs, _reasons = spectrum_needs_review(pipeline, ml_enabled=ml_enabled)
    status = compute_card_status(pipeline, ml_enabled=ml_enabled)
    amb = (pipeline.get("rule_assignments") or {}).get("ambiguity_labels") or []
    amb_titles = [str(a.get("title") or a.get("id") or "") for a in amb if a]
    caut_full_list = collect_pipeline_cautions(pipeline, max_total=24)
    caut_major, caut_minor = split_major_minor_cautions(caut_full_list)
    caut_short = caut_full_list[:4]
    fam_t, spec_t = families_and_specifics_from_assignments(pipeline)
    flags = ((pipeline.get("evidence") or {}).get("artifacts") or {}).get("flags") or {}
    major_art = any(
        flags.get(k)
        for k in (
            "water_vapor_or_moisture_like",
            "co2_region_elevated",
            "fingerprint_crowding",
            "atr_crystal_fingerprint_overlap",
            "possible_saturation",
            "weak_nitrile_region_spike",
        )
    )
    has_high_confidence = any(
        sc >= 0.52 and str(cc or "") in ("strong", "supported")
        for _lab, sc, cc in top_supported_assignments(pipeline, n=12)
    )
    has_major_cautions = len(caut_full_list) >= 4 or major_art
    has_ambiguity = bool(amb)
    has_ml_conflict = False
    if ml_enabled:
        for _lab, ent in (pipeline.get("consensus") or {}).get("top_labels") or []:
            if ent.get("agreement_status") in ("conflict", "ml_only_warning"):
                has_ml_conflict = True
                break
    return {
        "name": name,
        "anchor": anchor,
        "status": status,
        "top3": top3,
        "families_text": fam_t,
        "specifics_text": spec_t,
        "amb_titles": amb_titles,
        "caut_short": caut_short,
        "caut_major": caut_major,
        "caut_minor": caut_minor,
        "caut_full": caut_full_list,
        "ml_agree": ml_agreement_summary(pipeline, ml_enabled=ml_enabled),
        "needs_review": needs,
        "has_high_confidence": has_high_confidence,
        "has_major_cautions": has_major_cautions,
        "has_ambiguity": has_ambiguity,
        "has_ml_conflict": has_ml_conflict,
    }


KRONECKER_PI_EXTRA_CSS = """
.summary-table-section { margin: 8px 0 32px; max-width: 100%; }
.summary-table-heading { font-size: 1.2rem; font-weight: 650; margin: 0 0 10px; letter-spacing: -0.01em; }
.summary-table-wrap { margin-top: 2px; }
.tbl-summary { width: 100%; }
.tbl-summary th, .tbl-summary td { font-size: 11.5px; line-height: 1.35; padding: 8px 9px; vertical-align: top; }
.td-spectrum { max-width: 11rem; }
.section-sub { font-size: 0.82rem; margin: 0 0 6px; color: #64748b; font-weight: 600; }
.spec-card-header { border-bottom: 1px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 12px; }
.spec-card-title-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.spec-title-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.spec-card-title { margin: 0; font-size: 1.08rem; font-weight: 650; }
.spec-card-compact { margin: 4px 0; font-size: 0.84rem; line-height: 1.45; }
.spec-card-compact .lbl { color: #64748b; font-weight: 600; margin-right: 6px; }
.spec-inline-links { margin: 6px 0 2px; font-size: 0.82rem; }
.spec-inline-links a { color: #2563eb; text-decoration: none; font-weight: 500; }
.spec-inline-links a:hover { text-decoration: underline; }
.spec-title-after-plot { margin: 16px 0 12px; padding-bottom: 12px; border-bottom: 1px solid #e8edf2; }
.card { margin: 22px 0 36px; padding: 14px 16px 20px; border-color: #e8edf2; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
.badge-sci { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; letter-spacing: 0.02em; }
.badge-strong { background: #d1fae5; color: #065f46; }
.badge-supported { background: #e0f2fe; color: #075985; }
.badge-ambiguous { background: #fef9c3; color: #854d0e; }
.badge-attn { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
.badge-muted { background: #f4f4f5; color: #52525b; }
.band-evidence-map { margin: 16px 0 20px; }
.fg-justify-panels { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin: 14px 0 18px; }
.fg-justify-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; background: #fafbfc; }
.fg-justify-card h4 { margin: 0 0 6px; font-size: 0.92rem; }
.fg-justify-list { margin: 0; padding-left: 0; list-style: none; }
.fg-justify-list li { margin: 4px 0; line-height: 1.4; }
.ev-icon { font-weight: 700; margin-right: 4px; }
.ev-ok { color: #059669; }
.ev-part { color: #b45309; }
.ev-miss { color: #dc2626; }
.support-pill { font-size: 0.75rem; padding: 2px 7px; border-radius: 6px; background: #f1f5f9; color: #334155; }
.ambiguity-rich { margin: 10px 0 14px; }
.ambiguity-rich-list { margin: 6px 0 0 18px; font-size: 0.86rem; }
.tbl-explain-assign td.muted { color: #64748b; font-size: 0.82rem; }
.caution-soft { color: #57534e; font-size: 12px; margin: 6px 0 0 18px; }
.caution-soft li { margin: 2px 0; }
.caution-minor { margin: 6px 0; font-size: 0.82rem; }
.caution-minor summary { cursor: pointer; color: #64748b; font-weight: 500; }
.caution-minor-list { margin: 4px 0 0; }
.quick-interp ul { margin: 8px 0 0 18px; font-size: 0.88rem; }
.pre-figure-interp { margin-bottom: 10px; }
.post-figure-blocks { margin-top: 6px; }
.audit-details { margin: 10px 0; border: 1px solid #e8edf2; border-radius: 8px; padding: 0 12px 8px; background: #fcfcfd; }
.audit-details summary { cursor: pointer; padding: 8px 0; font-weight: 600; color: #475569; font-size: 0.9rem; }
.audit-details-body { padding-bottom: 6px; font-size: 0.88rem; }
.metadata-details { margin: 8px 0 10px; font-size: 0.84rem; }
.metadata-details summary { cursor: pointer; color: #64748b; }
pre.mono { font-size: 0.76rem; overflow-x: auto; background: #f8fafc; padding: 8px; border-radius: 6px; border: 1px solid #e8edf2; }
.table-scroll { overflow-x: auto; margin-top: 4px; }
.tbl { font-size: 11.5px; }
.tbl th, .tbl td { padding: 7px 8px; }
.tbl-zebra tbody tr:nth-child(even) { background: #f8fafc; }
.tbl-sticky thead th { position: sticky; top: 0; z-index: 2; background: #eef2f7; box-shadow: 0 1px 0 #e5e7eb; }
.tbl.small, .tbl.small th, .tbl.small td { font-size: 11px; }
.inline-details { display: inline; margin-left: 4px; }
.inline-details summary { display: inline; cursor: pointer; color: #2563eb; font-size: 0.8rem; }
.small { font-size: 0.8rem; }
main { max-width: 1180px; }
.nav-section-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin: 12px 0 6px; }
.fg-highlights { margin: 4px 0 6px; }
.fg-highlights h3 { font-size: 1rem; }
.fg-justify-below { margin: 10px 0 12px; max-width: 95ch; }
.fg-justify-below h3 { margin-top: 0; font-size: 1rem; }
.fg-just-p { font-size: 0.88rem; line-height: 1.45; margin: 6px 0; }
.expand-stack { margin-top: 4px; padding-top: 12px; border-top: 1px solid #eef2f7; }
.explain-assign-wrap { margin: 12px 0 14px; }
.explain-assign-heading { font-size: 1rem; margin: 0 0 6px; }
.explain-assign-scroll { max-height: 420px; overflow: auto; border: 1px solid #e8edf2; border-radius: 8px; }
.tbl-explain-assign th, .tbl-explain-assign td { max-width: 14rem; vertical-align: top; }
.tbl-explain-assign td:nth-child(12) { max-width: 22rem; }
#sidebar .badge-sci { margin-left: 4px; vertical-align: middle; transform: scale(0.95); }
.tbl th { font-weight: 600; }
"""
