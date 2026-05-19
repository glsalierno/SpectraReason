"""HTML section builders for evidence-first FTIR reports."""

from __future__ import annotations

import html
import json
from typing import Any


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _peak_hover_fragment(p: Any) -> str:
    """Rule / ML rows may store peaks as dicts or pre-formatted strings."""
    if isinstance(p, dict):
        return f"{float(p.get('wn_cm1', p.get('wn', 0))):.0f}"
    return str(p).strip()


def evidence_table_html(evidence: dict[str, Any]) -> str:
    matches = [m for m in (evidence.get("band_matches") or []) if m.get("matched")]
    rows = sorted(matches, key=lambda m: -float(m.get("support_score", 0)))[:20]
    parts = [
        "<h3>Matched FTIR band evidence</h3>",
        "<table class='tbl'><tr><th>Label</th><th>Region (cm⁻¹)</th><th>Mode</th>"
        "<th>Importance</th><th>Support</th></tr>",
    ]
    for m in rows:
        parts.append(
            f"<tr><td>{_esc(m.get('label'))}</td>"
            f"<td>{m.get('region_min_cm1'):.0f}–{m.get('region_max_cm1'):.0f}</td>"
            f"<td>{_esc(m.get('mode'))}</td>"
            f"<td>{_esc(m.get('importance'))}</td>"
            f"<td>{float(m.get('support_score', 0)):.3f}</td></tr>"
        )
    parts.append("</table>")
    ratios = evidence.get("ratios") or {}
    if ratios:
        parts.append("<p class='muted'>Ratios: " + ", ".join(f"{k}={v:.3f}" for k, v in ratios.items()) + "</p>")
    return "".join(parts)


def rule_assignments_html(rule_result: dict[str, Any], *, top_n: int = 12, v3_columns: bool = False) -> str:
    items = sorted(
        (rule_result.get("assignments") or {}).items(),
        key=lambda kv: -float(kv[1].get("score", 0)),
    )[:top_n]
    use_v3 = v3_columns or any(
        (e.get("confidence_class") or e.get("evidence_completeness")) for _, e in items if isinstance(e, dict)
    )
    header = (
        "<tr><th>Functional group</th><th>Rule score</th><th>Confidence (rule)</th>"
        "<th>Class</th><th>Evidence</th><th>Type</th><th>Supporting regions</th><th>Competing</th>"
        "<th>Missing / cautions</th><th>Summary</th></tr>"
        if use_v3
        else "<tr><th>Functional group</th><th>Rule score</th><th>Confidence</th><th>Summary</th></tr>"
    )
    parts = [
        "<h3>Evidence-supported assignment</h3>",
        "<table class='tbl'>",
        header,
    ]
    for lab, ent in items:
        if float(ent.get("score", 0)) < 0.12:
            continue
        miss = ent.get("missing_expected_bands") or []
        miss_s = "; ".join(str(x) for x in miss[:3]) if miss else "—"
        confs = ent.get("caution_flags") or []
        note = (confs[0][:120] + "…") if confs else "—"
        if use_v3:
            sup_b = ent.get("supporting_bands") or []
            sup_short = "; ".join(_esc(x) for x in sup_b[:2]) if sup_b else "—"
            comp_ex = ent.get("competing_explanation")
            comp_s = _esc(comp_ex) if comp_ex else "—"
            cev = ent.get("conflicting_evidence") or []
            miss_block = miss_s
            if cev:
                miss_block = f"{miss_s}; " + "; ".join(_esc(x) for x in cev[:2])
            parts.append(
                f"<tr><td>{_esc(lab)}</td><td>{float(ent.get('score', 0)):.3f}</td>"
                f"<td>{_esc(ent.get('confidence'))}</td>"
                f"<td>{_esc(ent.get('confidence_class', '—'))}</td>"
                f"<td>{_esc(ent.get('evidence_completeness', '—'))}</td>"
                f"<td>{_esc(ent.get('assignment_type', '—'))}</td>"
                f"<td>{sup_short}</td><td>{comp_s}</td>"
                f"<td>{miss_block} / {_esc(note)}</td>"
                f"<td>{_esc(ent.get('human_readable_summary', ''))}</td></tr>"
            )
        else:
            parts.append(
                f"<tr><td>{_esc(lab)}</td><td>{float(ent.get('score', 0)):.3f}</td>"
                f"<td>{_esc(ent.get('confidence'))}</td>"
                f"<td>{_esc(ent.get('human_readable_summary', ''))}</td></tr>"
            )
    parts.append("</table>")
    return "".join(parts)


def artifacts_block_html(evidence: dict[str, Any]) -> str:
    art = evidence.get("artifacts") or {}
    flags = art.get("flags") or {}
    active = [k for k, v in flags.items() if v]
    if not active and not art.get("cautions"):
        return ""
    parts = ["<h3>Artifact and interference flags</h3>", "<ul class='caution'>"]
    for k in active[:16]:
        parts.append(f"<li>{_esc(k.replace('_', ' '))}</li>")
    for c in (art.get("cautions") or [])[:8]:
        parts.append(f"<li>{_esc(c)}</li>")
    parts.append("</ul>")
    return "".join(parts)


def guardrails_diagnostics_html(rule_result: dict[str, Any]) -> str:
    rows = rule_result.get("guardrails_diagnostics") or []
    if not rows:
        return ""
    ver = _esc(rule_result.get("guardrails_version", ""))
    parts = [
        f"<h3>Guardrail diagnostics ({ver})</h3>",
        "<table class='tbl'><tr><th>Label</th><th>Score</th><th>Class</th>"
        "<th>Evidence</th><th>Req. fraction</th><th>Competitor</th><th>Comp. score</th></tr>",
    ]
    for r in rows[:20]:
        cs = r.get("competitor_score")
        cs_cell = f"{float(cs):.3f}" if cs is not None else "—"
        parts.append(
            f"<tr><td>{_esc(r.get('label'))}</td>"
            f"<td>{float(r.get('score', 0)):.3f}</td>"
            f"<td>{_esc(r.get('confidence_class'))}</td>"
            f"<td>{_esc(r.get('evidence_completeness'))}</td>"
            f"<td>{float(r.get('required_fraction', 0)):.2f}</td>"
            f"<td>{_esc(r.get('competitor'))}</td>"
            f"<td>{cs_cell}</td></tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def ambiguity_labels_html(rule_result: dict[str, Any]) -> str:
    amb = rule_result.get("ambiguity_labels") or []
    if not amb:
        return ""
    parts = ["<h3>Ambiguity / family fallback labels</h3>", "<ul>"]
    for a in amb:
        parts.append(
            f"<li><b>{_esc(a.get('title'))}</b> ({_esc(a.get('id'))}): {_esc(a.get('reason'))} "
            f"<span class='muted'>Related: {_esc(', '.join(a.get('related_labels') or []))}</span></li>"
        )
    parts.append("</ul>")
    return "".join(parts)


def ml_refinement_html(ml_block: dict[str, Any] | None, *, title: str) -> str:
    if not ml_block or not ml_block.get("per_label"):
        return ""
    ml_disp = str(ml_block.get("ml_score_display") or "")
    items = sorted(
        ml_block["per_label"].items(),
        key=lambda kv: -float(kv[1].get("ml_score") or kv[1].get("ml_probability") or 0),
    )[:12]
    parts = [
        f"<h3>{_esc(title)}</h3>",
        f"<p class='muted'>{_esc(ml_disp)} — FTIR band evidence remains primary; ML is secondary.</p>",
        "<table class='tbl'><tr><th>FG</th><th>Rule</th><th>ML</th>"
        "<th>Final</th><th>Agreement</th><th>Spectral context</th></tr>",
    ]
    for lab, ent in items:
        rule_s = float(ent.get("rule_score", 0))
        if ent.get("ml_probability") is None and ent.get("ml_score") is None and rule_s < 0.05:
            continue
        if ent.get("ml_probability") is not None:
            disp = str(ent.get("ml_score_label") or ml_disp or "").lower()
            if "calibrated svm probability" in disp or "calibrated" in disp:
                ml_cell = f"{float(ent['ml_probability']):.3f} (calibrated SVM probability)"
            else:
                ml_cell = f"{float(ent['ml_probability']):.3f} ({_esc(ent.get('ml_score_label', 'ML'))})"
        elif ent.get("ml_score") is not None:
            ml_cell = f"{float(ent['ml_score']):.4f} ({_esc(ent.get('ml_score_label', 'SVM score'))})"
        else:
            ml_cell = "—"
        ctx = ""
        if ent.get("supporting_peaks"):
            peaks_bits = [_peak_hover_fragment(p) for p in (ent.get("supporting_peaks") or [])[:4]]
            ctx += "Peaks: " + ", ".join(peaks_bits) + ". "
        if ent.get("missing_expected_bands"):
            ctx += "Missing bands noted. "
        if not ctx.strip():
            ctx = "See evidence table; ML never stands alone without spectral context in this report view."
        parts.append(
            f"<tr><td>{_esc(lab)}</td><td>{rule_s:.3f}</td><td>{ml_cell}</td>"
            f"<td>{float(ent.get('final_score', 0)):.3f}</td><td>{_esc(ent.get('agreement_status'))}</td>"
            f"<td>{_esc(ctx)}</td></tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def consensus_html(consensus: dict[str, Any], *, top_n: int = 12, with_heading: bool = True) -> str:
    items = consensus.get("top_labels") or []
    parts: list[str] = []
    if with_heading:
        parts.extend(
            [
                "<h3>Consensus interpretation</h3>",
                "<p class='hint'>Combines evidence-supported scores with optional ML refinement. "
                "Not ground truth.</p>",
            ]
        )
    parts.extend(
        [
        "<table class='tbl'><tr><th>FG</th><th>Rule</th><th>ML (basic)</th><th>ML (subtle)</th>"
        "<th>Final</th><th>Status</th></tr>",
    ]
    )
    for lab, ent in items[: max(1, int(top_n))]:
        if float(ent.get("final_score", 0)) < 0.12:
            continue
        parts.append(
            f"<tr><td>{_esc(lab)}</td><td>{float(ent.get('rule_score', 0)):.3f}</td>"
            f"<td>{_fmt_opt(ent.get('ml_probability_basic'))}</td>"
            f"<td>{_fmt_opt(ent.get('ml_probability_subtle'))}</td>"
            f"<td>{float(ent.get('final_score', 0)):.3f}</td>"
            f"<td>{_esc(ent.get('agreement_status'))}</td></tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _fmt_opt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.3f}"


def caution_block_html(pipeline_result: dict[str, Any]) -> str:
    cautions: list[str] = list(pipeline_result.get("warnings") or [])
    for _lab, ent in (pipeline_result.get("rule_assignments", {}).get("assignments") or {}).items():
        for c in ent.get("caution_flags") or []:
            if c not in cautions:
                cautions.append(f"{_lab}: {c}")
    ml = pipeline_result.get("ml_refinement") or {}
    for key in ("basic", "subtle", "legacy"):
        block = ml.get(key)
        if not block:
            continue
        for lab, ent in (block.get("per_label") or {}).items():
            if ent.get("agreement_status") == "ml_only_warning":
                cautions.append(f"{lab}: ML-only warning (weak band evidence)")
    if not cautions:
        return ""
    return "<h3>Caution flags</h3><ul class='caution'>" + "".join(
        f"<li>{_esc(c)}</li>" for c in cautions[:16]
    ) + "</ul>"


def justification_cards_html(pipeline_result: dict[str, Any], *, top_n: int = 6) -> str:
    """Per-label evidence + optional ML detail cards (no ML-only rows without context)."""
    consensus = pipeline_result.get("consensus") or {}
    top = consensus.get("top_labels") or []
    parts = ["<h3>Details</h3>", "<div class='expl-cards'>"]
    ml_blocks = pipeline_result.get("ml_refinement") or {}
    for lab, ent in top[:top_n]:
        if float(ent.get("final_score", 0)) < 0.1:
            continue
        rule = (pipeline_result.get("rule_assignments") or {}).get("assignments", {}).get(lab, {})
        ml_ent = None
        for key in ("basic", "subtle", "legacy"):
            block = ml_blocks.get(key) or {}
            pl = block.get("per_label") or {}
            if lab in pl:
                ml_ent = pl[lab]
                break
        peaks = rule.get("supporting_peaks") or []
        bands = rule.get("supporting_bands") or []
        missing = rule.get("missing_expected_bands") or []
        conflicts = rule.get("conflicting_evidence") or []
        peak_items = peaks[:8] if isinstance(peaks, list) else []
        peaks_parts: list[str] = [_peak_hover_fragment(p) for p in peak_items]
        peaks_txt = ", ".join(peaks_parts)
        status = (
            f"Agreement: {_esc(ent.get('agreement_status'))}. "
            f"Final score {float(ent.get('final_score', 0)):.3f}."
        )
        ml_line = ""
        if ml_ent:
            if ml_ent.get("ml_probability") is not None:
                ml_line = (
                    " ML (calibrated probability): "
                    f"{float(ml_ent['ml_probability']):.3f}."
                )
            elif ml_ent.get("ml_score") is not None:
                ml_line = (
                    f" ML ({_esc(ml_ent.get('ml_score_label', 'score'))}): "
                    f"{float(ml_ent['ml_score']):.4f}."
                )
        expl = _esc((ml_ent or {}).get("human_readable_summary") or rule.get("human_readable_summary") or "")
        parts.append(
            f"<div class='expl-card'><h4>{_esc(lab)}</h4>"
            f"<p><b>Final status:</b> {_esc(status)}</p>"
            f"<p><b>Rule score:</b> {float(rule.get('score', 0)):.3f}</p>"
            f"<p>{_esc(ml_line)}</p>"
            f"<p><b>Supporting peaks (cm⁻¹):</b> {_esc(peaks_txt)}</p>"
            f"<p><b>Supporting expected bands:</b> {_esc('; '.join(str(b) for b in bands[:6]))}</p>"
            f"<p><b>Missing expected bands:</b> {_esc('; '.join(str(b) for b in missing[:6]))}</p>"
            f"<p><b>Conflicts:</b> {_esc('; '.join(str(c) for c in conflicts[:4]))}</p>"
            f"<p>{expl}</p></div>"
        )
    parts.append("</div>")
    return "".join(parts)
