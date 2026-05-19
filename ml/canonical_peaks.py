"""
Canonical peak and band-evidence model for FTIR reports.

All plot labels, tables, hover, and CSV exports should reference peak_id from here.
"""

from __future__ import annotations

import math
from typing import Any, Literal

PEAK_LINK_TOL_CM1 = 12.0
ROUND_WN_DIGITS = 1

EvidenceSource = Literal["observed_peak", "region_activity", "deconv_component"]


def _round_wn(wn: float) -> float:
    return round(float(wn), ROUND_WN_DIGITS)


def assign_peak_ids(peaks: list[dict[str, Any]]) -> None:
    """Stable P0001… ids sorted by descending wavenumber."""
    ordered = sorted(peaks, key=lambda p: -float(p.get("wn_cm1", 0) or 0))
    for i, p in enumerate(ordered, start=1):
        pid = f"P{i:04d}"
        p["peak_id"] = pid


def peak_id_for_wavenumber(
    peaks: list[dict[str, Any]],
    wn: float,
    *,
    tol_cm1: float = PEAK_LINK_TOL_CM1,
) -> str | None:
    best_id: str | None = None
    best_d = float(tol_cm1) + 1.0
    for p in peaks:
        d = abs(float(p.get("center_cm1", p.get("wn_cm1", 0))) - float(wn))
        if d <= tol_cm1 and d < best_d:
            best_d = d
            best_id = str(p.get("peak_id", ""))
    return best_id or None


def _labels_for_band(band_id: str, pipeline: dict[str, Any]) -> list[str]:
    from reports.v4_evidence_report import _labels_for_band as _v4_labels

    return _v4_labels(band_id, pipeline)


def _rel_height(peak: dict[str, Any], y_max: float) -> float:
    if peak.get("rel_height") is not None:
        return float(peak["rel_height"])
    h = float(peak.get("height", 0) or 0)
    return h / (float(y_max) + 1e-9)


def _canonical_row_from_detected(p: dict[str, Any], *, y_max: float) -> dict[str, Any]:
    wn = float(p.get("wn_cm1", 0) or 0)
    rel = _rel_height(p, y_max)
    return {
        "peak_id": str(p.get("peak_id", "")),
        "center_cm1": wn,
        "height": float(p.get("height", 0) or 0),
        "prominence": float(p.get("local_prominence", p.get("prominence", 0)) or 0),
        "width": float(p.get("quality_width_cm1", 0) or 0),
        "quality_class": str(p.get("peak_quality", "moderate")),
        "detected_source": str(p.get("detected_source", "pick_spectral_peaks")),
        "plotted": False,
        "labeled": False,
        "label_text": "",
        "label_reason": "",
        "matched_band_ids": [],
        "matched_regions": [],
        "candidate_assignments": [],
        "evidence_roles": [],
        "warnings": list(p.get("warnings") or []),
        "rel_height": rel,
        "_raw": p,
    }


def build_canonical_peak_table(
    pipeline: dict[str, Any],
    *,
    label_min_height: float = 0.05,
    label_all_above_height: float | None = None,
    report_audience: str = "front",
) -> dict[str, Any]:
    """
    Build canonical_peaks and deduplicated evidence_rows on the pipeline.
    Mutates evidence peaks (peak_id) and band_matches (canonical_peak_id).
    """
    evidence = pipeline.get("evidence") or {}
    raw_peaks = list(evidence.get("peaks") or [])
    y_max = float((evidence.get("summary") or {}).get("y_max", 1.0) or 1.0)
    assign_peak_ids(raw_peaks)

    label_floor = float(label_all_above_height if label_all_above_height is not None else label_min_height)

    canonical: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for p in raw_peaks:
        row = _canonical_row_from_detected(p, y_max=y_max)
        canonical.append(row)
        by_id[row["peak_id"]] = row
        p["peak_id"] = row["peak_id"]

    # Link band_matches → canonical peak (never invent center from region midpoint).
    for bm in evidence.get("band_matches") or []:
        if not isinstance(bm, dict):
            continue
        bid = str(bm.get("band_id", ""))
        near = [x for x in (bm.get("peaks_near") or []) if isinstance(x, dict)]
        pid: str | None = None
        if near:
            best = max(near, key=lambda x: float(x.get("rel_height", x.get("height", 0)) or 0))
            pid = peak_id_for_wavenumber(canonical, float(best.get("wn_cm1", 0)))
            if pid and pid in by_id:
                if bid and bid not in by_id[pid]["matched_band_ids"]:
                    by_id[pid]["matched_band_ids"].append(bid)
                for npk in near:
                    npid = peak_id_for_wavenumber(canonical, float(npk.get("wn_cm1", 0)))
                    if npid and npid in by_id and bid and bid not in by_id[npid]["matched_band_ids"]:
                        by_id[npid]["matched_band_ids"].append(bid)
        bm["canonical_peak_id"] = pid
        bm["evidence_source"] = "observed_peak" if pid else "region_activity"

    # Plot / label flags (user: keep all peaks ≥ label_floor labeled).
    for row in canonical:
        rel = float(row["rel_height"])
        q = row["quality_class"]
        noise = q == "noise_like"
        row["plotted"] = (not noise) and rel >= label_floor
        row["labeled"] = row["plotted"]
        if row["labeled"]:
            row["label_text"] = f"{round(row['center_cm1']):.0f}"
            raw = row["_raw"]
            row["label_reason"] = str(raw.get("label_reason") or "height_prominence")
            raw["label_reason"] = row["label_reason"]

    evidence_rows = _build_evidence_rows(
        pipeline,
        canonical,
        by_id,
        audience=str(report_audience or "front").lower(),
    )

    stats = {
        "canonical_peak_count": len(canonical),
        "plotted_peak_count": sum(1 for r in canonical if r["plotted"]),
        "labeled_peak_count": sum(1 for r in canonical if r["labeled"]),
        "evidence_rows_raw": evidence_rows.get("_raw_count", 0),
        "evidence_rows_deduped": len(evidence_rows.get("rows") or []),
        "label_min_height": label_floor,
    }

    pack = {
        "peaks": canonical,
        "evidence_rows": evidence_rows.get("rows") or [],
        "peak_by_id": by_id,
        "stats": stats,
    }
    pipeline["canonical_peaks"] = pack
    return pack


def _build_evidence_rows(
    pipeline: dict[str, Any],
    canonical: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    audience: str,
) -> dict[str, Any]:
    evidence = pipeline.get("evidence") or {}
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    matches = [m for m in (evidence.get("band_matches") or []) if m.get("matched")]
    matches.sort(key=lambda m: -float(m.get("support_score", 0) or 0))

    raw_rows: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str, str]] = set()

    for bm in matches[:48]:
        bid = str(bm.get("band_id", ""))
        pid = bm.get("canonical_peak_id")
        if isinstance(pid, str) and pid not in by_id:
            pid = None
        lo = float(bm.get("region_min_cm1", 0) or 0)
        hi = float(bm.get("region_max_cm1", 0) or 0)
        region_range = f"{lo:.0f}–{hi:.0f} cm⁻¹" if hi > lo else "—"
        band_label = str(bm.get("label") or bm.get("subclass") or bid)
        labels = _labels_for_band(bid, pipeline)
        if not labels:
            labels = [""]
        for lab in labels:
            key = (pid, bid, lab)
            if key in seen:
                continue
            seen.add(key)
            if pid and pid in by_id:
                crow = by_id[pid]
                source: EvidenceSource = "observed_peak"
                peak_wn = round(float(crow["center_cm1"]))
                band_region = band_label
            else:
                source = "region_activity"
                peak_wn = None
                band_region = f"{band_label} ({region_range})"
            raw_rows.append(
                {
                    "peak_id": pid,
                    "band_id": bid,
                    "assignment_label": lab,
                    "source": source,
                    "peak_cm1": peak_wn,
                    "band_region": band_region,
                    "region_range": region_range,
                    "possible_functional_groups": [lab] if lab else [],
                    "support_score": float(bm.get("support_score", 0) or 0),
                    "peak_quality": str(bm.get("peak_quality") or ""),
                    "notes": [],
                }
            )

    rows = _merge_front_evidence_rows(raw_rows) if audience == "front" else raw_rows
    if audience == "front":
        rows = [r for r in rows if r.get("possible_functional_groups") or r.get("peak_cm1") is not None]
    return {"rows": rows, "_raw_count": len(raw_rows)}


def _merge_front_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate band/peak rows; combine FG labels; drop empty '—' only groups."""
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        pid = r.get("peak_id")
        peak_key = r.get("peak_cm1") if pid else r.get("band_id")
        bkey = (peak_key, r.get("band_region"), r.get("source"))
        if bkey not in buckets:
            buckets[bkey] = dict(r)
            buckets[bkey]["possible_functional_groups"] = list(r.get("possible_functional_groups") or [])
            buckets[bkey]["notes"] = list(r.get("notes") or [])
            continue
        b = buckets[bkey]
        for lab in r.get("possible_functional_groups") or []:
            if lab and lab not in b["possible_functional_groups"]:
                b["possible_functional_groups"].append(lab)
    out: list[dict[str, Any]] = []
    for b in buckets.values():
        fgs = [x for x in b.get("possible_functional_groups") or [] if x]
        if not fgs and b.get("source") == "region_activity":
            fgs = []
        b["possible_functional_groups"] = fgs
        out.append(b)
    out.sort(key=lambda r: (-(r.get("peak_cm1") or 0) if r.get("peak_cm1") else 0))
    return out


def peaks_as_legacy_dicts(
    pack: dict[str, Any] | None,
    *,
    plotted: bool | None = None,
    labeled: bool | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical rows to peak dicts used by Plotly builders."""
    if not pack:
        return []
    out: list[dict[str, Any]] = []
    for row in pack.get("peaks") or []:
        if plotted is not None and bool(row.get("plotted")) != plotted:
            continue
        if labeled is not None and bool(row.get("labeled")) != labeled:
            continue
        p = dict(row.get("_raw") or {})
        p["peak_id"] = row["peak_id"]
        p["wn_cm1"] = float(row["center_cm1"])
        p["label_reason"] = row.get("label_reason") or p.get("label_reason")
        out.append(p)
    return out


StaticLabelPolicy = Literal["key", "top", "all"]


def select_static_label_peak_ids(
    pack: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    policy: StaticLabelPolicy = "key",
    max_labels: int = 12,
) -> set[str]:
    """Subset of labeled peaks for PNG/SVG (interactive HTML may show more)."""
    labeled = [r for r in (pack.get("peaks") or []) if r.get("labeled")]
    if policy == "all":
        return {str(r["peak_id"]) for r in labeled}
    if policy == "top":
        ranked = sorted(labeled, key=lambda r: -float(r.get("prominence", 0) or r.get("rel_height", 0)))
        return {str(r["peak_id"]) for r in ranked[: max(1, int(max_labels))]}
    # key: peaks tied to supported assignments + strongest diagnostics
    key_ids: set[str] = set()
    assigns = (pipeline.get("rule_assignments") or {}).get("assignments") or {}
    for _lab, ent in assigns.items():
        if not isinstance(ent, dict):
            continue
        if float(ent.get("score", 0) or 0) < 0.22:
            continue
        for bid in ent.get("supporting_band_ids") or []:
            for row in labeled:
                if bid in (row.get("matched_band_ids") or []):
                    key_ids.add(str(row["peak_id"]))
    ranked = sorted(labeled, key=lambda r: -float(r.get("rel_height", 0)))
    for r in ranked:
        if r.get("quality_class") == "strong":
            key_ids.add(str(r["peak_id"]))
        if len(key_ids) >= max_labels:
            break
    if len(key_ids) < max_labels:
        for r in ranked:
            key_ids.add(str(r["peak_id"]))
            if len(key_ids) >= max_labels:
                break
    return key_ids


def validate_report_peak_consistency(
    pipeline: dict[str, Any],
    *,
    round_tol: float = 1.0,
) -> dict[str, Any]:
    """Validate canonical peak usage across pipeline artifacts."""
    issues: list[dict[str, Any]] = []
    pack = pipeline.get("canonical_peaks") or {}
    peaks = pack.get("peaks") or []
    by_id = pack.get("peak_by_id") or {}

    for row in peaks:
        if row.get("labeled") and not row.get("peak_id"):
            issues.append({"code": "labeled_missing_id", "row": row})
        if row.get("labeled") and not row.get("label_text"):
            issues.append({"code": "labeled_missing_text", "peak_id": row.get("peak_id")})

    for er in pack.get("evidence_rows") or []:
        pid = er.get("peak_id")
        if er.get("source") == "observed_peak":
            if not pid or pid not in by_id:
                issues.append({"code": "evidence_peak_id_missing", "row": er})
            elif er.get("peak_cm1") is not None:
                center = round(float(by_id[pid]["center_cm1"]))
                if abs(float(er["peak_cm1"]) - center) > round_tol:
                    issues.append(
                        {
                            "code": "evidence_peak_freq_mismatch",
                            "peak_id": pid,
                            "table_cm1": er["peak_cm1"],
                            "canonical_cm1": center,
                        }
                    )
        elif er.get("source") == "region_activity" and er.get("peak_cm1") is not None:
            issues.append({"code": "region_fabricated_peak", "row": er})

    evidence = pipeline.get("evidence") or {}
    for bm in evidence.get("band_matches") or []:
        if not bm.get("matched"):
            continue
        pid = bm.get("canonical_peak_id")
        near = bm.get("peaks_near") or []
        if near and not pid:
            issues.append({"code": "band_peak_not_linked", "band_id": bm.get("band_id")})
        if not near and pid:
            issues.append({"code": "region_has_fake_peak", "band_id": bm.get("band_id"), "peak_id": pid})

    keys = [
        (r.get("peak_id"), r.get("band_id"), r.get("assignment_label"))
        for r in (pack.get("evidence_rows") or [])
    ]
    if len(keys) != len(set(keys)):
        issues.append({"code": "duplicate_evidence_rows", "count": len(keys) - len(set(keys))})

    return {
        "ok": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues[:50],
        "stats": pack.get("stats") or {},
    }
