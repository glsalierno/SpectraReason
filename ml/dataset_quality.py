"""Dataset QA for ingested / merged spectral indexes."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class QAIssue:
    reference_id: str
    flag: str
    detail: str = ""


@dataclass
class QAReport:
    spectrum_count: int = 0
    issues: list[QAIssue] = field(default_factory=list)
    flag_counts: Counter[str] = field(default_factory=Counter)
    tag_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    duplicate_spectra: int = 0
    duplicate_identifiers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectrum_count": self.spectrum_count,
            "flag_counts": dict(self.flag_counts),
            "tag_counts": dict(self.tag_counts),
            "source_counts": dict(self.source_counts),
            "duplicate_spectra": self.duplicate_spectra,
            "duplicate_identifiers": self.duplicate_identifiers,
            "issue_sample": [
                {"reference_id": i.reference_id, "flag": i.flag, "detail": i.detail}
                for i in self.issues[:50]
            ],
        }


def _spectrum_hash(wn: np.ndarray, y: np.ndarray) -> str:
    wn_r = np.round(wn, 2)
    y_r = np.round(y, 4)
    return f"{wn_r.tobytes()!r}|{y_r.tobytes()!r}"


def audit_sqlite_index(db_path: Path, *, max_issues: int = 5000) -> QAReport:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT reference_id, metadata_json, wn_json, y_json FROM spectra")
    report = QAReport()
    seen_hash: dict[str, str] = {}
    seen_oid: dict[str, str] = defaultdict(str)

    for rid, mj, wj, yj in cur:
        report.spectrum_count += 1
        try:
            md = json.loads(mj) if mj else {}
            wn = np.asarray(json.loads(wj), dtype=float)
            y = np.asarray(json.loads(yj), dtype=float)
        except Exception:
            report.issues.append(QAIssue(rid, "parse_error"))
            report.flag_counts["parse_error"] += 1
            continue

        sid = str(md.get("source_id") or md.get("source") or "unknown")
        report.source_counts[sid] += 1
        for t in md.get("dataset_tags") or []:
            report.tag_counts[str(t)] += 1

        oid = str(md.get("original_identifier") or "")
        if oid:
            if oid in seen_oid and seen_oid[oid] != rid:
                report.duplicate_identifiers += 1
                _add(report, rid, "duplicate_identifier", oid, max_issues)
            else:
                seen_oid[oid] = rid

        if wn.size < 32:
            _add(report, rid, "too_few_points", str(wn.size), max_issues)
        span = float(np.nanmax(wn) - np.nanmin(wn)) if wn.size else 0.0
        if span < 200:
            _add(report, rid, "narrow_span", f"{span:.1f}", max_issues)
        if not np.all(np.isfinite(wn)) or not np.all(np.isfinite(y)):
            _add(report, rid, "non_finite", "", max_issues)
        yr = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
        if yr < 1e-6:
            _add(report, rid, "flat_spectrum", "", max_issues)
        if float(np.nanmax(y)) > 1.05 or float(np.nanmin(y)) < -0.05:
            _add(report, rid, "absorbance_out_of_range", f"min={np.nanmin(y):.3f} max={np.nanmax(y):.3f}", max_issues)
        if float(np.nanmax(y)) > 3.0:
            _add(report, rid, "extreme_absorbance", "", max_issues)
        dy = np.diff(y)
        if dy.size and float(np.nanstd(dy)) > 0.15:
            _add(report, rid, "extreme_noise", f"std={np.nanstd(dy):.3f}", max_issues)

        h = _spectrum_hash(wn, y)
        if h in seen_hash and seen_hash[h] != rid:
            report.duplicate_spectra += 1
            _add(report, rid, "duplicate_spectrum", seen_hash[h], max_issues)
        else:
            seen_hash[h] = rid

        xu = str(md.get("xunits") or "").upper()
        if xu and "CM" not in xu and "MICROM" not in xu and "1/CM" not in xu.replace(" ", ""):
            _add(report, rid, "suspicious_xunits", xu, max_issues)

        if not md.get("source_id") and not md.get("source_license"):
            _add(report, rid, "missing_provenance", "", max_issues)

    conn.close()
    return report


def _add(report: QAReport, rid: str, flag: str, detail: str, max_issues: int) -> None:
    report.flag_counts[flag] += 1
    if len(report.issues) < max_issues:
        report.issues.append(QAIssue(rid, flag, detail))


def write_audit_markdown(report: QAReport, out_path: Path, *, title: str = "Dataset ingestion audit") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- **Spectra audited:** {report.spectrum_count}",
        f"- **Duplicate spectra (hash):** {report.duplicate_spectra}",
        f"- **Duplicate original identifiers:** {report.duplicate_identifiers}",
        "",
        "## Flag counts",
        "",
    ]
    for flag, n in report.flag_counts.most_common():
        lines.append(f"- `{flag}`: {n}")
    lines.extend(["", "## Source counts", ""])
    for src, n in report.source_counts.most_common():
        lines.append(f"- `{src}`: {n}")
    lines.extend(["", "## Tag coverage", ""])
    for tag, n in report.tag_counts.most_common(30):
        lines.append(f"- `{tag}`: {n}")
    if report.issues:
        lines.extend(["", "## Sample issues (first 20)", ""])
        for iss in report.issues[:20]:
            lines.append(f"- `{iss.reference_id}` — **{iss.flag}** {iss.detail}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_index_cli(db_path: Path, out_md: Path | None = None) -> QAReport:
    report = audit_sqlite_index(db_path)
    out_md = out_md or Path("reports/dataset_ingestion_audit.md")
    write_audit_markdown(report, out_md)
    return report
