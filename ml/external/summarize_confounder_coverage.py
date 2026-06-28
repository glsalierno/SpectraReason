"""Summarize targeted confounder coverage across experimental SQLite indexes."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.external.confounder_targets import (
    ALL_TARGET_CLASSES,
    TARGET_BY_ID,
    classify_spectrum,
    match_target_class,
)
from ml.external.ingest_common import keyword_tags_from_metadata
from ml.external.spectrum_index import count_spectra


def _load_spectra(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT reference_id, metadata_json, source_path FROM spectra"
    ).fetchall()
    conn.close()
    out: list[dict[str, Any]] = []
    for rid, mj, sp in rows:
        md = json.loads(mj) if mj else {}
        md["reference_id"] = rid
        md["source_path"] = sp
        if not md.get("dataset_tags"):
            md["dataset_tags"] = keyword_tags_from_metadata(md)
        out.append(md)
    return out


def _acquisition_bucket(md: dict[str, Any]) -> str:
    mode = str(md.get("acquisition_mode") or "").upper()
    tags = {str(t).lower() for t in (md.get("dataset_tags") or [])}
    if mode == "ATR" or "atr" in tags:
        return "ATR"
    if mode == "transmission" or "transmission" in tags:
        return "transmission"
    if mode == "gas_phase":
        return "gas_phase"
    return "unknown"


def _fg_from_tags(md: dict[str, Any]) -> list[str]:
    fg_tags = (
        "nitro",
        "n_oxide",
        "nitroso",
        "amide",
        "enamine",
        "siloxane",
        "phenol",
        "heteroaromatic",
        "polymer",
    )
    tags = {str(t).lower() for t in (md.get("dataset_tags") or [])}
    return sorted(tags & set(fg_tags))


def summarize_confounder_coverage(
    *,
    sqlite_paths: list[Path] | None = None,
    experimental_dir: Path | None = None,
) -> dict[str, Any]:
    experimental_dir = experimental_dir or Path("data/experimental")
    if sqlite_paths is None:
        sqlite_paths = sorted(experimental_dir.glob("*.sqlite"))

    all_spectra: list[dict[str, Any]] = []
    per_db: dict[str, int] = {}
    for db in sqlite_paths:
        if not db.is_file():
            continue
        specs = _load_spectra(db)
        per_db[db.name] = len(specs)
        all_spectra.extend(specs)

    by_source: Counter[str] = Counter()
    by_tag: Counter[str] = Counter()
    by_acq: Counter[str] = Counter()
    by_fg: Counter[str] = Counter()
    by_target_class: Counter[str] = Counter()
    by_hn_class: Counter[str] = Counter()
    by_problem_tp: Counter[str] = Counter()
    by_problem_hn: Counter[str] = Counter()

    class_members: dict[str, list[dict[str, str]]] = defaultdict(list)

    for md in all_spectra:
        sid = str(md.get("source_id") or md.get("source") or "unknown")
        by_source[sid] += 1
        for t in md.get("dataset_tags") or []:
            by_tag[str(t).lower()] += 1
        by_acq[_acquisition_bucket(md)] += 1
        for fg in _fg_from_tags(md):
            by_fg[fg] += 1

        matched = classify_spectrum(md)
        for cid in matched:
            by_target_class[cid] += 1
            tc = TARGET_BY_ID[cid]
            class_members[cid].append(
                {
                    "reference_id": str(md.get("reference_id", "")),
                    "title": str(md.get("title") or md.get("name") or ""),
                    "source_id": sid,
                }
            )
            if tc.role == "hard_negative":
                by_hn_class[cid] += 1
                by_problem_hn[tc.problem] += 1
            elif tc.role == "true_positive":
                by_problem_tp[tc.problem] += 1

    gaps: list[dict[str, Any]] = []
    for tc in ALL_TARGET_CLASSES:
        have = by_target_class[tc.class_id]
        need = tc.minimum_count
        if have < need:
            gaps.append(
                {
                    "class_id": tc.class_id,
                    "role": tc.role,
                    "problem": tc.problem,
                    "have": have,
                    "minimum": need,
                    "missing": need - have,
                    "preferred_source": tc.preferred_source,
                    "example_compounds": list(tc.example_compounds)[:3],
                }
            )

    gaps.sort(key=lambda g: (-g["missing"], g["problem"], g["class_id"]))

    return {
        "total_spectra": len(all_spectra),
        "indexes": per_db,
        "by_source": dict(by_source),
        "by_tag": dict(by_tag.most_common()),
        "by_acquisition_mode": dict(by_acq),
        "by_functional_group_tag": dict(by_fg.most_common()),
        "by_target_class": dict(by_target_class),
        "by_hard_negative_class": dict(by_hn_class),
        "by_problem_true_positive_count": dict(by_problem_tp),
        "by_problem_hard_negative_count": dict(by_problem_hn),
        "coverage_gaps": gaps,
        "class_members_sample": {
            k: v[:5] for k, v in class_members.items() if v
        },
    }


def write_coverage_markdown(summary: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Confounder coverage summary",
        "",
        f"- **Total spectra:** {summary['total_spectra']}",
        f"- **Indexes:** {summary.get('indexes', {})}",
        "",
        "## By source",
        "",
    ]
    for src, n in sorted((summary.get("by_source") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"- `{src}`: {n}")

    lines.extend(["", "## By acquisition mode", ""])
    for mode, n in (summary.get("by_acquisition_mode") or {}).items():
        lines.append(f"- `{mode}`: {n}")

    lines.extend(["", "## By functional-group tag", ""])
    for fg, n in (summary.get("by_functional_group_tag") or {}).items():
        lines.append(f"- `{fg}`: {n}")

    lines.extend(["", "## Target class counts (true positive / hard negative)", ""])
    for tc in ALL_TARGET_CLASSES:
        n = (summary.get("by_target_class") or {}).get(tc.class_id, 0)
        mark = "✓" if n >= tc.minimum_count else "✗"
        lines.append(
            f"- {mark} `{tc.class_id}` ({tc.role}, {tc.problem}): **{n}** / min {tc.minimum_count}"
        )

    lines.extend(["", "## Coverage gaps (missing spectra)", ""])
    gaps = summary.get("coverage_gaps") or []
    if not gaps:
        lines.append("- All target classes meet minimum counts.")
    else:
        lines.append("| class | problem | role | have | min | missing | source |")
        lines.append("|-------|---------|------|------|-----|---------|--------|")
        for g in gaps[:25]:
            lines.append(
                f"| {g['class_id']} | {g['problem']} | {g['role']} | {g['have']} | "
                f"{g['minimum']} | {g['missing']} | {g['preferred_source']} |"
            )
        if len(gaps) > 25:
            lines.append(f"\n_…and {len(gaps) - 25} more gaps._")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifests_with_coverage(
    summary: dict[str, Any],
    manifest_dir: Path | None = None,
) -> dict[str, Path]:
    """Merge coverage stats into existing manifest JSON (preserves full class lists)."""
    manifest_dir = manifest_dir or Path("data/benchmark_sets")
    from ml.external.confounder_targets import manifest_spec

    written: dict[str, Path] = {}
    manifest_names = {
        c.benchmark_manifest for c in ALL_TARGET_CLASSES if c.benchmark_manifest
    }
    for name in sorted(manifest_names):
        path = manifest_dir / name
        if path.is_file():
            spec = json.loads(path.read_text(encoding="utf-8"))
        else:
            spec = manifest_spec(name)
        if not spec:
            continue
        for cls in spec.get("classes", []):
            cid = cls["class_id"]
            ingested = (summary.get("by_target_class") or {}).get(cid, 0)
            cls["ingested_count"] = ingested
            cls["gap"] = max(0, int(cls.get("minimum_count", 0)) - ingested)
        spec["coverage"] = {
            c["class_id"]: {
                "ingested": c.get("ingested_count", 0),
                "minimum": c.get("minimum_count", 0),
                "gap": c.get("gap", 0),
            }
            for c in spec.get("classes", [])
        }
        members = summary.get("class_members_sample") or {}
        for cls in spec.get("classes", []):
            cls["sample_spectra"] = members.get(cls["class_id"], [])
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        written[name] = path
    return written


def run_summarize(
    *,
    sqlite_index: Path | None = None,
    experimental_dir: Path | None = None,
    out_json: Path | None = None,
    out_md: Path | None = None,
    update_manifests: bool = True,
    update_expansion_audit: bool = True,
) -> dict[str, Any]:
    paths: list[Path] | None = None
    if sqlite_index is not None:
        paths = [sqlite_index]
    summary = summarize_confounder_coverage(
        sqlite_paths=paths,
        experimental_dir=experimental_dir,
    )

    out_json = out_json or Path("reports/confounder_coverage_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_md = out_md or Path("reports/confounder_coverage_summary.md")
    write_coverage_markdown(summary, out_md)

    if update_manifests:
        update_manifests_with_coverage(summary)

    if update_expansion_audit:
        from ml.external.generate_expansion_audit import generate_expansion_audit

        generate_expansion_audit(coverage_summary=summary)

    return summary
