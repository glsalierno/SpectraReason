"""Write ``reports/external_dataset_expansion_audit.md`` from registry + indexes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ml.dataset_quality import audit_sqlite_index
from ml.external.ingest_common import load_source_registry
from ml.external.spectrum_index import count_spectra


def generate_expansion_audit(
    *,
    registry_path: Path | None = None,
    experimental_dir: Path | None = None,
    out_path: Path | None = None,
    coverage_summary: dict | None = None,
) -> Path:
    registry_path = registry_path or Path("data/external_sources/source_registry.json")
    experimental_dir = experimental_dir or Path("data/experimental")
    out_path = out_path or Path("reports/external_dataset_expansion_audit.md")

    sources = load_source_registry(registry_path)
    lines = [
        "# External dataset expansion audit",
        "",
        f"Generated from `{registry_path}` and `{experimental_dir}`.",
        "",
        "## Sources in registry",
        "",
        "| source_id | status | license | est. count | output |",
        "|-----------|--------|---------|------------|--------|",
    ]
    for s in sources:
        if s.get("source_id") == "nistchemdata":
            continue
        lines.append(
            f"| {s.get('source_id')} | {s.get('ingestion_status')} | "
            f"{str(s.get('license', ''))[:40]}… | {s.get('spectrum_count_estimate')} | "
            f"`{s.get('processed_output', '')}` |"
        )

    lines.extend(["", "## Built experimental indexes", ""])
    tag_totals: Counter[str] = Counter()
    atr_n = 0
    total = 0
    for db in sorted(experimental_dir.glob("*.sqlite")):
        n = count_spectra(db)
        total += n
        rep = audit_sqlite_index(db)
        tag_totals.update(rep.tag_counts)
        atr_n += rep.tag_counts.get("atr", 0)
        lines.append(f"- `{db.name}`: **{n}** spectra")
        if rep.flag_counts:
            lines.append(f"  - QA flags: {dict(rep.flag_counts)}")

    lines.extend(
        [
            "",
            "## Coverage (tags across experimental indexes)",
            "",
            f"- **Total spectra (experimental SQLite):** {total}",
            f"- **ATR-tagged:** {atr_n}",
            f"- **heteroaromatic:** {tag_totals.get('heteroaromatic', 0)}",
            f"- **nitro:** {tag_totals.get('nitro', 0)}",
            f"- **n_oxide:** {tag_totals.get('n_oxide', 0)}",
            f"- **polymer:** {tag_totals.get('polymer', 0)}",
            f"- **siloxane:** {tag_totals.get('siloxane', 0)}",
            f"- **phenol:** {tag_totals.get('phenol', 0)}",
            f"- **amide:** {tag_totals.get('amide', 0)}",
            "",
            "## Legal / license notes",
            "",
            "- SDBS: research citation required; do not redistribute bulk exports.",
            "- Zenodo: verify DOI license before sharing merged SQLite.",
            "- Proprietary libraries: local use only via user plugin; never commit raw files.",
            "",
            "## Targeted confounder gaps",
            "",
        ]
    )

    if coverage_summary is None:
        try:
            from ml.external.summarize_confounder_coverage import summarize_confounder_coverage

            coverage_summary = summarize_confounder_coverage(experimental_dir=experimental_dir)
        except Exception:
            coverage_summary = None

    if coverage_summary:
        gaps = coverage_summary.get("coverage_gaps") or []
        lines.append(f"- **Classes below minimum:** {len(gaps)}")
        lines.append(f"- **Nitro TP / HN:** {coverage_summary.get('by_problem_true_positive_count', {}).get('nitro', 0)} / "
                     f"{coverage_summary.get('by_problem_hard_negative_count', {}).get('nitro', 0)}")
        lines.append(f"- **Amide TP / HN:** {coverage_summary.get('by_problem_true_positive_count', {}).get('amide', 0)} / "
                     f"{coverage_summary.get('by_problem_hard_negative_count', {}).get('amide', 0)}")
        lines.append(f"- **Siloxane TP / HN:** {coverage_summary.get('by_problem_true_positive_count', {}).get('siloxane', 0)} / "
                     f"{coverage_summary.get('by_problem_hard_negative_count', {}).get('siloxane', 0)}")
        lines.append("")
        lines.append("Top missing classes:")
        lines.append("")
        for g in gaps[:12]:
            lines.append(
                f"- `{g['class_id']}` ({g['role']}): need **{g['missing']}** more "
                f"(have {g['have']}/{g['minimum']}) — source `{g['preferred_source']}`"
            )
        lines.append("")
        lines.append("Full report: `reports/confounder_coverage_summary.md`")
        lines.append("")
    else:
        lines.append("_Run `python -m ml.external summarize-confounder-coverage` for gap table._")
        lines.append("")

    lines.extend(
        [
            "## Next recommended sources",
            "",
            "1. SDBS: nitro positives + N-oxide + nitroso + heteroaromatic (see TARGET_EXTERNAL_EXPANSION.md).",
            "2. SDBS: amide standards + enamine + imide + nicotinamide.",
            "3. Zenodo: PDMS/silicone ATR + nylon/epoxy C–O polymer negatives.",
            "4. Lab ATR: powder polymers with manifest.csv.",
            "",
            "## Ingestion failures",
            "",
            "See `reports/dataset_ingestion_audit.md` after each ingest run.",
            "",
            "## Production status",
            "",
            "Production `*_latest.joblib` and NIST NPZ datasets are **unchanged** until promotion checklist completes.",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
