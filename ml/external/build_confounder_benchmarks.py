"""Build confounder benchmark subsets from an experimental SQLite index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

BENCHMARK_DEFS: dict[str, dict[str, Any]] = {
    "nitro_vs_n_oxide": {
        "description": "Nitro vs N-oxide: overlapping NO stretches and aromatic context.",
        "required_tags_any": ["nitro", "n_oxide"],
        "expected_behavior": "Rules/ML should not collapse nitro and N-oxide; expect partial band overlap near 1300–1600 cm⁻¹.",
        "known_ambiguities": ["Aromatic nitro compounds vs heterocyclic N-oxides"],
    },
    "phenol_vs_alcohol": {
        "description": "Phenolic O–H vs aliphatic alcohol O–H broad bands.",
        "required_tags_any": ["phenol"],
        "keyword_any": ["alcohol", "phenol", "hydroxy"],
        "expected_behavior": "Phenol shows broader 3200–3600 cm⁻¹ envelope with aromatic fingerprint; alcohol narrower.",
        "known_ambiguities": ["Enols", "carboxylic acids with O–H"],
    },
    "siloxane_vs_C_O": {
        "description": "Si–O–Si vs C–O stretches in fingerprint.",
        "required_tags_any": ["siloxane"],
        "expected_behavior": "Strong Si–O–Si ~1000–1100 cm⁻¹; distinguish from ether C–O.",
        "known_ambiguities": ["Silicate fillers", "ester C–O near 1100 cm⁻¹"],
    },
    "amide_vs_enamine": {
        "description": "Amide C=O/N–H vs enamine C=C/N.",
        "required_tags_any": ["amide", "enamine"],
        "expected_behavior": "Amide shows 1650–1680 + 1500–1550; enamine lacks strong amide carbonyl pair.",
        "known_ambiguities": ["Imides", "lactams"],
    },
    "pyrrole_vs_secondary_amine": {
        "description": "Pyrrole N–H vs secondary amine N–H.",
        "keyword_any": ["pyrrol", "indol", "amine"],
        "expected_behavior": "Heteroaromatic ring modes 1400–1600; amine broader N–H without full aromatic ring set.",
        "known_ambiguities": ["Indoles", "anilines"],
    },
    "ATR_artifacts": {
        "description": "ATR baseline distortion and contact effects.",
        "required_tags_any": ["atr"],
        "optional_tags_any": ["baseline_drift", "moisture_like"],
        "expected_behavior": "Evidence pipeline should tolerate baseline curvature; guardrails flag moisture-like bands.",
        "known_ambiguities": ["Strong carbonyl ATR shifts"],
    },
    "crowded_fingerprint": {
        "description": "Dense fingerprint overlap (polymers, heteroaromatics).",
        "required_tags_any": ["polymer", "heteroaromatic"],
        "expected_behavior": "Peak picking remains stable; deconv optional.",
        "known_ambiguities": ["Any high-DoF organic solid"],
    },
}


def _match_benchmark(md: dict[str, Any], spec: dict[str, Any]) -> bool:
    tags = {str(t).lower() for t in (md.get("dataset_tags") or [])}
    blob = " ".join(str(md.get(k) or "") for k in ("title", "name", "formula")).lower()

    req_any = spec.get("required_tags_any")
    if req_any and not (tags & {t.lower() for t in req_any}):
        kw = spec.get("keyword_any")
        if not kw or not any(k in blob for k in kw):
            return False

    kw_any = spec.get("keyword_any")
    if kw_any and not spec.get("required_tags_any"):
        if not any(k in blob for k in kw_any):
            return False

    opt = spec.get("optional_tags_any")
    if opt and tags and not (tags & {t.lower() for t in opt}):
        return False
    return True


def build_confounder_benchmarks(
    sqlite_index: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or Path("data/benchmark_sets")
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(sqlite_index))
    rows = conn.execute("SELECT reference_id, metadata_json, source_path FROM spectra").fetchall()
    conn.close()

    summary: dict[str, Any] = {"benchmarks": {}, "sqlite": str(sqlite_index.resolve())}

    for name, spec in BENCHMARK_DEFS.items():
        members: list[dict[str, Any]] = []
        for rid, mj, sp in rows:
            md = json.loads(mj) if mj else {}
            if _match_benchmark(md, spec):
                members.append(
                    {
                        "reference_id": rid,
                        "source_path": sp,
                        "title": md.get("title"),
                        "tags": md.get("dataset_tags"),
                        "source_id": md.get("source_id"),
                    }
                )
        bench = {
            "name": name,
            "description": spec["description"],
            "expected_behavior": spec["expected_behavior"],
            "known_ambiguities": spec.get("known_ambiguities", []),
            "spectra": members,
            "count": len(members),
        }
        (out_dir / f"{name}.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
        summary["benchmarks"][name] = {"count": len(members)}

    (out_dir / "index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
