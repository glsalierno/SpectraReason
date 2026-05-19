#!/usr/bin/env python3
"""Generate reports/v4_smarts_label_audit.md from SMARTS library and optional dataset NPZ."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    from ml.fg_smarts_library import SMARTS_ENTRIES
    from ml.ftir_ontology import TRAINABLE_SPECIFIC_V4

    smarts_by_label = {e.label: e.smarts for e in SMARTS_ENTRIES}

    audit_labels = [
        "ester",
        "amide",
        "phenol",
        "aryl_ether",
        "siloxane",
        "heteroaromatic",
        "pyrrole_like_NH",
        "cyclic_amine",
        "urethane",
        "carbonate",
    ]

    lines = [
        "# v4 SMARTS weak-label audit",
        "",
        "Generated from `ml/fg_smarts_library.py` and training NPZ counts when available.",
        "",
        "| Label | SMARTS | Positives (NPZ) | ML recommendation | Notes |",
        "|-------|--------|-----------------|-------------------|-------|",
    ]

    npz_path = ROOT / "ml" / "runs" / "ds_v4_specific_spectral_evidence_v2_nist.npz"
    meta_path = ROOT / "ml" / "runs" / "ds_v4_specific_spectral_evidence_v2_nist.meta.json"
    counts: dict[str, int] = {}
    if npz_path.is_file() and meta_path.is_file():
        import numpy as np

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz_path, allow_pickle=True)
        Y = data["Y"]
        for j, lab in enumerate(meta.get("label_names") or []):
            counts[str(lab)] = int(Y[:, j].sum())

    for lab in audit_labels + [x for x in TRAINABLE_SPECIFIC_V4 if x not in audit_labels]:
        sm = smarts_by_label.get(lab, "")
        sm_disp = f"`{sm[:80]}...`" if len(str(sm)) > 80 else f"`{sm}`"
        pos = counts.get(lab, "—")
        if lab in ("urethane", "carbonate", "cyclic_amine", "pyrrole_like_NH"):
            rec = "specialist-only or rules-primary"
        elif lab in ("ester", "amide", "phenol", "siloxane", "heteroaromatic"):
            rec = "ML + strict guardrails"
        else:
            rec = "ML"
        note = "high FP risk — audit hard negatives" if lab in ("phenol", "ester", "siloxane") else ""
        lines.append(f"| {lab} | {sm_disp} | {pos} | {rec} | {note} |")

    lines += [
        "",
        "## Hard-negative groups (training diagnostics)",
        "",
        "See `ml/training_diagnostics.py` `HARD_NEGATIVE_PAIRS`.",
        "",
    ]
    out = ROOT / "reports" / "v4_smarts_label_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
