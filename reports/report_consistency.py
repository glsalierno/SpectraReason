"""Write report consistency audits to the run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.canonical_peaks import validate_report_peak_consistency


def write_report_consistency_audit(
    out_dir: Path,
    pipeline: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = validate_report_peak_consistency(pipeline)
    path = out_dir / "report_consistency_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if strict and not result.get("ok"):
        raise AssertionError(
            f"Report consistency failed ({result.get('issue_count')} issues). See {path}"
        )
    return result
