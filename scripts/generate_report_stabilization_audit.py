#!/usr/bin/env python3
"""Generate front/debug HTML + static PNG and write report_stabilization_audit.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PDA = _REPO / "examples" / "spectra" / "Catechol-120-80-9-IR.jdx"
OUT = _REPO / "reports" / "report_stabilization_proof"


def _run(
    *,
    audience: str,
    out_html: Path,
    static: bool,
) -> dict:
    from reports.structural_fg_svm_kronecker_report import run_batch

    out_html.parent.mkdir(parents=True, exist_ok=True)
    run_batch(
        input_paths=[PDA],
        model_path=None,
        basic_model_path=_REPO / "ml/runs/struct_fg_family_v4_ontology_latest.joblib",
        subtle_model_path=_REPO / "ml/runs/struct_fg_specific_v4_ontology_latest.joblib",
        out_path=out_html,
        page_title=f"Stabilization proof ({audience})",
        subtitle="canonical peaks + matplotlib static export",
        max_peaks=80,
        hover_top_fg=5,
        ml_mode="both",
        include_evidence=True,
        include_ml=True,
        report_style="product_v1",
        report_audience=audience,
        visual_theme="matlab",
        show_region_ruler=True,
        peak_sensitivity="sensitive",
        peak_label_preset="sensitive",
        label_all_above_height=0.05,
        export_static_figures=static,
        static_format="png",
        static_dpi=200,
        static_peak_label_policy="key",
        max_static_peak_labels=12,
        fingerprint_cluster_distance=0,
    )
    consistency_path = out_html.parent / "report_consistency_audit.json"
    consistency = {}
    if consistency_path.is_file():
        consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    pack = {}
    return {"consistency": consistency, "consistency_path": str(consistency_path.resolve())}


def main() -> int:
    if not PDA.is_file():
        print(f"Missing input: {PDA}", file=sys.stderr)
        return 1
    front_html = OUT / "front" / "REPORT.html"
    debug_html = OUT / "debug" / "REPORT.html"
    front = _run(audience="front", out_html=front_html, static=True)
    debug = _run(audience="debug", out_html=debug_html, static=False)

    lines = [
        "# Report stabilization audit\n",
        f"Input: `{PDA.resolve()}`\n\n",
        "## Outputs\n",
        f"- Front HTML: `{front_html.resolve()}`\n",
        f"- Debug HTML: `{debug_html.resolve()}`\n",
        f"- Front PNG: `{OUT.resolve()}/front/presentation/figures/`\n",
        f"- Consistency (front): `{front['consistency_path']}`\n",
        f"- Consistency (debug): `{debug['consistency_path']}`\n\n",
        "## Front consistency\n",
        f"```json\n{json.dumps(front['consistency'], indent=2)}\n```\n",
    ]
    audit_path = _REPO / "reports" / "report_stabilization_audit.md"
    audit_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {audit_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
