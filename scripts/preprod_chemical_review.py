#!/usr/bin/env python3
"""One-off preprod chemical review script."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.spectrum import load_processed_spectrum
from ml.ftir_pipeline import run_evidence_first_pipeline, load_model_bundle

ps = load_processed_spectrum(ROOT / "examples/spectra/Catechol-120-80-9-IR.jdx")
md = {"title": "Dopamine", "cas": "51-61-6", "formula": "C8H11NO2"}
pipe_e = run_evidence_first_pipeline(ps.wn, ps.y, md=md, ml_mode="none")
import joblib

art = joblib.load(ROOT / "models/struct_fg_v7_pubchem_mordred.joblib")
pipe_m = run_evidence_first_pipeline(
    ps.wn, ps.y, md=md, ml_mode="basic", fusion_mode="annotate", basic_model=art
)

def top_rules(pipe, n=8):
    a = pipe["rule_assignments"]["assignments"]
    return sorted(a.items(), key=lambda kv: -kv[1]["score"])[:n]

print("EVIDENCE ONLY top rules:")
for lab, e in top_rules(pipe_e):
    print(f"  {lab}: score={e['score']:.3f} conf={e['confidence']}")

print("\nBASIC ML consensus top:")
for lab, e in pipe_m["consensus"]["top_labels"][:8]:
    print(
        f"  {lab}: rule={e['rule_score']:.3f} ml={e.get('ml_probability_basic')} "
        f"agree={e['agreement_status']}"
    )

a = pipe_e["rule_assignments"]["assignments"]
checks = {
    "phenol_without_aromatic_only": a.get("phenol", {}).get("score", 0),
    "aromatic": a.get("aromatic", {}).get("score", 0),
    "primary_amine": a.get("primary_amine", {}).get("score", 0),
    "amide": a.get("amide", {}).get("score", 0),
    "siloxane": a.get("siloxane", {}).get("score", 0),
    "ether": a.get("ether", {}).get("score", 0),
}
print("\nFailure-mode scores:", json.dumps(checks, indent=2))
