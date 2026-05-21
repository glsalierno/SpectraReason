# ML artifacts (models + training matrices)

SpectraReason ships **pre-trained SVMs** and **homogenized NIST training matrices** so
collaborators can use ML immediately after clone.

## What is included in git

| Bundle | Path | Contents |
|--------|------|----------|
| **v4 production** | `data/training/bundled/v4_production/` | Family + specific `.joblib`, `.npz`, `.meta.json`, PubChem cache |
| **v7 legacy (Mordred)** | `data/training/bundled/v7_mordred/` | 303-D NPZ + v7 joblib (14 spectral + 32 RDKit + 256 Mordred) |
| **PubChem** | `data/training/bundled/pubchem_structure_cache_v7.json` | Structure cache (~44 MB) |

**Not included:** NIST SQLite index (`data/external/nist_index.sqlite`) — build locally.

## v4 feature layout (production)

- **434-D** `spectral+evidence_v2` per spectrum
- **Labels:** SMARTS weak supervision from PubChem structures (not Mordred columns in X)
- **~15.6k** training rows with `require_structure`

See `METHODS.md` § Feature representation for v4 SVM.

## v7 feature layout (legacy)

- **303-D** `spectral` + RDKit + Mordred + `has_structure`
- **~18k** rows in `struct_fg_v7_pubchem_mordred.npz`

## One-time setup

```bash
./scripts/setup_bundled_artifacts.sh   # copies into ml/runs/
export PYTHONPATH="$(pwd)"
```

Then reports use default paths:

- `ml/runs/struct_fg_family_v4_ontology_latest.joblib`
- `ml/runs/struct_fg_specific_v4_ontology_latest.joblib`

## Retrain without NIST rebuild

```bash
python -m ml.structural_fg_svm train \
  --dataset-prefix data/training/bundled/v4_production/ds_v4_family_spectral_evidence_v2_nist \
  --ontology v4 --model-kind family --out ml/runs
```

## ML role in the product

SVM scores are **advisory** (`fusion-mode annotate`). Rules + guardrails remain primary.
See `README.md` and `docs/PRODUCTION_DEFAULTS.md`.
