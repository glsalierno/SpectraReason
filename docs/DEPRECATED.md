# Deprecated artifacts and replacements

**Last updated:** 2026-05-16 (production stabilization)

Archived report folders (moved with manifest) live under `reports/_archive/2026-05/` — see `reports/archive_manifest_*.csv` and `reports/release_stabilization_audit.md`.

Items listed here are **kept in the tree** for reproducibility but should not be used for new work unless you are explicitly reproducing history.

---

## Report entry scripts

| Deprecated | Replacement |
|------------|-------------|
| `reports/structural_fg_svm_report.py` | `reports/structural_fg_svm_kronecker_report.py` with `--report-style product_v1` |
| `reports/kronecker_spectrum_report.py` | Same Kronecker report (spectrum-only subset) |
| `reports/structural_fg_lean_report.py` | `structural_fg_svm_kronecker_report.py` (full product) or evidence-only with `--ml-mode none` |
| `reports/structural_fg_svm_robustness_report.py` | Kronecker report + pytest robustness tests |
| `reports/structural_fg_robustness_report.py` | Same |

---

## Feature sets

| Deprecated | Replacement |
|------------|-------------|
| `spectral+evidence` / `evidence_v1` (~64-D evidence block) | `spectral+evidence_v2` (434-D) |
| v7 303-D spectral+RDKit+Mordred | v4 family + specific (`spectral+evidence_v2`) |
| `spectral+evidence_v2+peakcodebook` (734-D) experiment | Production 434-D until benchmark promotes |
| `spectral+evidence_v2+deconv` (514-D) experiment | Production 434-D; use deconv joblib only for comparison reports |

---

## Model artifacts

| Deprecated | Replacement |
|------------|-------------|
| `ml/runs/struct_fg_basic_v4_ontology_latest.joblib` | `struct_fg_family_v4_ontology_latest.joblib` + `struct_fg_specific_v4_ontology_latest.joblib` |
| `models/struct_fg_v7_pubchem_mordred.joblib` | v4 family + specific (ontology-aware) |
| Timestamped `struct_fg_*_20260516_*.joblib` in `ml/runs/` | `*_latest.joblib` symlinks/copies |
| Experiment deconv model (not promoted) | `struct_fg_specific_v4_ontology_latest.joblib` for production |

---

## Report output folders

| Deprecated (archived 2026-05) | Replacement |
|--------------------------------|-------------|
| `reports/preprod_*`, `reports/smoke_*`, `reports/model_training_*` | `reports/_archive/2026-05/` |
| `reports/ftir_powder_v4_svm*` | `reports/ftir_powder_v4_evidence_first/` |
| `reports/v4_ontology_20260516_*` | `product_v1_demo/` or regenerate with latest models |
| `reports/examples_spectra_*_v4_kronecker` | `examples/_evidence_pipeline_report/` |

See `reports/_archive/2026-05/MANIFEST_2026-05-17.md` for the full move list.

---

## Report styles

| Deprecated | Replacement |
|------------|-------------|
| `--report-style legacy` (default table-heavy) | `--report-style product_v1` (default in CLI) |
| Pre–product_v1 HTML without band shading / local hover | Regenerate with current `structural_fg_svm_kronecker_report.py` |

---

## Datasets (do not delete)

Failed/incomplete NPZ builds are documented in `ml/runs/experiments/README.md`. Only delete NPZ if explicitly marked failed/incomplete there.
