# Code ownership map (SpectraReason)

**Root:** repository root (`PYTHONPATH=.` when running commands)

---

## Data indexing

| Module / path | Responsibility |
|---------------|----------------|
| `nistchemdata.py` (if present) / NIST sibling repo | ZIP-native NIST IR index → SQLite |
| `../NIST/` | Index builds, raw archives |
| `ml/runs/ds_v4_*_nist.npz` | Training matrices from indexed spectra |

---

## Preprocessing

| Module | Responsibility |
|--------|----------------|
| `ftir_foundation.py` / `hidden_peak_workbench.py` | Load CSV/JDX, baseline, SG smooth, scale |
| `ml/ftir_peak_picking.py` | Peak detection for evidence and Kronecker stems |

---

## Evidence extraction

| Module | Responsibility |
|--------|----------------|
| `ml/ftir_evidence.py` | Peak/region/band evidence, artifacts, motifs |
| `ml/ftir_evidence_features.py` | Feature vector layouts (`evidence_v1`, `evidence_v2`) |
| `ml/ftir_interpretable_features.py` | Regional means, band shading windows |
| `ml/ftir_band_library.yaml` | Band definitions |

---

## Ontology

| Module | Responsibility |
|--------|----------------|
| `ml/ftir_ontology.py` | v4 families, categories, label buckets |
| `ml/fg_smarts_library.py` | Specific FG SMARTS (weak labels) |

---

## Rules / guardrails

| Module | Responsibility |
|--------|----------------|
| `ml/ftir_rules.py` | Rule-based FG assignment |
| `configs/rule_presets/` | Conservative / other presets |
| `ml/ftir_guardrails.py` | v3 soft FP control |
| `ml/ftir_ml_refinement.py` | ML advisory fusion, strict guardrails |
| `ml/ftir_atr.py` | ATR-specific siloxane overlap wording |

---

## ML training

| Module | Responsibility |
|--------|----------------|
| `ml/structural_fg_svm.py` | `build-dataset`, `train`, `predict` CLI |
| `ml/runs/struct_fg_*_latest.joblib` | Production model artifacts |
| `ml/runs/experiments/` | Non-production benchmarks (see README there) |

---

## ML refinement (inference)

| Module | Responsibility |
|--------|----------------|
| `ml/structural_fg_svm.py` | `predict_proba_row`, calibration |
| `ml/ftir_ml_refinement.py` | Annotate-mode score caps |

---

## Product report rendering

| Module | Responsibility |
|--------|----------------|
| `reports/structural_fg_svm_kronecker_report.py` | **Main CLI** — batch HTML reports |
| `reports/product_v1_report.py` | product_v1 layout, interpretation, Details |
| `reports/front_facing_report.py` | Front audience HTML (consensus, spectroscopist summary) |
| `reports/report_render.py` | Shared HTML/CSS/Plotly markers |
| `reports/kronecker_pi_layout.py` | Kronecker stem panels |
| `reports/v4_evidence_report.py` | Band maps, justification helpers |
| `reports/static_figure_export.py` | Matplotlib presentation PNGs (spectrum + region guide) |
| `reports/annotation_layout.py` | Ruler row heights, collision-aware peak labels |
| `reports/matlab_visual_theme.py` | MATLAB theme CSS, CSV export, `make_figures.m` |
| `reports/reproducibility_meta.py` | Git/model/package metadata JSON in Technical details |
| `reports/front_consensus.py` | Front consensus table + nitro support checks |
| `ml/canonical_peaks.py` | Canonical peak IDs for tables/labels/static export |
| `ml/ftir_region_ruler.py` | 1450–1650 ruler bands (C=C / amide II / N–O) |

---

## Release / ops

| Path | Responsibility |
|------|----------------|
| `scripts/release_stabilize.py` | Reference snapshots, archive manifest, vulture audit |
| `configs/production/` | Pinned production presets (structure) |
| `reports/reference_snapshots/` | Canonical front/debug/static regression HTML |
| `reports/_archive/` | Archived obsolete report folders |
| `tools/vulture_whitelist.py` | Dead-code audit allowlist hints |

---

## Legacy reports

| Module | Status |
|--------|--------|
| `reports/structural_fg_svm_report.py` | Legacy v7-style SVM HTML |
| `reports/kronecker_spectrum_report.py` | Spectrum + Kronecker only |
| `reports/structural_fg_lean_report.py` | Static lean HTML |
| `reports/structural_fg_*_robustness_report.py` | Robustness batch reports |

---

## Experiments

| Path | Responsibility |
|------|----------------|
| `ml/runs/experiments/v4_classification_improvement/` | Peakcodebook, baseline retrain attempts |
| `ml/runs/experiments/v4_deconv_specific/` | Deconv train + comparison model |
| `ml/runs/experiments/v4_deconv_benchmark/` | Deconv dataset benchmark |
| `ml/runs/experiments/v4_peakcodebook_deconv_specific/` | Combined feature dataset |
| `scripts/run_v4_deconv_training_benchmark.py` | Orchestration (if present) |

---

## Tests

| Path | Focus |
|------|-------|
| `ml/tests/test_product_v1_report.py` | product_v1 contract |
| `ml/tests/test_front_facing_report.py` | Front audience contract |
| `ml/tests/test_annotation_layout.py` | Static export panels, label layout |
| `ml/tests/test_matlab_visual_theme.py` | MATLAB theme + make_figures.m |
| `ml/tests/test_report_features.py` | Feature markers |
| `ml/tests/test_evidence_v2_v4_retrain.py` | Retrain smoke |
| `ml/tests/test_ftir_v3_guardrails.py` | Guardrails |
| `ml/tests/test_v4_classification_improvement.py` | Experiment helpers |

---

## Documentation

| File | Role |
|------|------|
| `CANONICAL_OUTPUTS.md` | Production path map |
| `reports/FIGURES_AND_EXPORT.md` | Static PNG + MATLAB figure workflow |
| `METHODS.md` | Manuscript-style methods |
| `context.md` | Living progress log |
| `docs/COMMANDS.md` | Copy-paste commands |
| `docs/DEPRECATED.md` | This registry |
