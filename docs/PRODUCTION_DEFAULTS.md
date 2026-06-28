# Production defaults (FTIR_SVM_v5)

Frozen **product** settings for reproducible front-facing reports. Override only deliberately for experiments.

| Setting | Production value | Notes |
|---------|------------------|-------|
| **ontology** | `v4` | SMARTS + band library v4 |
| **guardrails** | `v3` | `apply_v3_guardrails` + nitro/N-oxide/amide overlap |
| **fusion** | `annotate` | ML does not override rules; annotates consensus |
| **ml_guardrails** | `strict` | ML heads capped when rules disagree |
| **rules_preset** | `conservative` | Default in `ml/ftir_rules.py` evidence thresholds |
| **report_style** | `product_v1` | Interpretation panel + band map in technical details |
| **report_audience** | `front` | Spectroscopist summary, key evidence, collapsed metadata |
| **visual_theme** | `matlab` | White background, MATLAB-blue trace (deliverables) |
| **peak_sensitivity** | `sensitive` | More peaks detected for crowded fingerprints |
| **show_weak_peaks** | on | Plot faint peaks (not all labeled) |
| **region_ruler** | on (default for `product_v1`) | 1450–1650 label: C=C / amide II / N–O |
| **front_max_peak_labels** | `10` | Cap interactive peak labels in front mode |
| **peak_label thresholds (front)** | height 0.15, prominence 0.05 | See `resolve_peak_label_thresholds` |
| **peak_label thresholds (debug)** | height 0.05, prominence 0.025 | Full diagnostic labeling |
| **feature_set (training)** | `spectral+evidence_v2` | 434-D family/specific models |
| **ml_mode** | `both` | Family + specific joblibs |
| **shade** | band shading on, `label_band_shading` optional | Upper-mid tiered shading when enabled |

## Front vs debug

| | **Front** (`--report-audience front`) | **Debug** (`--report-audience debug` or `--report-density audit`) |
|---|--------------------------------------|---------------------------------------------------------------------|
| Summary | Spectroscopist prose + consensus per spectrum | Full interpretation panel |
| Tables | Key evidence + front consensus table | Generic summary + full assignment tables |
| Peak labels | ~10 prioritized | More labels; peak-picking summary in details |
| Metadata | Hidden unless `--show-metadata` | Shown in technical details |
| Raw ontology spam | Local motifs / NO₂ regions suppressed in consensus | Full diagnostics including local motifs |
| Reproducibility block | Collapsed under Technical details | Same JSON block, expanded details |

## Production models (do not relocate without updating all references)

- `c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5\ml\runs\struct_fg_family_v4_ontology_latest.joblib`
- `c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5\ml\runs\struct_fg_specific_v4_ontology_latest.joblib`

## Config paths (structure only)

- `configs/production/` — pinned YAML/JSON presets (future)
- `configs/experiments/` — non-production sweeps
- `ml/runs/production/` — symlink/copy target for promoted joblibs (optional)
- `ml/runs/experiments/` — dated training outputs
