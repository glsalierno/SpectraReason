# SVM spectral featurization audit — `struct_fg_basic_v4_ontology_latest`

**Artifact:** `ml/runs/struct_fg_basic_v4_ontology_latest.joblib`  
**Training dataset prefix (logical):** `ml/runs/ds_v4_basic_spectral_evidence_nist` (companion `.npz` / `.meta.json` from the structural FG SVM dataset builder)  
**Audit date:** 2026-05-15 (repository state)

---

## 1) Direct answer (main question)

**The v4 basic SVM does *not* use Kronecker / stem-delta peak features.**

- **Kronecker / discrete stem plots** in HTML reports (`structural_fg_svm_kronecker_report.py`, `find_peaks_simple` for figures) are **visualization only**. They are **not** concatenated into the feature row passed to `predict_proba_row` / `build_feature_row_layout`.
- The trained bundle is **`feature_set = spectral+evidence`**: **14-D fixed-window spectral statistics** (`ml.ftir_fg_svm.featurize`) **plus** a **49-D vector** from `ml.ftir_ontology.evidence_feature_vector` (band supports, ratios, two summary scalars, v4 local-motif supports), **plus** `has_structure_flag` → **64** features total.
- **Mordred** and **RDKit numeric blocks** are **absent** for this model (`n_mordred = 0`, `n_rdkit = 0`). **SMARTS binary columns are not used as *features*** (`n_smarts = 0`); SMARTS is used for **labels** (`label_source = smarts`, v4 ontology heads).

So the model is best described as: **spectral coarse windows + compact evidence summary**, with **no** full interpolated spectrum grid, **no** Kronecker codebook, and **no** explicit per-peak width/sharpness/SNR columns in `X` (those are computed inside `extract_spectral_evidence` to drive band/peak logic, but only a **scalar** `sum_n_peaks` and **band-level** support scores surface in `evidence_feature_vector`).

---

## 2) Bundle introspection (loaded joblib)

| Field | Value |
|--------|--------|
| `meta.feature_set` | `spectral+evidence` |
| `meta.feature_dim` / `n_features` | **64** |
| `meta.n_spectral` | **14** |
| `meta.n_evidence` | **49** |
| `meta.n_smarts` | **0** |
| `meta.n_mordred` | **0** |
| `meta.n_rdkit` | **0** |
| `meta.n_has_flag` | **1** |
| `meta.label_source` | `smarts` |
| `meta.ontology` | `v4` |
| `meta.model_kind` | `basic` |
| `meta.possible_structural_label_feature_leakage` | `False` |
| `meta.label_feature_smarts_leakage` | `False` |
| `meta.n_rows` | 15740 |
| `meta.train_estimator` | OneVsRest + per-label sigmoid calibration |
| `meta.ml_score_kind` | `calibrated_probability` |
| Top-level `calibration` | `method: sigmoid`, `fitted: true`, `cv: 5` |
| `labels` (9 OvR heads) | `hydroxy_containing`, `carbonyl_containing`, `nitrogen_containing`, `aromatic_system`, `c_o_containing`, `unsaturation_possible`, `nitro`, `nitrile`, `silicon_oxygen_family` |

**First 50 `meta.feature_names` (exact order in bundle):**

0–13: `spectral_y_global_*`, `spectral_cm1_<lo>_<hi>_mean_absorbance`, `spectral_cm1_*_std_absorbance` for windows (2500–3700), (1650–1820), (1200–1700), (900–1400), (650–900).  
14–44: `band_<band_id>` (31 library band support scores).  
45–49: `ratio_aromatic_to_ch_stretch`, `ratio_carbonyl_to_fingerprint`, `ratio_nitrile_to_fingerprint`, `ratio_oh_nh_to_fingerprint`, `ratio_siloxane_to_c_o`.  
50–51: `sum_oh_nh_broadness`, `sum_n_peaks`.  
52–62: `motif_<v4_local_motif_key>` (11 entries).  
63: `has_structure_flag`.

**Prefix counts (same order):** `spectral_*` 14, `band_*` 31, `ratio_*` 5, `sum_*` 2, `motif_*` 11, `has_structure_flag` 1.  
**Note:** No `art_*` columns appear in this bundle’s `feature_names` (artifact flags are not part of the stable evidence column list produced by the probe spectrum used to fix column order, and training calls `extract_spectral_evidence` without merging `ftir_artifacts.detect_spectral_artifacts` into that dict before `evidence_feature_vector`).

---

## 3) Feature categories (classification)

| Category | Used in this SVM? | Notes |
|----------|-------------------|--------|
| Raw / high-res interpolated spectral vector | **No** | Only 14 global + regional **mean/std** scalars. |
| Kronecker / stem delta codebook | **No** | Report-only; not in `build_feature_row_layout`. |
| Regional integrals (`regions[*].integral` in evidence) | **No** in `X` | Computed in `extract_spectral_evidence` but **not** exported by `evidence_feature_vector`. |
| Regional max / band-window stats | **Partially** | `band_*` scores combine **region rel_max** and **peak_support** near the band (`ftir_evidence.py`). |
| Peak count | **Partial** | Only **`sum_n_peaks`** (global count). No per-band counts in `X`. |
| Peak shape / width / SNR / isolation | **Not explicit in `X`** | `_enrich_peaks_with_quality` feeds band/peak logic; **no** dedicated `quality_*` columns in `evidence_feature_vector`. |
| Broadness | **Yes (1-D)** | `sum_oh_nh_broadness` from interpretable OH/NH broadness metric. |
| Ratio features | **Yes (5-D)** | As listed above (`ftir_evidence` region `rel_max` ratios). |
| Artifact flags | **No in this 64-D layout** | No `art_*` names in stored `feature_names`. |
| Rule / guardrail outputs | **No** | `ftir_rules.py` / `ftir_guardrails.py` run **after** spectrum featurization in the evidence-first **pipeline** for assignments, not as SVM inputs. |
| Mordred | **No** | `n_mordred = 0`. |
| SMARTS as *features* | **No** | `n_smarts = 0`. SMARTS used for **y** only (with documented leakage guard for `spectral+smarts*` feature sets). |
| `has_structure_flag` | **Yes** | 1-D binary from resolved RDKit mol in metadata. |

---

## 4) Training / inference code path (concise)

- **Dataset builder** (`ml/structural_fg_svm.py`, build-dataset): for each NIST row, `extract_spectral_evidence(wn, yy, peaks=None, config={"ontology":"v4"})` then `build_feature_row_layout` → `spectral+evidence` row; labels from `_infer_y_vector` with `label_source=smarts`, `ontology=v4`.
- **Inference** (`predict_proba_row`): same `featurize` + `extract_spectral_evidence(..., peaks=None)` + `evidence_feature_vector`; optional `prepare_nist_ftir_cm1` if `nist_ftir_cm1_prep_applied` in meta.

---

## 5) Missing opportunities (minimal upgrade list — *not implemented here*)

The following are **already partially present** in the evidence dict but **not** flattened into `evidence_feature_vector` today, or are natural extensions:

- Per-interpretable-region **`integral` / `max` / `std`** from `evidence["regions"]` (e.g. `regional_integral_carbonyl`, `regional_max_fingerprint`, …).
- **Per-band** `peak_count_<band>`, `nearest_peak_distance_<band>`, `peak_width_<band>`, `peak_sharpness_<band>`, `peak_isolation_<band>` aggregated from `peaks_near` + quality fields.
- Explicit **`broadness_carboxylic_acid_OH`** (or acid-specific broadness) if distinguishable from general OH/NH broadness.
- Extra **ratios** already named in your wishlist: e.g. **`OH_to_fingerprint_ratio`** (alias / duplicate naming of existing `ratio_oh_nh_to_fingerprint` — avoid double-counting), **`CO_to_carbonyl_ratio`**, **`SiO_overlap_competitor_ratio`** (could refine current `ratio_siloxane_to_c_o` or add organic-competitor denominator).

**Implementation locus (for a future PR + retrain):** extend `evidence_feature_vector` + `_stable_evidence_feature_names` probe, keep column order stable, bump `meta["n_evidence"]` / `feature_dim`, rebuild `.npz`, retrain.

---

## 6) Recommended next `feature_set` string (after you approve retrain)

- Short term (low risk): keep **`spectral+evidence`** and **append** new evidence scalars only → still one clear block after spectral 14-D.
- If you add many columns: consider renaming to **`spectral+evidence_v2`** in meta for traceability.
- Avoid **`spectral+smarts+mordred+evidence`** unless you need structure in `X`; it increases **leakage and distribution-shift risk** (see below).

---

## 7) Risks — especially leakage

- **Current v4 basic bundle:** **Labels from SMARTS**, **features spectral+evidence only** → **no structural label→feature leakage** for SMARTS (flags `possible_structural_label_feature_leakage: false`). `has_structure_flag` can correlate with label prevalence (structures present vs salts/polymers) but is not a duplicate SMARTS channel.
- **If you train with `spectral+smarts` or `...+smarts+...`:** the codebase explicitly warns when **label source and SMARTS features** co-present — that is **direct leakage** unless you split by molecule and hold out structures carefully.
- **Mordred / RDKit blocks:** can encode **global structure**; combined with spectrum they can **dominate** and hurt transfer to **unknown structures** or **polymers / salts** with weak/novel RDKit representation. Prefer **spectrum + hand-crafted evidence** for FTIR-first models unless you have a clear use case.

---

## 8) Summary table — “what does the model use?”

| Option | Applies to this v4 basic joblib? |
|--------|-----------------------------------|
| Only Kronecker deltas | **No** |
| Spectral + Kronecker | **No** |
| Spectral + evidence | **Yes** (this model) |
| Spectral + evidence + explicit shape/quality columns | **No** (shape/quality internal to evidence extraction, not in `X`) |
| Structural descriptors (Mordred / SMARTS-as-features) | **No** |

---

*End of audit. Retraining was intentionally not performed per project instructions.*
