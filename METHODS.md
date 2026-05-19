# Methods

## NIST FTIR reference data and local indexing

Experimental and reference infrared spectra were sourced from the **NIST Chemistry WebBook** IR collection, distributed through the community **NistChemData** mirror (Chernyshov, 2024), which packages WebBook-extracted JCAMP-DX (`.jdx`) records and related archives. Raw archives were ingested without full extraction using a ZIP-native indexer (`nistchemdata.py`) that assigns each spectrum a stable `reference_id`, stores JCAMP metadata as JSON, and writes paired wavenumber and absorbance arrays (`wn_json`, `y_json`) into a local **SQLite** index (version 7 build: `nistchemdata_ir_index_v7_fresh.sqlite`).

At index time, each spectrum was read with a JCAMP/CSV reader and passed through a unified preprocessing pipeline (`ftir_foundation.preprocess_spectrum`): wavenumbers were sorted ascending; transmittance was converted to absorbance when needed; a **rolling-minimum baseline** (151-point window) was subtracted; **Savitzky–Golay** smoothing (window 11, polynomial order 2) was applied; and absorbance was scaled to **[0, 1]** on the processed grid. These stored arrays were used for training feature extraction unless noted otherwise.

For dataset construction, an additional **NIST FTIR cm⁻¹ guard** was applied: metadata `xunits` were inspected, micrometer axes were converted to wavenumbers (ν = 10⁴/λ), and spectra with fewer than 32 points or span &lt; 200 cm⁻¹ were excluded so band-based features remained on a consistent wavenumber scale.

## Structure enrichment (PubChem)

Where CAS registry numbers, compound names, or file-path hints were available in NIST metadata, structures were resolved through the **PubChem PUG REST** API (Kim et al., 2023), with results cached on disk (`pubchem_structure_cache_v7.json` / `ml/runs/pubchem_train_writable.json`) to limit repeated requests. Successful hits were **canonicalized with RDKit** (`MolToSmiles`, isomeric). Across the v7 build, approximately **92.3%** of rows received a network-resolved structure, **4.3%** were ambiguous, and **0.6%** were misses, with no persistent network errors logged in the build metadata. v4 ontology training sets used **`--require-structure`** so only structure-resolved rows entered the SMARTS weak-label matrices (~15.7k spectra per family/specific build).

---

## Legacy multi-label SVM (v7 baseline, 303 dimensions)

*Retained for backward compatibility; not the primary v4 production path.*

Each indexed spectrum contributed one training row with **303 features**:

1. **Spectral block (14-D)** — Global mean, standard deviation, max, and min of processed absorbance, plus mean and standard deviation within five fixed wavenumber windows (cm⁻¹): 2500–3700 (X–H), 1650–1820 (carbonyl), 1200–1700, 900–1400, and 650–900 (`featurize`).

2. **RDKit block (32-D)** — Two-dimensional scalar descriptors when a molecule was built from SMILES or InChI.

3. **Mordred block (256-D)** — A fixed subset of **256** two-dimensional Mordred descriptors (`ignore_3D=True`).

4. **Structure flag (1-D)** — Binary `has_structure`; RDKit/Mordred zero-filled when absent.

**Weak labels (v7):** Twelve binary functional groups from **metadata keyword rules** (`FG_RULES` in `ml/ftir_fg_svm.py`): alcohol, amine, carbonyl, carboxylic acid, ester, ether, aromatic, halide, nitrile, nitro, alkene, alkyne.

**Estimator:** z-scored features; **OvR linear SVM** + **Platt sigmoid** calibration (`CalibratedClassifierCV`, internal CV). Artifact: `models/struct_fg_v7_pubchem_mordred.joblib` (18,157 rows in the historical full-matrix train).

---

## Evidence-first v4 ontology (current methodology)

### Ontology and assignment hierarchy

Functional groups are organized in **`ml/ftir_ontology.py`** into:

| Category | Role in reports | SVM training |
|----------|-----------------|--------------|
| **specific_fg** | Final interpretable assignments (phenol, amide, ether, …) | Specific model heads |
| **family** | Broad chemical families (hydroxy_containing, carbonyl_containing, …) | Family model heads |
| **local_motif** | Spectral windows (e.g. nitrile_alkyne_region) | Evidence only, not SVM *y* |
| **artifact** | Interference flags (moisture, CO₂, baseline drift, …) | Evidence only |
| **fallback** | Ambiguity buckets when evidence is incomplete | Rules only |

**Primary assignments** come from **spectral evidence and rule logic** (`ml/ftir_evidence.py`, `ml/ftir_rules.py`). Optional **machine learning refines advisory scores** but does not replace rules when `--fusion-mode annotate` is used (default for publication-style reports).

### Spectral evidence extraction

For each processed spectrum:

1. **Peaks** are detected (`find_peaks_simple`) and linked to a YAML **band library** (`ml/ftir_band_library.yaml`) with wavenumber windows, importance, and specificity.
2. **Regional statistics** are computed on fixed interpretable windows (`INTERPRETABLE_REGIONS` in `ml/ftir_interpretable_features.py`): OH/NH broad, C–H, carbonyl, C–O/fingerprint, aromatic C=C, nitrile/alkyne, Si–O overlap, etc.
3. **Ratios** (e.g. carbonyl-to-fingerprint, OH-to-fingerprint) summarize relative band strength.
4. **Artifact detectors** flag moisture-like OH, CO₂ spikes, baseline tilt, saturation, and fingerprint crowding.
5. Under **v4**, evidence is partitioned into **local motifs**, **family hints**, and **specific FG evidence** blocks for reporting.

### Rule-based assignment and v3 guardrails

Rules map band support scores to per-label confidences with required/supporting band groups (`ml/ftir_rules.py`). Optional **JSON presets** (`configs/rule_presets/`, e.g. `conservative`) tune thresholds.

**v3 guardrails** (`ml/ftir_guardrails.py`) apply **soft** false-positive control without deleting labels:

- **Paired-band requirements** (e.g. nitro: asymmetric + symmetric NO₂; amide: carbonyl + N–H or amide II; ester: C=O + C–O ester).
- **Single-band caps** and **competitor suppression** when a rival FG explains the same window better (e.g. siloxane vs ether/ester; phenol vs alcohol).
- **Artifact-aware down-weighting** when baseline instability or saturation is flagged.

Guardrails set `confidence_class` (e.g. `supported`, `tentative`, `local_possible`) and `evidence_completeness`; they are applied **before** ML refinement.

### ML advisory layer and soft gates

When enabled (`--ml-mode both`), two **independent OvR SVMs** provide calibrated probabilities:

| Model | Labels (examples) | Min positives at build |
|-------|-------------------|-------------------------|
| **Family** | hydroxy_containing, carbonyl_containing, nitrogen_containing, aromatic_system, C_O_containing, unsaturation_possible, nitro_family, silicon_oxygen_family | 20 |
| **Specific** | alcohol, phenol, amide, ester, ether, nitrile, nitro, siloxane, heteroaromatic, … (27 heads after ester SMARTS fix) | 10 |

**Important:** **SMARTS substructure matching** is used only to construct weak binary **training labels** *y* when `--label-source smarts` and a parseable structure are available. **SMARTS are not included in the feature vector *X*** for v4 production models (`n_smarts = 0` in dataset metadata).

**ML soft gates** (`ml/ftir_ml_refinement.py`, `--ml-guardrails strict` default):

- If spectral evidence is absent or partial but ML probability is high → `ml_only_warning` and capped `final_score`.
- High-risk labels (phenol, amide, ester, siloxane, nitrile, nitro, …) require matching evidence tiers before ML can reinforce a supported call.
- Silicon assignments require multiple Si-related regions before promotion above tentative.

Fusion modes: `annotate` (rules primary), `weighted`, `gate`, `ml_only` — production reports use **`annotate`**.

---

## Feature representation for v4 SVM (`spectral+evidence_v2`, 434 dimensions)

Training feature set **`spectral+evidence_v2`** (`evidence_feature_version = evidence_v2`):

| Block | Dimensions | Content |
|-------|------------|---------|
| Spectral | 14 | Same five-window means/stds as legacy `featurize` |
| Evidence v2 | 419 | Band supports; regional means/integrals/std; per-band peak counts and nearest-peak stats; peak width/sharpness/isolation/SNR proxies; broadness; extra ratios; `art_*` artifact flags; local motif supports |
| Structure flag | 1 | `has_structure_flag` (1 if RDKit-parseable SMILES/InChI at build time) |

Legacy **`spectral+evidence`** (~64-D evidence block, `evidence_v1`) remains supported for older joblib bundles.

Features are extracted **per spectrum at dataset build time** from the same preprocessed trace used in reports (evidence recomputed consistently at inference via `build_feature_row_layout`).

---

## Machine learning: dual v4 multi-label SVMs

**Software:** Python 3, NumPy, SciPy, scikit-learn (Pedregosa et al., 2011), joblib, RDKit (Landrum et al., 2024).

**Task.** Multi-label classification with **structure-derived weak labels**; family and specific tasks trained separately.

**Estimator.** `StandardScaler` on *X*; **OvR `LinearSVC`** (`class_weight='balanced'`, `dual=False`, `max_iter` large); each head wrapped in **`CalibratedClassifierCV`** with **sigmoid** calibration. Reported scores use **`ml_score_kind = calibrated_probability`** when calibration succeeds.

**Train/test split.** **`--split molecule`** (structure-level holdout), **20%** test fraction, **`random_state = 13`**.

**Per-label thresholds.** After calibration, thresholds are tuned on the held-out split (not fixed at 0.5). **High-risk labels** use a precision-biased objective. Thresholds and precision/recall/F1 are stored in joblib metadata and `*_threshold_summary.csv`.

**Hard-negative diagnostics.** For selected label pairs (e.g. phenol vs alcohol, siloxane vs ether), false-positive rates among hard-negative structures are logged at train time (`*_hard_negative_false_positives.csv`).

**Published artifacts (latest symlinks):**

- `ml/runs/struct_fg_family_v4_ontology_latest.joblib` — 8 heads; test macro-F1 ≈ 0.76 (family build, 2026-05-16)  
- `ml/runs/struct_fg_specific_v4_ontology_latest.joblib` — 27 heads; test macro-F1 ≈ 0.54 (specific build, 2026-05-16)

---

## Application spectra (non-NIST samples)

Powder and laboratory **CSV/JDX** files use the **same preprocessing** as the indexer (rolling-minimum baseline, Savitzky–Golay smoothing, normalization to [0, 1]). File-derived titles and CAS/formula hints supply metadata for rule masks only. **PubChem is not queried at inference** unless structures are supplied manually. The v4 pipeline runs **evidence → rules (+ guardrails) → optional dual ML → consensus** via `ml/ftir_pipeline.py`.

---

## Visualization and reporting

**Deliverable:** a self-contained **interactive HTML interpretation product** (`reports/structural_fg_svm_kronecker_report.py`). The chemistry engine (evidence → rules → guardrails → optional ML) is identical across modalities; **presentation** is selected by `--report-audience`.

**Philosophy:** one spectrum → one visual story → one concise interpretation → full audit depth on demand. From 2026-05-17, two standard modalities are maintained in parallel:

| Modality | CLI | Default when | Reader |
|----------|-----|--------------|--------|
| **Front-facing** | `--report-audience front` (or `--front-facing`) | `--report-style product_v1` + `balanced` / `summary` | Spectroscopists, publication, lab review |
| **Debug / audit** | `--report-audience debug` | `--report-density audit` | Pipeline development, ontology/rules tuning, ML diagnostics |

Implementation: `reports/front_facing_report.py` (spectroscopist summary, curated key evidence, front summary table); `reports/product_v1_report.py` (interpretation panel, full key evidence, debug Details). Regression: `reports/REPORT_PRODUCT_CONTRACT.md`; tests in `ml/tests/test_front_facing_report.py`, `ml/tests/test_product_v1_report.py`.

### Report styles (layout engine)

| Style | CLI | Notes |
|-------|-----|-------|
| **`product_v1`** (default) | `--report-style product_v1` | Pair with `front` or `debug` audience |
| **`legacy`** | `--report-style legacy` | Table-first layout without product_v1 curation |

### Front-facing layout (`--report-audience front`)

**Per spectrum (main view only):**

1. **Title** — Spectrum name (no status/ML badge clutter in header row).
2. **Spectrum (Plotly, dominant)** — Processed absorbance vs wavenumber (cm⁻¹); ~74% row height when ruler enabled. **Local hover** at each wavenumber (bands, motifs, local assignments) — no global probability dump.
3. **FTIR region ruler** (`--show-region-ruler`, default on) — Stacked horizontal bars for tentative ranges (O–H/N–H, sp² C–H, sp³ C–H, aldehyde C–H, C≡N/C≡C, C=O, C=C / amide II, C–O / fingerprint); activity-tiered fill (`ml/ftir_region_ruler.py`).
4. **Peak labels** — Diamond markers on up to **`--front-max-peak-labels`** (default 10) diagnostic peaks; weak peaks may plot unlabeled. `--label-all-diagnostic-peaks` is ignored in front mode.
5. **Kronecker panel** — Reduced height (~16% row); lighter stems; titled “Picked peaks”.
6. **Spectroscopist summary** — 1–3 sentence prose (`build_spectroscopist_summary`): top supported chemistry and main caveat; no raw scores or ontology jargon.
7. **Key evidence table** — Columns: Assignment | Key spectral evidence | Interpretation | Confidence; ≤6 rows; human-readable band phrases (no malformed tuple dumps).
8. **Cautions** — Short aggregated list (moisture, Si–O overlap, saturation, fingerprint crowding).
9. **ML check line** — Single human-readable status (e.g. “Rules dominant; ML did not change interpretation”) — not “ML Mixed”.
10. **Technical details (collapsible)** — Band map, full assignments, ML probabilities, rule scores, diagnostics; metadata only if `--show-metadata`.

**Batch summary (front):** Spectrum | Main interpretation | Key evidence | Main caution | Confidence | Link. Model paths and long caution paragraphs omitted from header (run settings in collapsed appendix).

**Quality-limited spectra:** when evidence is weak or artifacts dominate, a compact quality card replaces empty tables.

### Debug / audit layout (`--report-audience debug`)

Retains the pre–front-facing **product_v1** development view:

1. Spectrum + ruler (full Kronecker panel).
2. **Interpretation panel** — chips for main chemistry, specific assignments, cautions, confidence.
3. **Peak-picking summary** — detected / plotted / labeled counts.
4. **Key evidence table** — full product_v1 columns (may include more rows).
5. **Details** — band evidence map, justification panels, full assignment table, consensus, diagnostics.
6. **Metadata** — `<details>` block (open when `--report-density audit`).

`--label-all-diagnostic-peaks` and higher `--max-peak-labels` apply in debug mode.

### Shared visual options

| Element | CLI | Notes |
|---------|-----|-------|
| Band shading | `--show-band-shading` | Tiered regions (`ml/ftir_shade_regions.py`); optional `--shade-min-activity`, `--shade-faint-min`, `--shade-sensitive` |
| Shading labels | `--label-band-shading` | Short labels on shaded windows |
| Region ruler | `--show-region-ruler` | Default on for `product_v1` |
| Peak sensitivity | `--peak-sensitivity` | `conservative` → `very_sensitive` |
| Weak peaks | `--show-weak-peaks` | Plot weak peaks (muted); labeling still capped in front |

**Siloxane / Si–O presentation:** without paired silicon-region evidence, reports show **“Si–O overlap”** or **“C–O / Si–O overlap”** rather than promoting siloxane as primary supported chemistry. ATR-aware guardrails when `--mode ATR` or inferred ATR.

Scores in tables are **evidence scores** (capped rule support), not probabilities.

### Report options (production defaults)

| Flag | Purpose |
|------|---------|
| `--report-style product_v1` | Product layout engine (default) |
| `--report-audience front` | Spectroscopist deliverable (default for balanced `product_v1`) |
| `--report-audience debug` | Full metadata/diagnostics |
| `--front-max-peak-labels N` | Cap labeled peaks in front mode (default 10) |
| `--show-metadata` | Expose metadata in front mode |
| `--ontology v4` | v4 ontology buckets |
| `--guardrails v3` | Soft FP guardrails on rules |
| `--ml-mode both` + family/specific models | Dual advisory SVMs |
| `--fusion-mode annotate` | Rules-primary scores |
| `--ml-guardrails strict` | Cap weak-evidence / strong-ML conflicts |
| `--report-density audit` | Implies debug audience default |
| `--anonymize-metadata` | Basenames only in HTML/CSV |
| `--visual-theme matlab` | Publication-style Plotly/HTML (white, MATLAB-blue spectrum); hover preserved |
| `--export-static-figures` | Matplotlib PNGs under `<report-dir>/presentation/figures/` (or `--static-out`) |
| `--static-format` / `--static-dpi` | `png`, `svg`, or `pdf` at print scale |
| `--static-peak-label-policy key\|all` | Cap static PNG labels (`key`, default 12) or match interactive (`all`) |
| `--static-label-layout-mode smart` | Collision-aware peak labels on static PNG (`annotation_layout.py`) |
| `--label-all-above-height` | Label every picked peak with normalized absorbance ≥ threshold (e.g. `0.05`–`0.1`) |
| `--fingerprint-cluster-distance 0` | Show individual fingerprint labels when many peaks are labeled |

**Example (front-facing batch):**

```bash
cd SpectraReason
export PYTHONPATH="$(pwd)"

python reports/structural_fg_svm_kronecker_report.py batch \
  --inputs examples/spectra/Pyrrole_109-97-7-IR.jdx \
  --ontology v4 --guardrails v3 --ml-mode both \
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib \
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib \
  --fusion-mode annotate --ml-guardrails strict \
  --report-style product_v1 --report-audience front \
  --show-region-ruler --peak-sensitivity sensitive --show-weak-peaks \
  --front-max-peak-labels 10 \
  --out reports/demo_front/REPORT.html
```

**Example (debug/audit batch):** add `--report-audience debug --report-density audit --label-all-diagnostic-peaks --max-peak-labels 30 --out reports/product_v1_debug_demo/REPORT.html`.

Optional CSV export (`--export-csv`): consensus, rules long-form, band matches (full scientific record alongside HTML).

**Example (MATLAB-style + static export, FTIR_POWDER):** see `CANONICAL_OUTPUTS.md` and `reports/FIGURES_AND_EXPORT.md`. Typical flags: `--visual-theme matlab --export-static-figures --label-all-above-height 0.05 --show-region-ruler`. Static export writes **separate** `{stem}_spectrum_peaks.png` and `{stem}_region_guide.png` (avoid stacking ruler + spectrum in one slide). MATLAB: `cd` to `matlab_export` and run `make_figures` (collision-aware peak labels; edit font variables at top of script).

### Static figures and MATLAB post-processing

Full reference: **`reports/FIGURES_AND_EXPORT.md`**.

1. **Interactive HTML** — Plotly spectrum + ruler + capped or height-filtered labels; local hover only.
2. **Presentation PNGs** — `reports/static_figure_export.py`; smart label layout; optional `presentation/export_layout_audit.md`.
3. **MATLAB** — `matlab_export/make_figures.m` from `reports/matlab_visual_theme.py`; separate `_spectrum_peaks_matlab` / `_region_guide_matlab` panels; user-tuned `fontSpectrumPeakLabel` (default 14) and `fontSpectrumAxis` (default 18).

### Other report modes

1. **Static structural SVM report** (`structural_fg_svm_report.py`) — Matplotlib trace + OvR probability table (v7-oriented).

2. **Kronecker-only report** (`kronecker_spectrum_report.py`) — Spectrum + stems without full v4 pipeline tables.

3. **Robustness batch** (`structural_fg_svm_robustness_report.py`) — Perturbation stability (optional).

Batch outputs are self-contained HTML with sidebar navigation. Design audit: `reports/product_v1_design_audit.md`.

---

## Software availability

The v4 ontology pipeline, training scripts, trained family/specific models, and report generators are maintained in **`FTIR_SVM_v4/`** (this tree). Legacy v7 artifacts remain under `models/` and `data/training/`. See `context.md`, `CANONICAL_OUTPUTS.md`, and `reports/FIGURES_AND_EXPORT.md` for command lines, artifact paths, canonical front/debug demos (`reports/product_v1_front_demo/`, `reports/product_v1_debug_demo/`), FTIR_POWDER bundles (`reports/ftir_powder_pda_eg_con_new_front/`, `reports/ftir_powder_pda_eg_con_new_matlab/`), and audit reports (`reports/v4_evidence_v2_retraining_audit.md`, `reports/product_v1_design_audit.md`).

---

## References (APA 7)

Chernyshov, I. (2024). *NistChemData* [Data set]. GitHub. https://github.com/IvanChernyshov/NistChemData

Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline correction with asymmetric least squares smoothing. *Leiden University Medical Centre Report*, *1*(1), 5.

Kim, S., Chen, J., Cheng, T., Gindulyte, A., He, J., He, S., Li, Q., Shoemaker, B. A., Thiessen, P. A., Yu, B., Zaslavsky, L., Zhang, J., & Bolton, E. E. (2023). PubChem 2023 update. *Nucleic Acids Research*, *51*(D1), D1373–D1380. https://doi.org/10.1093/nar/gkac956

Landrum, G., et al. (2024). *RDKit: Open-source cheminformatics* (Version 2024.03.1) [Computer software]. https://www.rdkit.org

Lamprecht, M.-S., & Lamprecht, T. (2022). *JCAMP-DX*. IUPAC. https://iupac.org/what-we-do/digital-standards/jcamp-dx/

Moriwaki, H., Tian, Y.-S., Kawashita, N., & Takagi, T. (2018). Mordred: A molecular descriptor calculator. *Journal of Cheminformatics*, *10*, 4. https://doi.org/10.1186/s13321-018-0258-y

National Institute of Standards and Technology. (n.d.). *NIST Chemistry WebBook*. NIST Standard Reference Database Number 69. https://doi.org/10.18434/T4D303

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Plotly Technologies Inc. (2015). *Collaborative data science* [Computer software]. https://plot.ly

Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, *36*(8), 1627–1639. https://doi.org/10.1021/ac60214a047

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2
