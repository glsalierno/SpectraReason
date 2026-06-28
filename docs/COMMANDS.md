# Commands (FTIR_SVM_v5)

All commands assume:

```powershell
Set-Location "c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5"
$env:PYTHONPATH = (Get-Location).Path
```

**NIST index (typical):**  
`C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\NIST\reference_libraries\nistchemdata_ir_index_v7_fresh.sqlite`

Defaults: `docs/PRODUCTION_DEFAULTS.md` · Reproducibility: `docs/REPRODUCIBILITY.md`

---

## Production front-facing report

```powershell
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs "examples\spectra\1H-Indol-5-ol-1953-54-4-IR.jdx" `
  --ontology v4 --guardrails v3 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --fusion-mode annotate --ml-guardrails strict `
  --report-style product_v1 --report-audience front `
  --visual-theme matlab --show-region-ruler `
  --peak-sensitivity sensitive --show-weak-peaks `
  --front-max-peak-labels 10 `
  --export-csv reports/my_run/csv `
  --out reports/my_run/REPORT.html
```

---

## Debug / audit report

```powershell
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs "examples\spectra\1H-Indol-5-ol-1953-54-4-IR.jdx" `
  --ontology v4 --guardrails v3 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --fusion-mode annotate --ml-guardrails strict `
  --report-style product_v1 --report-audience debug `
  --show-region-ruler --peak-sensitivity sensitive `
  --export-csv reports/my_debug/csv `
  --out reports/my_debug/REPORT.html
```

---

## Regenerate reference snapshots

```powershell
python scripts/release_stabilize.py --snapshots-only
```

Outputs: `reports/reference_snapshots/front/REPORT.html`, `debug/REPORT.html`, `static_figures/`.

---

## Run production report (`product_v1`, legacy block)

```powershell
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs "examples\spectra\1H-Indol-5-ol-1953-54-4-IR.jdx" `
  --ontology v4 --guardrails v3 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --fusion-mode annotate --ml-guardrails strict `
  --report-style product_v1 --report-density balanced `
  --show-band-shading --label-band-shading `
  --include-evidence --include-ml --include-consensus `
  --export-csv examples/_evidence_pipeline_report/csv `
  --out examples/_evidence_pipeline_report/REPORT_all_examples_spectra.html
```

Batch multiple inputs by repeating paths in `--inputs`.

---

## Run deconv comparison report

```powershell
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs "examples\spectra\1H-Indol-5-ol-1953-54-4-IR.jdx" `
  --ontology v4 --guardrails v3 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/experiments/v4_deconv_specific/struct_fg_specific_v4_ontology_20260517_150309.joblib `
  --fusion-mode annotate --ml-guardrails strict `
  --report-style product_v1 --report-density balanced `
  --show-band-shading --label-band-shading `
  --include-evidence --include-ml --include-consensus `
  --export-csv examples/_evidence_pipeline_report_deconv/csv `
  --out examples/_evidence_pipeline_report_deconv/REPORT_all_examples_spectra.html
```

---

## Train family (production)

```powershell
python -m ml.structural_fg_svm build-dataset `
  --nist-index "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\NIST\reference_libraries\nistchemdata_ir_index_v7_fresh.sqlite" `
  --out-prefix ml/runs/ds_v4_family_spectral_evidence_v2_nist `
  --model-kind family --label-source smarts --feature-set spectral+evidence_v2 `
  --ontology v4 --pipeline-version v4_ontology --min-label-positives 20 `
  --require-structure --enrich-pubchem --pubchem-cache ml/runs/pubchem_train_writable.json

python -m ml.structural_fg_svm train `
  --dataset-prefix ml/runs/ds_v4_family_spectral_evidence_v2_nist `
  --version v4_ontology --ontology v4 --model-kind family `
  --calibration sigmoid --split molecule --min-label-positives 20 `
  --hard-negative-mode on --random-state 13 --out ml/runs
```

---

## Train specific (production)

```powershell
python -m ml.structural_fg_svm build-dataset `
  --nist-index "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\NIST\reference_libraries\nistchemdata_ir_index_v7_fresh.sqlite" `
  --out-prefix ml/runs/ds_v4_specific_spectral_evidence_v2_nist `
  --model-kind specific --label-source smarts --feature-set spectral+evidence_v2 `
  --ontology v4 --pipeline-version v4_ontology --min-label-positives 10 `
  --require-structure --enrich-pubchem --pubchem-cache ml/runs/pubchem_train_writable.json

python -m ml.structural_fg_svm train `
  --dataset-prefix ml/runs/ds_v4_specific_spectral_evidence_v2_nist `
  --version v4_ontology --ontology v4 --model-kind specific `
  --calibration sigmoid --split molecule --min-label-positives 10 `
  --hard-negative-mode on --random-state 13 --out ml/runs
```

Experiment trains: add `--no-update-latest` and `--out ml/runs/experiments/<name>`.

---

## Run tests

```powershell
python -m pytest ml/tests/ -q
```

Focused report regression:

```powershell
python -m pytest ml/tests/test_product_v1_report.py ml/tests/test_report_features.py ml/tests/test_kronecker_report_density.py -q
```

---

## Static presentation figures + MATLAB (`pda_eg` example)

Writes `presentation/figures/{stem}_spectrum_peaks.png` and `{stem}_region_guide.png`, plus `matlab_export/make_figures.m`. Details: `reports/FIGURES_AND_EXPORT.md`.

```powershell
$pda = "c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\FTIR_POWDER\pda_eg_con_new.CSV"

python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs $pda `
  --ontology v4 --guardrails v3 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --fusion-mode annotate --ml-guardrails strict `
  --report-style product_v1 --report-audience front `
  --visual-theme matlab --show-region-ruler `
  --label-all-above-height 0.05 --fingerprint-cluster-distance 0 `
  --export-static-figures --static-peak-label-policy key --max-static-peak-labels 12 `
  --export-csv reports/ftir_powder_pda_eg_con_new_matlab/csv `
  --out reports/ftir_powder_pda_eg_con_new_matlab/REPORT.html
```

MATLAB (after Python run):

```matlab
cd('c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5\reports\ftir_powder_pda_eg_con_new_matlab\matlab_export');
make_figures
```

---

## Run report regression test

See `reports/report_regression_checklist.md`. Quick automated check:

```powershell
python -m pytest ml/tests/test_product_v1_report.py -q
```

---

## Clean temporal reports and caches

```powershell
python scripts/cleanup_temporal.py --dry-run   # preview
python scripts/cleanup_temporal.py             # archive to reports/_archive/2026-05/
```

Keeps: `reference_snapshots`, `product_v1_*_demo`, `ftir_powder_pda_eg_con_new_matlab`, `examples_matlab_pyrrole_indol`, active coverage audits.

---

## Archive old reports (manifest-first)

```powershell
python scripts/release_stabilize.py --dry-run-archive   # preview
python scripts/release_stabilize.py --skip-vulture --skip-archive  # snapshots only
python scripts/release_stabilize.py   # full: snapshots + archive + vulture + audit
```

Writes `reports/archive_manifest_YYYYMMDD.csv` and moves clutter to `reports/_archive/2026-05/`.

Do **not** archive: `product_v1_front_demo`, `product_v1_debug_demo`, `reference_snapshots`, `ftir_powder_pda_eg_con_new_matlab`, latest joblibs, NIST indexes, NPZ datasets.

---

## Vulture dead-code audit (optional dev)

```powershell
pip install -r requirements-dev.txt
python -m vulture . --min-confidence 80 --exclude ".venv,env,__pycache__,reports/_archive,ml/runs,data,*.joblib,*.npz"
```

Or: `python scripts/release_stabilize.py --skip-archive --skip-snapshots` (writes `reports/vulture_dead_code_audit.md`).

---

## External / open dataset ingestion (experimental tier)

Does **not** update production `*_latest.joblib`. See `docs/EXTERNAL_DATASETS.md`.

```powershell
# List registered sources
python -m ml.external list-sources

# Validate pipeline on bundled examples (JCAMP → SQLite → QA → benchmarks)
python -m ml.external demo-ingest-examples --out-db data/experimental/examples_index.sqlite

# SDBS first batch (subfolders + manifest — see docs/SDBS_FIRST_BATCH.md)
python -m ml.external ingest-sdbs
# After JCAMP files are in raw/sdbs/{nitro_positive,...}/:
python -m ml.external merge-indexes data/experimental/examples_index.sqlite data/experimental/sdbs_ir_index.sqlite --out-db data/experimental/merged_external_index.sqlite
python -m ml.external dataset-qa --sqlite-index data/experimental/merged_external_index.sqlite
python -m ml.external summarize-confounder-coverage

# Open polymer / Zenodo local drops
python -m ml.external ingest-open-polymer --raw-dir data/external_sources/raw/open_polymer

# Generic JCAMP folder plugin
python -m ml.external ingest-jcamp-folder `
  --out-db data/experimental/user_jcamp.sqlite `
  --library-path data/external_sources/raw/user_jcamp `
  --library-source jcamp --source-id user_jcamp

# Merge experimental indexes
python -m ml.external merge-indexes `
  data/experimental/sdbs_ir_index.sqlite `
  data/experimental/open_polymer_ir_index.sqlite `
  --out-db data/experimental/merged_external_index.sqlite

# QA audit
python -m ml.external dataset-qa --sqlite-index data/experimental/merged_external_index.sqlite

# Confounder benchmark JSON subsets
python -m ml.external build-confounder-benchmarks `
  --sqlite-index data/experimental/merged_external_index.sqlite

# Experimental NPZ (writes under ml/runs/experimental/)
python -m ml.external build-external-dataset `
  --sqlite-index data/experimental/merged_external_index.sqlite `
  --out-prefix ml/runs/experimental/ds_external_family

# Targeted confounder coverage (updates manifests + expansion audit)
python -m ml.external summarize-confounder-coverage

# Expansion audit only
python -c "from ml.external.generate_expansion_audit import generate_expansion_audit; generate_expansion_audit()"
```

Target class definitions: `docs/TARGET_EXTERNAL_EXPANSION.md`

Train on experimental NPZ only with `--no-update-latest` and `--out ml/runs/experimental/<name>`.

---

## Manuscript figures, interactive curation, and spectral chunks

Full reference: `reports/CURATION_AND_CHUNKS.md`

```powershell
$csv = "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\csvs\Dopamine.CSV"
$out = "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\ftir figs\Dopamine"

python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs $csv `
  --ontology v4 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --report-audience front --visual-theme matlab `
  --export-paper-figures `
  --export-interactive-curation `
  --export-region-stacks `
  --export-chunk-data `
  --chunk-collage `
  --save-label-overrides `
  --label-overrides $out `
  --paper-out "$out\presentation\paper_figures" `
  --out "$out\REPORT.html"
```

After curating in the browser, re-run with `--apply-label-overrides --label-overrides $out`.  
After editing ranges in `REPORT.html`, download `ranges_config.json` and pass `--regions-file` (alias `--ranges-file`).

Combined multi-spectrum stacks: batch all CSV paths with `--out ...\ftir figs\_combined_region_stacks\REPORT.html`.

Focused tests:

```powershell
python -m pytest ml/tests/test_paper_ftir_figures.py ml/tests/test_interactive_curation.py ml/tests/test_peak_snap.py ml/tests/test_chunk_export.py -q
```
