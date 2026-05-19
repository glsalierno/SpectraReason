# Commands (SpectraReason)

All commands assume repository root on `PYTHONPATH`:

```bash
cd SpectraReason
export PYTHONPATH="$(pwd)"    # Windows: $env:PYTHONPATH = (Get-Location).Path
```

Defaults: [`PRODUCTION_DEFAULTS.md`](PRODUCTION_DEFAULTS.md) · Reproducibility: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

**NIST index (local, not in git):** place SQLite at `data/external/nist_index.sqlite`
(symlink or copy). See `data/README_DATA.md`.

---

## Production front-facing report

```bash
python reports/structural_fg_svm_kronecker_report.py batch \
  --inputs examples/spectra/1H-Indol-5-ol-1953-54-4-IR.jdx \
  --ontology v4 --guardrails v3 --ml-mode both \
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib \
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib \
  --fusion-mode annotate --ml-guardrails strict \
  --report-style product_v1 --report-audience front \
  --visual-theme matlab --show-region-ruler \
  --peak-sensitivity sensitive --show-weak-peaks \
  --front-max-peak-labels 10 \
  --export-csv reports/my_run/csv \
  --out reports/my_run/REPORT.html
```

---

## Debug / audit report

```bash
python reports/structural_fg_svm_kronecker_report.py batch \
  --inputs examples/spectra/1H-Indol-5-ol-1953-54-4-IR.jdx \
  --ontology v4 --guardrails v3 --ml-mode both \
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib \
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib \
  --fusion-mode annotate --ml-guardrails strict \
  --report-style product_v1 --report-audience debug \
  --show-region-ruler --peak-sensitivity sensitive \
  --export-csv reports/my_debug/csv \
  --out reports/my_debug/REPORT.html
```

---

## Static presentation figures + MATLAB export

```bash
python reports/structural_fg_svm_kronecker_report.py batch \
  --inputs examples/spectra/Benzoic\ acid\ -\ 65-85-0-IR.jdx \
  --ontology v4 --guardrails v3 --ml-mode both \
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib \
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib \
  --fusion-mode annotate --ml-guardrails strict \
  --report-style product_v1 --report-audience front \
  --visual-theme matlab --show-region-ruler \
  --export-static-figures --static-peak-label-policy key --max-static-peak-labels 12 \
  --out reports/static_demo/REPORT.html
```

Outputs: `presentation/figures/*.png`, `matlab_export/make_figures.m`. See `reports/FIGURES_AND_EXPORT.md`.

MATLAB (optional):

```matlab
cd('reports/static_demo/matlab_export');
make_figures
```

---

## Regenerate reference snapshots

```bash
python scripts/release_stabilize.py --snapshots-only
```

Outputs: `reports/reference_snapshots/front/REPORT.html`, `debug/REPORT.html`, `static_figures/`.

---

## Benchmark generation (dataset + train)

**Family:**

```bash
python -m ml.structural_fg_svm build-dataset \
  --nist-index data/external/nist_index.sqlite \
  --out-prefix ml/runs/ds_v4_family_spectral_evidence_v2_nist \
  --model-kind family --label-source smarts --feature-set spectral+evidence_v2 \
  --ontology v4 --pipeline-version v4_ontology --min-label-positives 20 \
  --require-structure --enrich-pubchem --pubchem-cache ml/runs/pubchem_train_writable.json

python -m ml.structural_fg_svm train \
  --dataset-prefix ml/runs/ds_v4_family_spectral_evidence_v2_nist \
  --version v4_ontology --ontology v4 --model-kind family \
  --calibration sigmoid --split molecule --min-label-positives 20 \
  --hard-negative-mode on --random-state 13 --out ml/runs
```

**Specific:** same with `--model-kind specific` and `--min-label-positives 10`.

Experiments: `--no-update-latest --out ml/runs/experiments/<name>`.

---

## External ingestion smoke

```bash
python -m ml.structural_fg_svm resolve-structure --name "benzoic acid" --cas 65-85-0
```

---

## Tests

```bash
python -m pytest ml/tests/ -q
```

Focused report regression:

```bash
python -m pytest ml/tests/test_product_v1_report.py ml/tests/test_report_features.py -q
```

---

## Release snapshot generation (full stabilize)

```bash
python scripts/release_stabilize.py --dry-run-archive    # preview archive moves
python scripts/release_stabilize.py --skip-vulture --skip-archive  # snapshots only
python scripts/release_stabilize.py   # snapshots + archive + vulture + audit
```

---

## Vulture dead-code audit

```bash
pip install -r requirements-dev.txt
python -m vulture . --min-confidence 80 \
  --exclude ".venv,env,__pycache__,reports/_archive,ml/runs,data,*.joblib,*.npz"
```

Or: `python scripts/release_stabilize.py --skip-archive --skip-snapshots`.
