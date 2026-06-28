# SpectraReason `v5` branch

Active development line synced from the local **FTIR_SVM_v5** lab tree (2026-06).

## What landed on `v5` (vs `main`)

- **Intensity modes** — `lib/intensity_modes.py`; native vs absorbance vs difference; apparent transmittance with 95% baseline cap for blank-subtracted spectra
- **Scaled blank subtraction** — `scripts/subtract_solvent_blank.py`
- **Manuscript + curation** — interactive peak curation, chunk/region stack export, label overrides, spectrum feedback
- **External data framework** — `ml/external/` ingest, provenance, confounder coverage CLI
- **Report CLI flags** — `--allow-apparent-transmittance`, `--export-interactive-curation`, `--export-chunk-data`, etc.

See `reports/CURATION_AND_CHUNKS.md` and `docs/COMMANDS.md`.

## Quickstart (unchanged)

```powershell
git checkout v5
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
.\scripts\setup_bundled_artifacts.ps1
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs examples/spectra/Catechol-120-80-9-IR.jdx `
  --ontology v4 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --report-audience front --visual-theme matlab `
  --export-paper-figures --export-interactive-curation `
  --out reports/demo_front/REPORT.html
```

## Not in git (by design)

- Proprietary lab CSVs and article figure bundles (OneDrive `POC_PDA_ODA article/`)
- NIST SQLite index, SDBS raw JCAMP downloads
- Production `ml/runs/*_latest.joblib` scratch copies (use bundled artifacts + setup script)

## Merge policy

Keep **`main`** collaborator-stable; merge **`v5` → `main`** after tests and doc review when ready for a tagged release.
