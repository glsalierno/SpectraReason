# Bundled ML artifacts (ready to use)

Pre-built **NIST-homogenized** training matrices and **production SVM** models ship in this
folder so collaborators can run reports and retrain without rebuilding from SQLite.

**Total download (approx.):** ~95 MB (v4) + ~65 MB (v7 legacy) + ~44 MB (shared PubChem cache).

## v4 production (evidence-first reports)

Path: `data/training/bundled/v4_production/`

| File | Role |
|------|------|
| `struct_fg_family_v4_ontology_latest.joblib` | Family OvR SVM (8 heads), calibrated |
| `struct_fg_specific_v4_ontology_latest.joblib` | Specific OvR SVM (~27 heads), calibrated |
| `ds_v4_family_spectral_evidence_v2_nist.npz` | Training matrix **X**, **Y** (family) |
| `ds_v4_specific_spectral_evidence_v2_nist.npz` | Training matrix **X**, **Y** (specific) |
| `*.meta.json` | Build provenance (15.6k rows, 434-D `spectral+evidence_v2`) |
| `pubchem_train_writable.json` | PubChem SMILES/InChI cache used at build time |

**Feature vector (434-D):** 14 spectral window stats + 419 evidence-v2 columns + `has_structure` flag.  
**Not in X:** Mordred/RDKit descriptors (SMARTS weak labels only, via PubChem structures).

**Rows:** ~15,635 structure-resolved NIST spectra (see `n_rows` in meta).

## v7 legacy (spectral + RDKit + Mordred)

Path: `data/training/bundled/v7_mordred/`

| File | Role |
|------|------|
| `struct_fg_v7_pubchem_mordred.npz` | **303-D** matrix (14 + 32 RDKit + 256 Mordred + flag) |
| `struct_fg_v7_pubchem_mordred.meta.json` | Label names, column layout |
| `struct_fg_v7_pubchem_mordred.joblib` | Legacy 12-label OvR SVM |

Use for backward-compatible v7 reports (`--ml-mode legacy`).

## Shared PubChem cache

`data/training/bundled/pubchem_structure_cache_v7.json` — duplicate of the v4 build cache for convenience.

## Install paths expected by the code

After clone, run once:

```powershell
# Windows
.\scripts\setup_bundled_artifacts.ps1
```

```bash
# Linux/macOS
./scripts/setup_bundled_artifacts.sh
```

This copies/symlinks into `ml/runs/` where `docs/COMMANDS.md` and reports expect models.

## Rebuild from NIST (optional)

You still need a local NIST SQLite index (not in git). See `data/README_DATA.md` and
`docs/EXTERNAL_DATASETS.md`.
