# External datasets

SpectraReason ingests **public or lab-licensed** reference data through a
plugin-style library architecture. **This repository does not redistribute**
NIST indexes, vendor libraries, or proprietary experimental archives.

## Design principles

1. **Provenance first** — every ingested spectrum records source, license, and
   retrieval date in dataset metadata JSON.
2. **Local indexing** — large SQLite/JCAMP archives stay on disk outside git.
3. **Structure enrichment** — PubChem PUG REST with on-disk cache (gitignored).
4. **No surprise commits** — `data/external_sources/raw/` and `*.sqlite` are
   gitignored; only manifests and ingestion code ship in git.

## Supported workflows

### NIST Chemistry WebBook (via NistChemData mirror)

1. Download the community mirror archive to `data/external_sources/raw/` (local).
2. Build or symlink a SQLite index (see `METHODS.md`, `data/README_DATA.md`).
3. Point `build-dataset` at the index path:

```bash
python -m ml.structural_fg_svm build-dataset \
  --nist-index data/external/nist_index.sqlite \
  --out-prefix ml/runs/ds_v4_family_spectral_evidence_v2_nist \
  --model-kind family --label-source smarts \
  --feature-set spectral+evidence_v2 --ontology v4 \
  --require-structure --enrich-pubchem
```

**Legal:** NIST data are public domain; respect NIST/NistChemData terms and do not
commit the SQLite file.

### SDBS (AIST Japan)

Use for spot-checking aromatic/heteroaromatic benchmarks. Ingest JCAMP into the
same preprocessing pipeline; store raw files under `data/external_sources/raw/sdbs/`
(gitignored). Document CAS/name in manifest CSV under `data/benchmark_sets/`.

### Zenodo / open polymer corpora

1. Record DOI, version, and license in `data/benchmark_sets/<name>_manifest.json`.
2. Convert to internal NPZ only on developer machines (`ml/runs/`, gitignored).
3. Ship **manifest + evaluation scripts**, not the corpus itself.

## Plugin library architecture

| Component | Role |
|-----------|------|
| `ml/ftir_band_library.yaml` | Wavenumber windows and band semantics (versioned) |
| `ml/ftir_ontology.py` | Label taxonomy (family / specific / motif / artifact) |
| `ml/fg_smarts_library.py` | SMARTS weak labels when structure resolves |
| `ml/structural_fg_svm.py` | Dataset build + train CLI |
| `configs/rule_presets/` | Conservative vs sensitive evidence thresholds |

Adding a new external source means: indexer → metadata schema → optional SMARTS
enrichment → benchmark manifest — **not** copying spectra into git.

## What collaborators must not commit

- NIST `*.sqlite` indexes
- Vendor ATR libraries
- Customer polymer powder CSVs
- PubChem caches > few MB (use local `ml/runs/pubchem_train_writable.json`)

## Redistribution policy

Publications may **cite** NIST/WebBook and Zenodo DOIs. HTML reports generated
from lab spectra are **not** redistributable unless the underlying data license
allows it. When in doubt, export band tables and figures without embedded arrays.
