# External FTIR datasets

Expand training/reference diversity using **legally accessible** sources only. Production v4 NIST models remain frozen until external data passes QA and benchmark review.

## Allowed

| Source | License (summary) | Ingestion |
|--------|-------------------|-----------|
| SDBS (AIST Japan) | Research use, cite AIST; no bulk redistribution | User-downloaded JCAMP → `ingest-sdbs` |
| Zenodo / university open FTIR | Per-record (often CC BY) | Local download → `ingest-open-polymer` |
| Raman Open Database | Open terms on site | JCAMP adapter when FTIR available |
| User JCAMP / CSV | User license | `--library-path` plugin |
| NIST NistChemData | Public domain (US Gov) | Existing production indexer (frozen) |

## Not allowed

- Scraping or redistributing **Wiley KnowItAll**, **Sadtler**, or other proprietary commercial libraries
- Committing proprietary spectral files to git
- Automated bulk download of SDBS against their terms

## Provenance philosophy

Every ingested spectrum stores in `metadata_json`:

- `source_id`, `source_name`, `source_license`
- `original_identifier`, `ingestion_date`
- `preprocessing_version` (matches `lib.ftir_foundation.preprocess_spectrum`)
- `dataset_tier` = `experimental` until promotion
- `dataset_tags` for confounder coverage

Reports in **debug** audience can surface provenance when metadata is present.

## Directory layout

```
data/external_sources/     # registry, README, raw downloads
data/experimental/         # SQLite indexes (not production)
data/benchmark_sets/       # confounder JSON subsets
ml/external/               # ingestion adapters
ml/dataset_quality.py      # QA audits
ml/runs/experimental/      # NPZ + models (never *_latest)
```

## Add a new source

1. Verify license and redistribution terms.
2. Add an entry to `data/external_sources/source_registry.json`.
3. Implement or reuse an importer in `ml/external/` (prefer `import_jcamp_folder` / `import_csv_bundle`).
4. Place raw files under `data/external_sources/raw/<source_id>/`.
5. Run ingestion → `dataset-qa` → `build-confounder-benchmarks`.
6. Document commands in `docs/COMMANDS.md`.
7. Update `reports/external_dataset_expansion_audit.md` after first ingest.

## Plugin architecture

```powershell
python -m ml.external ingest-jcamp-folder `
  --out-db data/experimental/my_lib.sqlite `
  --library-path "D:\MyLicensedLibrary\jcamp" `
  --library-source jcamp `
  --source-id user_licensed_local
```

`library_source` values: `jcamp`, `csv`, `sqlite` (passthrough copy).

## Promotion to production

1. Merge indexes → `merge-indexes`
2. `dataset-qa` — resolve blocking flags
3. Review `data/benchmark_sets/*.json`
4. `build-external-dataset` → train under `ml/runs/experimental/` with `--no-update-latest`
5. Regression reports on `examples/spectra` + powder CSVs
6. Manual sign-off before copying artifacts to `ml/runs/*_latest.joblib`

## Preprocessing

All importers call the same pipeline as NIST indexing:

`read_spectrum` / JCAMP parse → `preprocess_spectrum` → `prepare_nist_ftir_cm1` → SQLite `wn_json` / `y_json`.

Do not fork preprocessing for external data.
