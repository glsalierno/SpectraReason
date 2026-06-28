# External FTIR data sources

Legally accessible, research-permitted spectral libraries **only**. Proprietary Wiley/KnowItAll/Sadtler bulk content must not be scraped or committed.

## Layout

| Path | Purpose |
|------|---------|
| `source_registry.json` | Licenses, URLs, ingestion status |
| `raw/sdbs/` | User-downloaded SDBS JCAMP exports (see `docs/SDBS_FIRST_BATCH.md`) |
| `raw/sdbs/sdbs_download_manifest.csv` | Per-compound SDBS ID/URL + download tracking |
| `raw/open_polymer/` | Zenodo/university polymer ATR downloads |
| `raw/user_jcamp/` | Private licensed libraries (gitignored) |
| `../experimental/` | SQLite indexes built by importers |

## Per-source checklist

For each source, record:

1. **Name** and **URL**
2. **License / terms** and whether **redistribution** is allowed
3. **Access date** when files were obtained
4. **Ingestion script** under `ml/external/`
5. **Processed output** SQLite under `data/experimental/`
6. **Quality notes** (ATR vs transmission, units, noise)

## Ingestion

```powershell
Set-Location "<FTIR_SVM_v5>"
$env:PYTHONPATH = (Get-Location).Path

python -m ml.external list-sources
python -m ml.external ingest-sdbs --raw-dir data/external_sources/raw/sdbs
python -m ml.external ingest-jcamp-folder --out-db data/experimental/user.sqlite --library-path data/external_sources/raw/user_jcamp --library-source jcamp
python -m ml.external merge-indexes data/experimental/sdbs_ir_index.sqlite data/experimental/open_polymer_ir_index.sqlite --out-db data/experimental/merged_external_index.sqlite
python -m ml.external dataset-qa --sqlite-index data/experimental/merged_external_index.sqlite
```

See `docs/EXTERNAL_DATASETS.md` and `docs/COMMANDS.md`.

## Production separation

Experimental indexes are **not** used for `*_latest.joblib` training until:

- `dataset-qa` passes
- Confounder benchmarks reviewed
- Regression vs production NIST models

Promotion writes to `ml/runs/` only after explicit sign-off (see `CANONICAL_OUTPUTS.md`).
