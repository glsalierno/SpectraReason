# SDBS first batch (25–35 spectra)

Curated downloads only — **no bulk mirror**. Export **FTIR JCAMP** from [SDBS](https://sdbs.db.aist.go.jp) and record each compound’s SDBS page/ID in the manifest.

## Folder layout

```
data/external_sources/raw/sdbs/
  sdbs_download_manifest.csv
  nitro_positive/
  n_oxide_hard_negative/
  amide_positive/
  amide_hard_negative/
  siloxane_confounds/
```

After download, each `.jdx` (or `.dx`) file lives in the matching subfolder. Filename can match compound name (e.g. `nitrobenzene.jdx`).

## Manifest

Edit `sdbs_download_manifest.csv` for each row:

| Column | Purpose |
|--------|---------|
| `batch_folder` | Subfolder name (must match) |
| `compound_name` | Common name |
| `sdbs_id` | SDBS compound ID if shown on site |
| `sdbs_url` | Full URL to the SDBS record page |
| `local_filename` | Your saved file name (e.g. `nitrobenzene.jdx`) |
| `cas` | Optional CAS |
| `notes` | Transmission/KBr, quality, etc. |
| `download_date` | ISO date you obtained the file |
| `downloaded` | `y` when file is in place |

## Suggested compounds (first batch)

### `nitro_positive/` (10)

- nitrobenzene
- o-nitrotoluene
- m-nitrotoluene
- p-nitrotoluene
- m-nitroaniline
- p-nitroaniline
- nitromethane
- nitroethane
- 1-nitronaphthalene
- 2-nitrophenol

### `n_oxide_hard_negative/` (8)

- pyridine N-oxide
- quinoline N-oxide
- 4-methylpyridine N-oxide
- nitrosobenzene
- pyridine
- quinoline
- imidazole
- indole

### `amide_positive/` (6)

- acetamide
- benzamide
- nicotinamide
- caprolactam
- acetanilide
- formamide

### `amide_hard_negative/` (6)

- pyrrole
- indole
- carbazole
- succinimide
- phthalimide
- acrylamide

### `siloxane_confounds/` (5) — C–O / ether-ester (siloxane positives come later from open polymer)

- diethyl ether
- anisole
- phenetole
- ethyl acetate
- methyl benzoate

**Total:** 35 target rows in manifest (adjust if a compound has no FTIR on SDBS).

## After files are in place

```powershell
Set-Location "<FTIR_SVM_v5>"
$env:PYTHONPATH = (Get-Location).Path

python -m ml.external ingest-sdbs
python -m ml.external merge-indexes `
  data/experimental/examples_index.sqlite `
  data/experimental/sdbs_ir_index.sqlite `
  --out-db data/experimental/merged_external_index.sqlite
python -m ml.external dataset-qa --sqlite-index data/experimental/merged_external_index.sqlite
python -m ml.external summarize-confounder-coverage
```

Review:

- `reports/confounder_coverage_summary.md`
- `data/external_sources/raw/sdbs/sdbs_ingest_audit.json`
- Manifest `downloaded` column vs files on disk

## Next batch decision

After coverage summary:

| If gap is largest in… | Next batch focus |
|----------------------|------------------|
| `nitro_hn_n_oxide`, `nitro_hn_nitroso` | More N-oxides / nitroso on SDBS |
| `amide_hn_enamine`, `amide_hn_imide` | Enamines, imides, conjugated amides |
| `siloxane_positive`, `siloxane_hn_polymer_co` | Zenodo PDMS + nylon/epoxy ATR (`raw/open_polymer/`) |

Production `*_latest.joblib` stays frozen until gaps are closed and benchmarks pass.
