# Data layout (SpectraReason)

| Path | In git? | Purpose |
|------|---------|---------|
| `examples/spectra/` | Yes | Collaborator-safe demo CSV/JDX |
| `data/benchmark_sets/` | Manifests only | Benchmark chemistry lists |
| `data/training/bundled/` | **Yes (LFS)** | Production SVMs, NIST NPZ matrices, PubChem cache |
| `data/external_sources/raw/` | **No** (gitignored) | Downloaded archives (NIST, Zenodo, …) |
| `data/experimental/` | **No** (gitignored) | Lab-only spectra |
| `ml/runs/` | **No** (scratch) | Installed copies after `setup_bundled_artifacts.*` |

## Bundled ML (ready after clone)

See [`training/bundled/README.md`](training/bundled/README.md) and [`docs/ML_ARTIFACTS.md`](../docs/ML_ARTIFACTS.md).

| Bundle | Feature dim | Notes |
|--------|-------------|-------|
| `bundled/v4_production/` | 434-D `spectral+evidence_v2` | Production family + specific SVMs |
| `bundled/v7_mordred/` | 303-D spectral+RDKit+Mordred | Legacy v7 reports |

Run once after clone:

```powershell
.\scripts\setup_bundled_artifacts.ps1
```

## NIST index (external, optional for rebuild)

Only needed to **rebuild** training matrices from SQLite, not for bundled inference/retrain.

```bash
mkdir -p data/external
ln -s /path/to/nistchemdata_ir_index_v7_fresh.sqlite data/external/nist_index.sqlite
```

Windows (PowerShell, admin for symlink):

```powershell
New-Item -ItemType Directory -Force data/external
cmd /c mklink data\external\nist_index.sqlite C:\path\to\nistchemdata_ir_index_v7_fresh.sqlite
```

Never commit the SQLite file. See `docs/EXTERNAL_DATASETS.md`.
