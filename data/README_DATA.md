# Data layout (SpectraReason)

| Path | In git? | Purpose |
|------|---------|---------|
| `examples/spectra/` | Yes | Collaborator-safe demo CSV/JDX |
| `data/benchmark_sets/` | Manifests only | Benchmark chemistry lists |
| `data/external_sources/raw/` | **No** (gitignored) | Downloaded archives (NIST, Zenodo, …) |
| `data/experimental/` | **No** (gitignored) | Lab-only spectra |
| `data/training/` | Meta only | Legacy v7 NPZ/cache **not** committed by default |
| `ml/runs/` | **No** | Joblibs, NPZ training matrices, PubChem cache |

## NIST index (external)

Build or symlink locally:

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

## Legacy v7 training bundle (optional local)

| File | Approx. size | Purpose |
|------|--------------|---------|
| `training/struct_fg_v7_pubchem_mordred.npz` | ~19 MB | Historical v7 train matrix |
| `training/pubchem_structure_cache_v7.json` | ~44 MB | PubChem cache for `build-dataset` |

Obtain from maintainer release zip or rebuild from NIST. Not required for v4 production reports.
