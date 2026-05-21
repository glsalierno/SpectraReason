# Collaborator quickstart

Welcome to **SpectraReason**. This guide gets you from clone to a demo report in
under ten minutes without leaking proprietary data.

## 1. Clone the private repository

```bash
git clone <private-repo-url> SpectraReason
cd SpectraReason
```

Use GitHub **private** visibility and invite collaborators explicitly. Do not
fork proprietary spectra into a public repo.

## 2. Python environment

**Recommended:** Python 3.11 on Windows or Linux.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

**RDKit** (structure-aware training):

```bash
conda install -c conda-forge rdkit
```

**PYTHONPATH** must be the repository root:

```bash
export PYTHONPATH="$(pwd)"
# Windows: $env:PYTHONPATH = (Get-Location).Path
```

## 3. Git LFS and bundled ML artifacts

Large training files use **Git LFS**. After clone:

```bash
git lfs install
git lfs pull
```

Install production models into `ml/runs/` (one-time):

```powershell
# Windows
.\scripts\setup_bundled_artifacts.ps1
```

```bash
# Linux/macOS
./scripts/setup_bundled_artifacts.sh
```

This copies from `data/training/bundled/v4_production/`:

- `struct_fg_family_v4_ontology_latest.joblib`
- `struct_fg_specific_v4_ontology_latest.joblib`
- Homogenized NIST training matrices (`*.npz`) and PubChem cache

**v7 legacy (Mordred):** `data/training/bundled/v7_mordred/` — see [`docs/ML_ARTIFACTS.md`](ML_ARTIFACTS.md).

Rules-only smoke reports work without running the setup script (`--ml-mode none`).

## 4. Run the demo front-facing report

```bash
python reports/structural_fg_svm_kronecker_report.py batch \
  --inputs examples/spectra/Catechol-120-80-9-IR.jdx \
  --ontology v4 --guardrails v3 --ml-mode both \
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib \
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib \
  --fusion-mode annotate --ml-guardrails strict \
  --report-style product_v1 --report-audience front \
  --visual-theme matlab --show-region-ruler \
  --out reports/demo_front/REPORT.html
```

Open `reports/demo_front/REPORT.html` in a browser. Confirm:

- Peak hover shows local context
- Region ruler appears in the 1450–1650 cm⁻¹ window
- Consensus table distinguishes **supported** vs **tentative** labels

## 5. Where to put your spectra

| Location | Purpose |
|----------|---------|
| `examples/spectra/` | Only **sanitized, shareable** demo files |
| `data/experimental/` | Your lab spectra (**gitignored**) |
| `data/external_sources/raw/` | Downloaded vendor/NIST archives (**gitignored**) |

Never commit powder libraries, customer samples, or licensed databases.

## 6. Debug vs production

| Mode | Flag | Use when |
|------|------|----------|
| Production front | `--report-audience front` | Deliverables, slides |
| Debug / audit | `--report-audience debug` | Method development |
| Experiments | `ml/runs/experiments/` | New features, deconv models |

Production defaults: `docs/PRODUCTION_DEFAULTS.md`.

## 7. Benchmarks

Read `docs/BENCHMARK_PHILOSOPHY.md` before changing training data. Benchmark
**manifests** live under `data/benchmark_sets/`; spectral libraries stay local.

## 8. Tests

```bash
python -m pytest ml/tests/ -q
```

## 9. Reference snapshots

```bash
python scripts/release_stabilize.py --snapshots-only
```

Compare output to `reports/reference_snapshots/`.

## 10. Getting help

- Commands catalog: `docs/COMMANDS.md`
- External data policy: `docs/EXTERNAL_DATASETS.md`
- Issues: private GitHub tracker (no proprietary attachments)
