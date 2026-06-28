# Reproducibility (FTIR_SVM_v5 production release)

## Environment

| Item | Requirement |
|------|-------------|
| **Python** | 3.10+ (3.11 recommended on Windows) |
| **Core packages** | `requirements.txt` at repo root |
| **PYTHONPATH** | Repository root (see below) |

```powershell
Set-Location "c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5"
$env:PYTHONPATH = (Get-Location).Path
python -m pip install -r requirements.txt
```

Optional dev tools: `pip install -r requirements-dev.txt` (includes `vulture`).

## Regenerate a production report

See `docs\COMMANDS.md` for the full front-facing command. Minimal smoke:

```powershell
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs "examples\spectra\Catechol-120-80-9-IR.jdx" `
  --ontology v4 --guardrails v3 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --fusion-mode annotate --ml-guardrails strict `
  --report-style product_v1 --report-audience front `
  --visual-theme matlab --show-region-ruler `
  --peak-sensitivity sensitive --show-weak-peaks `
  --out reports/reference_snapshots/front/REPORT.html
```

## Embedded report metadata

Each `product_v1` HTML report includes a collapsed **Reproducibility metadata** JSON block (Technical details) with:

- UTC generation timestamp
- Git commit (12-char) when `.git` is available
- Run settings line (ontology, guardrails, peak thresholds, theme, models)
- Model paths and SHA-256 prefixes
- Band library file hash
- Python + key package versions

Implementation: `reports/reproducibility_meta.py`.

## Verify model hashes

```powershell
python -c "from reports.reproducibility_meta import _sha256_file; from pathlib import Path; p=Path('ml/runs/struct_fg_family_v4_ontology_latest.joblib'); print(p, _sha256_file(p))"
```

Compare to values in `reports/release_stabilization_audit.md` or the report JSON block.

## Cite / document a run

Record in publications or lab notebooks:

1. Repository path and git commit from the report metadata block
2. `--report-audience`, `--visual-theme`, `--peak-sensitivity`, label thresholds
3. Family + specific joblib SHA-256 prefixes
4. NIST index path if ML training is referenced (`docs\COMMANDS.md`)

## Frozen in this release

- Ontology v4 + band library (Python module)
- v3 guardrails + conservative rules preset
- Production family/specific joblibs (paths above; files not moved)
- `product_v1` front/debug presentation contract
- Reference snapshots under `reports/reference_snapshots/`

Not frozen: experimental deconv model under `ml/runs/experiments/`.

## Reference snapshots

Regenerate all reference bundles:

```powershell
python scripts/release_stabilize.py --snapshots-only
```

See `reports/reference_snapshots/README.md` for spectra list and expected qualitative behavior.
