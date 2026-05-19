# Reproducibility (SpectraReason v5)

## Environment

| Item | Requirement |
|------|-------------|
| **Python** | 3.10+ (3.11 recommended on Windows) |
| **Core packages** | `requirements.txt` at repo root |
| **PYTHONPATH** | Repository root |
| **RDKit** | Required for SMARTS weak-label training |
| **Kaleido** | Optional (`pip install kaleido`) for Plotly static PNG |

```bash
cd SpectraReason
export PYTHONPATH="$(pwd)"
python -m pip install -r requirements.txt
```

Optional dev: `pip install -r requirements-dev.txt`.

## Frozen production defaults

| Component | Version / path |
|-----------|----------------|
| Ontology | **v4** (`ml/ftir_ontology.py`) |
| Band library | **v4** (`ml/ftir_band_library.yaml` + Python module) |
| Guardrails | **v3** (`ml/ftir_guardrails.py`) |
| Rules preset | **conservative** (`configs/rule_presets/conservative.json`) |
| Report contract | **product_v1** |
| Feature set (ML) | `spectral+evidence_v2` (434-D) |

See [`PRODUCTION_DEFAULTS.md`](PRODUCTION_DEFAULTS.md).

## Production models (local artifacts)

Not committed to git. Expected paths after training or maintainer handoff:

- `ml/runs/struct_fg_family_v4_ontology_latest.joblib`
- `ml/runs/struct_fg_specific_v4_ontology_latest.joblib`

Verify hashes:

```bash
python -c "from reports.reproducibility_meta import _sha256_file; from pathlib import Path; p=Path('ml/runs/struct_fg_family_v4_ontology_latest.joblib'); print(p, _sha256_file(p))"
```

## Regenerate a production report

```bash
python reports/structural_fg_svm_kronecker_report.py batch \
  --inputs examples/spectra/Catechol-120-80-9-IR.jdx \
  --ontology v4 --guardrails v3 --ml-mode both \
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib \
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib \
  --fusion-mode annotate --ml-guardrails strict \
  --report-style product_v1 --report-audience front \
  --visual-theme matlab --show-region-ruler \
  --out reports/reference_snapshots/front/REPORT.html
```

## Embedded report metadata

Each `product_v1` HTML report includes a collapsed **Reproducibility metadata** JSON block with:

- UTC generation timestamp
- Git commit (12-char) when `.git` is available
- Run settings line (ontology, guardrails, peak thresholds, theme, models)
- Model paths and SHA-256 prefixes
- Band library file hash
- Python + key package versions

Implementation: `reports/reproducibility_meta.py`.

## Model / retraining philosophy

- **Rules are primary** for supported vs tentative calls in production (`fusion-mode annotate`).
- **ML retrains** require NIST index + PubChem cache locally; outputs stay in `ml/runs/`.
- **Experiments** never overwrite `*_latest.joblib` without maintainer review (`--no-update-latest`).
- **Deconv models** under `ml/runs/experiments/` are not production-frozen.

## Benchmark strategy

Confounder-aware SMARTS labels, hard negatives, and explainability-first evaluation.
See [`BENCHMARK_PHILOSOPHY.md`](BENCHMARK_PHILOSOPHY.md).

## External dataset policy

No proprietary redistribution through git. See [`EXTERNAL_DATASETS.md`](EXTERNAL_DATASETS.md).

## Reference snapshots

```bash
python scripts/release_stabilize.py --snapshots-only
```

Catalog: `reports/reference_snapshots/README.md`.

## Cite a run

Record from the report metadata block:

1. Git commit hash
2. `--report-audience`, `--visual-theme`, `--peak-sensitivity`
3. Family + specific joblib SHA-256 prefixes
4. NIST index build ID if ML training is referenced
