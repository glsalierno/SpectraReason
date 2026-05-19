# v3_guarded evidence-first FTIR pipeline

## Motivation

Broad FTIR fingerprint and C–O regions produce **overlapping** evidence. A single band or a single high-SVM head is easy to misread (e.g. siloxane vs aryl ether, nitrile vs noise, nitro vs aromatic fingerprint). **v3_guarded** generalizes the earlier **paired-band nitro** idea: use **evidence networks**, **soft competitor suppression**, **ratio and artifact context**, and **explicit confidence classes**—without hard-zeroing weak but real chemistry.

## False-positive classes addressed

- **Single-band / low-specificity overlap** (siloxane, ether, ester, phenol, nitrile, …)
- **Incomplete motifs** (ester without C=O+C–O, amide without I+NH context, acid without broad OH, phenol without aromatic+aryl C–O)
- **Cross-family competition** (siloxane vs organic C–O; nitrile vs alkyne; amide vs amine)
- **Artifacts** (moisture-like OH, CO₂ region, fingerprint crowding, baseline tilt)

## Design

| Layer | Role |
|--------|------|
| `ml/ftir_evidence.py` | Peak **quality** proxies (sharpness, isolation, SNR-like) for sharp-band labels |
| `ml/ftir_artifacts.py` | Soft **artifact flags** + cautions |
| `ml/ftir_guardrails.py` | **v3** caps, competitors, ambiguity **fallback labels** |
| `ml/ftir_rules.py` | Rule scoring + `guardrails_mode`: `none` / `v2` / `v3` |
| `ml/ftir_ml_refinement.py` | `--ml-guardrails strict|moderate|off` caps ML fusion when evidence is weak |
| Reports | Kronecker CLI: `--guardrails`, sections C–D for diagnostics + ambiguity |

**Soft suppression:** scores are multiplied down toward a **floor** (0.10) when a strong **competitor** exists and required evidence is incomplete—never silently raised by ML alone in **strict** mode.

## Versioning and artifacts

- Pipeline / rule version string: **`v3_guarded`** (returned on rule results when `--guardrails v3`).
- Trained SVM naming: use `python -m ml.structural_fg_svm train ... --version v3_guarded` (alias of `--pipeline-version`) to emit e.g. `struct_fg_basic_v3_guarded_<UTCtimestamp>.joblib` under `ml/runs/` (or `--out`).
- HTML reports: use `--out` under `reports/v3_guarded_<run>/REPORT.html` (full path on your machine).

## CLI (Kronecker interactive report)

Defaults match production-minded settings:

| Flag | Default | Meaning |
|------|---------|---------|
| `--guardrails` | `v3` | `v3` = v3_guarded; `v2` = confidence classes only; `none` = legacy rule scores only |
| `--ml-guardrails` | `strict` | Limits ML-weighted / gate / ml_only promotion when evidence is tentative |
| `--show-ambiguity-labels` | on | Family fallbacks (e.g. `hydroxy_containing`) |
| `--show-artifact-flags` | on | Artifact / interference block |

### Evidence-only (recommended production)

```text
python reports/structural_fg_svm_kronecker_report.py batch ^
  --inputs "c:\path\to\data\*.CSV" ^
  --ml-mode none ^
  --guardrails v3 ^
  --rules-preset conservative ^
  --show-ambiguity-labels ^
  --show-artifact-flags ^
  --include-evidence --include-consensus --no-include-ml ^
  --export-csv "c:\path\to\reports\v3_guarded_production\csv" ^
  --out "c:\path\to\reports\v3_guarded_production"
```

### ML-assisted (secondary)

```text
python reports/structural_fg_svm_kronecker_report.py batch ^
  --inputs "c:\path\to\data\*.CSV" ^
  --ml-mode basic ^
  --basic-model "c:\path\to\FTIR_SVM_v2\ml\runs\struct_fg_basic_v3_guarded_<timestamp>.joblib" ^
  --fusion-mode annotate ^
  --ml-guardrails strict ^
  --guardrails v3 ^
  --rules-preset conservative ^
  --show-ambiguity-labels ^
  --show-artifact-flags ^
  --include-evidence --include-ml --include-consensus ^
  --export-csv "c:\path\to\reports\v3_guarded_ml\csv" ^
  --out "c:\path\to\reports\v3_guarded_ml"
```

## Train with v3 artifact tag

```text
cd SpectraReason   # repository root; export PYTHONPATH=.
set PYTHONPATH=%CD%
python -m ml.structural_fg_svm train --dataset-prefix data\training\struct_fg_v7_pubchem_mordred ^
  --model-kind basic --remap-legacy-labels --min-label-positives 20 ^
  --version v3_guarded --calibration sigmoid --split molecule --out ml\runs
```

`--hard-negative-mode` is reserved (recorded in training metadata; mining not yet wired).

## Limitations

- Guardrails are **heuristic**; unusual polymers, salts, or heavy overlap can still confuse the library.
- **SMARTS/SVM heads** do not include every ambiguity fallback id—fallbacks are **report-level** interpretive labels.
- Peak “quality” metrics are **proxies**, not instrument-grade SNR.

## Tests

See `ml/tests/test_ftir_v3_guardrails.py` for synthetic paired-band, ratio, sharpness, ML strict cap, and v2 skip behaviors.
