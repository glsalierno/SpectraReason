# Benchmark philosophy

SpectraReason benchmarks measure **interpretive correctness under ambiguity**,
not raw multiclass accuracy alone. A high SVM score on a crowded fingerprint is
less valuable than a guarded, explainable assignment.

## Core tenets

1. **Confounder-aware training** — negatives include chemically similar classes
   (nitro vs N-oxide, amide vs pyrrole/enamine, siloxane vs C–O ester ether).
2. **Hard negatives** — `hard-negative-mode on` up-weights false positives that
   survive naive band matching.
3. **Ambiguity preserved** — benchmarks score whether the report states
   *tentative* vs *supported*, not whether a single label wins.
4. **Explainability over score** — a correct tentative call beats a wrong
   supported call.

## Representative confounders

| Pair | Why it matters |
|------|----------------|
| **Nitro vs N-oxide** | Mid-IR strong bands overlap; requires asymmetric+symmetric NO₂ pairing |
| **Amide vs pyrrole/enamine** | N–H and C=O windows overlap in heterocycles |
| **Siloxane vs C–O** | ATR fingerprint Si–O can mimic ester/ether C–O |
| **Phenol vs alcohol** | Broad O–H; aromatic context needed |
| **ATR artifacts** | Baseline curvature and contact effects mimic moisture/OH |

## Evaluation layers

| Layer | Benchmark question |
|-------|-------------------|
| Band library | Are peaks linked to the right windows? |
| Rules | Are required/supporting bands enforced? |
| Guardrails v3 | Are competitors suppressed and artifacts down-weighted? |
| ML advisory | Does fusion **annotate** without overriding rules? |
| Front report | Is consensus prose honest about ambiguity? |

## Dataset construction

- **SMARTS weak labels** on structure-resolved NIST rows (`--require-structure`).
- **Family** and **specific** heads trained separately (`spectral+evidence_v2`).
- **Manifests** in `data/benchmark_sets/` describe held-out chemistries; spectra
  remain local.

## What we do not optimize for

- Single-number accuracy on imbalanced FG lists
- Label coverage at the expense of false supported calls
- Black-box fusion that hides rule failures

## Using benchmarks in the lab

1. Add a manifest row (chemistry, expected supported/tentative labels).
2. Run the spectrum through production defaults (`docs/PRODUCTION_DEFAULTS.md`).
3. Compare HTML consensus + CSV exports to manifest — file issues when guardrails
   or ruler text disagree with expert reading.

See also `docs/REPRODUCIBILITY.md` and `METHODS.md` for training methodology.
