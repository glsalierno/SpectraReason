# Targeted external FTIR expansion

Designed **negative-space coverage** for three production pain points: nitro, amide, and siloxane. Expansion is measurable via `python -m ml.external summarize-confounder-coverage`.

**Do not retrain production models** until gaps below minimum counts are closed and benchmarks pass.

---

## Problem 1 — Nitro hard negatives

| Class ID | Role | Example compounds | Tag / keyword rule | SMARTS | Source | FTIR ambiguity |
|----------|------|-------------------|--------------------|--------|--------|----------------|
| `nitro_positive` | true positive | nitrobenzene, p-nitrotoluene, DNT, nitromethane | tag `nitro`; kw nitro/dinitro (exclude nitroso, n-oxide) | `[N+](=O)[O-]` | SDBS | ν_as ~1520–1550, ν_s ~1340–1370 |
| `nitro_hn_n_oxide` | hard negative | pyridine 1-oxide, quinoline 1-oxide | tag `n_oxide` + `heteroaromatic` | `[n+][O-]` | SDBS | N–O overlaps nitro region; no symmetric NO₂ pair |
| `nitro_hn_nitroso` | hard negative | nitrosobenzene, NDMA | tag `nitroso` | `[NX2]=O` | SDBS | N=O ~1450–1500; single N–O |
| `nitro_hn_heteroaromatic` | hard negative | pyridine, pyrrole, indole | tag `heteroaromatic` | `a1naaaa1` | SDBS | Ring modes 1400–1600 crowd fingerprint |
| `nitro_hn_enamine` | hard negative | morpholine enamine | tag `enamine` | `[CX3]=[CX3][NX3]` | SDBS | C=C + C–N; no NO₂ |

**Minimum targets:** 10 nitro positives, 8 N-oxides, 5 nitroso, 10 heteroaromatic HN, 5 enamine HN.

**Manifest:** `data/benchmark_sets/nitro_vs_noxide_manifest.json`

---

## Problem 2 — Amide hard negatives

| Class ID | Role | Example compounds | Tag / keyword rule | SMARTS | Source | FTIR ambiguity |
|----------|------|-------------------|--------------------|--------|--------|----------------|
| `amide_positive` | true positive | acetamide, benzamide, caprolactam, nylon 6 | tag `amide` | `C(=O)N` | SDBS | Amide I ~1650–1680, II ~1550 |
| `amide_hn_enamine` | hard negative | morpholine enamine | tag `enamine` | `[CX3]=[CX3][NX3]` | SDBS | Weak/absent amide I |
| `amide_hn_pyrrole` | hard negative | pyrrole, indole, carbazole | kw pyrrol/indol | `[nH]1cccc1` | SDBS | N–H ~3200–3400; no amide I |
| `amide_hn_imide` | hard negative | phthalimide, succinimide | kw imide | `C(=O)NC(=O)` | SDBS | Twin carbonyls |
| `amide_hn_conjugated_amide` | hard negative | nicotinamide, acrylamide | tag `amide` + aromatic | `C(=O)Nc` | SDBS | Lowered amide I |

**Minimum targets:** 10 amide positives, 8 enamine, 8 pyrrole, 5 imide, 5 conjugated amide.

**Manifest:** `data/benchmark_sets/amide_vs_enamine_manifest.json`

---

## Problem 3 — Si–O hard negatives

| Class ID | Role | Example compounds | Tag / keyword rule | SMARTS | Source | FTIR ambiguity |
|----------|------|-------------------|--------------------|--------|--------|----------------|
| `siloxane_positive` | true positive | PDMS, HMDS, D4 cyclosiloxane | tag `siloxane` | `[Si][OX2][Si]` | Zenodo / open polymer | Si–O–Si ~1000–1100; CH₃ ~1260 |
| `siloxane_hn_ether_ester` | hard negative | diethyl ether, ethyl acetate, PEG | kw ether/ester | `[OD2]([#6])[#6]` | SDBS | C–O ~1050–1150 |
| `siloxane_hn_polymer_co` | hard negative | nylon, PET, PMMA, epoxy | tag `polymer` | — | open polymer ATR | C=O / C–O; no Si–O–Si |
| `siloxane_hn_atr_polymer` | supporting | nylon ATR, epoxy coating | tag `polymer` + `atr` | — | lab / Zenodo | Baseline drift, contact effects |

**Minimum targets:** 8 siloxane positives, 10 ether/ester, 8 C–O polymers, 6 ATR polymer supporting.

**Manifest:** `data/benchmark_sets/siloxane_vs_CO_manifest.json`

---

## Acquisition mode targets

| Mode | Why |
|------|-----|
| **ATR** | Primary lab data; siloxane & polymer confounders |
| **Transmission** | SDBS KBr / solution standards for nitro/amide positives |
| **Baseline drift / moisture_like** | Tag via spectral heuristics after ingest |

---

## Workflow

1. Download targeted SDBS / Zenodo subsets into `data/external_sources/raw/`.
2. Ingest: `python -m ml.external ingest-sdbs` (etc.).
3. Measure: `python -m ml.external summarize-confounder-coverage`
4. Review: `reports/confounder_coverage_summary.md`, `reports/external_dataset_expansion_audit.md`
5. Close gaps listed in manifest `coverage.gap` fields.
6. Only then: experimental NPZ + train under `ml/runs/experimental/` with `--no-update-latest`.

---

## Scientific justification for retraining

Retraining is justified when:

- Every `minimum_count` in manifests is met (or waived with documented reason).
- Hard-negative:class ratio ≥ 1:1 per problem for ingested external set.
- Benchmark qualitative review on `data/benchmark_sets/*.json` members passes.
- No regression on `examples/spectra` and powder CSVs vs production reports.

---

## Code references

- Class definitions: `ml/external/confounder_targets.py`
- Coverage CLI: `ml/external/summarize_confounder_coverage.py`
- Tagging heuristics: `ml/external/tagging.py`, `ml/external/ingest_common.py`
