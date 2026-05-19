# Release notes — SpectraReason v5 (production)

**Codename:** SpectraReason (private repo `SpectraReason`, formerly FTIR_SVM_v4 workspace)

## Highlights

### Evidence-first architecture

- Ontology **v4** separates local motifs, families, specific functional groups,
  artifacts, and fallback ambiguity buckets.
- Band library + regional statistics + artifact detectors feed rule assignment
  before any ML score is shown.

### Product front-facing reports (`product_v1`)

- Spectroscopist summary prose and key-evidence tables
- Collapsed reproducibility metadata (git hash, model SHA-256, band-library hash)
- MATLAB-style visual theme for publication figures

### Ambiguity-aware interpretation

- Explicit **supported / tentative / local_possible** classes
- Front consensus suppresses raw ontology spam; debug mode shows full diagnostics
- Region ruler for 1450–1650 cm⁻¹ (C=C vs amide II vs N–O)

### Guardrails v3

- Paired-band requirements (nitro, amide, ester, …)
- **NO₂ vs N-oxide** competitor logic
- **Si–O / ATR** overlap handling vs ether/ester calls

### Optional ML advisory

- Family + specific OvR SVMs with `fusion-mode annotate` (production default)
- `ml-guardrails strict` caps ML when rules disagree

### External ingestion framework

- NIST index workflow, Zenodo manifests, SDBS spot checks
- Plugin-style band library + SMARTS library (see `docs/EXTERNAL_DATASETS.md`)

### Exports

- Interactive Plotly HTML with local peak hover
- Static PNG bundle + `matlab_export/make_figures.m`
- Canonical peak model for consistent labeling

## Production defaults

| Setting | Value |
|---------|--------|
| ontology | v4 |
| guardrails | v3 |
| fusion | annotate |
| ml_guardrails | strict |
| rules_preset | conservative |
| report_style | product_v1 |
| report_audience | front |
| visual_theme | matlab |
| region_ruler | enabled |

## Upgrade notes for collaborators

- Place joblibs under `ml/runs/` locally (not in git).
- Regenerate reference snapshots: `python scripts/release_stabilize.py --snapshots-only`
- Run `python -m pytest ml/tests/ -q` after pulling

## Known experimental (non-production)

- Deconvolution-specific models under `ml/runs/experiments/`
- Peakcodebook feature ablations (`docs/DEPRECATED.md` lists legacy paths)
