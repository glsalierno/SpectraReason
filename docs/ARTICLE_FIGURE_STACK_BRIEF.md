# Article FTIR stack figures (PDA / ODA POC)

Standalone article-export utility for multi-spectrum comparison stacks. **Does not modify** production report code (`structural_fg_svm_kronecker_report.py`).

> **Preferred (2026-06):** integrated chunk export in the main report CLI — see `reports/CURATION_AND_CHUNKS.md`.  
> POC article bundles: `C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\ftir figs\`  
> Combined stacks: `...\ftir figs\_combined_region_stacks\stacks\`

**Script:** `scripts/export_spectrum_stack.py`  
**Preprocessing:** `lib.ftir_foundation.preprocess_spectrum` (single source of truth)

---

## Purpose

Produce publication-ready **stacked spectrum figures** for the four POC article cases:

| ID | Sample | Input | Stack in %T? |
|----|--------|-------|--------------|
| `dopamine_powder` | Dopamine powder | `FTIR_POWDER\Dopamine_Powder.CSV` | Yes |
| `pda_eg_air_corrected` | PDA/EG (air corrected) | `FTIR_POWDER\pda_eg_con_new_minus_air_scaled.CSV` | No (absorbance diff) |
| `oda_ethanol` | ODA in ethanol | `PDA_ODA\ODA_in_Ethanol_blank_subtracted.CSV` | No (blank subtracted) |
| `pda_oda` | PDA–ODA | `PDA_ODA\pda_oda.CSV` | Yes |

Per-case SVM reports live under:

`C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\FTIR\{case}\`

Optional peak CSVs: `{case}\matlab_export\{stem}_peaks.csv`

---

## Two-output figure set (default run)

### 1. Normalized absorbance stack (primary article figure)

- **All four** spectra included.
- Each trace independently preprocessed and normalized to **[0, 1]**.
- Vertical offset: trace *i* at `i × offset_step` (default **1.15**).
- Right-side sample labels; **no dense peak labels** by default.
- Wavenumber range default **400–4000 cm⁻¹**, inverted x-axis.

Outputs:

- `article_ftir_stack_normalized_absorbance.png`
- `article_ftir_stack_normalized_absorbance.svg`
- `article_ftir_stack_normalized_absorbance.pdf`

### 2. Transmittance (%T) stack

- **Only true %T inputs:** Dopamine powder, PDA–ODA.
- Absorbance-difference and blank-subtracted spectra **excluded** (avoids chemically misleading mixed-axis plots).
- Offsets computed from each trace’s **in-window span** × `offset_step`.

Outputs:

- `article_ftir_stack_transmittance.png`
- `article_ftir_stack_transmittance.svg`
- `article_ftir_stack_transmittance.pdf`

### 3. Fingerprint zoom (normalized absorbance)

- Range default **400–1500 cm⁻¹** (fingerprint + mid-IR).
- Same offset policy as full-range normalized stack.
- Default: **unlabeled** (clean comparison).

Output:

- `article_ftir_stack_fingerprint_normalized_absorbance.png` (+ svg/pdf when `--formats` includes them)

---

## Label policy

| Mode | Behavior |
|------|----------|
| Default stack | No peak wavenumber labels; right-side trace IDs only |
| `--label-peaks` | Up to `--max-labels-per-trace` (default 5) diagnostic peaks per trace |
| Peak source | Report `*_peaks.csv` if present; else top heights from processed trace |
| Collision | Fingerprint clustering (18 cm⁻¹) + `apply_peak_label_layout` (smart) |
| Labeled output | Separate file: `*_labeled.{fmt}` (fingerprint zoom recommended) |

Production SVM reports keep separate spectrum / region-guide panels to avoid label overlap; stacks follow the same philosophy unless `--label-peaks` is set.

---

## CLI

```powershell
Set-Location "c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5"
$env:PYTHONPATH = (Get-Location).Path
$env:MPLBACKEND = "Agg"

python scripts/export_spectrum_stack.py `
  --out-root "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\FTIR\stacks" `
  --wn-min 400 --wn-max 4000 `
  --offset-step 1.15 `
  --dpi 300 `
  --formats png svg pdf
```

Optional flags:

| Flag | Purpose |
|------|---------|
| `--label-peaks` | Emit labeled fingerprint stack |
| `--max-labels-per-trace N` | Cap labels (default 5) |
| `--fingerprint-wn-min` / `--fingerprint-wn-max` | Fingerprint window (default 400–1500) |
| `--manifest JSON` | Override default case list |
| `--report-root DIR` | Base dir for auto-resolving `*_peaks.csv` |

---

## Offset model

Normalized absorbance (unit span per trace):

```
y_plot = y_norm + i * offset_step    # i = 0, 1, 2, 3
```

Transmittance (%T in window):

```
span_i = max(y_i) - min(y_i)
y_plot = y_i + sum(span_j * offset_step for j < i)
```

Right-side label anchor: `(wn_max - 0.02 * wn_span, offset_i + 0.5)`

---

## Visual theme

- House colors (MATLAB-inspired): blue `#0072bd`, orange `#d95319`, green `#77ac30`, purple `#7e2f8e`
- 300 dpi PNG default; vector via svg/pdf
- White background, light grid `#e6e6e6`
- y-axis ticks hidden on stacks (offsets are artificial)

---

## Invariants

- Do **not** fork preprocessing; use `preprocess_spectrum`.
- Do **not** retrain or overwrite `ml/runs/*_latest.joblib`.
- Scaled diff file: `pda_eg_con_new_minus_air_scaled.CSV` (precomputed; sample/blank grids differ by 1 point).
- Deliverables use **full absolute paths** in logs and `stack_manifest.json`.

---

## Related

- `reports/CURATION_AND_CHUNKS.md` — **integrated** chunk stacks, collages, curation (preferred)
- `scripts/export_paper_spectra.py` — single-spectrum transmittance + normalized panels
- `reports/FIGURES_AND_EXPORT.md` — production report static export
- `reports/annotation_layout.py` — peak label collision layout
