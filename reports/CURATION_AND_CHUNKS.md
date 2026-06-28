# Interactive curation, manuscript figures, and spectral chunks

Manuscript-ready FTIR exports with **interactive peak curation**, **custom wavenumber ranges**, and **machine-readable chunk data** for overlay/stack/collage experiments.

**Main CLI:** `reports/structural_fg_svm_kronecker_report.py batch`  
**Does not retrain** or overwrite `ml/runs/*_latest.joblib`.

Preprocessing: `lib.ftir_foundation.preprocess_spectrum` (single source of truth).

## Intensity modes and transmittance rules

Classification: `lib/intensity_modes.py`

| Raw category | Native %T | Normalized absorbance | Apparent %T (`--allow-apparent-transmittance`) |
|--------------|-----------|----------------------|------------------------------------------------|
| `transmittance_percent` | Yes | Yes (A = −log10(T/100)) | N/A |
| `absorbance` | No | Yes | Optional T_app = 100·10^(−A), labeled **Apparent Transmittance (%)** |
| `absorbance_difference` | No | Yes | Optional with flag + warning |

Difference stems (`blank_subtracted`, `minus`, `diff`, `scaled`, …) and `ODA_in_Ethanol_blank_subtracted`, `pda_eg_con_new_minus_air_scaled` are forced to `absorbance_difference`.

CLI: `--force-intensity-mode`, `--allow-apparent-transmittance`, `--apparent-transmittance-label`.

When native %T is unavailable, the curation section shows an explanatory banner (and `--allow-apparent-transmittance` is documented there).

---

## Report styles

| Output | Purpose |
|--------|---------|
| `REPORT.html` | Full product report: Plotly spectrum, interactive curation, range editor, download links (no embedded static PNGs) |
| `MANUSCRIPT_REPORT.html` | Concise manuscript HTML with links to static figures and chunk exports |
| `presentation/paper_figures/` | Publication PNG/SVG/PDF: transmittance + normalized absorbance with horizontal leader-line labels |

---

## Static figure label policy

| Element | Static manuscript / chunk figures | Interactive curation |
|---------|-----------------------------------|----------------------|
| Spectrum line | Yes | Yes |
| Leader lines (#d95319) | Yes | Yes (Plotly) |
| Horizontal wavenumber labels | Yes | Yes |
| Orange peak-tip dots | **Off by default** (`--show-peak-markers false`) | Optional via **Show candidate markers** checkbox (default on) |

Leader lines and labels are always drawn when a peak is selected for labeling. Peak-tip markers are optional everywhere.

---

## Interactive peak curation

Enable with `--export-interactive-curation`.

**UI (per spectrum):**

- Click candidate markers to toggle inclusion in the label table.
- **Show candidate markers** checkbox (default checked): semi-transparent orange dots for clicking; labels and leader lines remain when hidden.
- **Manual peaks:**
  - Click directly on the spectrum trace, or
  - Type wavenumber + **Add peak**.
- Snap within ±15–30 cm⁻¹ to local max (absorbance) or min (transmittance); falls back to nearest point with `snap_status=nearest_point`.
- Download `{stem}_label_overrides.json` before closing the browser session.

**Manual peak fields:** `source=manual`, `show_label=true`, `mode`, `added_by` (`click`|`typed`), `requested_wavenumber_cm1`, `snapped_wavenumber_cm1`, `snap_window_cm1`, `snap_target`, `snap_status`, plus shift/label overrides.

**CSV outputs (under `presentation/paper_figures/`):**

- `{stem}_peaks_selected.csv`
- `{stem}_peaks_manual.csv`
- `{stem}_peaks_candidates.csv`

**Persistence:** Save overrides JSON into the bundle directory (same folder as `REPORT.html`). Re-run with `--apply-label-overrides` to regenerate static figures and chunk labels from saved curation.

---

## Custom wavenumber ranges / chunks

Default ranges (900–400 cm⁻¹ excluded from automatic labels unless manually forced):

| Name | cm⁻¹ window |
|------|-------------|
| `OH_NH_stretch` | 3000–3700 |
| `CH_stretch` | 2800–3000 |
| `C_O_aromatic_NH` | 1500–1800 |
| `ring_CN` | 1200–1500 |
| `CO_fingerprint` | 900–1200 |

**Modules:** `reports/discussion_regions.py`, `reports/range_editor.py`, `reports/chunk_export.py`

**Config file:** `{bundle}/stacks/ranges_config.json`

```json
{
  "range_set_name": "Custom FTIR discussion ranges",
  "ranges": [
    {
      "name": "OH_NH_stretch",
      "wn_min": 3000,
      "wn_max": 3700,
      "color": "#0072bd",
      "show_in_stacks": true,
      "label_policy": "selected_only"
    }
  ]
}
```

Min/max may be entered in either order; plots use IR convention (high wavenumber left).

**REPORT.html editor:** add / duplicate / delete ranges, edit name/bounds/color/stack visibility, download or load `ranges_config.json`. After editing, re-run batch with `--regions-file` or `--ranges-file` pointing at the saved JSON.

---

## Chunk export layout

Enable with `--export-region-stacks` (writes under `{bundle}/stacks/`).

| Artifact | Description |
|----------|-------------|
| `{range}_single_{mode}.png/svg/pdf` | One spectrum, one range (single-input batches only) |
| `{range}_{mode}_stack.png/svg/pdf` | Offset stack of all input spectra in range |
| `ranges_collage_{mode}.png/svg/pdf` | Multi-range side-by-side panel |
| `{range}_chunk_data.json` / `.csv` | Machine-readable spectra + selected peaks for overlay experiments |
| `ranges_config.json` | Active range set |
| `chunks_manifest.json` | Index of all chunk outputs |
| `CHUNKS_INDEX.md` | Human-readable index |

**Modes:** `normalized_absorbance`, `transmittance` (transmittance skipped when input is not valid %T).

**Offset model:** `offset_step = robust_span × (1 + offset_gap)` with default `offset_gap=0.15` (99th − 1st percentile span per trace).

**Combined multi-spectrum comparison:** run batch with all inputs and `--out ...\_combined_region_stacks\REPORT.html`.

---

## CLI flags (curation + chunks)

| Flag | Default | Purpose |
|------|---------|---------|
| `--export-paper-figures` | off | Manuscript static figures + `MANUSCRIPT_REPORT.html` |
| `--export-interactive-curation` | off | Plotly curation UI in `REPORT.html` |
| `--export-region-stacks` | off | Singles, stacks, collages, chunk data under `stacks/` |
| `--export-chunk-data` / `--no-export-chunk-data` | on when stacks enabled | `{range}_chunk_data.json/csv |
| `--chunk-collage` / `--no-chunk-collage` | on | Multi-range collage figures |
| `--chunk-modes` | `--stack-modes` | `normalized_absorbance` `transmittance` |
| `--regions-file` / `--ranges-file` | — | Custom `ranges_config.json` |
| `--offset-gap` | 0.15 | Stack vertical spacing |
| `--show-peak-markers` | **false** | Orange peak-tip dots on static/chunk figures |
| `--save-label-overrides` / `--no-save-label-overrides` | on | Write auto `{stem}_label_overrides.json` templates |
| `--apply-label-overrides` | off | Apply saved overrides to paper + chunk exports |
| `--label-overrides DIR` | bundle dir | Override JSON directory |

---

## Example: POC article bundle (single spectrum)

```powershell
Set-Location "c:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\chunks\FTIR_SVM_v5"
$env:PYTHONPATH = (Get-Location).Path

$csv = "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\csvs\Dopamine.CSV"
$out = "C:\Users\glsal\OneDrive - UMass Lowell\TURI\Research\AI\AT-10\PDA\POC_PDA_ODA article\ftir figs\Dopamine"

python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs $csv `
  --ontology v4 --ml-mode both `
  --family-model ml/runs/struct_fg_family_v4_ontology_latest.joblib `
  --specific-model ml/runs/struct_fg_specific_v4_ontology_latest.joblib `
  --report-audience front --visual-theme matlab `
  --export-paper-figures `
  --export-spectrum-feedback `
  --export-interactive-curation `
  --export-region-stacks `
  --export-chunk-data `
  --chunk-collage `
  --save-label-overrides `
  --label-overrides $out `
  --paper-out "$out\presentation\paper_figures" `
  --out "$out\REPORT.html"
```

After curating peaks in the browser and saving `{stem}_label_overrides.json`:

```powershell
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs $csv `
  ... `
  --apply-label-overrides `
  --label-overrides $out `
  --out "$out\REPORT.html"
```

After editing ranges in the HTML editor, download `ranges_config.json` and re-run with:

```powershell
  --regions-file "$out\stacks\ranges_config.json"
```

---

## Module map

| Module | Role |
|--------|------|
| `reports/paper_ftir_figures.py` | Manuscript transmittance + normalized absorbance figures |
| `reports/manuscript_report.py` | Lightweight `MANUSCRIPT_REPORT.html` |
| `reports/interactive_curation.py` | Plotly curation plots + table + manual peak UI |
| `reports/label_overrides.py` | Override JSON schema, auto templates, merge |
| `reports/peak_snap.py` | Manual peak snap to local extrema |
| `reports/discussion_regions.py` | Default ranges, load/save `ranges_config.json` |
| `reports/range_editor.py` | HTML range editor section + chunk file links |
| `reports/chunk_export.py` | Singles, stacks, collages, chunk data export |
| `reports/region_stack_export.py` | Shared trace prep + thin wrapper to `chunk_export` |

---

## Tests

| Test file | Focus |
|-----------|-------|
| `ml/tests/test_paper_ftir_figures.py` | Manuscript figures, leader labels, `--show-peak-markers` |
| `ml/tests/test_interactive_curation.py` | Curation HTML, range editor, marker toggle |
| `ml/tests/test_peak_snap.py` | Snap logic, manual peak merge |
| `ml/tests/test_chunk_export.py` | Chunk smoke + legacy artifact cleanup |

```powershell
python -m pytest ml/tests/test_paper_ftir_figures.py ml/tests/test_interactive_curation.py ml/tests/test_peak_snap.py ml/tests/test_chunk_export.py -q
```

---

## Related

- `reports/FIGURES_AND_EXPORT.md` — interactive HTML, static presentation PNGs, MATLAB path
- `docs/COMMANDS.md` — copy-paste command recipes
- `docs/ARTICLE_FIGURE_STACK_BRIEF.md` — standalone `scripts/export_spectrum_stack.py` (legacy article utility)
- `docs/CODEMAP.md` — full module ownership
