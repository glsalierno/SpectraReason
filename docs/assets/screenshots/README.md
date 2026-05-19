# Report screenshots

Curated static previews for README and collaborator onboarding.

| File | Description |
|------|-------------|
| `front_report.png` | Front-facing combined figure (Catechol) |
| `region_ruler.png` | FTIR region ruler panel (1450–1650 cm⁻¹) |
| `matlab_spectrum_peaks.png` | MATLAB-style spectrum + peak labels |

Interactive equivalents:

- Front: `reports/reference_snapshots/front/REPORT.html`
- Debug: `reports/reference_snapshots/debug/REPORT.html`

Regenerate PNGs:

```bash
export PYTHONPATH="$(pwd)"
python scripts/release_stabilize.py --snapshots-only --force-snapshots
cp reports/reference_snapshots/static_figures/presentation/figures/Catechol-120-80-9-IR_combined.png docs/assets/screenshots/front_report.png
```
