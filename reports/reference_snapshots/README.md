# Reference snapshots (production defaults)

Bundled HTML examples for **SpectraReason** production settings. Open in a
browser without a server.

## Spectra

| ID | Relative path | Role |
|----|---------------|------|
| catechol | `examples/spectra/Catechol-120-80-9-IR.jdx` | canonical |
| nylon_amide | `examples/spectra/Nylon_T.CSV` | canonical |
| benzoic_acid | `examples/spectra/Benzoic acid - 65-85-0-IR.jdx` | canonical |
| pyrrole_carboxylic | `examples/spectra/1H-Pyrrole-2-carboxylic acid-634-97-9-IR.jdx` | canonical |
| indol_5_ol | `examples/spectra/1H-Indol-5-ol-1953-54-4-IR.jdx` | canonical |
| indole | `examples/spectra/Indole_120-72-9-IR.jdx` | canonical |
| polydopamine_powder | `examples/spectra/Polydopamine_Powder.CSV` | canonical |
| dopamine_powder | `examples/spectra/Dopamine_Powder.CSV` | canonical |

## Regenerate

```bash
export PYTHONPATH="$(pwd)"
python scripts/release_stabilize.py --snapshots-only
```

## Outputs

- Front: `reports/reference_snapshots/front/REPORT.html`
- Debug: `reports/reference_snapshots/debug/REPORT.html`
- Static figures: `reports/reference_snapshots/static_figures/`

## Expected qualitative behavior

- **Catechol / indole / pyrrole:** aromatic + O–H/N–H; heteroaromatic cautions; no supported nitro from mid-region alone
- **Nylon:** amide I/II pattern; amide supported when paired bands present
- **PDA / polydopamine / dopamine powder:** broad O–H/N–H, aromatic; siloxane not supported without Si–O dominance
- **Benzoic acid:** carboxylic C=O/O–H; not confused with nitro

## Known ambiguities

- 1450–1650 cm⁻¹: C=C vs amide II vs heterocyclic N–O (ruler + guardrails)
- ATR fingerprint: C–O vs Si–O overlap without siloxane call
- Pyrrole-carboxylic: amide/enamine/pyrrole overlap cards in front mode
