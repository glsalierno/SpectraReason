# Reference snapshots (production defaults)

Workspace: `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4`

## Spectra

| ID | Path | Role |
|----|------|------|
| catechol | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\Catechol-120-80-9-IR.jdx` | canonical |
| nylon_amide | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\Nylon_T.CSV` | canonical |
| benzoic_acid | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\Benzoic acid - 65-85-0-IR.jdx` | canonical |
| pyrrole_carboxylic | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\1H-Pyrrole-2-carboxylic acid-634-97-9-IR.jdx` | canonical |
| indol_5_ol | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\1H-Indol-5-ol-1953-54-4-IR.jdx` | canonical |
| indole | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\Indole_120-72-9-IR.jdx` | canonical |
| polydopamine_powder | `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\examples\spectra\Polydopamine_Powder.CSV` | canonical |
| pda_eg | `<RESEARCH_ROOT>\AI\AT-10\PDA\FTIR_POWDER\pda_eg_con_new.CSV` | canonical |

## Commands

```powershell
Set-Location "<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4"
$env:PYTHONPATH = (Get-Location).Path
python scripts/release_stabilize.py --snapshots-only
```

## Outputs

- Front: `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\reports\reference_snapshots\front\REPORT.html`
- Debug: `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\reports\reference_snapshots\debug\REPORT.html`
- Static figures: `<RESEARCH_ROOT>\AI\AT-10\PDA\chunks\FTIR_SVM_v4\reports\reference_snapshots\static_figures`

## Expected qualitative behavior

- **Catechol / indole / pyrrole:** aromatic + O–H/N–H; heteroaromatic cautions; no supported nitro from mid-region alone
- **Nylon:** amide I/II pattern; amide supported when paired bands present
- **PDA / polydopamine / pda_eg:** broad O–H/N–H, aromatic; siloxane not supported without Si–O dominance
- **Benzoic acid:** carboxylic C=O/O–H; not confused with nitro

## Known ambiguities

- 1450–1650 cm⁻¹: C=C vs amide II vs heterocyclic N–O (ruler + guardrails)
- ATR fingerprint: C–O vs Si–O overlap without siloxane call
- Pyrrole-carboxylic: amide/enamine/pyrrole overlap cards in front mode
