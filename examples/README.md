# Example spectra (collaborator-safe)

Files in `examples/spectra/` are **literature / reference** spectra (JCAMP-DX and similar)
for demos, tests, and reference snapshots. **No lab powder libraries** are committed.

## Included chemistries

| File | Teaching point |
|------|----------------|
| `Catechol-120-80-9-IR.jdx` | Aromatic + phenolic O–H |
| `Benzoic acid - 65-85-0-IR.jdx` | Carboxylic acid (not nitro) |
| `1H-Indol-5-ol-1953-54-4-IR.jdx` | Heteroaromatic + O–H ambiguity |
| `Indole_120-72-9-IR.jdx` | Aromatic N–H region |
| `1H-Pyrrole-2-carboxylic acid-634-97-9-IR.jdx` | Amide / pyrrole overlap |
| `Pyrrole_109-97-7-IR.jdx` | Heteroaromatic baseline |
| `Nylon_T.CSV` | Amide I/II (generic nylon reference) |
| `DopamineHCl_B6008951-IR.jdx` | Public-style reference JDX (not lab powder CSV) |

## Lab / proprietary data (not in git)

Place local FTIR powder measurements under `data/experimental/` (gitignored), e.g.:

- PDA, polydopamine, or custom polymer ATR/CSV exports
- Never commit `FTIR_POWDER/` paths or vendor `.SPA` archives

## Screenshots

Pre-rendered report previews: `docs/assets/screenshots/`.
