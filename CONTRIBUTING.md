# Contributing to SpectraReason

Thank you for helping improve evidence-first FTIR interpretation. This project
prioritizes **explainability**, **reproducibility**, and **safe collaboration**.

## Before you start

1. Read `README.md` and `docs/COLLABORATOR_QUICKSTART.md`.
2. Never commit proprietary spectra, licensed libraries, or absolute local paths.
3. Keep production defaults in `docs/PRODUCTION_DEFAULTS.md` unless you are
   running an explicit experiment under `ml/runs/experiments/` (local only).

## Development setup

```bash
git clone <private-repo-url> SpectraReason
cd SpectraReason
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH="$(pwd)"   # Windows: $env:PYTHONPATH = (Get-Location).Path
```

## What to commit

| OK | Avoid |
|----|--------|
| Python source, tests, configs | `*.joblib`, `*.npz`, `*.sqlite` |
| `examples/spectra/` demo files | `data/experimental/` |
| `docs/`, report templates | Generated `reports/output_*` |
| `reports/reference_snapshots/*.html` | Personal OneDrive paths |
| Benchmark manifests (JSON/CSV) | Raw NIST or vendor libraries |

## Pull request checklist

- [ ] `python -m pytest ml/tests/ -q` passes
- [ ] No absolute paths, usernames, or `file://` links in committed HTML/MD
- [ ] Front-facing report smoke test if you touched `reports/` or `ml/ftir_*`
- [ ] Updated `docs/COMMANDS.md` if CLI flags changed
- [ ] Vulture findings reviewed (do not delete dynamic report hooks)

## Code style

- Match existing module layout (`ml/`, `reports/`, `lib/`).
- Prefer extending ontology/rules/ML layers separately — do not fold ML scores
  into rule conclusions when `fusion-mode annotate` is the production default.
- Comments only for non-obvious spectroscopy or guardrail rationale.

## Reporting issues

Open a private GitHub issue with: spectrum format, anonymized settings line,
and whether the problem is rules, guardrails, ML advisory, or presentation.
