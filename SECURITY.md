# Security Policy

## Supported versions

SpectraReason is maintained as a **private research repository**. Security fixes
apply to the current `main` branch used for production v5 reports.

## Reporting a vulnerability

If you discover a security issue (e.g. path traversal in report export, unsafe
deserialization, or credential leakage in logs):

1. **Do not** open a public issue with exploit details.
2. Email the repository maintainers privately with steps to reproduce.
3. Allow reasonable time for a fix before disclosure.

## Data handling

- Do not commit API keys, `.env` files, or licensed spectral databases.
- PubChem caches may contain public structure data only; keep caches local
  (`ml/runs/` or `data/training/`, both gitignored).
- HTML reports embed spectrum arrays for interactivity; share reports only with
  collaborators cleared for the underlying experimental data.

## Dependencies

Keep `requirements.txt` pinned at the minor version level where practical.
Run `pip audit` periodically in your virtual environment.
