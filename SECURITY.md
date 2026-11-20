# Security Policy

## Supported versions

SpectraReason is a public research repository. Security fixes apply to the
current `main` branch.

## Reporting a vulnerability

If you discover a security issue (e.g. path traversal in report export, unsafe
deserialization, or credential leakage in logs):

1. **Do not** open a public issue with exploit details.
2. Use [GitHub private vulnerability reporting](https://github.com/glsalierno/SpectraReason/security/advisories/new)
   if the repository offers it. Otherwise, contact a maintainer privately using
   a contact method on their GitHub profile. Include the affected version or
   commit, minimal reproduction steps, and the impact. Do not send credentials,
   unpublished spectra, or licensed datasets.
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
