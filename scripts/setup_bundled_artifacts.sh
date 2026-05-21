#!/usr/bin/env bash
# Copy bundled ML artifacts into ml/runs/ for production commands.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLED="$ROOT/data/training/bundled"
RUNS="$ROOT/ml/runs"
mkdir -p "$RUNS"
install() { cp -f "$BUNDLED/$1" "$RUNS/$2"; echo "Installed $2"; }
install v4_production/struct_fg_family_v4_ontology_latest.joblib struct_fg_family_v4_ontology_latest.joblib
install v4_production/struct_fg_specific_v4_ontology_latest.joblib struct_fg_specific_v4_ontology_latest.joblib
install v4_production/ds_v4_family_spectral_evidence_v2_nist.npz ds_v4_family_spectral_evidence_v2_nist.npz
install v4_production/ds_v4_family_spectral_evidence_v2_nist.meta.json ds_v4_family_spectral_evidence_v2_nist.meta.json
install v4_production/ds_v4_specific_spectral_evidence_v2_nist.npz ds_v4_specific_spectral_evidence_v2_nist.npz
install v4_production/ds_v4_specific_spectral_evidence_v2_nist.meta.json ds_v4_specific_spectral_evidence_v2_nist.meta.json
install v4_production/pubchem_train_writable.json pubchem_train_writable.json
echo "Done. export PYTHONPATH=$ROOT"
