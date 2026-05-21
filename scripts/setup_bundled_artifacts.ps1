# Copy bundled ML artifacts into ml/runs/ for production commands.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bundled = Join-Path $Root "data/training/bundled"
$Runs = Join-Path $Root "ml/runs"
New-Item -ItemType Directory -Force -Path $Runs | Out-Null

$maps = @(
    @("v4_production/struct_fg_family_v4_ontology_latest.joblib", "struct_fg_family_v4_ontology_latest.joblib"),
    @("v4_production/struct_fg_specific_v4_ontology_latest.joblib", "struct_fg_specific_v4_ontology_latest.joblib"),
    @("v4_production/ds_v4_family_spectral_evidence_v2_nist.npz", "ds_v4_family_spectral_evidence_v2_nist.npz"),
    @("v4_production/ds_v4_family_spectral_evidence_v2_nist.meta.json", "ds_v4_family_spectral_evidence_v2_nist.meta.json"),
    @("v4_production/ds_v4_specific_spectral_evidence_v2_nist.npz", "ds_v4_specific_spectral_evidence_v2_nist.npz"),
    @("v4_production/ds_v4_specific_spectral_evidence_v2_nist.meta.json", "ds_v4_specific_spectral_evidence_v2_nist.meta.json"),
    @("v4_production/pubchem_train_writable.json", "pubchem_train_writable.json")
)
foreach ($m in $maps) {
    $src = Join-Path $Bundled $m[0]
    $dst = Join-Path $Runs $m[1]
    if (-not (Test-Path $src)) { Write-Warning "Missing bundled file: $src"; continue }
    Copy-Item $src $dst -Force
    Write-Host "Installed $($m[1])"
}
Write-Host "Done. Set PYTHONPATH to repo root and run reports per README.md."
