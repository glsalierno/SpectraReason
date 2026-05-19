param(
    [string[]]$Inputs = @("examples/spectra/*"),
    [string]$OutDir = "reports/output_interactive"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = (Get-Location).Path
$paths = @()
foreach ($pat in $Inputs) {
    $paths += @(Get-ChildItem -Path $pat -File | ForEach-Object { $_.FullName })
}
if (-not $paths.Count) { throw "No input spectra matched: $Inputs" }
python reports/structural_fg_svm_kronecker_report.py batch `
  --inputs @paths `
  --model models/struct_fg_v7_pubchem_mordred.joblib `
  --out-dir $OutDir `
  --title "Structural FG SVM — interactive Kronecker"
