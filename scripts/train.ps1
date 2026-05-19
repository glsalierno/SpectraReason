# Re-train OvR SVM from bundled NPZ
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = (Get-Location).Path
python -m ml.structural_fg_svm train `
  --dataset-prefix data/training/struct_fg_v7_pubchem_mordred `
  --model-out models/struct_fg_v7_pubchem_mordred.joblib `
  --n-jobs 1
