@echo off
REM Quick basic SVM train from bundled v7 NPZ (no NIST rebuild). ~10-20 min on full set.
cd /d "%~dp0.."
set PYTHONPATH=%CD%
if not exist "data\training\struct_fg_v7_pubchem_mordred.meta.json" (
  if exist "models\struct_fg_v7_pubchem_mordred.meta.json" (
    copy /Y "models\struct_fg_v7_pubchem_mordred.meta.json" "data\training\struct_fg_v7_pubchem_mordred.meta.json"
  )
)
if not exist "ml\runs" mkdir "ml\runs"
python -m ml.structural_fg_svm train --dataset-prefix data/training/struct_fg_v7_pubchem_mordred --model-kind basic --remap-legacy-labels --min-label-positives 30 --model-out ml/runs/struct_fg_basic_smoke.joblib --n-jobs 1
pause
