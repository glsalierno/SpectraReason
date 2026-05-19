@echo off
cd /d "%~dp0.."
set PYTHONPATH=%CD%
if not exist "ml\runs" mkdir "ml\runs"
python -m ml.structural_fg_svm train --dataset-prefix data/training/struct_fg_subtle --model-kind subtle --n-jobs 1
pause
