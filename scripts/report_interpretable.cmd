@echo off
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python reports/structural_fg_svm_kronecker_report.py batch --inputs examples/spectra/Dopamine_Powder.CSV --ml-mode legacy --model models/struct_fg_v7_pubchem_mordred.joblib --fusion-mode annotate --include-evidence --include-ml --include-consensus --out reports/svm_interpretable_smoke
echo.
echo Report: %CD%\reports\svm_interpretable_smoke\REPORT.html
pause
