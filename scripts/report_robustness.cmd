@echo off
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python reports/structural_fg_svm_robustness_report.py batch --inputs examples/spectra/Dopamine_Powder.CSV --model models/struct_fg_v7_pubchem_mordred.joblib --out-dir reports/structural_fg_svm_robustness_smoke
echo.
echo HTML: %CD%\reports\structural_fg_svm_robustness_smoke\REPORT.html
pause
