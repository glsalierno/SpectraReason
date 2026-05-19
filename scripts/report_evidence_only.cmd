@echo off
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python reports/structural_fg_svm_kronecker_report.py batch --inputs examples/spectra/Dopamine_Powder.CSV --ml-mode none --include-evidence --no-include-ml --out reports/evidence_only_demo
echo Report: %CD%\reports\evidence_only_demo\REPORT.html
pause
