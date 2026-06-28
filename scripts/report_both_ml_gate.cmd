@echo off
REM Optional: evidence + legacy basic + gate fusion (when basic/subtle joblibs exist, edit paths)
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python reports/structural_fg_svm_kronecker_report.py batch --inputs examples/spectra/Dopamine_Powder.CSV --ml-mode both --fusion-mode gate --model models/struct_fg_v7_pubchem_mordred.joblib --include-evidence --include-ml --include-consensus --export-csv reports/export_both_demo/csv --out reports/export_both_demo
pause
