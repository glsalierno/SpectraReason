@echo off
REM Side-by-side lean reports: evidence-only vs legacy SVM (annotate)
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python reports/structural_fg_lean_report.py compare ^
  --inputs examples/spectra/Dopamine_Powder.CSV ^
  --out-dir reports/lean_compare ^
  --write-json
echo.
echo Open: %CD%\reports\lean_compare\index.html
pause
