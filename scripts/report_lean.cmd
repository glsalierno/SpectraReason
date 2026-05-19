@echo off
REM Evidence-only lean report (use report_lean_compare.cmd for evidence vs SVM pair)
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python reports/structural_fg_lean_report.py batch --inputs examples/spectra/Catechol-120-80-9-IR.jdx --ml-mode none --out-dir reports/lean_demo
echo Report: %CD%\reports\lean_demo\REPORT.html
pause

