@echo off
REM Optional: scan example spectra, write calibration_summary.csv (+ optional CSV export)
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python -m ml.calibrate_rules batch --inputs examples/spectra/*.CSV examples/spectra/*.jdx --out-dir reports/calibration_scan --export-csv --suggest-preset
pause
