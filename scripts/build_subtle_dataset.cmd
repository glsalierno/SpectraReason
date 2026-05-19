@echo off
REM Build subtle-label NPZ from NIST (run before train_subtle.cmd).
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python -m ml.structural_fg_svm build-dataset --nist-index ..\NIST\reference_libraries\nistchemdata_ir_index_v7_fresh.sqlite --out-prefix data/training/struct_fg_subtle --model-kind subtle --enrich-pubchem --pubchem-cache data/training/pubchem_structure_cache_v7.json --pubchem-delay 0.25
pause
