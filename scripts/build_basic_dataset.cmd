@echo off
REM Build basic-label NPZ from NIST SQLite (long; needs network if --enrich-pubchem).
cd /d "%~dp0.."
set PYTHONPATH=%CD%
python -m ml.structural_fg_svm build-dataset --nist-index ..\NIST\reference_libraries\nistchemdata_ir_index_v7_fresh.sqlite --out-prefix data/training/struct_fg_basic --model-kind basic --enrich-pubchem --pubchem-cache data/training/pubchem_structure_cache_v7.json --pubchem-delay 0.25
pause
