@echo off
REM Run from the SpectraReason root:  call scripts\env.cmd
cd /d "%~dp0.."
set PYTHONPATH=%CD%
echo PYTHONPATH=%PYTHONPATH%
