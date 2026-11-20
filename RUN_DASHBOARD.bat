@echo off
REM Opens the MarketCast research dashboard in your browser (http://127.0.0.1:8765, this PC only).
REM Keep this window open while you use the page; close it (or press Ctrl+C) to stop the dashboard.
cd /d "%~dp0"
set PY=%USERPROFILE%\.conda\envs\marketcast-data\python.exe
if not exist "%PY%" (echo Missing env: conda env create -f environment-data.yml & pause & exit /b 1)
set PYTHONPATH=%~dp0src
set OMP_NUM_THREADS=2
"%PY%" -m marketcast app
pause
