@echo off
REM MarketCast: download all datasets into this folder, then build training tables.
REM Double-click to run. Safe to close and re-run: finished files are skipped.
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY="%USERPROFILE%\anaconda3\python.exe"
if not exist %PY% if not "%PY%"=="python" set PY="%USERPROFILE%\.conda\python.exe"
echo Using %PY%
%PY% scripts\marketcast_download.py %*
%PY% scripts\marketcast_build_panel.py
echo.
echo All done. See README_DATA.md for what each folder contains.
pause
