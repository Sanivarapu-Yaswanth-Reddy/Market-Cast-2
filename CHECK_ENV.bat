@echo off
REM Double-click on ANY machine that has the MarketCast folder. Writes reports\env_report.txt
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY="%USERPROFILE%\anaconda3\python.exe"
%PY% scripts\check_env.py
echo.
pause
