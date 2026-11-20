@echo off
REM Builds data\processed from data\raw using the marketcast-data conda environment, then writes reports\data_quality.txt
REM Close Chrome and other heavy programs first (the full build can use several GB of RAM).
cd /d "%~dp0"
set PY=%USERPROFILE%\.conda\envs\marketcast-data\python.exe
if not exist "%PY%" (
  echo Could not find %PY%
  echo Create the environment first:  conda env create -f environment-data.yml
  pause
  exit /b 1
)
echo Using %PY%
"%PY%" scripts\marketcast_build_panel.py
if errorlevel 1 (echo BUILD FAILED - see messages above & pause & exit /b 1)
"%PY%" scripts\validate_processed.py
echo.
echo Done. Report: reports\data_quality.txt
pause
