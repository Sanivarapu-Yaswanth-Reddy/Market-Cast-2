@echo off
REM Installs the marketcast package (editable) into the marketcast-data env and runs the test suite.
cd /d "%~dp0"
set PY=%USERPROFILE%\.conda\envs\marketcast-data\python.exe
if not exist "%PY%" (echo Missing env: conda env create -f environment-data.yml & pause & exit /b 1)
"%PY%" -m pip install -e . --no-deps -q
"%PY%" -m pytest -q
"%PY%" -m marketcast data inventory
echo.
pause
