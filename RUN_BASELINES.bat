@echo off
REM Baseline runs for gate G2 (06 P2.5). Safe to re-run: each (model, fold) overwrites its own score file.
REM   B0, B1: every test date of all nine folds (minutes per fold)
REM   B5, B2: the four "fast" folds, every 5th date (LightGBM fit and GARCH refits are slow; widen later)
REM Output: reports\scores\<model>\<fold>_test.parquet, then reports\scorecard_baselines.md. Log: logs\baselines_run.log
cd /d "%~dp0"
set PY=%USERPROFILE%\.conda\envs\marketcast-data\python.exe
if not exist "%PY%" (echo Missing env: conda env create -f environment-data.yml & exit /b 1)
set PYTHONPATH=%~dp0src
if "%OMP_NUM_THREADS%"=="" set OMP_NUM_THREADS=3
if not exist logs mkdir logs
(
  "%PY%" -m marketcast baselines run --model B0 --folds all
  "%PY%" -m marketcast baselines run --model B1 --folds all
  "%PY%" -m marketcast baselines run --model B5 --folds fast --every 5
  "%PY%" -m marketcast baselines run --model B2 --folds fast --every 5
  "%PY%" -m marketcast eval scorecard --out reports\scorecard_baselines.md
) >> logs\baselines_run.log 2>&1
echo Baselines finished. See reports\scorecard_baselines.md
