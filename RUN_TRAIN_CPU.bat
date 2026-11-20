@echo off
REM Reduced-model training on this CPU-only PC (configs\model\p4_gen_cpu.yaml, fold F9, seed 0).
REM Resumable: if it is interrupted (reboot, sleep), double-click again and it continues from the last checkpoint.
REM Gate G2 override is explicit and recorded in runs\registry.jsonl (owner decision 2026-11-16).
REM Progress: logs\train_cpu.log, runs\p4-gen-cpu2-F9-s0\metrics.jsonl, or the dashboard's "Training" tab.
cd /d "%~dp0"
set PY=%USERPROFILE%\.conda\envs\marketcast-data\python.exe
if not exist "%PY%" (echo Missing env: conda env create -f environment-data.yml & exit /b 1)
set PYTHONPATH=%~dp0src
if "%OMP_NUM_THREADS%"=="" set OMP_NUM_THREADS=4
if not exist logs mkdir logs
set RESUME=
if exist runs\p4-gen-cpu2-F9-s0\ckpt_last.pt set RESUME=--resume
"%PY%" -m marketcast train --config model/p4_gen_cpu.yaml --fold F9 --seed 0 --set train.allow_before_g2=true %RESUME% >> logs\train_cpu.log 2>&1
"%PY%" -m marketcast eval score --run p4-gen-cpu2-F9-s0 --split val --paths 32 --every 5 >> logs\train_cpu.log 2>&1
echo Training finished. See logs\train_cpu.log
