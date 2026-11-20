@echo off
REM FULL model training on the GPU machine (env marketcast-gpu). Copy first: the repo + data\targets, data\features,
REM data\graph (1.8 GB, see scripts\sync_to_gpu.md). Usage:   RUN_TRAIN_GPU.bat [fold] [seed]     default F4 0
REM Order: smoke check -> Phase 3 deterministic -> Phase 4 generative (warm-started) -> unrolled K=5 -> score.
REM Every stage is resumable: re-run the same command after an interruption.
REM NOTE: rule 8 blocks training until gate G2 is signed off (dev_plan\state\G2.done). To override on purpose, set
REM       OVERRIDE=--set train.allow_before_g2=true   before calling this file; it is recorded in the run registry.
cd /d "%~dp0"
set PY=%USERPROFILE%\.conda\envs\marketcast-gpu\python.exe
if not exist "%PY%" set PY=python
set PYTHONPATH=%~dp0src
set FOLD=%1
if "%FOLD%"=="" set FOLD=F4
set SEED=%2
if "%SEED%"=="" set SEED=0
if not exist logs mkdir logs
set R3=
if exist runs\p3-det-%FOLD%-s%SEED%\ckpt_last.pt set R3=--resume
set R4=
if exist runs\p4-gen-%FOLD%-s%SEED%\ckpt_last.pt set R4=--resume
set R5=
if exist runs\p4-unroll5-%FOLD%-s%SEED%\ckpt_last.pt set R5=--resume

"%PY%" scripts\check_env.py
"%PY%" -m marketcast train --config model/p3_det.yaml --fold %FOLD% --smoke || (echo SMOKE TEST FAILED & exit /b 1)
"%PY%" -m marketcast train --config model/p3_det.yaml --fold %FOLD% --seed %SEED% %OVERRIDE% %R3% || exit /b 1
"%PY%" -m marketcast train --config model/p4_gen.yaml --fold %FOLD% --seed %SEED% %OVERRIDE% %R4% --set train.init_from=runs/p3-det-%FOLD%-s%SEED%/ckpt_best.pt || exit /b 1
"%PY%" -m marketcast train --config model/p4_unroll.yaml --fold %FOLD% --seed %SEED% %OVERRIDE% %R5% --set train.init_from=runs/p4-gen-%FOLD%-s%SEED%/ckpt_best.pt || exit /b 1
"%PY%" -m marketcast eval score --run p3-det-%FOLD%-s%SEED% --split val --every 5
"%PY%" -m marketcast eval score --run p4-unroll5-%FOLD%-s%SEED% --split val --paths 16 --every 5
echo Done. Copy runs\ and reports\scores\ back to the data PC.
