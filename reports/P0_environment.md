# P0.2 Environment record

## Data PC (DESKTOP-LQDHANP), from `reports/env_report.txt`, 2026-11-16

| Item | Value |
|---|---|
| OS | Windows 10/11 build 26300, x64 |
| Python | 3.11.18, conda env `marketcast-data` (`C:\Users\PC\.conda\envs\marketcast-data`) |
| conda | `C:\ProgramData\anaconda3` |
| Key packages | numpy 2.4.6, pandas 3.1.1, pyarrow 26.0.1, duckdb 2.0.0, scipy 1.17.1, statsmodels 0.15.0, arch 9.0.0, lightgbm 4.7.0, scikit-learn 1.9.1, scoringrules 0.10.0, properscoring 0.1, mlflow 3.18.0, pytest 9.2.0 |
| torch / PyG | not installed (training happens on the GPU machine) |
| RAM | 15.7 GB (about 7.5 GB free in normal use) |
| Disk | 207 GB free of 476 GB |
| CPU threads | 8 |
| git | not on PATH (GitHub Desktop is installed) |
| Project location | inside OneDrive (keep large data out of sync if it causes trouble; decision Q-03) |

Notes: the full in-memory equity build peaks at several GB; close browsers before `BUILD_DATA.bat`. P0.3 moves to per-year processing.

## GPU training machine: TO FILL

Run on that machine: `nvidia-smi`, then `CHECK_ENV.bat` inside env `marketcast-gpu` (`conda env create -f environment-gpu.yml`).

| Item | Value |
|---|---|
| GPU model | NVIDIA RTX 50-series (owner, 2026-11-16) |
| VRAM | 8 GB (owner, 2026-11-16) |
| Driver / CUDA | ? |
| torch build | must be a CUDA 12.8+ wheel (cu128 or newer) for Blackwell |
| Check | `CHECK_ENV.bat` shows "matmul on GPU: OK" |
| RAM / disk | ? |

Model sizing: 8 GB is the default-config threshold (06 P0.2). Start with d = 256, 8 layers, bf16, micro-batch 1 date x 2 samples with gradient accumulation; drop to d = 192 if a 10-step toy run exceeds about 7 GB peak (`torch.cuda.max_memory_allocated`).
