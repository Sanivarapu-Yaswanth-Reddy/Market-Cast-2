# 09 Environment setup: libraries, repositories, machines

Read after `00_START_HERE.md`. This is the exact list of what must be installed where. P0.2 turns it into `environment.yml` / `pyproject.toml` and records pinned versions in `reports/P0_environment.md`.

## 1. Two machines, two environments

| | Data PC (Windows, project root `C:\Users\PC\OneDrive\Desktop\Market Cast`) | GPU machine (Windows or Linux, RTX 50-series, 8 GB VRAM) |
|---|---|---|
| Does | downloads, panel build, PIT store, features, baselines (CPU), tests | model training, hyper-parameter runs, GPU inference |
| Env file | `env/environment-data.yml` → conda env `marketcast-data` | `env/environment-gpu.yml` → conda env `marketcast-gpu` |
| Gets data | everything in `data/` | only `data/processed/`, feature and graph caches (copy by LAN share or external drive, never OneDrive) |
| Check | `CHECK_ENV.bat` | `CHECK_ENV.bat` plus `nvidia-smi` |

The files `env/check_env.py`, `env/CHECK_ENV.bat`, `env/environment-*.yml` are to be copied into the project as `scripts/check_env.py`, `CHECK_ENV.bat` (root), `environment-data.yml` and `environment-gpu.yml` (root). `CHECK_ENV.bat` writes `reports\env_report.txt`, which is how an agent that cannot open a terminal on the machine reads its state.

## 2. Setup commands

Use **Anaconda Prompt** (conda is not on the plain PATH).

```
cd "C:\Users\PC\OneDrive\Desktop\Market Cast"
conda env create -f environment-data.yml        # data PC
conda activate marketcast-data
python scripts\check_env.py
```
GPU machine: same with `environment-gpu.yml`. Install git (GitHub Desktop is on the data PC; it bundles git) and clone the private repo.

If `conda env create` is slow or fails on one package: create the env without the failing line, then `pip install` that package inside it. Record what you did in `reports/P0_environment.md`.

`RUN_DOWNLOAD.bat` uses whatever `python` is first on PATH; run it from the `marketcast-data` prompt (or edit `set PY=`) so the builder sees pandas, pyarrow and numpy.

## 3. Library list and why

| Package | Where | Used for |
|---|---|---|
| numpy, pandas, pyarrow | both | panel build (`marketcast_build_panel.py` needs all three), Parquet |
| duckdb | both | PIT as-of store (P1.4) |
| scipy, statsmodels | both | tests (Kupiec, Christoffersen, DM reference), Markov-switching trend baseline BT3 |
| arch (`arch-py` on conda-forge) | both | GARCH, GJR, skew-t baselines B2, B3 |
| lightgbm, scikit-learn | both | quantile baseline B5, logistic trend baseline |
| scoringrules, properscoring | both (dev) | reference values for CRPS tests (P2.4) |
| pytest, hypothesis, ruff, black | both | tests, property tests for PIT rules, style |
| torch (CUDA 12.8+ build) | GPU | the model |
| torch-geometric | GPU (and data PC for graph-cache tests, CPU torch) | `HeteroData` graph caches and message passing |
| mlflow | GPU | run logging (one experiment per phase) |
| imdlib | data PC, optional | IMD rainfall (`--with-imd`) |
| chronos-forecasting, timesfm | GPU, optional | zero-shot foundation-model baseline B6 only; weights come from Hugging Face (see §5) |

## 4. RTX 50-series (Blackwell) specifics

* Needs a PyTorch wheel built for CUDA 12.8 or newer (PyTorch 2.7+). The YAML uses the `cu128` index. A default `pip install torch` may pull a build without `sm_120` kernels; the symptom is a warning that the GPU's capability is not compatible, or a CUDA error on the first op.
* `check_env.py` prints the card, VRAM, compute capability, the torch arch list, and runs a small GPU matmul. Acceptance: "matmul on GPU: OK" and `sm_120` (or a `compute_120` PTX entry) in the arch list.
* If cu128 wheels do not resolve for the chosen Python version, try Python 3.12, or the newer `cu129` / `cu130` index shown on pytorch.org's install selector. Do not fall back to a nightly unless stable fails; record the exact version.
* Driver: current NVIDIA Game Ready/Studio driver; it supports newer CUDA runtimes than the wheel's, so the system CUDA toolkit is not required.
* 8 GB VRAM: bf16 autocast, micro-batch of a few dates, gradient accumulation to the target batch, activation checkpointing in the processor layers if needed. Default config (d = 256, 8 to 12 layers, about 11 to 15 M parameters) is expected to fit; if not, use the fallback in P0.2 (d = 192, 8 layers). Close browsers while training (Windows reserves some VRAM for the display).
* PyG: `pip install torch-geometric` alone is enough on PyG 2.3+. The optional wheels (`pyg-lib`, `torch-scatter`, `torch-sparse`) only add speed and are matched to a torch and CUDA version; skip them until profiling says message passing is the bottleneck.

## 5. Repositories and external code

Dependencies come from PyPI, not by vendoring repos. Repos below are references to read; names are from memory of the ecosystem, so open each and confirm license and current status at P0.2 before relying on it.

| Repo | Role | Use |
|---|---|---|
| `pyg-team/pytorch_geometric` | dependency | graph data structures, attention layers (read `HeteroData`, GATv2 / transformer conv for the fabric attention) |
| `google-deepmind/graphcast` | design reference only | encode-process-decode structure and training recipe (JAX; do not depend on it; check weight licensing before any reuse) |
| `google-deepmind/weatherbench2` | evaluation reference | how CRPS, spread-skill, rank histograms are computed for ensembles; cross-check our metric library |
| `frazane/scoringrules` | dependency (dev) | fair CRPS and energy score reference implementations for unit tests |
| `bashtage/arch` | dependency | GARCH family |
| `amazon-science/chronos-forecasting`, `google-research/timesfm` | optional dependency | B6 zero-shot baselines |

## 6. Network and policy limits that affect setup

* From the sandbox the shell reaches only package registries and GitHub; the data PC and GPU machine have normal internet but some sites block scripts (FRED timed out from Python; use the browser).
* Foundation-model weights download from Hugging Face on the machine that runs B6. If that host is blocked there, download in a browser and place the files under `models_cache/` (git-ignored).
* Never put real credentials in `environment*.yml` or the repo. MLflow stays local (`mlruns/`) unless the owner decides otherwise.

## 7. Checklist for Fable (P0.2 additions)

1. Run `CHECK_ENV.bat` on the data PC; fix every `MISSING` line; commit the report summary (not secrets) to `reports/P0_environment.md`.
2. Same on the GPU machine; paste `nvidia-smi` and the GPU section of the report.
3. Pin versions: `conda env export --from-history > environment.lock.yml` and `pip freeze > requirements.lock.txt` on each machine.
4. `pytest -q` runs on both machines.
5. Confirm a 10-step toy training loop on the GPU in bf16 with peak memory printed (`torch.cuda.max_memory_allocated`).
