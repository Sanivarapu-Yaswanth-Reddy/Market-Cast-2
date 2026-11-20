# 07 Repo layout and conventions

## 1. Layout (project root = the Market Cast folder)

```
Market Cast/
  README_DATA.md                 data catalog (exists)
  RUN_DOWNLOAD.bat               downloader + builder launcher (exists)
  scripts/                       existing standalone scripts (keep working; thin wrappers later)
  dev_plan/                      this plan (read-only for the agent except plan_status state)
    state/                       <task>.done markers written by plan_status.py
  pyproject.toml                 package "marketcast", Python ≥ 3.11
  environment.yml                conda env "marketcast"
  configs/
    data.yaml                    paths, date ranges, series kept
    pit_lags.yaml                publication-lag table (02 §2)
    index_aliases.yaml
    edges_global.yaml            curated global → industry exposures
    edges_supply.yaml, edges_group.yaml   (P4.5)
    folds.yaml, heldout_companies.txt, scenario_tags.csv
    costs.yaml                   transaction-cost assumptions
    model/{p3_det.yaml, p4_gen.yaml}
    train/{default.yaml}
  src/marketcast/
    data/        build_panel.py, calendar.py, security_master.py, corporate_actions.py, universe.py, sector_map.py, derivatives.py, global_align.py, masks.py
    pit/         store.py (DuckDB API), lags.py
    features/    stock.py, global_nodes.py, forcings.py, normalize.py, build.py
    targets/     returns.py, realized_vol.py, labels.py
    graph/       fabric.py, significance.py, corr_edges.py, curated.py
    model/       encoder.py, fabric_attn.py, processor.py, fgn.py, heads.py, rollout.py, marketcast.py
    train/       loop.py, losses.py, data_loader.py, logging.py
    baselines/   ewma.py, hist_sim.py, garch.py, dcc.py, factor.py, lgbm_quantile.py, foundation.py, trend.py, sector.py
    eval/        forecast.py, metrics.py, folds.py, slices.py, scorecard.py, leakage.py, economic.py
    readout/     trend.py, sectors.py, shock.py
    decision/    portfolio.py, contract.py
    explain/     narrate.py, validate_numbers.py
    live/        nightly.py, freeze.py, score_live.py
    cli.py       `python -m marketcast <command>`
  tests/         mirrors src; data-free tests use tiny synthetic fixtures in tests/fixtures/
  notebooks/     exploration only; nothing a gate depends on lives here
  reports/       progress_log.md, P*/G* reports, scorecards, scores/, live/, vault_log.md
  data/          see 02 §1 (git-ignored)
  mlruns/        local experiment tracking (git-ignored)
```

## 2. CLI (one entry point, every step re-runnable and idempotent)

```
python -m marketcast data build-panel            # P0.3
python -m marketcast data calendar|master|corp-actions|universe|sectors|derivatives|global|masks
python -m marketcast pit check                   # PIT property tests on the live store
python -m marketcast targets build
python -m marketcast folds make
python -m marketcast baselines run --model B2 --folds fast|all
python -m marketcast features build --fold F4
python -m marketcast graph build --fold F4
python -m marketcast train --config configs/model/p3_det.yaml --fold F4 --seed 0
python -m marketcast eval score --model MC-det --folds all
python -m marketcast eval scorecard --out reports/scorecard_<name>.md
python -m marketcast live nightly
```

## 3. Environment

`environment.yml` (pin minor versions at P0.2 and record them):
python 3.11, pandas, pyarrow, duckdb, numpy, scipy, statsmodels, arch, lightgbm, scikit-learn, pytorch (CUDA build if a GPU exists), torch-geometric, xarray (optional), pyyaml, mlflow, matplotlib, pytest, hypothesis, scoringrules, properscoring, imdlib (optional), tqdm, rich.

Windows notes: use `pathlib` everywhere; quote the project path (it contains a space); OneDrive may lock files during sync, so write to a temp file then `os.replace`; long paths can exceed 260 chars, so keep partition names short.

## 4. Conventions

* **Determinism:** set seeds for Python, NumPy and Torch; `torch.backends.cudnn.deterministic = True` for evaluation runs; record seeds in run metadata.
* **Config:** every run is fully described by a YAML file plus CLI overrides; the resolved config is saved with the run and hashed (sha256 of canonical JSON).
* **Data snapshot hash:** sha256 over the sorted list of (path, size, mtime) of the raw files used, saved with each run.
* **Dates:** `datetime.date` for sessions, tz-aware `pd.Timestamp(tz="Asia/Kolkata")` for `available_at`. Never naive timestamps for availability.
* **Keys:** ISIN strings, uppercase; never ticker joins.
* **Units:** returns as fractions (0.01 = 1%); money in ₹; flows in ₹ crore as published, converted where used.
* **Typing and style:** type hints, `ruff` + `black`; functions that touch time take an explicit `as_of` argument (no hidden "today").
* **Tests:** every formula in `03` has a hand-computed unit test; every PIT rule has a property test; CI target `pytest -q` under 5 minutes on synthetic fixtures.
* **Logging:** `logging` module, INFO to console, DEBUG to `logs/<command>_<date>.log`.
* **Git:** commit per task (`P1.4: PIT store with as-of API`); never commit `data/`, `mlruns/` or secrets. Tag gate passes (`gate-G1`).
* **Experiments:** one MLflow experiment per phase; run names `<phase>-<variant>-<fold>-s<seed>`; never delete runs (they count for multiple testing).

## 5. Compute plan

| Stage | Where | Approx. cost |
|---|---|---|
| P0–P2 (data, baselines) | owner's PC, CPU | hours; DCC on 500 names is the slowest (composite likelihood keeps it tractable) |
| P3 dev runs | owner's NVIDIA GPU machine (second computer); Kaggle/Colab fallback | 1–3 h per fold run |
| P3/P4 gates | all 9 folds × 4 seeds (P4) | ≈ 36 runs × 2–4 h; rent one GPU only with owner approval |
| Live | owner's PC nightly | minutes (inference only) |

Moving data to the GPU machine or a notebook: copy `data/features/<fold>/` and `data/graph/<fold>/` only (not raw). Keep one fold per upload.
