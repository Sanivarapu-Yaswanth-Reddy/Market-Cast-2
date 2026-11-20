# MarketCast code guide

How the code is organised, how to run it, and what is and is not finished. The design lives in `dev_plan/`;
this file is the map from that plan to the code. Every module starts with a docstring that says what it does,
which plan section it implements, and where it deliberately differs.

## 1. The pipeline in one picture

```
data/raw  ──build-panel──▶  processed/equity_daily, fo_daily, indices, fred          (P0.3)
                │
     calendar ─ master ─ universe ─ sectors ─ derivatives ─ global                   (P1.1 – P1.8)
                │
        folds make  (walk-forward folds + frozen held-out list)                      (P2.2)
                │
        targets build ──▶ data/targets   "model panel": returns, sigma, z, masks      (P1.3, P1.9, P2.1)
                │
   graph build ──▶ data/graph            features build ──▶ data/features            (P3.2, P3.1)
                │                                   │
                └────────────── train ──────────────┘ ──▶ runs/<run>/ckpt_best.pt    (P3.4, P4.1, P4.2)
                                   │
            baselines run ─────────┴── eval score ──▶ reports/scores ──▶ eval scorecard
```

Everything is one command line: `python -m marketcast <group> <command>` (`--help` lists them).

## 2. Commands

On the **data PC** (all of this has already been run once on 2026-11-16):

```
python -m marketcast data all            # the whole data foundation, in order (about 25 minutes from raw)
python -m marketcast graph build         # sector tree + monthly correlation edges (about 1 minute)
python -m marketcast features build      # feature store + per-fold normalisers (about 5 minutes)
python -m marketcast pit check           # property test of as-of queries on the live store
python -m pytest -q                      # 58 tests; the real-data ones skip if caches are missing
```

Baselines (CPU, data PC). `--folds fast` = F2, F4, F6, F8; `all` = F1–F9. The vault fold is refused.

```
python -m marketcast baselines run --model B0 --folds all
python -m marketcast baselines run --model B1 --folds all
python -m marketcast baselines run --model B2 --folds all --every 5     # GARCH: slow, see garch.py
python -m marketcast baselines run --model B5 --folds all
python -m marketcast eval scorecard --out reports/scorecard_baselines.md
```

On the **GPU machine** (copy the repo plus `data/targets`, `data/features`, `data/graph`; nothing else):

```
python -m marketcast train --config model/p3_det.yaml --fold F4 --smoke      # 1 minute, checks the code path
python -m marketcast train --config model/p3_det.yaml --fold F4 --seed 0     # Phase 3
python -m marketcast train --config model/p4_gen.yaml --fold F4 --seed 0     # Phase 4, stage S1
python -m marketcast train --config model/p4_unroll.yaml --fold F4 --seed 0 --set train.init_from=runs/p4-gen-F4-s0/ckpt_best.pt
python -m marketcast eval score --run p4-gen-F4-s0 --split val --paths 16 --every 5
```

`--resume` continues an interrupted run. `--set train.key=value` overrides any config value.
**Training is blocked until gate G2 is signed off** (00 rule 8): a non-smoke run stops with a message unless
`dev_plan/state/G2.done` exists or you pass `--set train.allow_before_g2=true`.

### One-click launchers (double-click in Explorer)

| File | What it does |
|---|---|
| `RUN_DASHBOARD.bat` | opens the research dashboard at http://127.0.0.1:8765 (this PC only) |
| `RUN_TRAIN_CPU.bat` | reduced model (`configs/model/p4_gen_cpu.yaml`, fold F9) on this PC; resumes if interrupted; scores the validation year at the end |
| `RUN_BASELINES.bat` | B0 and B1 on all folds, B5 and B2 on the fast folds (every 5th date), then the scorecard |
| `RUN_TRAIN_GPU.bat [fold] [seed]` | full model on the GPU machine: smoke check → Phase 3 → Phase 4 → unrolled → score |

### Dashboard (`python -m marketcast app`)

Four tabs sharing one filter row (scenario source, decision date, horizon, number of scenarios):
**Portfolio risk** (holdings picker, outcome fan with what actually happened, VaR / expected shortfall, drawdown
odds, per-holding ranges), **Market and sectors** (trend odds, sector exposure), **Model scores** (whatever is in
`reports/scores`), **Training** (loss curves of every run, refreshed every 30 s).
Sources: baseline B1 (instant) and any generative run under `runs/` (rolled out on the CPU: minutes the first
time per date, then cached in `data/forecasts/app_cache`). Dates from 2026-01-01 are refused (vault), and a
model's own training period is flagged as in-sample. Code: `src/marketcast/app/` (engine = numbers,
server = standard-library HTTP bound to 127.0.0.1, `static/index.html` = the page).

## 3. Where things are

| Folder | What | Start reading at |
|---|---|---|
| `src/marketcast/data/` | raw → processed tables and reference tables | `build_panel.py`, `wide.py` |
| `src/marketcast/pit/` | point-in-time clock and DuckDB store | `lags.py` |
| `src/marketcast/targets/` | return / volatility formulas, labels, the model panel | `returns.py`, `build.py` |
| `src/marketcast/features/` | stock channels, global nodes, forcings, normalisers | `stock.py` |
| `src/marketcast/graph/` | sector fabric: tree, significance, correlation edges | `fabric.py` |
| `src/marketcast/model/` | encoder, edge attention, FGN noise, processor, heads, rollout | `marketcast.py` |
| `src/marketcast/train/` | batch assembly, losses, training loop, run log | `loop.py`, `data_loader.py` |
| `src/marketcast/eval/` | folds, forecast object + scorer, metrics, scorecard | `forecast.py`, `folds.py` |
| `src/marketcast/baselines/` | B0, B1, B2, B5, trend baselines, runner | `run.py` |
| `src/marketcast/readout/`, `decision/`, `explain/` | path readouts, portfolio JSON, number validator | `decision/portfolio.py` |
| `configs/` | every tunable or curated input (lags, events, sectors, edges, folds, model, train) | each file's header |
| `tests/` | formulas by hand, PIT rules, model structure, real-data leakage checks | `test_real_data.py` |

## 4. Conventions worth knowing before changing anything

* **Row t = what happened on session t.** Nothing on disk is stored pre-shifted. The loader pairs decision
  date t with targets t+1…t+k, so `eval/folds.py` alone owns purging, embargo and the vault.
* **`key` = company_id** everywhere (never a ticker). `SYM:<symbol>` keys are pre-2011 names with no ISIN.
* **Held-out companies** are excluded at four points: sigma-floor median and market aggregate
  (`targets/build.py`), sector-rank reference set and normaliser fit (`features/`), correlation peers
  (`graph/fabric.py`), and training batches (`train/data_loader.py`). `tests/test_real_data.py` checks it.
* **The vault** (decision dates ≥ 2026-01-01) raises `PermissionError` unless `allow_vault=True`, which
  appends to `reports/vault_log.md`. Data builders do process 2026 rows; models and scorers cannot.
* **Raw features, per-fold statistics.** `data/features/stock.npy` holds un-normalised channels once;
  each fold only adds `norm.json`. The rollout needs raw values to update price channels.
* **Price channels are defined twice on purpose**: numpy in `features/stock.py` (history) and torch in
  `model/rollout.py` (simulated days). `test_rollout_price_channels_match_numpy_builder` keeps them equal.
* **Windows**: `marketcast/__init__.py` fixes two silent native crashes (MKL DLL path without
  `conda activate`; duplicate OpenMP runtime with pip torch). Import `marketcast` before heavy numerics.

## 5. Status against the plan (2026-11-16)

| Plan task | Code | Run on real data | Open |
|---|---|---|---|
| P0.3 per-year builder | done | yes | — |
| P1.1 calendar, aliases | done | yes | event dates in `configs/calendar_events.yaml` unverified; raw data has no weekend sessions |
| P1.2 security master | done | yes | 72 reused tickers listed in `reports/P1_symbol_conflicts.csv` |
| P1.3 corporate actions, returns | done | yes | hand verification of events and the 20 stock-day check (needs a person) |
| P1.4 PIT store | done | yes | — |
| P1.5 universe | done | yes | traded-value overlap with Nifty 500 is 92.4% vs the 95% target |
| P1.6 sector map | done | yes | fails the ≤ 5% unclassified test in early years; 3-level tree (no basic industry) |
| P1.7 derivatives | done | yes | F&O ban list not downloaded |
| P1.8 global / macro | done | yes | conservative lags, no vintages |
| P1.9 masks | done | yes | merger window and dividend ex-date masks are empty (no data) |
| P2.1 targets, P2.2 folds | done | yes | — |
| P2.4 metrics | done (earlier) | — | — |
| P2.5 baselines | B0, B0g, B1, B2, B5, BT1, BT2 | spot-checked only | full fold runs; B3, B4, B6, BT3, BS1–2 not written |
| P2.6 leakage suite, scorecard | tests 1, 2, 3, 5, 6 + scorecard | yes | random-label control (needs training); slices, CIs, plots |
| P3.1 features, P3.2 graph | done | yes | — |
| P3.3 model, P3.4 training loop | done | CPU smoke only | real training on the GPU machine |
| P4.1 FGN, P4.2 rollout | done | CPU smoke only | same |
| P4.4 readouts, P6.3 portfolio, P6.4 validator | done | synthetic tests | shock-token input path (Phase 5), LLM narration |
| P5 events, P6.1–6.2 live pipeline | not started | — | — |

No task has been marked done in `dev_plan/state/`: several acceptance criteria need the owner
(see `reports/progress_log.md`).
