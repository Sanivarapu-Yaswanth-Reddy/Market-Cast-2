# Progress log

Newest first. One entry per working session: what was done, numbers produced by code, what is next.

## 2026-11-16 (later the same day: CPU training started, baselines started, dashboard)

Owner decisions: train a small model on this PC now; run baselines in parallel and override gate G2 for that run
(recorded in `runs/registry.jsonl`); build a local web dashboard; leave the six other Nifty 50 names held out.

- **Training started** (`RUN_TRAIN_CPU.bat`, detached): `p4-gen-cpu-F9-s0`, 1.0 M parameters, fold F9 (train to
  2023-11-01, validate 2024), 4,000 steps at about 2 s per step. Validation score (CRPS z + CRPS v) 0.944 untrained
  → 0.892 at step 250. The batch file scores the validation year when training ends. **Not finished when this
  entry was written**; this is a chain test, not the gate model (that is `p4_gen.yaml` on the GPU machine).
- **Update 20:45**: that first run (`p4-gen-cpu-F9-s0`, lr 5e-4, 1 date per step) diverged after step 250 (training
  loss 1.18 → 1.68) and was stopped. Its replacement `p4-gen-cpu2-F9-s0` (lr 1.5e-4, 2 dates per step, 2,000 steps,
  about 4 s per step) scored 1.118 at step 200, still worse than untrained (0.944). An overfit check on 8 fixed
  dates (`--set train.overfit_dates=8`, run `overfit-gen-F9-s0`) took CRPS z from 0.65 to 0.46 and the loss from
  1.06 to about 0.65 in 240 steps, so the model and loss do learn; the full-data run simply has not had enough
  steps. `p4-gen-cpu2` was resumed from step 200 and is running. **Whether it ends better than persistence is
  not known yet.**
- **Baselines started** (`RUN_BASELINES.bat`, detached): B0, B1 all folds; B5, B2 fast folds every 5th date; then
  `reports/scorecard_baselines.md`. First results, B0 test split: CRPS h=1 0.554 / 0.567 / 0.557 on F1 / F2 / F3,
  90% coverage 0.885 / 0.878 / 0.885.
- **Dashboard** (`RUN_DASHBOARD.bat`, `src/marketcast/app/`): verified in the browser with baseline B1 on real data
  and through the API with the in-training checkpoint. A 32-scenario model rollout took 261 s while training and
  baselines shared the CPU; results are cached on disk per checkpoint.
- **GPU machine**: `RUN_TRAIN_GPU.bat` written, not run (no access from here).
- 59 tests pass.
- **Next**: when the two background jobs finish, read `reports/scorecard_baselines.md` and the Training / Model
  scores tabs; write the G2 gate report; run `RUN_TRAIN_GPU.bat` on the GPU machine.

## 2026-11-16 (coding session: everything except training)

Scope set by the owner: write and verify the code now; training waits for the GPU machine. Map of the code and
per-task status: `README_CODE.md`.

- **Data rebuilt per year** (`python -m marketcast data build-panel`): equity 8.64 M rows (4.8% fund/ETF) 2005-01-03 to 2026-11-13, 5,398
  sessions; F&O 4,163 files, 0 skipped. The pending "fix 3" rebuild is included: share-only `|ret_1d| > 0.5` fell from
  about 10,000 to 586. Known-event checks: see `reports/data_quality.txt`.
- **Numbers produced by code**
  - Calendar: 5,398 sessions; 16 special (thin Muhurat sessions, detected), 1 outage. **The raw data contains no weekend
    sessions** (budget Saturdays 2015-02-28, 2020-02-01, 2025-02-01 are absent): the downloader appears to skip weekends.
  - Index aliases: 61 renames detected. Official Nifty 50 rows: 3,469; 74 sessions since 2012-07 have no index file.
  - Security master: 4,192 keys (3,739 shares, 453 funds); 1,096 exits, reason known for 79; 72 tickers reused by more
    than one company (`reports/P1_symbol_conflicts.csv`).
  - Universe: MC500 has 500 members at all 43 rebalances, median 7.3% replaced per rebalance. Overlap with Nifty 500:
    80.8% by count (target 80), **92.4% by traded value (target 95: miss)**.
  - Sector map: 751 companies from constituent lists, 245 from the hand map, 1,254 keyword, 1,489 unclassified.
    Unclassified share of MC500 traded value: 6.6% in 2005, below 1% from 2008 (acceptance 5%: fails 2005 only).
  - Derivatives: NIFTY ATM IV vs India VIX correlation 0.940 over 2014-05 to 2025-12 (target 0.9).
  - Targets: z std inside 0.9–1.2 in every year; `|ℓ| > 30%` on unflagged rows 0.0022% (acceptance 0.05%).
  - Held-out: 231 companies = 20 named + 211 stratified, 15.2% of the 1,515 ever in MC500.
  - Graph: 27 industries, 23 sectors, 13 macro sectors; 234 monthly correlation refreshes, mean 2.87 of 10 neighbours
    replaced (rule: at most 3). Feature store: 5,398 × 2,399 × 53 channels; normalised IQR 1.34–1.68 on train.
  - PIT property check on the live store: 0 violations in 500 queries.
- **Verified**: 58 tests pass (formulas by hand, PIT rules a–d, fold boundaries, model structure, losses, rollout,
  portfolio engine, real-data leakage checks). CPU smoke runs of `p3_det`, `p4_gen` and `p4_unroll` on fold F4 complete
  (3 steps each, tiny model) and a smoke checkpoint was rolled out and scored. Baselines B0 and B1 spot-checked on 5
  validation dates (B0: CRPS 0.541 at h = 1, 90% coverage 0.895).
- **Not verified**: any real training run; GPU memory and speed of the 17.0 M-parameter default; bf16 autocast;
  DataLoader workers on Windows; full-fold baseline runs.
- **Found and fixed**: (1) conda MKL crashes silently when Python starts without `conda activate` (how the .bat files
  start it); (2) pip torch and conda NumPy each load an OpenMP runtime and abort on the first BLAS call. Both are
  handled in `marketcast/__init__.py`. CPU torch 2.15.0 was installed into `marketcast-data` for these tests.
- **Needs the owner**
  1. ~~Infosys in the stratified held-out set.~~ **Decided by the owner the same day: excluded** (D-27,
     `configs/heldout_exclusions.yaml`); `INE018I01` took its slot; targets, graph and features were rebuilt and the
     58 tests re-run. Still open: six other Nifty 50 names the sha rule picked (Adani Ports, Hindustan Unilever,
     Jio Financial, L&T, Tata Motors PV, Trent). Hindalco is held out on purpose (named 20).
  2. Verify `configs/calendar_events.yaml`, `configs/sector_manual.yaml`, `configs/edges_global.yaml` (all written from
     memory, flagged unreviewed) and decide on re-downloading weekend sessions.
  3. P1.3 hand checks (20 stock-days, events of the 30 largest stocks) and the G1 / G2 gate reports.
  4. `git init` through GitHub Desktop; no commit was made (git is not on PATH).
- **Next**: full baseline runs (B0, B1, B2, B5 on all folds) → scorecard → G2 report → training on the GPU machine.
  Not written yet: B3 DCC, B4 factor model, B6, BT3, BS1–2, slices/CIs/plots in the scorecard, Phase 5, live pipeline.
- No task marked done in `dev_plan/state/` this session.

## 2026-11-16 (setup agent, before Fable)

- **Data**: NSE raw download complete (P0.1 report: every acceptance check passes). Processed tables built by `BUILD_DATA.bat`: equity_daily (8.34 M rows, 2005-01-03 to 2026-11-13), indices_daily (295,794 rows), fo_daily (785,783 rows, 2010 to 2026), fred_global (40 series).
- **Data issues found and fixed in `scripts/marketcast_build_panel.py`**:
  1. NSE PREVCLOSE is not adjusted for splits or bonuses (both file formats). Splits and bonuses are now inferred from price gaps matching standard ratios (D-22). Spot checks: RELIANCE 2017-09-07 0.5, HDFCBANK 2019-09-19 0.5, GARFIBRES 2025-01-03 0.2, KAMDHENU 2025-01-08 0.1, SHRIRAMFIN 2025-01-10 0.2.
  2. NSE PREVCLOSE is sometimes a placeholder (1.40, 5.00, 37.10 against closes of 540, 616, 7,574). Returns now use the company's own previous close.
  3. company_id = ISIN[:9] merged different ETFs (INF ISINs share 9 characters within a fund house) and side-by-side share classes (ordinary + DVR). The rule now applies only to INE ISINs that replace each other in time.
  - **Fix 3 needs a rebuild**: run `BUILD_DATA.bat` (about 15 minutes; FRED, indices and F&O are skipped because they exist). Then read `reports/data_quality.txt`: the share-only `|ret_1d| > 0.5` count should drop from about 10,000 to a few hundred.
- **Code**: P0.2 scaffold (package `src/marketcast`, CLI `python -m marketcast`, pytest config), P0.1 inventory module, P2.4 metric library. 22 tests pass under pandas 3.1.1, numpy 2.x, scipy 1.17.1, statsmodels 0.15.0, scoringrules 0.10.0.
- **Not done**: git init (git is not on PATH on the data PC; use GitHub Desktop: File → Add local repository → Market Cast → create repository), GPU machine record (P0.2), P0.3 per-year refactor.
- **Next**: rebuild data → P0.3 (move builder into `src/marketcast/data/build_panel.py`, per-year partitions) → P1.1, P1.2.
