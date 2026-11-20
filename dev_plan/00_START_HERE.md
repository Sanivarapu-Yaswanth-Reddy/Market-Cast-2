# MarketCast development plan: start here

Written 2026-11-16 for the agent (Fable) that will build MarketCast. Read this file fully, then the others in order. Everything you need to start work today is in this folder. The design source of truth is the doc "MarketCast: Adapting WeatherNext to Stock Market Forecasting" (https://claude.ai/artifact/E8cLgVbfFh1kcL3Y9dQS6v). These files turn that doc into an executable plan. If they disagree with the doc, the doc wins on product intent and these files win on implementation detail; log the conflict in `08_DECISIONS_RISKS_OPEN_QUESTIONS.md`.

## 1. Mission in five sentences

MarketCast is a probabilistic scenario engine for Indian equities. Every evening it produces M coherent sample paths (64 to 1,000) of the next 1 to 20 trading days for every stock in a 500-stock universe. From those same paths it reads four outputs: per-stock return quantiles, per-stock and index volatility, Nifty 50 trend odds, and the most-affected sectors. The model is a WeatherNext/GraphCast-style encode-process-decode graph network over a "sector fabric", made generative with FGN-style shared low-dimensional noise and trained with fair CRPS. It must beat GARCH, DCC-GARCH, historical simulation, a factor model and LightGBM on proper scores and calibration, out of sample, before anything ships. It never outputs buy or sell calls.

## 2. Where things stand right now (2026-11-16, 01:30 IST)

| Item | State |
|---|---|
| Design doc | Complete (problem, targets, 10-layer architecture, harness, 6-phase plan, edge cases, risks) |
| Raw data download | **Running** in a console window on the owner's PC (`RUN_DOWNLOAD.bat`). Indices done through Oct 2016 at last check. Order: imports → reference → indices → global → factors → cm (cash bhavcopy) → delivery → fo. Expected to take a few hours in total; the cash, delivery and F&O groups had not started |
| Already on disk | FRED 40 series, IIMA India 4-factor library (11 files, 2025-12 release), NSDL FPI monthly flows 2002–Nov 2026, RBI policy-rate changes (119 rows), NSE reference snapshot 2026-11-16 (EQUITY_L, delisted, ETF list, bulk/block deals, 37 index constituent lists), ~1,000+ daily all-index files |
| Scripts | `scripts/marketcast_download.py` (resumable downloader, logs to `data/manifest.csv`), `scripts/marketcast_build_panel.py` (raw → `data/processed/*.parquet`; tested on synthetic data only) |
| Data catalog | `README_DATA.md` (what each folder holds, PIT notes, gaps) |
| Code repo for the model | **Does not exist yet.** Task P0.2 creates it |
| Processed tables | **Not built yet.** The builder runs automatically when the download finishes |

## 3. Reading order

1. `00_START_HERE.md` (this file)
2. `01_PRODUCT_SPEC.md`: what is forecast, for whom, how success is judged
3. `02_DATA_CONTRACT.md`: every table, its columns, its point-in-time rule, and the known gaps
4. `03_FEATURES_AND_TARGETS.md`: exact formulas for targets, features, normalization, masks
5. `04_MODEL_ARCHITECTURE.md`: layers L0–L9 with shapes, hyperparameters, losses, rollout
6. `05_EVALUATION_HARNESS.md`: MarketBench (splits, purge, metrics, baselines, scorecards)
7. `06_BUILD_PLAN.md`: **the algorithmic plan**, phases P0–P6, every task with inputs, outputs, algorithm and acceptance test
8. `07_REPO_LAYOUT_AND_CONVENTIONS.md`: code layout, configs, tests, compute
9. `08_DECISIONS_RISKS_OPEN_QUESTIONS.md`: decisions already taken, risks, questions for the owner
10. `plan.yaml` + `plan_status.py`: the same task graph in machine-readable form; run `python dev_plan/plan_status.py` to see what is done and what is ready
11. `scenario_tags_seed.csv`: dated historical episodes for evaluation slices (needs review)
12. `heldout_named_20.csv`: the 20 companies never used in training (chosen 2026-11-16)
13. `09_ENVIRONMENT_SETUP.md`: libraries, conda env files, RTX 50-series install, reference repos, `CHECK_ENV.bat`

## 4. Non-negotiable rules

These come from the doc. Breaking any of them makes every later result worthless, so treat them as invariants and write tests for them.

1. **Point-in-time or nothing.** Every stored fact has `valid_date` (the period it describes) and `available_at` (the IST timestamp it became public). Training rows for decision date t may only read facts with `available_at ≤ cutoff(t)`. Cutoff rule in `02_DATA_CONTRACT.md` §2.
2. **company_id is the key.** company_id = first 9 characters of the equity ISIN, because a face-value split issues a new ISIN for the same company. Never join on ticker.
3. **Survivorship-inclusive.** Delisted, merged and suspended companies stay in the universe up to their exit, with a terminal return.
4. **Raw is immutable.** Never edit `data/raw/`. Every derived table is rebuilt by code from raw.
5. **Walk-forward only.** No random splits. Purge 20 trading days and embargo 20 more around every boundary. Labels (trend, drawdown) are built inside each fold.
6. **The vault is sealed.** Calendar 2026 (from 2026-01-01) is the final holdout. Nothing reads it until the Phase 4 gate is signed off, and then exactly once per model version. Log every opening in `reports/vault_log.md`.
7. **Held-out stocks stay held out.** The 20 named companies in `heldout_named_20.csv` (one or two per NSE industry, large to small caps), plus a stratified top-up to 15% of companies (rule in `05_EVALUATION_HARNESS.md` §3), never enter any training batch, feature normalizer, graph statistic fit or training-time market aggregate.
8. **Harness before model.** Phase 2 (all baselines scored) must finish before any neural model trains.
9. **Log every run.** Every training or evaluation run gets a run id, config hash, git commit, data snapshot hash and metrics in the experiment log. The run count feeds multiple-testing corrections.
10. **Numbers come from code, words come from the language layer.** No LLM ever produces or edits a forecast number.
11. **No public stock-specific views.** Output stays internal until the owner has legal advice (SEBI Research Analyst / Investment Adviser rules).

## 5. First session checklist (do these in order)

1. Run `python dev_plan/plan_status.py`. It shows which outputs already exist.
2. **P0.1** Check whether the download finished: open `data/manifest.csv`, count `ok` rows per group, and compare with the targets in `06_BUILD_PLAN.md` P0.1. If the console is still running, do not start a second downloader. Work on P0.2 (repo scaffold) and P2.3/P2.4 (scenario tags, metric library), which need no data.
3. **P0.2** Create the code package `src/marketcast/` per `07_REPO_LAYOUT_AND_CONVENTIONS.md`, with `pyproject.toml`, an environment file, pytest and git.
4. **P0.3** Once raw data is complete, run the panel builder (rewrite it to process by year if memory fails, see the task) and record row counts.
5. Continue down `06_BUILD_PLAN.md` in dependency order. At each gate, write the gate report in `reports/` and stop for the owner's sign-off when the plan says so.

## 6. Environment facts

* Owner's machine: Windows, project root `C:\Users\PC\OneDrive\Desktop\Market Cast`. Python is available through Anaconda (`RUN_DOWNLOAD.bat` detects `python` or the Anaconda path). This PC does the data work. Training runs on a **second computer with an NVIDIA GPU** that the owner can access; P0.2 records both machines (`nvidia-smi` on the GPU machine: model, VRAM, driver, CUDA version).
* The folder sits inside **OneDrive**. A full raw build is about 4–5 GB, and OneDrive will try to sync it. Recommend to the owner either pausing sync or moving `data\` outside OneDrive with a junction. Never delete anything to save space without asking.
* On this network, plain Python requests to FRED time out, and some sites block scripts. NSE archive files (`nsearchives.nseindia.com`) work at a polite rate (the downloader uses 2 workers, 0.5 s delay, 30 s timeout, 5-minute cooldown after 12 consecutive errors). Do not call NSE's internal `/api/` endpoints. Use published archive files only.
* Training compute: the model is 5–15 M parameters. One run should take a few hours on a single modern GPU. Use the owner's GPU machine: copy the repo (git) plus `data/features/<fold>/` and `data/graph/<fold>/` only, never raw data. Kaggle or Colab are the fallback; rent a GPU only with the owner's approval.

## 7. Escalate to the owner (Yaswanth) before you

* spend money (paid data, GPU rental, APIs);
* publish, share or expose any forecast outside the machine;
* change a gate threshold, the target definitions, the trend rule or the vault boundary;
* delete or move data;
* conclude a gate passed when it is borderline (write the numbers and ask).

## 8. Reporting

After each work session, append to `reports/progress_log.md`: date, tasks touched, what changed, numbers produced, blockers, and the next task. Gate reports use the template in `05_EVALUATION_HARNESS.md` §9.
