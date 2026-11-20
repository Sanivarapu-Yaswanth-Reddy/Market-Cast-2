# 06 Build plan: the algorithm

Seven phases (P0 bootstrap plus the doc's six). Each phase ends in a gate. A gate must pass, with a written report, before the next phase starts. Within a phase, tasks run in dependency order; tasks with no dependency between them may run in parallel. The same graph is in `plan.yaml`; `python dev_plan/plan_status.py` shows progress.

Timeline (from the doc, about six months part-time): P1 weeks 1–4, P2 weeks 5–7, P3 weeks 8–11, P4 weeks 12–16, P5 weeks 17–20, P6 weeks 21–26 and beyond.

Task format: **ID name** · depends · outputs · algorithm · acceptance.

## Master loop

```
while not all gates passed:
    task = first task in plan.yaml whose deps are done and that is not done
    read its spec below; implement in src/marketcast/ with tests
    run it; write outputs to the listed paths; log the run
    check acceptance; if it fails, fix or record a blocker in reports/progress_log.md
    mark done:  python dev_plan/plan_status.py --done <ID> --note "<one line>"
    if task is a gate: write reports/G<n>_gate.md and wait for owner sign-off
```

---

## P0 Bootstrap (days 0–2)

**P0.1 Verify raw download** · deps: none · out: `reports/P0_raw_inventory.md`
```
counts = files on disk per raw folder and year; manifest status counts per group
expected sessions/year ≈ 245–250; compare each year against the cash-bhavcopy count (the calendar)
list failures (status not ok) from manifest; re-run marketcast_download.py for failed groups only (it skips existing files)
```
Acceptance: cm_bhavcopy ≥ 5,300 files 2005→latest; delivery within 1% of cm count; fo ≥ 4,100 (2010→); indices ≥ 3,450 (2012-07→); no year with a gap > 3 consecutive sessions vs cm. If the downloader is still running, do not start another; skip to P0.2, P2.3, P2.4.

**P0.2 Repo scaffold** · deps: none · out: `pyproject.toml`, `environment.yml`, `src/marketcast/__init__.py`, `tests/`, `configs/`, `.gitignore`, `reports/progress_log.md`
Layout and conventions in `07_REPO_LAYOUT_AND_CONVENTIONS.md`. Environment steps and the library list are in `09_ENVIRONMENT_SETUP.md` (run `CHECK_ENV.bat` on both machines first). `git init`; `.gitignore` excludes `data/`. Record both machines in `reports/P0_environment.md`: the data PC (Python, RAM, free disk) and the GPU training machine (GPU model and VRAM from `nvidia-smi`, driver, CUDA, torch build, RAM, disk). Write `scripts/sync_to_gpu.md` describing how code (git) and per-fold feature/graph caches move to the GPU machine and how run logs come back. Size the model and batch to the measured VRAM (≥ 8 GB fits the default config; below that, use d = 192 and 8 layers).
Acceptance: `pytest` runs (0 tests is fine); `python -c "import marketcast"` works.

**P0.3 Build processed tables** · deps: P0.1, P0.2 · out: `data/processed/{equity_daily,indices_daily,fo_daily,fred_global}`
Move `scripts/marketcast_build_panel.py` logic into `src/marketcast/data/build_panel.py`. Change the equity and F&O builders to process one year at a time and write Parquet partitions `equity_daily/year=YYYY/part.parquet` (the in-memory concat of 20 years will not fit comfortably on a laptop). The corporate-action factor needs the previous raw close across year boundaries: carry the last close per key from year y−1 into year y. The back-adjusted level columns need all later factors; compute them in a second pass, or drop them (models use returns).
Acceptance: row counts per year logged; spot-check 3 dates per format era (pre-2011 no ISIN, 2011–2024 old format, post-2024-07-08 UDiFF) against the raw CSV.

---

## P1 Data foundation (weeks 1–4) → Gate G1

**P1.1 Trading calendar and index aliases** · deps: P0.3 · out: `data/pit/ref_trading_calendar.parquet`, `configs/index_aliases.yaml`
```
sessions = sorted dates with a cash bhavcopy file
session_quality: special (Muhurat sessions, weekend budget sessions: 2015-02-28, 2020-02-01, 2025-02-01 — verify each), outage (2021-02-24), partial (manually listed), else normal
calendar_days_since_prev; expiry flags from F&O expiry column (monthly = last expiry of the month for index futures; weekly = index option expiries)
rbi_policy_day from rbi_policy_rate_changes.csv plus MPC schedules (scheduled vs unscheduled flag; unscheduled: 2020-03-27, 2020-05-22, 2022-05-04 — verify)
budget_day, election_result_day (2009-05-18, 2014-05-16, 2019-05-23, 2024-06-04)
index_aliases: map old names (S&P CNX Nifty, CNX Nifty, CNX IT, …) to current names by matching overlapping-date closes
```
Acceptance: every year has 240–252 sessions; every alias maps to exactly one current index; no index has a gap > 5 sessions after aliasing.

**P1.2 Security master and symbol history** · deps: P0.3 · out: `data/pit/ref_security_master.parquet`, `data/pit/ref_symbol_history.parquet`
```
for each ISIN: first/last trade date, symbols used with valid_from/valid_to, series history
join EQUITY_L (listing date, face value, name) and delisted.csv (encoding latin-1; delisting date, reason)
pre-2011 rows without ISIN: map symbol → ISIN by the first later appearance; detect conflicts (one symbol → several ISINs over time) and resolve by date ranges; unresolved → key "SYM:<symbol>" with flag
company_id = ISIN[:9]; build ref.isin_lineage (company_id, isin, valid_from, valid_to); a face-value split changes the ISIN serial, not the company
```
Acceptance: no ISIN with overlapping symbol intervals; ≥ 99% of 2011+ traded value has an ISIN; every inferred split/bonus with an ISIN change keeps its company_id (exceptions listed); conflicts listed in `reports/P1_symbol_conflicts.csv`.

**P1.3 Corporate actions and returns** · deps: P1.2 · out: `data/pit/ref_corporate_actions.parquet`, `data/processed/equity_returns/`, `reports/P1_corp_action_check.md`
```
events = rows with corp_action_flag (ratio inferred from price gaps; NSE PREVCLOSE is NOT adjusted for splits or bonuses) plus rows with jump_unexplained; classify the latter as demerger, rights, relisting or genuine move using the official table (G15) or curated entries; then check against a known-event list (RELIANCE 2017-09-07 0.5, HDFCBANK 2019-09-19 0.5, GARFIBRES 2025-01-03 0.2, KAMDHENU 2025-01-08 0.1, SHRIRAMFIN 2025-01-10 0.2) and compute precision on a hand-labelled sample
dividends: ingest any source obtained for gap G2; else empty table with the masking rule (03 §5)
compute r, ℓ, o, GK, RV², v (03 §1–2); terminal returns for delisted ISINs (02 §7)
gate sample: pick 20 random (isin, date) pairs stratified by era; compare adjusted return and close with the exchange's published values (raw file + NSE corporate action notice); record matches
```
Acceptance: 20/20 stock-days match (G1 criterion); every inferred split/bonus among the 30 largest stocks verified by hand; share of |ℓ| > 30% days < 0.05% after excluding flagged events.

**P1.4 PIT store** · deps: P1.1, P1.2 · out: `data/pit/marketcast.duckdb`, `src/marketcast/pit/store.py`, `configs/pit_lags.yaml`, `tests/test_pit.py`
Implement the API in `02_DATA_CONTRACT.md` §6. Long-format macro/global table `(series, valid_date, available_at, value)` with available_at per the §2 rules and lag table. Every view exposes `available_at`.
Acceptance: tests pass for (a) a 7 pm announcement lands on the next session; (b) a FRED value dated US d is unavailable on India date d and available on d+1; (c) CPI for month m is unavailable before m-end + 15 days; (d) an as-of query never returns a row with available_at > cutoff (property test on 10,000 random queries).

**P1.5 Universe MC500 (and MC1000)** · deps: P1.3, P1.4 · out: `data/pit/ref_universe.parquet`, `reports/P1_universe_check.md`
Algorithm in `02_DATA_CONTRACT.md` §5.
Acceptance: 500 members every period; overlap with the 2026-11-16 Nifty 500 snapshot ≥ 80% by count, ≥ 95% by traded value; delisted members appear before their exit; turnover per rebalance reported (expect 5–12%).

**P1.6 Sector map** · deps: P1.2 · out: `data/pit/ref_sector_map.parquet`, `reports/P1_sector_coverage.md`
```
level source priority: NSE four-level classification snapshot (obtain, G3) > constituent-list Industry label > manual mapping
build the tree: basic_industry → industry → sector → macro_sector → market
delisted / old names: map by (a) the last known label of that ISIN if it ever appeared in a constituent list, (b) name keywords with a reviewed rule table, (c) "unclassified"
valid_from = classification snapshot date; history before the first snapshot uses it with flag hindsight_class = true (decision D-08)
```
Acceptance: ≤ 5% of MC500 traded value unclassified in any year; every node has ≥ 1 child; report node counts per level.

**P1.7 Derivatives features** · deps: P0.3, P1.1 · out: `data/processed/fo_features/`
Basis, OI z-scores, days to expiry, PCR, ATM implied volatility by Black-76 (forward = near-month future settle, rate = call-money rate from FRED monthly, lagged; skip if |moneyness| of nearest strike > 5%). Verify and ingest F&O ban files (`nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_DDMMYYYY.csv` — verify the URL pattern on two dates before bulk download; if it fails, mark gap).
Acceptance: India-VIX-like check: the 30-day ATM IV of NIFTY options correlates ≥ 0.9 with India VIX over 2013–2025.

**P1.8 Global and macro alignment** · deps: P1.4 · out: `data/processed/global_daily_pit/`
Produce the 16 global node vectors (03 §8) on the India session calendar using `store.global_for_india_date` and `store.macro_asof`. Include RBI repo series and the monthly FPI series.
Acceptance: for 20 random India dates, every value equals the hand lookup obeying the PIT rule; no forward fill beyond 10 sessions for daily series (flag stale instead).

**P1.9 Masks** · deps: P1.3, P1.7 · out: `data/processed/masks/`
Circuit inference (03 §5), market halts (2009-05-18, 2020-03-13, 2020-03-23 — verify), stale, session_bad, history_short, merger windows, terminal_unknown, dividend ex-date suspects.
Acceptance: 50 random circuit flags reviewed, precision ≥ 90%; mask shares per year reported.

**P1.10 Gate G1: data foundation** · deps: P1.1–P1.9 · out: `reports/G1_gate.md`
Criteria: (1) 20 random stock-days match exchange records after adjustment (P1.3); (2) as-of queries never return a later filing (P1.4 property test); (3) universe and sector checks pass; (4) leakage tests 1, 3 and 5 from `05` §9 implemented and passing on the data layer. Owner sign-off.

---

## P2 Harness and baselines (weeks 5–7) → Gate G2

**P2.1 Targets and labels** · deps: P1.5, P1.9 · out: `data/targets/`
Store per (date, isin): ℓ_{t}, v_{t}, σ̂_{t}, z_{t}, target mask, liquidity weight. Forward windows are assembled at load time (never stored pre-shifted, so the fold logic owns the purge). Market series: MC500 aggregate and Nifty 50 (2012-07+).
Acceptance: z has mean ≈ 0 and std 0.9–1.2 per year on MC500; no target uses masked days.

**P2.2 Folds and held-out stocks** · deps: P1.5 · out: `configs/folds.yaml`, `configs/heldout_companies.txt`, `tests/test_folds.py`
Folds and purge rule in `05` §1–2; held-out rule in `05` §3 (named 20 from `dev_plan/heldout_named_20.csv`, resolved to company_id via symbol history, plus the stratified top-up).
Acceptance: boundary test passes; all 20 named companies resolved with full history from their listing date; held-out share 15% ± 1% of companies and within ±3 pp per sector bucket.

**P2.3 Scenario tags** · deps: none · out: `configs/scenario_tags.csv`
Start from `dev_plan/scenario_tags_seed.csv`. Verify every date against a reliable source; add the automatic slices (05 §8) as code, not rows. Ask the owner to review the episode list (doc: "an analyst can label the major episodes in a week").
Acceptance: every row has start, end, tag, source; `verified = true` for all rows used in a gate.

**P2.4 Metric library** · deps: P0.2 · out: `src/marketcast/eval/metrics.py`, `tests/test_metrics.py`
Every metric in `05` §4, vectorised (NumPy/Torch). Tests: fair CRPS equals `scoringrules.crps_ensemble(..., estimator="fair")`; CRPS of N(0,1) samples vs closed form within 1% at M = 10,000; Kupiec and Christoffersen reproduce a textbook example; DM test reproduces statsmodels-based reference.
Acceptance: all tests pass.

**P2.5 Baselines** · deps: P2.1, P2.2, P2.4 · out: `src/marketcast/baselines/`, `reports/scores/B*/`
Implement B0–B5 and BT1–3, BS1–2 (`05` §6), each producing the `Forecast` object, M = 256 for scoring. Build PIT factors for B4 and validate against IIMA (corr ≥ 0.9). Run fast-mode folds first, then all nine.
Acceptance: every baseline runs on all folds without NaNs; B2 parameters are stable (α + β < 1 for ≥ 98% of fits); runtime per fold logged.

**P2.6 Leakage suite and scorecard** · deps: P2.3, P2.5 · out: `tests/test_leakage.py`, `reports/scorecard_baselines.md`
All six leakage tests (`05` §9). Scorecard per template (`05` §10) across all test years and slices.

**P2.7 Gate G2: harness** · deps: P2.6 · out: `reports/G2_gate.md`
Criteria: one-page scorecard for all baselines across all test years and scenario slices; leakage suite green; owner sign-off. No neural training before this gate.

---

## P3 Deterministic volatility model (weeks 8–11) → Gate G3

**P3.1 Feature store** · deps: P2.7, P1.6, P1.7, P1.8 · out: `data/features/<fold>/`
Compute every channel in `03` §7–9; fit normalizers per fold on training dates and training stocks; write memory-mapped tensors per date: stock history [N, 60, F], global history [16, 60, F_g], forcings [80, F_f], masks. Cache size target ≤ 25 GB in float16; if larger, store daily feature rows and assemble windows in the data loader.
Acceptance: PIT assertion on 10,000 cells; normalized channels have median ≈ 0 and IQR ≈ 1.35 on train; validation statistics reported, not refit.

**P3.2 Graph builder** · deps: P1.5, P1.6, P2.2 · out: `src/marketcast/graph/`, `data/graph/<fold>/`
Per date: node list (MC500 members + their tree ancestors + globals), `up/down` edges from the sector map valid on t, significance w (02 §4.5; D-07), monthly `corr` edges (04 §L4), curated `global` exposure edges from `configs/edges_global.yaml`. Output as PyTorch Geometric `HeteroData` cached per month with daily node masks.
Acceptance: graph statistics per year (nodes, edges by type, max in-degree); no edge uses data after t (unit test); corr edges change by ≤ 3 of 10 per refresh.

**P3.3 Model code** · deps: P0.2 · out: `src/marketcast/model/{encoder,fabric,processor,heads,fgn,rollout}.py`, `tests/test_model_shapes.py`
Implement 04 L2, L4–L5, L7 deterministic variant; FGN code may be stubbed (zero noise). Unit tests: shape checks; permutation equivariance over stock order; masked nodes do not affect others; a node's significance prior shifts attention as expected.

**P3.4 Training loop and logging** · deps: P3.1, P3.2, P3.3 · out: `src/marketcast/train/`, experiment log (MLflow local `mlruns/` or W&B)
Loss for P3: QLIKE-consistent log-variance loss (04 L7). Log run id, config hash, git commit, data snapshot hash, fold, seed, metrics.
Acceptance: overfits a 20-date subset (loss → near zero); a full fast-mode run completes; training curve and validation QLIKE logged.

**P3.5 Fabric ablation** · deps: P3.4 · out: `reports/P3_fabric_ablation.md`
Variants: flat (no edges), tree only, tree + corr, tree + corr + global exposures. Same folds, same seeds. Keep cross-links only if they improve QLIKE for stocks whose linked neighbours had a shock in the previous 5 days (doc rule), with DM test.

**P3.6 Gate G3** · deps: P3.5 · out: `reports/G3_gate.md`
Criteria: best variant beats GARCH (B2) and LightGBM (B5) on QLIKE out of sample at h = 1, 5, 20, including the stress slices, with DM p < 0.05 on the pooled test years; held-out stocks no worse than held-in by more than 5% relative QLIKE.

---

## P4 Generative core (weeks 12–16) → Gate G4

**P4.1 FGN and sampling heads** · deps: P3.6 · out: model code + run logs
Add L6 (zero-init conditional norms), return and volatility sample heads, fair CRPS with 2 samples per input. Warm-start the processor from the best P3 run.
Acceptance: ensemble spread grows in stress slices (spread–error correlation > 0.3 across dates); CRPS beats the P3 model turned into a Gaussian with predicted variance.

**P4.2 Rollout and unrolled fine-tuning** · deps: P4.1 · out: `src/marketcast/model/rollout.py` complete
L8 algorithm; fine-tune K = 5 then K = 10. Monitor path statistics (04 L8).
Acceptance: 20-day CRPS of rolled-out paths beats the direct 20-day head of P3 or is within 2%; extreme-path share < 0.1%.

**P4.3 Seeds and full scoring** · deps: P4.2 · out: `reports/scores/MC-gen/`
Four seeds × all folds; 64-path forecasts (4 × 16).

**P4.4 Trend, sector and shock readouts** · deps: P4.3 · out: `src/marketcast/readout/`, scores for T3/T4
Path-based trend, drawdown and stress probabilities; trend cross-check head; sector aggregation and abnormal returns; shock mode with shared noise; event-study validation on 10 real past shocks (e.g. crude spikes, IL&FS, RBI surprises).
Acceptance: the head-based and path-based trend probabilities differ by < 0.1 on average and are both calibrated (reliability within bootstrap bands).

**P4.5 Curated cross-links (optional in P4, required before P6)** · deps: P4.3 · out: `configs/edges_supply.yaml`, `configs/edges_group.yaml`, ablation report
Curate supplier–customer and business-group links for the top 100 MC500 names, each with a source and `available_at`. Spillover event study per the doc (Cohen and Frazzini 2008 motivation). Keep only if they improve neighbour CRPS after shocks.

**P4.6 Gate G4** · deps: P4.4 · out: `reports/G4_gate.md`
Criteria: 90% intervals cover 88–92% in calm and in stressed slices; energy score beats DCC-GARCH (B3) with DM p < 0.05; CRPS beats B0–B5; VaR backtests pass at 95% and 99%. Trend probabilities beat the base rate on Brier and sector ranks beat sector-average, or those outputs stay off the product. After owner sign-off: open the vault once (`reports/vault_log.md`) and report the same table.

---

## P5 Text and event heads (weeks 17–20) → Gate G5

**P5.1 Event data acquisition** · deps: P4.6 · out: `data/raw/events/`, `data/pit/ref_events.parquet`
Results dates, board meetings, dividends (fills G2), corporate announcements (G5) from exchange announcement archives (browser export or a licensed feed; ask the owner before paying). Each record: isin, event type, first-seen timestamp, source, text. GDELT India filter as a second source (G14).

**P5.2 Event structuring** · deps: P5.1 · out: `data/processed/event_tokens/`
Rule-based parsers first (dates, amounts, event types). Text embeddings from a small open model whose training data predates the test period. Any LLM-based extraction is evaluated only on dates after that model's knowledge cutoff (doc rule). Deduplicate into event clusters keeping the earliest timestamp; link rumours and denials.

**P5.3 Event assimilation and heads** · deps: P5.2 · out: model code (L3, event heads), runs
Co-train event heads (results-day abnormal return, P(DD ≥ 10% in 20d), P(circuit hit)).

**P5.4 Gate G5** · deps: P5.3 · out: `reports/G5_gate.md`
Criteria: ablation shows the text module improves CRPS on event days, measured only after the text model's knowledge cutoff; event heads calibrated.

---

## P6 Live forecasting and product prototype (weeks 21–26+) → Gate G6

**P6.1 Nightly pipeline** · deps: P4.6 · out: `src/marketcast/live/`, `scripts/run_nightly.bat`
```
20:30 IST on each session: download today's raw files → validate (counts, schema, stale check) → rebuild today's rows
→ features → forecast (latest model per seed) → write forecasts/live/<date>.npz + summary JSON
→ freeze: sha256 of the file, append (date, hash, created_at, model_version) to forecasts/live/ledger.csv, commit the ledger to git
must finish before 09:15 IST next session; if inputs are stale, write a stale flag and skip (never forecast silently)
```
Schedule with Windows Task Scheduler (owner approves).

**P6.2 Live scoring** · deps: P6.1 · out: `reports/live/` daily and monthly
Score each frozen forecast as outcomes arrive (1, 5, 20 days later); compare live calibration with backtest calibration.

**P6.3 Portfolio engine (L9)** · deps: P4.6 · out: `src/marketcast/decision/`, JSON contract tests
Implements 04 L9 and the JSON in `01` §7.

**P6.4 Explanation layer** · deps: P6.3 · out: `src/marketcast/explain/`
LLM receives only the JSON; a validator extracts every number from the text and checks it against the JSON (exact match after formatting rules); failures fall back to a template sentence.

**P6.5 Household app hook** · deps: P6.3 · out: `docs/app_integration.md`
Outcome-fan data contract for the app (portfolio-level only, education framing, no stock calls).

**P6.6 Gate G6** · deps: P6.2 · out: `reports/G6_gate.md`
Criteria: ≥ 3 months of live forecasts whose 90% coverage is within ±3 pp of the backtest and whose CRPS is within 10% of backtest; zero frozen-forecast integrity failures. Legal advice obtained before anything public.

---

## Ongoing duties (every phase)

* Log every run; keep the run count for multiple-testing corrections.
* Quarterly: re-run the reference download (dated folders accumulate membership history), retrain, and record skill decay between refits.
* Keep `reports/progress_log.md` current.
* Re-check the RBI policy table after every MPC meeting.
