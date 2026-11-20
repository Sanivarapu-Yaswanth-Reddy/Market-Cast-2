# 02 Data contract

This file defines every dataset, its schema, its point-in-time (PIT) rule and its known gaps. It is L0 of the architecture, the "market reanalysis". Most of the project's real work and most of its moat live here.

## 1. Layers on disk

```
data/
  raw/          immutable downloads (never edited; see README_DATA.md for sources)
  manifest.csv  one row per fetched file: fetched_at_utc, group, status, bytes, path, url
  processed/    tables rebuilt from raw by code (Parquet, partitioned by year where large)
  pit/          marketcast.duckdb: PIT views over processed tables + reference tables
  features/     model-ready feature tensors per fold (built by P3.1)
  targets/      targets and labels per fold (built by P2.1)
  forecasts/    frozen forecasts (backtest and live), append-only
reports/        scorecards, gate reports, progress log, vault log
```

## 2. Point-in-time rule (the most important definition in the project)

Every fact row carries:

* `valid_date`: the trading date or period the value describes;
* `available_at`: the earliest IST timestamp at which the value was public (timezone-aware, Asia/Kolkata).

**Information set for decision date t:** a fact may be used for a forecast made "at the close of t" if and only if

```
available_at ≤ cutoff(t)   where   cutoff(t) = t 15:30 IST            (news, global, macro, filings)
                           and NSE end-of-day files for date t are included (they describe the 15:30 close)
```

Consequences:

* Exchange EOD files for date t (cash bhavcopy, delivery, F&O bhavcopy, all-indices close) are part of I_t even though they are published around 18:00–20:00 IST, because they describe the session that closed at 15:30.
* An announcement published at 15:31 or later on day t belongs to day t+1.
* US/European closes for calendar date t happen after 15:30 IST on t, so they belong to I_{t+1}. For India date t, use global values dated (US date) ≤ t−1.
* The live forecast is computed in the evening of t and frozen (hashed, timestamped) before 09:15 IST on t+1.

### `available_at` rules per source

| Source | valid_date | available_at rule | Notes |
|---|---|---|---|
| NSE cash bhavcopy, delivery MTO, F&O bhavcopy, all-indices close | trade date d | d 15:30 IST (EOD exception above) | published same evening |
| NSE reference snapshots (EQUITY_L, delisted, constituents) | snapshot date | snapshot date 23:59 IST | only valid **from** that date forward; history must come from elsewhere |
| Bulk/block deals | deal date d | d 23:59 IST → use from t = d+1 | published after close |
| FRED daily US/global series (DGS10, DGS2, T10Y2Y, DFF, DTWEXBGS, DEX*, DCOIL*, VIXCLS, SP500, NASDAQCOM, DJIA, NIKKEI225, BAML*, USEPUINDXD) | US date d | d+1 06:00 IST | Nikkei closes before Indian open: still use d+1 rule for simplicity in v1; v2 may use same-day Nikkei (closes 11:30 IST) |
| FRED DEXINUS (USD/INR noon buying rate NY) | d | d+1 06:00 IST | prefer RBI reference rate later |
| FRED monthly India macro | month m | **conservative publication lag** (table below) | FRED gives latest revised values; ALFRED vintages needed for revisable series (IIP, GDP) |
| IMF commodity prices (P*USDM) | month m | end of m + 35 days | monthly averages |
| NSDL FPI monthly net investment | month m | end of m + 1 day 09:00 IST | daily series is a gap |
| RBI policy rate changes | effective date | announcement date 12:00 IST | 2025 rows flagged for verification; unscheduled changes flagged |
| IIMA factor returns | d | d 15:30 IST (treated as computable from EOD) | breakpoints use annual data; acceptable for baseline only. Own PIT factors in P2.5 |
| Curated graph edges (supplier, group) | — | date the source document became public | no edge exists before this date |
| Announcements / news (Phase 5) | event | first-seen timestamp; > 15:30 → next day | dedupe clusters keep earliest timestamp |

**Conservative monthly lags (v1, until vintages are obtained):** CPI +15 days after month end; IIP +45; trade +20; M3 +15; call rate monthly average +5; FX reserves +10; India 10y yield monthly +5; policy uncertainty +10; GDP annual (World Bank) +180. Store the lag table in `configs/pit_lags.yaml` and unit-test that no feature row violates it.

## 3. Raw inventory (state at 2026-11-16) and expected size

| Group / folder | Coverage | Expected files | State |
|---|---|---|---|
| `raw/nse/cm_bhavcopy/<YYYY>/` | 2005-01-03 → latest (old zip format to 2024-07-05; UDiFF from 2024-07-08) | ≈ 5,400 | downloading |
| `raw/nse/delivery_mto/<YYYY>/` | 2005 → latest, `MTO_YYYYMMDD.DAT` | ≈ 5,400 | queued |
| `raw/nse/fo_bhavcopy/<YYYY>/` | 2010-01 → latest (same format switch) | ≈ 4,200 | queued |
| `raw/nse/indices_close_all/<YYYY>/` | 2012-07-02 → latest | ≈ 3,500 | in progress (through Oct 2016 at last check) |
| `raw/nse/reference/20261116/` | snapshot | 6 files + 37 constituent lists | done |
| `raw/global/fred/` | decades | 40 CSVs | done (`_series_index.csv` not written; descriptions are in the downloader's FRED dict) |
| `raw/factors/iima_india/` | 1993-10 → 2025-12 | 11 CSVs | done |
| `raw/factors/ken_french/` | — | zips | **missing** (site blocked in browser; downloader retries) |
| `raw/flows/nsdl_fpi/fpi_monthly_net_investment_2002_2026.csv` | 2002-01 → 2026-11, ₹ crore | 1 file, 299 rows | done |
| `raw/macro/rbi/rbi_policy_rate_changes.csv` | 2005 → 2025-12 | 1 file, 119 rows | done (2025 dates to verify; check for 2026 changes) |
| `raw/nse/cm_volatility/`, `raw/weather/imd/` | optional (`--with-volt`, `--with-imd`) | — | not run |

Holiday 404s are not logged to the manifest; trading days are defined by the presence of a cash bhavcopy file (P1.1).

## 4. Processed tables (schemas)

Types: `date` (calendar date), `ts` (timestamp with tz Asia/Kolkata), `str`, `f64`, `i64`, `bool`. Primary keys in **bold**.

### 4.1 `processed/equity_daily` (from builder; partition by year)

| Column | Type | Meaning |
|---|---|---|
| **date**, **key** | date, str | key = ISIN, or `SYM:<symbol>` when ISIN unknown (pre-2011 rows are backfilled from the symbol's first later ISIN) |
| company_id | str | **the identity key for modelling** (D-20): first 9 characters of an equity ISIN (`INE` + 4-character issuer code + `01` equity type), so a face-value split that issues a new ISIN (serial digits change, e.g. `INE628A01036`) keeps one company. Verify on the data: for every inferred split, check the ISIN changed and the first 9 characters did not; list exceptions |
| isin, symbol, series | str | series kept: EQ, BE, BZ (BE/BZ = trade-for-trade / surveillance; keep and flag) |
| open, high, low, close, last, prev_close | f64 | raw ₹; `prev_close` is NSE's own, already adjusted on ex-dates for splits/bonus/rights |
| volume, turnover, trades | f64 | shares, ₹, trade count |
| mto_qty, deliv_qty, deliv_pct | f64 | from MTO join on (date, symbol, series) |
| adj_factor | f64 | R_t, inferred split/bonus/consolidation ratio: every earlier price is multiplied by R_t (1/k for a k-for-1 split, > 1 for a consolidation); 1.0 otherwise. **NSE's PREVCLOSE is not adjusted for these** (verified on RELIANCE 2017-09-07, HDFCBANK 2019-09-19, GARFIBRES 2025-01-03), so events are inferred from open and close gaps matching a standard ratio (builder `marketcast_build_panel.py`) and must be verified in P1.3 |
| corp_action_flag | bool | an inferred split, bonus or consolidation on this row |
| jump_unexplained | bool | close/prev_close beyond ±40% with no matching standard ratio (demerger, rights, genuine crash, relisting); not adjusted, needs the official corporate-action table (G15) |
| gap_days | f64 | calendar days since this company's previous row; > 10 means a suspension or a missing download, so no event is inferred across it |
| adj_open…adj_prev_close | f64 | back-adjusted to the latest date (**not PIT-safe for levels**: use only for plots; models use returns) |
| ret_1d | f64 | close/prev_close − 1: PIT-safe, CA-adjusted, **price return only (dividends missing)** |
| days_listed | i64 | count of prior rows for key |

Known issues to fix in P1.3: dividends are not in `prev_close` adjustment, so `ret_1d` is a price return; pre-2011 ISIN backfill misses renamed tickers; the builder concatenates all years in memory (rewrite to per-year if it fails).

### 4.2 `processed/indices_daily`
`date, index, open, high, low, close, chg, chg_pct, volume, turnover_cr, pe, pb, div_yield`. About 145 indices including `Nifty 50`, `Nifty 500`, sector indices and `India VIX`. Old names (S&P CNX Nifty, CNX …) must be mapped to current names in `configs/index_aliases.yaml` (P1.1).

### 4.3 `processed/fo_daily`
Per (date, underlying): `fut_oi, fut_chg_oi, fut_near_settle, fut_near_expiry, call_oi, put_oi, pcr_oi`. P1.7 adds: futures basis = fut_near_settle / spot − 1 annualized by days to expiry; OI z-scores; days to expiry; ATM implied vol (Black-76 inversion of near-month ATM call and put settle prices) where options exist.

### 4.4 `processed/fred_global`
Wide table on a calendar date index, raw values, no fill. PIT alignment happens in the `pit.global_asof` view (§6).

### 4.5 Reference tables built in Phase 1

| Table | Key | Columns | Built by |
|---|---|---|---|
| `ref.trading_calendar` | date | is_trading_day, session_quality ∈ {normal, special, partial, outage}, calendar_days_since_prev, is_monthly_expiry, is_weekly_expiry, is_rbi_policy_day, is_budget_day, is_election_result_day, is_index_rebalance_effective | P1.1 |
| `ref.security_master` | isin | first_date, last_date, listing_date (EQUITY_L), delisting_date and reason (delisted.csv, Latin-1), company_name, face_value | P1.2 |
| `ref.symbol_history` | (isin, valid_from) | symbol, series, valid_to | P1.2 |
| `ref.corporate_actions` | (isin, ex_date, type) | type ∈ {split, bonus, rights, dividend, merger, demerger}, factor, cash_amount, source, available_at | P1.3 |
| `ref.universe_mc500` | (rebalance_date, isin) | rank, avg_traded_value_6m, in_universe, effective_from, effective_to | P1.5 |
| `ref.sector_map` | (isin, valid_from) | macro_sector, sector, industry, basic_industry, source, confidence | P1.6 |
| `ref.significance` | (date, isin) | float_mcap (when available), traded_value_med60, w_parent (share within parent node), method | P3.2 |
| `ref.edges_curated` | (src, dst, type, valid_from) | weight, source_doc, available_at | P3.2 / P4 |

## 5. Universe definition: MC500 (decision D-03)

Historical Nifty 500 membership is not on disk (gap G1). Instead of waiting, define a self-built PIT universe that imitates the index methodology:

1. Rebalance dates: last trading day of March and September (Nifty broad indices reconstitute semi-annually).
2. Eligible: series EQ (or BE/BZ if it traded as EQ within the last 6 months), listed ≥ 6 months (≥ 3 months for very large new listings, rank ≤ 100), traded on ≥ 90% of sessions in the last 6 months, not an ETF (eq_etfseclist) or REIT/InvIT.
3. Rank by 6-month average daily traded value (`turnover`). Replace with float-adjusted market cap once shares-outstanding and free-float history exist (gap G4).
4. Top 500 enter. Buffer: an incumbent stays while rank ≤ 600. A new name enters if rank ≤ 400 or a slot is free.
5. Effective from the next trading day after the rebalance date until the next rebalance.
6. A member that delists mid-period stays until its last trading day, then takes a terminal return (P1.3).

Validation: overlap with the 2026-11-16 Nifty 500 constituent snapshot should be ≥ 80% by count and ≥ 95% by weight. Report the gap; it is expected because ranking differs.

Training may use a wider MC1000 (same rule, top 1,000) as an ablation; evaluation is always on MC500.

## 6. PIT store design (DuckDB)

`data/pit/marketcast.duckdb` holds views and small tables. Large tables stay as Parquet and are attached with `read_parquet(...)`.

Required access patterns (implement in `src/marketcast/pit/store.py`, all covered by tests):

```python
store.panel(start, end, universe="MC500", cols=[...])          # EOD panel, rows already PIT by construction
store.asof(table, at_ts, keys=None)                             # latest row per key with available_at <= at_ts
store.global_for_india_date(d)                                   # global values with US date <= d-1 (rule §2)
store.macro_asof(d)                                              # monthly series with lag table applied
store.universe(d, name="MC500")                                  # members effective on d
store.sector_tree(d)                                             # sector_map rows valid on d
```

DuckDB pattern for as-of joins:

```sql
SELECT p.date, p.isin, m.value
FROM panel p
ASOF LEFT JOIN macro_long m
  ON m.series = 'INDCPIALLMINMEI' AND p.cutoff_ts >= m.available_at
```

## 7. Data edge-case rules (from the doc; each becomes a test)

| Case | Rule |
|---|---|
| Split, bonus, consolidation | inferred ratio in `adj_factor` (NSE PREVCLOSE is unadjusted); verify every inferred event against a known-event list and the official table (G15); the builder skips prices below 5 and gaps over 10 days; return across an event is `close / (prev_close × adj_factor) − 1` |
| Rights issues, small ratio changes | invisible to the ratio rule (price effect under about 5%); stay in the return; use G15 when available |
| Special dividends | total-return target needs dividends (gap G2); until then flag ex-dates where `ret_1d < −3%` and the stock has an announced dividend > 2% of price, and mask those targets |
| Mergers (e.g. HDFC Ltd into HDFC Bank, July 2023) | retire absorbed ISIN at last trading day with terminal return from swap ratio; carry curated edges to survivor; flag a ±20-day merger window |
| Demergers (e.g. Jio Financial from Reliance, July–Aug 2023) | spawn child node with inherited sector/group edges; the ex-date shows an unmatched jump, so the row is flagged `jump_unexplained` and its target is masked until the official table (G15) or a curated entry supplies the adjustment |
| Delistings and buyouts | keep in universe to last trading day; terminal return = buyout price / last close − 1, else −100% for liquidation, else 0 with `terminal_unknown` flag (exclude from targets, report count) |
| New listings | mask history < 60 days; borrow parent industry state; add lock-in expiry forcings when known |
| Ticker changes | company_id key; symbol history table |
| ISIN change on face-value split | company_id (ISIN[:9]) links old and new ISINs; `ref.isin_lineage(company_id, isin, valid_from, valid_to)` |
| Illiquid / stale | `stale = volume == 0 or (high == low == close == prev_close)`; liquidity filter in universe; EWMA volatility floor (03 §3) |
| Restated filings | store original and restated versions with their own available_at |
| After-close announcements | > 15:30 IST → next trading day |
| NSE vs BSE | NSE is primary for every stock in v1; BSE only fills gaps (not downloaded yet) |
| Circuit limits | infer band hits (03 §5); censored targets |
| Market-wide circuit breakers (e.g. 2020-03-13 and 2020-03-23 halts; 2009-05-18 upper circuit) | censoring event for all nodes that day |
| ASM/GSM surveillance, trade-for-trade (BE/BZ) | series flag now; ASM/GSM history is a gap (G7) |
| F&O ban | `fo_secban` archive files (verify URL in P1.7); ban flag feature |
| T+1 settlement (completed 2023-01-27), expiry-day shifts | rule-change dates as forcings; skill reported before/after |
| Outages (NSE, 2021-02-24) | session_quality = outage; drop day from targets |
| Muhurat and weekend budget sessions | session_quality = special; exclude from targets; `calendar_days_since_prev` forcing |

## 8. Gaps and how to close them (priority order)

| ID | Gap | Needed by | Route | Fallback for v1 |
|---|---|---|---|---|
| G1 | Historical index membership | eval realism | NSE/niftyindices "Index changes" press releases compiled into a dated table | MC500 rule (§5) |
| G2 | Dividends history (total returns) | P2 targets | BSE corporate actions archive; NSE corporate-actions exports (manual, browser); company filings | price returns + ex-date masking; quantify bias (dividend yield of Nifty ≈ 1–1.5%/yr, small at 1–20 day horizons) |
| G3 | NSE four-level industry classification, current and history | P3.2 | NSE "Industry Classification" document (macro-economic sector → sector → industry → basic industry); dated snapshots from now on | constituent-list `Industry` labels as the industry level; manual mapping upward; delisted names mapped by hand/heuristic with `confidence` |
| G4 | Shares outstanding, free float, promoter holding and pledges | P3.2 significance, flows layer | quarterly shareholding-pattern filings (BSE XBRL) | 60-day median traded value as significance |
| G5 | Results dates, board meetings, announcements | P5 event heads, forcings | NSE/BSE corporate announcements (browser export or licensed feed); BSE XBRL financials | none: event heads wait for this |
| G6 | Daily FPI and DII flows | flows features | NSDL daily FPI reports (page per day); NSE publishes only the latest day of DII | monthly FPI (on disk) as a slow input |
| G7 | ASM/GSM history, circuit band history | censoring, features | NSE archives where available; otherwise infer | inferred band hits |
| G8 | Consensus estimates, business-group links | expectations layer, group edges | paid (CMIE Prowess, Refinitiv, Bloomberg, Capitaline); group links can be curated by hand for the top 100 | omit; ablation later |
| G9 | Macro vintages (ALFRED / MOSPI release calendars) | honest macro features | ALFRED vintages; MOSPI and RBI release calendars | conservative lag table (§2) |
| G10 | India VIX and Nifty before 2012-07 | longer index history | NSE/niftyindices historical downloads | market node built from MC500 cap/traded-value-weighted aggregate; CBOE VIX before 2012 |
| G11 | Ken French factors | global factor context | the downloader's g_factors retry, or a manual browser download | not needed for v1 |
| G12 | IMD rainfall, reservoir levels | weather nodes | `--with-imd` (imdlib), CWC bulletins | omit until P3 ablation slot |
| G13 | AMFI monthly MF flows, CDSL/NSDL demat counts | behavioural layer | amfiindia.com monthly reports | omit |
| G15 | Official corporate-action table (ex-dates, split and bonus ratios, rights, demergers, dividends) | P1.3 | NSE corporate-actions export via the owner's browser, or a licensed feed; BSE corporate actions as a cross-check | inferred ratios only; demergers and rights unresolved |
| G14 | News text at scale | P5 | GDELT 2.0 (BigQuery) filtered to India; exchange announcements | omit until P5 |

**Terms.** NSE data is for personal and research use; a commercial API needs a licensed feed. Cite IIMA as Agarwalla, Jacob and Varma (2013). Never redistribute raw files.
