# 03 Features and targets

Exact definitions. Implement each formula once in `src/marketcast/features/` and `src/marketcast/targets/`, unit-test it on hand-computed examples, and never re-derive it elsewhere. Notation: i = stock, t = trading day, prices are raw ₹ from `equity_daily`, `pc_t` = NSE `prev_close` on day t (already adjusted for splits/bonus/rights on ex-dates), D_{i,t} = cash dividend with ex-date t (0 until gap G2 is filled).

## 1. Returns

```
r_{i,t}  = (close_t + D_{i,t}) / pc_t − 1            # total return when D known, else price return
ℓ_{i,t}  = ln(1 + r_{i,t})                           # log return; all aggregation uses ℓ
L_{i,t,h} = Σ_{k=1..h} ℓ_{i,t+k}                     # h-day forward log return
```

Never compute returns from back-adjusted levels. Terminal return on delisting day per `02_DATA_CONTRACT.md` §7.

## 2. Realized variance proxy (daily OHLC only; no intraday data)

Overnight plus Garman–Klass intraday, computed on the same price basis (`pc_t` is on day t's basis):

```
o_t   = ln(open_t / pc_t)                                       # overnight gap
GK_t  = 0.5·ln(high_t/low_t)² − (2·ln2 − 1)·ln(close_t/open_t)²
RV²_t = o_t² + max(GK_t, 0)
if high_t == low_t:  RV²_t = o_t² + ln(close_t/open_t)²        # no range recorded
v_{i,t} = 0.5·ln(max(RV²_t, 1e−6))                              # log realized vol (floor = 0.1% daily)
RV_{i,t,h} = sqrt(Σ_{k=1..h} RV²_{t+k});   v_{i,t,h} = ln RV_{i,t,h}
```

Index RV (Nifty 50 and sectors) uses the same formula on `indices_daily` OHLC. Before 2012-07 (no index files), use the market node aggregate return and the square of its return as RV² (flag `rv_method = close_only`).

## 3. Trailing volatility and the scaled return target

EWMA (RiskMetrics) on log returns, known at the close of t:

```
s²_{i,t} = λ·s²_{i,t−1} + (1 − λ)·ℓ²_{i,t},     λ = 0.94
init: s²_{i,t0} = sample variance of the first 20 valid ℓ;  valid only after 60 observations
σ̂_{i,t} = max( sqrt(s²_{i,t}),  floor_t ),   floor_t = max(0.004, 0.25·median_j sqrt(s²_{j,t}))
```

Primary target (doc's "volatility-scaled return", in log form for additivity; decision D-05):

```
z_{i,t+1}   = ℓ_{i,t+1} / σ̂_{i,t}
Z_{i,t,h}   = L_{i,t,h} / (σ̂_{i,t}·sqrt(h))        # used only for scoring h-day marginals
```

Targets are **never clipped**. Inputs are.

## 4. Market trend, drawdown and stress labels (built inside each fold only)

* R_{t,h} = Nifty 50 L_{t,h}; σ_{t,h} = σ̂_{N50,t}·√h. Class up / sideways / down by ±0.5σ_{t,h} (01 §3).
* DD_{t,h} = min_{1≤k≤h} (exp(L_{N50,t,k}) − 1). Events DD ≤ −0.05 and ≤ −0.10.
* Stress_t = VIX_IN_t ≥ q90_{252}(VIX_IN) or RV20_{N50,t} ≥ 1.5·median_{252}(RV20_{N50}).
* Purge: a label for t uses data through t+h, so decision dates within h days before any split boundary are dropped from the earlier split (05 §2).

## 5. Censoring and masks

**Circuit-band inference.** Applies only to stocks with no F&O contract on day t (F&O stocks have dynamic bands, not fixed circuits). With r = close/pc − 1 and bands B = {0.02, 0.05, 0.10, 0.20}:

```
upper_hit = (close == high) and any(|r − b| ≤ 0.0015 for b in B)
lower_hit = (close == low)  and any(|r + b| ≤ 0.0015 for b in B)
censored_{i,t} = upper_hit or lower_hit or market_halt_t
```

Effects: the day-t target is masked (loss weight 0); feature `circuit_carry_{i,t}` = +1/−1/0 enters day t+1 inputs (pressure carries over). Validate the rule on 50 random flagged cases against intraday news or exchange notices; target precision ≥ 90%.

**Other masks** (all boolean, all stored): `stale` (volume == 0 or OHLC all equal to pc), `session_bad` (outage/special), `history_short` (< 60 valid days), `merger_window`, `terminal_unknown`, `dividend_exdate_unadjusted` (G2 not filled and suspected dividend drop), `heldout_stock`.

Target mask = not(censored or stale or session_bad or terminal_unknown or dividend_exdate_unadjusted). Input mask per feature = feature not missing.

## 6. Loss weights

```
w_{i,t} = sqrt(median_{60d}(turnover_i)) / mean_{j ∈ U_t} sqrt(median_{60d}(turnover_j))
```

Normalized to mean 1 per date, capped at 5. This is the market's version of WeatherNext's latitude weighting.

## 7. Stock-node input features (target: 60–120 channels)

Every feature is computed from data with available_at ≤ cutoff(t). Windows are in trading days. "z(x, n)" = robust z-score of x vs its own trailing n-day median and MAD (per stock, PIT). All ratios use σ̂ from §3.

| Group | Feature (count) | Definition |
|---|---|---|
| Returns | z_t (1) | ℓ_t / σ̂_{t−1} (the encoder sees 60 lags of this) |
| | momentum (4) | L over past 5, 20, 60, 120 days / (σ̂_t·√n) |
| | 52-week position (2) | (ln close_t − ln max_252)/σ̂_t, (ln close_t − ln min_252)/σ̂_t, on a returns-reconstructed price index |
| | drawdown (2) | max drawdown over 20 and 60 days |
| | shape (3) | 60-day skewness and kurtosis of z; downside semivariance share |
| Volatility | v_t and trailing (5) | v_t; ln RV over 5, 20, 60; ln σ̂_t |
| | vol structure (3) | ln(RV5/RV60); vol of v over 20; gap o_t/σ̂_{t−1} |
| | range (2) | ln(high/low)/σ̂; close location (close−low)/(high−low) |
| Liquidity | activity (5) | z(ln turnover, 60), z(ln volume, 60), z(ln trades, 60), z(ln turnover/trades, 60), ln median turnover 60 |
| | delivery (3) | deliv_pct, z(deliv_pct, 60), deliv_pct − 20-day mean |
| | illiquidity (2) | 20-day mean of ln(\|r\|/turnover) (Amihud); stale share over 20 days |
| | status (4) | series ∈ {BE, BZ}; circuit_carry; ln(1 + days_listed); history_short |
| Derivatives | availability (1) | has_fo |
| | futures (4) | ln fut_oi, Δln fut_oi, z(Δoi, 60), annualized basis = (fut_near_settle/close − 1)·365/days_to_expiry |
| | options (4) | pcr_oi, z(pcr_oi, 60), ln ATM IV (Black-76 on near-month ATM settle), ln(IV/RV20) |
| | ban (1) | fo_ban flag (when the archive is verified) |
| Market-relative | beta and correlation (4) | EWMA beta to market node (halflife 60), corr 60 to market, residual-vol share, rank of 20-day return inside its industry (0–1) |
| Corporate | (3) | corp_action_flag today, days since last corporate action (capped), merger_window |
| Fundamentals (after G4/G5) | (≈10, later) | earnings yield, book-to-price, ROE, debt/equity, promoter holding, pledge %, last results surprise (needs consensus), all by filing date |
| Text (P5) | event tokens | attached by cross-attention, not as channels |

Missing values: set to 0 after normalization and add a mask channel per feature group (not per feature) so the model knows the group is absent (e.g. `has_fo = 0` already covers derivatives).

## 8. Global node features (about 16 nodes; one vector each, PIT-aligned by the §2 global rule)

| Node | Channels |
|---|---|
| MARKET (MC500 aggregate, Nifty 50 when available) | z, v, momentum 5/20/60, breadth (share of members with ℓ > 0), dispersion (cross-sectional std of z), average pairwise corr 60 (estimated on 100 largest) |
| VIX_IN | ln level, Δln, z(·, 252) |
| USDINR | Δln 1/5/20, ln RV20 |
| BRENT, WTI | Δln 1/5/20 |
| US_RATES | DGS2, DGS10, DFF levels; Δ 1/5/20 (bp); T10Y2Y |
| USD | DTWEXBGS Δln 1/5/20; DEXCHUS, DEXJPUS, DEXUSEU Δln 1/5 |
| US_EQ | SP500, NASDAQCOM, DJIA ℓ 1/5/20 (SP500 and DJIA on FRED cover only the last 10 years; use NASDAQCOM for long history) |
| ASIA_EQ | NIKKEI225 ℓ 1/5/20 |
| GLOBAL_VOL | ln VIXCLS, Δ |
| CREDIT | BAMLH0A0HYM2, BAMLC0A0CM levels and Δ5/20 |
| INDIA_RATES | repo (from RBI file), days since last change, direction of last change, call rate, India 10y yield |
| INDIA_MACRO | CPI yoy, IIP yoy, M3 yoy, exports yoy, imports yoy, FX reserves Δ, policy uncertainty (all lagged per lag table) |
| FLOWS | FPI equity net (₹ cr) of last complete month, scaled by market-cap proxy; 3-month sum |
| METALS | copper, aluminium, iron ore, coal monthly Δln (lagged) |
| AGRI | wheat, rice, sugar, cotton, rubber monthly Δln (lagged); later monsoon anomaly (G12) |
| POLICY_UNC | USEPUINDXD ln and z |

## 9. Forcings (known in advance for every future day t+k of the rollout)

day of week; days since previous session; is_monthly_expiry; is_weekly_expiry (index); days to next expiry; is_rbi_policy_day (pre-announced schedule only; unscheduled moves are events); is_budget_day; is_election_result_day; is_index_rebalance_effective; month; fiscal-year-end window (last 10 sessions of March); holiday-adjacent (before/after a break of ≥ 3 calendar days); rollout step k (1–20); is_results_day per stock (after G5); lock-in expiry per stock (when known).

Expiry calendars are derived from the F&O bhavcopy `expiry` column: every listed contract's expiry is public from listing, so it is PIT-safe to use as a forcing.

## 10. Normalization

* Robust per-feature scaling: (x − median)/(1.4826·MAD), statistics fitted on the **training fold and training stocks only**, stored with the fold, never refit on validation or test.
* Clip normalized inputs to [−8, 8]. Never clip targets.
* No per-stock normalization of targets beyond σ̂ (it is already the scale).

## 11. Graph-level inputs (built in P3.2; defined in 04 §L4)

Edge features: edge type one-hot, log w_j (significance prior), for correlation edges the shrunk correlation value, for curated edges a confidence score.
