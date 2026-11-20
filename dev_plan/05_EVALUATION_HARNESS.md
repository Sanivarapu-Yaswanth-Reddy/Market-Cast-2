# 05 Evaluation harness ("MarketBench")

Built before the model, trusted more than the model. A model that only looks good inside a weak harness is worth nothing. Code lives in `src/marketcast/eval/`; every metric has unit tests against hand-computed or reference-library values (`scoringrules`, `properscoring`, `arch`).

## 1. Walk-forward folds

Decision dates start 2008-01-01 (2005–2007 is warm-up for 252-day features). Derivatives channels are masked before 2010; India VIX and official index channels before 2012-07.

| Fold | Train decision dates | Validation | Test |
|---|---|---|---|
| F1 | 2008-01-01 → 2015-12-31 | 2016 | 2017 |
| F2 | → 2016 | 2017 | 2018 |
| F3 | → 2017 | 2018 | 2019 |
| F4 | → 2018 | 2019 | 2020 (COVID) |
| F5 | → 2019 | 2020 | 2021 |
| F6 | → 2020 | 2021 | 2022 |
| F7 | → 2021 | 2022 | 2023 |
| F8 | → 2022 | 2023 | 2024 |
| F9 | → 2023 | 2024 | 2025 |
| **Vault** | → 2024 | 2025 | **2026-01-01 → latest** (sealed; 00 rule 6) |

Fast mode for development: F2, F4, F6, F8 only. Gates always use all nine folds.

## 2. Purge and embargo

Horizon H = 20 trading days, embargo E = 20 trading days. With sessions indexed by n:

```
last_train_decision ≤ first_val_decision − (H + E)        # target windows cannot touch validation, plus a 20-session gap
last_val_decision   ≤ first_test_decision − (H + E)
```

Labels (trend, drawdown, stress) and every fitted statistic (normalizers, correlation edges, significance where fitted, baseline parameters) are built inside each fold from that fold's training data only. Unit test: for every fold, max(train target end) < min(val decision) − E.

## 3. Held-out stocks

Two layers, both keyed on `company_id` (02 §4.1, decision D-20), never on a single ISIN:

1. **Named 20** (`dev_plan/heldout_named_20.csv`): one company from each of 19 NSE industries, with Financial Services taking two because it is the largest; 5 large, 9 mid and 6 small caps by the 2026 index lists. All were listed on NSE before 2008, so they span every test fold and most tagged episodes. Mega-caps that dominate the market node (top Nifty 50 weights) were deliberately left out, so holding names out does not distort the market aggregate. These names are the headline test of generalization rather than memorization. Report them as their own slice, and also one by one.
2. **Stratified 15%**: bucket every remaining company ever in MC500 by (sector, size tercile by median traded value over its life). Inside each bucket sort by `sha256(company_id)` and hold out enough that, with the named 20, about 15% of companies are held out in each bucket.

Held-out companies never appear in training batches, normalizer fits, correlation-edge statistics, significance fits or the MC500 market aggregate used as a training feature. The official Nifty indices still include some of them, which is accepted: an index level carries no stock-specific information. At test time they are inserted into the graph (inductive inference). Every scorecard reports held-in, held-out and named-20 results separately. The combined list is frozen in `configs/heldout_companies.txt` and versioned.

## 4. Metrics

All scores are computed per (date, stock, horizon), stored in `reports/scores/<model>/<fold>.parquet`, then aggregated with liquidity weights (03 §6) and unweighted.

| Metric | Definition | Used for |
|---|---|---|
| Fair CRPS | `(1/M)Σ|x^m − y| − 1/(2M(M−1))Σ_{m≠m'}|x^m − x^{m'}|` | z at h = 1, 5, 20 (h-day uses Z_{i,t,h}); v |
| Energy score (fair) | `(1/M)Σ‖x^m − y‖ − 1/(2M(M−1))Σ_{m≠m'}‖x^m − x^{m'}‖` on vectors | joint: Nifty 50 members, each sector basket (10–30 names), 20 random 20-stock portfolios per date |
| Variogram score (p = 0.5) | `Σ_{i<j}(|y_i − y_j|^p − (1/M)Σ_m|x_i^m − x_j^m|^p)²` | same baskets; sensitive to correlation errors |
| Correlation error | ‖C_forecast − C_realized‖_F / N over 20-day windows, stress months reported separately | joint |
| QLIKE | `RV²/σ²_f − ln(RV²/σ²_f) − 1`, σ²_f = ensemble mean of exp(2v̂) aggregated to h | volatility |
| Coverage | share of outcomes inside central 50/80/90/95% intervals | calibration |
| PIT | u = (1/M)Σ1[x^m < y] + U·(1/M)Σ1[x^m = y]; 10-bin histogram; report max bin deviation | calibration |
| Kupiec POF | `LR_uc = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x}(x/T)^x] ~ χ²₁` | VaR 95/99 |
| Christoffersen | `LR_ind` from violation transition counts (n00, n01, n10, n11); `LR_cc = LR_uc + LR_ind ~ χ²₂` | VaR clustering |
| Brier (multi-class) | Σ_c (p_c − o_c)²; skill = 1 − BS/BS_base-rate | trend, drawdown, stress |
| Log score | −ln max(p_observed, 1e−4) | trend |
| Reliability | 10-bin calibration curve with block-bootstrap bands | trend, sectors |
| Spearman ρ, top-3/bottom-3 hit rate | predicted vs realized sector abnormal returns | sectors |
| Diebold–Mariano | mean loss difference across dates; Newey–West HAC with lag H + 5, or circular block bootstrap (block 20) | every model-vs-baseline claim |
| Deflated Sharpe ratio | Bailey & López de Prado (2014), with N = number of logged configurations | any portfolio result |

Multi-day VaR and trend tests use non-overlapping windows (every h-th date) or block bootstrap; overlapping labels make the effective sample about 1/h of the row count.

## 5. Economic-value test

Reference portfolio: MC500 members, inverse-forecast-volatility weights, scaled to a 15% annualized target volatility using each model's forecast portfolio volatility, rebalanced weekly (last session of the week), long-only, max 5% per name. Costs per side (configurable in `configs/costs.yaml`; the owner should confirm current rates): brokerage 0.03%, STT 0.1% (delivery, both sides), stamp duty 0.015% (buy), exchange and SEBI fees ≈ 0.004%, GST 18% on brokerage and fees, slippage 0.05%. Compare across models: realized volatility vs target, max drawdown, 95% CVaR of weekly returns, turnover, and Sharpe with DSR. The claim is "lower drawdowns with the same target", never "higher returns".

## 6. Baselines (all emit the same forecast object, §7)

| ID | Baseline | Specification |
|---|---|---|
| B0 | EWMA random walk | z ~ Student-t(ν) (ν fitted per fold; Gaussian variant too); joint via EWMA correlation of z (λ = 0.97) with Ledoit–Wolf shrinkage; volatility forecast = σ̂ (persistence) |
| B1 | Filtered historical simulation | sample M past dates from the last 500 sessions; take that date's cross-sectional vector of z (joint structure preserved), multiply by current σ̂; multi-day via stationary bootstrap (mean block 5); missing stocks use their industry's mean z that day |
| B2 | GARCH(1,1), GJR-GARCH(1,1,1) | per stock, skew-t innovations (`arch`), rolling 1,000-day window (min 500), refit monthly; multi-step by simulation; joint via B0 correlation (Gaussian copula) |
| B3 | DCC-GARCH(1,1) | GJR marginals + DCC on standardized residuals; for N ≈ 500 use composite likelihood over 2,000 random pairs with correlation targeting (Engle, Shephard and Sheppard 2008); refit monthly; simulate |
| B4 | Factor model | PIT daily factors built from the panel: MKT, SMB, HML, WML (validate: correlation with IIMA factors ≥ 0.9 over overlapping dates) + industry factors; EWMA betas (250 days); factor covariance EWMA (λ = 0.97), multivariate-t simulation; residuals GARCH(1,1), independent |
| B5 | LightGBM quantile | same stock features as the model (no graph), separate models per target and horizon, quantiles {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99}; sort to fix crossing; piecewise-linear inverse CDF with exponential tails → samples; joint via B0 Gaussian copula |
| B6 | Zero-shot foundation models (optional) | Chronos-Bolt and TimesFM on each stock's return series → marginal samples; copula as B5 |
| BT1–3 | Trend baselines | training-fold base rate; momentum logistic on 20-day Nifty momentum; 2-state Markov-switching model (statsmodels) simulated forward |
| BS1–2 | Sector baselines | sector-beta (β_s × B0 market path → sector paths); sector-average (each sector's training-fold mean abnormal return and volatility, unconditional) |

## 7. Forecast object and storage

`src/marketcast/eval/forecast.py`:

```python
@dataclass
class Forecast:
    model_id: str; fold: str; date: date; isins: np.ndarray       # [N]
    ell: np.ndarray        # [M, H, N] float32 sampled daily log returns (scored in memory)
    v: np.ndarray          # [M, H, N] float32 sampled daily log realized vol
    index_ell: dict[str, np.ndarray]   # optional official-index paths, [M, H]
    meta: dict             # config hash, git commit, data snapshot hash, seed list
```

Score in memory, then persist per-(date, stock, horizon) scores. Persist full paths (float16) only for every 5th test date and for all tagged scenario dates, to keep disk use near 1–2 GB per model.

## 8. Slices

Every metric is reported for: all days; each tag in `dev_plan/scenario_tags_seed.csv` (after owner review); stress vs calm (fixed rule, 01 §3; calm = VIX_IN below its trailing 30th percentile); results seasons (approx. 10 Jan–15 Feb, 10 Apr–20 May, 10 Jul–15 Aug, 10 Oct–15 Nov); expiry days; RBI policy days; budget days; size terciles; F&O vs non-F&O; held-in vs held-out stocks; before vs after rule changes (T+1 2023-01-27; SEBI F&O changes from 2024-11-20).

## 9. Leakage tests (run in CI on every change to data or features)

1. **PIT assertion:** sample 10,000 (date, stock, feature) cells; assert every source row has available_at ≤ cutoff(date).
2. **Truncation invariance:** rebuild features from a raw snapshot truncated at date d; forecasts for dates ≤ d must be identical (bitwise for deterministic parts, same seeds for stochastic parts).
3. **Canary:** add a feature equal to the next-day target; the PIT assertion must fail.
4. **Random-label control:** a model trained on permuted targets must score no better than B0.
5. **Fold-boundary test:** §2 inequality for every fold.
6. **Held-out isolation:** no held-out ISIN in any training batch, normalizer fit or edge statistic.

## 10. Scorecard and gate report templates

`reports/scorecard_<model>_<date>.md`: one table per target with rows = models (B0–B6, MarketCast variants) and columns = metric × horizon, value ± 95% block-bootstrap CI, DM p-value vs best baseline, then the slice table (rows = slices, columns = key metric per model), then calibration plots (PIT, coverage, reliability) saved as PNG next to it.

`reports/G<n>_gate.md`:
```
Gate: G<n> <name>          Date:          Model/data versions:
Criteria (copied from 06_BUILD_PLAN.md):
| criterion | threshold | measured | CI | pass? |
Slices that fail:
Runs logged since last gate: <count>   (feeds DSR / multiple testing)
Known caveats:
Decision requested from owner: pass / fail / pass with conditions
```
