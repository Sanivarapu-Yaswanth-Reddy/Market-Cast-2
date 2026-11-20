# 01 Product spec

## 1. Problem

Indian investors and their advisers decide from point predictions and backward-looking risk numbers, so they are routinely surprised by how much, and how *together*, their holdings can fall. Four gaps: point targets hide uncertainty; standard risk models assume yesterday's world; stocks are modelled one at a time, so crisis correlation is missed; news is read but not quantified relative to expectations.

## 2. Formal problem

Given everything publicly known at the close of trading day t (cutoff rule in `02_DATA_CONTRACT.md` §2), produce a **joint** probability distribution over the next H = 1…20 trading days of

* volatility-scaled returns z_{i,t+k} and
* log realized volatility v_{i,t+k}

for every stock i in a universe of N ≈ 500 (MC500, see `02_DATA_CONTRACT.md` §5). Represent it as M coherent sample paths, M = 64 for research scoring and up to 1,000 for product queries. All four outputs are read from the same paths, so they cannot contradict each other.

## 3. The four targets

| Target | Forecast | Horizons | Output form | Primary score | Baselines | Honest expectation |
|---|---|---|---|---|---|---|
| T1 Stock returns | z_{i,t+1..t+h}, aggregated to h-day log return | 1, 5, 20 | per-stock quantiles from paths (never a point) | fair CRPS (on z), energy score and variogram score (joint) | EWMA random walk, GARCH family, historical simulation, DCC-GARCH, factor model, LightGBM quantiles | little skill in the median; value in spread, skew, tails |
| T2 Volatility | realized vol of each stock and of Nifty 50 | 1, 5, 20 | distribution of RV | QLIKE on ensemble-mean variance; coverage of RV ranges | same | most forecastable target, main source of skill |
| T3 Market trend | P(up), P(sideways), P(down) for Nifty 50; P(drawdown ≥ 5%), P(drawdown ≥ 10%); P(stress regime) | 5, 20 | probabilities summing to 1 | Brier, log score, reliability | long-run base rate, momentum rule, 2-state regime model | direction ≈ base rate; drawdown and stress odds are where skill is plausible |
| T4 Affected sectors | each sector's abnormal return (vs market) and Δvol, in outlook mode and shock mode | 1, 5, 20 | ranked list + probabilities (e.g. P(sector in worst 3)); shock path trace | Spearman ρ, top-3/bottom-3 hit rate, calibration | sector-beta, sector-average | useful after clear shocks; unconditional daily ranks near noise |

**Trend rule (fixed before training).** Let R_{t,h} be the Nifty 50 cumulative log return over t+1…t+h and σ_{t,h} = σ̂_{N50,t}·√h (EWMA, §3 of `03_FEATURES_AND_TARGETS.md`). Up if R > +0.5σ_{t,h}; down if R < −0.5σ_{t,h}; otherwise sideways. The 0.5 may be tuned only on validation years; always report skill for thresholds {0.25, 0.5, 0.75}.

**Drawdown event.** Within horizon h, DD_{t,h} = min_{1≤k≤h} (P_{t+k}/P_t − 1). Event if DD ≤ −5% (or −10%). Path-based, read from each sample path.

**Stress regime.** Fixed-rule definition, so it never shifts with refitting: India VIX ≥ its trailing-252-day 90th percentile **or** Nifty 50 20-day realized vol ≥ 1.5× its trailing-252-day median. P(stress at t+h) = share of paths satisfying the rule at t+h (computed with simulated realized vol; India VIX is not simulated, so the path version uses the RV leg only, and that difference is reported).

**Sector modes.**
* *Outlook mode*: rank sectors by P(abnormal return in worst 3) across the day's paths.
* *Shock mode*: re-run the same noise draws with a named shock token (crude +10%, a rate surprise, a guidance cut at a bellwether), report Δ_s = mean_m(A_s^{m,shock} − A_s^{m,base}). Shared noise isolates the shock. Label results "model-implied"; validate against real past shocks with an event study.

## 4. Success criteria (must hold out of sample: walk-forward test years and held-out stocks)

1. Beat GARCH, DCC-GARCH, historical simulation and the factor model on CRPS (single stocks) and energy score (joint moves). Report Diebold–Mariano tests with block-bootstrap or HAC errors.
2. Calibrated: nominal 90% ranges cover 88–92% of outcomes, in calm and stressed slices separately; 50/80/95% reported too; PIT histograms near flat.
3. VaR backtests pass (Kupiec unconditional coverage and Christoffersen conditional coverage) at 95% and 99%, for stocks and for a set of reference portfolios.
4. A simple risk-targeted portfolio has lower drawdowns after costs using MarketCast risk than using baseline risk.
5. T3 and T4 outputs ship only if they beat their base-rate baselines (Brier for trend, sector-average for ranks). Otherwise they stay internal.

## 5. Out of scope

Intraday trading, order execution, buy/sell recommendations to the public, any claim of beating the market on returns.

## 6. Users, in build order

| Phase | User | Surface | Constraint |
|---|---|---|---|
| Research (now → Phase 5) | Owner, internal | scorecards, notebooks, CLI | no regulatory exposure |
| Household app (after Phase 6 gate) | comfortable households aspiring to the next wealth tier | "outcome fan" and "what could happen" views for goal planning | portfolio-level risk education only, no stock calls; legal review first |
| B2B risk API (later) | PMS, registered advisers, wealth platforms, lenders | portfolio stress-testing API | licensed NSE data feed required for commercial use |

## 7. Product output contract (L9; JSON emitted per portfolio query)

```json
{
  "as_of": "2026-11-16", "model_version": "mc-0.4.1+seed4", "data_snapshot": "sha256:…",
  "horizon_days": 20, "n_paths": 1000,
  "portfolio": {"value_inr": 1500000, "holdings": [{"isin": "INE…", "weight": 0.18}]},
  "outcome_range": {"p05": -0.071, "p50": 0.004, "p95": 0.058},
  "var": {"95": -0.052, "99": -0.083}, "es": {"95": -0.068, "99": -0.101},
  "prob_drawdown": {"5pct": 0.21, "10pct": 0.06},
  "risk_contributions": [{"isin": "INE…", "share_of_es95": 0.31}],
  "tail_story": {"filter": "nifty_20d < -0.05", "share_paths": 0.09, "co_falling_holdings": ["INE…"]},
  "market_trend": {"up": 0.34, "sideways": 0.38, "down": 0.28},
  "sectors_most_exposed": [{"sector": "Private Banks", "p_worst3": 0.47}],
  "flags": {"stale_inputs": false, "ood_score": 0.12, "unsupported_assets": []}
}
```

The explanation layer receives only this JSON and writes plain language; a validator checks every number in the text matches the JSON exactly, and rejects the text otherwise.

## 8. Example user sentence (illustrative numbers)

"In 90 of 100 simulated months, your portfolio ends between −7% and +6%. The worst 5% of scenarios are dominated by a market-wide sell-off in which three of your holdings fall together. Your single biggest risk is concentration in private banks."
