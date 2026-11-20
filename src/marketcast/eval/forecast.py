"""The forecast object every model and baseline emits, the realised "truth" it is scored against, and the
scorer (05 §4, §7). One scorer for everything is the point of MarketBench: a model cannot look good by
being scored differently.

FORECAST FORMS
    path form      ell [M, H, N] daily log returns and v [M, H, N] daily log realized vol  (MarketCast, B0, B1, B2)
    marginal form  zh {h: [M, N]} samples of the h-day scaled return and var_h {h: [N]} variance forecasts
                   (B5 LightGBM, the deterministic Phase 3 model: no day-by-day paths)
    A path forecast is converted to the marginal form inside the scorer, so both are scored identically.

SCORED QUANTITIES per (date, stock, horizon h in 1/5/20)
    Z_{t,h} = sum_{k<=h} l_{t+k} / (sigma_hat_t sqrt(h))      -> fair CRPS, PIT, interval hits (50/80/90/95), VaR hits
    RV2_{t,h} = sum_{k<=h} RV2_{t+k}                           -> QLIKE against the ensemble-mean variance
    plus per date: fair energy score and variogram score of the 50 most significant valid stocks (joint skill)
A stock-horizon is scored only if every day of the window is a valid target (no partial windows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marketcast.eval import metrics as MT

HORIZONS = (1, 5, 20)
LEVELS = (0.5, 0.8, 0.9, 0.95)
BASKET = 50


@dataclass
class Forecast:
    model_id: str
    fold: str
    date: pd.Timestamp
    keys: np.ndarray                                   # [N] company keys (never tickers)
    ell: np.ndarray | None = None                      # [M, H, N] float32 sampled daily log returns
    v: np.ndarray | None = None                        # [M, H, N] float32 sampled daily log realized vol
    zh: dict[int, np.ndarray] | None = None            # marginal form: h -> [M, N]
    var_h: dict[int, np.ndarray] | None = None         # marginal form: h -> [N] forecast of sum RV2 over h days
    index_ell: dict[str, np.ndarray] = field(default_factory=dict)   # optional official-index paths [M, H]
    meta: dict = field(default_factory=dict)           # config hash, git commit, data snapshot hash, seeds


@dataclass
class Truth:
    """What actually happened after decision date t for the forecast's stocks."""
    ell: np.ndarray        # [H, N] realised daily log returns t+1..t+H (NaN = none)
    rv2: np.ndarray        # [H, N]
    mask: np.ndarray       # [H, N] bool: day is a valid target
    sigma: np.ndarray      # [N] sigma_hat_t
    weight: np.ndarray     # [N] liquidity weight (03 §6)
    heldout: np.ndarray    # [N] bool


def truth_for(panel, row: int, cols: np.ndarray, horizon: int = 20) -> Truth:
    """Realised outcomes for decision row `row`. The only place evaluation code reads the future."""
    t_len = len(panel.dates)
    end = min(row + horizon, t_len - 1)
    n, got = len(cols), end - row

    def fut(name, fill, dtype):
        a = np.full((horizon, n), fill, dtype=dtype)
        if got > 0:
            a[:got] = np.asarray(panel[name][row + 1: end + 1][:, cols])
        return a

    w = np.sqrt(np.nan_to_num(np.asarray(panel["med_turn60"][row, cols], dtype=np.float64), nan=0.0))
    w = np.minimum(w / max(w.mean(), 1e-12), 5.0) if n else w
    return Truth(fut("ell", np.nan, np.float64), fut("rv2", np.nan, np.float64), fut("target_mask", False, bool),
                 np.asarray(panel["sigma"][row, cols], dtype=np.float64), w, np.asarray(panel.heldout[cols], dtype=bool))


def to_marginals(fc: Forecast, sigma: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """(zh, var_h) for the standard horizons, from whichever form the forecast carries."""
    if fc.zh is not None:
        return fc.zh, (fc.var_h or {})
    if fc.ell is None:
        return {}, (fc.var_h or {})
    cum = np.cumsum(fc.ell.astype(np.float64), axis=1)                       # [M, H, N]
    zh = {h: cum[:, h - 1] / (sigma[None, :] * np.sqrt(h)) for h in HORIZONS if h <= cum.shape[1]}
    var_h = {}
    if fc.v is not None:
        cv = np.cumsum(np.exp(2.0 * fc.v.astype(np.float64)), axis=1).mean(axis=0)     # ensemble-mean variance, [H, N]
        var_h = {h: cv[h - 1] for h in HORIZONS if h <= cv.shape[0]}
    return zh, var_h


def score_forecast(fc: Forecast, tr: Truth, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-(stock, horizon) score rows and a per-date dict of joint scores."""
    rng = rng or np.random.default_rng(0)
    zh, var_h = to_marginals(fc, tr.sigma)
    rows, joint = [], {}
    cum_ok = np.cumsum(~tr.mask, axis=0) == 0                                  # [H, N]: all days up to k are valid
    cum_ell = np.cumsum(np.nan_to_num(tr.ell), axis=0)
    cum_rv2 = np.cumsum(np.nan_to_num(tr.rv2), axis=0)
    for h in HORIZONS:
        if h > tr.ell.shape[0]:
            continue
        ok = cum_ok[h - 1] & np.isfinite(tr.sigma)
        if not ok.any():
            continue
        df = pd.DataFrame({"key": fc.keys[ok], "h": h, "weight": tr.weight[ok], "heldout": tr.heldout[ok]})
        obs = cum_ell[h - 1, ok] / (tr.sigma[ok] * np.sqrt(h))
        df["z_obs"] = obs
        if h in zh:
            x = np.asarray(zh[h], dtype=np.float64)[:, ok].T                   # [n, M] (member axis last for the metric library)
            df["crps"] = MT.crps_fair(x, obs)
            df["pit"] = MT.pit_values(x, obs, rng)
            for lv in LEVELS:
                lo, hi = np.quantile(x, [(1 - lv) / 2, 1 - (1 - lv) / 2], axis=1)
                df[f"in{int(lv * 100)}"] = (obs >= lo) & (obs <= hi)
            for q in (0.05, 0.01):                                             # VaR violation = outcome below the q-quantile
                df[f"hit{int(q * 100):02d}"] = obs < np.quantile(x, q, axis=1)
            df["spread"] = x.std(axis=1)
            df["abs_err_mean"] = np.abs(x.mean(axis=1) - obs)
            # joint skill on the most significant valid names (same basket for every model on this date)
            top = np.argsort(-tr.weight[ok])[:BASKET]
            if len(top) >= 5 and x.shape[1] >= 2:
                xs = x[top].T                                                  # [M, D]
                joint[f"energy_h{h}"] = float(MT.energy_score_fair(xs, obs[top]))
                joint[f"variogram_h{h}"] = float(MT.variogram_score(xs, obs[top], p=0.5))
                joint[f"basket_n_h{h}"] = int(len(top))
        if h in var_h:
            vf = np.asarray(var_h[h], dtype=np.float64)[ok]
            rv = cum_rv2[h - 1, ok]
            good = (vf > 0) & (rv > 0)
            df["qlike"] = np.where(good, MT.qlike(np.where(good, rv, 1.0), np.where(good, vf, 1.0)), np.nan)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out.insert(0, "date", fc.date)
    return out, joint


def save_scores(model_id: str, fold: str, split: str, per_stock: list[pd.DataFrame], per_date: list[dict]) -> tuple:
    """reports/scores/<model>/<fold>_<split>.parquet (+ _dates.parquet). Idempotent: overwrites the same run."""
    from marketcast import REPORTS
    from marketcast.util import write_parquet

    d = REPORTS / "scores" / model_id
    a = write_parquet(pd.concat(per_stock, ignore_index=True) if per_stock else pd.DataFrame(), d / f"{fold}_{split}.parquet")
    b = write_parquet(pd.DataFrame(per_date), d / f"{fold}_{split}_dates.parquet")
    return a, b
