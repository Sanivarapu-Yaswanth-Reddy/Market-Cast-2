"""Market trend, drawdown and stress labels (01 §3, 03 §4).

All three use FIXED rules (no fitted parameter), so they are the same in every fold; what makes them
"fold-local" is that the fold logic (eval/folds.py) drops decision dates whose label window crosses a split
boundary. Labels look FORWARD by construction: they are targets, never features.

    trend(t, h)     R = sum_{k=1..h} l_{t+k};  s = sigma_hat_t * sqrt(h)
                    up if R > +thr*s, down if R < -thr*s, else sideways        (thr = 0.5; report 0.25/0.5/0.75)
                    classes: 0 = down, 1 = sideways, 2 = up;  -1 = undefined
    drawdown(t, h)  DD = min_{1<=k<=h} (exp(sum_{j<=k} l_{t+j}) - 1);  events DD <= -5% and DD <= -10%
    stress(t)       VIX_t >= trailing-252 90th percentile  OR  RV20_t >= 1.5 x trailing-252 median of RV20
                    (a same-day state label; the forecast target is stress at t+h)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketcast.targets.returns import forward_sum

TREND_THRESHOLD = 0.5
CLASS_NAMES = ["down", "sideways", "up"]


def trend_labels(ell: np.ndarray, sigma: np.ndarray, h: int, thr: float = TREND_THRESHOLD) -> np.ndarray:
    """int8 [T]: class of the h-day forward market return relative to +-thr * sigma_t * sqrt(h); -1 if undefined."""
    r = forward_sum(ell, h)
    band = thr * np.asarray(sigma, dtype=float) * np.sqrt(h)
    lab = np.where(r > band, 2, np.where(r < -band, 0, 1)).astype(np.int8)
    lab[~np.isfinite(r) | ~np.isfinite(band)] = -1
    return lab


def trend_from_paths(cum_ell: np.ndarray, sigma_t: float, h: int, thr: float = TREND_THRESHOLD) -> np.ndarray:
    """Path-based P(down), P(sideways), P(up): cum_ell [M] = each scenario's h-day market log return."""
    band = thr * sigma_t * np.sqrt(h)
    return np.array([(cum_ell < -band).mean(), ((cum_ell >= -band) & (cum_ell <= band)).mean(), (cum_ell > band).mean()])


def forward_drawdown(ell: np.ndarray, h: int) -> np.ndarray:
    """DD_{t,h} [T] = worst cumulative simple return reached within the next h sessions (<= 0 or NaN)."""
    ell = np.asarray(ell, dtype=float)
    t = len(ell)
    out = np.full(t, np.nan)
    for i in range(t - h):
        w = ell[i + 1: i + 1 + h]
        if np.isfinite(w).all():
            out[i] = min(np.expm1(np.cumsum(w)).min(), 0.0)
    return out


def drawdown_from_paths(ell_paths: np.ndarray) -> np.ndarray:
    """Per-scenario drawdown: ell_paths [M, H] daily log returns -> [M] min_k (exp(cumsum) - 1), capped at 0."""
    return np.minimum(np.expm1(np.cumsum(ell_paths, axis=1)).min(axis=1), 0.0)


def stress_labels(vix: pd.Series, rv2: pd.Series) -> pd.DataFrame:
    """Fixed-rule stress regime per session (both legs reported; the path version can only use the RV leg)."""
    rv20 = np.sqrt(rv2.rolling(20, min_periods=10).mean())
    vix_leg = vix >= vix.rolling(252, min_periods=120).quantile(0.90)
    rv_leg = rv20 >= 1.5 * rv20.rolling(252, min_periods=120).median()
    out = pd.DataFrame({"rv20": rv20, "vix_leg": vix_leg.fillna(False), "rv_leg": rv_leg.fillna(False)})
    out["stress"] = out["vix_leg"] | out["rv_leg"]
    # calm slice of 05 §8: VIX below its trailing-252 30th percentile
    out["calm"] = (vix < vix.rolling(252, min_periods=120).quantile(0.30)).fillna(False)
    return out
