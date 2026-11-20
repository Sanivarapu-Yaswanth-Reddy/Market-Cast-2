"""T3 readouts from sample paths (01 §3, 06 P4.4): market trend odds, drawdown odds and stress odds.

All inputs are scenario paths of the MARKET's daily log return, [M, H]. Two ways to get them:
    * `market_paths_from_members`: significance-weighted aggregate of the simulated stock returns (always
      available; by construction consistent with every stock and sector scenario), or
    * the model's market head (`mkt_z` x sigma), which targets the official Nifty 50.
Gate G4 compares the two (they should differ by < 0.1 on average and both be calibrated).

Nothing here is a model: it is counting scenarios. P(event) = share of paths in which the event happens.
"""
from __future__ import annotations

import numpy as np

from marketcast.targets.labels import TREND_THRESHOLD, drawdown_from_paths, trend_from_paths

STRESS_MULT = 1.5


def market_paths_from_members(ell: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """ell [M, H, N] stock log returns, weight [N] significance (sums to anything) -> market log returns [M, H].

    Simple returns are aggregated (a portfolio's return is the weighted mean of simple, not log, returns) with
    weights fixed at the decision date, then converted back to logs.
    """
    w = np.asarray(weight, dtype=np.float64)
    w = w / w.sum()
    return np.log1p((np.expm1(ell.astype(np.float64)) * w).sum(axis=-1))


def trend_probabilities(mkt_ell: np.ndarray, sigma_t: float, horizons=(5, 20), thr: float = TREND_THRESHOLD) -> dict:
    """{h: {"down": p, "sideways": p, "up": p}} with the fixed +-thr * sigma * sqrt(h) rule; probabilities sum to 1."""
    cum = np.cumsum(mkt_ell, axis=1)
    out = {}
    for h in horizons:
        if h <= cum.shape[1]:
            p = trend_from_paths(cum[:, h - 1], sigma_t, h, thr)
            out[h] = {"down": float(p[0]), "sideways": float(p[1]), "up": float(p[2])}
    return out


def drawdown_probabilities(mkt_ell: np.ndarray, horizons=(5, 20), levels=(0.05, 0.10)) -> dict:
    """{h: {"5pct": P(DD <= -5%), "10pct": ...}}: path-based, the worst point inside the horizon counts."""
    out = {}
    for h in horizons:
        if h <= mkt_ell.shape[1]:
            dd = drawdown_from_paths(mkt_ell[:, :h])
            out[h] = {f"{int(lv * 100)}pct": float((dd <= -lv).mean()) for lv in levels}
    return out


def stress_probability(mkt_ell: np.ndarray, hist_rv2: np.ndarray, median_rv20: float, horizons=(5, 20)) -> dict:
    """P(stress at t+h) by the RV leg of the fixed rule: RV20 >= 1.5 x trailing-252 median of RV20.

    hist_rv2 [>= 20] = the market's last realised daily variances (oldest first); simulated days append their
    squared return (no intraday range is simulated, so this is the close-to-close proxy). India VIX is not
    simulated, so the VIX leg is absent here; scorecards must report that difference (01 §3).
    """
    m = mkt_ell.shape[0]
    hist = np.broadcast_to(np.asarray(hist_rv2, dtype=np.float64)[-20:], (m, 20))
    full = np.concatenate([hist, mkt_ell.astype(np.float64) ** 2], axis=1)
    out = {}
    for h in horizons:
        if h <= mkt_ell.shape[1]:
            rv20 = np.sqrt(full[:, h: h + 20].mean(axis=1))
            out[h] = float((rv20 >= STRESS_MULT * median_rv20).mean())
    return out
