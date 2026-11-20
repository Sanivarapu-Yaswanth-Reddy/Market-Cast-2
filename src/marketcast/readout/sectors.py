"""T4 readouts (01 §3, 04 §L7): sector scenarios are AGGREGATES of stock scenarios, never a separate head.

    R_s^m   = significance-weighted simple return of sector s's member stocks in scenario m over the horizon
    A_s^m   = R_s^m - R_market^m                                   (abnormal return)
    outlook = rank sectors by P(A_s is among the worst 3 sectors)  across the day's scenarios
    shock   = re-run the SAME noise draws with a shock token; Delta_s = mean_m (A_s^{m,shock} - A_s^{m,base})

Because both runs of the shock mode share every noise draw, the difference isolates the shock ("model-implied",
to be validated against real past shocks with an event study, 06 P4.4). The shock-token input path belongs to
the event layer (Phase 5); `shock_delta` below is the readout half and already works on any two path sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sector_returns(ell: np.ndarray, weight: np.ndarray, sector: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ell [M, H, N], weight [N], sector [N] integer codes -> (R [M, S], market R [M], sector ids [S]) at horizon h.

    Sectors with no member on this date are dropped (ids tells which remain).
    """
    r = np.expm1(ell[:, :h].astype(np.float64).sum(axis=1))                 # [M, N] h-day simple returns
    w = np.asarray(weight, dtype=np.float64)
    ids = np.unique(sector)
    onehot = (sector[:, None] == ids[None, :]).astype(np.float64) * w[:, None]      # [N, S]
    rs = (r @ onehot) / onehot.sum(axis=0)
    rm = (r * w).sum(axis=1) / w.sum()
    return rs, rm, ids


def outlook(ell: np.ndarray, weight: np.ndarray, sector: np.ndarray, names: list[str], h: int = 20, k: int = 3) -> pd.DataFrame:
    """Per sector: P(in worst k), P(in best k), median and 5-95% abnormal return. Sorted most-exposed first."""
    rs, rm, ids = sector_returns(ell, weight, sector, h)
    a = rs - rm[:, None]
    order = np.argsort(a, axis=1)                                            # ascending: worst first
    worst = np.zeros_like(a, dtype=bool)
    best = np.zeros_like(a, dtype=bool)
    rows = np.arange(a.shape[0])[:, None]
    kk = min(k, a.shape[1])
    worst[rows, order[:, :kk]] = True
    best[rows, order[:, -kk:]] = True
    q = np.quantile(a, [0.05, 0.5, 0.95], axis=0)
    out = pd.DataFrame({"sector": [names[i] for i in ids], "p_worst3": worst.mean(axis=0), "p_best3": best.mean(axis=0),
                        "abn_p05": q[0], "abn_p50": q[1], "abn_p95": q[2], "vol": rs.std(axis=0)})
    return out.sort_values("p_worst3", ascending=False, ignore_index=True)


def shock_delta(ell_base: np.ndarray, ell_shock: np.ndarray, weight: np.ndarray, sector: np.ndarray, names: list[str], h: int = 5) -> pd.DataFrame:
    """Delta_s = mean over scenarios of (abnormal return under the shock - abnormal return in the base run).

    The two path sets MUST come from the same noise draws; the paired difference then has far lower Monte-Carlo
    error than a difference of two independent means (its standard error is reported).
    """
    if ell_base.shape != ell_shock.shape:
        raise ValueError("base and shock paths must have identical shapes (same noise draws)")
    rb, mb, ids = sector_returns(ell_base, weight, sector, h)
    rs, ms, _ = sector_returns(ell_shock, weight, sector, h)
    d = (rs - ms[:, None]) - (rb - mb[:, None])
    out = pd.DataFrame({"sector": [names[i] for i in ids], "delta_abn": d.mean(axis=0), "se": d.std(axis=0, ddof=1) / np.sqrt(len(d)),
                        "delta_market": float((ms - mb).mean())})
    return out.sort_values("delta_abn", ignore_index=True)
