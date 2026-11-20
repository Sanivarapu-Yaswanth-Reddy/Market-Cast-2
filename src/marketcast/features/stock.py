"""Stock-node input channels (03 §7). ONE definition per channel, written to the feature store by build.py.

Every channel at row t uses data up to and including session t (trailing windows only), so the row is the
state a forecaster holds at the close of t. Raw (un-normalised) values are stored; the per-fold robust
normaliser is applied by the loader (features/normalize.py), never baked in.

CHANNEL GROUPS
    PRICE_CHANNELS   derived purely from the stock's own return / realized-variance history. These are the
                     channels the autoregressive rollout RECOMPUTES from simulated returns (04 §L8), so their
                     formulas are mirrored in torch by model/rollout.py::price_channels_step and the two
                     implementations are compared in tests/test_rollout_consistency.py. If you change a formula
                     here, change it there and keep the test green.
    HELD_CHANNELS    everything else (liquidity, delivery, derivatives, market-relative, corporate, masks).
                     The rollout holds these at their last observed value (decision D-13).

DEVIATIONS FROM 03 §7 (deliberate, each logged in 08 as D-24)
    * 52-week position and drawdown are scaled by sigma*sqrt(window) (not sigma) so they are O(1).
    * "drawdown n" is the distance below the trailing n-day high (not the worst peak-to-trough inside the
      window): it can be updated in O(1) per rollout step.
    * skewness / kurtosis are population moments (same formula in numpy and torch).
    * Not built (no data): fo_ban flag, merger_window, fundamentals.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from marketcast.data.wide import lag, robust_z, roll

PRICE_CHANNELS = ["z", "mom5", "mom20", "mom60", "mom120", "pos52_hi", "pos52_lo", "dd20", "dd60", "skew60", "kurt60",
                  "semi60", "v", "lnrv5", "lnrv20", "lnrv60", "ln_sigma", "rv5_60", "volofvol20", "gap", "range", "close_loc"]
HELD_CHANNELS = ["z_ln_turnover", "z_ln_volume", "z_ln_trades", "z_ln_ticket", "ln_med_turn60", "deliv_pct", "z_deliv_pct",
                 "deliv_vs_mean20", "amihud20", "stale_share20", "series_surv", "circuit_carry", "ln_days_listed",
                 "has_fo", "ln_fut_oi", "dln_fut_oi", "z_dln_fut_oi", "basis_ann", "pcr_oi", "z_pcr_oi", "ln_atm_iv", "ln_iv_rv20",
                 "beta", "corr_mkt", "resid_share", "sector_rank20", "corp_action", "since_corp_action",
                 "m_price", "m_deliv", "m_trades"]
CHANNELS = PRICE_CHANNELS + HELD_CHANNELS
# 0/1 flags and already-bounded channels are passed through un-normalised (median 0, scale 1).
PASSTHROUGH = {"series_surv", "circuit_carry", "has_fo", "corp_action", "since_corp_action", "m_price", "m_deliv", "m_trades",
               "close_loc", "semi60", "sector_rank20", "stale_share20", "deliv_pct"}
# Rollout proxy for the range channel: E[ln(H/L)] of a driftless diffusion = sigma * sqrt(4 ln 2) (Parkinson).
PARKINSON_K = float(np.sqrt(4.0 * np.log(2.0)))


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def _ln(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def price_channels(ell: np.ndarray, rv2: np.ndarray, v: np.ndarray, sigma: np.ndarray, z: np.ndarray, o: np.ndarray,
                   high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, [T, K] float32) for PRICE_CHANNELS, in order."""
    sig_prev = lag(sigma)
    yield "z", z.astype(np.float32)
    for n in (5, 20, 60, 120):
        yield f"mom{n}", _div(roll(ell, n, "sum", n // 2), sigma * np.sqrt(n))
    p = np.cumsum(np.nan_to_num(ell, nan=0.0), axis=0).astype(np.float32)      # log price index (returns-reconstructed)
    p[~np.isfinite(sigma)] = np.nan                                           # undefined before sigma is valid
    s252 = sigma * np.sqrt(252.0)
    yield "pos52_hi", _div(p - roll(p, 252, "max", 60), s252)
    yield "pos52_lo", _div(p - roll(p, 252, "min", 60), s252)
    for n in (20, 60):
        yield f"dd{n}", _div(p - roll(p, n, "max", n // 2), sigma * np.sqrt(n))
    del p
    m1, m2 = roll(z, 60, "mean", 30), roll(z * z, 60, "mean", 30)
    m3, m4 = roll(z ** 3, 60, "mean", 30), roll(z ** 4, 60, "mean", 30)
    var = np.maximum(m2 - m1 * m1, 1e-8)
    yield "skew60", ((m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5).astype(np.float32)
    yield "kurt60", ((m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) / (var * var) - 3.0).astype(np.float32)
    yield "semi60", _div(roll(np.where(z < 0, z * z, np.where(np.isfinite(z), 0.0, np.nan)), 60, "mean", 30), np.maximum(m2, 1e-8))
    del m1, m2, m3, m4, var
    yield "v", v.astype(np.float32)
    lnrv = {n: (0.5 * _ln(roll(rv2, n, "mean", max(2, n // 2)))).astype(np.float32) for n in (5, 20, 60)}
    for n in (5, 20, 60):
        yield f"lnrv{n}", lnrv[n]
    yield "ln_sigma", _ln(sigma)
    yield "rv5_60", (lnrv[5] - lnrv[60]).astype(np.float32)
    vm, vm2 = roll(v, 20, "mean", 10), roll(v * v, 20, "mean", 10)
    yield "volofvol20", np.sqrt(np.maximum(vm2 - vm * vm, 0.0)).astype(np.float32)   # population std (ddof 0)
    yield "gap", _div(o, sig_prev)
    yield "range", _div(_ln(high / np.where(low > 0, low, np.nan)), sig_prev)
    rng = high - low
    loc = np.where(rng > 0, (close - low) / np.where(rng > 0, rng, 1.0), 0.5).astype(np.float32)
    loc[~np.isfinite(close)] = np.nan
    yield "close_loc", loc


def ewm_beta(x: np.ndarray, m: np.ndarray, halflife: float = 60.0, min_obs: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EWMA beta, correlation and residual-variance share of each column of x [T, K] against market m [T].

    Exponentially weighted second moments (no mean removal: daily means are ~0), updated only on days when
    both the stock and the market have a return. PIT: row t uses rows <= t.
    """
    a = 1.0 - 0.5 ** (1.0 / halflife)
    t, k = x.shape
    sxy, sxx, smm = np.zeros(k), np.zeros(k), np.zeros(k)
    n = np.zeros(k)
    beta = np.full((t, k), np.nan, dtype=np.float32)
    corr = np.full((t, k), np.nan, dtype=np.float32)
    for i in range(t):
        ok = np.isfinite(x[i]) & np.isfinite(m[i])
        if ok.any():
            xi = np.clip(x[i, ok], -0.35, 0.35)                # same cap idea as the EWMA vol update
            sxy[ok] = (1 - a) * sxy[ok] + a * xi * m[i]
            sxx[ok] = (1 - a) * sxx[ok] + a * xi * xi
            smm[ok] = (1 - a) * smm[ok] + a * m[i] * m[i]
            n[ok] += 1
        good = n >= min_obs
        with np.errstate(divide="ignore", invalid="ignore"):
            beta[i, good] = sxy[good] / smm[good]
            corr[i, good] = sxy[good] / np.sqrt(sxx[good] * smm[good])
    corr = np.clip(corr, -1, 1)
    return beta, corr, (1.0 - corr * corr).astype(np.float32)


def sector_rank(x: np.ndarray, sector_code: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rank (0-1) of x[t, k] among the REFERENCE companies of k's sector on day t.

    reference [T, K] bool marks the companies that define the distribution (training universe members, never
    held-out stocks), so a held-out company's return cannot move anybody else's feature. rank = share of
    reference values strictly below x (0.5 when the sector has fewer than 3 reference values).
    """
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for s in np.unique(sector_code):
        cols = np.flatnonzero(sector_code == s)
        xs, ref = x[:, cols], reference[:, cols]
        for t in range(x.shape[0]):
            r = xs[t][ref[t] & np.isfinite(xs[t])]
            fin = np.isfinite(xs[t])
            if not fin.any():
                continue
            if r.size < 3:
                out[t, cols[fin]] = 0.5
                continue
            r.sort()
            out[t, cols[fin]] = np.searchsorted(r, xs[t][fin], side="left") / r.size
    return out


def held_channels(*, ell, rv2, sigma, z, turnover, volume, trades, deliv_pct, stale, series_surv, circuit_carry, days_listed,
                  has_fo, fo: dict[str, np.ndarray], mkt_ell, sector_code, reference, corp_action, med_turn60, traded
                  ) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, [T, K] float32) for HELD_CHANNELS, in order."""
    ln_turn, ln_vol, ln_tr = _ln(turnover), _ln(volume), _ln(trades)
    yield "z_ln_turnover", robust_z(ln_turn, 60)
    yield "z_ln_volume", robust_z(ln_vol, 60)
    yield "z_ln_trades", robust_z(ln_tr, 60)
    yield "z_ln_ticket", robust_z(ln_turn - ln_tr, 60)
    yield "ln_med_turn60", _ln(med_turn60)
    d = (deliv_pct / 100.0).astype(np.float32)
    yield "deliv_pct", d
    yield "z_deliv_pct", robust_z(d, 60)
    yield "deliv_vs_mean20", (d - roll(d, 20, "mean", 10)).astype(np.float32)
    yield "amihud20", roll(_ln(np.abs(ell) + 1e-4) - ln_turn, 20, "mean", 10)
    yield "stale_share20", roll(np.where(traded, stale.astype(np.float32), np.nan), 20, "mean", 5)
    yield "series_surv", series_surv.astype(np.float32)
    yield "circuit_carry", circuit_carry.astype(np.float32)
    yield "ln_days_listed", np.log1p(days_listed).astype(np.float32)
    # ---- derivatives (all NaN -> 0 after normalisation; has_fo is the group mask)
    yield "has_fo", has_fo.astype(np.float32)
    ln_oi = _ln(fo["fut_oi"])
    dln_oi = (ln_oi - lag(ln_oi)).astype(np.float32)
    yield "ln_fut_oi", ln_oi
    yield "dln_fut_oi", dln_oi
    yield "z_dln_fut_oi", robust_z(dln_oi, 60)
    yield "basis_ann", fo["basis_ann"].astype(np.float32)
    yield "pcr_oi", fo["pcr_oi"].astype(np.float32)
    yield "z_pcr_oi", robust_z(fo["pcr_oi"], 60)
    ln_iv = _ln(fo["atm_iv"])
    yield "ln_atm_iv", ln_iv
    yield "ln_iv_rv20", (ln_iv - 0.5 * np.log(252.0) - 0.5 * _ln(roll(rv2, 20, "mean", 10))).astype(np.float32)
    # ---- market-relative
    beta, corr, resid = ewm_beta(ell, mkt_ell)
    yield "beta", beta
    yield "corr_mkt", corr
    yield "resid_share", resid
    del beta, corr, resid
    yield "sector_rank20", sector_rank(_div(roll(ell, 20, "sum", 10), sigma * np.sqrt(20.0)), sector_code, reference)
    # ---- corporate
    ca = corp_action.astype(bool)
    yield "corp_action", ca.astype(np.float32)
    idx = np.where(ca, np.arange(ca.shape[0])[:, None], -10**6)
    since = np.arange(ca.shape[0])[:, None] - np.maximum.accumulate(idx, axis=0)
    yield "since_corp_action", (np.minimum(since, 250) / 250.0).astype(np.float32)
    # ---- group masks (03 §7: one mask per feature group)
    yield "m_price", (np.isfinite(z) & np.isfinite(sigma)).astype(np.float32)
    yield "m_deliv", np.isfinite(deliv_pct).astype(np.float32)
    yield "m_trades", np.isfinite(trades).astype(np.float32)
