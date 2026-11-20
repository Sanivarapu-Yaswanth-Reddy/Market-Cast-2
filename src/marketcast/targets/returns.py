"""Return, realized-variance and trailing-volatility formulas (03 §1-§3). ONE implementation, used everywhere.

Every function works on arrays shaped [T] or [T, K] (time first) and treats NaN as "no observation".
Nothing here looks forward: the value at row t depends on rows <= t only.

Price basis: `pc` is the previous close expressed on day t's share basis (own previous close x inferred
split/bonus factor, built in data/build_panel.py). NSE's own PREVCLOSE is never used (decision D-22).
"""
from __future__ import annotations

import numpy as np

EWMA_LAMBDA = 0.94          # RiskMetrics decay (03 §3)
EWMA_INIT_N = 20            # s2 starts as the sample variance of the first 20 valid log returns
EWMA_VALID_N = 60           # sigma is "valid" only after 60 observations
SIGMA_ABS_FLOOR = 0.004     # 0.4% daily
SIGMA_REL_FLOOR = 0.25      # x cross-sectional median sigma
RV2_FLOOR = 1e-6            # (0.1% daily)^2
# |l| used in the EWMA UPDATE is capped at ln(1.4): the same threshold that flags `jump_unexplained`.
# Without the cap one unadjusted demerger row (e.g. -60%) would inflate sigma for ~100 sessions and crush
# every z that follows. Targets themselves are never clipped (03 §3). Logged as decision D-23.
EWMA_UPDATE_CAP = float(np.log(1.4))


def log_return(close: np.ndarray, pc: np.ndarray, dividend: np.ndarray | float = 0.0) -> np.ndarray:
    """l_t = ln((close_t + D_t) / pc_t). D = 0 until dividend history exists (gap G2) -> price return."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log((close + dividend) / pc)
    out[~np.isfinite(out)] = np.nan
    return out


def rv2_gk(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """Daily realized-variance proxy: overnight gap^2 + Garman-Klass intraday (03 §2, decision D-06).

        o    = ln(open/pc)
        GK   = 0.5 ln(high/low)^2 - (2 ln2 - 1) ln(close/open)^2
        RV2  = o^2 + max(GK, 0)            and, when high == low (no range recorded), o^2 + ln(close/open)^2
    NaN when any input is missing or non-positive.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.log(open_ / pc)
        co = np.log(close / open_)
        gk = 0.5 * np.log(high / low) ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
        rv2 = o ** 2 + np.maximum(gk, 0.0)
        rv2 = np.where(high == low, o ** 2 + co ** 2, rv2)
    rv2 = np.where(np.isfinite(rv2), rv2, np.nan)
    return rv2


def log_rv(rv2: np.ndarray) -> np.ndarray:
    """v_t = 0.5 ln(max(RV2_t, 1e-6)): log realized volatility with a 0.1% daily floor."""
    with np.errstate(invalid="ignore"):
        return 0.5 * np.log(np.maximum(rv2, RV2_FLOOR))


def ewma_var(ell: np.ndarray, lam: float = EWMA_LAMBDA, init_n: int = EWMA_INIT_N,
             cap: float | None = EWMA_UPDATE_CAP) -> tuple[np.ndarray, np.ndarray]:
    """EWMA variance s2 down the time axis, per column. Returns (s2 [T, K], n_obs [T, K]).

        s2_t = lam * s2_{t-1} + (1 - lam) * l_t^2
    * initialised on the day the `init_n`-th valid return arrives, as the sample variance of those returns;
    * a NaN return leaves s2 unchanged (a non-trading day carries yesterday's estimate);
    * s2 is NaN before initialisation. n_obs counts valid returns so far (validity needs >= 60, 03 §3).
    """
    x = np.asarray(ell, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    valid = np.isfinite(x)
    xc = np.where(valid, x, 0.0)
    if cap is not None:
        xc = np.clip(xc, -cap, cap)
    n_obs = np.cumsum(valid, axis=0)
    c1, c2 = np.cumsum(xc, axis=0), np.cumsum(xc * xc, axis=0)
    s2 = np.full(x.shape, np.nan)
    cur = np.full(x.shape[1], np.nan)
    for t in range(x.shape[0]):
        v = valid[t]
        start = v & (n_obs[t] == init_n)
        if start.any():
            mean = c1[t, start] / init_n
            cur[start] = np.maximum((c2[t, start] - init_n * mean * mean) / (init_n - 1), 0.0)
        upd = v & (n_obs[t] > init_n)
        if upd.any():
            cur[upd] = lam * cur[upd] + (1.0 - lam) * xc[t, upd] ** 2
        s2[t] = cur
    if squeeze:
        return s2[:, 0], n_obs[:, 0]
    return s2, n_obs


def sigma_hat(s2: np.ndarray, n_obs: np.ndarray, floor_members: np.ndarray | None = None,
              valid_n: int = EWMA_VALID_N) -> tuple[np.ndarray, np.ndarray]:
    """sigma_hat_t = max(sqrt(s2_t), floor_t), floor_t = max(0.004, 0.25 * median_j sqrt(s2_{j,t})) (03 §3).

    floor_members [T, K] bool restricts the cross-section used for the median (training companies only, so
    held-out stocks never influence a normaliser; 00 rule 7). Returns (sigma [T, K], floor [T]).
    sigma is NaN until the company has `valid_n` observations.
    """
    sd = np.sqrt(s2)
    ref = np.where(floor_members, sd, np.nan) if floor_members is not None else sd
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(n_obs >= valid_n, ref, np.nan), axis=1)
    floor = np.maximum(SIGMA_ABS_FLOOR, SIGMA_REL_FLOOR * np.nan_to_num(med, nan=0.0))
    sig = np.maximum(sd, floor[:, None])
    sig[~(n_obs >= valid_n) | ~np.isfinite(sd)] = np.nan
    return sig, floor


def scaled_return(ell: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z_t = l_t / sigma_hat_{t-1}  (the target for decision date t-1; 03 §3, decision D-05). Never clipped."""
    prev = np.full_like(sigma, np.nan)
    prev[1:] = sigma[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ell / prev
    z[~np.isfinite(z)] = np.nan
    return z


def forward_sum(x: np.ndarray, h: int) -> np.ndarray:
    """L_{t,h} = sum_{k=1..h} x_{t+k}. NaN if ANY day in the window is missing (no partial windows).

    This is the only forward-looking helper in the package; it is used to build labels and to score
    forecasts, never to build features.
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0), axis=0)
    bad = np.cumsum(~np.isfinite(x), axis=0)
    out = np.full(x.shape, np.nan)
    if h < len(x):
        out[:-h] = np.where(bad[h:] - bad[:-h] == 0, c[h:] - c[:-h], np.nan)
    return out
