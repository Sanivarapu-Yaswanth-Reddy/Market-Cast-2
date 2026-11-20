"""Metric library (P2.4). Definitions follow dev_plan/05_EVALUATION_HARNESS.md section 4.

Conventions
-----------
* Ensembles put the member axis LAST for scalar targets: samples[..., M], obs[...].
* Multivariate ensembles: samples[..., M, D], obs[..., D].
* Every score is "lower is better" unless its name says skill / hit rate.
* NumPy only, so the same code scores baselines on the data PC and model output on the GPU machine
  (convert tensors with .cpu().numpy()). Training losses live in marketcast.train.losses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- ensemble scores (scalar)


def crps_fair(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair (unbiased) ensemble CRPS.

    (1/M) sum_m |x_m - y|  -  1/(2 M (M-1)) sum_{m != m'} |x_m - x_m'|

    The pairwise term is computed in O(M log M) from order statistics:
    sum_{m != m'} |x_m - x_m'| = 2 sum_i (2i - M - 1) x_(i), i = 1..M.
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(obs, dtype=float)
    m = x.shape[-1]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    term1 = np.abs(x - y[..., None]).mean(axis=-1)
    xs = np.sort(x, axis=-1)
    w = 2.0 * np.arange(1, m + 1) - m - 1
    pair_sum = 2.0 * (xs * w).sum(axis=-1)
    return term1 - pair_sum / (2.0 * m * (m - 1))


def crps_ensemble_nrg(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Standard (biased, 'energy form') ensemble CRPS: same as properscoring.crps_ensemble."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(obs, dtype=float)
    m = x.shape[-1]
    term1 = np.abs(x - y[..., None]).mean(axis=-1)
    xs = np.sort(x, axis=-1)
    w = 2.0 * np.arange(1, m + 1) - m - 1
    return term1 - (2.0 * (xs * w).sum(axis=-1)) / (2.0 * m * m)


def crps_normal(mu, sigma, obs) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) (Gneiting et al. 2005)."""
    z = (np.asarray(obs, float) - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


# --------------------------------------------------------------------------- multivariate scores


def energy_score_fair(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair energy score. samples[..., M, D], obs[..., D].

    (1/M) sum_m ||x_m - y|| - 1/(2 M (M-1)) sum_{m != m'} ||x_m - x_m'||
    Memory O(M^2) per item; fine for M <= ~1,000.
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(obs, dtype=float)
    m = x.shape[-2]
    if m < 2:
        raise ValueError("fair energy score needs at least 2 members")
    term1 = np.linalg.norm(x - y[..., None, :], axis=-1).mean(axis=-1)
    diff = x[..., :, None, :] - x[..., None, :, :]
    pair = np.linalg.norm(diff, axis=-1).sum(axis=(-1, -2))  # diagonal is zero
    return term1 - pair / (2.0 * m * (m - 1))


def variogram_score(samples: np.ndarray, obs: np.ndarray, p: float = 0.5, weights: np.ndarray | None = None) -> np.ndarray:
    """Variogram score of order p (Scheuerer and Hamill 2015).

    sum_{i<j} w_ij (|y_i - y_j|^p - (1/M) sum_m |x_i^m - x_j^m|^p)^2 ; samples[..., M, D], obs[..., D].
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(obs, dtype=float)
    d = y.shape[-1]
    iu, ju = np.triu_indices(d, k=1)
    vy = np.abs(y[..., iu] - y[..., ju]) ** p
    vx = (np.abs(x[..., iu] - x[..., ju]) ** p).mean(axis=-2)
    w = 1.0 if weights is None else np.asarray(weights, float)[iu, ju]
    return (w * (vy - vx) ** 2).sum(axis=-1)


def correlation_error(c_forecast: np.ndarray, c_realized: np.ndarray) -> float:
    """||C_f - C_r||_F / N."""
    cf, cr = np.asarray(c_forecast, float), np.asarray(c_realized, float)
    return float(np.linalg.norm(cf - cr, "fro") / cf.shape[0])


# --------------------------------------------------------------------------- volatility and calibration


def qlike(rv2: np.ndarray, sigma2_f: np.ndarray) -> np.ndarray:
    """QLIKE loss: RV^2/sigma_f^2 - ln(RV^2/sigma_f^2) - 1 (0 when the forecast equals the realisation)."""
    r = np.asarray(rv2, float) / np.asarray(sigma2_f, float)
    return r - np.log(r) - 1.0


def interval_coverage(samples: np.ndarray, obs: np.ndarray, levels=(0.5, 0.8, 0.9, 0.95)) -> dict[float, float]:
    """Share of outcomes inside the central ensemble interval at each level (pooled over all items)."""
    x = np.asarray(samples, float)
    y = np.asarray(obs, float)
    out = {}
    for lv in levels:
        lo = np.quantile(x, (1 - lv) / 2, axis=-1)
        hi = np.quantile(x, 1 - (1 - lv) / 2, axis=-1)
        out[lv] = float(np.mean((y >= lo) & (y <= hi)))
    return out


def pit_values(samples: np.ndarray, obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomised PIT: u = (1/M) sum 1[x < y] + U (1/M) sum 1[x = y]."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(samples, float)
    y = np.asarray(obs, float)[..., None]
    below = (x < y).mean(axis=-1)
    tie = (x == y).mean(axis=-1)
    return below + rng.uniform(size=below.shape) * tie


def pit_max_bin_deviation(u: np.ndarray, bins: int = 10) -> float:
    """Largest |share in a bin - 1/bins| of a PIT histogram (0 = perfectly calibrated)."""
    h, _ = np.histogram(np.asarray(u).ravel(), bins=bins, range=(0, 1))
    return float(np.max(np.abs(h / h.sum() - 1.0 / bins)))


# --------------------------------------------------------------------------- VaR backtests


@dataclass
class TestResult:
    stat: float
    pvalue: float
    detail: dict


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(violations: np.ndarray, p: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures test. violations: 0/1 hits; p: expected rate (0.05 for VaR95)."""
    v = np.asarray(violations).astype(int).ravel()
    t, x = len(v), int(v.sum())
    ll0 = _xlogy(t - x, 1 - p) + _xlogy(x, p)
    phat = x / t
    ll1 = _xlogy(t - x, 1 - phat) + _xlogy(x, phat)
    lr = -2.0 * (ll0 - ll1)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)), {"T": t, "x": x, "rate": phat})


def christoffersen(violations: np.ndarray, p: float) -> tuple[TestResult, TestResult]:
    """Christoffersen (1998) independence and conditional-coverage tests -> (LR_ind, LR_cc)."""
    v = np.asarray(violations).astype(int).ravel()
    a, b = v[:-1], v[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_ind1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr_ind = -2.0 * (ll_ind0 - ll_ind1)
    uc = kupiec_pof(v, p)
    lr_cc = uc.stat + lr_ind
    counts = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    return (TestResult(lr_ind, float(stats.chi2.sf(lr_ind, 1)), counts),
            TestResult(lr_cc, float(stats.chi2.sf(lr_cc, 2)), counts))


# --------------------------------------------------------------------------- probabilistic classification


def brier_multiclass(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """sum_c (p_c - o_c)^2 per item; probs[..., C], outcome[...] integer class."""
    p = np.asarray(probs, float)
    o = np.eye(p.shape[-1])[np.asarray(outcome, int)]
    return ((p - o) ** 2).sum(axis=-1)


def brier_skill(probs: np.ndarray, outcome: np.ndarray, base_rate: np.ndarray) -> float:
    """1 - BS / BS(base rate); base_rate is the training-fold class frequency vector."""
    bs = brier_multiclass(probs, outcome).mean()
    base = np.broadcast_to(np.asarray(base_rate, float), np.asarray(probs).shape)
    return float(1.0 - bs / brier_multiclass(base, outcome).mean())


def log_score(probs: np.ndarray, outcome: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """-ln max(p_observed, floor)."""
    p = np.asarray(probs, float)
    po = np.take_along_axis(p, np.asarray(outcome, int)[..., None], axis=-1)[..., 0]
    return -np.log(np.maximum(po, floor))


def reliability_curve(p: np.ndarray, o: np.ndarray, bins: int = 10, n_boot: int = 0, block: int = 20,
                      rng: np.random.Generator | None = None):
    """Binary reliability: mean forecast and observed frequency per probability bin.

    With n_boot > 0, returns 5-95% observed-frequency bands from a circular block bootstrap over time order.
    """
    p = np.asarray(p, float).ravel()
    o = np.asarray(o, float).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)

    def freq(pp, oo, ii):
        mf = np.array([pp[ii == k].mean() if (ii == k).any() else np.nan for k in range(bins)])
        of = np.array([oo[ii == k].mean() if (ii == k).any() else np.nan for k in range(bins)])
        n = np.bincount(ii, minlength=bins)
        return mf, of, n

    mf, of, n = freq(p, o, idx)
    if not n_boot:
        return mf, of, n
    rng = rng or np.random.default_rng(0)
    boots = np.array([freq(p[s], o[s], idx[s])[1] for s in _cbb_indices(len(p), block, n_boot, rng)])
    lo, hi = np.nanpercentile(boots, [5, 95], axis=0)
    return mf, of, n, lo, hi


# --------------------------------------------------------------------------- ranking (sectors)


def spearman_rho(pred: np.ndarray, realized: np.ndarray) -> float:
    return float(stats.spearmanr(pred, realized).statistic)


def top_bottom_hit_rate(pred: np.ndarray, realized: np.ndarray, k: int = 3) -> float:
    """Share of the predicted top-k and bottom-k items that are in the realised top-k / bottom-k."""
    pred, realized = np.asarray(pred, float), np.asarray(realized, float)
    pt, pb = set(np.argsort(-pred)[:k]), set(np.argsort(pred)[:k])
    rt, rb = set(np.argsort(-realized)[:k]), set(np.argsort(realized)[:k])
    return (len(pt & rt) + len(pb & rb)) / (2 * k)


# --------------------------------------------------------------------------- forecast comparison


def _cbb_indices(n: int, block: int, n_boot: int, rng: np.random.Generator):
    nb = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        yield (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n


def newey_west_var(d: np.ndarray, lags: int) -> float:
    """HAC (Bartlett) variance of the mean of d."""
    d = np.asarray(d, float)
    n = len(d)
    e = d - d.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return s / n


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1, method: str = "nw",
                    lags: int | None = None, n_boot: int = 2000, block: int = 20,
                    rng: np.random.Generator | None = None) -> TestResult:
    """DM test on per-date losses (already averaged across stocks). Negative stat = model A better.

    method "nw": Newey-West HAC with lag h + 5 (05 section 4), normal reference.
    method "bootstrap": circular block bootstrap (block 20) of the mean loss difference, two-sided.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    mean = d.mean()
    if method == "nw":
        lags = h + 5 if lags is None else lags
        se = math.sqrt(newey_west_var(d, lags))
        z = mean / se
        return TestResult(z, float(2 * stats.norm.sf(abs(z))), {"mean_diff": mean, "lags": lags, "n": len(d)})
    rng = rng or np.random.default_rng(0)
    centred = d - mean
    bm = np.array([centred[s].mean() for s in _cbb_indices(len(d), block, n_boot, rng)])
    pval = float((np.abs(bm) >= abs(mean)).mean())
    return TestResult(mean / bm.std(), pval, {"mean_diff": mean, "block": block, "n": len(d)})


EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """SR0: expected maximum Sharpe ratio of n_trials unskilled strategies (Bailey and Lopez de Prado 2014)."""
    if n_trials < 2:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_sr) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, var_sr_trials: float) -> TestResult:
    """Deflated Sharpe ratio (per-period SR, not annualised).

    returns: the selected strategy's per-period returns; n_trials: number of configurations logged;
    var_sr_trials: variance of the per-period Sharpe ratios across those trials.
    DSR = Phi( (SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2) ), kurt = non-excess kurtosis.
    """
    r = np.asarray(returns, float)
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr * sr)
    return TestResult(float(stats.norm.cdf(z)), float(1 - stats.norm.cdf(z)),
                      {"sr": sr, "sr0": sr0, "skew": skew, "kurt": kurt, "T": t})
