"""B2  GJR-GARCH(1,1,1) with skew-t innovations per stock, Gaussian copula across stocks (05 §6).

    estimation   `arch` on each stock's last 1,000 valid daily log returns (minimum 500), refit once per calendar
                 month (first forecast date of the month), parameters cached; data up to the forecast date only
    filtering    the GJR recursion is re-run with the cached parameters up to the forecast date, giving the
                 conditional variance for t+1
    simulation   correlated normals (B0's shrunk EWMA correlation) -> uniforms -> skew-t innovations, then
                     sigma2_{k+1} = omega + (alpha + gamma * 1[e_k < 0]) e_k^2 + beta sigma2_k
    fallback     a stock with too little history or a failed / non-stationary fit uses sigma_hat persistence with
                 Gaussian innovations (counted and reported by the runner)

`Garch(gjr=False)` is plain GARCH(1,1) (model id B2g11). Returns are scaled by 100 for the optimiser.
This baseline is CPU-heavy: about 500 fits per month of test data. Budget roughly 10-20 minutes per fold.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from marketcast.baselines.common import H, Baseline, Ctx, chol, shrunk_corr
from marketcast.baselines.ewma import CORR_WINDOW
from marketcast.eval.forecast import Forecast

WINDOW, MIN_OBS, SCALE = 1000, 500, 100.0


def gjr_filter(x: np.ndarray, mu: float, omega: float, alpha: float, gamma: float, beta: float) -> float:
    """Conditional variance for the day AFTER the last observation of x (returns in % units)."""
    e = x - mu
    s2 = float(np.var(e))
    for ei in e:
        s2 = omega + (alpha + gamma * (ei < 0.0)) * ei * ei + beta * s2
    return s2


class Garch(Baseline):
    model_id = "B2"

    def __init__(self, gjr: bool = True):
        self.gjr = gjr
        if not gjr:
            self.model_id = "B2g11"
        self.cache: dict[tuple[int, int, int], np.ndarray | None] = {}
        self.n_fit, self.n_bad = 0, 0

    def fit(self, ctx: Ctx) -> None:
        super().fit(ctx)

    def _params(self, k: int, row: int) -> np.ndarray | None:
        """[mu, omega, alpha, gamma, beta, eta, lam] for company column k, refit monthly. None = use the fallback."""
        d = self.ctx.dates[row]
        key = (k, d.year, d.month)
        if key in self.cache:
            return self.cache[key]
        from arch import arch_model

        x = self.ctx.ell[: row + 1, k]
        x = x[np.isfinite(x)][-WINDOW:] * SCALE
        par = None
        if len(x) >= MIN_OBS and np.std(x) > 0:
            self.n_fit += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1 if self.gjr else 0, q=1, dist="skewt", rescale=False
                                     ).fit(disp="off", show_warning=False, options={"maxiter": 200})
                p = res.params.to_numpy(dtype=float)
                mu, omega, alpha = p[0], p[1], p[2]
                gamma, beta, eta, lam = (p[3], p[4], p[5], p[6]) if self.gjr else (0.0, p[3], p[4], p[5])
                if np.all(np.isfinite(p)) and omega > 0 and alpha + beta + 0.5 * gamma < 0.999 and eta > 2.05:
                    par = np.array([mu, omega, alpha, gamma, beta, eta, lam])
            except Exception:
                par = None
            if par is None:
                self.n_bad += 1
        self.cache[key] = par
        return par

    def forecast(self, row: int, n_paths: int, rng: np.random.Generator) -> Forecast:
        from arch.univariate import SkewStudent

        c = self.ctx
        cols = c.cols(row)
        n = len(cols)
        lmat = chol(shrunk_corr(c.z[max(0, row - CORR_WINDOW + 1): row + 1][:, cols]))
        g = rng.standard_normal((n_paths, H, n)) @ lmat.T
        u = np.clip(stats.norm.cdf(g), 1e-6, 1 - 1e-6)
        ell = np.empty((n_paths, H, n))
        lnsig = np.empty((n_paths, H, n))
        dist = SkewStudent()
        sig_hat = c.sigma[row, cols].astype(np.float64)
        for j, k in enumerate(cols):
            par = self._params(int(k), row)
            if par is None:                                           # fallback: EWMA persistence, Gaussian
                ell[:, :, j] = g[:, :, j] * sig_hat[j]
                lnsig[:, :, j] = np.log(sig_hat[j])
                continue
            mu, omega, alpha, gamma, beta, eta, lam = par
            x = c.ell[: row + 1, k]
            x = x[np.isfinite(x)][-WINDOW:] * SCALE
            s2 = np.full(n_paths, gjr_filter(x, mu, omega, alpha, gamma, beta))
            e_std = dist.ppf(u[:, :, j], [eta, lam])                  # unit-variance skew-t innovations
            for step in range(H):
                e = np.sqrt(s2) * e_std[:, step]
                ell[:, step, j] = (mu + e) / SCALE
                lnsig[:, step, j] = 0.5 * np.log(s2) - np.log(SCALE)
                s2 = omega + (alpha + gamma * (e < 0.0)) * e * e + beta * s2
        v = lnsig + self.mu_v + self.sd_v * rng.standard_normal(lnsig.shape)
        return self._wrap(row, cols, ell, v)

    def stability(self) -> dict:
        """Acceptance (06 P2.5): alpha + beta < 1 for >= 98% of fits."""
        return {"fits": self.n_fit, "rejected": self.n_bad, "accepted_share": 1.0 - self.n_bad / max(self.n_fit, 1)}
