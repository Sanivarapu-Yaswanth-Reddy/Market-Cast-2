"""B0  EWMA random walk (05 §6): the reference every claim is measured against.

    z_{t+k} ~ multivariate Student-t(nu) with EWMA (lambda = 0.97) Ledoit-Wolf-shrunk correlation of past z,
              independent across days, scale = pooled training std of z
    l_{t+k} = z_{t+k} * sigma_hat_t                       (volatility persistence over the horizon)
    v_{t+k} = ln sigma_hat_t + training offset + noise     (see Ctx.vol_offset)

nu is chosen per fold by maximum likelihood on pooled training z over a small grid; `B0g` is the Gaussian
variant (nu = infinity). The multivariate t uses ONE chi-square draw per (scenario, day) for all stocks, which
gives the joint tail dependence a Gaussian copula lacks.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from marketcast.baselines.common import H, Baseline, Ctx, chol, shrunk_corr
from marketcast.eval.forecast import Forecast

NU_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0)
CORR_WINDOW = 250


class EwmaRW(Baseline):
    model_id = "B0"

    def __init__(self, gaussian: bool = False):
        self.gaussian = gaussian
        if gaussian:
            self.model_id = "B0g"

    def fit(self, ctx: Ctx) -> None:
        super().fit(ctx)
        rows, m = ctx.train_cells(every=3)
        z = ctx.z[rows + 1][m]
        z = z[np.isfinite(z)]
        self.z_std = float(z.std())
        self.nu = np.inf
        if not self.gaussian:
            best = -np.inf
            for nu in NU_GRID:                                   # unit-variance t scaled to the sample std
                ll = stats.t.logpdf(z, nu, scale=self.z_std * np.sqrt((nu - 2.0) / nu)).sum()
                if ll > best:
                    best, self.nu = ll, nu

    def draw_z(self, row: int, cols: np.ndarray, n_paths: int, rng: np.random.Generator, horizon: int = H) -> np.ndarray:
        """Correlated z draws [M, H, N] for the stocks `cols` using only data up to `row`."""
        c = self.ctx
        lmat = chol(shrunk_corr(c.z[max(0, row - CORR_WINDOW + 1): row + 1][:, cols]))
        g = rng.standard_normal((n_paths, horizon, len(cols))) @ lmat.T
        if np.isfinite(self.nu):
            chi = rng.chisquare(self.nu, size=(n_paths, horizon, 1))
            g = g * np.sqrt((self.nu - 2.0) / chi)               # unit-variance multivariate t
        return g * self.z_std

    def forecast(self, row: int, n_paths: int, rng: np.random.Generator) -> Forecast:
        c = self.ctx
        cols = c.cols(row)
        sig = c.sigma[row, cols].astype(np.float64)
        z = self.draw_z(row, cols, n_paths, rng)
        return self._wrap(row, cols, z * sig[None, None, :], self._persist_v(np.log(sig), z.shape, rng))
