"""Shared plumbing for baselines (05 §6): the fold context, the correlation estimator and the base class.

Every baseline
    * is fitted on the fold's TRAINING decision dates and NON-held-out companies only (`fit`),
    * forecasts a decision date using data up to and including that date only (`forecast`),
    * returns the same `Forecast` object as MarketCast, so the same scorer judges it.
The node set of a date is the same rule the model uses at evaluation: MC500 members (held-out included) with
a valid price state today.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketcast.eval.folds import Split, split_rows
from marketcast.eval.forecast import Forecast
from marketcast.targets.build import ModelPanel

H = 20


@dataclass
class Ctx:
    fold: str
    panel: ModelPanel
    split: Split
    z: np.ndarray           # [T, K] scaled returns (row t = day t)
    ell: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    tmask: np.ndarray
    member: np.ndarray
    heldout: np.ndarray     # [K]

    @classmethod
    def open(cls, fold: str, allow_vault: bool = False) -> "Ctx":
        p = ModelPanel()
        f32 = lambda n: np.asarray(p[n], dtype=np.float32)        # noqa: E731
        return cls(fold, p, split_rows(fold, p.dates, allow_vault=allow_vault), f32("z"), f32("ell"), f32("v"), f32("sigma"),
                   np.asarray(p["target_mask"]), np.asarray(p["member500"]), p.heldout)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates

    def cols(self, row: int) -> np.ndarray:
        return np.flatnonzero(self.member[row] & np.isfinite(self.sigma[row]) & np.isfinite(self.z[row]))

    def train_cells(self, every: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(rows [n], mask [n, K]) of training decision dates x non-held-out members with a valid next-day target."""
        rows = self.split.train[::every]
        rows = rows[rows + 1 < len(self.dates)]
        m = self.member[rows] & ~self.heldout[None, :] & self.tmask[rows + 1] & np.isfinite(self.sigma[rows])
        return rows, m

    def vol_offset(self) -> tuple[float, float]:
        """Mean and std of v_{t+1} - ln sigma_hat_t on training cells: maps an EWMA/GARCH volatility level to the
        Garman-Klass log realized vol the scorer uses (they are different estimators of the same thing)."""
        rows, m = self.train_cells(every=3)
        d = (self.v[rows + 1] - np.log(self.sigma[rows]))[m]
        d = d[np.isfinite(d)]
        return float(d.mean()), float(d.std())


def shrunk_corr(zwin: np.ndarray, lam: float = 0.97) -> np.ndarray:
    """EWMA-weighted correlation of the columns of zwin [W, N] (oldest first) with Ledoit-Wolf shrinkage.

    Rows are scaled by sqrt of their EWMA weight so that X'X is the EWMA second-moment matrix; Ledoit-Wolf then
    picks the shrinkage intensity toward a scaled identity. Missing values count as 0 (the mean of z).
    Returns a positive-definite correlation matrix [N, N].
    """
    from sklearn.covariance import ledoit_wolf

    w, n = zwin.shape
    x = np.clip(np.nan_to_num(zwin, nan=0.0), -6.0, 6.0).astype(np.float64)
    wt = lam ** np.arange(w - 1, -1, -1)
    x = x * np.sqrt(wt / wt.sum() * w)[:, None]
    cov, _ = ledoit_wolf(x, assume_centered=True)
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    c = cov / np.outer(d, d)
    return 0.5 * (c + c.T)


def chol(c: np.ndarray) -> np.ndarray:
    """Cholesky factor with a jitter ladder (a shrunk correlation matrix is PD, but be safe)."""
    for eps in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            return np.linalg.cholesky(c + eps * np.eye(len(c)))
        except np.linalg.LinAlgError:
            continue
    return np.eye(len(c))


class Baseline:
    """Interface: fit once per fold, then forecast any decision row of that fold's validation / test split."""
    model_id = "B?"

    def fit(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.mu_v, self.sd_v = ctx.vol_offset()

    def forecast(self, row: int, n_paths: int, rng: np.random.Generator) -> Forecast:
        raise NotImplementedError

    # helpers shared by the path baselines
    def _wrap(self, row: int, cols: np.ndarray, ell: np.ndarray, v: np.ndarray) -> Forecast:
        c = self.ctx
        return Forecast(self.model_id, c.fold, c.dates[row], c.panel.keys[cols], ell=ell.astype(np.float32), v=v.astype(np.float32))

    def _persist_v(self, ln_sigma: np.ndarray, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        """Volatility persistence: v_hat = ln sigma_hat_t + training offset + Gaussian noise (training spread)."""
        return ln_sigma[None, None, :] + self.mu_v + self.sd_v * rng.standard_normal(shape)
