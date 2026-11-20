"""B1  Filtered historical simulation (05 §6).

For each scenario, draw a sequence of H past sessions from the last 500 (stationary bootstrap, mean block 5,
so volatility clusters and multi-day moves survive), take each drawn day's whole cross-section of scaled
returns z (the joint structure of that real day is preserved), and re-scale by today's sigma_hat:

    l_{t+k}^m = z_{s(m,k)} * sigma_hat_t          v_{t+k}^m = (v_s - ln sigma_hat_{s-1}) + ln sigma_hat_t

A stock with no z on a drawn day (not listed yet, suspended) gets that day's mean z of its sector among the
stocks that do have one, as the plan specifies; if the sector is empty that day, 0.
"""
from __future__ import annotations

import numpy as np

from marketcast.baselines.common import H, Baseline, Ctx
from marketcast.eval.forecast import Forecast

POOL, MEAN_BLOCK = 500, 5


class HistSim(Baseline):
    model_id = "B1"

    def fit(self, ctx: Ctx) -> None:
        super().fit(ctx)
        from marketcast.graph.fabric import load_tree

        tree = load_tree()
        self.sector = np.asarray(tree["industry_sector"])[np.asarray(tree["stock_industry"])]
        self.n_sec = len(tree["sectors"])

    def forecast(self, row: int, n_paths: int, rng: np.random.Generator) -> Forecast:
        c = self.ctx
        cols = c.cols(row)
        n = len(cols)
        lo = max(1, row - POOL + 1)
        pool = np.arange(lo, row + 1)
        # stationary bootstrap of source rows: continue the block with prob 1 - 1/MEAN_BLOCK, else jump
        src = np.empty((n_paths, H), dtype=np.int64)
        src[:, 0] = rng.integers(0, len(pool), n_paths)
        for k in range(1, H):
            jump = rng.random(n_paths) < 1.0 / MEAN_BLOCK
            nxt = np.where(src[:, k - 1] + 1 < len(pool), src[:, k - 1] + 1, 0)
            src[:, k] = np.where(jump, rng.integers(0, len(pool), n_paths), nxt)
        zp = c.z[pool][:, cols].astype(np.float64)                         # [P, N]
        sec = self.sector[cols]
        onehot = np.zeros((n, self.n_sec))
        onehot[np.arange(n), sec] = 1.0
        ok = np.isfinite(zp)
        sec_mean = (np.where(ok, zp, 0.0) @ onehot) / np.maximum(ok.astype(float) @ onehot, 1.0)   # [P, n_sec]
        zp = np.where(ok, zp, sec_mean[:, sec])
        sig = c.sigma[row, cols].astype(np.float64)
        dv = c.v[pool][:, cols] - np.log(c.sigma[pool - 1][:, cols])       # realised log-vol relative to yesterday's sigma
        dv = np.where(np.isfinite(dv), dv, self.mu_v)
        ell = zp[src] * sig[None, None, :]
        v = dv[src] + np.log(sig)[None, None, :]
        return self._wrap(row, cols, ell, v)
