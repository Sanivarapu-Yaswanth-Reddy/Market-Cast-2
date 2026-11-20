"""B5  LightGBM quantile baseline (05 §6): "the same stock features, no graph".

    features   the stock's own normalised channel row on the decision date (features/stock.py, fold normaliser)
    targets    Z_{t,h} for h = 1, 5, 20: eleven quantile models each (pinball loss)
               ln(RV2_{t,h} / (h sigma_hat_t^2)): one L2 model per horizon for the variance forecast
    sampling   predicted quantiles are sorted (fixes crossing), turned into an inverse CDF that is piecewise-linear
               between quantiles with exponential tails beyond the 1% / 99% points, and sampled through a Gaussian
               copula with B0's shrunk correlation, so the joint scores are comparable with the other baselines

This is the strongest "no-graph" competitor and the one Gate G3 must beat on QLIKE. It emits the MARGINAL
forecast form (per-horizon samples), not day-by-day paths. Fitting takes several minutes per fold
(36 boosters on a few hundred thousand training cells).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from marketcast.baselines.common import Baseline, Ctx, chol, shrunk_corr
from marketcast.baselines.ewma import CORR_WINDOW
from marketcast.eval.forecast import HORIZONS, Forecast
from marketcast.targets.returns import forward_sum

QUANTILES = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99)
FIT_EVERY, MAX_CELLS = 5, 400_000
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)


def inverse_cdf(q_levels: np.ndarray, q_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample a distribution known at a few quantiles. q_values [N, Q] (sorted along Q), u [M, N] uniforms -> [M, N].

    Inside [q_1, q_Q]: linear interpolation. Outside: exponential tails whose rate matches the slope of the
    outermost segment, so the density is continuous in shape and no sample is ever clipped at the last quantile.
    """
    q = np.asarray(q_levels, dtype=float)
    out = np.empty(u.shape)
    for j in range(q_values.shape[0]):
        v = q_values[j]
        x = np.interp(u[:, j], q, v)
        lo, hi = u[:, j] < q[0], u[:, j] > q[-1]
        if lo.any():
            scale = max((v[1] - v[0]) / np.log(q[1] / q[0]), 1e-6)       # left tail: P(X < x) = q0 * exp((x - v0) / scale)
            x[lo] = v[0] + scale * np.log(u[lo, j] / q[0])
        if hi.any():
            scale = max((v[-1] - v[-2]) / np.log((1 - q[-2]) / (1 - q[-1])), 1e-6)
            x[hi] = v[-1] - scale * np.log((1 - u[hi, j]) / (1 - q[-1]))
        out[:, j] = x
    return out


class LgbmQuantile(Baseline):
    model_id = "B5"

    def fit(self, ctx: Ctx) -> None:
        super().fit(ctx)
        import lightgbm as lgb

        from marketcast.train.data_loader import Store

        self.store = Store(ctx.fold)                                      # feature memmap + this fold's normaliser
        rows, m = ctx.train_cells(every=FIT_EVERY)
        rng = np.random.default_rng(0)
        xs, ys, yv = [], {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}
        fwd = {h: forward_sum(ctx.ell.astype(np.float64), h) for h in HORIZONS}
        okw = {h: forward_sum(np.where(ctx.tmask, 0.0, np.nan), h) == 0 for h in HORIZONS}     # every day of the window valid
        rv2 = np.asarray(ctx.panel["rv2"], dtype=np.float64)
        frv = {h: forward_sum(rv2, h) for h in HORIZONS}
        keep_p = min(1.0, MAX_CELLS / max(int(m.sum()), 1))
        for i, r in enumerate(rows):
            cols = np.flatnonzero(m[i] & (rng.random(m.shape[1]) < keep_p))
            if not len(cols):
                continue
            xs.append(self.store.snorm.apply(np.asarray(self.store.feats[r, cols], dtype=np.float32)))
            sig = ctx.sigma[r, cols].astype(np.float64)
            for h in HORIZONS:
                good = okw[h][r, cols]
                ys[h].append(np.where(good, fwd[h][r, cols] / (sig * np.sqrt(h)), np.nan))
                with np.errstate(divide="ignore", invalid="ignore"):
                    yv[h].append(np.where(good, np.log(frv[h][r, cols] / (h * sig * sig)), np.nan))
        x = np.concatenate(xs)
        self.q_models, self.v_models, self.v_resid = {}, {}, {}
        for h in HORIZONS:
            y, yvv = np.concatenate(ys[h]), np.concatenate(yv[h])
            ok = np.isfinite(y)
            self.q_models[h] = [lgb.LGBMRegressor(objective="quantile", alpha=a, **LGB).fit(x[ok], y[ok]) for a in QUANTILES]
            okv = np.isfinite(yvv)
            mv = lgb.LGBMRegressor(objective="regression", **LGB).fit(x[okv], yvv[okv])
            self.v_models[h] = mv
            self.v_resid[h] = float(np.var(yvv[okv] - mv.predict(x[okv])))    # log-normal mean correction below
        self.n_train = int(len(x))

    def forecast(self, row: int, n_paths: int, rng: np.random.Generator) -> Forecast:
        c = self.ctx
        cols = c.cols(row)
        x = self.store.snorm.apply(np.asarray(self.store.feats[row, cols], dtype=np.float32))
        lmat = chol(shrunk_corr(c.z[max(0, row - CORR_WINDOW + 1): row + 1][:, cols]))
        sig = c.sigma[row, cols].astype(np.float64)
        zh, var_h = {}, {}
        for h in HORIZONS:
            qv = np.sort(np.column_stack([m.predict(x) for m in self.q_models[h]]), axis=1)
            u = np.clip(stats.norm.cdf(rng.standard_normal((n_paths, len(cols))) @ lmat.T), 1e-6, 1 - 1e-6)
            zh[h] = inverse_cdf(np.array(QUANTILES), qv, u)
            var_h[h] = h * sig * sig * np.exp(self.v_models[h].predict(x) + 0.5 * self.v_resid[h])
        return Forecast(self.model_id, c.fold, c.dates[row], c.panel.keys[cols], zh=zh, var_h=var_h)
