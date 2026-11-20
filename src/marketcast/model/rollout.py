"""L8  Autoregressive rollout (04 §L8): M coherent sample paths of the next H sessions.

    for k = 1..H (all scenarios at once on the GPU):
        (z_hat, v_hat) = model(state, forcing of day t+k, eps^{m,k})
        l_hat = z_hat * sigma_hat(state)                        # back to a log return
        clip l_hat to the +-20% fixed-circuit band for stocks without F&O; remember the hit (circuit_carry)
        update the stock's PRICE channels from (l_hat, v_hat) and append the new day to its 60-day window
        hold every non-price channel and all global nodes at their last value (decisions D-13, "v1")

STATE carried per scenario and stock: the last 252 log returns, 60 scaled returns, 60 realized variances,
20 log vols, the EWMA variance s2 and sigma_hat. These are exactly what features/stock.py needs to produce the
PRICE_CHANNELS row of a new day; `price_channels_step` below is its torch mirror (same formulas, same
windows, same minimum counts) and tests/test_rollout_consistency.py checks the two against each other.

KNOWN APPROXIMATIONS inside the horizon (all v1, all documented in 08 as D-24)
    * gap channel <- 0, range <- sigma-implied Parkinson value, close location <- 0.5
    * market-relative channels (beta, correlation, sector rank), liquidity, delivery and F&O channels are held
    * the sigma floor is frozen at its decision-date value; the true circuit band (2/5/10/20%) is unknown, so
      the widest (20%) is used
Gradients flow through the samples (they are deterministic functions of eps), which is what unrolled
fine-tuning (training stage S3) needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from marketcast.features.stock import CHANNELS, PARKINSON_K, PRICE_CHANNELS
from marketcast.targets.returns import EWMA_LAMBDA, EWMA_UPDATE_CAP

N_PRICE = len(PRICE_CHANNELS)
CIRCUIT_BAND = 0.20
_CARRY = CHANNELS.index("circuit_carry")
_MPRICE = CHANNELS.index("m_price")


@dataclass
class RolloutState:
    """Raw (un-normalised) per-stock history needed to recompute price channels. Shapes [S, N, ...]."""
    ell: torch.Tensor      # [S, N, 252] daily log returns, oldest first (NaN = no observation)
    z: torch.Tensor        # [S, N, 60]
    rv2: torch.Tensor      # [S, N, 60]
    v: torch.Tensor        # [S, N, 20]
    s2: torch.Tensor       # [S, N] EWMA variance
    sigma: torch.Tensor    # [S, N] sigma_hat (floored)
    floor: torch.Tensor    # scalar: sigma floor on the decision date
    has_fo: torch.Tensor   # [N] bool: no fixed circuit band

    @classmethod
    def from_batch(cls, b, n_samples: int) -> "RolloutState":
        t = b.targets
        ex = lambda x: x.unsqueeze(0).expand(n_samples, *x.shape).clone()       # noqa: E731
        return cls(ex(t["ell_hist"]), ex(t["z_hist"]), ex(t["rv2_hist"]), ex(t["v_hist"]), ex(t["s2"]), ex(t["sigma"]),
                   t["sigma_floor"], t["has_fo"])


def _nanmean(x: torch.Tensor, min_count: int) -> torch.Tensor:
    ok = torch.isfinite(x)
    n = ok.sum(-1)
    m = torch.where(ok, x, torch.zeros_like(x)).sum(-1) / n.clamp_min(1)
    return torch.where(n >= min_count, m, torch.full_like(m, float("nan")))


def _nansum(x: torch.Tensor, min_count: int) -> torch.Tensor:
    ok = torch.isfinite(x)
    s = torch.where(ok, x, torch.zeros_like(x)).sum(-1)
    return torch.where(ok.sum(-1) >= min_count, s, torch.full_like(s, float("nan")))


def price_channels_step(st: RolloutState, sigma_prev: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    """PRICE_CHANNELS row [S, N, 22] (raw units) for the day that was just appended to `st`.

    Mirrors features/stock.py::price_channels. `st` already contains the new day as the LAST element of each
    buffer and the updated sigma; sigma_prev is sigma_hat before the update (the scale of today's z and range).
    """
    sig = st.sigma
    out = [st.z[..., -1]]
    for n in (5, 20, 60, 120):
        out.append(_nansum(st.ell[..., -n:], n // 2) / (sig * math.sqrt(n)))
    p = torch.nan_to_num(st.ell, nan=0.0).cumsum(-1)                     # log price index inside the 252-day window
    s252 = sig * math.sqrt(252.0)
    out.append((p[..., -1] - p.max(-1).values) / s252)
    out.append((p[..., -1] - p.min(-1).values) / s252)
    for n in (20, 60):
        out.append((p[..., -1] - p[..., -n:].max(-1).values) / (sig * math.sqrt(n)))
    z = st.z
    m1, m2, m3, m4 = (_nanmean(z ** k, 30) for k in (1, 2, 3, 4))
    var = (m2 - m1 * m1).clamp_min(1e-8)
    out.append((m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5)
    out.append((m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) / (var * var) - 3.0)
    neg = torch.where(z < 0, z * z, torch.where(torch.isfinite(z), torch.zeros_like(z), z))
    out.append(_nanmean(neg, 30) / m2.clamp_min(1e-8))
    out.append(v_new)
    lnrv = {n: 0.5 * torch.log(_nanmean(st.rv2[..., -n:], max(2, n // 2))) for n in (5, 20, 60)}
    out += [lnrv[5], lnrv[20], lnrv[60], torch.log(sig), lnrv[5] - lnrv[60]]
    vm, vm2 = _nanmean(st.v, 10), _nanmean(st.v * st.v, 10)
    out.append((vm2 - vm * vm).clamp_min(0.0).sqrt())
    out.append(torch.zeros_like(sig))                                    # gap: no overnight/intraday split is simulated
    out.append(PARKINSON_K * torch.exp(v_new) / sigma_prev)              # range implied by the simulated volatility
    out.append(torch.full_like(sig, 0.5))                                # close location
    return torch.stack(out, dim=-1)


def advance(st: RolloutState, z_hat: torch.Tensor, v_hat: torch.Tensor) -> tuple[RolloutState, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one simulated day. Returns (new state, l_hat [S, N], sigma_prev, circuit flag in {-1, 0, +1})."""
    sigma_prev = st.sigma
    ell = z_hat.float() * sigma_prev
    lo, hi = math.log(1.0 - CIRCUIT_BAND), math.log(1.0 + CIRCUIT_BAND)
    fixed = ~st.has_fo.unsqueeze(0)
    clipped = torch.where(fixed, ell.clamp(lo, hi), ell)
    carry = torch.where(fixed & (ell >= hi), 1.0, 0.0) - torch.where(fixed & (ell <= lo), 1.0, 0.0)
    ell = clipped
    upd = ell.clamp(-EWMA_UPDATE_CAP, EWMA_UPDATE_CAP)
    s2 = EWMA_LAMBDA * st.s2 + (1.0 - EWMA_LAMBDA) * upd * upd
    sigma = torch.maximum(s2.sqrt(), st.floor.to(s2.dtype))
    push = lambda buf, x: torch.cat([buf[..., 1:], x.unsqueeze(-1)], dim=-1)   # noqa: E731
    v_hat = v_hat.float()
    new = replace(st, ell=push(st.ell, ell), z=push(st.z, ell / sigma_prev), rv2=push(st.rv2, torch.exp(2.0 * v_hat)),
                  v=push(st.v, v_hat), s2=s2, sigma=sigma)
    return new, ell, sigma_prev, carry


def rollout(model, b, eps: torch.Tensor, norm_median: torch.Tensor, norm_scale: torch.Tensor, clip: float = 8.0) -> dict:
    """Roll the generative model forward. eps [H, S, noise_dim]; norm_* are the stock normaliser vectors [F_s].

    Returns {"ell": [S, H, N] log returns, "v": [S, H, N] log vols, "mkt_z": [S, H], "z": [S, H, N] model-space
    scaled returns, "extreme_share": share of scenario-stocks whose |cumulative return| ever exceeds 60%}.
    Step 1 reuses the batch exactly as the single-step model sees it, so a 1-step rollout equals forward().
    """
    if model.cfg.mode != "gen":
        raise ValueError("rollout needs the generative model (mode='gen')")
    h_steps, s = eps.shape[0], eps.shape[1]
    if h_steps > b.f_fut.shape[0]:
        raise ValueError(f"rollout of {h_steps} steps but only {b.f_fut.shape[0]} future forcing rows in the batch")
    st = RolloutState.from_batch(b, s)
    x = b.x_s.unsqueeze(0).expand(s, -1, -1, -1)                         # [S, N, T, F] normalised window
    mask, f_hist = b.mask_s, b.f_hist
    med_p, sc_p = norm_median[:N_PRICE].to(x.dtype), norm_scale[:N_PRICE].to(x.dtype)
    ells, vs, zs, mk = [], [], [], []
    cur = b
    for k in range(1, h_steps + 1):
        if k == 1:
            h0 = model.encode(b)
            out = model.simulate(h0, b, eps[0], step=1)
        else:
            out = model.simulate(model.encode(cur), cur, eps[k - 1], step=k, ln_sigma=torch.log(st.sigma))
        st, ell, sigma_prev, carry = advance(st, out["z"], out["v"])
        ells.append(ell)
        vs.append(out["v"].float())
        zs.append(out["z"].float())
        mk.append(out["mkt_z"].float())
        if k == h_steps:
            break
        raw = price_channels_step(st, sigma_prev, out["v"].float())
        new_row = x[:, :, -1, :].clone()                                 # held channels keep their last normalised value
        new_row[..., :N_PRICE] = torch.nan_to_num(((raw.to(x.dtype) - med_p) / sc_p).clamp(-clip, clip), nan=0.0)
        new_row[..., _CARRY] = carry.to(x.dtype)                         # pressure from a simulated circuit hit carries over
        new_row[..., _MPRICE] = 1.0
        x = torch.cat([x[:, :, 1:], new_row.unsqueeze(2)], dim=2)
        mask = torch.cat([mask[:, 1:], torch.ones_like(mask[:, :1])], dim=1)
        f_hist = torch.cat([f_hist[1:], b.f_fut[k - 1: k]], dim=0)       # the simulated day's forcing joins the history
        cur = replace(b, x_s=x, mask_s=mask, f_hist=f_hist)
    ell_p, v_p = torch.stack(ells, dim=1), torch.stack(vs, dim=1)
    extreme = (ell_p.cumsum(1).abs() > math.log(1.6)).any(1).float().mean()
    return {"ell": ell_p, "v": v_p, "z": torch.stack(zs, dim=1), "mkt_z": torch.stack(mk, dim=1), "extreme_share": extreme}
