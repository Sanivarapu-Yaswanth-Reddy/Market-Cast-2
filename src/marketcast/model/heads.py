"""L7  Output heads (04 §L7).

GENERATIVE (mode = "gen"; one SAMPLE per scenario, trained with fair CRPS)
    z_hat   [S, N]   next-day volatility-scaled return of each stock
    v_hat   [S, N]   next-day log realized vol = ln(sigma_hat_t) + delta    (residual on today's log vol)
    mkt_z   [S]      next-day scaled return of the official market index (consistency head)
    trend   [2, 3]   logits over {down, sideways, up} for h = 5 and 20 (cross-check head; averaged over scenarios)

DETERMINISTIC (mode = "det"; Phase 3; trained with the QLIKE-consistent loss RV2/exp(s) + s)
    s       [1, N, 3]  predicted LOG VARIANCE of the 1-, 5- and 20-day realized variance:
                       s_h = 2 ln(sigma_hat_t) + ln(h) + delta_h     (h-day heads are the "direct" heads that the
                       Phase 4 rollout is later compared against)

Every final linear layer is zero-initialised, so an untrained model outputs the persistence forecast
(z = 0, v = ln sigma_hat, variance = h * sigma_hat^2): training starts from the EWMA baseline, not from noise.

Sector outputs have NO head: a sector's return in a scenario is the significance-weighted sum of its member
stocks' simulated returns (readout/sectors.py), so sector and stock scenarios can never disagree.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from marketcast.model.fgn import CondLN

DET_HORIZONS = (1, 5, 20)
TREND_HORIZONS = (5, 20)


def _mlp(d: int, out: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Heads(nn.Module):
    def __init__(self, d: int, cond_dim: int = 128, mode: str = "gen"):
        super().__init__()
        if mode not in ("gen", "det"):
            raise ValueError("mode must be 'gen' or 'det'")
        self.mode = mode
        self.norm = CondLN(d, cond_dim)
        if mode == "gen":
            self.ret, self.vol, self.mkt = _mlp(d, 1), _mlp(d, 1), _mlp(d, 1)
        else:
            self.var = _mlp(d, len(DET_HORIZONS))
            self.register_buffer("ln_h", torch.tensor([math.log(h) for h in DET_HORIZONS]), persistent=False)
        self.trend = _mlp(d, len(TREND_HORIZONS) * 3)

    def forward(self, h: torch.Tensor, c, node_type: torch.Tensor, n_stock: int, market_idx: int, ln_sigma: torch.Tensor) -> dict:
        """h [S, N_total, d]; ln_sigma [N_stock] (deterministic mode) or [S, N_stock] (inside a rollout)."""
        y = self.norm(h, c, node_type).float()
        hs, hm = y[:, :n_stock], y[:, market_idx]
        ls = ln_sigma if ln_sigma.dim() == 2 else ln_sigma.unsqueeze(0)
        out = {"trend": self.trend(hm).mean(0).view(len(TREND_HORIZONS), 3)}
        if self.mode == "gen":
            out["z"] = self.ret(hs).squeeze(-1)
            out["v"] = ls + self.vol(hs).squeeze(-1)
            out["mkt_z"] = self.mkt(hm).squeeze(-1)
        else:
            out["s"] = 2.0 * ls.unsqueeze(-1) + self.ln_h + self.var(hs)
        return out
