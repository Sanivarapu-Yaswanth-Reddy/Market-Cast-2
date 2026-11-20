"""L6  Stochastic core, FGN style (04 §L6).

    eps ~ N(0, I_16)          ONE draw per scenario per rollout step, shared by every node and every layer
    c   = MLP(16 -> 128 -> 128, SiLU)(eps)
    CondLN(x) = LayerNorm(x) * (1 + gamma(c)) + beta(c)      gamma, beta: Linear(128 -> d) per node type

gamma and beta are ZERO-INITIALISED, so at the start of training every scenario is identical and the network
is the deterministic model (which is also why a P3 checkpoint can warm-start P4). Because a single
16-dimensional vector modulates the whole market, the cheapest way for the network to lower CRPS is to learn
coherent market moods (risk-off, sector rotation) rather than independent noise per stock.

With c = None the layer is a plain LayerNorm: that is the Phase 3 deterministic variant.
"""
from __future__ import annotations

import torch
from torch import nn

NOISE_DIM = 16
N_NODE_TYPES = 6        # stock, industry, sector, macro, market, global (graph/fabric.py node_type codes)


class NoiseEmbed(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM, cond_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(noise_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim), nn.SiLU())

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        return self.net(eps)


class CondLN(nn.Module):
    """LayerNorm whose gain and bias are shifted by the noise embedding, separately per node type."""

    def __init__(self, d: int, cond_dim: int = 128, n_types: int = N_NODE_TYPES):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.d, self.n_types = d, n_types
        # Factorised per-type maps: one Linear(cond -> d) shared by all node types, times a learned per-type gain
        # vector. A full Linear per node type costs 6x the parameters (12 M across the processor) for no
        # demonstrated benefit; this keeps "per layer and per node type" at ~0.26 M per layer.
        self.gamma = nn.Linear(cond_dim, d)
        self.beta = nn.Linear(cond_dim, d)
        self.type_gain_g = nn.Parameter(torch.ones(n_types, d))
        self.type_gain_b = nn.Parameter(torch.ones(n_types, d))
        for lin in (self.gamma, self.beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None, node_type: torch.Tensor) -> torch.Tensor:
        """x [S, N, d]; c [S, cond_dim] or None; node_type [N] long."""
        y = self.ln(x)
        if c is None:
            return y
        g = self.gamma(c).unsqueeze(1) * self.type_gain_g[node_type].unsqueeze(0)      # [S, N, d]
        b = self.beta(c).unsqueeze(1) * self.type_gain_b[node_type].unsqueeze(0)
        return y * (1.0 + g.to(y.dtype)) + b.to(y.dtype)
