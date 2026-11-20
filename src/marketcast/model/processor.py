"""L5  Processor: a stack of graph-transformer layers over the sector fabric (04 §L5).

One layer =
    UP      level-synchronous, bottom-up:  industry <- stocks, sector <- industries, macro <- sectors,
            market <- macros + all global nodes.   Run in order, so a single stock's signal reaches the market
            node inside ONE layer.
    DOWN    top-down: macro <- market, ..., stock <- industry, each scaled by a learned per-node gate
            g_i = sigmoid(Linear(h_i))  (the node's sensitivity to its parent: a learned "beta").
    ACROSS  stock <- correlation peers;  sector <- curated global exposures.
    MLP     node-wise, behind a noise-conditioned norm.
Every pass is pre-norm + residual, and every norm is a CondLN, so the scenario noise can steer each stage.

The fabric here has three hierarchy levels below the market (industry, sector, macro sector) because the
NSE basic-industry level is not on disk (gap G3); the level count comes from the edge names, not from code.
The "thin basic industry mixes with its parent" rule of 04 §L5 is therefore not needed yet.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from marketcast.model.fabric_attn import EdgeAttention
from marketcast.model.fgn import CondLN

N_LEVELS = 4                 # up0..up3 / down0..down3


class ProcessorLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, cond_dim: int, dropout: float = 0.1, hub_cap: bool = True):
        super().__init__()
        self.up = EdgeAttention(d, n_heads, n_rel=N_LEVELS, lambda_init=1.0, hub_cap=hub_cap)
        self.down = EdgeAttention(d, n_heads, n_rel=N_LEVELS, lambda_init=0.5, hub_cap=False)   # one parent per child: no hubs
        self.across = EdgeAttention(d, n_heads, n_rel=1, lambda_init=0.5, hub_cap=hub_cap)
        self.glob = EdgeAttention(d, n_heads, n_rel=1, lambda_init=0.5, hub_cap=hub_cap)
        self.n_up, self.n_down, self.n_across, self.n_mlp = (CondLN(d, cond_dim) for _ in range(4))
        self.gate = nn.Linear(d, 1)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor | None, node_type: torch.Tensor, edges: dict) -> torch.Tensor:
        for lvl in range(N_LEVELS):                                   # UP (sequential on purpose)
            src, dst, lw, ft = edges[f"up{lvl}"]
            if src.numel():
                h = h + self.drop(self.up(self.n_up(h, c, node_type), src, dst, lw, ft, rel=lvl))
        for lvl in reversed(range(N_LEVELS)):                         # DOWN with the sensitivity gate
            src, dst, lw, ft = edges[f"down{lvl}"]
            if src.numel():
                hn = self.n_down(h, c, node_type)
                h = h + torch.sigmoid(self.gate(hn)) * self.drop(self.down(hn, src, dst, lw, ft, rel=lvl))
        src, dst, lw, ft = edges["corr"]                              # ACROSS
        gs, gd, glw, gft = edges["glob"]
        if src.numel() or gs.numel():
            hn = self.n_across(h, c, node_type)
            if src.numel():
                h = h + self.drop(self.across(hn, src, dst, lw, ft))
            if gs.numel():
                h = h + self.drop(self.glob(hn, gs, gd, glw, gft))
        return h + self.drop(self.mlp(self.n_mlp(h, c, node_type)))


class Processor(nn.Module):
    def __init__(self, d: int = 256, n_layers: int = 8, n_heads: int = 8, d_ff: int = 1024, cond_dim: int = 128,
                 dropout: float = 0.1, hub_cap: bool = True, grad_checkpoint: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(ProcessorLayer(d, n_heads, d_ff, cond_dim, dropout, hub_cap) for _ in range(n_layers))
        self.grad_checkpoint = grad_checkpoint

    def forward(self, h: torch.Tensor, c: torch.Tensor | None, node_type: torch.Tensor, edges: dict) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpoint and self.training and h.requires_grad:
                # trades compute for memory on the 8 GB card (09 §4): activations are recomputed in backward
                h = checkpoint(lambda x, c_=c, layer_=layer: layer_(x, c_, node_type, edges), h, use_reentrant=False)
            else:
                h = layer(h, c, node_type, edges)
        return h
