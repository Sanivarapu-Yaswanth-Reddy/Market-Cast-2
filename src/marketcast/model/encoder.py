"""L2  Temporal encoder (04 §L2): turns each node's 60-day history into one state vector.

    x[n, tau, :] -> Linear -> + learned position embedding -> 4 pre-LN Transformer blocks (masked)
                 -> attention pooling with one learned query (masked) -> Linear(d_e -> d)

Shared by all stocks (one instance); the global nodes use a second instance with its own weights.
Masked days (not traded, not listed yet) are excluded from attention and from pooling. A node whose whole
window is masked would give NaN attention rows, so the most recent position is always kept visible; such
nodes are then overwritten by the "new listing" rule in marketcast.py.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, f_in: int, d_e: int = 128, d_out: int = 256, n_layers: int = 4, n_heads: int = 4, d_ff: int = 512,
                 dropout: float = 0.1, max_len: int = 60):
        super().__init__()
        self.inp = nn.Linear(f_in, d_e)
        self.pos = nn.Parameter(torch.zeros(max_len, d_e))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_e, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_e)
        self.query = nn.Parameter(torch.zeros(d_e))
        nn.init.normal_(self.query, std=0.02)
        self.key = nn.Linear(d_e, d_e)
        self.out = nn.Linear(d_e, d_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B, T, F] (already normalised, missing = 0), mask [B, T] bool (True = real observation) -> [B, d_out]."""
        t = x.shape[1]
        mask = mask.clone()
        mask[:, -1] = True                                        # never a fully masked row (see module docstring)
        h = self.inp(x) + self.pos[-t:].to(x.dtype)               # right-aligned: position -1 is always "today"
        h = self.norm(self.blocks(h, src_key_padding_mask=~mask))
        score = (self.key(h) * self.query.to(h.dtype)).sum(-1) / h.shape[-1] ** 0.5
        score = score.float().masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1).to(h.dtype)
        return self.out((w.unsqueeze(-1) * h).sum(1))
