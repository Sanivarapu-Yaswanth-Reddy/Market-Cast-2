"""L4  Edge attention with the significance prior (04 §L4).

For one edge set (src -> dst), per head h and relation r (e.g. the tree level):

    score_ij = q_i . k_j / sqrt(d_head)  +  lambda_{r,h} * log w_j  +  b_{r,h}  +  u_{r,h} * feat_ij
    alpha_ij = softmax over the senders j of receiver i
    hub cap  : for receivers with >= 3 senders, alpha <- min(alpha, 0.5), then renormalise once
    out_i    = W_o  sum_j alpha_ij * LayerNorm(v_j)

* log w_j is the sender's significance (traded-value share inside the parent) and enters as an additive
  prior, so with zero learned attention a parent simply takes the significance-weighted mean of its children.
* feat_ij is one scalar edge feature (the shrunk correlation on `corr` edges, 0 elsewhere).
* Receivers with no senders get a zero message (their state passes through unchanged).

Implemented with plain index operations on an edge list (no PyTorch Geometric): tensors are [S, N, d] where
S = number of noise scenarios; the edge list is shared across scenarios.
"""
from __future__ import annotations

import math

import torch
from torch import nn

HUB_CAP, HUB_MIN_SENDERS = 0.5, 3


def scatter_softmax(score: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of score [S, E, H] within groups of equal dst [E] (n receivers). float32 inside for stability."""
    score = score.float()
    idx = dst.view(1, -1, 1).expand_as(score)
    mx = score.new_full((score.shape[0], n, score.shape[2]), float("-inf")).scatter_reduce(1, idx, score, "amax", include_self=True)
    ex = (score - mx.gather(1, idx)).exp()
    den = score.new_zeros((score.shape[0], n, score.shape[2])).scatter_add(1, idx, ex)
    return ex / den.gather(1, idx).clamp_min(1e-20)


class EdgeAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, n_rel: int = 1, lambda_init: float = 0.5, hub_cap: bool = True):
        super().__init__()
        if d % n_heads:
            raise ValueError("d must be divisible by n_heads")
        self.d, self.h, self.dh, self.hub_cap = d, n_heads, d // n_heads, hub_cap
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d)
        self.msg_norm = nn.LayerNorm(d)
        self.lam = nn.Parameter(torch.full((n_rel, n_heads), float(lambda_init)))    # significance prior strength
        self.bias = nn.Parameter(torch.zeros(n_rel, n_heads))
        self.feat = nn.Parameter(torch.zeros(n_rel, n_heads))                        # weight on the scalar edge feature
        nn.init.zeros_(self.o.weight)        # every pass starts as the identity (residual branch = 0)
        nn.init.zeros_(self.o.bias)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, log_w: torch.Tensor, feat: torch.Tensor,
                rel: int = 0, return_alpha: bool = False):
        """h [S, N, d] (already normalised by the caller); src/dst [E] long; log_w/feat [E]. Returns [S, N, d]."""
        s, n, _ = h.shape
        if src.numel() == 0:
            out = torch.zeros_like(h)
            return (out, None) if return_alpha else out
        e = src.numel()
        q = self.q(h)[:, dst].view(s, e, self.h, self.dh)
        k = self.k(h)[:, src].view(s, e, self.h, self.dh)
        v = self.msg_norm(self.v(h))[:, src].view(s, e, self.h, self.dh)
        score = (q * k).sum(-1).float() / math.sqrt(self.dh)
        score = score + (self.lam[rel].float() * log_w.float().unsqueeze(-1) + self.bias[rel].float()
                         + self.feat[rel].float() * feat.float().unsqueeze(-1)).unsqueeze(0)
        alpha = scatter_softmax(score, dst, n)                                       # [S, E, H]
        if self.hub_cap:
            deg = torch.bincount(dst, minlength=n)
            capped = torch.where((deg[dst] >= HUB_MIN_SENDERS).view(1, -1, 1), alpha.clamp(max=HUB_CAP), alpha)
            idx = dst.view(1, -1, 1).expand_as(capped)
            den = capped.new_zeros((s, n, self.h)).scatter_add(1, idx, capped)
            alpha = capped / den.gather(1, idx).clamp_min(1e-20)
        msg = (alpha.to(v.dtype).unsqueeze(-1) * v).reshape(s, e, self.d)
        agg = h.new_zeros((s, n, self.d)).index_add_(1, dst, msg)
        out = self.o(agg)
        return (out, alpha) if return_alpha else out
