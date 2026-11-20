"""The MarketCast network: encode -> process -> decode on the sector fabric (04, layers L2-L7).

TWO-STAGE FORWARD (kept separate because the first stage has no noise and is the expensive one)
    encode(batch)             histories -> initial node states h0 [N_total, d]     (once per date)
    simulate(h0, batch, eps)  h0 + target-day forcing -> processor (noise-conditioned) -> heads   (per scenario)
    forward(batch, eps)       = simulate(encode(batch), ...)

`Batch` is everything the model needs for ONE decision date. It is produced by train/data_loader.py and by
the rollout (which rewrites the history tensors with simulated days). Node order follows graph/fabric.py:
[stocks | industries | sectors | macro sectors | market | globals].

HIERARCHY INITIALISATION: an industry starts as the significance-weighted mean of its stocks' encoded states,
a sector as the weighted mean of its industries, and so on up to the market; a learned node-type embedding is
added to every node. New listings (fewer than `new_listing_min` valid days in the window) take their
industry's state plus a learned "new listing" embedding instead of their own thin history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from marketcast.model.encoder import TemporalEncoder
from marketcast.model.fgn import N_NODE_TYPES, NOISE_DIM, NoiseEmbed
from marketcast.model.heads import Heads
from marketcast.model.processor import N_LEVELS, Processor


@dataclass
class ModelConfig:
    f_stock: int                    # stock channels per day
    f_global: int                   # global-node channels per day
    f_forcing: int                  # calendar forcings per day
    mode: str = "gen"               # "gen" (FGN samples, CRPS) or "det" (log-variance, QLIKE)
    d: int = 256
    d_e: int = 128
    enc_layers: int = 4
    enc_heads: int = 4
    enc_ff: int = 512
    proc_layers: int = 8
    proc_heads: int = 8
    proc_ff: int = 1024
    cond_dim: int = 128
    noise_dim: int = NOISE_DIM
    dropout: float = 0.1
    hist_len: int = 60
    max_steps: int = 20             # rollout horizon (scales the step forcing)
    new_listing_min: int = 20
    hub_cap: bool = True
    grad_checkpoint: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Batch:
    x_s: torch.Tensor               # [N, T, F_s]  normalised stock history (or [S, N, T, F_s] inside a rollout)
    mask_s: torch.Tensor            # [N, T] bool  day is a real observation
    x_g: torch.Tensor               # [G, T, F_g]  normalised global-node history
    f_hist: torch.Tensor            # [T, F_f]     forcings of the history days
    f_fut: torch.Tensor             # [H, F_f]     forcings of target days t+1 .. t+H
    ln_sigma: torch.Tensor          # [N]          ln sigma_hat_t (base of the volatility head)
    node_type: torch.Tensor         # [N_total] long
    edges: dict                     # edge type -> (src, dst, log_w, feat) tensors
    child_w: dict                   # "up{l}" -> child weight inside its parent (hierarchy initialisation)
    n_stock: int = 0
    market_idx: int = 0
    targets: dict = field(default_factory=dict)     # z, v, mask, weight, ... (see train/data_loader.py)
    meta: dict = field(default_factory=dict)        # row, date, cols (numpy), for bookkeeping

    def to(self, device) -> "Batch":
        mv = lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x     # noqa: E731
        return Batch(mv(self.x_s), mv(self.mask_s), mv(self.x_g), mv(self.f_hist), mv(self.f_fut), mv(self.ln_sigma), mv(self.node_type),
                     {k: tuple(mv(t) for t in v) for k, v in self.edges.items()}, {k: mv(v) for k, v in self.child_w.items()},
                     self.n_stock, self.market_idx, {k: mv(v) for k, v in self.targets.items()}, self.meta)


class MarketCast(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        enc = dict(d_e=cfg.d_e, d_out=cfg.d, n_layers=cfg.enc_layers, n_heads=cfg.enc_heads, d_ff=cfg.enc_ff, dropout=cfg.dropout, max_len=cfg.hist_len)
        self.stock_enc = TemporalEncoder(cfg.f_stock + cfg.f_forcing, **enc)
        self.global_enc = TemporalEncoder(cfg.f_global + cfg.f_forcing, **enc)
        self.type_emb = nn.Embedding(N_NODE_TYPES, cfg.d)
        self.global_id_emb = nn.Embedding(64, cfg.d)                 # which global node (VIX, oil, ...): they are not exchangeable
        self.new_listing = nn.Parameter(torch.zeros(cfg.d))
        self.forcing = nn.Sequential(nn.Linear(cfg.f_forcing + 1, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
        self.noise = NoiseEmbed(cfg.noise_dim, cfg.cond_dim)
        self.processor = Processor(cfg.d, cfg.proc_layers, cfg.proc_heads, cfg.proc_ff, cfg.cond_dim, cfg.dropout, cfg.hub_cap, cfg.grad_checkpoint)
        self.heads = Heads(cfg.d, cfg.cond_dim, cfg.mode)

    # ------------------------------------------------------------------ stage 1: encode (no noise)
    def encode(self, b: Batch) -> torch.Tensor:
        """Returns h0 [N_total, d], or [S, N_total, d] when b.x_s carries a scenario axis (rollout)."""
        if b.x_s.dim() == 4:
            s = b.x_s.shape[0]
            hs = self._encode_stocks(b.x_s.flatten(0, 1), b.mask_s.repeat(s, 1) if b.mask_s.dim() == 2 else b.mask_s.flatten(0, 1), b.f_hist)
            hs = hs.view(s, b.n_stock, -1)
        else:
            hs = self._encode_stocks(b.x_s, b.mask_s, b.f_hist).unsqueeze(0)
        g = b.x_g.shape[0]
        xg = torch.cat([b.x_g, b.f_hist.unsqueeze(0).expand(g, -1, -1)], dim=-1)
        hg = self.global_enc(xg, torch.ones(xg.shape[:2], dtype=torch.bool, device=xg.device))
        hg = hg + self.global_id_emb(torch.arange(g, device=xg.device))
        h0 = self._init_hierarchy(hs, hg, b)
        return h0 if b.x_s.dim() == 4 else h0[0]

    def _encode_stocks(self, x: torch.Tensor, mask: torch.Tensor, f_hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, f_hist.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)], dim=-1)
        return self.stock_enc(x, mask)

    def _init_hierarchy(self, hs: torch.Tensor, hg: torch.Tensor, b: Batch) -> torch.Tensor:
        """hs [S', N, d] stocks, hg [G, d] globals -> [S', N_total, d] with hierarchy nodes filled bottom-up."""
        s, n, d = hs.shape
        n_total = b.node_type.numel()
        h = hs.new_zeros((s, n_total, d))
        valid_days = b.mask_s.sum(-1) if b.mask_s.dim() == 2 else b.mask_s.view(s, n, -1).sum(-1)[0]
        weak = valid_days < self.cfg.new_listing_min                                  # [N] new listings
        h[:, :n] = hs
        for lvl in range(N_LEVELS):
            src, dst, _, _ = b.edges[f"up{lvl}"]
            if src.numel() == 0:
                continue
            w = b.child_w[f"up{lvl}"].to(hs.dtype)
            k = w.numel()                                    # tree children only (up3 also lists global -> market edges)
            src, dst = src[:k], dst[:k]
            if lvl == 0:
                w = w * (~weak[src]).to(w.dtype)             # a thin history does not vote on its industry's state
            den = h.new_zeros(n_total).index_add_(0, dst, w).clamp_min(1e-8)
            h = h.index_add(1, dst, h[:, src] * (w / den[dst]).view(1, -1, 1))
            if lvl == 0 and bool(weak.any()):
                ind_of = torch.zeros(n, dtype=torch.long, device=h.device).index_copy_(0, src, dst)
                idx = torch.nonzero(weak).squeeze(-1)
                h[:, idx] = h[:, ind_of[idx]] + self.new_listing.to(h.dtype)
        h[:, n_total - hg.shape[0]:] = hg.unsqueeze(0).to(h.dtype)
        return h + self.type_emb(b.node_type).to(h.dtype)

    # ------------------------------------------------------------------ stage 2: simulate one step
    def simulate(self, h0: torch.Tensor, b: Batch, eps: torch.Tensor | None, step: int = 1, n_samples: int | None = None,
                 ln_sigma: torch.Tensor | None = None) -> dict:
        """One forecast step. eps [S, noise_dim] (None = deterministic). step = rollout step k (1-based)."""
        if h0.dim() == 2:
            s = eps.shape[0] if eps is not None else (n_samples or 1)
            h = h0.unsqueeze(0).expand(s, -1, -1)
        else:
            h = h0
        f = torch.cat([b.f_fut[step - 1], b.f_fut.new_tensor([step / self.cfg.max_steps])])
        h = h + self.forcing(f).to(h.dtype)
        c = self.noise(eps) if eps is not None else None
        h = self.processor(h, c, b.node_type, b.edges)
        return self.heads(h, c, b.node_type, b.n_stock, b.market_idx, b.ln_sigma if ln_sigma is None else ln_sigma)

    def forward(self, b: Batch, eps: torch.Tensor | None = None, n_samples: int | None = None) -> dict:
        return self.simulate(self.encode(b), b, eps, 1, n_samples)

    def sample_noise(self, n: int, device=None, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(n, self.cfg.noise_dim, device=device, generator=generator)


def count_parameters(model: nn.Module) -> dict:
    groups = {"encoders": ["stock_enc", "global_enc"], "processor": ["processor"], "heads_fgn_misc": ["heads", "noise", "forcing", "type_emb",
                                                                                                   "global_id_emb", "new_listing"]}
    out = {g: sum(p.numel() for n, p in model.named_parameters() if n.split(".")[0] in names) for g, names in groups.items()}
    out["total"] = sum(p.numel() for p in model.parameters())
    return out
