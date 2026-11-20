"""Tiny synthetic market graph for data-free model tests (no files, no real prices).

Layout mirrors graph/fabric.py: [stocks | industries | sectors | macros | market | globals].
"""
from __future__ import annotations

import numpy as np
import torch

from marketcast.features.stock import CHANNELS
from marketcast.model.marketcast import Batch, ModelConfig

F_G, F_F, T, H, N_GLOB = 9, 18, 60, 20, 16
TINY = dict(d=32, d_e=32, enc_layers=1, enc_heads=2, enc_ff=64, proc_layers=2, proc_heads=4, proc_ff=64, cond_dim=32)


def tiny_config(mode: str = "gen") -> ModelConfig:
    return ModelConfig(f_stock=len(CHANNELS), f_global=F_G, f_forcing=F_F, mode=mode, **TINY)


def make_batch(n: int = 12, seed: int = 0, n_ind: int = 4, n_sec: int = 2, n_mac: int = 1, with_state: bool = True) -> Batch:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)                              # noqa: E731
    ind = torch.arange(n) % n_ind
    ind_sec = torch.arange(n_ind) % n_sec
    sec_mac = torch.zeros(n_sec, dtype=torch.long)
    o_ind, o_sec = n, n + n_ind
    o_mac = o_sec + n_sec
    o_mkt = o_mac + n_mac
    o_glob = o_mkt + 1
    n_total = o_glob + N_GLOB
    w = torch.rand(n, generator=g) + 0.1
    ind_tot = torch.zeros(n_ind).index_add_(0, ind, w)
    sec_tot = torch.zeros(n_sec).index_add_(0, ind_sec, ind_tot)
    sh = [w / ind_tot[ind], ind_tot / sec_tot[ind_sec], sec_tot / sec_tot.sum(), torch.ones(n_mac)]
    ups = [(torch.arange(n), o_ind + ind), (o_ind + torch.arange(n_ind), o_sec + ind_sec),
           (o_sec + torch.arange(n_sec), o_mac + sec_mac), (o_mac + torch.arange(n_mac), torch.full((n_mac,), o_mkt))]
    edges, child_w = {}, {}
    for lvl, ((src, dst), s) in enumerate(zip(ups, sh)):
        child_w[f"up{lvl}"] = s
        lw = torch.log(s)
        if lvl == 3:
            gs = o_glob + torch.arange(N_GLOB)
            edges["up3"] = (torch.cat([src, gs]), torch.cat([dst, torch.full((N_GLOB,), o_mkt)]), torch.cat([lw, torch.zeros(N_GLOB)]),
                            torch.zeros(len(src) + N_GLOB))
        else:
            edges[f"up{lvl}"] = (src, dst, lw, torch.zeros(len(src)))
        edges[f"down{lvl}"] = (dst, src, lw, torch.zeros(len(src)))
    # correlation edges: each stock receives from its 3 "next" stocks
    dst = torch.arange(n).repeat_interleave(3)
    src = (dst + torch.tensor([1, 2, 3]).repeat(n)) % n
    edges["corr"] = (src, dst, torch.full((3 * n,), float(np.log(1 / 3))), torch.rand(3 * n, generator=g))
    edges["glob"] = (o_glob + torch.arange(3), o_sec + torch.arange(3) % n_sec, torch.zeros(3), torch.zeros(3))
    node_type = torch.cat([torch.zeros(n), torch.ones(n_ind), torch.full((n_sec,), 2), torch.full((n_mac,), 3), torch.tensor([4.0]),
                           torch.full((N_GLOB,), 5)]).long()
    sigma = 0.01 + 0.02 * torch.rand(n, generator=g)
    tg = {"z": rnd(H, n), "v": torch.log(sigma).expand(H, n) + 0.1 * rnd(H, n), "mask": torch.ones(H, n, dtype=torch.bool),
          "weight": torch.ones(n), "sigma": sigma, "heldout": torch.zeros(n, dtype=torch.bool),
          "rv2_h": (sigma ** 2).unsqueeze(-1) * torch.tensor([1.0, 5.0, 20.0]) * torch.exp(0.2 * rnd(n, 3)),
          "rv2_mask": torch.ones(n, 3, dtype=torch.bool), "mkt_z": torch.tensor(0.3), "mkt_valid": torch.tensor(True),
          "mkt_sigma": torch.tensor(0.01), "trend": torch.tensor([1, 2]), "sig_w": w / w.sum()}
    if with_state:
        ell = sigma.unsqueeze(-1) * rnd(n, 252)
        tg.update(ell_hist=ell, z_hist=ell[:, -60:] / sigma.unsqueeze(-1), rv2_hist=(sigma ** 2).unsqueeze(-1) * torch.exp(0.3 * rnd(n, 60)),
                  v_hist=torch.log(sigma).unsqueeze(-1) + 0.1 * rnd(n, 20), s2=sigma ** 2, sigma_floor=torch.tensor(0.004),
                  has_fo=torch.arange(n) % 2 == 0)
    return Batch(rnd(n, T, len(CHANNELS)), torch.ones(n, T, dtype=torch.bool), rnd(N_GLOB, T, F_G), rnd(T, F_F), rnd(H, F_F), torch.log(sigma),
                 node_type, edges, child_w, n_stock=n, market_idx=o_mkt, targets=tg, meta={"row": 0, "date": "synthetic", "cols": np.arange(n)})
