"""Assembles one `Batch` per decision date from the feature store, the model panel and the fabric (L1).

WHAT A BATCH IS: the whole market graph of one date (04 "A batch is 2 decision dates x 2 noise samples";
dates are looped with gradient accumulation, noise samples are a tensor axis).

    decision date t (row) ->
        stocks    = universe members on t with a valid price state today   [training: held-out removed]
        x_s       = normalised stock channels of days t-59..t              [N, 60, F_s]
        x_g       = normalised global-node channels of days t-59..t        [16, 60, F_g]
        f_hist    = forcings of t-59..t ;  f_fut = forcings of t+1..t+20
        graph     = Fabric.assemble(t, stocks)
        targets   = what happened on t+1..t+K (never stored shifted: paired here, so the fold owns the purge)

LEAKAGE GUARDS IN THIS FILE
    * rows come from eval.folds.split_rows (purged, embargoed, vault-checked); `StoreDataset` re-checks the vault
    * train=True drops held-out companies from the node set, so they are in no batch, no loss, no graph
    * every input index is <= t; every target index is > t (asserted in tests/test_data_loader.py)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from marketcast.eval.folds import assert_not_vault
from marketcast.features.build import FEATURES, load_normalizers
from marketcast.features.stock import CHANNELS
from marketcast.graph.fabric import Fabric
from marketcast.model.marketcast import Batch
from marketcast.targets.build import ModelPanel
from marketcast.targets.labels import trend_labels

HIST, HORIZON = 60, 20
WEIGHT_CAP = 5.0
DET_H = (1, 5, 20)
_MPRICE = CHANNELS.index("m_price")


class Store:
    """Read-only handles on everything a batch needs. Cheap to construct; arrays are memory-mapped."""

    def __init__(self, fold: str, use_edges: tuple[str, ...] = ("tree", "corr", "glob"), universe: str = "member500"):
        self.fold, self.universe = fold, universe
        self.panel = ModelPanel()
        meta = json.loads((FEATURES / "meta.json").read_text(encoding="utf-8"))
        if meta["keys"] != self.panel.keys.tolist() or meta["stock_channels"] != CHANNELS:
            raise RuntimeError("feature store does not match the model panel / channel list; re-run `features build`")
        self.feats = np.load(FEATURES / "stock.npy", mmap_mode="r")
        self.glob = np.load(FEATURES / "global.npy", mmap_mode="r")
        self.forc = np.load(FEATURES / "forcings.npy", mmap_mode="r")
        self.snorm, self.gnorm = load_normalizers(fold)
        self.fabric = Fabric(self.panel, use_edges)
        self.dates = self.panel.dates
        self.heldout = self.panel.heldout
        mkt = self.panel.market()
        self.mkt = mkt
        self.m_ell, self.m_sigma = mkt["m_ell"].to_numpy(float), mkt["m_sigma"].to_numpy(float)
        self.m_bad = mkt["session_bad"].to_numpy(bool)
        self.floor = mkt["sigma_floor"].to_numpy(float)
        self.trend = np.stack([trend_labels(self.m_ell, self.m_sigma, h) for h in (5, 20)], axis=1)    # [T, 2]
        self.dims = {"f_stock": len(CHANNELS), "f_global": int(self.glob.shape[1] * 0 + self.glob.shape[2]), "f_forcing": int(self.forc.shape[1])}

    def stock_cols(self, row: int, train: bool) -> np.ndarray:
        member = np.asarray(self.panel[self.universe][row])
        ok = member & (np.asarray(self.feats[row, :, _MPRICE]) > 0)
        if train:
            ok &= ~self.heldout
        return np.flatnonzero(ok)

    def make(self, row: int, train: bool, k_steps: int = 1, with_state: bool = False, max_nodes: int | None = None,
             rng: np.random.Generator | None = None) -> Batch:
        """Build the batch of decision row `row`. k_steps = number of future days with targets (1 = single-step).
        with_state adds the raw history buffers the rollout needs. max_nodes subsamples stocks (smoke tests)."""
        p = self.panel
        if row < HIST - 1 + 252:
            raise ValueError(f"decision row {row} is too early: need {HIST + 251} sessions of history")
        cols = self.stock_cols(row, train)
        if max_nodes and len(cols) > max_nodes:
            cols = np.sort((rng or np.random.default_rng(row)).choice(cols, max_nodes, replace=False))
        n = len(cols)
        lo = row - HIST + 1
        raw = np.asarray(self.feats[lo: row + 1][:, cols], dtype=np.float32)                 # [T, N, F]
        x_s = torch.from_numpy(self.snorm.apply(raw).transpose(1, 0, 2).copy())
        mask_s = torch.from_numpy((raw[:, :, _MPRICE] > 0).T.copy())
        g_raw = np.asarray(self.glob[lo: row + 1], dtype=np.float32)                         # [T, G, C]
        t_len, g_n, g_c = g_raw.shape
        x_g = torch.from_numpy(self.gnorm.apply(g_raw.reshape(t_len, g_n * g_c)).reshape(t_len, g_n, g_c).transpose(1, 0, 2).copy())
        f_hist = torch.from_numpy(np.array(self.forc[lo: row + 1], dtype=np.float32))          # np.array: writable copy of the memmap
        f_fut = torch.from_numpy(np.array(self.forc[row + 1: row + 1 + HORIZON], dtype=np.float32))
        gr = self.fabric.assemble(row, cols, include_heldout_edges=not train)
        edges = {k: (torch.from_numpy(a.astype(np.int64)), torch.from_numpy(b.astype(np.int64)), torch.from_numpy(c), torch.from_numpy(d))
                 for k, (a, b, c, d) in gr.edges.items()}
        child_w = {k: torch.from_numpy(v) for k, v in gr.child_weight.items()}
        sigma = np.asarray(p["sigma"][row, cols], dtype=np.float32)

        # ---- targets: rows row+1 .. row+k (clipped at the panel end: missing days are simply masked)
        t_end = min(row + k_steps, len(self.dates) - 1)
        fut = slice(row + 1, t_end + 1)
        pad = k_steps - (t_end - row)
        def _fut(name, fill):                                                                # noqa: E306
            a = np.asarray(p[name][fut][:, cols])
            if pad:
                a = np.concatenate([a, np.full((pad, n), fill, dtype=a.dtype)])
            return a
        ell_f = _fut("ell", np.nan).astype(np.float32)
        tmask = _fut("target_mask", False).astype(bool)
        tz = ell_f / sigma[None, :]                                                          # decision-date sigma units
        tmask &= np.isfinite(tz)
        wt = np.sqrt(np.nan_to_num(np.asarray(p["med_turn60"][row, cols], dtype=np.float64), nan=0.0))
        wt = np.minimum(wt / max(wt.mean(), 1e-12), WEIGHT_CAP) if n else wt                 # 03 §6: mean 1, capped at 5
        wt = wt / max(wt.mean(), 1e-12) if n else wt
        tg = {"z": torch.from_numpy(tz), "v": torch.from_numpy(_fut("v", np.nan).astype(np.float32)), "mask": torch.from_numpy(tmask),
              "weight": torch.from_numpy(wt.astype(np.float32)), "sigma": torch.from_numpy(sigma),
              "heldout": torch.from_numpy(self.heldout[cols].copy())}
        # deterministic targets: realized variance summed over 1 / 5 / 20 days; a window is valid only if complete
        rv2_h, rv2_m = np.full((n, len(DET_H)), np.nan, dtype=np.float32), np.zeros((n, len(DET_H)), dtype=bool)
        for i, h in enumerate(DET_H):
            if row + h < len(self.dates):
                w_rv = np.asarray(p["rv2"][row + 1: row + 1 + h][:, cols], dtype=np.float64)
                w_ok = np.asarray(p["target_mask"][row + 1: row + 1 + h][:, cols]).all(axis=0) & np.isfinite(w_rv).all(axis=0)
                rv2_h[:, i] = np.where(w_ok, w_rv.sum(axis=0), np.nan)
                rv2_m[:, i] = w_ok
        tg["rv2_h"], tg["rv2_mask"] = torch.from_numpy(rv2_h), torch.from_numpy(rv2_m)
        # market head and trend cross-check labels
        nxt = row + 1
        mz = self.m_ell[nxt] / self.m_sigma[row] if nxt < len(self.dates) else np.nan
        valid = bool(np.isfinite(mz)) and not self.m_bad[min(nxt, len(self.dates) - 1)]
        tg["mkt_z"] = torch.tensor(float(mz) if valid else 0.0)
        tg["mkt_valid"] = torch.tensor(valid)
        tg["mkt_sigma"] = torch.tensor(float(self.m_sigma[row]) if np.isfinite(self.m_sigma[row]) else 1.0)
        tg["trend"] = torch.from_numpy(self.trend[row].astype(np.int64))
        sw = np.nan_to_num(np.asarray(p["med_turn60"][row, cols], dtype=np.float64), nan=0.0)
        tg["sig_w"] = torch.from_numpy((sw / max(sw.sum(), 1e-12)).astype(np.float32))
        if with_state:                                                                       # raw buffers for the rollout
            hist = lambda name, m: torch.from_numpy(np.asarray(p[name][row - m + 1: row + 1][:, cols], dtype=np.float32).T.copy())   # noqa: E731
            tg.update(ell_hist=hist("ell", 252), z_hist=hist("z", 60), rv2_hist=hist("rv2", 60), v_hist=hist("v", 20),
                      s2=torch.from_numpy(np.asarray(p["s2"][row, cols], dtype=np.float32)),
                      sigma_floor=torch.tensor(float(self.floor[row])), has_fo=torch.from_numpy(np.asarray(p["has_fo"][row, cols]).astype(bool)))
        return Batch(x_s, mask_s, x_g, f_hist, f_fut, torch.from_numpy(np.log(sigma)), torch.from_numpy(gr.node_type), edges, child_w,
                     n_stock=n, market_idx=gr.offsets["market"], targets=tg,
                     meta={"row": int(row), "date": str(self.dates[row].date()), "cols": cols})


class StoreDataset(torch.utils.data.Dataset):
    """Map-style dataset over decision rows. The Store is opened lazily in each worker process (memmaps do not
    survive pickling on Windows)."""

    def __init__(self, fold: str, rows: np.ndarray, train: bool, k_steps: int = 1, with_state: bool = False, max_nodes: int | None = None,
                 use_edges: tuple[str, ...] = ("tree", "corr", "glob"), universe: str = "member500", allow_vault: bool = False):
        self.fold, self.rows, self.train, self.k_steps, self.with_state = fold, np.asarray(rows, dtype=np.int64), train, k_steps, with_state
        self.max_nodes, self.use_edges, self.universe = max_nodes, use_edges, universe
        self._store: Store | None = None
        dates = ModelPanel().dates
        assert_not_vault(dates[self.rows], allow_vault, who=f"StoreDataset({fold})")
        if train and len(self.rows) and dates[self.rows].max() + pd.Timedelta(days=0) >= pd.Timestamp("2026-01-01"):
            raise PermissionError("training rows inside the vault")

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.fold, self.use_edges, self.universe)
        return self._store

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Batch:
        return self.store.make(int(self.rows[i]), self.train, self.k_steps, self.with_state, self.max_nodes)


def loader(ds: StoreDataset, shuffle: bool, seed: int = 0, workers: int = 0) -> torch.utils.data.DataLoader:
    """One Batch per iteration (batch_size=None: the 'batch' is already a whole market graph)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=shuffle, generator=g, num_workers=workers,
                                       persistent_workers=workers > 0, collate_fn=None)
