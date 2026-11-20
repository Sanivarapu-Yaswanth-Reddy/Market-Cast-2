"""P3.2  Sector fabric: the graph the processor runs on (04 §L4).

NODES for decision date t (order is fixed and the model relies on it)
    [ stocks (N_s, varies by date) | industries | sectors | macro sectors | MARKET | 16 global nodes ]
    The hierarchy and global blocks have the same size every day, so only the stock block changes.

EDGE TYPES (each is a (src, dst) index list; attention is "dst attends to its srcs")
    up0..up3    child -> parent, level by level (stock->industry->sector->macro->market); up3 also carries
                every global node -> MARKET
    down3..down0 parent -> child (reverse)
    corr        peer stock -> stock: each stock's top-10 peers by shrunk 120-day correlation of z
    glob        global node -> sector (curated exposure table configs/edges_global.yaml)

SIGNIFICANCE PRIOR w (decision D-07): a child's share of the 60-session median traded value inside its parent
    on date t (float-adjusted market cap is not available: gap G4). Enters attention as  + lambda * log w.

CACHES (data/graph/, built by `python -m marketcast graph build`; fold-independent because every statistic
    is a trailing window ending on its own refresh date)
    tree.json     the static hierarchy for the model-panel companies
    corr.npz      monthly correlation edges, two blocks:
                    held-in block: peers and receivers are non-held-out MC1000 members
                    held-out block: receivers are held-out members, peers are held-in members
                  Training graphs use only the first block (no held-out node exists in training). Evaluation
                  graphs add the second, so inserting a held-out stock never changes a held-in stock's
                  correlation neighbourhood (inductive insertion, 05 §3).

No PyTorch Geometric: graphs are plain index arrays and the attention layer does its own scatter-softmax
(model/fabric_attn.py). One dependency fewer on a brand-new GPU architecture; logged as decision D-23.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marketcast import DATA
from marketcast.features.global_nodes import NODES as GLOBAL_NODES
from marketcast.util import get_logger, load_yaml, save_npz, write_json

log = get_logger("graph.fabric")
GRAPH = DATA / "graph"
CORR_WINDOW, CORR_MIN_OBS, TOP_K, MAX_REPLACE = 120, 80, 10, 3
EDGE_TYPES = ["up0", "up1", "up2", "up3", "down3", "down2", "down1", "down0", "corr", "glob"]


# --------------------------------------------------------------------------- static tree
def build_tree(keys: np.ndarray) -> dict:
    from marketcast.data.sector_map import load_sector_map

    sm = load_sector_map().set_index("key").reindex(keys)
    sm[["macro_sector", "sector", "industry"]] = sm[["macro_sector", "sector", "industry"]].fillna("Unclassified")
    ind_label = sm["sector"] + " | " + sm["industry"]                 # industry names are unique within a sector
    industries = sorted(ind_label.unique())
    sectors = sorted(sm["sector"].unique())
    macros = sorted(sm["macro_sector"].unique())
    ind_sector = {i: i.split(" | ")[0] for i in industries}
    sec_macro = sm.drop_duplicates("sector").set_index("sector")["macro_sector"].to_dict()
    tree = {"industries": industries, "sectors": sectors, "macros": macros,
            "stock_industry": [industries.index(x) for x in ind_label],
            "industry_sector": [sectors.index(ind_sector[i]) for i in industries],
            "sector_macro": [macros.index(sec_macro[s]) for s in sectors],
            "keys": list(map(str, keys))}
    write_json(tree, GRAPH / "tree.json")
    log.info("tree: %d industries, %d sectors, %d macro sectors for %d companies", len(industries), len(sectors), len(macros), len(keys))
    return tree


def load_tree() -> dict:
    p = GRAPH / "tree.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m marketcast graph build`")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- correlation edges
def _corr_block(zw: np.ndarray) -> np.ndarray:
    """Pairwise-complete correlation of the columns of zw [W, n] (NaN = missing), clipped inputs."""
    v = np.isfinite(zw)
    x = np.clip(np.where(v, zw, 0.0), -6.0, 6.0)
    n = np.maximum(v.sum(axis=0), 1)
    x = np.where(v, x - x.sum(axis=0) / n, 0.0)
    sd = np.sqrt((x * x).sum(axis=0) / n)
    x = x / np.where(sd > 0, sd, 1.0)
    pair_n = v.T.astype(np.float32) @ v.astype(np.float32)
    c = (x.T @ x) / np.maximum(pair_n, 1.0)
    c[pair_n < CORR_MIN_OBS / 2] = 0.0
    return np.clip(c, -1.0, 1.0)


def _stable_topk(score: np.ndarray, prev: np.ndarray | None, allowed: np.ndarray) -> np.ndarray:
    """Top-K by score with at most MAX_REPLACE changes versus `prev` (04 §L4 stability rule).

    score [n] over candidate positions (already -inf for self / invalid); prev = previous neighbour positions
    still present today (may be shorter than K); allowed = positions that may be chosen.
    """
    k = min(TOP_K, int(allowed.sum()))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    top = np.argpartition(-score, k - 1)[:k]
    top = top[np.argsort(-score[top])]
    if prev is None or len(prev) == 0:
        return top
    new = [j for j in top if j not in set(prev.tolist())]
    if len(new) <= MAX_REPLACE:
        return top
    keep_prev = prev[np.argsort(-score[prev])][: k - MAX_REPLACE]
    out = np.concatenate([keep_prev, np.array(new[:MAX_REPLACE], dtype=np.int64)])
    if len(out) < k:                                              # previous list was short: top up from the ranking
        extra = [j for j in top if j not in set(out.tolist())][: k - len(out)]
        out = np.concatenate([out, np.array(extra, dtype=np.int64)])
    return out.astype(np.int64)


def build_corr_edges(panel, tree: dict, first_date: str = "2007-06-01") -> dict:
    """Monthly top-10 peers by shrunk correlation: rho_tilde = 0.5 rho_hat + 0.5 mean rho of the sector pair."""
    z = np.asarray(panel["z"], dtype=np.float32)
    m1000 = np.asarray(panel["member1000"])
    heldout = panel.heldout
    dates = panel.dates
    sec_of = np.asarray(tree["industry_sector"])[np.asarray(tree["stock_industry"])]
    n_sec = len(tree["sectors"])
    s = pd.Series(np.arange(len(dates)), index=dates)
    month_rows = s.groupby([dates.year, dates.month]).last().to_numpy()
    month_rows = month_rows[(month_rows >= CORR_WINDOW) & (dates[month_rows] >= pd.Timestamp(first_date))]
    blocks = {"in": ([], [], []), "ho": ([], [], [])}
    ptr = {"in": [0], "ho": [0]}
    prev: dict[int, np.ndarray] = {}                                   # receiver key index -> previous peer key indices
    changed, total = 0, 0
    for r in month_rows:
        win = z[r - CORR_WINDOW + 1: r + 1]
        ok = m1000[r] & (np.isfinite(win).sum(axis=0) >= CORR_MIN_OBS)
        cin = np.flatnonzero(ok & ~heldout)                            # held-in candidates (peers AND receivers)
        cho = np.flatnonzero(ok & heldout)                             # held-out receivers
        c_all = _corr_block(win[:, np.concatenate([cin, cho])])
        n_in = len(cin)
        c = c_all[:n_in, :n_in].copy()
        onehot = np.zeros((n_in, n_sec), dtype=np.float32)
        onehot[np.arange(n_in), sec_of[cin]] = 1.0
        np.fill_diagonal(c, 0.0)
        cnt = onehot.T @ (1.0 - np.eye(n_in, dtype=np.float32)) @ onehot
        rho_bar = (onehot.T @ c @ onehot) / np.maximum(cnt, 1.0)       # [n_sec, n_sec] mean pair correlation
        pos_of = {int(kk): i for i, kk in enumerate(cin)}
        for name, recv_keys, rho_rows in (("in", cin, c), ("ho", cho, c_all[n_in:, :n_in])):
            src_l, dst_l, rho_l = blocks[name]
            for i, kk in enumerate(recv_keys):
                shrunk = 0.5 * rho_rows[i] + 0.5 * rho_bar[sec_of[kk], sec_of[cin]]
                allowed = np.ones(n_in, dtype=bool)
                if name == "in":
                    allowed[i] = False
                score = np.where(allowed, shrunk, -np.inf)
                p = prev.get(int(kk))
                p_pos = np.array([pos_of[j] for j in p if j in pos_of and (name == "ho" or pos_of[j] != i)], dtype=np.int64) if p is not None else None
                nb = _stable_topk(score, p_pos, allowed)
                peers = cin[nb]
                if p is not None:
                    changed += len(set(peers.tolist()) - set(p.tolist()))
                    total += len(peers)
                prev[int(kk)] = peers
                src_l.append(peers)
                dst_l.append(np.full(len(peers), kk, dtype=np.int64))
                rho_l.append(shrunk[nb].astype(np.float32))
            ptr[name].append(sum(len(a) for a in src_l))
    out = {"month_rows": month_rows.astype(np.int64)}
    for name in ("in", "ho"):
        src_l, dst_l, rho_l = blocks[name]
        out[f"src_{name}"] = np.concatenate(src_l).astype(np.int32) if src_l else np.zeros(0, np.int32)
        out[f"dst_{name}"] = np.concatenate(dst_l).astype(np.int32) if dst_l else np.zeros(0, np.int32)
        out[f"rho_{name}"] = np.concatenate(rho_l) if rho_l else np.zeros(0, np.float32)
        out[f"ptr_{name}"] = np.asarray(ptr[name], dtype=np.int64)
    save_npz(GRAPH / "corr.npz", **out)
    log.info("corr edges: %d monthly refreshes %s..%s; %d held-in edges, %d held-out-receiver edges; "
             "mean neighbours replaced per refresh %.2f of %d (rule: <= %d)", len(month_rows), dates[month_rows[0]].date(),
             dates[month_rows[-1]].date(), len(out["src_in"]), len(out["src_ho"]), TOP_K * changed / max(total, 1), TOP_K, MAX_REPLACE)
    return out


# --------------------------------------------------------------------------- per-date assembly
@dataclass
class DateGraph:
    n_stock: int
    n_nodes: int
    offsets: dict[str, int]                         # first node id of each block
    node_type: np.ndarray                           # [n_nodes] 0 stock,1 industry,2 sector,3 macro,4 market,5 global
    edges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    # edges[type] = (src [E], dst [E], log_w [E], feat [E])   feat = shrunk correlation for `corr`, else 0
    child_weight: dict[str, np.ndarray] = field(default_factory=dict)   # per up-level: weight of each child in its parent


class Fabric:
    """Loads the caches once and assembles the graph of any decision date in ~1 ms."""

    def __init__(self, panel, use: tuple[str, ...] = ("tree", "corr", "glob")):
        self.panel = panel
        self.use = set(use)                                    # ablation switch (P3.5): subset of {tree, corr, glob}
        self.tree = load_tree()
        if self.tree["keys"] != list(panel.keys):
            raise RuntimeError("graph tree and model panel have different companies; re-run `graph build`")
        self.stock_ind = np.asarray(self.tree["stock_industry"], dtype=np.int64)
        self.ind_sec = np.asarray(self.tree["industry_sector"], dtype=np.int64)
        self.sec_mac = np.asarray(self.tree["sector_macro"], dtype=np.int64)
        self.n_ind, self.n_sec, self.n_mac, self.n_glob = len(self.ind_sec), len(self.sec_mac), len(self.tree["macros"]), len(GLOBAL_NODES)
        self.corr = dict(np.load(GRAPH / "corr.npz")) if "corr" in self.use else None
        cfg = load_yaml("edges_global.yaml").get("exposures") or {}
        pairs, unknown = [], []
        for g, secs in cfg.items():
            for sname in secs:
                if g in GLOBAL_NODES and sname in self.tree["sectors"]:
                    pairs.append((GLOBAL_NODES.index(g), self.tree["sectors"].index(sname)))
                else:
                    unknown.append(f"{g}->{sname}")
        if unknown:
            log.warning("edges_global.yaml entries ignored (unknown node or sector): %s", unknown)
        self.glob_pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.med_turn = panel["med_turn60"]

    def assemble(self, row: int, cols: np.ndarray, include_heldout_edges: bool = False) -> DateGraph:
        """Graph for decision date `row` over stock columns `cols` (indices into panel.keys, in batch order)."""
        n = len(cols)
        o_ind, o_sec = n, n + self.n_ind
        o_mac, o_mkt = o_sec + self.n_sec, o_sec + self.n_sec + self.n_mac
        o_glob = o_mkt + 1
        n_nodes = o_glob + self.n_glob
        node_type = np.concatenate([np.zeros(n), np.ones(self.n_ind), np.full(self.n_sec, 2), np.full(self.n_mac, 3), [4],
                                    np.full(self.n_glob, 5)]).astype(np.int64)
        g = DateGraph(n, n_nodes, {"stock": 0, "industry": o_ind, "sector": o_sec, "macro": o_mac, "market": o_mkt, "global": o_glob}, node_type)
        w = np.nan_to_num(np.asarray(self.med_turn[row, cols], dtype=np.float64), nan=0.0) + 1.0   # +1 rupee: no zero weights
        ind = self.stock_ind[cols]
        ind_tot = np.bincount(ind, weights=w, minlength=self.n_ind)
        sec_tot = np.bincount(self.ind_sec, weights=ind_tot, minlength=self.n_sec)
        mac_tot = np.bincount(self.sec_mac, weights=sec_tot, minlength=self.n_mac)
        zero = lambda k: np.zeros(k, dtype=np.float32)                                      # noqa: E731
        lw = lambda a: np.log(np.maximum(a, 1e-8)).astype(np.float32)                       # noqa: E731
        if "tree" in self.use:
            live_i, live_s, live_m = np.flatnonzero(ind_tot > 0), np.flatnonzero(sec_tot > 0), np.flatnonzero(mac_tot > 0)
            sh0 = w / ind_tot[ind]
            sh1 = ind_tot[live_i] / sec_tot[self.ind_sec[live_i]]
            sh2 = sec_tot[live_s] / mac_tot[self.sec_mac[live_s]]
            sh3 = mac_tot[live_m] / max(mac_tot.sum(), 1e-8)
            ups = [(np.arange(n), o_ind + ind, sh0), (o_ind + live_i, o_sec + self.ind_sec[live_i], sh1),
                   (o_sec + live_s, o_mac + self.sec_mac[live_s], sh2), (o_mac + live_m, np.full(len(live_m), o_mkt), sh3)]
            for lvl, (src, dst, sh) in enumerate(ups):
                g.child_weight[f"up{lvl}"] = sh.astype(np.float32)
                if lvl == 3:                                    # the market also attends to every global node (w = 1)
                    gs = o_glob + np.arange(self.n_glob)
                    g.edges["up3"] = (np.concatenate([src, gs]), np.concatenate([dst, np.full(self.n_glob, o_mkt)]),
                                      np.concatenate([lw(sh), zero(self.n_glob)]), zero(len(src) + self.n_glob))
                else:
                    g.edges[f"up{lvl}"] = (src, dst, lw(sh), zero(len(src)))
                g.edges[f"down{lvl}"] = (dst, src, lw(sh), zero(len(src)))       # parent -> child, prior = child's share
        if "corr" in self.use and self.corr is not None and n:
            c = self.corr
            m = int(np.searchsorted(c["month_rows"], row, side="right")) - 1
            if m >= 0:
                pos = np.full(len(self.panel.keys), -1, dtype=np.int64)
                pos[cols] = np.arange(n)
                srcs, dsts, rhos = [], [], []
                for name in ("in", "ho") if include_heldout_edges else ("in",):
                    a, b = c[f"ptr_{name}"][m], c[f"ptr_{name}"][m + 1]
                    srcs.append(pos[c[f"src_{name}"][a:b]])
                    dsts.append(pos[c[f"dst_{name}"][a:b]])
                    rhos.append(c[f"rho_{name}"][a:b])
                src, dst, rho = np.concatenate(srcs), np.concatenate(dsts), np.concatenate(rhos)
                ok = (src >= 0) & (dst >= 0)
                src, dst, rho = src[ok], dst[ok], rho[ok]
                peer_tot = np.bincount(dst, weights=w[src], minlength=n)
                g.edges["corr"] = (src, dst, lw(w[src] / np.maximum(peer_tot[dst], 1e-8)), rho.astype(np.float32))
        if "glob" in self.use and len(self.glob_pairs):
            g.edges["glob"] = (o_glob + self.glob_pairs[:, 0], o_sec + self.glob_pairs[:, 1], zero(len(self.glob_pairs)), zero(len(self.glob_pairs)))
        for et in EDGE_TYPES:                                   # every type is always present (possibly empty)
            if et not in g.edges:
                e = np.zeros(0, dtype=np.int64)
                g.edges[et] = (e, e, zero(0), zero(0))
        return g


def main() -> int:
    from marketcast.targets.build import ModelPanel

    panel = ModelPanel()
    GRAPH.mkdir(parents=True, exist_ok=True)
    tree = build_tree(panel.keys)
    build_corr_edges(panel, tree)
    fab = Fabric(panel)
    rows = [panel.dates.searchsorted(pd.Timestamp(d)) for d in ("2008-06-30", "2015-06-30", "2023-06-30")]
    for r in rows:                                               # graph statistics (06 P3.2 acceptance)
        cols = np.flatnonzero(np.asarray(panel["member500"][r]) & ~panel.heldout)
        g = fab.assemble(int(r), cols)
        deg = np.bincount(g.edges["corr"][1], minlength=g.n_stock) if len(g.edges["corr"][1]) else np.zeros(1)
        log.info("graph %s: %d stocks, %d nodes; edges %s; max corr in-degree %d", panel.dates[r].date(), g.n_stock, g.n_nodes,
                 {k: len(v[0]) for k, v in g.edges.items() if len(v[0])}, int(deg.max()))
    return 0
