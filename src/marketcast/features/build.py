"""P3.1  Feature store.

LAYOUT (data/features/)                                   [copy this folder + data/targets + data/graph to the GPU PC]
    stock.npy        float16 memmap [T, K, F_s]   RAW stock channels (features/stock.py), row t = state at close of t
    global.npy       float32 [T, 16, 9]           RAW global-node channels (features/global_nodes.py)
    forcings.npy     float32 [T + 25, F_f]        calendar forcings, incl. 25 projected future sessions
    meta.json        dates, keys, channel names, price-channel indices
    <fold>/norm.json per-fold robust normalisers for stock and global channels (fit on train dates/stocks only)

WHY RAW + PER-FOLD STATS (instead of one normalised tensor per fold as sketched in 06 P3.1)
    Raw channels are fold-independent (trailing windows only), so they are computed once (~1.4 GB) and the
    ten folds differ only by a few hundred numbers in norm.json. The loader normalises on the fly, which is
    cheap, and the rollout needs raw values anyway to update price channels. Logged as decision D-24.

HELD-OUT ISOLATION: normaliser samples, the sector-rank reference set and the market series all exclude
held-out companies. Their own feature rows are still computed (they are scored inductively at test time).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from marketcast import DATA
from marketcast.features import forcings as FO
from marketcast.features import global_nodes as GN
from marketcast.features import stock as ST
from marketcast.features.normalize import Normalizer, fit
from marketcast.util import commit, get_logger, load_yaml, save_npy, write_json

log = get_logger("features.build")
FEATURES = DATA / "features"
N_FUTURE = 25                 # projected sessions appended to the forcing table (>= rollout horizon 20)
FIT_EVERY = 3                 # normaliser sample: every 3rd training date


def build_stock(panel) -> list[str]:
    from marketcast.data.build_panel import read_equity
    from marketcast.data.derivatives import load_stock_features
    from marketcast.data.sector_map import load_sector_map
    from marketcast.data.wide import load_wide, pivot

    keys, dates = panel.keys, panel.dates
    t, k = len(dates), len(keys)
    w = load_wide(["high", "low", "close", "volume", "turnover", "trades", "deliv_pct", "corp_action_flag", "days_listed"], keys=keys)
    if not (w.dates.equals(dates) and (w.keys == keys).all()):
        raise RuntimeError("equity panel and model panel are out of sync; re-run `targets build`")
    e = read_equity(["date", "key", "series"], shares_only=True)
    e = e[e["key"].isin(set(keys))]
    surv = np.zeros((t, k), dtype=np.float32)
    surv[dates.get_indexer(e["date"]), pd.Index(keys).get_indexer(e["key"])] = e["series"].isin(["BE", "BZ"]).to_numpy(np.float32)
    del e
    fo_long = load_stock_features()
    fo_long = fo_long[fo_long["key"].isin(set(keys))]
    fo = pivot(fo_long, ["fut_oi", "basis_ann", "pcr_oi", "atm_iv"], dates=dates, keys=keys).a
    sm = load_sector_map().set_index("key")["sector"].reindex(keys).fillna("Unclassified")
    sector_code = pd.Categorical(sm).codes
    heldout = panel.heldout
    reference = np.asarray(panel["member1000"]) & ~heldout[None, :]
    mkt = panel.market()

    FEATURES.mkdir(parents=True, exist_ok=True)
    tmp = FEATURES / f"stock.npy.tmp{os.getpid()}"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(t, k, len(ST.CHANNELS)))
    ell, rv2, v, sigma, z = (np.asarray(panel[n], dtype=np.float32) for n in ("ell", "rv2", "v", "sigma", "z"))
    gens = [ST.price_channels(ell, rv2, v, sigma, z, np.asarray(panel["o"], dtype=np.float32), w["high"], w["low"], w["close"]),
            ST.held_channels(ell=ell, rv2=rv2, sigma=sigma, z=z, turnover=w["turnover"], volume=w["volume"], trades=w["trades"],
                             deliv_pct=w["deliv_pct"], stale=np.asarray(panel["stale"]), series_surv=surv,
                             circuit_carry=np.asarray(panel["circuit_carry"]), days_listed=np.nan_to_num(w["days_listed"], nan=0.0),
                             has_fo=np.asarray(panel["has_fo"]), fo=fo, mkt_ell=mkt["mkt_ell"].to_numpy(dtype=np.float64),
                             sector_code=sector_code, reference=reference, corp_action=np.nan_to_num(w["corp_action_flag"], nan=0.0) > 0,
                             med_turn60=np.asarray(panel["med_turn60"], dtype=np.float32), traded=np.isfinite(w["close"]))]
    j = 0
    for gen in gens:
        for name, arr in gen:
            if name != ST.CHANNELS[j]:
                raise RuntimeError(f"channel order mismatch: got {name}, expected {ST.CHANNELS[j]}")
            # float16 holds +-65504; anything larger is a data artefact and is stored as NaN (-> 0 after normalisation)
            a = np.where(np.abs(arr) < 6.0e4, arr, np.nan)
            out[:, :, j] = a.astype(np.float16)
            j += 1
            log.info("  channel %2d/%d %-18s finite %.1f%%", j, len(ST.CHANNELS), name, 100 * np.isfinite(a).mean())
    if j != len(ST.CHANNELS):
        raise RuntimeError(f"wrote {j} channels, expected {len(ST.CHANNELS)}")
    out.flush()
    del out
    commit(tmp, FEATURES / "stock.npy")
    return ST.CHANNELS


def build_global_and_forcings(panel) -> tuple[list[list[str]], np.ndarray]:
    from marketcast.data.calendar import extend_calendar, load_calendar
    from marketcast.data.global_align import load_levels

    g, names = GN.build(load_levels(), panel.market())
    save_npy(g, FEATURES / "global.npy")
    cal = extend_calendar(load_calendar(), N_FUTURE)
    f = FO.build(cal, (load_yaml("calendar_events.yaml").get("rule_changes") or {}))
    save_npy(f, FEATURES / "forcings.npy")
    return names, g


def fit_fold_normalizers(panel, fold: str, global_arr: np.ndarray, global_names: list[list[str]]) -> dict:
    """Fit and store data/features/<fold>/norm.json. Reads only training-split rows of non-held-out members."""
    from marketcast.eval.folds import split_rows

    split = split_rows(fold, panel.dates) if fold != "Vault" else split_rows(fold, panel.dates[panel.dates < "2026-01-01"])
    rows = split.train[::FIT_EVERY]
    feats = np.load(FEATURES / "stock.npy", mmap_mode="r")
    ref = np.asarray(panel["member1000"]) & ~panel.heldout[None, :]
    m_price = ST.CHANNELS.index("m_price")
    chunks = []
    for r in rows:
        x = np.asarray(feats[r, ref[r]], dtype=np.float32)
        chunks.append(x[x[:, m_price] > 0])
    sample = np.concatenate(chunks) if chunks else np.zeros((0, len(ST.CHANNELS)), dtype=np.float32)
    stock_norm = fit(sample, ST.CHANNELS, ST.PASSTHROUGH)
    gflat = global_arr[split.train].reshape(len(split.train), -1)
    gnames = [f"{GN.NODES[i]}.{c}" for i, cs in enumerate(global_names) for c in cs]
    gpass = {n for n in gnames if n.endswith(".has_data") or ".pad" in n}
    glob_norm = fit(gflat, gnames, gpass)
    out = {"fold": fold, "n_train_dates": int(len(split.train)), "n_cells": int(len(sample)),
           "train_first": str(panel.dates[split.train[0]].date()), "train_last": str(panel.dates[split.train[-1]].date()),
           "stock": stock_norm.to_dict(), "global": glob_norm.to_dict()}
    write_json(out, FEATURES / fold / "norm.json")
    # acceptance (06 P3.1): normalised training channels have median ~ 0 and IQR ~ 1.35
    y = (sample - stock_norm.median) / stock_norm.scale
    idx = [i for i, n in enumerate(ST.CHANNELS) if n not in ST.PASSTHROUGH]
    with np.errstate(all="ignore"):
        q25, q50, q75 = np.nanpercentile(y[:, idx], [25, 50, 75], axis=0)
    log.info("  %s: %d train dates, %d cells; |median| max %.3f; IQR range %.2f-%.2f (target 1.35)", fold, len(split.train),
             len(sample), float(np.nanmax(np.abs(q50))), float(np.nanmin(q75 - q25)), float(np.nanmax(q75 - q25)))
    return out


def load_normalizers(fold: str) -> tuple[Normalizer, Normalizer]:
    p = FEATURES / fold / "norm.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m marketcast features build`")
    d = json.loads(p.read_text(encoding="utf-8"))
    return Normalizer.from_dict(d["stock"]), Normalizer.from_dict(d["global"])


def main(folds: list[str] | None = None, skip_stock: bool = False) -> int:
    from marketcast.eval.folds import load_folds
    from marketcast.targets.build import ModelPanel

    panel = ModelPanel()
    if not skip_stock or not (FEATURES / "stock.npy").exists():
        log.info("stock channels: %d sessions x %d companies x %d channels", len(panel.dates), len(panel.keys), len(ST.CHANNELS))
        build_stock(panel)
    names, g = build_global_and_forcings(panel)
    write_json({"dates": [str(d.date()) for d in panel.dates], "keys": panel.keys.tolist(), "stock_channels": ST.CHANNELS,
                "price_channels": ST.PRICE_CHANNELS, "global_nodes": GN.NODES, "global_channels": names,
                "forcing_names": FO.FORCING_NAMES, "n_future_forcings": N_FUTURE,
                "built_at": pd.Timestamp.now().isoformat(timespec="seconds")}, FEATURES / "meta.json")
    for fold in folds or list(load_folds()["folds"]):
        fit_fold_normalizers(panel, fold, g, names)
    return 0
