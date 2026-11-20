"""P1.3 + P1.9 + P2.1  Returns, masks, targets and the market series: the "model panel".

This is the hand-over point between the data foundation and everything model-side. It writes

  data/processed/equity_returns/part.parquet   long (date, key, ell, o, rv2, v) for EVERY company share (P1.3)
  data/processed/masks/part.parquet            long masks for every company share (P1.9)
  data/targets/                                the MODEL PANEL: wide float32 .npy arrays [T, K] restricted to
      meta.json                                companies ever in MC1000 (+ named held-out), plus
      ell, v, rv2, sigma, z, o                 03 §1-§3 quantities (z[t] = ell[t] / sigma[t-1])
      med_turn60                               60-session median traded value (significance + loss weights)
      target_mask, censored, circuit_carry, stale, jump_unexplained, has_fo
      member500, member1000                    PIT universe membership
      heldout (K,)                             1 = company is held out (00 rule 7)
      market.parquet                           per-session market series: MC500 aggregate (ex-held-out),
                                               official Nifty 50 (l, RV2, v, sigma) and India VIX
  reports/P1_corp_action_check.md, reports/P2_targets_check.md

FORWARD WINDOWS ARE NEVER STORED. Row t holds what happened ON session t. The loader pairs decision date t
with target row t+1 (and t+1..t+h for multi-day scores), so the fold logic alone owns purging (06 P2.1).

HELD-OUT ISOLATION: the sigma floor's cross-sectional median and the market aggregate are computed over
non-held-out companies only, so no held-out price ever enters a statistic that training stocks see.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marketcast import DATA, PROCESSED, REPORTS
from marketcast.data import masks as M
from marketcast.data.wide import Wide, load_wide, roll
from marketcast.targets import returns as R
from marketcast.util import get_logger, md_table, save_npy, write_json, write_parquet, write_text

log = get_logger("targets.build")
TARGETS = DATA / "targets"
WIDE_COLS = ["open", "high", "low", "close", "pc", "volume", "turnover", "adj_factor", "corp_action_flag", "jump_unexplained"]
PANEL_ARRAYS = ["ell", "v", "rv2", "sigma", "s2", "z", "o", "med_turn60", "target_mask", "censored", "circuit_carry", "stale",
                "jump_unexplained", "has_fo", "member500", "member1000"]


def nifty_series(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Official Nifty 50 and India VIX aligned to sessions (NaN before the index files start, 2012-07)."""
    from marketcast.data.calendar import canonical_indices

    ix = canonical_indices(["open", "high", "low", "close", "chg"])
    n = ix[ix["index"] == "Nifty 50"].set_index("date").reindex(sessions)
    pc = n["close"] - n["chg"]                                  # the index's own previous close
    pc = pc.where(pc > 0, n["close"].shift(1))
    out = pd.DataFrame(index=sessions)
    out["n50_close"] = n["close"]
    out["n50_ell"] = R.log_return(n["close"].to_numpy(float), pc.to_numpy(float))
    out["n50_rv2"] = R.rv2_gk(n["open"].to_numpy(float), n["high"].to_numpy(float), n["low"].to_numpy(float),
                              n["close"].to_numpy(float), pc.to_numpy(float))
    out["vix"] = ix[ix["index"] == "India VIX"].set_index("date")["close"].reindex(sessions)
    return out


def build() -> dict:
    from marketcast.data.calendar import load_calendar
    from marketcast.data.derivatives import load_stock_features
    from marketcast.data.universe import ever_members, load_universe, membership
    from marketcast.eval.folds import load_heldout

    cal = load_calendar()
    w = load_wide(WIDE_COLS, shares_only=True)
    if not (w.dates == pd.DatetimeIndex(cal["date"])).all():
        raise RuntimeError("trading calendar and equity panel disagree on sessions; rebuild the calendar")
    heldout_keys, named_keys = load_heldout()
    is_ho_all = np.isin(w.keys, list(heldout_keys))
    close, pc = w["close"].astype(np.float64), w["pc"].astype(np.float64)
    open_, high, low = (w[c].astype(np.float64) for c in ("open", "high", "low"))

    # ---- 03 §1-§3
    ell = R.log_return(close, pc)
    rv2 = R.rv2_gk(open_, high, low, close, pc)
    v = R.log_rv(rv2)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.log(open_ / pc)
    o[~np.isfinite(o)] = np.nan
    s2, n_obs = R.ewma_var(ell)
    sigma, floor = R.sigma_hat(s2, n_obs, floor_members=np.broadcast_to(~is_ho_all, s2.shape))
    z = R.scaled_return(ell, sigma)

    # ---- masks (03 §5)
    fo = load_stock_features()[["date", "key"]]
    has_fo = np.zeros(w.shape, dtype=bool)
    ti, ki = w.dates.get_indexer(fo["date"]), pd.Index(w.keys).get_indexer(fo["key"])
    ok = (ti >= 0) & (ki >= 0)
    has_fo[ti[ok], ki[ok]] = True
    up, dn = M.circuit_hits(close, high, low, pc, has_fo)
    halt = cal["is_market_halt"].to_numpy(bool)
    censored = up | dn | (halt[:, None] & np.isfinite(close))
    carry = np.zeros(w.shape, dtype=np.float32)                 # +1 / -1 enters the NEXT day's inputs (03 §5)
    carry[1:] = up[:-1].astype(np.float32) - dn[:-1].astype(np.float32)
    stale = M.stale_mask(w["volume"], open_, high, low, close, pc)
    session_bad = cal["session_quality"].isin(["special", "partial", "outage"]).to_numpy()
    jump = np.nan_to_num(w["jump_unexplained"], nan=0.0) > 0
    traded = np.isfinite(close)
    terminal = M.terminal_rows(traded)
    tmask = M.target_mask(z, v, censored, stale, session_bad, jump)
    med_turn60 = roll(w["turnover"], 60, "median", min_periods=20)

    # ---- long outputs for all companies (plan outputs of P1.3 and P1.9)
    full = Wide(w.dates, w.keys, {"ell": ell.astype(np.float32), "o": o.astype(np.float32), "rv2": rv2.astype(np.float32), "v": v.astype(np.float32)})
    write_parquet(full.to_long(), PROCESSED / "equity_returns" / "part.parquet")
    mk = Wide(w.dates, w.keys, {"close": w["close"]})
    long_m = mk.to_long()
    ti, ki = w.dates.get_indexer(long_m["date"]), pd.Index(w.keys).get_indexer(long_m["key"])
    for name, arr in [("censored", censored), ("upper_circuit", up), ("lower_circuit", dn), ("stale", stale), ("jump_unexplained", jump),
                      ("history_short", ~(n_obs >= R.EWMA_VALID_N)), ("terminal_unknown", terminal), ("has_fo", has_fo), ("target_mask", tmask)]:
        long_m[name] = arr[ti, ki]
    long_m["session_bad"] = session_bad[ti]
    long_m["merger_window"] = False                 # gap G15: no official corporate-action table yet
    long_m["dividend_exdate_unadjusted"] = False    # gap G2: no dividend history yet
    write_parquet(long_m.drop(columns="close"), PROCESSED / "masks" / "part.parquet")

    # ---- model panel: restrict columns to companies ever in MC1000 (+ any named held-out outside it)
    uni = load_universe()
    keep_keys = np.union1d(ever_members("MC1000"), np.array(sorted(named_keys & set(w.keys)), dtype=str))
    keep = np.isin(w.keys, keep_keys)
    keys = w.keys[keep]
    m500 = membership(w.dates, keys, "MC500", uni)
    m1000 = membership(w.dates, keys, "MC1000", uni)
    is_ho = np.isin(keys, list(heldout_keys))
    TARGETS.mkdir(parents=True, exist_ok=True)
    arrays = {"ell": ell, "v": v, "rv2": rv2, "sigma": sigma, "s2": s2, "z": z, "o": o, "med_turn60": med_turn60, "target_mask": tmask,
              "censored": censored, "circuit_carry": carry, "stale": stale, "jump_unexplained": jump, "has_fo": has_fo}
    for name, arr in arrays.items():
        a = arr[:, keep]
        save_npy(a.astype(np.float32) if a.dtype.kind == "f" else a, TARGETS / f"{name}.npy")
    save_npy(m500, TARGETS / "member500.npy")
    save_npy(m1000, TARGETS / "member1000.npy")
    save_npy(is_ho, TARGETS / "heldout.npy")
    write_json({"dates": [str(d.date()) for d in w.dates], "keys": keys.tolist(), "arrays": PANEL_ARRAYS + ["heldout"],
                "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "note": "row t = what happened on session t; forward windows are assembled by the loader"}, TARGETS / "meta.json")

    # ---- market series: traded-value-weighted MC500 aggregate, held-out companies excluded
    wgt = np.nan_to_num(np.vstack([np.full((1, keep.sum()), np.nan), med_turn60[:-1, keep]]), nan=0.0)   # weights known at t-1
    el_k = ell[:, keep]
    use = m500 & ~is_ho[None, :] & np.isfinite(el_k) & (wgt > 0)
    wsum = np.where(use, wgt, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        simple = np.where(use, wgt * np.expm1(el_k), 0.0).sum(axis=1) / wsum       # aggregate SIMPLE returns, then log
        mkt_ell = np.where(wsum > 0, np.log1p(simple), np.nan)
    mkt = nifty_series(w.dates)
    mkt["mkt_ell"] = mkt_ell
    mkt["mkt_n"] = use.sum(axis=1)
    mkt["mkt_breadth"] = np.where(use.sum(axis=1) > 0, (use & (el_k > 0)).sum(axis=1) / np.maximum(use.sum(axis=1), 1), np.nan)
    mkt["mkt_dispersion"] = np.nanstd(np.where(use, z[:, keep], np.nan), axis=1)
    # market "price" target = Nifty 50 when official data exists, else the aggregate (03 §2 rv_method=close_only)
    mkt["m_ell"] = mkt["n50_ell"].where(mkt["n50_ell"].notna(), mkt["mkt_ell"])
    mkt["m_rv2"] = mkt["n50_rv2"].where(mkt["n50_rv2"].notna(), mkt["mkt_ell"] ** 2)
    mkt["rv_method"] = np.where(mkt["n50_rv2"].notna(), "ohlc", "close_only")
    ms2, mn = R.ewma_var(mkt["m_ell"].to_numpy())
    msig = np.sqrt(ms2)
    msig[mn < R.EWMA_VALID_N] = np.nan
    mkt["m_sigma"] = np.maximum(msig, R.SIGMA_ABS_FLOOR)
    mkt["m_z"] = R.scaled_return(mkt["m_ell"].to_numpy(), mkt["m_sigma"].to_numpy())
    mkt["m_v"] = R.log_rv(mkt["m_rv2"].to_numpy())
    mkt["session_bad"] = session_bad
    mkt["sigma_floor"] = floor
    write_parquet(mkt.rename_axis("date").reset_index(), TARGETS / "market.parquet")

    stats = _reports(w, ell, z, tmask, censored, stale, jump, terminal, m500, keep, is_ho, mkt)
    log.info("model panel: %d sessions x %d companies (%d held-out); %s", len(w.dates), keep.sum(), int(is_ho.sum()), stats)
    return stats


def _reports(w, ell, z, tmask, censored, stale, jump, terminal, m500, keep, is_ho, mkt) -> dict:
    yrs = w.dates.year
    zk, mk = z[:, keep], tmask[:, keep]
    rows = []
    for y in np.unique(yrs):
        r = yrs == y
        sel = m500[r] & mk[r]
        zz = zk[r][sel]
        memb = m500[r]
        rows.append({"year": int(y), "z_mean": float(np.mean(zz)) if zz.size else np.nan, "z_std": float(np.std(zz)) if zz.size else np.nan,
                     "z_kurt": float(pd.Series(zz).kurt()) if zz.size > 10 else np.nan,
                     "target_share_of_member_days": float(sel.sum() / max(memb.sum(), 1)),
                     "censored_pct": float(100 * (censored[r][:, keep] & memb).sum() / max(memb.sum(), 1)),
                     "stale_pct": float(100 * (stale[r][:, keep] & memb).sum() / max(memb.sum(), 1))})
    zt = pd.DataFrame(rows)
    ok_years = zt[(zt["year"] >= 2008)]
    in_band = ((ok_years["z_std"] >= 0.9) & (ok_years["z_std"] <= 1.2)).mean()
    write_text("\n".join([
        "# P2.1 Targets check", "", f"Generated {pd.Timestamp.now().isoformat(timespec='seconds')} by `python -m marketcast targets build`.", "",
        "Acceptance (06 P2.1): z has mean ~ 0 and std 0.9-1.2 per year on MC500; no target uses masked days.", "",
        f"- years (2008+) with z std inside 0.9-1.2: {in_band:.0%}", f"- model-panel companies: {int(keep.sum())}, held-out: {int(is_ho.sum())}",
        f"- sessions with an official Nifty 50 row: {int(mkt['n50_ell'].notna().sum())} of {len(mkt)}", "",
        md_table(zt), ""]), REPORTS / "P2_targets_check.md")
    big = np.abs(ell) > 0.30
    n_ret = int(np.isfinite(ell).sum())
    ca = np.nan_to_num(w["corp_action_flag"], nan=0.0) > 0
    L = ["# P1.3 Corporate-action and return check", "", f"Generated {pd.Timestamp.now().isoformat(timespec='seconds')}.", "",
         "Returns are close / (own previous close x inferred factor) - 1 (decision D-22). Price returns only: dividends are missing (gap G2).", "",
         f"- stock-days with a return: {n_ret:,}", f"- inferred split/bonus/consolidation events: {int(ca.sum()):,}",
         f"- unexplained jumps (> 40%, no standard ratio; masked from targets): {int(jump.sum()):,}",
         f"- |log return| > 30%: {int(big.sum()):,} ({100 * big.sum() / n_ret:.4f}% of stock-days); "
         f"after excluding flagged rows: {int((big & ~jump).sum()):,} ({100 * (big & ~jump).sum() / n_ret:.4f}%; acceptance < 0.05%)",
         f"- exits (last traded row before the panel end): {int(terminal.sum()):,}; terminal payoff unknown for all of them (no buyout/merger table), "
         "so no target is built after the last traded day", "",
         "## Not done here (needs a human or an official table, gap G15)", "",
         "- The 20 random stock-day comparison against exchange records (gate G1 criterion 1).",
         "- Hand verification of every inferred event among the 30 largest stocks.",
         "- Classification of unexplained jumps into demerger / rights / relisting / genuine move.", ""]
    write_text("\n".join(L), REPORTS / "P1_corp_action_check.md")
    return {"z_std_in_band": float(in_band), "abs_ell_gt_30pct_unflagged_pct": float(100 * (big & ~jump).sum() / n_ret)}


# --------------------------------------------------------------------------- loader used by everything downstream
class ModelPanel:
    """Memory-mapped view of data/targets. Arrays are opened lazily and read-only."""

    def __init__(self, root=None):
        self.root = TARGETS if root is None else root
        meta_p = self.root / "meta.json"
        if not meta_p.exists():
            raise FileNotFoundError(f"{meta_p} missing; run `python -m marketcast targets build`")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        self.dates = pd.DatetimeIndex(meta["dates"])
        self.keys = np.array(meta["keys"], dtype=str)
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self._cache[name]

    @property
    def heldout(self) -> np.ndarray:
        return np.asarray(self["heldout"], dtype=bool)

    def market(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "market.parquet").set_index("date")


def main() -> int:
    build()
    return 0
