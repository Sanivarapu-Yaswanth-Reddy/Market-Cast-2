#!/usr/bin/env python3
"""
MarketCast panel builder
========================

Turns the raw files from marketcast_download.py into training-ready tables:

  data/processed/equity_daily.parquet   one row per (date, ISIN): OHLC, volume, turnover,
                                        trades, delivery qty/%, corporate-action adjustment
                                        factor and adjusted prices
  data/processed/indices_daily.parquet  ~145 NSE indices incl. India VIX: OHLC, P/E, P/B, DY
  data/processed/fo_daily.parquet       per (date, underlying): futures OI/volume/settle,
                                        options call/put OI, put-call OI ratio
  data/processed/fred_global.parquet    FRED series aligned on one date index (raw, no fill)

Needs pandas (Anaconda has it). Parquet needs pyarrow; without it CSV.gz is written.

Point-in-time notes
  * Bhavcopy rows are published the same evening, so every column is knowable at
    the close of `date`. Delivery (MTO) is also same-day.
  * The adjustment factor uses NSE's own PREVCLOSE, which the exchange adjusts on the
    ex-date of splits, bonuses and rights:  f_t = PREVCLOSE_t / CLOSE_{t-1}.
    A factor far from 1 on an ex-date is a corporate action, not a crash. Adjusted
    prices are computed backward from the latest date (standard back-adjustment);
    for strict point-in-time training, use raw prices plus f_t, which never looks ahead.
  * Delisted companies stay in the panel up to their last trading day (no survivorship
    filter is applied here). Join data/raw/nse/reference/*/delisted.csv for exit reasons.

Usage:  python scripts/marketcast_build_panel.py [--series EQ,BE,BZ] [--only equity,indices,fo,fred]
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pandas is required: pip install pandas pyarrow")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CM_MAP = {  # old bhavcopy / UDiFF -> unified
    "SYMBOL": "symbol", "TckrSymb": "symbol", "SERIES": "series", "SctySrs": "series",
    "ISIN": "isin", "OPEN": "open", "OpnPric": "open", "HIGH": "high", "HghPric": "high",
    "LOW": "low", "LwPric": "low", "CLOSE": "close", "ClsPric": "close", "LAST": "last",
    "LastPric": "last", "PREVCLOSE": "prev_close", "PrvsClsgPric": "prev_close",
    "TOTTRDQTY": "volume", "TtlTradgVol": "volume", "TOTTRDVAL": "turnover",
    "TtlTrfVal": "turnover", "TOTALTRADES": "trades", "TtlNbOfTxsExctd": "trades",
    "TIMESTAMP": "date", "TradDt": "date",
}
FO_MAP = {
    "INSTRUMENT": "instr", "FinInstrmTp": "instr", "SYMBOL": "underlying", "TckrSymb": "underlying",
    "EXPIRY_DT": "expiry", "XpryDt": "expiry", "STRIKE_PR": "strike", "StrkPric": "strike",
    "OPTION_TYP": "opt_type", "OptnTp": "opt_type", "CLOSE": "close", "ClsPric": "close",
    "SETTLE_PR": "settle", "SttlmPric": "settle", "CONTRACTS": "contracts",
    "TtlTradgVol": "volume_units", "OPEN_INT": "oi", "OpnIntrst": "oi",
    "CHG_IN_OI": "chg_oi", "ChngInOpnIntrst": "chg_oi", "TIMESTAMP": "date", "TradDt": "date",
    "UndrlygPric": "underlying_px",
}
INSTR = {"FUTSTK": "STF", "FUTIDX": "IDF", "OPTSTK": "STO", "OPTIDX": "IDO"}


def save(df: "pd.DataFrame", name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"   wrote {name}.parquet  rows={len(df):,}")
    except Exception:
        df.to_csv(OUT / f"{name}.csv.gz", index=False)
        print(f"   wrote {name}.csv.gz  rows={len(df):,} (install pyarrow for parquet)")


def read_zip_csv(p: Path) -> "pd.DataFrame":
    with zipfile.ZipFile(p) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def file_date(p: Path) -> "pd.Timestamp":
    m = re.search(r"(\d{8})", p.name)
    return pd.Timestamp(m.group(1))


# ------------------------------------------------------------------ equity
def read_mto(p: Path) -> "pd.DataFrame":
    rows = []
    for line in p.read_text(errors="ignore").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if parts and parts[0] == "20" and len(parts) >= 7:
            rows.append(parts[2:7])
    df = pd.DataFrame(rows, columns=["symbol", "series", "mto_qty", "deliv_qty", "deliv_pct"])
    df["date"] = file_date(p)
    return df


def build_equity(series: set[str]) -> None:
    files = sorted((RAW / "nse" / "cm_bhavcopy").rglob("*.zip"))
    print(f"\n== equity panel from {len(files)} bhavcopy files ==")
    mto_by_date = {file_date(m): m for m in (RAW / "nse" / "delivery_mto").rglob("*.DAT")}
    print(f"   delivery files indexed: {len(mto_by_date)} (joined one day at a time)")
    frames = []
    for i, p in enumerate(files):
        try:
            df = read_zip_csv(p).rename(columns=CM_MAP)
        except Exception as e:
            print(f"   skip {p.name}: {e}")
            continue
        df = df[[c for c in set(CM_MAP.values()) if c in df.columns]]
        df["series"] = df["series"].str.strip()
        df = df[df["series"].isin(series)]
        df["date"] = file_date(p)
        m = mto_by_date.get(df["date"].iloc[0]) if len(df) else None
        if m is not None:
            try:
                dm = read_mto(m).drop_duplicates(["symbol", "series"])
                dm["symbol"] = dm["symbol"].astype(str)
                df["symbol"] = df["symbol"].astype(str).str.strip()
                df = df.merge(dm.drop(columns=["date"]), on=["symbol", "series"], how="left")
            except Exception as e:
                print(f"   delivery skipped for {p.name}: {e}")
        frames.append(df)
        if (i + 1) % 500 == 0:
            print(f"   parsed {i+1}/{len(files)}")
    if not frames:
        print("   no bhavcopy files found; run the downloader first")
        return
    eq = pd.concat(frames, ignore_index=True)
    del frames
    num = ["open", "high", "low", "close", "last", "prev_close", "volume", "turnover", "trades"]
    for c in num:
        if c in eq.columns:
            eq[c] = pd.to_numeric(eq[c], errors="coerce")
    eq["symbol"] = eq["symbol"].str.strip()

    # ISIN backfill: pre-2011 bhavcopies lack ISIN; map symbol->ISIN using the earliest
    # later observation of that symbol (ticker changes remain a known gap).
    if "isin" not in eq.columns:
        eq["isin"] = np.nan
    eq["isin"] = eq["isin"].str.strip()
    first_isin = (eq.dropna(subset=["isin"]).sort_values("date")
                  .drop_duplicates("symbol")[["symbol", "isin"]].set_index("symbol")["isin"])
    eq["isin"] = eq["isin"].fillna(eq["symbol"].map(first_isin))
    eq["key"] = eq["isin"].fillna("SYM:" + eq["symbol"])

    for c in ["mto_qty", "deliv_qty", "deliv_pct"]:
        if c in eq.columns:
            eq[c] = pd.to_numeric(eq[c], errors="coerce")

    # company_id = ISIN[:9] (issuer + security type): survives face-value-split ISIN serial changes (decision D-20)
    # Only company shares (ISIN prefix INE) are joined on ISIN[:9]. ETFs and fund units (INF...) share their first
    # 9 characters across many different schemes of one fund house, so they keep the full ISIN.
    # Within an INE[:9] group, ISINs are joined only if they REPLACE each other in time (face-value split: old ISIN
    # stops, new starts). ISINs that trade side by side (e.g. ordinary + DVR share) stay separate.
    is_ine = eq["isin"].str.startswith("INE").fillna(False)
    span = eq.loc[is_ine].groupby("isin")["date"].agg(["min", "max"]).reset_index()
    span["pref"] = span["isin"].str[:9]
    span = span.sort_values(["pref", "min"])
    span["prev_max"] = span.groupby("pref")["max"].shift(1)
    span["overlap"] = (span["prev_max"] - span["min"]).dt.days > 7
    coexist = set(span.loc[span.groupby("pref")["overlap"].transform("any"), "pref"])
    join = is_ine & ~eq["isin"].str[:9].isin(coexist).fillna(False)
    eq["company_id"] = eq["isin"].where(~join, eq["isin"].str[:9]).fillna("SYM:" + eq["symbol"])
    print(f"   ISIN prefixes kept separate (share classes trading side by side): {len(coexist)}")
    eq["is_fund"] = eq["isin"].str.startswith("INF").fillna(False)
    eq["key"] = eq["company_id"]
    # one row per (date, company): if an old and a new ISIN both trade on a date, keep the more liquid row
    eq = eq.sort_values(["key", "date", "turnover"], ignore_index=True)
    n0 = len(eq)
    eq = eq.drop_duplicates(["key", "date"], keep="last", ignore_index=True)
    print(f"   same-company same-day duplicates dropped: {n0 - len(eq):,}")
    eq["gap_days"] = eq.groupby("key")["date"].diff().dt.days      # >10 can mean a missing download or a long suspension

    # Splits / bonuses / consolidations. NSE's PREVCLOSE is NOT adjusted for them (verified: RELIANCE 2017-09-07,
    # HDFCBANK 2019-09-19, GARFIBRES 2025-01-03), so they are inferred from price jumps that match a standard ratio.
    # adj_factor_t = R means: prices before t are multiplied by R (R = 1/k for a k-for-1 split; R > 1 for a consolidation).
    STD = np.array([1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/10, 1/20, 2/3, 3/4, 4/5, 5/6, 2.0, 3.0, 5.0, 10.0])
    # NSE's PREVCLOSE is unreliable on first-trading days after listings and some restructurings (rows seen with
    # prev_close 1.40 / 5 / 37.10 against closes of 540 / 616 / 7,574), so the reference is this company's own previous close.
    own_prev = eq.groupby("key")["close"].shift(1)
    ref = own_prev.where((own_prev > 0) & (eq["gap_days"].fillna(99) <= 10))
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.log(eq["open"].where(eq["open"] > 0) / ref)
        lc = np.log(eq["close"].where(eq["close"] > 0) / ref)
    big = (lo.abs() > np.log(1.22)) & (ref >= 5)   # penny-tick noise excluded; ref is NaN across gaps
    lstd = np.log(STD)
    lo_a, lc_a = lo.to_numpy(), lc.to_numpy()
    cand = np.flatnonzero(big.to_numpy())                 # only a few thousand rows, keeps memory small
    ok = np.zeros(len(eq), dtype=bool)
    fac = np.ones(len(eq))
    if len(cand):
        d = np.abs(lo_a[cand][:, None] - lstd[None, :])
        jj = np.argmin(d, axis=1)
        nr = lstd[jj]
        good = (np.abs(lo_a[cand] - nr) < 0.06) & (np.abs(lc_a[cand] - nr) < 0.16)
        ok[cand[good]] = True
        fac[cand[good]] = STD[jj[good]]
    eq["adj_factor"] = fac
    eq["corp_action_flag"] = ok
    # large moves that match no standard ratio (demergers, rights, genuine crashes, relistings): flagged, not adjusted
    eq["jump_unexplained"] = (~ok) & (np.abs(lc_a) > np.log(1.4))
    # back-adjust: multiply history by product of all later factors
    later = (eq.groupby("key")["adj_factor"]
               .transform(lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1.0)))
    for c in ["open", "high", "low", "close", "prev_close"]:
        eq[f"adj_{c}"] = eq[c] * later
    # return across an inferred event uses the event ratio; otherwise the plain return
    # NaN on a company's first row and across gaps > 10 days; raw NSE prev_close stays in the table for reference
    eq["ret_1d"] = eq["close"] / (ref * eq["adj_factor"]) - 1
    eq["days_listed"] = eq.groupby("key").cumcount()
    save(eq, "equity_daily")
    flags = eq.loc[eq["corp_action_flag"] | eq["jump_unexplained"],
                   ["date", "symbol", "isin", "company_id", "prev_close", "open", "close", "adj_factor",
                    "corp_action_flag", "jump_unexplained", "gap_days"]]
    save(flags, "corporate_action_events_inferred")


# ------------------------------------------------------------------ indices
def build_indices() -> None:
    files = sorted((RAW / "nse" / "indices_close_all").rglob("*.csv"))
    print(f"\n== indices panel from {len(files)} files ==")
    if not files:
        return
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p, dtype=str)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    ix = pd.concat(frames, ignore_index=True).rename(columns={
        "Index Name": "index", "Index Date": "date", "Open Index Value": "open",
        "High Index Value": "high", "Low Index Value": "low", "Closing Index Value": "close",
        "Points Change": "chg", "Change(%)": "chg_pct", "Volume": "volume",
        "Turnover (Rs. Cr.)": "turnover_cr", "P/E": "pe", "P/B": "pb", "Div Yield": "div_yield"})
    ix["date"] = pd.to_datetime(ix["date"], format="%d-%m-%Y", errors="coerce")
    for c in ["open", "high", "low", "close", "chg", "chg_pct", "volume", "turnover_cr", "pe", "pb", "div_yield"]:
        ix[c] = pd.to_numeric(ix[c].replace("-", np.nan), errors="coerce")
    # old names (S&P CNX Nifty, CNX Nifty Junior...) are kept as-is; map them in features
    save(ix.sort_values(["index", "date"]), "indices_daily")


# ------------------------------------------------------------------ F&O
def build_fo() -> None:
    files = sorted((RAW / "nse" / "fo_bhavcopy").rglob("*.zip"))
    print(f"\n== F&O features from {len(files)} files ==")
    out = []
    bad = 0
    for i, p in enumerate(files):
        try:
            agg = _fo_one(p)
        except Exception as e:
            bad += 1
            if bad <= 20:
                print(f"   skip {p.name}: {type(e).__name__}: {e}")
            continue
        if agg is not None:
            out.append(agg)
        if (i + 1) % 500 == 0:
            print(f"   parsed {i+1}/{len(files)}")
    print(f"   F&O files parsed {len(out)}, skipped {bad}")
    if out:
        save(pd.concat(out, ignore_index=True), "fo_daily")


def _fo_one(p: Path):
    if True:
        df = read_zip_csv(p).rename(columns=FO_MAP)
        df["instr"] = df["instr"].str.strip().replace(INSTR)
        for c in ["close", "settle", "oi", "chg_oi", "contracts", "volume_units", "strike", "underlying_px"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["underlying"] = df["underlying"].str.strip()
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce", format="mixed")
        d = file_date(p)
        fut = df[df["instr"].isin(["STF", "IDF"])].sort_values("expiry")
        near = fut.groupby("underlying").head(1).set_index("underlying")
        f_agg = fut.groupby("underlying").agg(fut_oi=("oi", "sum"), fut_chg_oi=("chg_oi", "sum"))
        f_agg["fut_near_settle"] = near["settle"]
        f_agg["fut_near_expiry"] = near["expiry"]
        opt = df[df["instr"].isin(["STO", "IDO"])]
        o_agg = opt.pivot_table(index="underlying", columns="opt_type", values="oi", aggfunc="sum")
        o_agg = o_agg.rename(columns={"CE": "call_oi", "PE": "put_oi"})
        agg = f_agg.join(o_agg[[c for c in ["call_oi", "put_oi"] if c in o_agg]], how="outer")
        if "call_oi" in agg and "put_oi" in agg:
            agg["pcr_oi"] = agg["put_oi"] / agg["call_oi"].replace(0, np.nan)
        agg["date"] = d
        return agg.reset_index()


# ------------------------------------------------------------------ FRED
def build_fred() -> None:
    files = [p for p in (RAW / "global" / "fred").glob("*.csv") if not p.name.startswith("_")]
    print(f"\n== FRED: {len(files)} series ==")
    series = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df.columns = ["date", p.stem]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[p.stem] = pd.to_numeric(df[p.stem], errors="coerce")
            series.append(df.set_index("date")[p.stem])
        except Exception as e:
            print(f"   skip {p.name}: {e}")
    if series:
        save(pd.concat(series, axis=1).sort_index().reset_index(), "fred_global")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="EQ,BE,BZ", help="equity series to keep")
    ap.add_argument("--only", default="equity,indices,fo,fred")
    ap.add_argument("--force", action="store_true", help="rebuild FRED and indices even if their output exists")
    a = ap.parse_args()
    todo = set(a.only.split(","))
    if "fred" in todo and (a.force or not (OUT / "fred_global.parquet").exists()):
        build_fred()
    if "indices" in todo and (a.force or not (OUT / "indices_daily.parquet").exists()):
        build_indices()
    if "equity" in todo:
        build_equity(set(a.series.split(",")))
    if "fo" in todo:
        build_fo()


if __name__ == "__main__":
    main()
