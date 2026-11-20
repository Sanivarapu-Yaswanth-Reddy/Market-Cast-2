"""Sanity report on data/processed. Reads only needed columns (low memory). Writes reports/data_quality.txt
Usage: python scripts/validate_processed.py"""
from __future__ import annotations
from pathlib import Path
import datetime as dt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
out: list[str] = []
def log(s=""):
    print(s); out.append(str(s))

log(f"MarketCast processed-data report {dt.datetime.now().isoformat(timespec='seconds')}")
for f in sorted(P.glob("*.parquet")):
    log(f"  {f.name:42s} {f.stat().st_size/2**20:8.1f} MB")

ep = P / "equity_daily.parquet"
if ep.exists():
    cols = ["date", "symbol", "isin", "key", "close", "ret_1d", "adj_factor", "corp_action_flag", "deliv_pct", "series", "turnover"]
    import pyarrow.parquet as pq
    have = set(pq.read_schema(ep).names)
    e = pd.read_parquet(ep, columns=[c for c in cols + ["is_fund", "jump_unexplained"] if c in have])
    log(f"\n== equity_daily: {len(e):,} rows, {e['key'].nunique():,} keys, {e['symbol'].nunique():,} symbols, "
        f"{e['date'].min().date()} to {e['date'].max().date()}, {e['date'].nunique():,} trading dates")
    y = e.groupby(e["date"].dt.year).agg(days=("date", "nunique"), rows=("date", "size"), symbols=("symbol", "nunique"),
                                         deliv_cov=("deliv_pct", lambda s: s.notna().mean()),
                                         events=("corp_action_flag", "sum"))
    log("\nBy year:\n" + y.round(3).to_string())
    d = e.groupby("date").size()
    log(f"\nRows/day: min {d.min()} median {int(d.median())} max {d.max()}; days with <300 rows: {(d<300).sum()}")
    dates = sorted(e["date"].unique()); gaps = pd.Series(dates).diff().dt.days
    big = [(str(pd.Timestamp(dates[i]).date()), int(g)) for i, g in enumerate(gaps) if g and g > 5]
    log(f"Calendar gaps > 5 days between consecutive dates ({len(big)}): {big[:15]}")
    if "is_fund" in e:
        log(f"\nFund/ETF rows (ISIN INF...): {e['is_fund'].mean():.2%} of rows; stats below are for company shares only")
        e = e[~e["is_fund"]]
    if "jump_unexplained" in e:
        log(f"Unexplained jumps (>40%, no standard ratio): {int(e['jump_unexplained'].sum()):,}")
    r = e["ret_1d"]
    log(f"\nret_1d: mean {r.mean():.5f}  std {r.std():.4f}  |ret|>0.5: {(r.abs()>0.5).sum()}  (circuit bands bound most names; large ones are listing days, bad splits or relisting)")
    top = e.loc[r.abs().nlargest(10).index, ["date", "symbol", "series", "close", "ret_1d", "adj_factor"]]
    log("Largest |ret_1d|:\n" + top.to_string(index=False))
    dup = e.duplicated(["date", "symbol", "series"]).sum()
    log(f"\nDuplicate (date, symbol, series): {dup}")
    log(f"Missing ISIN: {e['isin'].isna().mean():.3%}   ret_1d missing: {r.isna().mean():.3%}")
    log("\nKnown-event checks (verify by eye; these are spot checks, not assertions):")
    for sym, day, note in [("RELIANCE", "2017-09-07", "1:1 bonus, expect factor about 0.5"),
                           ("HDFCBANK", "2019-09-19", "2:1 split (face value 2 to 1), expect about 0.5 and an ISIN change"),
                           ("YESBANK", "2020-03-06", "moratorium crash, expect large negative return, no factor")]:
        row = e[(e.symbol == sym) & (e.date == day)]
        log(f"  {sym} {day} ({note}): " + (row[['close','ret_1d','adj_factor','isin']].to_string(index=False, header=False) if len(row) else "no row"))
    hb = e[e.symbol == "HDFCBANK"].drop_duplicates("isin")[["isin"]]
    log(f"  HDFCBANK ISINs seen: {list(hb['isin'])}  -> if more than one, the raw ISIN splits its history; company_id (first 9 chars) must join them")
else:
    log("equity_daily.parquet not found")

for name, key, extra in [("indices_daily", "index", None), ("fo_daily", "symbol", None)]:
    f = P / f"{name}.parquet"
    if f.exists():
        x = pd.read_parquet(f)
        dc = "date" if "date" in x.columns else x.columns[0]
        log(f"\n== {name}: {len(x):,} rows, {x[key].nunique() if key in x else '?'} {key}s, {pd.to_datetime(x[dc]).min().date()} to {pd.to_datetime(x[dc]).max().date()}")
fg = P / "fred_global.parquet"
if fg.exists():
    x = pd.read_parquet(fg); log(f"\n== fred_global: {x.shape[0]:,} rows, {x.shape[1]-1} series")

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "data_quality.txt").write_text("\n".join(out), encoding="utf-8")
print("\nSaved reports/data_quality.txt")
