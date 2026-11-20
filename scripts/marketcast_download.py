#!/usr/bin/env python3
"""
MarketCast data downloader
==========================

Builds the raw layer of the MarketCast point-in-time "market reanalysis" (L0 in the
design doc) from public, published archive files only. Pure standard library: no
pip installs needed (IMD rainfall is the one optional extra, see --with-imd).

Everything is resumable: files already on disk are skipped, so you can stop the
script (Ctrl+C) and run it again at any time. Every download is logged to
data/manifest.csv with the UTC time it was fetched (useful for point-in-time audits).

Usage (from the Market Cast folder):
    python scripts/marketcast_download.py                 # default full build
    python scripts/marketcast_download.py --quick         # last ~1 year only, to test
    python scripts/marketcast_download.py --only reference,global,factors
    python scripts/marketcast_download.py --start 2005-01-01 --fo-start 2010-01-01
    python scripts/marketcast_download.py --with-imd      # also IMD gridded rainfall

Groups (run in this order):
    reference  NSE security master, delisted list, ETF list, 35 index constituent lists
    global     FRED: US rates, dollar, USD/INR, Brent, WTI, VIX, S&P 500, Nasdaq, Nikkei,
               credit spreads, plus India CPI, IIP, 10y yield, call money rate
    factors    IIM Ahmedabad India four-factor library (survivorship-bias adjusted),
               Ken French US/developed factors
    indices    NSE ind_close_all: OHLC + P/E, P/B, div yield for ~145 indices incl.
               India VIX and all sector indices, daily from Jul 2012
    cm         NSE cash-market bhavcopy (every EQ/BE/etc. security, OHLC, volume,
               turnover, trades, ISIN, adjusted PREVCLOSE), daily
    delivery   NSE MTO security-wise delivery position (delivery qty and %), daily
    fo         NSE F&O bhavcopy (all futures and options contracts: OI, change in OI,
               settle price), daily
    volatility NSE CMVOLT daily/annualised volatility per security (optional, --with-volt)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the Market Cast folder
DATA = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
NSE_ARCH = "https://nsearchives.nseindia.com"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
UDIFF_START = dt.date(2024, 7, 8)   # NSE switched CM and F&O bhavcopy to UDiFF format

_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "missing": 0, "fail": 0, "bytes": 0}
_consec_fail = [0]
_cooldown_until = [0.0]


# --------------------------------------------------------------------------- helpers
def log_manifest(group: str, url: str, path: Path, status: str, nbytes: int) -> None:
    with _lock:
        new = not MANIFEST.exists()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["fetched_at_utc", "group", "status", "bytes", "path", "url"])
            try:
                rel = path.relative_to(ROOT).as_posix()
            except ValueError:
                rel = str(path)
            w.writerow([dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        group, status, nbytes, rel, url])


def fetch(url: str, dest: Path, group: str, referer: str | None = None,
          retries: int = 3, min_bytes: int = 1, delay: float = 0.25,
          timeout: int = 60) -> str:
    """Download url to dest. Returns ok / skip / missing / fail."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        with _lock:
            _stats["skip"] += 1
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        headers["Referer"] = referer
    for attempt in range(retries):
        try:
            wait = _cooldown_until[0] - time.time()
            if wait > 0:
                time.sleep(wait)
            time.sleep(delay + random.random() * delay)          # be polite
            req = urllib.request.Request(url, headers=headers)
            t_start = time.time()
            chunks = []
            with urllib.request.urlopen(req, timeout=timeout) as r:
                while True:
                    c = r.read(65536)
                    if not c:
                        break
                    chunks.append(c)
                    if time.time() - t_start > timeout * 2:
                        raise TimeoutError("response too slow")
            body = b"".join(chunks)
            if len(body) < min_bytes or body[:15].lower().startswith(b"<!doctype html"):
                status = "missing"
                break
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(body)
            os.replace(tmp, dest)
            with _lock:
                _stats["ok"] += 1
                _stats["bytes"] += len(body)
                _consec_fail[0] = 0
            log_manifest(group, url, dest, "ok", len(body))
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                status = "missing"            # holiday / not published: normal
                break
            status = f"http{e.code}"
            time.sleep(2 ** attempt * 2)
        except Exception as e:                # network hiccup
            status = f"err:{type(e).__name__}"
            with _lock:
                _consec_fail[0] += 1
                if _consec_fail[0] >= 12 and _cooldown_until[0] < time.time():
                    print("   ...server is stalling requests; pausing 5 minutes, then resuming", flush=True)
                    _cooldown_until[0] = time.time() + 300
                    _consec_fail[0] = 0
            time.sleep(2 ** attempt * 2)
    with _lock:
        _stats["missing" if status == "missing" else "fail"] += 1
    if status != "missing":
        log_manifest(group, url, dest, status, 0)
    return status


def weekdays(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def run_parallel(jobs, workers: int, label: str) -> None:
    jobs = list(jobs)
    if not jobs:
        return
    print(f"\n== {label}: {len(jobs)} candidate files (holidays show up as 'missing') ==")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, *j[:3], **(j[3] if len(j) > 3 else {})) for j in jobs]
        for f in as_completed(futs):
            done += 1
            if done % 200 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"   {label}: {done}/{len(jobs)}  ok={_stats['ok']} skip={_stats['skip']} "
                      f"missing={_stats['missing']} fail={_stats['fail']}  "
                      f"{_stats['bytes']/1e6:,.0f} MB  {el/60:.1f} min", flush=True)


# --------------------------------------------------------------------------- groups
INDEX_LISTS = [
    # broad market (universe definition)
    "ind_nifty50list", "ind_niftynext50list", "ind_nifty100list", "ind_nifty200list",
    "ind_nifty500list", "ind_niftymidcap100list", "ind_niftymidcap150list",
    "ind_niftysmallcap100list", "ind_niftysmallcap250list", "ind_niftymicrocap250_list",
    "ind_niftytotalmarket_list",
    # sectoral / thematic (sector-fabric membership)
    "ind_niftybanklist", "ind_nifty_privatebanklist", "ind_niftypsubanklist",
    "ind_niftyfinancelist", "ind_niftyfinancialservices25_50list", "ind_niftyitlist",
    "ind_niftyfmcglist", "ind_niftypharmalist", "ind_niftyhealthcarelist",
    "ind_niftyautolist", "ind_niftymetallist", "ind_niftyrealtylist", "ind_niftyenergylist",
    "ind_niftyoilgaslist", "ind_niftymedialist", "ind_niftyconsumerdurableslist",
    "ind_niftyconsumptionlist", "ind_niftycommoditieslist", "ind_niftyinfralist",
    "ind_niftycpselist", "ind_niftyservicelist", "ind_niftymnclist",
    "ind_niftyindiadefence_list",
]


def g_reference(a) -> None:
    stamp = dt.date.today().strftime("%Y%m%d")        # snapshot date: these files change
    base = DATA / "nse" / "reference" / stamp
    jobs = [
        (f"{NSE_ARCH}/content/equities/EQUITY_L.csv", base / "EQUITY_L.csv", "reference"),
        (f"{NSE_ARCH}/content/equities/delisted.csv", base / "delisted.csv", "reference"),
        (f"{NSE_ARCH}/content/equities/eq_etfseclist.csv", base / "eq_etfseclist.csv", "reference"),
        (f"{NSE_ARCH}/content/equities/bulk.csv", base / "bulk_deals_latest.csv", "reference"),
        (f"{NSE_ARCH}/content/equities/block.csv", base / "block_deals_latest.csv", "reference"),
    ]
    for n in INDEX_LISTS:
        jobs.append((f"{NSE_ARCH}/content/indices/{n}.csv",
                     base / "index_constituents" / f"{n}.csv", "reference"))
    run_parallel(jobs, 4, "reference (NSE security master + index constituents)")


FRED = {
    # global nodes
    "DGS10": "US 10y Treasury yield", "DGS2": "US 2y Treasury yield",
    "T10Y2Y": "US 10y-2y spread", "DFF": "Fed funds effective rate",
    "DTWEXBGS": "Broad US dollar index", "DEXINUS": "USD/INR daily",
    "DCOILBRENTEU": "Brent crude", "DCOILWTICO": "WTI crude",
    "VIXCLS": "CBOE VIX", "SP500": "S&P 500 (last 10y only)", "NASDAQCOM": "Nasdaq Composite",
    "DJIA": "Dow Jones (last 10y)", "NIKKEI225": "Nikkei 225",
    "BAMLH0A0HYM2": "US high-yield OAS", "BAMLC0A0CM": "US IG corporate OAS",
    "DEXCHUS": "CNY per USD", "DEXJPUS": "JPY per USD", "DEXUSEU": "USD per EUR",
    "PCOPPUSDM": "Copper price (monthly)", "PALUMUSDM": "Aluminium price (monthly)",
    "PIORECRUSDM": "Iron ore price (monthly)", "PCOALAUUSDM": "Coal price (monthly)",
    "PNGASEUUSDM": "EU natural gas (monthly)", "PWHEAMTUSDM": "Wheat (monthly)",
    "PRICENPQUSDM": "Rice (monthly)", "PSUGAISAUSDM": "Sugar (monthly)",
    "PCOTTINDUSDM": "Cotton (monthly)", "PRUBBUSDM": "Rubber (monthly)",
    "USEPUINDXD": "US economic policy uncertainty (daily)",
    # India macro (monthly / annual)
    "INDCPIALLMINMEI": "India CPI all items", "INDIRLTLT01STM": "India 10y govt bond yield",
    "IRSTCI01INM156N": "India call money / interbank rate",
    "INDPROINDMISMEI": "India industrial production", "MABMM301INM189S": "India M3",
    "XTEXVA01INM667S": "India exports", "XTIMVA01INM667S": "India imports",
    "MKTGDPINA646NWDB": "India GDP (annual, WB)", "INDEPUINDXM": "India policy uncertainty",
    "CCUSMA02INM618N": "INR per USD monthly avg", "TRESEGINM052N": "India FX reserves ex-gold",
}


def g_global(a) -> None:
    base = DATA / "global" / "fred"
    have = [sid for sid in FRED if (base / f"{sid}.csv").exists()]
    if len(have) == len(FRED):
        print("\n== FRED: all series already present ==")
    else:
        probe = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", base / "DGS10.csv",
                      "global", min_bytes=50, retries=1, timeout=20)
        if probe not in ("ok", "skip"):
            print("\n== FRED does not answer Python on this network. Skipping; the browser bundle "
                  "(marketcast__global__fred__bundle.csv in Downloads) is imported instead. ==")
            return
    jobs = [(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", base / f"{sid}.csv",
             "global", {"min_bytes": 50, "timeout": 30, "retries": 2}) for sid in FRED]
    run_parallel(jobs, 3, "global + India macro (FRED)")
    with open(base / "_series_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "description", "url"])
        for sid, desc in FRED.items():
            w.writerow([sid, desc, f"https://fred.stlouisfed.org/series/{sid}"])


def g_factors(a) -> None:
    iima = "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/DATA/"
    rel = a.iima_release
    files = [
        "FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv",
        "FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted.csv",
        "Size_Value_Portfolio_Returns_Daily_SurvivorshipBiasAdjusted.csv",
        "Size_Momentum_Portfolio_Returns_Daily_SurvivorshipBiasAdjusted.csv",
        "Size_and_Value_Break_Points_for_Size_Value_Portfolios.csv",
        "Size_and_Momentum_Break_Points_for_Size_Momentum_Portfolios.csv",
    ]
    jobs = [(iima + f"{rel}_{fn}", DATA / "factors" / "iima_india" / f"{rel}_{fn}", "factors",
             {"referer": "https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/",
              "min_bytes": 100, "timeout": 30, "retries": 2}) for fn in files]
    kf = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    for fn in ["F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip",
               "Emerging_5_Factors_CSV.zip", "Emerging_MOM_Factor_CSV.zip"]:
        jobs.append((kf + fn, DATA / "factors" / "ken_french" / fn, "factors",
                     {"min_bytes": 100, "timeout": 30, "retries": 2}))
    run_parallel(jobs, 2, "factor libraries (IIMA India 4-factor, Ken French)")


def g_imports(a) -> None:
    """Move files the browser saved as marketcast__<dir>__<dir>__<name> from Downloads into data/raw."""
    dl = Path.home() / "Downloads"
    found = sorted(dl.glob("marketcast__*")) if dl.exists() else []
    found = [p for p in found if not p.name.endswith((".crdownload", ".tmp", ".part"))]
    if not found:
        return
    print(f"\n== importing {len(found)} browser-downloaded file(s) from Downloads ==")
    for p in found:
        parts = p.name.split("__")[1:]
        if len(parts) < 2:
            continue
        if parts[-1] == "bundle.csv" and parts[:2] == ["global", "fred"]:
            _split_fred_bundle(p, DATA / "global" / "fred")
        else:
            dest = DATA.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, dest)
            log_manifest("imports", "browser:" + p.name, dest, "ok", dest.stat().st_size)
            print(f"   {p.name} -> {dest.relative_to(ROOT)}")
            continue
        p.unlink()


def _split_fred_bundle(p: Path, base: Path) -> None:
    """Long CSV (series_id,observation_date,value) -> one fredgraph-style CSV per series."""
    base.mkdir(parents=True, exist_ok=True)
    series: dict[str, list] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series.setdefault(row["series_id"], []).append((row["observation_date"], row["value"]))
    for sid, rows in series.items():
        dest = base / f"{sid}.csv"
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["observation_date", sid])
            w.writerows(rows)
        log_manifest("imports", f"browser:fred:{sid}", dest, "ok", dest.stat().st_size)
    print(f"   FRED bundle -> {len(series)} series in {base.relative_to(ROOT)}")


def _cm_url(d: dt.date):
    if d >= UDIFF_START:
        return (f"{NSE_ARCH}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
                f"BhavCopy_NSE_CM_{d:%Y%m%d}.csv.zip")
    m = MONTHS[d.month - 1]
    return (f"{NSE_ARCH}/content/historical/EQUITIES/{d.year}/{m}/cm{d:%d}{m}{d.year}bhav.csv.zip",
            f"cm{d:%Y%m%d}bhav.csv.zip")


def _fo_url(d: dt.date):
    if d >= UDIFF_START:
        return (f"{NSE_ARCH}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
                f"BhavCopy_NSE_FO_{d:%Y%m%d}.csv.zip")
    m = MONTHS[d.month - 1]
    return (f"{NSE_ARCH}/content/historical/DERIVATIVES/{d.year}/{m}/fo{d:%d}{m}{d.year}bhav.csv.zip",
            f"fo{d:%Y%m%d}bhav.csv.zip")


def g_daily(a, group: str) -> None:
    end = a.end
    if group == "cm":
        start, label = a.start, "NSE cash-market bhavcopy"
        mk = lambda d: _cm_url(d)
    elif group == "delivery":
        start, label = max(a.start, dt.date(2005, 1, 1)), "NSE delivery position (MTO)"
        mk = lambda d: (f"{NSE_ARCH}/archives/equities/mto/MTO_{d:%d%m%Y}.DAT", f"MTO_{d:%Y%m%d}.DAT")
    elif group == "fo":
        start, label = max(a.fo_start, dt.date(2000, 6, 12)), "NSE F&O bhavcopy"
        mk = lambda d: _fo_url(d)
    elif group == "indices":
        start, label = max(a.start, dt.date(2012, 7, 1)), "NSE all-indices close (incl. India VIX, P/E, P/B)"
        mk = lambda d: (f"{NSE_ARCH}/content/indices/ind_close_all_{d:%d%m%Y}.csv",
                        f"ind_close_all_{d:%Y%m%d}.csv")
    elif group == "volatility":
        start, label = max(a.start, dt.date(2011, 1, 1)), "NSE CMVOLT security volatility"
        mk = lambda d: (f"{NSE_ARCH}/archives/nsccl/volt/CMVOLT_{d:%d%m%Y}.CSV", f"CMVOLT_{d:%Y%m%d}.csv")
    else:
        raise ValueError(group)
    folder = {"cm": "cm_bhavcopy", "delivery": "delivery_mto", "fo": "fo_bhavcopy",
              "indices": "indices_close_all", "volatility": "cm_volatility"}[group]
    jobs = []
    for d in weekdays(start, end):
        url, name = mk(d)
        jobs.append((url, DATA / "nse" / folder / str(d.year) / name, group,
                     {"referer": "https://www.nseindia.com/", "min_bytes": 200,
                      "timeout": 30, "delay": 0.5}))
    run_parallel(jobs, a.workers, label)


def g_imd(a) -> None:
    """IMD 0.25-degree gridded daily rainfall via imdlib, plus an all-India daily mean CSV."""
    try:
        import imdlib as imd  # type: ignore
    except ImportError:
        print("\n== IMD rainfall: installing imdlib (pip) ==")
        os.system(f'"{sys.executable}" -m pip install --quiet imdlib')
        try:
            import imdlib as imd  # type: ignore
        except ImportError:
            print("   imdlib could not be installed; skipping IMD. Run: pip install imdlib")
            return
    out = DATA / "weather" / "imd"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = max(a.start.year, 1901), min(a.end.year, dt.date.today().year - 1)
    print(f"\n== IMD gridded rainfall {y0}-{y1} (~25 MB per year) ==")
    for y in range(y0, y1 + 1):
        if (out / "rain" / f"{y}.grd").exists():
            continue
        try:
            imd.get_data("rain", y, y, fn_format="yearwise", file_dir=str(out))
        except Exception as e:
            print(f"   rain {y}: failed ({e})")
    try:
        import numpy as np  # noqa
        data = imd.open_data("rain", y0, y1, "yearwise", str(out))
        ds = data.get_xarray()
        ds = ds.where(ds["rain"] != -999.0)
        s = ds["rain"].mean(dim=["lat", "lon"]).to_pandas()
        s.name = "all_india_mean_rain_mm"
        s.to_csv(out / "all_india_daily_mean_rainfall.csv")
        print("   wrote all_india_daily_mean_rainfall.csv")
    except Exception as e:
        print(f"   could not build all-India series ({e}); raw .grd files are still saved")


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2005-01-01", help="first date for daily NSE files")
    p.add_argument("--fo-start", default="2010-01-01", help="first date for F&O (large: ~1 MB/day)")
    p.add_argument("--end", default=dt.date.today().isoformat())
    p.add_argument("--only", default="", help="comma list of groups to run")
    p.add_argument("--quick", action="store_true", help="last ~1 year only (smoke test)")
    p.add_argument("--with-imd", action="store_true", help="also download IMD gridded rainfall")
    p.add_argument("--with-volt", action="store_true", help="also download CMVOLT volatility files")
    p.add_argument("--workers", type=int, default=2, help="parallel NSE downloads (keep small)")
    p.add_argument("--iima-release", default="2025-12", help="IIMA data library release tag")
    a = p.parse_args()
    a.start = dt.date.fromisoformat(a.start)
    a.fo_start = dt.date.fromisoformat(a.fo_start)
    a.end = dt.date.fromisoformat(a.end)
    if a.quick:
        a.start = a.fo_start = a.end - dt.timedelta(days=365)

    groups = ["imports", "reference", "indices", "global", "factors", "cm", "delivery", "fo"]
    if a.with_volt:
        groups.append("volatility")
    if a.with_imd:
        groups.append("imd")
    if a.only:
        groups = [g.strip() for g in a.only.split(",") if g.strip()]
    else:
        groups.append("imports")      # pick up bundles that finished downloading meanwhile

    print(f"MarketCast downloader -> {DATA}")
    print(f"groups={groups}  daily {a.start}..{a.end}  F&O from {a.fo_start}")
    t0 = time.time()
    try:
        for g in groups:
            if g == "imports":
                g_imports(a)
            elif g == "reference":
                g_reference(a)
            elif g == "global":
                g_global(a)
            elif g == "factors":
                g_factors(a)
            elif g == "imd":
                g_imd(a)
            else:
                g_daily(a, g)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. ok={_stats['ok']} already-had={_stats['skip']} "
          f"not-published={_stats['missing']} failed={_stats['fail']}  "
          f"downloaded {_stats['bytes']/1e6:,.0f} MB")
    if _stats["fail"]:
        print("Some files failed (see data/manifest.csv). Re-run to retry them.")


if __name__ == "__main__":
    main()
