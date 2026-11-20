# MarketCast data catalog

This folder holds the raw layer of MarketCast's point-in-time "market reanalysis" (L0 in the design doc) and the scripts that build and refresh it. Every file comes from a public, published source. Nothing is scraped from internal web APIs.

## How to run

```
cd "C:\Users\PC\OneDrive\Desktop\Market Cast"
python scripts\marketcast_download.py            # full build, resumable (Ctrl+C and re-run any time)
python scripts\marketcast_build_panel.py         # turn raw files into training tables
```

* `--quick` grabs only the last year (smoke test). `--with-imd` adds IMD gridded rainfall. `--with-volt` adds NSE's per-security volatility files.
* A full default build is roughly 4 to 5 GB and about 25,000 requests. It takes a few hours at the polite default rate. OneDrive will try to sync all of it, so consider pausing OneDrive sync or moving `data\` outside OneDrive.
* `data\manifest.csv` logs every file with its source URL and fetch time (UTC).
* FRED doesn't answer Python on this network, and IIMA may not either. Those files were fetched through the browser into your **Downloads** folder as `marketcast__*` files. The downloader moves them into `data\raw\` automatically the next time it runs. So once the first run finishes, double-click `RUN_DOWNLOAD.bat` once more: it skips everything already downloaded, imports the bundles and rebuilds the tables.
* `delisted.csv` is Latin-1 encoded: `pd.read_csv(path, encoding="latin-1")`.

## What is in it, mapped to the MarketCast layers

| Folder | Source | Coverage | Feeds | Why it matters |
|---|---|---|---|---|
| `raw/nse/cm_bhavcopy/` | NSE cash-market bhavcopy (old format to 5 Jul 2024, UDiFF after) | daily, 2005 onward (back to 1995 with `--start`) | Targets (returns, realized vol) and L1 price features | Every listed security, including stocks later delisted, so the panel is survivorship-free |
| `raw/nse/delivery_mto/` | NSE security-wise delivery position | daily, 2005 onward | L1 delivery z-scores | Delivery % separates conviction buying from intraday churn |
| `raw/nse/fo_bhavcopy/` | NSE F&O bhavcopy, every future and option contract | daily, 2010 onward by default (data exists from 2000) | Derivatives and positioning layer | OI, change in OI, put-call ratio, futures basis, max-pain near expiry |
| `raw/nse/indices_close_all/` | NSE all-indices daily close | daily, Jul 2012 onward | Global and sector nodes, market heads, trend target | About 145 indices: Nifty 50/500, every sector index, **India VIX**, plus index P/E, P/B and dividend yield |
| `raw/nse/reference/<date>/` | NSE security master (EQUITY_L), delisted list, ETF list, 34 index constituent lists | snapshot on download date | Universe, ISIN keys, sector-fabric warp | Constituent lists carry NSE's industry label; delisted list gives exit dates for terminal returns |
| `raw/global/fred/` | FRED (St. Louis Fed) | daily or monthly, decades | Global nodes and India macro | US 2y/10y, Fed funds, dollar index, USD/INR, Brent, WTI, CBOE VIX, Nasdaq, Nikkei, credit spreads, IMF commodity prices; India CPI, IIP, 10y yield, call rate, M3, trade, FX reserves |
| `raw/factors/iima_india/` | IIM Ahmedabad four-factor data library (Agarwalla, Jacob, Varma), survivorship-bias adjusted | daily, Oct 1993 to Dec 2025 | Factor-model baseline in MarketBench | Market, SMB, HML, WML for India, ready-made |
| `raw/factors/ken_french/` | Ken French data library | daily | Global factor context | US 5-factor + momentum, emerging-market factors |
| `raw/weather/imd/` (optional) | IMD Pune 0.25° gridded daily rainfall via `imdlib` | 1901 onward | Weather nodes (agri, FMCG, fertilizer, tractor, power) | Plus an all-India daily mean rainfall series |

## Processed tables (after `marketcast_build_panel.py`)

* `processed/equity_daily.parquet`: one row per date and ISIN. Raw OHLC, volume, turnover, trades and delivery, plus `adj_factor` from NSE's own ex-date-adjusted PREVCLOSE, back-adjusted prices, and a point-in-time-safe `ret_1d = close / prev_close − 1`.
* `processed/corporate_action_events_inferred.parquet`: every ex-date the factor detected (splits, bonuses, rights). Spot-check these against known events for the Phase 1 gate.
* `processed/indices_daily.parquet`, `processed/fo_daily.parquet` (futures OI, near-month settle, call and put OI, PCR per underlying) and `processed/fred_global.parquet`.

## Point-in-time rules already respected

* Bhavcopy, delivery, F&O and index files are published the same evening, so all their fields are knowable at the close of that date.
* FRED values are **latest revised** numbers, not first releases. For macro series that get revised (IIP, GDP), use ALFRED vintages before trusting a backtest.
* Constituent lists are today's snapshot. Re-run the reference group regularly and keep each dated folder, so membership history accumulates from now on.
* Pre-2011 bhavcopies have no ISIN column. The builder back-fills ISIN from the symbol's first later appearance, so ticker renames before 2011 remain a known gap.

## Gaps that still need another source

| Need (from the doc) | Best route |
|---|---|
| Historical index membership (who was in the Nifty 500 on a past date) | NSE index-change press releases (niftyindices.com, "Index Changes"), compiled into a dated table |
| NSE's four-level industry classification over time | NSE publishes the current mapping; history needs dated snapshots or a vendor |
| Corporate announcements, results dates, board meetings | NSE/BSE announcement pages (manual or licensed feed); BSE XBRL for financials |
| FII/FPI and DII daily flows | NSDL FPI Monitor (fpi.nsdl.co.in) daily reports; NSE publishes only the latest day |
| Mutual fund flows | AMFI monthly reports (amfiindia.com) |
| RBI repo rate history and policy dates | RBI DBIE (dbie.rbi.org.in) and RBI MPC press releases |
| Consensus estimates and business-group links | Paid (CMIE Prowess, Refinitiv, Bloomberg, Capitaline) |
| News and event text at scale | GDELT 2.0 event and GKG tables (free, via BigQuery), filtered to India |
| WeatherNext forecasts as inputs | Google Earth Engine or BigQuery WeatherNext datasets |

## Terms of use

NSE data is for personal and research use. Check NSE's terms before redistributing or using it commercially (the doc's Phase 3 B2B API will need a licensed feed). Cite the IIMA library as Agarwalla, Jacob and Varma (2013), IIMA Working Paper 2013-09-05. FRED data carries each original publisher's terms.
