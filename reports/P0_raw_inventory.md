# P0.1 Raw download inventory

Generated 2026-11-16T07:57:33 by `python -m marketcast data inventory`. Calendar = dates with a cash-market bhavcopy.

> First version built from `data/manifest.csv` (files with status ok) on 2026-11-16, because the agent could not run commands on the data PC. Re-run `python -m marketcast data inventory` on the data PC to regenerate it from disk.

## Acceptance checks

| Check | Result | Detail |
|---|---|---|
| cm_bhavcopy >= 5,300 files | PASS | 5,398 |
| delivery_mto: no gap > 3 sessions vs cm | PASS | longest 0 from - |
| fo_bhavcopy >= 4,100 files | PASS | 4,163 |
| fo_bhavcopy: no gap > 3 sessions vs cm | PASS | longest 1 from 2013-10-09 |
| indices_close_all >= 3,450 files | PASS | 3,531 |
| indices_close_all: no gap > 3 sessions vs cm | PASS | longest 2 from 2015-03-12 |
| delivery within 1% of cm | PASS | 5,398 vs 5,398 |

## Daily NSE series

| folder | series | files | first | last | missing_vs_cm | longest_gap_sessions | gap_starts |
|---|---|---|---|---|---|---|---|
| cm_bhavcopy | cash-market bhavcopy | 5398 | 2005-01-03 | 2026-11-13 | 0 | 0 |  |
| delivery_mto | delivery (MTO) | 5398 | 2005-01-03 | 2026-11-13 | 0 | 0 |  |
| fo_bhavcopy | F&O bhavcopy | 4163 | 2010-01-04 | 2026-11-13 | 2 | 1 | 2013-10-09 |
| indices_close_all | index closes | 3531 | 2012-07-02 | 2026-11-13 | 12 | 2 | 2015-03-12 |

## Files per year

| year | cm_bhavcopy | delivery_mto | fo_bhavcopy | indices_close_all |
|---|---|---|---|---|
| 2005 | 249 | 249 | 0 | 0 |
| 2006 | 247 | 247 | 0 | 0 |
| 2007 | 249 | 249 | 0 | 0 |
| 2008 | 246 | 246 | 0 | 0 |
| 2009 | 242 | 242 | 0 | 0 |
| 2010 | 251 | 251 | 251 | 0 |
| 2011 | 247 | 247 | 247 | 0 |
| 2012 | 247 | 247 | 247 | 123 |
| 2013 | 248 | 248 | 247 | 247 |
| 2014 | 243 | 243 | 243 | 241 |
| 2015 | 247 | 247 | 247 | 239 |
| 2016 | 246 | 246 | 246 | 245 |
| 2017 | 248 | 248 | 248 | 248 |
| 2018 | 246 | 246 | 246 | 246 |
| 2019 | 244 | 244 | 244 | 244 |
| 2020 | 250 | 250 | 250 | 250 |
| 2021 | 248 | 248 | 247 | 248 |
| 2022 | 248 | 248 | 248 | 248 |
| 2023 | 245 | 245 | 245 | 245 |
| 2024 | 246 | 246 | 246 | 246 |
| 2025 | 248 | 248 | 248 | 248 |
| 2026 | 213 | 213 | 213 | 213 |

## Other raw folders (file counts)

- factors: 15
- flows: 1
- global: 40
- nse/reference: 39

## Manifest (latest status per file)

| group | status | n |
|---|---|---|
| cm | ok | 5398 |
| delivery | ok | 5398 |
| factors | ok | 4 |
| fo | ok | 4163 |
| imports | ok | 52 |
| indices | ok | 3531 |
| reference | ok | 39 |
