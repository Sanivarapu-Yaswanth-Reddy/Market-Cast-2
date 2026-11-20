# P1.6 Sector coverage

Generated 2026-11-16T15:53:41. Snapshot classification applied to all history (`hindsight_class`, D-08). Tree: market -> macro_sector -> sector -> industry (gap G3: no basic-industry level).

## Node counts

- macro sectors: 13
- sectors: 23
- industries: 27
- companies: 3739

## Label source (all companies ever traded)

| source | companies |
|---|---|
| unclassified | 1489 |
| keyword | 1254 |
| constituent | 751 |
| manual | 245 |

## Share of MC500 traded value by label source, % (acceptance: unclassified <= 5 in every year)

| year | constituent | keyword | manual | unclassified |
|---|---|---|---|---|
| 2005 | 53.17 | 11.7 | 28.55 | 6.57 |
| 2006 | 59.99 | 9.64 | 27.18 | 3.18 |
| 2007 | 59.29 | 11.41 | 27.36 | 1.94 |
| 2008 | 57.04 | 14.09 | 27.95 | 0.92 |
| 2009 | 61.96 | 11.66 | 25.81 | 0.56 |
| 2010 | 66.18 | 9.15 | 22.76 | 1.9 |
| 2011 | 72.03 | 7.41 | 19.01 | 1.55 |
| 2012 | 74.86 | 7.14 | 17.42 | 0.58 |
| 2013 | 77.43 | 6.59 | 15.54 | 0.44 |
| 2014 | 80.35 | 4.85 | 14.45 | 0.35 |
| 2015 | 82.88 | 4.36 | 12.43 | 0.34 |
| 2016 | 82.39 | 4.58 | 12.47 | 0.56 |
| 2017 | 83.57 | 4.21 | 11.85 | 0.37 |
| 2018 | 84.95 | 3.83 | 11.1 | 0.12 |
| 2019 | 90.98 | 1.67 | 7.3 | 0.04 |
| 2020 | 94.22 | 1.02 | 4.69 | 0.07 |
| 2021 | 93.81 | 1.28 | 4.85 | 0.05 |
| 2022 | 92.98 | 1.34 | 5.63 | 0.05 |
| 2023 | 95.98 | 0.75 | 3.23 | 0.03 |
| 2024 | 97.1 | 0.96 | 1.94 | 0 |
| 2025 | 98.52 | 0.64 | 0.81 | 0.03 |
| 2026 | 98.21 | 1.07 | 0.66 | 0.07 |

**FAILS in years [2005]** - extend KEYWORDS or obtain the NSE classification file (G3).

Keyword-labelled value is low-confidence (0.4) and should be reviewed by hand for the largest names.
