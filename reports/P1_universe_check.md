# P1.5 Universe check

Generated 2026-11-16T15:38:53 by `python -m marketcast data universe`.

## MC500

- rebalances: 43 (2005-09-30 to 2026-09-30)
- members per period: min 500, max 500 (target 500)
- names replaced per rebalance (excluding the first): median 7.300000000000001%, max 13.4% (expected 5-12%)

## MC1000

- rebalances: 43 (2005-09-30 to 2026-09-30)
- members per period: min 747, max 1000 (target 1000)
- names replaced per rebalance (excluding the first): median 4.300000000000001%, max 14.1% (expected 5-12%)

## Overlap with the official Nifty 500 snapshot

- snapshot 20261116: 501 constituents, 497 resolved to a key
- MC500 (rebalance 2026-09-30) members also in Nifty 500: 404 of 500 = 80.8% by count (target >= 80%)
- share of MC500 traded value that is in Nifty 500: 92.4% (target >= 95%)
- the gap is expected: MC500 ranks by traded value, the index by free-float market cap (gap G4).
