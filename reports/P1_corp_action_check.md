# P1.3 Corporate-action and return check

Generated 2026-11-16T19:31:53.

Returns are close / (own previous close x inferred factor) - 1 (decision D-22). Price returns only: dividends are missing (gap G2).

- stock-days with a return: 8,219,154
- inferred split/bonus/consolidation events: 2,071
- unexplained jumps (> 40%, no standard ratio; masked from targets): 2,370
- |log return| > 30%: 2,553 (0.0311% of stock-days); after excluding flagged rows: 183 (0.0022%; acceptance < 0.05%)
- exits (last traded row before the panel end): 1,096; terminal payoff unknown for all of them (no buyout/merger table), so no target is built after the last traded day

## Not done here (needs a human or an official table, gap G15)

- The 20 random stock-day comparison against exchange records (gate G1 criterion 1).
- Hand verification of every inferred event among the 30 largest stocks.
- Classification of unexplained jumps into demerger / rights / relisting / genuine move.
