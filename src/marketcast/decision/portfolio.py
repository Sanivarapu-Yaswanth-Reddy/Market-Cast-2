"""L9  Decision layer: portfolio risk numbers read from sample paths (04 §L9) and the JSON contract (01 §7).

Deterministic arithmetic on paths; no model, no randomness, no language model (00 rule 10: numbers come from
code). Given paths l [M, H, N] and buy-and-hold value weights w:

    P_{i,k}^m = exp(sum_{j<=k} l_{i,j}^m)                 price relative
    R_k^m     = sum_i w_i (P_{i,k}^m - 1)                 portfolio return after k days
    outcome range = 5/50/95% quantiles of R_H
    VaR_q  = -Quantile_{1-q}(R_H);   ES_q = -mean(R_H | R_H <= -VaR_q)
    P(drawdown >= x) = share of scenarios with min_k R_k <= -x
    Euler ES contribution of holding i = mean(w_i (P_{i,H} - 1) | tail) / mean(R_H | tail)   (shares sum to 1)

This module returns PORTFOLIO-LEVEL risk only. It must never be turned into a per-stock recommendation
(00 rule 11); holdings the model does not cover are reported in flags.unsupported_assets, never dropped
silently.
"""
from __future__ import annotations

import numpy as np

ROUND = 6


def portfolio_paths(ell: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R [M, H] cumulative portfolio returns, contrib [M, N] each holding's contribution to R_H)."""
    rel = np.expm1(np.cumsum(ell.astype(np.float64), axis=1))              # [M, H, N] P - 1
    return rel @ w, rel[:, -1, :] * w[None, :]


def var_es(r_h: np.ndarray, q: float) -> tuple[float, float, np.ndarray]:
    """(VaR_q, ES_q, tail mask). Both are reported as POSITIVE losses; the JSON contract stores them as returns."""
    cut = np.quantile(r_h, 1.0 - q)
    tail = r_h <= cut
    return float(-cut), float(-r_h[tail].mean()), tail


def risk_report(ell: np.ndarray, keys: np.ndarray, holdings: dict[str, float], *, as_of: str, model_version: str, data_snapshot: str,
                value_inr: float | None = None, market_ell: np.ndarray | None = None, tail_filter: float | None = -0.05,
                trend: dict | None = None, sectors: list | None = None, stale_inputs: bool = False, ood_score: float | None = None) -> dict:
    """Build the product JSON of 01 §7 for one portfolio.

    holdings: key -> weight (weights of covered holdings are renormalised to sum to 1 and the uncovered ones are
    listed). market_ell [M, H]: market paths for the "tail story" (scenarios where the market falls by more than
    `tail_filter` over the horizon).
    """
    idx = {k: i for i, k in enumerate(map(str, keys))}
    covered = {k: float(v) for k, v in holdings.items() if k in idx and v > 0}
    unsupported = sorted(k for k in holdings if k not in idx)
    if not covered:
        raise ValueError("no holding is covered by the forecast universe")
    tot = sum(covered.values())
    w = np.zeros(len(keys))
    for k, v in covered.items():
        w[idx[k]] = v / tot
    r, contrib = portfolio_paths(ell, w)
    r_h = r[:, -1]
    m, h = r.shape
    p05, p50, p95 = np.quantile(r_h, [0.05, 0.5, 0.95])
    var95, es95, tail95 = var_es(r_h, 0.95)
    var99, es99, _ = var_es(r_h, 0.99)
    worst = r.min(axis=1)
    denom = r_h[tail95].mean()
    shares = contrib[tail95].mean(axis=0) / denom if abs(denom) > 1e-12 else np.zeros(len(keys))
    rc = sorted(({"key": k, "share_of_es95": round(float(shares[idx[k]]), ROUND)} for k in covered), key=lambda d: -d["share_of_es95"])
    out = {
        "as_of": as_of, "model_version": model_version, "data_snapshot": data_snapshot, "horizon_days": int(h), "n_paths": int(m),
        "portfolio": {"value_inr": value_inr, "holdings": [{"key": k, "weight": round(v / tot, ROUND)} for k, v in covered.items()]},
        "outcome_range": {"p05": round(float(p05), ROUND), "p50": round(float(p50), ROUND), "p95": round(float(p95), ROUND)},
        "var": {"95": round(-var95, ROUND), "99": round(-var99, ROUND)},
        "es": {"95": round(-es95, ROUND), "99": round(-es99, ROUND)},
        "prob_drawdown": {"5pct": round(float((worst <= -0.05).mean()), ROUND), "10pct": round(float((worst <= -0.10).mean()), ROUND)},
        "risk_contributions": rc,
        "flags": {"stale_inputs": bool(stale_inputs), "ood_score": ood_score, "unsupported_assets": unsupported},
    }
    if market_ell is not None and tail_filter is not None:
        sel = market_ell.sum(axis=1) < np.log1p(tail_filter)
        story = {"filter": f"market_{h}d < {tail_filter}", "share_paths": round(float(sel.mean()), ROUND), "co_falling_holdings": []}
        if sel.sum() >= 5:                                   # which holdings fall in most of those scenarios
            fell = (contrib[sel] < 0).mean(axis=0)
            story["co_falling_holdings"] = [k for k in covered if fell[idx[k]] >= 0.8]
            story["portfolio_p50_in_tail"] = round(float(np.median(r_h[sel])), ROUND)
        out["tail_story"] = story
    if trend is not None:
        out["market_trend"] = trend
    if sectors is not None:
        out["sectors_most_exposed"] = sectors
    return out
