"""Forcings (03 §9): calendar facts known IN ADVANCE for any future session, so the rollout may read them for
t+1 ... t+20 without leaking.

Each row describes one session. The model concatenates the row of day tau to every stock's inputs for day tau
(history) and feeds the row of the TARGET day to the processor (plus the rollout step k, added at run time).

DELIBERATELY EXCLUDED
    is_rbi_policy_day : the file on disk lists only days on which the rate CHANGED. Feeding that as a forcing
                        would tell the model in advance that a change happens on the target day (a leak).
                        It returns when a pre-announced MPC meeting calendar is compiled (decision D-25).
    is_results_day, lock-in expiry : per-stock, need gap G5.

Expiry flags come from contracts NIFTY actually listed; a listed contract's expiry is public from listing.
Rows past the last real session are projected weekdays (calendar.extend_calendar) and are only ever used as
future forcings of the most recent decision dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORCING_NAMES = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "ln_days_since_prev", "is_monthly_expiry", "is_weekly_expiry",
                 "days_to_monthly_expiry", "is_budget_day", "is_election_result_day", "is_index_rebalance_effective",
                 "month_sin", "month_cos", "fy_end_window", "holiday_adjacent", "after_t_plus_1", "after_fo_curbs"]


def build(cal: pd.DataFrame, rule_changes: dict | None = None) -> np.ndarray:
    """Calendar frame (one row per session, real or projected) -> [n_rows, F_f] float32."""
    rule_changes = rule_changes or {}
    d = pd.DatetimeIndex(cal["date"])
    f = np.zeros((len(cal), len(FORCING_NAMES)), dtype=np.float32)
    wd = np.minimum(d.weekday.to_numpy(), 4)                       # a weekend special session counts as Friday
    f[np.arange(len(cal)), wd] = 1.0
    f[:, 5] = np.log1p(cal["calendar_days_since_prev"].to_numpy(dtype=float))
    for j, c in [(6, "is_monthly_expiry"), (7, "is_weekly_expiry"), (9, "is_budget_day"), (10, "is_election_result_day"),
                 (11, "is_index_rebalance_effective"), (14, "fy_end_window"), (15, "holiday_adjacent")]:
        f[:, j] = cal[c].fillna(False).to_numpy(dtype=bool) if c in cal.columns else 0.0
    f[:, 8] = cal["days_to_monthly_expiry"].fillna(30).to_numpy(dtype=float) / 30.0
    ang = 2.0 * np.pi * (d.month.to_numpy() - 1) / 12.0
    f[:, 12], f[:, 13] = np.sin(ang), np.cos(ang)
    f[:, 16] = d >= pd.Timestamp(str(rule_changes.get("t_plus_1_complete", "2023-01-27")))
    f[:, 17] = d >= pd.Timestamp(str(rule_changes.get("sebi_fo_curbs_start", "2024-11-20")))
    return f
