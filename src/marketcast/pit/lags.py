"""Point-in-time clock: the cutoff of a decision date and the available_at of every external series (02 §2).

THE RULE (decision D-11): a fact may feed a forecast "made at the close of session t" iff
    available_at <= cutoff(t) = t 15:30 Asia/Kolkata.
NSE end-of-day files for t are the one exception (they describe that 15:30 close) and are handled by
construction in the panel, not through this clock.

All timestamps are tz-aware (Asia/Kolkata). Naive timestamps for availability are a bug (07 §4).
"""
from __future__ import annotations

import pandas as pd

from marketcast.util import IST, load_yaml

CUTOFF_HHMM = "15:30"


def cutoff(d) -> pd.Timestamp:
    """Decision cutoff of session d: d 15:30 IST."""
    return pd.Timestamp(f"{pd.Timestamp(d).date()} {CUTOFF_HHMM}", tz=IST)


def cutoffs(dates) -> pd.DatetimeIndex:
    return (pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=15, minutes=30)).tz_localize(IST)


def load_lags() -> dict:
    return load_yaml("pit_lags.yaml")


def infer_freq(valid_dates: pd.DatetimeIndex) -> str:
    """'daily', 'monthly' or 'annual' from the median spacing of observations."""
    if len(valid_dates) < 3:
        return "monthly"
    gap = pd.Series(valid_dates).diff().dt.days.median()
    return "daily" if gap <= 5 else ("monthly" if gap <= 45 else "annual")


def rule_for(series: str, valid_dates: pd.DatetimeIndex, lags: dict | None = None) -> dict:
    lags = lags or load_lags()
    if series in (lags.get("series") or {}):
        return dict(lags["series"][series])
    freq = infer_freq(valid_dates)
    if freq == "annual":
        return {"rule": "year_lag", "lag_days": 180, "max_stale_days": 800}
    return dict(lags["defaults"][freq])


def available_at(valid_dates: pd.DatetimeIndex, rule: dict) -> pd.DatetimeIndex:
    """Vectorised available_at for one series under its publication rule (tz-aware IST)."""
    d = pd.DatetimeIndex(valid_dates).normalize()
    kind = rule["rule"]
    if kind == "next_day":
        ts = d + pd.Timedelta(days=1, hours=6)
    elif kind == "month_lag":
        month_end = d + pd.offsets.MonthEnd(0)
        ts = month_end + pd.Timedelta(days=int(rule["lag_days"]), hours=9)
    elif kind == "year_lag":
        year_end = d + pd.offsets.YearEnd(0)
        ts = year_end + pd.Timedelta(days=int(rule["lag_days"]), hours=9)
    elif kind == "same_day":
        hh, mm = str(rule.get("time", "12:00")).split(":")
        ts = d + pd.Timedelta(hours=int(hh), minutes=int(mm))
    else:
        raise ValueError(f"unknown PIT rule {kind!r}")
    return pd.DatetimeIndex(ts).tz_localize(IST)


def session_for_event(ts: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First session whose cutoff is at or after `ts` (an announcement at 15:31 belongs to the next session)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("event timestamps must be tz-aware (Asia/Kolkata)")
    cuts = cutoffs(sessions)
    i = cuts.searchsorted(ts, side="left")
    return None if i >= len(sessions) else pd.Timestamp(sessions[i])
