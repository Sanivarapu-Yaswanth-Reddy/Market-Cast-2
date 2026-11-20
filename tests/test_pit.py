"""P1.4 acceptance tests for the point-in-time rules (06 P1.4 a-d) and fold boundaries (05 §9 test 5). Data-free."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marketcast.data.global_align import asof_levels
from marketcast.eval import folds as F
from marketcast.pit.lags import available_at, cutoff, session_for_event
from marketcast.pit.store import PITStore
from marketcast.util import IST

SESSIONS = pd.bdate_range("2007-01-01", "2026-06-30")
LAGS = {"defaults": {"daily": {"rule": "next_day", "max_stale_days": 15}, "monthly": {"rule": "month_lag", "lag_days": 30, "max_stale_days": 120}},
        "series": {"CPI": {"rule": "month_lag", "lag_days": 15, "max_stale_days": 120}}}


def _long():
    d = pd.bdate_range("2023-01-02", "2023-03-31")
    daily = pd.DataFrame({"series": "US10Y", "valid_date": d, "value": np.arange(len(d), dtype=float),
                          "available_at": available_at(d, LAGS["defaults"]["daily"])})
    m = pd.to_datetime(["2023-01-01", "2023-02-01"])
    cpi = pd.DataFrame({"series": "CPI", "valid_date": m, "value": [100.0, 101.0], "available_at": available_at(m, LAGS["series"]["CPI"])})
    return pd.concat([daily, cpi], ignore_index=True)


def test_a_after_close_announcement_lands_on_next_session():
    s = pd.bdate_range("2023-01-02", periods=5)
    assert session_for_event(pd.Timestamp("2023-01-03 19:00", tz=IST), s) == s[2]
    assert session_for_event(pd.Timestamp("2023-01-03 15:30", tz=IST), s) == s[1]        # at the cutoff: same day
    assert session_for_event(pd.Timestamp("2023-01-03 15:31", tz=IST), s) == s[2]
    with pytest.raises(ValueError):
        session_for_event(pd.Timestamp("2023-01-03 19:00"), s)                           # naive timestamps are refused


def test_b_us_value_unavailable_same_india_date_available_next():
    long = _long()
    lv = asof_levels(long, pd.bdate_range("2023-01-02", "2023-01-06"), LAGS)
    # value dated Tue 3 Jan (index 1) must NOT be visible on India date 3 Jan, and must be on 4 Jan
    assert lv.loc["2023-01-03", "US10Y"] == 0.0 and lv.loc["2023-01-04", "US10Y"] == 1.0


def test_c_cpi_unavailable_before_month_end_plus_lag():
    lv = asof_levels(_long(), pd.bdate_range("2023-02-01", "2023-03-20"), LAGS)
    assert np.isnan(lv.loc["2023-02-14", "CPI"])              # January CPI: 31 Jan + 15 days = 15 Feb 09:00
    assert lv.loc["2023-02-15", "CPI"] == 100.0
    assert lv.loc["2023-03-14", "CPI"] == 100.0 and lv.loc["2023-03-15", "CPI"] == 101.0


def test_truncation_invariance_of_asof_levels():
    """05 §9 test 2 on the data layer: cutting the fact table at d cannot change rows of sessions <= d."""
    long, sess = _long(), pd.bdate_range("2023-01-02", "2023-03-31")
    full = asof_levels(long, sess, LAGS)
    cut = pd.Timestamp("2023-02-10")
    trunc = asof_levels(long[long["available_at"] <= cutoff(cut)], sess[sess <= cut], LAGS)
    # a series with no public value yet is simply absent from the truncated table: that equals "all NaN"
    pd.testing.assert_frame_equal(full.loc[:cut], trunc.reindex(columns=full.columns))


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 89 * 24 * 60))
def test_d_asof_never_returns_a_later_fact(minutes):
    long = _long()
    with PITStore(":memory:") as store:
        store.register("global_long", long)
        ts = pd.Timestamp("2023-01-01", tz=IST) + pd.Timedelta(minutes=minutes)
        got = store.asof("global_long", ts)
        assert (got["available_at"] <= ts).all()
        for s, g in long[long["available_at"] <= ts].groupby("series"):
            assert got.loc[got["series"] == s, "available_at"].iloc[0] == g["available_at"].max()
        with pytest.raises(ValueError):
            store.asof("global_long", ts.tz_localize(None))


def test_canary_future_fact_is_caught():
    """05 §9 test 3: a fact stamped after the cutoff must be invisible at the cutoff."""
    long = _long()
    leak = pd.DataFrame({"series": ["CANARY"], "valid_date": [pd.Timestamp("2023-02-01")], "value": [1.0],
                         "available_at": [pd.Timestamp("2023-02-02 09:15", tz=IST)]})
    lv = asof_levels(pd.concat([long, leak]), pd.bdate_range("2023-02-01", "2023-02-02"), LAGS)
    assert np.isnan(lv.loc["2023-02-01", "CANARY"]) and lv.loc["2023-02-02", "CANARY"] == 1.0


def test_fold_boundaries_purge_embargo_and_vault():
    cfg = F.default_folds()
    assert F.check_boundaries(SESSIONS, cfg) == []
    for name in [f"F{i}" for i in range(4, 10)]:
        s = F.split_rows(name, SESSIONS, cfg)
        assert s.train.max() + F.H < s.val.min() - F.E + 1 and s.val.max() + F.H < s.test.min() - F.E + 1
        assert SESSIONS[s.test.max() + F.H] < F.VAULT_START        # no target of any non-vault fold reads 2026
    with pytest.raises(PermissionError):
        F.split_rows("Vault", SESSIONS, cfg)
    with pytest.raises(PermissionError):
        F.assert_not_vault(pd.DatetimeIndex(["2026-01-02"]))
    F.assert_not_vault(pd.DatetimeIndex(["2025-12-31"]))
