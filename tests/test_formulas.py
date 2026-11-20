"""Hand-computed checks of the formulas in 03_FEATURES_AND_TARGETS.md and of the panel-builder rules. Data-free."""
import numpy as np
import pandas as pd
import pytest

from marketcast.data import masks as M
from marketcast.data.build_panel import assign_identity, finalize_year, infer_corporate_actions
from marketcast.data.universe import select_members
from marketcast.targets import returns as R
from marketcast.targets.labels import drawdown_from_paths, forward_drawdown, trend_from_paths, trend_labels


def test_log_return_and_rv2_by_hand():
    assert R.log_return(np.array([110.0]), np.array([100.0]))[0] == pytest.approx(np.log(1.1))
    # open 102, high 105, low 99, close 104, pc 100
    o, co, hl = np.log(1.02), np.log(104 / 102), np.log(105 / 99)
    want = o ** 2 + 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    got = R.rv2_gk(np.array([102.0]), np.array([105.0]), np.array([99.0]), np.array([104.0]), np.array([100.0]))[0]
    assert got == pytest.approx(want)
    # no range recorded: overnight^2 + close-to-open^2 (here both zero except the gap)
    flat = R.rv2_gk(np.array([101.0]), np.array([101.0]), np.array([101.0]), np.array([101.0]), np.array([100.0]))[0]
    assert flat == pytest.approx(np.log(1.01) ** 2)
    assert R.log_rv(np.array([0.0]))[0] == pytest.approx(0.5 * np.log(1e-6))          # floor = 0.1% daily
    assert np.isnan(R.rv2_gk(np.array([np.nan]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))[0])


def test_ewma_init_recursion_and_nan_hold():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, 80)
    x[30] = np.nan                                           # a non-trading day
    s2, n = R.ewma_var(x, cap=None)
    valid = np.flatnonzero(np.isfinite(x))
    t0 = valid[19]                                           # day the 20th valid return arrives
    assert np.isnan(s2[t0 - 1]) and s2[t0] == pytest.approx(np.var(x[valid[:20]], ddof=1))
    t1 = valid[20]
    assert s2[t1] == pytest.approx(0.94 * s2[t0] + 0.06 * x[t1] ** 2)
    assert s2[30] == s2[29] and n[30] == n[29]               # NaN leaves the estimate unchanged
    # the update cap bounds the effect of one absurd return
    y = x.copy()
    y[60] = -1.5
    capped, _ = R.ewma_var(y)
    assert capped[60] <= 0.94 * capped[59] + 0.06 * R.EWMA_UPDATE_CAP ** 2 + 1e-12


def test_sigma_floor_validity_and_scaled_return_alignment():
    s2 = np.full((70, 3), 1e-8)
    s2[:, 1] = 0.02 ** 2
    n = np.tile(np.arange(1, 71)[:, None], (1, 3))
    sig, floor = R.sigma_hat(s2, n)
    assert np.isnan(sig[58, 0]) and np.isfinite(sig[59, 0])                      # valid from the 60th observation
    assert floor[-1] == pytest.approx(0.004) and sig[-1, 0] == pytest.approx(0.004) and sig[-1, 1] == pytest.approx(0.02)
    ell = np.zeros((70, 3))
    ell[65, 1] = 0.01
    z = R.scaled_return(ell, sig)
    assert z[65, 1] == pytest.approx(0.01 / sig[64, 1])                          # divides by YESTERDAY's sigma


def test_forward_sum_is_strictly_forward_and_complete():
    x = np.arange(10, dtype=float)
    f = R.forward_sum(x, 3)
    assert f[0] == 1 + 2 + 3 and np.isnan(f[-1]) and np.isnan(f[7])
    x[5] = np.nan
    f = R.forward_sum(x, 3)
    assert np.isnan(f[2]) and np.isnan(f[4]) and f[5] == 6 + 7 + 8               # any missing day voids the window


def test_trend_and_drawdown_labels():
    ell = np.array([0.0, 0.02, 0.0, -0.03, 0.001, 0.0, 0.0])
    lab = trend_labels(ell, np.full(7, 0.01), 1)
    assert list(lab[:4]) == [2, 1, 0, 1] and lab[-1] == -1                       # +-0.5 sigma band; last row has no future
    p = trend_from_paths(np.array([-0.05, 0.0, 0.001, 0.05]), 0.01, 4)
    assert p.sum() == pytest.approx(1.0) and list(p) == [0.25, 0.5, 0.25]
    dd = drawdown_from_paths(np.array([[0.01, -0.08, 0.10], [0.01, 0.01, 0.01]]))
    assert dd[0] == pytest.approx(np.expm1(-0.07)) and dd[1] == 0.0
    assert forward_drawdown(np.array([0.0, -0.1, 0.2, 0.0]), 2)[0] == pytest.approx(np.expm1(-0.1))


def test_corporate_action_inference():
    #            2:1 split   crash, no ratio  normal   penny stock (ignored)
    ref = np.array([1000.0, 100.0, 100.0, 2.0])
    open_ = np.array([501.0, 60.0, 101.0, 1.0])
    close = np.array([505.0, 55.0, 102.0, 1.0])
    fac, ok, unexpl = infer_corporate_actions(open_, close, ref)
    assert list(fac) == [0.5, 1.0, 1.0, 1.0] and list(ok) == [True, False, False, False]
    assert list(unexpl) == [False, True, False, True]


def test_identity_and_cross_year_carry():
    df = pd.DataFrame({"symbol": ["AAA", "AAA", "FUND", "OLD"], "isin": ["INE111A01012", "INE111A01020", "INF222B01011", None],
                       "date": pd.to_datetime(["2020-01-01"] * 4)})
    out = assign_identity(df, {"OLD": "INE999Z01015"}, coexist=set())
    assert list(out["key"]) == ["INE111A01", "INE111A01", "INF222B01011", "INE999Z01"]      # split-proof; funds keep the full ISIN
    assert list(out["is_fund"]) == [False, False, True, False]
    assert assign_identity(df, {}, coexist={"INE111A01"})["key"].iloc[0] == "INE111A01012"   # side-by-side share classes stay apart
    assert assign_identity(df, {}, set())["key"].iloc[3] == "SYM:OLD"

    def year(dates, closes, opens):
        n = len(dates)
        return pd.DataFrame({"date": pd.to_datetime(dates), "key": "K", "company_id": "K", "isin": "I", "symbol": "S", "series": "EQ",
                             "is_fund": False, "open": opens, "high": closes, "low": closes, "close": closes, "last": closes,
                             "prev_close": closes, "volume": 1.0, "turnover": 1.0, "trades": 1.0, "mto_qty": 1.0, "deliv_qty": 1.0,
                             "deliv_pct": 50.0}, index=range(n))

    y1, carry = finalize_year(year(["2019-12-30", "2019-12-31"], [100.0, 200.0], [100.0, 200.0]), None, {})
    y2, carry = finalize_year(year(["2020-01-01", "2020-01-02"], [101.0, 102.0], [100.0, 101.0]), carry, {})
    assert np.isnan(y1["ret_1d"].iloc[0])                                        # first ever row: no return
    assert y2["adj_factor"].iloc[0] == 0.5 and y2["corp_action_flag"].iloc[0]     # 200 -> 100 at the year boundary = split
    assert y2["ret_1d"].iloc[0] == pytest.approx(101 / 100 - 1)                   # return is adjusted for it
    assert list(y2["days_listed"]) == [2, 3]


def test_masks():
    close, high, low, pc = (np.array([[v]]) for v in (105.0, 105.0, 100.0, 100.0))
    up, dn = M.circuit_hits(close, high, low, pc, np.array([[False]]))
    assert up[0, 0] and not dn[0, 0]
    assert not M.circuit_hits(close, high, low, pc, np.array([[True]]))[0][0, 0]          # F&O stocks have no fixed band
    assert not M.circuit_hits(np.array([[104.0]]), high, low, pc, np.array([[False]]))[0][0, 0]
    traded = np.zeros((30, 2), dtype=bool)
    traded[:5, 0] = True
    traded[:, 1] = True
    term = M.terminal_rows(traded)
    assert term[4, 0] and term.sum() == 1


def test_universe_buffer_rule():
    rank = np.arange(1, 11, dtype=float)                      # 10 names, universe size 5 -> stay <= 6, enter <= 4
    eligible = np.ones(10, dtype=bool)
    incumbent = np.zeros(10, dtype=bool)
    incumbent[[0, 1, 2, 5, 6]] = True                         # ranks 1, 2, 3, 6, 7
    m = select_members(rank, eligible, incumbent, 5)
    assert m.sum() == 5 and m[5] and not m[6]                 # rank 6 stays (buffer), rank 7 leaves
    assert m[3] and not m[4]                                  # rank 4 enters by right; rank 5 does not displace the buffered incumbent
