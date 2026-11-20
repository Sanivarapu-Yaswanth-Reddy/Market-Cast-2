"""P2.4 acceptance tests (06_BUILD_PLAN.md): reference libraries, closed forms, independent re-derivations."""
import math

import numpy as np
import pytest
from scipy import stats

from marketcast.eval import metrics as M

rng = np.random.default_rng(42)


def _brute_fair_crps(x, y):
    m = len(x)
    t1 = np.abs(x - y).mean()
    t2 = sum(abs(a - b) for a in x for b in x) / (2 * m * (m - 1))
    return t1 - t2


def test_fair_crps_matches_brute_force():
    x = rng.normal(size=(7, 13))
    y = rng.normal(size=7)
    got = M.crps_fair(x, y)
    want = np.array([_brute_fair_crps(x[i], y[i]) for i in range(7)])
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_fair_crps_matches_scoringrules():
    sr = pytest.importorskip("scoringrules")
    x = rng.normal(size=(50, 32))
    y = rng.normal(size=50)
    ref = np.asarray(sr.crps_ensemble(y, x, estimator="fair"))
    np.testing.assert_allclose(M.crps_fair(x, y), ref, rtol=1e-9)


def test_nrg_crps_matches_properscoring():
    ps = pytest.importorskip("properscoring")
    x = rng.normal(size=(40, 25))
    y = rng.normal(size=40)
    np.testing.assert_allclose(M.crps_ensemble_nrg(x, y), ps.crps_ensemble(y, x), rtol=1e-9)


def test_fair_crps_vs_normal_closed_form():
    # M = 10,000 N(0,1) members, within 1% of the closed form (plan acceptance). One ensemble has about 1.5%
    # Monte Carlo error at some y, so the fair estimator is averaged over 20 independent ensembles (it is unbiased).
    x = np.random.default_rng(7).normal(size=(20, 10_000))
    for y in (-2.0, -0.5, 0.0, 0.7, 1.9):
        got = float(M.crps_fair(x, np.full(20, y)).mean())
        want = float(M.crps_normal(0.0, 1.0, y))
        assert abs(got - want) / want < 0.01, (y, got, want)


def test_energy_score_fair_reduces_to_crps_in_1d():
    x = rng.normal(size=(5, 20, 1))
    y = rng.normal(size=(5, 1))
    np.testing.assert_allclose(M.energy_score_fair(x, y), M.crps_fair(x[..., 0], y[..., 0]), rtol=1e-12)


def test_energy_score_fair_matches_scoringrules():
    sr = pytest.importorskip("scoringrules")
    x = rng.normal(size=(6, 30, 4))
    y = rng.normal(size=(6, 4))
    try:
        fn = getattr(sr, "es_ensemble", None) or sr.energy_score
        ref = np.asarray(fn(y, x, estimator="fair"))
    except TypeError:
        pytest.skip("this scoringrules version has no fair energy-score estimator")
    np.testing.assert_allclose(M.energy_score_fair(x, y), ref, rtol=1e-9)


def test_variogram_brute_force():
    x = rng.normal(size=(15, 4))
    y = rng.normal(size=4)
    want = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            want += (abs(y[i] - y[j]) ** 0.5 - np.mean(np.abs(x[:, i] - x[:, j]) ** 0.5)) ** 2
    assert math.isclose(float(M.variogram_score(x, y)), want, rel_tol=1e-12)


def test_variogram_detects_wrong_correlation():
    # truth strongly correlated; a forecast with the right marginals but independent members should score worse
    g = np.random.default_rng(3)
    cov = np.array([[1, 0.9], [0.9, 1]])
    ys = g.multivariate_normal([0, 0], cov, size=4000)
    good = g.multivariate_normal([0, 0], cov, size=(4000, 50))
    bad = g.normal(size=(4000, 50, 2))
    assert M.variogram_score(good, ys).mean() < M.variogram_score(bad, ys).mean()


def test_qlike_zero_at_truth_and_positive_elsewhere():
    assert M.qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(0.0)
    assert (M.qlike(np.array([2.0, 2.0]), np.array([1.0, 4.0])) > 0).all()


def test_coverage_and_pit_calibrated_ensemble():
    g = np.random.default_rng(11)
    x = g.normal(size=(20_000, 200))
    y = g.normal(size=20_000)
    cov = M.interval_coverage(x, y)
    for lv, c in cov.items():
        assert abs(c - lv) < 0.015, (lv, c)
    assert M.pit_max_bin_deviation(M.pit_values(x, y)) < 0.01


def test_kupiec_independent_derivation():
    # T = 250, VaR99 with 6 violations (a standard textbook setting)
    v = np.zeros(250, int)
    v[[10, 50, 90, 130, 170, 210]] = 1
    r = M.kupiec_pof(v, 0.01)
    t, x, p = 250, 6, 0.01
    lr = -2 * ((t - x) * math.log(1 - p) + x * math.log(p)) + 2 * ((t - x) * math.log(1 - x / t) + x * math.log(x / t))
    assert r.stat == pytest.approx(lr, rel=1e-12)
    assert r.stat == pytest.approx(3.5554, abs=1e-3)      # hand value: rejects at 10%, not at 5%
    assert 0.05 < r.pvalue < 0.10
    assert M.kupiec_pof(np.zeros(100), 0.01).stat == pytest.approx(-2 * 100 * math.log(0.99))


def test_christoffersen_clustered_vs_spread():
    spread = np.zeros(500, int)
    spread[::50] = 1                         # 10 hits, never consecutive
    clustered = np.zeros(500, int)
    clustered[200:210] = 1                   # same 10 hits, all consecutive
    ind_s, cc_s = M.christoffersen(spread, 0.02)
    ind_c, cc_c = M.christoffersen(clustered, 0.02)
    assert ind_c.stat > 20 and ind_c.pvalue < 1e-4
    assert ind_c.stat > ind_s.stat
    assert cc_c.stat == pytest.approx(M.kupiec_pof(clustered, 0.02).stat + ind_c.stat)
    assert ind_c.detail == {"n00": 488, "n01": 1, "n10": 1, "n11": 9}


def test_brier_log_score_and_skill():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    out = np.array([0, 2])
    np.testing.assert_allclose(M.brier_multiclass(probs, out), [0.09 + 0.04 + 0.01, 0.01 + 0.01 + 0.04])
    np.testing.assert_allclose(M.log_score(probs, out), [-math.log(0.7), -math.log(0.8)])
    assert M.brier_skill(probs, out, [1 / 3, 1 / 3, 1 / 3]) > 0
    assert M.log_score(np.array([[0.0, 1.0]]), np.array([0]))[0] == pytest.approx(-math.log(1e-4))


def test_reliability_perfect_forecaster():
    g = np.random.default_rng(5)
    p = g.uniform(size=50_000)
    o = (g.uniform(size=50_000) < p).astype(float)
    mf, of, n, lo, hi = M.reliability_curve(p, o, n_boot=50)
    assert np.nanmax(np.abs(mf - of)) < 0.03
    assert (lo <= of + 1e-9).all() and (hi >= of - 1e-9).all()


def test_ranking_metrics():
    pred = np.arange(10.0)
    assert M.spearman_rho(pred, pred * 2) == pytest.approx(1.0)
    assert M.top_bottom_hit_rate(pred, pred) == 1.0
    assert M.top_bottom_hit_rate(pred, -pred) == 0.0


def test_diebold_mariano_matches_statsmodels_hac():
    sm = pytest.importorskip("statsmodels.api")
    g = np.random.default_rng(9)
    a = g.normal(1.0, 1, 600) + 0.3 * np.convolve(g.normal(size=604), np.ones(5) / 5, "valid")
    b = g.normal(1.1, 1, 600)
    lags = 1 + 5
    res = M.diebold_mariano(a, b, h=1)
    ref = sm.OLS(a - b, np.ones(600)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    assert res.stat == pytest.approx(float(ref.tvalues[0]), rel=1e-9)
    assert res.pvalue == pytest.approx(float(ref.pvalues[0]), rel=1e-6)


def test_diebold_mariano_bootstrap_detects_real_difference():
    g = np.random.default_rng(1)
    a = g.normal(0.0, 1, 1000)
    res_same = M.diebold_mariano(a, a + g.normal(0, 0.01, 1000), method="bootstrap", n_boot=500)
    res_diff = M.diebold_mariano(a, a + 0.3 + g.normal(0, 0.1, 1000), method="bootstrap", n_boot=500)
    assert res_diff.pvalue < 0.01
    assert res_same.pvalue > 0.01


def test_deflated_sharpe():
    g = np.random.default_rng(2)
    r = g.normal(0.001, 0.01, 1000)                       # per-period SR about 0.1
    one = M.deflated_sharpe_ratio(r, n_trials=1, var_sr_trials=0.0)
    many = M.deflated_sharpe_ratio(r, n_trials=1000, var_sr_trials=0.001)
    assert one.detail["sr0"] == 0.0 and many.detail["sr0"] > 0
    assert many.stat < one.stat                            # more trials -> less confidence
    # SR0 formula against a direct evaluation
    g_ = M.EULER_GAMMA
    sr0 = math.sqrt(0.001) * ((1 - g_) * stats.norm.ppf(1 - 1 / 1000) + g_ * stats.norm.ppf(1 - 1 / (1000 * math.e)))
    assert many.detail["sr0"] == pytest.approx(sr0)
