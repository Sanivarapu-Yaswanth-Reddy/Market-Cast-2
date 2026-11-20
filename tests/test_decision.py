"""L9 portfolio engine, number validator, readouts and the forecast scorer on synthetic paths. Data-free."""
import numpy as np
import pandas as pd
import pytest

from marketcast.baselines.lgbm_quantile import inverse_cdf
from marketcast.decision.portfolio import portfolio_paths, risk_report, var_es
from marketcast.eval.forecast import Forecast, Truth, score_forecast
from marketcast.explain.validate_numbers import template_sentence, validate
from marketcast.readout import sectors as RS
from marketcast.readout import trend as RT

RNG = np.random.default_rng(0)
ELL = 0.015 * RNG.standard_normal((2000, 20, 4)) + 0.01 * RNG.standard_normal((2000, 20, 1))     # common factor
KEYS = np.array(["A", "B", "C", "D"])


def test_portfolio_identities():
    w = np.array([0.5, 0.3, 0.2, 0.0])
    r, contrib = portfolio_paths(ELL, w)
    assert np.allclose(contrib.sum(axis=1), r[:, -1])                       # holdings' contributions add up to the portfolio
    one = np.zeros((1, 2, 4))
    one[0, :, 0] = np.log(1.1)
    assert portfolio_paths(one, np.array([1.0, 0, 0, 0]))[0][0, -1] == pytest.approx(0.21)
    var, es, tail = var_es(r[:, -1], 0.95)
    assert es >= var > 0 and tail.mean() == pytest.approx(0.05, abs=0.002)


def test_risk_report_contract_and_validator():
    mkt = RT.market_paths_from_members(ELL, np.ones(4))
    rep = risk_report(ELL, KEYS, {"A": 0.6, "B": 0.4, "ZZZ": 0.1}, as_of="2025-01-01", model_version="test", data_snapshot="sha256:x",
                      market_ell=mkt, trend=RT.trend_probabilities(mkt, 0.012)[20])
    assert rep["flags"]["unsupported_assets"] == ["ZZZ"]                    # never silently dropped
    assert sum(h["weight"] for h in rep["portfolio"]["holdings"]) == pytest.approx(1.0)
    assert sum(c["share_of_es95"] for c in rep["risk_contributions"]) == pytest.approx(1.0, abs=1e-4)
    assert rep["var"]["99"] <= rep["var"]["95"] <= 0 and rep["es"]["95"] <= rep["var"]["95"]
    assert rep["outcome_range"]["p05"] < rep["outcome_range"]["p50"] < rep["outcome_range"]["p95"]
    assert sum(rep["market_trend"].values()) == pytest.approx(1.0)
    ok, bad = validate(template_sentence(rep), rep)
    assert ok, bad
    ok, bad = validate("Your portfolio could lose 42.7% next month.", rep)
    assert not ok and bad == ["42.7%"]                                      # an invented number is rejected
    with pytest.raises(ValueError):
        risk_report(ELL, KEYS, {"ZZZ": 1.0}, as_of="x", model_version="x", data_snapshot="x")


def test_trend_drawdown_stress_readouts():
    mkt = RT.market_paths_from_members(ELL, np.array([1.0, 1.0, 1.0, 1.0]))
    t = RT.trend_probabilities(mkt, 0.012)
    assert all(abs(sum(v.values()) - 1) < 1e-9 for v in t.values())
    dd = RT.drawdown_probabilities(mkt)
    assert dd[20]["10pct"] <= dd[20]["5pct"] and dd[5]["5pct"] <= dd[20]["5pct"]         # nested events
    s = RT.stress_probability(mkt, np.full(40, 0.012 ** 2), 0.012)
    assert 0 <= s[5] <= 1 and RT.stress_probability(mkt, np.full(40, 0.012 ** 2), 1e-6)[20] == 1.0


def test_sector_readouts_and_shock_pairing():
    sec = np.array([0, 0, 1, 2])
    out = RS.outlook(ELL, np.ones(4), sec, ["s0", "s1", "s2"], h=20, k=1)
    assert out["p_worst3"].sum() == pytest.approx(1.0) and set(out["sector"]) == {"s0", "s1", "s2"}
    shock = ELL.copy()
    shock[:, 0, 2] -= 0.05                                                  # sector 1 hit on day 1, same noise otherwise
    d = RS.shock_delta(ELL, shock, np.ones(4), sec, ["s0", "s1", "s2"], h=5).set_index("sector")
    assert d.loc["s1", "delta_abn"] < -0.02 and d.loc["s0", "delta_abn"] > 0 and d["se"].max() < 0.002


def test_scorer_rewards_the_true_distribution():
    m, n = 300, 1500
    sigma = np.full(n, 0.02)
    truth_ell = sigma * RNG.standard_normal((20, n))
    tr = Truth(truth_ell, np.full((20, n), 0.02 ** 2), np.ones((20, n), bool), sigma, np.ones(n), np.zeros(n, bool))
    keys = np.array([f"K{i}" for i in range(n)])

    def fc(scale):
        return Forecast("t", "F", pd.Timestamp("2020-01-01"), keys, ell=(scale * sigma * RNG.standard_normal((m, 20, n))).astype(np.float32),
                        v=np.full((m, 20, n), np.log(0.02 * scale), dtype=np.float32))

    good, _ = score_forecast(fc(1.0), tr)
    wide, joint = score_forecast(fc(3.0), tr)
    g, w = good.groupby("h").mean(numeric_only=True), wide.groupby("h").mean(numeric_only=True)
    assert (g["crps"] < w["crps"]).all() and (g["qlike"] < w["qlike"]).all()
    assert abs(g.loc[1, "in90"] - 0.9) < 0.04 and w.loc[1, "in90"] > 0.99      # calibrated vs over-dispersed
    assert g.loc[1, "qlike"] < 0.01 and "energy_h20" in joint
    tr.mask[3, :10] = False                                                 # one bad day voids the 5- and 20-day windows only
    part, _ = score_forecast(fc(1.0), tr)
    assert (part["h"] == 1).sum() == n and (part["h"] == 5).sum() == n - 10


def test_inverse_cdf_reproduces_quantiles_and_has_tails():
    q = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    vals = np.array([[-2.3, -0.67, 0.0, 0.67, 2.3]])
    u = np.array([[0.01], [0.5], [0.99], [0.001], [0.999]])
    x = inverse_cdf(q, vals, u)[:, 0]
    assert np.allclose(x[:3], [-2.3, 0.0, 2.3]) and x[3] < -2.3 and x[4] > 2.3
