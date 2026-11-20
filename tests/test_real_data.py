"""Integration checks on the REAL caches (data/targets, data/features, data/graph). Skipped when they are not built,
so the data-free suite still runs anywhere. These are the leakage tests of 05 §9 that need data:

    1  PIT / index discipline of a batch: inputs end at t, targets start at t+1
    6  held-out isolation: no held-out company in any training batch; they do appear at evaluation
    +  rollout consistency: the torch price-channel update reproduces the numpy feature builder on real history
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marketcast import DATA                                              # noqa: E402

pytestmark = pytest.mark.skipif(not ((DATA / "features" / "F4" / "norm.json").exists() and (DATA / "graph" / "corr.npz").exists()),
                                reason="real feature/graph caches not built on this machine")


@pytest.fixture(scope="module")
def store():
    from marketcast.train.data_loader import Store

    return Store("F4")


@pytest.fixture(scope="module")
def row(store):
    from marketcast.eval.folds import split_rows

    return int(split_rows("F4", store.dates).train[-50])


def test_training_batch_has_no_heldout_and_eval_batch_does(store, row):
    tr = store.make(row, train=True)
    ev = store.make(row, train=False)
    assert not store.heldout[tr.meta["cols"]].any()
    assert store.heldout[ev.meta["cols"]].any() and ev.n_stock > tr.n_stock
    # inserting held-out stocks must not change a held-in stock's correlation neighbourhood
    def corr_in(b):                                                      # noqa: E306
        s, d = b.edges["corr"][0].numpy(), b.edges["corr"][1].numpy()
        c = b.meta["cols"]
        return {(int(c[i]), int(c[j])) for i, j in zip(s, d)}
    held_in = set(tr.meta["cols"].tolist())
    assert corr_in(tr) == {(a, b_) for a, b_ in corr_in(ev) if b_ in held_in}


def test_inputs_end_at_t_and_targets_start_after(store, row):
    b = store.make(row, train=True, k_steps=3)
    cols, p = b.meta["cols"], store.panel
    # the last input day is exactly session t: its z channel equals the panel's z[t] after normalisation
    z_idx = 0
    want = store.snorm.apply(np.asarray(store.feats[row, cols], dtype=np.float32))[:, z_idx]
    assert np.allclose(b.x_s[:, -1, z_idx].numpy(), want, atol=1e-5)
    for k in range(3):                                                   # target k is session t + 1 + k
        ref = np.asarray(p["ell"][row + 1 + k, cols]) / np.asarray(p["sigma"][row, cols])
        m = b.targets["mask"][k].numpy()
        assert np.allclose(b.targets["z"][k].numpy()[m], ref[m], rtol=1e-5)
    assert abs(float(b.targets["weight"].mean()) - 1.0) < 1e-4 and torch.isfinite(b.x_s).all()


def test_graph_shape_and_tree_weights(store, row):
    b = store.make(row, train=True)
    n = b.n_stock
    src, dst, lw, _ = b.edges["up0"]
    assert len(src) == n and int(b.node_type[b.market_idx]) == 4
    share = torch.zeros(int(b.node_type.numel())).index_add_(0, dst, torch.exp(lw))
    assert torch.allclose(share[dst], torch.ones(n), atol=1e-4)           # children's significance sums to 1 per parent
    deg = torch.bincount(b.edges["corr"][1], minlength=n)
    assert int(deg.max()) <= 10


def test_dashboard_engine_report_and_vault_guard():
    """The dashboard's numbers come from the same decision/readout code and respect the vault."""
    from marketcast.app.engine import Engine

    eng = Engine()
    comp = eng.companies("2024-06-28")["companies"]
    hold = {c["key"]: 1.0 for c in comp[:5]}
    hold["NOT_A_KEY"] = 1.0
    d = eng.report("B1", "2024-06-28", hold, horizon=10, n_paths=64)
    r = d["report"]
    assert r["horizon_days"] == 10 and r["n_paths"] == 64 and r["flags"]["unsupported_assets"] == ["NOT_A_KEY"]
    assert len(d["fan"]["p50"]) == 11 and d["fan"]["p05"][-1] <= d["fan"]["p50"][-1] <= d["fan"]["p95"][-1]
    assert abs(sum(d["market"]["trend"].values()) - 1) < 1e-9 and len(d["realised"]) == 11
    assert abs(sum(h["weight"] for h in r["portfolio"]["holdings"]) - 1) < 1e-4
    with pytest.raises(PermissionError):
        eng.report("B1", "2026-03-02", {}, 20, 64)
    late = eng.report("B1", "2025-12-29", {}, 20, 64)          # outcome only up to the last 2025 session
    assert late["info"]["realised_days"] < 20


def test_rollout_price_channels_match_numpy_builder(store, row):
    """Replay the REAL next day through the rollout update and compare with the stored feature row of that day."""
    from marketcast.features.stock import PRICE_CHANNELS
    from marketcast.model.rollout import RolloutState, advance, price_channels_step

    b = store.make(row, train=False, with_state=True)
    cols, p = b.meta["cols"], store.panel
    nxt = row + 1
    ell1, v1 = np.asarray(p["ell"][nxt, cols], np.float32), np.asarray(p["v"][nxt, cols], np.float32)
    # stocks with a clean, long, unclipped history on both days (the documented approximations do not apply to them)
    hist_ok = np.isfinite(np.asarray(p["ell"][row - 320: nxt + 1][:, cols])).all(axis=0)
    ok = hist_ok & np.isfinite(ell1) & np.isfinite(v1) & (np.abs(ell1) < 0.15) & (np.asarray(p["sigma"][nxt, cols]) > 1.01 * store.floor[nxt])
    assert ok.sum() > 100
    st = RolloutState.from_batch(b, 1)
    st.has_fo = torch.ones_like(st.has_fo)                                # no circuit clipping in this replay
    z1 = torch.from_numpy(ell1 / np.asarray(p["sigma"][row, cols], np.float32)).unsqueeze(0)
    st2, ell_hat, sig_prev, _ = advance(st, torch.nan_to_num(z1), torch.nan_to_num(torch.from_numpy(v1)).unsqueeze(0))
    got = price_channels_step(st2, sig_prev, torch.from_numpy(v1).unsqueeze(0))[0].numpy()
    want = np.asarray(store.feats[nxt, cols], dtype=np.float32)[:, : len(PRICE_CHANNELS)]
    assert np.allclose(st2.sigma[0].numpy()[ok], np.asarray(p["sigma"][nxt, cols])[ok], rtol=2e-3)       # EWMA update agrees
    skip = {"gap", "range", "close_loc"}                                 # replaced by proxies inside a rollout (documented)
    for j, name in enumerate(PRICE_CHANNELS):
        if name in skip:
            continue
        a, w = got[ok, j], want[ok, j]
        fin = np.isfinite(a) & np.isfinite(w)
        err = np.abs(a[fin] - w[fin]) / (1.0 + np.abs(w[fin]))
        assert fin.mean() > 0.95 and np.quantile(err, 0.99) < 0.02, (name, float(np.quantile(err, 0.99)))
