"""P3.3 model tests on a synthetic graph: shapes, determinism at init, equivariance, masking, the significance
prior, losses and the rollout. Data-free; skipped when torch is not installed (data-only environments)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marketcast.eval import metrics as MT                               # noqa: E402
from marketcast.features.stock import CHANNELS, PRICE_CHANNELS          # noqa: E402
from marketcast.model.fabric_attn import EdgeAttention                  # noqa: E402
from marketcast.model.marketcast import MarketCast, count_parameters    # noqa: E402
from marketcast.model.rollout import N_PRICE, rollout                   # noqa: E402
from marketcast.train import losses as LS                               # noqa: E402
from synth import H, make_batch, tiny_config                            # noqa: E402


def _model(mode="gen", seed=0):
    torch.manual_seed(seed)
    return MarketCast(tiny_config(mode)).eval()


def _wake(model):
    """Zero-initialised branches make an untrained model trivially invariant; randomise them for structural tests."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                p.normal_(std=0.05)
    return model


def test_shapes_gen_and_det():
    b = make_batch(12)
    out = _model("gen")(b, torch.randn(3, 16))
    assert out["z"].shape == (3, 12) and out["v"].shape == (3, 12) and out["mkt_z"].shape == (3,) and out["trend"].shape == (2, 3)
    out = _model("det")(b, None, 1)
    assert out["s"].shape == (1, 12, 3)


def test_untrained_model_is_persistence_forecast():
    """Zero-init heads: z = 0, v = ln sigma, variance = h * sigma^2, and noise has no effect yet."""
    b = make_batch(10)
    out = _model("gen")(b, torch.randn(4, 16))
    assert torch.allclose(out["z"], torch.zeros(4, 10), atol=1e-6)
    assert torch.allclose(out["v"], b.ln_sigma.expand(4, 10), atol=1e-6)
    s = _model("det")(b, None, 1)["s"][0]
    want = 2 * b.ln_sigma.unsqueeze(-1) + torch.log(torch.tensor([1.0, 5.0, 20.0]))
    assert torch.allclose(s, want, atol=1e-5)


def test_noise_changes_output_and_same_noise_repeats():
    m, b = _wake(_model("gen")), make_batch(10)
    e = torch.randn(2, 16)
    a, c = m(b, e)["z"], m(b, e)["z"]
    assert torch.allclose(a, c)
    assert not torch.allclose(a[0], a[1])


def test_permutation_equivariance_over_stock_order():
    m, b = _wake(_model("gen")), make_batch(12)
    perm = torch.randperm(12, generator=torch.Generator().manual_seed(3))
    inv = torch.argsort(perm)
    n = b.n_stock

    def remap(idx):                                      # node ids: stocks are permuted, the other blocks stay
        return torch.where(idx < n, inv[idx.clamp(max=n - 1)], idx)

    edges = {}
    for k, (s, d, lw, ft) in b.edges.items():
        edges[k] = (remap(s), remap(d), lw, ft)
    import dataclasses
    bp = dataclasses.replace(b, x_s=b.x_s[perm], mask_s=b.mask_s[perm], ln_sigma=b.ln_sigma[perm], edges=edges,
                             child_w={**b.child_w, "up0": b.child_w["up0"]})
    # up0 edges keep their order (src = old stock id remapped), so child weights stay aligned with the edge list
    e = torch.randn(2, 16)
    assert torch.allclose(m(b, e)["z"][:, perm], m(bp, e)["z"], atol=1e-4)


def test_isolated_stock_does_not_affect_others():
    """A stock with no edges at all cannot influence any other node."""
    m, b = _wake(_model("gen")), make_batch(12)
    k = 5
    edges = {name: tuple(t[(s != k) & (d != k)] for t in (s, d, lw, ft)) for name, (s, d, lw, ft) in b.edges.items()}
    import dataclasses
    cw = {**b.child_w, "up0": b.child_w["up0"][torch.arange(12) != k]}
    b1 = dataclasses.replace(b, edges=edges, child_w=cw)
    x2 = b.x_s.clone()
    x2[k] += 5.0
    b2 = dataclasses.replace(b1, x_s=x2)
    e = torch.randn(2, 16)
    z1, z2 = m(b1, e)["z"], m(b2, e)["z"]
    others = torch.arange(12) != k
    assert torch.allclose(z1[:, others], z2[:, others], atol=1e-5)
    assert not torch.allclose(z1[:, k], z2[:, k])


def test_significance_prior_shifts_attention():
    torch.manual_seed(0)
    att = EdgeAttention(16, 4, lambda_init=1.0, hub_cap=False)
    h = torch.zeros(1, 4, 16)                                   # identical nodes: only the prior can break the tie
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([3, 3, 3])
    lw = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    _, alpha = att(h, src, dst, lw, torch.zeros(3), return_alpha=True)
    assert torch.allclose(alpha[0, :, 0], torch.tensor([0.7, 0.2, 0.1]), atol=1e-5)
    att_cap = EdgeAttention(16, 4, lambda_init=1.0, hub_cap=True)
    _, a2 = att_cap(h, src, dst, lw, torch.zeros(3), return_alpha=True)
    assert a2[0, 0, 0] < 0.7 and torch.allclose(a2.sum(1), torch.ones(1, 4), atol=1e-5)      # hub cap then renormalise


def test_fair_crps_matches_numpy_metric():
    x, y = torch.randn(7, 5, 3), torch.randn(5, 3)
    ref = MT.crps_fair(x.permute(1, 2, 0).numpy(), y.numpy())
    assert np.allclose(LS.crps_fair(x, y).numpy(), ref, atol=1e-5)
    assert np.allclose(LS.crps_fair(x[:2], y).numpy(), MT.crps_fair(x[:2].permute(1, 2, 0).numpy(), y.numpy()), atol=1e-5)


def test_losses_backward_and_masking():
    for mode in ("gen", "det"):
        m, b = _model(mode).train(), make_batch(10)
        out = m(b, torch.randn(2, 16)) if mode == "gen" else m(b, None, 1)
        loss, parts = LS.gen_loss(out, b.targets, w_cons=0.05) if mode == "gen" else LS.det_loss(out, b.targets)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # a NaN target under a False mask must not poison the loss
    b = make_batch(10)
    b.targets["z"][0, 0] = float("nan")
    b.targets["mask"][0, 0] = False
    loss, _ = LS.gen_loss(_model("gen")(b, torch.randn(2, 16)), b.targets)
    assert torch.isfinite(loss)


def test_rollout_shapes_step1_and_gradient():
    m, b = _wake(_model("gen")), make_batch(10)
    med, sc = torch.zeros(len(CHANNELS)), torch.ones(len(CHANNELS))
    eps = torch.randn(4, 3, 16)
    out = rollout(m, b, eps, med, sc)
    assert out["ell"].shape == (3, 4, 10) and out["v"].shape == (3, 4, 10) and torch.isfinite(out["ell"]).all()
    one = m(b, eps[0])                                           # step 1 of a rollout is the single-step forecast
    assert torch.allclose(out["z"][:, 0], one["z"], atol=1e-5)
    m.train()
    out = rollout(m, b, torch.randn(3, 2, 16), med, sc)
    loss, _ = LS.rollout_loss(out, b.targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert N_PRICE == len(PRICE_CHANNELS) and H >= 4


def test_parameter_budget_of_default_config():
    from marketcast.model.marketcast import ModelConfig

    n = count_parameters(MarketCast(ModelConfig(f_stock=len(CHANNELS), f_global=9, f_forcing=18)))
    assert 5e6 < n["total"] < 20e6, n          # plan: 11-15 M at 8-10 layers
