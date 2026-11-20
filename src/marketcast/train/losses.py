"""Training losses (04 "Fair CRPS" and §L7). Torch mirrors of the NumPy metrics in eval/metrics.py; the two are
compared in tests/test_losses.py so the number optimised is the number reported.

    fair CRPS   (1/S) sum_m |x_m - y|  -  1/(2 S (S-1)) sum_{m != m'} |x_m - x_m'|        (sample axis FIRST here)
    total       sum_i mask_i w_i [CRPS(z_i) + alpha_v CRPS(v_i)] / sum_i mask_i w_i
                + w_mkt * CRPS(market z) + w_trend * cross-entropy(trend) + w_cons * consistency penalty
    QLIKE-consistent (deterministic model):  RV2 / exp(s) + s      (minimised at s = ln E[RV2])

All reductions ignore masked targets; a date with no valid target contributes 0 (and is counted, so the
caller can skip it).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def crps_fair(samples: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """samples [S, ...], obs [...] -> fair CRPS [...]. O(S log S) through the sorted-sample identity."""
    s = samples.shape[0]
    if s < 2:
        raise ValueError("fair CRPS needs at least 2 samples")
    term1 = (samples - obs.unsqueeze(0)).abs().mean(0)
    if s == 2:                                                   # the training case: one pair
        return term1 - 0.5 * (samples[0] - samples[1]).abs()
    xs, _ = samples.sort(dim=0)
    w = (2.0 * torch.arange(1, s + 1, device=samples.device, dtype=samples.dtype) - s - 1).view(-1, *([1] * (samples.dim() - 1)))
    return term1 - (xs * w).sum(0) / (s * (s - 1))


def masked_weighted_mean(loss: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """sum(mask * w * loss) / sum(mask * w); 0 when nothing is valid. NaN losses at masked positions are ignored."""
    w = mask.to(loss.dtype) if weight is None else mask.to(loss.dtype) * weight.to(loss.dtype)
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    return (loss * w).sum() / w.sum().clamp_min(1e-8)


def qlike_logvar(s: torch.Tensor, rv2: torch.Tensor) -> torch.Tensor:
    """RV2 / exp(s) + s, the QLIKE-consistent loss for a predicted log variance s."""
    return rv2 * torch.exp(-s) + s


def gen_loss(out: dict, tg: dict, alpha_v: float = 1.0, w_mkt: float = 0.1, w_trend: float = 0.1, w_cons: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Single-step generative loss for one date. out: model outputs with a sample axis; tg: batch.targets."""
    z, v = tg["z"][0], tg["v"][0]
    mask, w = tg["mask"][0], tg["weight"]
    l_z = masked_weighted_mean(crps_fair(out["z"].float(), torch.nan_to_num(z)), mask, w)
    l_v = masked_weighted_mean(crps_fair(out["v"].float(), torch.nan_to_num(v)), mask, w)
    loss = l_z + alpha_v * l_v
    parts = {"crps_z": l_z.detach(), "crps_v": l_v.detach()}
    if w_mkt and bool(tg["mkt_valid"]):
        l_m = crps_fair(out["mkt_z"].float(), tg["mkt_z"])
        loss = loss + w_mkt * l_m
        parts["crps_mkt"] = l_m.detach()
    if w_trend:
        l_t = trend_loss(out["trend"], tg["trend"])
        loss = loss + w_trend * l_t
        parts["trend_ce"] = l_t.detach()
    if w_cons and bool(tg["mkt_valid"]):
        # market-consistency penalty: the market head and the significance-weighted member aggregate should agree
        agg = (out["z"].float() * tg["sigma"].unsqueeze(0) * tg["sig_w"].unsqueeze(0)).sum(-1) / tg["mkt_sigma"]
        l_c = ((out["mkt_z"].float() - agg) ** 2).mean()
        loss = loss + w_cons * l_c
        parts["cons"] = l_c.detach()
    return loss, parts


def rollout_loss(paths: dict, tg: dict, alpha_v: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Unrolled loss (stage S3): mean over steps of the single-step CRPS, in decision-date sigma units.

    Simulated returns are compared with realised returns after both are divided by sigma_hat_t, so every step
    is on the same scale as step 1 (where the two definitions coincide).
    """
    k = paths["ell"].shape[1]
    sig = tg["sigma"].view(1, 1, -1)
    zs = paths["ell"] / sig                                          # [S, K, N]
    tz, tv, mask, w = tg["z"][:k], tg["v"][:k], tg["mask"][:k], tg["weight"]
    l_z = masked_weighted_mean(crps_fair(zs, torch.nan_to_num(tz)), mask, w.unsqueeze(0).expand_as(mask))
    l_v = masked_weighted_mean(crps_fair(paths["v"], torch.nan_to_num(tv)), mask, w.unsqueeze(0).expand_as(mask))
    return l_z + alpha_v * l_v, {"crps_z": l_z.detach(), "crps_v": l_v.detach()}


def det_loss(out: dict, tg: dict, w_trend: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Deterministic (Phase 3) loss: QLIKE-consistent on the 1/5/20-day realized variance + trend cross-entropy."""
    s = out["s"][0].float()                                          # [N, 3]
    rv2, mask = tg["rv2_h"], tg["rv2_mask"]                          # [N, 3]
    per = qlike_logvar(s, torch.nan_to_num(rv2, nan=1.0))
    # report the proper QLIKE (>= 0, 0 when exact) next to the optimised form
    ratio = torch.nan_to_num(rv2, nan=1.0) * torch.exp(-s)
    q = ratio - torch.log(ratio.clamp_min(1e-12)) - 1.0
    w = tg["weight"].unsqueeze(-1).expand_as(per)
    loss = masked_weighted_mean(per, mask, w)
    parts = {f"qlike_h{h}": masked_weighted_mean(q[:, i], mask[:, i], tg["weight"]).detach() for i, h in enumerate((1, 5, 20))}
    if w_trend:
        l_t = trend_loss(out["trend"], tg["trend"])
        loss = loss + w_trend * l_t
        parts["trend_ce"] = l_t.detach()
    return loss, parts


def trend_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over {down, sideways, up} for each horizon; label -1 = undefined (ignored)."""
    if not bool((labels >= 0).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.float(), labels, ignore_index=-1)
