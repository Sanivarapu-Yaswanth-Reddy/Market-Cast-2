"""P3.4 / P4  Training loop (04 "Training recipe").

    python -m marketcast train --config model/p4_gen.yaml --fold F4 --seed 0            # on the GPU machine
    python -m marketcast train --config model/p3_det.yaml --fold F4 --smoke             # code-path check, CPU, ~1 min

RECIPE
    AdamW (weight decay 0.05, no decay on biases / norms / embeddings), peak lr 3e-4, linear warm-up then cosine
    decay to 3e-6, gradient clip 1.0, bfloat16 autocast on CUDA, EMA of weights 0.999 (the EMA is what gets
    evaluated and checkpointed as "best"). One optimiser step = `dates_per_step` decision dates x `n_samples`
    noise scenarios, accumulated date by date so only one market graph is on the GPU at a time (8 GB card).
    Early stopping on the validation metric with a patience of 5 evaluations.

STAGES (selected by the config, see configs/model/)
    det           Phase 3: log-variance heads, QLIKE-consistent loss
    gen, k = 1    S1: single-step fair CRPS on z and v (+ market head, + trend cross-entropy)
    gen, k > 1    S3: unrolled fine-tuning through k rollout steps (lower lr, warm-start with train.init_from)

SAFETY RAILS
    * 00 rule 8: no neural training before gate G2 is signed off. A non-smoke run refuses to start until
      dev_plan/state/G2.done exists, unless train.allow_before_g2=true is passed explicitly (and then says so
      in the run registry).
    * rows come from eval.folds.split_rows: purged, embargoed, vault-checked; held-out stocks are not in any
      training batch (data_loader).
    * every run is registered (train/logging.py) with config hash, git commit and data snapshot hash.
    * checkpoints are written atomically; --resume continues from ckpt_last.pt with optimiser, EMA and step.
"""
from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from marketcast import DATA, ROOT
from marketcast.eval.folds import split_rows
from marketcast.model.marketcast import MarketCast, ModelConfig, count_parameters
from marketcast.model.rollout import rollout
from marketcast.train import losses as LS
from marketcast.train.data_loader import Store, StoreDataset, loader
from marketcast.train.logging import RUNS, RunLogger
from marketcast.util import deep_update, get_logger, load_yaml, set_seed, snapshot_hash

log = get_logger("train")
SMOKE_MODEL = {"d": 32, "d_e": 32, "enc_layers": 1, "enc_heads": 2, "enc_ff": 64, "proc_layers": 2, "proc_heads": 4, "proc_ff": 64, "cond_dim": 32}
SMOKE_TRAIN = {"max_steps": 3, "eval_every": 3, "eval_dates": 2, "eval_samples": 4, "workers": 0, "warmup_steps": 1, "max_nodes": 48,
               "amp": False, "log_every": 1, "ckpt_every": 3, "dates_per_step": 1}


# --------------------------------------------------------------------------- config
def resolve_config(path: str, overrides: list[str], smoke: bool) -> dict:
    cfg = deep_update(load_yaml("train/default.yaml"), load_yaml(path))
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        import yaml

        v = yaml.safe_load(val)
        if isinstance(v, str):               # YAML 1.1 reads "5e-4" as text (it wants "5.0e-4"); accept both
            try:
                v = float(v)
            except ValueError:
                pass
        node[parts[-1]] = v
    if smoke:
        cfg["model"] = {**cfg.get("model", {}), **SMOKE_MODEL}
        cfg["train"] = {**cfg["train"], **SMOKE_TRAIN}
    return cfg


def build_model(cfg: dict, dims: dict) -> MarketCast:
    allowed = {f.name for f in fields(ModelConfig)}
    unknown = set(cfg.get("model", {})) - allowed
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")
    return MarketCast(ModelConfig(**dims, **cfg.get("model", {})))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() < 2 or "emb" in n or "norm" in n or n.endswith(("lam", "feat", "query", "pos")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    prog = min(1.0, (step - tc["warmup_steps"]) / max(1, tc["max_steps"] - tc["warmup_steps"]))
    return tc["lr_min"] + 0.5 * (tc["lr"] - tc["lr_min"]) * (1.0 + math.cos(math.pi * prog))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))            # fast start so early evaluations are meaningful
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model: torch.nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        return backup


def save_ckpt(path: Path, **obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    for k in range(8):                                              # OneDrive may hold the old file briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == 7:
                raise
            time.sleep(0.5 * (k + 1))


# --------------------------------------------------------------------------- one date
def date_loss(model: MarketCast, b, tc: dict, norm: tuple[torch.Tensor, torch.Tensor], n_samples: int, gen: torch.Generator | None):
    """Loss of one decision date under the configured stage. Returns (loss, parts) or None if nothing is trainable."""
    if b.n_stock < 8:
        return None
    mode, k = model.cfg.mode, int(tc.get("k_steps", 1))
    dev = b.x_s.device
    if mode == "det":
        if not bool(b.targets["rv2_mask"].any()):
            return None
        return LS.det_loss(model(b, None, 1), b.targets, tc["w_trend"])
    if not bool(b.targets["mask"][0].any()):
        return None
    if k <= 1:
        eps = torch.randn(n_samples, model.cfg.noise_dim, device=dev, generator=gen)
        return LS.gen_loss(model(b, eps), b.targets, tc["alpha_v"], tc["w_mkt"], tc["w_trend"], tc["w_cons"])
    eps = torch.randn(k, n_samples, model.cfg.noise_dim, device=dev, generator=gen)
    return LS.rollout_loss(rollout(model, b, eps, norm[0], norm[1]), b.targets, tc["alpha_v"])


@torch.no_grad()
def evaluate(model: MarketCast, dl, tc: dict, norm, device, amp_ctx) -> dict:
    """Validation metric (lower is better): CRPS(z) + alpha_v CRPS(v) for gen, mean QLIKE over horizons for det."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)          # same noise every evaluation: comparable numbers
    tot, n = {}, 0
    eval_tc = {**tc, "k_steps": 1}
    for b in dl:
        b = b.to(device)
        with amp_ctx():
            res = date_loss(model, b, eval_tc, norm, int(tc["eval_samples"]), gen)
        if res is None:
            continue
        _, parts = res
        for k_, v in parts.items():
            tot[k_] = tot.get(k_, 0.0) + float(v)
        n += 1
    model.train()
    out = {k_: v / max(n, 1) for k_, v in tot.items()}
    if model.cfg.mode == "det":
        out["metric"] = float(np.mean([out.get(f"qlike_h{h}", np.nan) for h in (1, 5, 20)]))
    else:
        out["metric"] = out.get("crps_z", np.nan) + tc["alpha_v"] * out.get("crps_v", np.nan)
    out["n_dates"] = n
    return out


# --------------------------------------------------------------------------- main
def main(config: str, fold: str, seed: int, resume: bool = False, smoke: bool = False, overrides: list[str] | None = None) -> int:
    cfg = resolve_config(config, overrides or [], smoke)
    tc = cfg["train"]
    if not smoke and not tc.get("allow_before_g2") and not (ROOT / "dev_plan" / "state" / "G2.done").exists():
        log.error("Gate G2 (harness and baselines) is not signed off: 00 rule 8 forbids neural training before it. "
                  "Use --smoke to test the code path, or pass --set train.allow_before_g2=true to override on purpose.")
        return 2
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)     # noqa: E731
    run_name = f"{cfg.get('name', Path(config).stem)}-{fold}-s{seed}" + ("-smoke" if smoke else "")
    run_dir = RUNS / run_name

    store = Store(fold, tuple(tc["edges"]), tc["universe"])
    split = split_rows(fold, store.dates)
    first_ok = 60 + 252                                             # rows with a full history window
    tr_rows = split.train[split.train >= first_ok]
    va_rows = split.val[:: max(1, len(split.val) // int(tc["eval_dates"]))][: int(tc["eval_dates"])]
    if tc.get("overfit_dates"):
        # 06 P3.4 acceptance check: "overfits a 20-date subset". Train AND validate on the same few dates; a
        # healthy model drives the loss far below the persistence level. Diagnostic only, never a result.
        tr_rows = tr_rows[np.linspace(0, len(tr_rows) - 1, int(tc["overfit_dates"])).astype(int)]
        va_rows = tr_rows
    if len(tr_rows) == 0 or len(va_rows) == 0:
        raise RuntimeError(f"fold {fold} has no usable train/val decision dates")
    k_steps = int(tc.get("k_steps", 1))
    mk = dict(use_edges=tuple(tc["edges"]), universe=tc["universe"], max_nodes=tc.get("max_nodes"))
    ds_tr = StoreDataset(fold, tr_rows, train=True, k_steps=k_steps, with_state=k_steps > 1, **mk)
    ds_va = StoreDataset(fold, va_rows, train=True, k_steps=1, **mk)       # validation also excludes held-out stocks
    dl_va = loader(ds_va, shuffle=False, workers=0)
    model = build_model(cfg, store.dims).to(device)
    norm = (torch.from_numpy(store.snorm.median).to(device), torch.from_numpy(store.snorm.scale).to(device))
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"], betas=(0.9, 0.95))
    ema = EMA(model, tc["ema"])
    step, best, bad = 0, float("inf"), 0
    ckpt_last, ckpt_best = run_dir / "ckpt_last.pt", run_dir / "ckpt_best.pt"
    if resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        opt.load_state_dict(ck["opt"])
        step, best, bad = ck["step"], ck["best"], ck["bad"]
        log.info("resumed %s at step %d (best %.5f)", run_name, step, best)
    elif tc.get("init_from"):
        ck = torch.load(ROOT / tc["init_from"], map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck.get("ema", ck["model"]), strict=False)
        log.info("warm start from %s: %d missing, %d unexpected keys (heads differ between det and gen by design)",
                 tc["init_from"], len(missing), len(unexpected))
        ema = EMA(model, tc["ema"])
    data_hash = snapshot_hash([DATA / "targets" / "meta.json", DATA / "features" / "meta.json", DATA / "features" / fold / "norm.json",
                               DATA / "graph" / "tree.json", DATA / "graph" / "corr.npz"])
    rl = RunLogger(run_name, cfg, fold, seed, data_hash, smoke, resume)
    npar = count_parameters(model)
    log.info("run %s | device %s | amp %s | params %s | train dates %d (%s..%s) | val dates %d | stage %s k=%d", run_name, device, use_amp, npar,
             len(tr_rows), store.dates[tr_rows[0]].date(), store.dates[tr_rows[-1]].date(), len(va_rows), model.cfg.mode, k_steps)
    rl.log(step, event="start", **{f"params_{k}": v for k, v in npar.items()}, g2_override=bool(tc.get("allow_before_g2")))

    def ckpt(path: Path, state: dict) -> None:
        save_ckpt(path, model=state, ema=ema.shadow, opt=opt.state_dict(), step=step, best=best, bad=bad, cfg=cfg,
                  model_cfg=model.cfg.to_dict(), fold=fold, seed=seed, norm_median=store.snorm.median, norm_scale=store.snorm.scale,
                  stock_channels=store.snorm.names)

    gen = torch.Generator(device=device).manual_seed(seed + 1000 * step)
    model.train()
    status, t0, acc = "finished", time.time(), {}
    try:
        epoch = 0
        while step < tc["max_steps"]:
            it = iter(loader(ds_tr, shuffle=True, seed=seed + epoch + step, workers=int(tc["workers"])))
            epoch += 1
            exhausted = False
            while step < tc["max_steps"] and not exhausted:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, tc)
                opt.zero_grad(set_to_none=True)
                used = 0
                for _ in range(int(tc["dates_per_step"])):
                    b = next(it, None)
                    if b is None:
                        exhausted = True
                        break
                    b = b.to(device)
                    with amp_ctx():
                        res = date_loss(model, b, tc, norm, int(tc["n_samples"]), gen)
                    if res is None:
                        continue
                    loss, parts = res
                    if not torch.isfinite(loss):
                        log.warning("non-finite loss on %s: date skipped", b.meta["date"])
                        continue
                    (loss / int(tc["dates_per_step"])).backward()
                    used += 1
                    acc["loss"] = acc.get("loss", 0.0) + float(loss.detach())
                    for k_, v in parts.items():
                        acc[k_] = acc.get(k_, 0.0) + float(v)
                    acc["n"] = acc.get("n", 0) + 1
                if not used:
                    continue
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()
                step += 1
                ema.update(model, step)
                if step % int(tc["log_every"]) == 0 and acc.get("n"):
                    n = acc.pop("n")
                    avg = {k_: v / n for k_, v in acc.items()}
                    acc = {}
                    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                    rl.log(step, lr=lr_at(step, tc), grad_norm=float(gn), gpu_gb=mem, **{f"train_{k_}": v for k_, v in avg.items()})
                    log.info("step %6d  loss %.4f  lr %.2e  |g| %.2f  %.1f GB  %.0fs", step, avg["loss"], lr_at(step, tc), float(gn), mem, time.time() - t0)
                if step % int(tc["ckpt_every"]) == 0:
                    ckpt(ckpt_last, model.state_dict())
                if step % int(tc["eval_every"]) == 0 or step == tc["max_steps"]:
                    backup = ema.copy_to(model)
                    ev = evaluate(model, dl_va, tc, norm, device, amp_ctx)
                    rl.log(step, **{f"val_{k_}": v for k_, v in ev.items()})
                    improved = ev["metric"] < best - 1e-6
                    if improved:
                        best, bad = ev["metric"], 0
                        ckpt(ckpt_best, copy.deepcopy(model.state_dict()))       # EMA weights are in the model right now
                    else:
                        bad += 1
                    model.load_state_dict(backup, strict=False)
                    log.info("eval step %d: %s%s (best %.5f, patience %d/%d)", step, {k_: round(v, 5) for k_, v in ev.items()},
                             " *" if improved else "", best, bad, tc["patience"])
                    if bad >= int(tc["patience"]):
                        status = "early_stopped"
                        raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        status = "interrupted"
        log.warning("interrupted: writing the last checkpoint so --resume can continue")
    ckpt(ckpt_last, model.state_dict())
    rl.finish(status, best_val_metric=best, steps=step, minutes=round((time.time() - t0) / 60, 1), params=npar["total"])
    log.info("run %s %s at step %d; best val metric %.5f; artefacts in %s", run_name, status, step, best, run_dir)
    return 0
