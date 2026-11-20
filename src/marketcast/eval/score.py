"""Scoring of trained MarketCast runs (06 P4.3) through the shared scorer in eval/forecast.py.

    python -m marketcast eval score --run p4-gen-F4-s0 --split val --paths 16 --every 5

For each scored decision date the EMA ("best") checkpoint rolls out `paths` scenarios for 20 sessions over the
EVALUATION graph (MC500 members including held-out companies, inserted inductively), and the paths are scored
against what happened. Output: reports/scores/<run>/<fold>_<split>.parquet with one row per
(date, stock, horizon), flagged held-in / held-out. The deterministic Phase 3 model has no paths: its
1/5/20-day variance forecasts are scored with QLIKE only.

The test split of the Vault fold cannot be scored from here without allow_vault (00 rule 6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marketcast.eval.folds import split_rows
from marketcast.eval.forecast import Forecast, save_scores, score_forecast, truth_for
from marketcast.model.marketcast import MarketCast, ModelConfig
from marketcast.model.rollout import rollout
from marketcast.train.data_loader import HORIZON, Store
from marketcast.train.logging import RUNS
from marketcast.util import get_logger

log = get_logger("eval.score")


def load_run(run: str, device: torch.device, which: str = "best") -> tuple[MarketCast, dict]:
    """Rebuild the model of a run from its checkpoint (falls back to ckpt_last if no best exists yet)."""
    d = Path(run) if Path(run).exists() else RUNS / run
    p = d / f"ckpt_{which}.pt"
    if not p.exists():
        p = d / "ckpt_last.pt"
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint in {d}")
    ck = torch.load(p, map_location=device, weights_only=False)
    model = MarketCast(ModelConfig(**ck["model_cfg"])).to(device)
    # ckpt_best holds EMA weights in "model"; ckpt_last holds raw weights with the EMA alongside
    state = ck["model"] if p.name == "ckpt_best.pt" else {**ck["model"], **ck["ema"]}
    model.load_state_dict(state)
    model.eval()
    return model, ck


@torch.no_grad()
def forecast_date(model: MarketCast, store: Store, row: int, paths: int, device, norm, seed: int = 0, chunk: int = 16,
                  model_id: str = "MC") -> tuple[Forecast, np.ndarray]:
    """One decision date -> Forecast (paths for gen, variance marginals for det). Returns (forecast, stock cols)."""
    b = store.make(row, train=False, k_steps=1, with_state=True).to(device)
    cols = b.meta["cols"]
    date = store.dates[row]
    if model.cfg.mode == "det":
        s = model(b, None, 1)["s"][0].float().cpu().numpy()                   # [N, 3] log variance for h = 1, 5, 20
        return Forecast(model_id, store.fold, date, store.panel.keys[cols], var_h={h: np.exp(s[:, i]) for i, h in enumerate((1, 5, 20))}), cols
    gen = torch.Generator(device=device).manual_seed(seed * 100003 + row)      # reproducible: same seed + date = same paths
    ells, vs = [], []
    for start in range(0, paths, chunk):
        m = min(chunk, paths - start)
        eps = torch.randn(HORIZON, m, model.cfg.noise_dim, device=device, generator=gen)
        out = rollout(model, b, eps, norm[0], norm[1])
        ells.append(out["ell"].cpu().numpy().astype(np.float32))
        vs.append(out["v"].cpu().numpy().astype(np.float32))
    return Forecast(model_id, store.fold, date, store.panel.keys[cols], ell=np.concatenate(ells), v=np.concatenate(vs)), cols


def score_run(run: str, split: str = "val", paths: int = 16, every: int = 5, allow_vault: bool = False) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run(run, device)
    fold, tc = ck["fold"], ck["cfg"]["train"]
    store = Store(fold, tuple(tc["edges"]), tc["universe"])
    rows = getattr(split_rows(fold, store.dates, allow_vault=allow_vault), split)[::every]
    norm = (torch.from_numpy(ck["norm_median"]).to(device), torch.from_numpy(ck["norm_scale"]).to(device))
    per_stock, per_date = [], []
    for i, row in enumerate(rows):
        fc, cols = forecast_date(model, store, int(row), paths, device, norm, seed=ck["seed"], model_id=Path(run).name)
        df, joint = score_forecast(fc, truth_for(store.panel, int(row), cols))
        per_stock.append(df)
        per_date.append({"date": fc.date, **joint})
        if (i + 1) % 10 == 0:
            log.info("scored %d/%d dates", i + 1, len(rows))
    a, _ = save_scores(Path(run).name, fold, split, per_stock, per_date)
    allv = pd.concat(per_stock, ignore_index=True) if per_stock else pd.DataFrame()
    if len(allv):
        cols_ = [c for c in ("crps", "qlike", "in90") if c in allv]
        log.info("%s %s %s: %d dates\n%s", Path(run).name, fold, split, len(rows), allv.groupby(["h", "heldout"])[cols_].mean().round(4).to_string())
    log.info("scores written to %s", a)
    return 0
