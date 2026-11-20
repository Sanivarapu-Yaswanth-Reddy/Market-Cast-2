"""P2.5  Baseline runner: fit per fold, forecast every scored date, score with the shared scorer, log the run.

    python -m marketcast baselines run --model B0 --folds fast            # F2, F4, F6, F8, test split
    python -m marketcast baselines run --model B2 --folds all --every 5   # GARCH is slow: every 5th date

IMPLEMENTED    B0 (EWMA random walk, Student-t), B0g (Gaussian), B1 (filtered historical simulation),
               B2 (GJR-GARCH skew-t) / B2g11, B5 (LightGBM quantiles); trend baselines BT1-BT2 in trend.py
NOT YET        B3 DCC-GARCH, B4 factor model (needs own PIT factors), B6 zero-shot foundation models,
               BT3 Markov-switching, BS1-BS2 sector baselines. They are listed in reports/progress_log.md.

Scores go to reports/scores/<model>/<fold>_<split>.parquet; every invocation is appended to runs/registry.jsonl
(the run count feeds multiple-testing corrections, 00 rule 9). The vault fold is refused (00 rule 6).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from marketcast.baselines.common import Baseline, Ctx
from marketcast.eval.folds import FAST_FOLDS, load_folds
from marketcast.eval.forecast import save_scores, score_forecast, truth_for
from marketcast.train.logging import RunLogger
from marketcast.util import get_logger, snapshot_hash

log = get_logger("baselines")


def make(model: str) -> Baseline:
    if model in ("B0", "B0g"):
        from marketcast.baselines.ewma import EwmaRW

        return EwmaRW(gaussian=model == "B0g")
    if model == "B1":
        from marketcast.baselines.hist_sim import HistSim

        return HistSim()
    if model in ("B2", "B2g11"):
        from marketcast.baselines.garch import Garch

        return Garch(gjr=model == "B2")
    if model == "B5":
        from marketcast.baselines.lgbm_quantile import LgbmQuantile

        return LgbmQuantile()
    raise ValueError(f"baseline {model!r} is not implemented (available: B0, B0g, B1, B2, B2g11, B5)")


def run_fold(model: str, fold: str, split: str = "test", paths: int = 256, every: int = 1, max_dates: int | None = None) -> pd.DataFrame:
    """Fit and score one baseline on one fold. Returns the per-(date, stock, horizon) score table."""
    from marketcast import DATA

    t0 = time.time()
    ctx = Ctx.open(fold)
    bl = make(model)
    bl.fit(ctx)
    rows = getattr(ctx.split, split)[::every]
    if max_dates:
        rows = rows[:max_dates]
    rl = RunLogger(f"{model}-{fold}-{split}", {"baseline": model, "paths": paths, "every": every, "split": split, "phase": "P2"}, fold, 0,
                   snapshot_hash([DATA / "targets" / "meta.json"]))
    per_stock, per_date = [], []
    for i, row in enumerate(rows):
        rng = np.random.default_rng([abs(hash(model)) % (2**31), int(row)])      # same paths on every re-run
        fc = bl.forecast(int(row), paths, rng)
        cols = pd.Index(ctx.panel.keys).get_indexer(fc.keys)
        df, joint = score_forecast(fc, truth_for(ctx.panel, int(row), cols), rng)
        per_stock.append(df)
        per_date.append({"date": fc.date, **joint})
        if (i + 1) % 25 == 0:
            log.info("%s %s %s: %d/%d dates, %.0fs", model, fold, split, i + 1, len(rows), time.time() - t0)
    save_scores(model, fold, split, per_stock, per_date)
    allv = pd.concat(per_stock, ignore_index=True) if per_stock else pd.DataFrame()
    summ = {}
    if len(allv):
        if allv.isna().all(axis=None):
            raise RuntimeError(f"{model} produced only NaN scores on {fold}")
        g = allv.groupby("h")
        summ = {f"{c}_h{h}": float(v) for c in ("crps", "qlike", "in90") if c in allv for h, v in g[c].mean().items()}
    extra = bl.stability() if hasattr(bl, "stability") else {}
    rl.finish("finished", n_dates=int(len(rows)), minutes=round((time.time() - t0) / 60, 2), **summ, **extra)
    log.info("%s %s %s done: %d dates in %.1f min; %s %s", model, fold, split, len(rows), (time.time() - t0) / 60,
             {k: round(v, 4) for k, v in summ.items()}, extra)
    return allv


def main(model: str, folds: str = "fast", split: str = "test", paths: int = 256, every: int = 1) -> int:
    names = FAST_FOLDS if folds == "fast" else ([f for f in load_folds()["folds"] if f != "Vault"] if folds == "all" else folds.split(","))
    if "Vault" in names:
        raise PermissionError("the vault fold is sealed (00 rule 6); baselines are scored on F1-F9 only")
    for fold in names:
        run_fold(model, fold, split, paths, every)
    return 0
