"""Dashboard back end: turns a (source, date, portfolio) question into the numbers the page draws.

No web code here (that is app/server.py); every public method returns plain JSON-serialisable dicts, so the
same engine can be driven from a notebook or a test.

SOURCES OF SCENARIOS
    "B1"            filtered historical simulation (baselines/hist_sim.py): always available, no training needed
    "<run name>"    any generative MarketCast run under runs/ that has a checkpoint (rolled out on this machine)
A source produces sample paths for ONE decision date; everything shown (portfolio fan, VaR/ES, drawdown odds,
market trend odds, sector exposure) is read from those same paths (decision/, readout/), so the panels cannot
contradict each other.

GUARDS
    * Vault (00 rule 6): decision dates from 2026-01-01 are refused, and realised outcomes are only drawn up
      to the last 2025 session. The dashboard is a research viewer, not the live forecast (that is Phase 6).
    * Internal use only (00 rule 11): the server binds to 127.0.0.1; nothing here is a buy or sell call.
    * A run is "in sample" for dates inside its own training window; the response says so, because numbers on
      dates the model was fitted to are not evidence of skill.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

from marketcast import DATA, REPORTS
from marketcast.decision.portfolio import portfolio_paths, risk_report
from marketcast.eval.folds import VAULT_START, load_folds, split_rows
from marketcast.eval.forecast import Forecast
from marketcast.util import save_npz
from marketcast.explain.validate_numbers import template_sentence
from marketcast.readout import sectors as RS
from marketcast.readout import trend as RT
from marketcast.targets.build import ModelPanel
from marketcast.train.logging import RUNS, read_metrics
from marketcast.util import get_logger

log = get_logger("app")
FIRST_DATE = pd.Timestamp("2017-01-01")          # first walk-forward test year (05 §1)
FAN_Q = (0.05, 0.25, 0.5, 0.75, 0.95)
CACHE_SIZE = 6
DIAGNOSTIC = ("overfit", "bench")      # run-name prefixes of diagnostic runs: logged, but not offered in the dashboard


def _f(x, nd: int = 6):
    """Rounded float for JSON; NaN/inf become None (JSON has no NaN)."""
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


class Engine:
    def __init__(self):
        from marketcast.data.security_master import load_master
        from marketcast.graph.fabric import load_tree

        self.panel = ModelPanel()
        self.dates = self.panel.dates
        m = load_master().set_index("key").reindex(self.panel.keys)
        self.symbol = m["symbol"].fillna("").to_numpy()
        self.name = m["company_name"].fillna("").to_numpy()
        tree = load_tree()
        self.sector_names = tree["sectors"]
        self.sector_of = np.asarray(tree["industry_sector"])[np.asarray(tree["stock_industry"])]
        self.vault_row = int(self.dates.searchsorted(VAULT_START, "left"))
        self.folds = load_folds()
        self.market = self.panel.market()
        self._lock = threading.Lock()                 # one heavy computation at a time
        self._cache: OrderedDict = OrderedDict()
        self._b1: dict = {}
        self._runs: dict = {}

    # ------------------------------------------------------------------ catalogue
    def sources(self) -> list[dict]:
        out = [{"id": "B1", "label": "Baseline B1: filtered historical simulation", "kind": "baseline", "first_date": str(FIRST_DATE.date())}]
        for d in sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []:
            cfg_p = d / "config.json"
            if not cfg_p.exists() or not ((d / "ckpt_best.pt").exists() or (d / "ckpt_last.pt").exists()):
                continue
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            if cfg.get("smoke") or d.name.startswith(DIAGNOSTIC) or (cfg["config"].get("model") or {}).get("mode") != "gen":
                continue
            fold = cfg["fold"]
            train_end = pd.Timestamp(self.folds["folds"][fold]["train"][1])
            out.append({"id": d.name, "label": f"MarketCast run {d.name}", "kind": "model", "fold": fold,
                        "train_end": str(train_end.date()), "first_date": str(FIRST_DATE.date()),
                        "has_best": (d / "ckpt_best.pt").exists()})
        return out

    def meta(self) -> dict:
        ok = (self.dates >= FIRST_DATE) & (np.arange(len(self.dates)) < self.vault_row)
        d = self.dates[ok]
        return {"first_date": str(d[0].date()), "last_date": str(d[-1].date()), "sources": self.sources(),
                "vault_start": str(VAULT_START.date()), "horizons": [5, 10, 20], "paths": [64, 128, 256, 512]}

    def snap_row(self, date: str) -> int:
        """Latest session on or before `date`, inside the allowed (non-vault) range."""
        d = pd.Timestamp(date)
        if d >= VAULT_START:
            raise PermissionError(f"{d.date()} is inside the sealed 2026 holdout (rule 6); choose a date up to {self.dates[self.vault_row - 1].date()}")
        row = int(self.dates.searchsorted(d, "right")) - 1
        if row < 0 or self.dates[row] < FIRST_DATE:
            raise ValueError(f"no session on or before {d.date()} in the dashboard range (from {FIRST_DATE.date()})")
        return row

    def _fold_for(self, row: int) -> str:
        """Walk-forward fold whose TEST year contains the date (so a baseline never uses that year for fitting)."""
        return f"F{min(max(self.dates[row].year - 2016, 1), 9)}"

    def companies(self, date: str) -> dict:
        row = self.snap_row(date)
        cols = np.flatnonzero(np.asarray(self.panel["member500"][row]) & np.isfinite(np.asarray(self.panel["sigma"][row]))
                              & np.isfinite(np.asarray(self.panel["z"][row])))
        w = np.nan_to_num(np.asarray(self.panel["med_turn60"][row, cols], dtype=float))
        order = np.argsort(-w)
        return {"date": str(self.dates[row].date()),
                "companies": [{"key": str(self.panel.keys[cols[i]]), "symbol": str(self.symbol[cols[i]]), "name": str(self.name[cols[i]]),
                               "sector": self.sector_names[self.sector_of[cols[i]]], "share": _f(w[i] / w.sum())} for i in order]}

    # ------------------------------------------------------------------ scenario paths
    def _paths(self, source: str, row: int, n_paths: int) -> dict:
        key = (source, row, n_paths)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if source == "B1":
            from marketcast.baselines.common import Ctx
            from marketcast.baselines.hist_sim import HistSim

            fold = self._fold_for(row)
            if fold not in self._b1:
                bl = HistSim()
                bl.fit(Ctx.open(fold))
                self._b1[fold] = bl
            fc = self._b1[fold].forecast(row, n_paths, np.random.default_rng([1, row]))
            info = {"source": "B1", "fold": fold, "in_sample": False}
        else:
            import torch

            from marketcast.eval.score import forecast_date, load_run
            from marketcast.train.data_loader import Store

            if source not in self._runs or self._runs[source]["mtime"] != self._ckpt_mtime(source):
                model, ck = load_run(source, torch.device("cpu"))
                tc = ck["cfg"]["train"]
                self._runs[source] = {"model": model, "ck": ck, "store": Store(ck["fold"], tuple(tc["edges"]), tc["universe"]),
                                      "mtime": self._ckpt_mtime(source)}
                for k in [k for k in self._cache if k[0] == source]:          # a newer checkpoint invalidates old paths
                    del self._cache[k]
            r = self._runs[source]
            ck = r["ck"]
            # Rolling a model out on a CPU is slow (minutes), so finished path sets are kept on disk, keyed by the
            # checkpoint they came from: asking the same (run, date, scenarios) again is instant, and a newer
            # checkpoint never serves stale paths.
            disk = DATA / "forecasts" / "app_cache" / source / f"{row}_{n_paths}_{int(r['mtime'])}.npz"
            if disk.exists():
                z = np.load(disk, allow_pickle=False)
                fc = Forecast(source, ck["fold"], self.dates[row], z["keys"], ell=z["ell"].astype(np.float32), v=z["v"].astype(np.float32))
            else:
                norm = (torch.from_numpy(ck["norm_median"]), torch.from_numpy(ck["norm_scale"]))
                fc, _ = forecast_date(r["model"], r["store"], row, n_paths, torch.device("cpu"), norm, seed=ck["seed"], model_id=source)
                save_npz(disk, keys=np.asarray(fc.keys, dtype=str), ell=fc.ell.astype(np.float16), v=fc.v.astype(np.float16))
            train_rows = split_rows(ck["fold"], self.dates).train
            info = {"source": source, "fold": ck["fold"], "in_sample": bool(len(train_rows) and row <= train_rows.max()),
                    "checkpoint_step": int(ck["step"]), "run_finished": (RUNS / source / "summary.json").exists()}
        cols = pd.Index(self.panel.keys).get_indexer(fc.keys)
        out = {"ell": fc.ell, "v": fc.v, "cols": cols, "info": info}
        self._cache[key] = out
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return out

    @staticmethod
    def _ckpt_mtime(run: str) -> float:
        d = RUNS / run
        p = d / "ckpt_best.pt" if (d / "ckpt_best.pt").exists() else d / "ckpt_last.pt"
        return p.stat().st_mtime

    # ------------------------------------------------------------------ the main question
    def report(self, source: str, date: str, holdings: dict[str, float], horizon: int = 20, n_paths: int = 256,
               value_inr: float | None = None) -> dict:
        row = self.snap_row(date)
        horizon = int(min(max(horizon, 1), 20))
        with self._lock:
            sc = self._paths(source, row, int(n_paths))
        cols, ell = sc["cols"], sc["ell"][:, :horizon]
        keys = self.panel.keys[cols]
        sig_w = np.nan_to_num(np.asarray(self.panel["med_turn60"][row, cols], dtype=float)) + 1.0
        mkt = RT.market_paths_from_members(ell, sig_w)
        m_sigma = float(self.market["m_sigma"].iloc[row])
        if not holdings:                                            # empty portfolio = the significance-weighted market
            holdings = {str(k): float(w) for k, w in zip(keys, sig_w / sig_w.sum())}
            label = "Market portfolio (all MC500 members, weighted by traded value)"
        else:
            label = f"{len(holdings)} holdings"
        trend = RT.trend_probabilities(mkt, m_sigma, horizons=(horizon,))[horizon]
        sec = RS.outlook(ell, sig_w, self.sector_of[cols], self.sector_names, h=horizon, k=3)
        rep = risk_report(ell, keys, holdings, as_of=str(self.dates[row].date()), model_version=source, data_snapshot="local",
                          value_inr=value_inr, market_ell=mkt, trend=trend,
                          sectors=[{"sector": r.sector, "p_worst3": _f(r.p_worst3, 4)} for r in sec.head(5).itertuples()])
        # ---- fan: quantiles of the cumulative portfolio return for each day, plus what actually happened
        idx = {str(k): i for i, k in enumerate(keys)}
        w = np.zeros(len(keys))
        for h_ in rep["portfolio"]["holdings"]:
            w[idx[h_["key"]]] = h_["weight"]
        r_paths, contrib = portfolio_paths(ell, w)
        fan = {f"p{int(q * 100):02d}": [0.0] + [_f(v) for v in np.quantile(r_paths, q, axis=0)] for q in FAN_Q}
        last_ok = min(row + horizon, self.vault_row - 1, len(self.dates) - 1)
        realised = [0.0]
        if last_ok > row:
            fut = np.nan_to_num(np.asarray(self.panel["ell"][row + 1: last_ok + 1][:, cols], dtype=float))
            realised += [_f(v) for v in np.expm1(np.cumsum(fut, axis=0)) @ w]
        days = [str(self.dates[min(row + k, len(self.dates) - 1)].date()) if row + k < len(self.dates) else f"t+{k}" for k in range(horizon + 1)]
        # ---- per-holding table (internal research view; ranges, never targets)
        cum = np.expm1(ell.astype(np.float64).sum(axis=1))
        sigma = np.asarray(self.panel["sigma"][row, cols], dtype=float)
        es_share = {c["key"]: c["share_of_es95"] for c in rep["risk_contributions"]}
        table = []
        for h_ in sorted(rep["portfolio"]["holdings"], key=lambda d: -d["weight"])[:60]:
            i = idx[h_["key"]]
            q = np.quantile(cum[:, i], [0.05, 0.5, 0.95])
            table.append({"key": h_["key"], "symbol": str(self.symbol[cols[i]]), "name": str(self.name[cols[i]]),
                          "sector": self.sector_names[self.sector_of[cols[i]]], "weight": h_["weight"], "vol_ann": _f(sigma[i] * np.sqrt(252), 4),
                          "p05": _f(q[0], 4), "p50": _f(q[1], 4), "p95": _f(q[2], 4), "es_share": es_share.get(h_["key"])})
        dd = RT.drawdown_probabilities(mkt, horizons=(horizon,))[horizon]
        return {"report": rep, "label": label, "fan": fan, "realised": realised, "days": days, "sentence": template_sentence(rep),
                "holdings_table": table, "n_holdings": len(rep["portfolio"]["holdings"]),
                "market": {"trend": trend, "drawdown": dd, "sigma_daily": _f(m_sigma),
                           "p05": _f(np.quantile(np.expm1(mkt.sum(axis=1)), 0.05)), "p50": _f(np.quantile(np.expm1(mkt.sum(axis=1)), 0.5)),
                           "p95": _f(np.quantile(np.expm1(mkt.sum(axis=1)), 0.95))},
                "sectors": [{"sector": r.sector, "p_worst3": _f(r.p_worst3, 4), "p_best3": _f(r.p_best3, 4), "abn_p05": _f(r.abn_p05, 4),
                             "abn_p50": _f(r.abn_p50, 4), "abn_p95": _f(r.abn_p95, 4)} for r in sec.itertuples()],
                "info": {**sc["info"], "n_stocks": int(len(cols)), "n_paths": int(ell.shape[0]), "horizon": horizon,
                         "realised_days": len(realised) - 1}}

    # ------------------------------------------------------------------ training runs and scores
    def runs(self) -> list[dict]:
        out = []
        for d in sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []:
            cfg_p = d / "config.json"
            if not cfg_p.exists():
                continue
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            if cfg.get("smoke") or d.name.startswith(DIAGNOSTIC) or "baseline" in cfg.get("config", {}):
                continue
            ms = read_metrics(d)
            tr = [(m["step"], m["train_loss"]) for m in ms if "train_loss" in m]
            va = [(m["step"], m["val_metric"]) for m in ms if "val_metric" in m]
            summ = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
            tcfg = cfg["config"].get("train", {})
            out.append({"run": d.name, "fold": cfg["fold"], "seed": cfg["seed"], "mode": (cfg["config"].get("model") or {}).get("mode"),
                        "status": summ.get("status", "running or interrupted"), "max_steps": tcfg.get("max_steps"),
                        "step": tr[-1][0] if tr else 0, "best_val": _f(min(v for _, v in va)) if va else None,
                        "last_time": ms[-1]["time"] if ms else None,
                        "train": [[s, _f(v, 5)] for s, v in tr], "val": [[s, _f(v, 5)] for s, v in va]})
        return out

    def scores(self) -> dict:
        """Mean scores per model / horizon / held-in vs held-out from reports/scores (whatever has been run so far)."""
        root = REPORTS / "scores"
        rows = []
        for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            for split in ("test", "val"):
                files = [p for p in sorted(d.glob(f"*_{split}.parquet")) if not p.stem.endswith("_dates")]
                parts = [pd.read_parquet(p) for p in files]
                parts = [p for p in parts if len(p)]
                if not parts:
                    continue
                df = pd.concat(parts, ignore_index=True)
                for (h, ho), g in df.groupby(["h", "heldout"]):
                    r = {"model": d.name, "split": split, "folds": len(files), "h": int(h), "stocks": "held-out" if ho else "held-in",
                         "dates": int(g["date"].nunique())}
                    for c in ("crps", "qlike", "in50", "in90", "hit05"):
                        r[c] = _f(g[c].mean(), 4) if c in g and g[c].notna().any() else None
                    rows.append(r)
        return {"rows": rows}
