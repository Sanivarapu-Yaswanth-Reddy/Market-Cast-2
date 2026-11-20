"""Experiment log (00 rule 9): every training or evaluation run leaves a permanent, countable record.

    runs/<run_name>/config.json      resolved config, config hash, git commit, data snapshot hash, fold, seed
    runs/<run_name>/metrics.jsonl    one JSON line per logged step / evaluation
    runs/<run_name>/summary.json     final status and best validation metric
    runs/registry.jsonl              ONE LINE PER RUN EVENT, append-only. Never delete lines: the number of
                                     distinct runs is the N of the deflated Sharpe ratio / multiple-testing
                                     corrections (05 §4), so a deleted run is a hidden trial.

MLflow is optional: if it is installed and train.mlflow is true, the same numbers are mirrored to ./mlruns.
The JSONL files are the source of truth because they need no server and survive a copy between machines.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from marketcast import ROOT
from marketcast.util import _json_default, config_hash, git_commit, write_json

RUNS = ROOT / "runs"


def count_runs(include_smoke: bool = False) -> int:
    """Number of distinct logged runs (the multiple-testing N)."""
    p = RUNS / "registry.jsonl"
    if not p.exists():
        return 0
    seen = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if include_smoke or not r.get("smoke"):
            seen.add(r.get("run_id"))
    return len(seen)


class RunLogger:
    def __init__(self, run_name: str, cfg: dict, fold: str, seed: int, data_hash: str, smoke: bool = False, resume: bool = False):
        self.dir = RUNS / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"run_id": f"{run_name}@{config_hash(cfg)[:12]}", "run_name": run_name, "config_hash": config_hash(cfg),
                     "git_commit": git_commit(), "data_snapshot": data_hash, "fold": fold, "seed": seed, "smoke": smoke}
        write_json({**self.meta, "config": cfg}, self.dir / "config.json")
        self._metrics = self.dir / "metrics.jsonl"
        self._registry("resumed" if resume else "started")
        self._mlflow = None
        if (cfg.get("train") or {}).get("mlflow"):
            try:
                import mlflow

                mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
                mlflow.set_experiment(str(cfg.get("phase", "marketcast")))
                mlflow.start_run(run_name=run_name)
                mlflow.log_params({"fold": fold, "seed": seed, "config_hash": self.meta["config_hash"][:12]})
                self._mlflow = mlflow
            except Exception as e:      # logging must never kill a training run
                print(f"[run-log] MLflow disabled: {type(e).__name__}: {e}")

    def _registry(self, status: str, **extra) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        rec = {**self.meta, "status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        with open(RUNS / "registry.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def log(self, step: int, **metrics) -> None:
        rec = {"step": step, "time": time.strftime("%H:%M:%S"), **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}}
        with open(self._metrics, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics({k: v for k, v in rec.items() if isinstance(v, float)}, step=step)
            except Exception:
                pass

    def finish(self, status: str, **summary) -> None:
        write_json({**self.meta, "status": status, **summary}, self.dir / "summary.json")
        self._registry(status, **summary)
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception:
                pass


def read_metrics(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
