"""Single entry point: python -m marketcast <group> <command>   (07 §2).

Every command is re-runnable and idempotent. Heavy imports happen inside the handlers so `--help` is instant
and the data PC (no GPU, CPU torch only) never needs the training stack for data commands.

PIPELINE ORDER (each step reads only what earlier steps wrote; `data all` runs the data block in order)
    data build-panel -> data calendar -> data master -> data universe -> data sectors -> data derivatives
    -> data global -> folds make -> targets build -> features build -> graph build
    -> baselines run -> train -> eval score -> eval scorecard
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from marketcast import ROOT, __version__


def _run_script(name: str, extra: list[str]) -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts" / name), *extra])


def _data_all(a) -> int:
    """The whole data foundation in dependency order (P0.3 - P2.2). Safe to re-run: finished stages are skipped
    or cheaply rebuilt."""
    from marketcast.data import build_panel, calendar, derivatives, global_align, sector_map, security_master, universe
    from marketcast.eval import folds
    from marketcast.targets import build as tbuild

    build_panel.main(force=a.force, workers=a.workers)
    for step in (calendar.main, security_master.main, universe.main, sector_map.main, derivatives.main,
                 global_align.main, folds.main, tbuild.main):
        rc = step()
        if rc:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="marketcast", description=f"MarketCast {__version__}")
    sub = ap.add_subparsers(dest="group", required=True)

    d = sub.add_parser("data", help="data foundation (P0-P1)").add_subparsers(dest="cmd", required=True)
    d.add_parser("inventory", help="P0.1 raw download inventory -> reports/P0_raw_inventory.md").add_argument("--out", default=None)
    bp = d.add_parser("build-panel", help="P0.3 raw -> data/processed (per-year, resumable)")
    bp.add_argument("--only", default="fred,indices,equity,fo")
    bp.add_argument("--force", action="store_true", help="re-parse years even if their output is up to date")
    bp.add_argument("--workers", type=int, default=4)
    for name, hlp in [("calendar", "P1.1 trading calendar + index aliases"), ("master", "P1.2 security master, symbol history"),
                      ("universe", "P1.5 MC500 / MC1000"), ("sectors", "P1.6 sector map"), ("derivatives", "P1.7 F&O features"),
                      ("global", "P1.8 global/macro PIT alignment"), ("validate", "rewrite reports/data_quality.txt")]:
        d.add_parser(name, help=hlp)
    da = d.add_parser("all", help="run the whole data foundation in order (build-panel ... targets)")
    da.add_argument("--force", action="store_true")
    da.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("pit", help="PIT store").add_subparsers(dest="cmd", required=True)
    p.add_parser("check", help="property test of as-of queries on the live store").add_argument("--n", type=int, default=2000)

    sub.add_parser("folds", help="P2.2 folds and held-out list").add_subparsers(dest="cmd", required=True).add_parser("make")
    sub.add_parser("targets", help="P2.1 returns, masks, targets").add_subparsers(dest="cmd", required=True).add_parser("build")
    sub.add_parser("features", help="P3.1 feature store").add_subparsers(dest="cmd", required=True).add_parser("build")
    sub.add_parser("graph", help="P3.2 sector fabric caches").add_subparsers(dest="cmd", required=True).add_parser("build")

    b = sub.add_parser("baselines", help="P2.5 baselines").add_subparsers(dest="cmd", required=True)
    br = b.add_parser("run", help="score a baseline on walk-forward folds")
    br.add_argument("--model", required=True, help="B0, B0g, B1, B2, B5")
    br.add_argument("--folds", default="fast", help="fast | all | comma list (F2,F4)")
    br.add_argument("--split", default="test", choices=["val", "test"])
    br.add_argument("--paths", type=int, default=256)
    br.add_argument("--every", type=int, default=1, help="score every n-th decision date (speed)")

    t = sub.add_parser("train", help="P3.4 / P4 training (run on the GPU machine)")
    t.add_argument("--config", default="model/p4_gen.yaml")
    t.add_argument("--fold", default="F4")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true", help="continue from the run's last checkpoint")
    t.add_argument("--smoke", action="store_true", help="tiny model, a few steps, CPU-friendly: checks the code path only")
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. train.max_steps=200")

    e = sub.add_parser("eval", help="scoring and scorecards").add_subparsers(dest="cmd", required=True)
    es = e.add_parser("score", help="score a trained run's checkpoint on a fold split")
    es.add_argument("--run", required=True, help="run directory under runs/")
    es.add_argument("--split", default="val", choices=["val", "test"])
    es.add_argument("--paths", type=int, default=16)
    es.add_argument("--every", type=int, default=5)
    ec = e.add_parser("scorecard", help="aggregate reports/scores/* into a markdown scorecard")
    ec.add_argument("--out", default=None)

    ap_app = sub.add_parser("app", help="local research dashboard in the browser (this PC only)")
    ap_app.add_argument("--port", type=int, default=8765)
    ap_app.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    sub.add_parser("env", help="environment report (scripts/check_env.py)")
    sub.add_parser("status", help="plan status (dev_plan/plan_status.py --next)")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    g, c = a.group, getattr(a, "cmd", None)
    if g == "data":
        if c == "inventory":
            from marketcast.data.inventory import write_report

            print(write_report(a.out))
            return 0
        if c == "build-panel":
            from marketcast.data.build_panel import main as m

            return m(a.only, a.force, a.workers)
        if c == "validate":
            from marketcast.data.build_panel import quality_report

            print(quality_report())
            return 0
        if c == "all":
            return _data_all(a)
        mod = {"calendar": "calendar", "master": "security_master", "universe": "universe", "sectors": "sector_map",
               "derivatives": "derivatives", "global": "global_align"}[c]
        import importlib

        return importlib.import_module(f"marketcast.data.{mod}").main()
    if g == "pit":
        from marketcast.pit.store import check

        res = check(a.n)
        print(res)
        return 1 if res["violations"] else 0
    if g == "folds":
        from marketcast.eval.folds import main as m

        return m()
    if g == "targets":
        from marketcast.targets.build import main as m

        return m()
    if g == "features":
        from marketcast.features.build import main as m

        return m()
    if g == "graph":
        from marketcast.graph.fabric import main as m

        return m()
    if g == "baselines":
        from marketcast.baselines.run import main as m

        return m(a.model, a.folds, a.split, a.paths, a.every)
    if g == "train":
        from marketcast.train.loop import main as m

        return m(a.config, a.fold, a.seed, a.resume, a.smoke, a.set)
    if g == "eval":
        if c == "score":
            from marketcast.eval.score import score_run

            return score_run(a.run, a.split, a.paths, a.every)
        from marketcast.eval.scorecard import main as m

        return m(a.out)
    if g == "app":
        from marketcast.app.server import main as m

        return m(a.port, not a.no_browser)
    if g == "env":
        return _run_script("check_env.py", [])
    if g == "status":
        return subprocess.call([sys.executable, str(ROOT / "dev_plan" / "plan_status.py"), "--next"])
    return 1
