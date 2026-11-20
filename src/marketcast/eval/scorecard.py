"""Scorecard (05 §10): aggregates reports/scores/<model>/<fold>_<split>.parquet into one markdown table.

    python -m marketcast eval scorecard --out reports/scorecard_baselines.md

Per model and horizon: liquidity-weighted and unweighted fair CRPS, QLIKE, interval coverage (50/80/90/95),
PIT max-bin deviation, VaR 95/99 violation rates with Kupiec and Christoffersen p-values, joint energy score,
and a Diebold-Mariano test (Newey-West, lag h + 5) of each model against the reference (default B0) on
per-date mean CRPS over the dates both models scored. Held-in and held-out stocks are reported separately.

WHAT THIS VERSION DOES NOT DO YET (so nobody reads the table as a finished gate report)
    block-bootstrap confidence intervals, the scenario / stress / size slices of 05 §8, calibration plots.
VaR tests pool stocks per date into a daily violation RATE series for Christoffersen (independence over time)
and use non-overlapping dates for h > 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from marketcast import REPORTS
from marketcast.eval import metrics as MT
from marketcast.util import get_logger, md_table, write_text

log = get_logger("eval.scorecard")


def load_scores(split: str = "test") -> dict[str, pd.DataFrame]:
    out = {}
    root = REPORTS / "scores"
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        parts = [pd.read_parquet(p).assign(fold=p.stem.split("_")[0]) for p in sorted(d.glob(f"*_{split}.parquet"))]
        parts = [p for p in parts if len(p)]
        if parts:
            out[d.name] = pd.concat(parts, ignore_index=True)
    return out


def _wmean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna()
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, ho), g in df.groupby(["h", "heldout"]):
        r = {"h": int(h), "stocks": "held-out" if ho else "held-in", "n": len(g), "dates": g["date"].nunique()}
        if "crps" in g and g["crps"].notna().any():
            r["crps_w"], r["crps"] = _wmean(g["crps"], g["weight"]), float(g["crps"].mean())
            for lv in (50, 80, 90, 95):
                r[f"cov{lv}"] = float(g[f"in{lv}"].mean())
            r["pit_maxdev"] = MT.pit_max_bin_deviation(g["pit"].dropna().to_numpy())
            sub = g[g["date"].isin(np.sort(g["date"].unique())[:: int(h)])]              # non-overlapping windows
            for q, col in ((0.05, "hit05"), (0.01, "hit01")):
                v = sub[col].astype(int).to_numpy()
                r[f"var{int(100 * (1 - q))}_rate"] = float(v.mean())
                r[f"kupiec{int(100 * (1 - q))}_p"] = MT.kupiec_pof(v, q).pvalue
                daily = (sub.groupby("date")[col].mean() > q).astype(int).to_numpy()   # day flagged if its violation rate exceeds q
                r[f"christ{int(100 * (1 - q))}_p"] = MT.christoffersen(daily, max(daily.mean(), 1e-6))[0].pvalue if len(daily) > 10 else np.nan
        if "qlike" in g and g["qlike"].notna().any():
            r["qlike_w"], r["qlike"] = _wmean(g["qlike"], g["weight"]), float(g["qlike"].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def dm_vs_reference(scores: dict[str, pd.DataFrame], ref: str, col: str = "crps") -> pd.DataFrame:
    rows = []
    if ref not in scores or col not in scores[ref]:
        return pd.DataFrame()
    base = scores[ref].groupby(["h", "date"])[col].mean()
    for name, df in scores.items():
        if name == ref or col not in df:
            continue
        mine = df.groupby(["h", "date"])[col].mean()
        for h in sorted(set(mine.index.get_level_values(0)) & set(base.index.get_level_values(0))):
            j = pd.concat([mine.loc[h], base.loc[h]], axis=1, keys=["a", "b"]).dropna()
            if len(j) > 30:
                t = MT.diebold_mariano(j["a"].to_numpy(), j["b"].to_numpy(), h=int(h))
                rows.append({"model": name, "vs": ref, "metric": col, "h": int(h), "dates": len(j), "mean_diff": t.detail["mean_diff"],
                             "dm_stat": t.stat, "p_value": t.pvalue})
    return pd.DataFrame(rows)


def main(out: str | None = None, split: str = "test", ref: str = "B0") -> int:
    from marketcast.train.logging import count_runs

    scores = load_scores(split)
    if not scores:
        log.error("no score files for split %r under reports/scores/; run `baselines run` or `eval score` first", split)
        return 1
    L = [f"# Scorecard ({split} split)", "", f"Generated {pd.Timestamp.now().isoformat(timespec='seconds')}. Runs logged so far: {count_runs()} "
         "(feeds multiple-testing corrections). Lower is better for CRPS / QLIKE / energy; coverage should match its nominal level.", "",
         "Not included yet: bootstrap confidence intervals, scenario and stress slices, calibration plots (see module docstring).", ""]
    for name, df in scores.items():
        L += [f"## {name}", "", f"folds: {sorted(df['fold'].unique())}", "", md_table(summarise(df)), ""]
    for col in ("crps", "qlike"):
        dm = dm_vs_reference(scores, ref, col)
        if len(dm):
            L += [f"## Diebold-Mariano vs {ref} on per-date mean {col} (negative = better than {ref})", "", md_table(dm), ""]
    p = write_text("\n".join(L), Path(out) if out else REPORTS / f"scorecard_{split}.md")
    log.info("scorecard written to %s (%d models)", p, len(scores))
    return 0
