"""Trend baselines BT1 and BT2 (05 §6) for the T3 target: P(down), P(sideways), P(up) of the market at h = 5, 20.

    BT1  training-fold base rate (the benchmark the Brier SKILL is measured against)
    BT2  multinomial logistic regression on the market's trailing 20-day scaled momentum
    BT3  two-state Markov-switching model: not implemented yet

Labels come from targets/labels.py (fixed +-0.5 sigma sqrt(h) rule). Scores use NON-OVERLAPPING dates
(every h-th decision date), because overlapping h-day labels make the effective sample about 1/h of the rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketcast.eval import metrics as MT
from marketcast.eval.folds import split_rows
from marketcast.targets.build import ModelPanel
from marketcast.targets.labels import trend_labels


def score_fold(fold: str, split: str = "test", horizons=(5, 20)) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    p = ModelPanel()
    mkt = p.market()
    ell, sig = mkt["m_ell"].to_numpy(float), mkt["m_sigma"].to_numpy(float)
    mom = pd.Series(ell).rolling(20, min_periods=10).sum().to_numpy() / (sig * np.sqrt(20.0))
    sp = split_rows(fold, p.dates)
    rows = []
    for h in horizons:
        lab = trend_labels(ell, sig, h)
        tr = sp.train[(lab[sp.train] >= 0) & np.isfinite(mom[sp.train])][::h]
        te = getattr(sp, split)
        te = te[(lab[te] >= 0) & np.isfinite(mom[te])][::h]
        if len(tr) < 30 or len(te) < 5:
            continue
        base = np.bincount(lab[tr], minlength=3) / len(tr)
        p1 = np.broadcast_to(base, (len(te), 3))
        clf = LogisticRegression(max_iter=500).fit(mom[tr, None], lab[tr])
        p2 = np.zeros((len(te), 3))
        p2[:, clf.classes_] = clf.predict_proba(mom[te, None])
        for name, pr in (("BT1", p1), ("BT2", p2)):
            rows.append({"fold": fold, "model": name, "h": h, "n": len(te), "brier": float(MT.brier_multiclass(pr, lab[te]).mean()),
                         "brier_skill": MT.brier_skill(pr, lab[te], base), "log_score": float(MT.log_score(pr, lab[te]).mean()),
                         "base_down": base[0], "base_sideways": base[1], "base_up": base[2]})
    return pd.DataFrame(rows)
