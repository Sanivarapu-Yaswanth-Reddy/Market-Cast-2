"""Robust per-channel normalisation (03 §10), fitted per fold on TRAINING dates and TRAINING stocks only.

    x_norm = clip((x - median) / (1.4826 * MAD), -8, 8);   missing -> 0 after normalisation

The statistics are stored next to the fold (data/features/<fold>/norm.json) and never refit on validation
or test data. Flags and bounded channels are passed through (median 0, scale 1). Targets are never
normalised or clipped here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIP = 8.0


@dataclass
class Normalizer:
    names: list[str]
    median: np.ndarray        # [F]
    scale: np.ndarray         # [F]  (1.4826 * MAD, or the fallback described in fit())

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x[..., F] raw -> normalised float32 with NaN -> 0 and clipping to +-8."""
        y = (x.astype(np.float32) - self.median.astype(np.float32)) / self.scale.astype(np.float32)
        np.clip(y, -CLIP, CLIP, out=y)
        y[~np.isfinite(y)] = 0.0
        return y

    def to_dict(self) -> dict:
        return {"names": self.names, "median": self.median.tolist(), "scale": self.scale.tolist(), "clip": CLIP}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(list(d["names"]), np.asarray(d["median"], dtype=np.float32), np.asarray(d["scale"], dtype=np.float32))


def fit(sample: np.ndarray, names: list[str], passthrough: set[str] | None = None) -> Normalizer:
    """Fit on sample[n, F] (rows = training cells; NaN = missing).

    scale = 1.4826 * MAD; if the MAD is 0 (a channel that is constant for most cells, e.g. a sparse count) the
    standard deviation is used; if that is 0 too, 1. A channel with no finite training value gets (0, 1).
    """
    passthrough = passthrough or set()
    f = sample.shape[1]
    med, scale = np.zeros(f, dtype=np.float32), np.ones(f, dtype=np.float32)
    for j, name in enumerate(names):
        if name in passthrough:
            continue
        col = sample[:, j].astype(np.float64)
        col = col[np.isfinite(col)]
        if col.size < 10:
            continue
        m = np.median(col)
        mad = 1.4826 * np.median(np.abs(col - m))
        if mad <= 1e-8:
            mad = float(col.std())
        med[j] = m
        scale[j] = mad if mad > 1e-8 else 1.0
    return Normalizer(list(names), med, scale)
