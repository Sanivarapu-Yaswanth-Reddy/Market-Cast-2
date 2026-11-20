"""Shared helpers used by every layer: atomic file writes, config loading, logging, hashing.

WHY THIS FILE EXISTS
--------------------
* The project lives inside OneDrive, which can lock a file mid-sync. Every writer therefore writes to a
  temporary sibling file and then calls ``os.replace`` (atomic on Windows and POSIX), retrying a few times
  if the sync client holds a lock (07 §3 "Windows notes").
* Every run must be reproducible and logged (00 rule 9), so the config hash and data-snapshot hash
  helpers live here and are used by training, baselines and scoring alike.

Nothing in here knows about markets; keep it that way.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from marketcast import CONFIGS, ROOT

IST = "Asia/Kolkata"  # timezone of every `available_at` timestamp (07 §4: never naive)


# --------------------------------------------------------------------------- logging
def get_logger(name: str = "marketcast") -> logging.Logger:
    """INFO to console, DEBUG to logs/<name>_<date>.log (07 §4). Safe to call many times."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        d = ROOT / "logs"
        d.mkdir(exist_ok=True)
        fh = logging.FileHandler(d / f"{name.replace('.', '_')}_{time.strftime('%Y%m%d')}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:  # read-only location: console logging is enough
        pass
    log.propagate = False
    return log


# --------------------------------------------------------------------------- atomic writes
def _replace_with_retry(tmp: Path, dst: Path, tries: int = 8) -> None:
    """os.replace with back-off: OneDrive/antivirus may hold the destination for a moment."""
    for k in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))


def atomic_path(dst: str | Path) -> tuple[Path, Path]:
    """Return (temporary path, final path); the caller writes the first and calls commit()."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(dst.name + f".tmp{os.getpid()}"), dst


def commit(tmp: Path, dst: Path) -> Path:
    _replace_with_retry(tmp, dst)
    return dst


def write_parquet(df: pd.DataFrame, dst: str | Path) -> Path:
    """Atomic Parquet write (zstd). Index is never written: keys are always explicit columns."""
    tmp, dst = atomic_path(dst)
    df.to_parquet(tmp, index=False, compression="zstd")
    return commit(tmp, dst)


def write_text(text: str, dst: str | Path) -> Path:
    tmp, dst = atomic_path(dst)
    tmp.write_text(text, encoding="utf-8")
    return commit(tmp, dst)


def write_json(obj: Any, dst: str | Path) -> Path:
    return write_text(json.dumps(obj, indent=2, default=_json_default), dst)


def write_yaml(obj: Any, dst: str | Path, header: str = "") -> Path:
    body = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=120)
    return write_text((header.rstrip() + "\n" if header else "") + body, dst)


def save_npy(arr: np.ndarray, dst: str | Path) -> Path:
    """Atomic .npy write (np.save appends '.npy' to names without it, so write through a handle)."""
    tmp, dst = atomic_path(dst)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    return commit(tmp, dst)


def save_npz(dst: str | Path, **arrays: np.ndarray) -> Path:
    tmp, dst = atomic_path(dst)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    return commit(tmp, dst)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- configs and hashes
def load_yaml(path: str | Path) -> dict:
    """Load a YAML file; relative names are looked up under configs/."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIGS / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, over: dict) -> dict:
    """Recursive dict merge used for `--set a.b=c` style overrides (returns a new dict)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_hash(cfg: dict) -> str:
    """sha256 of the canonical JSON of a resolved config (07 §4). First 12 hex chars are the short id."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=_json_default).encode()).hexdigest()


def snapshot_hash(paths: Iterable[str | Path]) -> str:
    """Data snapshot hash: sha256 over the sorted (path, size, mtime) of the files a run read (07 §4).

    Directories are expanded recursively. Missing paths are recorded as missing rather than raising,
    so a run on the GPU machine (which has no raw data) still gets a stable, honest hash.
    """
    items: list[tuple[str, int, int]] = []
    for p in paths:
        p = Path(p)
        files = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
        for q in files:
            if q.exists():
                st = q.stat()
                items.append((q.as_posix(), st.st_size, int(st.st_mtime)))
            else:
                items.append((q.as_posix(), -1, -1))
    h = hashlib.sha256()
    for it in sorted(items):
        h.update(repr(it).encode())
    return "sha256:" + h.hexdigest()


def git_commit() -> str:
    """Current git commit, or 'nogit' (git is not on PATH on the data PC; see reports/P0_environment.md)."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) Torch (07 §4 determinism)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = "{:.4g}") -> str:
    """Markdown table without the optional `tabulate` dependency (used by every report writer)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
