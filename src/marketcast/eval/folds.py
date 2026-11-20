"""P2.2  Walk-forward folds, purge/embargo, held-out companies, and the vault guard (05 §1-§3; 00 rules 5-7).

FOLDS (configs/folds.yaml, written by `python -m marketcast folds make`)
    F1..F9 and Vault, expanding training window from 2008-01-01. Each split is a set of DECISION dates.
    Purge + embargo: with horizon H = 20 and embargo E = 20 sessions,
        last decision date of a split  <=  first decision date of the next split - (H + E) sessions
    so no 20-day target window of the earlier split can touch the later one, plus a 20-session gap.
    The same trimming is applied before the vault boundary, so no non-vault target ever reads 2026 data.

HELD-OUT COMPANIES (configs/heldout_companies.txt)
    named 20 (dev_plan/heldout_named_20.csv) + a stratified top-up to ~15% of companies ever in MC500:
    bucket = (sector, size tercile by lifetime median traded value); inside a bucket order by sha256(key) and
    hold out round(15% * bucket size) minus the named ones already in it. Deterministic, no RNG.

VAULT (00 rule 6): any decision date >= 2026-01-01 is refused by `assert_not_vault` unless the caller passes
    allow_vault=True, which appends a line to reports/vault_log.md. Data BUILDERS may process 2026 rows (they
    must, to keep tables current); models, baselines and scorers go through `split_rows`, which enforces this.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketcast import CONFIGS, REPORTS, ROOT
from marketcast.util import get_logger, load_yaml, write_text, write_yaml

log = get_logger("eval.folds")
H, E = 20, 20
VAULT_START = pd.Timestamp("2026-01-01")
FOLDS_PATH = CONFIGS / "folds.yaml"
HELDOUT_PATH = CONFIGS / "heldout_companies.txt"
FAST_FOLDS = ["F2", "F4", "F6", "F8"]
HELDOUT_SHARE = 0.15


def default_folds() -> dict:
    folds = {}
    for i in range(1, 10):
        folds[f"F{i}"] = {"train": ["2008-01-01", f"{2014 + i}-12-31"], "val": [f"{2015 + i}-01-01", f"{2015 + i}-12-31"],
                          "test": [f"{2016 + i}-01-01", f"{2016 + i}-12-31"]}
    folds["Vault"] = {"train": ["2008-01-01", "2024-12-31"], "val": ["2025-01-01", "2025-12-31"], "test": ["2026-01-01", "2099-12-31"]}
    return {"horizon": H, "embargo": E, "vault_start": str(VAULT_START.date()), "fast": FAST_FOLDS, "folds": folds}


def make_folds_file() -> dict:
    cfg = default_folds()
    write_yaml(cfg, FOLDS_PATH, header="# Walk-forward folds (05 §1). Ranges are DECISION dates before purge/embargo; "
               "marketcast.eval.folds.split_rows applies the\n# (H + E)-session trim at every boundary. Changing a boundary "
               "needs the owner's approval (00 §7).")
    return cfg


def load_folds() -> dict:
    return load_yaml(FOLDS_PATH) if FOLDS_PATH.exists() else default_folds()


def assert_not_vault(dates, allow_vault: bool = False, who: str = "") -> None:
    """Refuse decision dates inside the vault unless explicitly opened (and then log the opening)."""
    d = pd.DatetimeIndex(dates)
    if len(d) and d.max() >= VAULT_START:
        if not allow_vault:
            raise PermissionError(f"decision dates up to {d.max().date()} are inside the sealed vault (>= {VAULT_START.date()}); "
                                  "00 rule 6. Pass allow_vault=True only after the G4 gate is signed off.")
        p = REPORTS / "vault_log.md"
        prev = p.read_text(encoding="utf-8") if p.exists() else "# Vault log\n\nEvery opening of the 2026 holdout (00 rule 6).\n\n"
        write_text(prev + f"- {pd.Timestamp.now().isoformat(timespec='seconds')} opened by `{who or 'unknown'}` for "
                   f"{d.min().date()}..{d.max().date()} ({len(d)} decision dates)\n", p)
        log.warning("VAULT OPENED by %s: logged in reports/vault_log.md", who)


@dataclass
class Split:
    fold: str
    train: np.ndarray      # row numbers (into the session index) of decision dates
    val: np.ndarray
    test: np.ndarray


def split_rows(fold: str, sessions: pd.DatetimeIndex, cfg: dict | None = None, allow_vault: bool = False) -> Split:
    """Decision-date rows of one fold after purge and embargo.

    A decision date t needs its targets t+1..t+H to lie strictly before the next split's first decision date
    minus E sessions; equivalently row(t) <= row(first of next split) - (H + E). The test split is trimmed the
    same way against the vault start and against the end of the data (targets must exist).
    """
    cfg = cfg or load_folds()
    f = cfg["folds"][fold]
    h, e = int(cfg.get("horizon", H)), int(cfg.get("embargo", E))
    sessions = pd.DatetimeIndex(sessions)
    rows = {}
    for name in ("train", "val", "test"):
        lo, hi = pd.Timestamp(f[name][0]), pd.Timestamp(f[name][1])
        rows[name] = np.flatnonzero((sessions >= lo) & (sessions <= hi))
    vault_row = int(sessions.searchsorted(VAULT_START, "left"))
    is_vault_fold = len(rows["test"]) and sessions[rows["test"][0]] >= VAULT_START
    if is_vault_fold:
        assert_not_vault(sessions[rows["test"]], allow_vault, who=f"split_rows({fold})")
    # train and val: purge H + embargo E before the next split's first decision date
    nxt_val = rows["val"][0] if len(rows["val"]) else len(sessions)
    nxt_test = rows["test"][0] if len(rows["test"]) else len(sessions)
    rows["train"] = rows["train"][rows["train"] <= nxt_val - (h + e)]
    rows["val"] = rows["val"][rows["val"] <= nxt_test - (h + e)]
    # test: its 20-day targets must exist and (outside the vault fold) must end before the vault starts
    end = len(sessions) if is_vault_fold else min(vault_row, len(sessions))
    rows["test"] = rows["test"][rows["test"] + h <= end - 1]
    return Split(fold, rows["train"], rows["val"], rows["test"])


def check_boundaries(sessions: pd.DatetimeIndex, cfg: dict | None = None) -> list[str]:
    """05 §9 test 5: return a list of violations (empty = pass). Vault fold is checked without opening it."""
    cfg = cfg or load_folds()
    h, e = int(cfg.get("horizon", H)), int(cfg.get("embargo", E))
    bad = []
    for name in cfg["folds"]:
        if name == "Vault":
            continue
        s = split_rows(name, sessions, cfg)
        for a, b, la, lb in [(s.train, s.val, "train", "val"), (s.val, s.test, "val", "test")]:
            if len(a) and len(b) and not (a.max() + h < b.min() - e + 1):
                bad.append(f"{name}: {la} target end row {a.max() + h} not < {lb} start {b.min()} - embargo {e}")
        if len(s.test) and sessions[min(s.test.max() + h, len(sessions) - 1)] >= VAULT_START:
            bad.append(f"{name}: test targets reach the vault")
    return bad


# --------------------------------------------------------------------------- held-out companies
def _sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def resolve_named(named: pd.DataFrame, sym_hist: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Named-20 rows -> keys, through the symbol history (ticker as of today) with the ISIN prefix as a check."""
    keys = set(master["key"])
    out = []
    for r in named.itertuples(index=False):
        h = sym_hist[sym_hist["symbol"] == r.symbol].sort_values("valid_to")
        key = str(h["key"].iloc[-1]) if len(h) else None
        if key is None or not key.startswith(str(r.company_id)):
            alt = r.company_id if r.company_id in keys else (r.isin_2026 if r.isin_2026 in keys else None)
            if key is None:
                key = alt
            elif alt is not None and alt != key:
                log.warning("named held-out %s: symbol history says %s, csv says %s; using the csv id", r.symbol, key, alt)
                key = alt
        out.append({"symbol": r.symbol, "key": key})
    return pd.DataFrame(out)


def read_named() -> pd.DataFrame:
    """dev_plan/heldout_named_20.csv. Its last column is free text with unquoted commas, so a strict CSV parser
    fails; the first seven fields are clean, so split each line at most seven times."""
    lines = (ROOT / "dev_plan" / "heldout_named_20.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = [ln.split(",", len(header) - 1) for ln in lines[1:] if ln.strip()]
    return pd.DataFrame(rows, columns=header).apply(lambda s: s.str.strip())


def build_heldout() -> pd.DataFrame:
    """Write configs/heldout_companies.txt and return the table (key, kind, sector, tercile)."""
    from marketcast import PIT
    from marketcast.data.sector_map import load_sector_map
    from marketcast.data.security_master import SYMHIST_PATH, load_master
    from marketcast.data.universe import load_universe

    master = load_master()
    named = resolve_named(read_named(), pd.read_parquet(SYMHIST_PATH), master)
    if named["key"].isna().any():
        raise RuntimeError(f"named held-out companies not resolved: {named.loc[named['key'].isna(), 'symbol'].tolist()}")
    uni = load_universe()
    ever = pd.DataFrame({"key": np.sort(uni.loc[uni["universe"] == "MC500", "key"].unique())})
    ever = ever.merge(load_sector_map()[["key", "sector"]], on="key", how="left").merge(master[["key", "median_turnover"]], on="key", how="left")
    ever["sector"] = ever["sector"].fillna("Unclassified")
    ever["tercile"] = pd.qcut(ever["median_turnover"].rank(method="first"), 3, labels=["small", "mid", "large"]).astype(str)
    ever["named"] = ever["key"].isin(set(named["key"]))
    ever["sha"] = ever["key"].map(_sha)
    ever["heldout"] = ever["named"]
    # owner-approved exclusions (configs/heldout_exclusions.yaml): mega-caps the stratified rule must skip
    excl_path = CONFIGS / "heldout_exclusions.yaml"
    never = set((load_yaml(excl_path).get("never_hold_out") or {})) if excl_path.exists() else set()
    clash = never & set(named["key"])
    if clash:
        raise RuntimeError(f"{sorted(clash)} are in both the named held-out list and heldout_exclusions.yaml")
    ever["excluded"] = ever["key"].isin(never)
    for _, g in ever.groupby(["sector", "tercile"]):
        need = int(round(HELDOUT_SHARE * len(g))) - int(g["named"].sum())
        if need > 0:
            pick = g[~g["named"] & ~g["excluded"]].sort_values("sha").index[:need]
            ever.loc[pick, "heldout"] = True
    ho = ever[ever["heldout"]].copy()
    ho["kind"] = np.where(ho["named"], "named", "stratified")
    missing_named = sorted(set(named["key"]) - set(ever["key"]))     # named company never in MC500: still held out
    extra = pd.DataFrame({"key": missing_named, "kind": "named", "sector": "?", "tercile": "?"})
    ho = pd.concat([ho[["key", "kind", "sector", "tercile"]], extra], ignore_index=True).sort_values(["kind", "key"])
    sym = named.set_index("key")["symbol"]
    lines = ["# Held-out companies (05 §3, D-14). FROZEN: regenerate only to replace a name that stopped trading.",
             f"# {len(ho)} keys = {int((ho['kind'] == 'named').sum())} named + {int((ho['kind'] == 'stratified').sum())} stratified; "
             f"{len(ho) / max(len(ever), 1):.1%} of the {len(ever)} companies ever in MC500.", "# key  kind  [symbol]"]
    lines += [f"{r.key}  {r.kind}  {sym.get(r.key, '')}".rstrip() for r in ho.itertuples()]
    write_text("\n".join(lines) + "\n", HELDOUT_PATH)
    share = ever.groupby("sector")["heldout"].mean()
    log.info("held-out: %d keys (%.1f%% of %d ever-MC500); per-sector share min %.2f max %.2f; named missing from MC500 history: %s",
             len(ho), 100 * len(ho) / len(ever), len(ever), share.min(), share.max(), missing_named)
    return ho


def load_heldout() -> tuple[set[str], set[str]]:
    """Return (all held-out keys, the named subset). Raises if the frozen file is missing."""
    if not HELDOUT_PATH.exists():
        raise FileNotFoundError(f"{HELDOUT_PATH} missing; run `python -m marketcast folds make`")
    allk, named = set(), set()
    for line in HELDOUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        allk.add(parts[0])
        if len(parts) > 1 and parts[1] == "named":
            named.add(parts[0])
    return allk, named


def main() -> int:
    make_folds_file()
    build_heldout()
    return 0
