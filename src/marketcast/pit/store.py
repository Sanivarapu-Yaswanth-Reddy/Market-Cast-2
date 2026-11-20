"""P1.4  Point-in-time store: one DuckDB connection with views over the Parquet tables (02 §6, decision D-02).

Large tables stay as Parquet; DuckDB only holds VIEW definitions, re-created on every connect so the
.duckdb file never goes stale when a table is rebuilt (and stays valid if the project folder moves).

ACCESS PATTERNS (all covered by tests/test_pit.py)
    store.panel(start, end, universe="MC500", cols=[...])   EOD rows; PIT by construction (EOD exception, 02 §2)
    store.asof("global_long", at_ts, keys=[...])            latest row per series with available_at <= at_ts
    store.global_for_india_date(d)                          asof at cutoff(d): US date <= d-1 automatically
    store.macro_asof(d)                                     same clock; monthly lags already inside available_at
    store.universe(d, name="MC500")                         members effective on d
    store.sector_tree(d)                                    sector_map rows valid on d (see hindsight note there)

Every as-of query filters on `available_at <= ?` in SQL, so "never returns a later fact" is a property of
the query text, not of caller discipline.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from marketcast import PIT, PROCESSED
from marketcast.pit.lags import cutoff
from marketcast.util import IST

DB_PATH = PIT / "marketcast.duckdb"


def _q(p: Path) -> str:
    """Path -> SQL string literal (forward slashes; single quotes doubled)."""
    return "'" + p.as_posix().replace("'", "''") + "'"


class PITStore:
    """Open with `PITStore()` (file-backed) or `PITStore(":memory:")` for tests with custom tables."""

    def __init__(self, db: str | Path | None = None, read_only: bool = False, processed: Path = PROCESSED, pit: Path = PIT):
        self.processed, self.pit = Path(processed), Path(pit)
        target = str(db) if db is not None else str(DB_PATH)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(target, read_only=read_only)
        self.con.execute(f"SET TimeZone='{IST}'")
        if not read_only:
            self._create_views()

    # ---------------------------------------------------------------- views
    def _create_views(self) -> None:
        eq = self.processed / "equity_daily"
        views = {
            "equity_daily": (eq.exists() and any(eq.glob("year=*/part.parquet")),
                             f"SELECT * FROM read_parquet({_q(eq / 'year=*' / 'part.parquet')}, hive_partitioning=false)"),
            "indices_daily": ((self.processed / "indices_daily.parquet").exists(),
                              f"SELECT * FROM read_parquet({_q(self.processed / 'indices_daily.parquet')})"),
            "global_long": ((self.pit / "global_long.parquet").exists(),
                            f"SELECT * FROM read_parquet({_q(self.pit / 'global_long.parquet')})"),
            "universe": ((self.pit / "ref_universe.parquet").exists(),
                         f"SELECT * FROM read_parquet({_q(self.pit / 'ref_universe.parquet')})"),
            "sector_map": ((self.pit / "ref_sector_map.parquet").exists(),
                           f"SELECT * FROM read_parquet({_q(self.pit / 'ref_sector_map.parquet')})"),
            "trading_calendar": ((self.pit / "ref_trading_calendar.parquet").exists(),
                                 f"SELECT * FROM read_parquet({_q(self.pit / 'ref_trading_calendar.parquet')})"),
            "security_master": ((self.pit / "ref_security_master.parquet").exists(),
                                f"SELECT * FROM read_parquet({_q(self.pit / 'ref_security_master.parquet')})"),
        }
        for name, (ok, sql) in views.items():
            if ok:
                self.con.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}")

    def register(self, name: str, df: pd.DataFrame) -> None:
        """Expose a DataFrame as a table (tests, ad-hoc fact tables such as events in Phase 5)."""
        self.con.register(f"_df_{name}", df)
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM _df_{name}")

    def tables(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT table_name FROM information_schema.tables").fetchall()]

    # ---------------------------------------------------------------- queries
    def panel(self, start, end, universe: str | None = "MC500", cols: list[str] | None = None) -> pd.DataFrame:
        """EOD equity rows in [start, end]; with `universe`, only rows of members effective on each date."""
        sel = ", ".join(f"e.{c}" for c in dict.fromkeys(["date", "key", *(cols or ["close", "pc", "ret_1d", "turnover"])]))
        if universe:
            sql = (f"SELECT {sel} FROM equity_daily e JOIN universe u ON u.key = e.key AND u.universe = ? "
                   "AND e.date >= u.effective_from AND e.date <= u.effective_to WHERE e.date BETWEEN ? AND ? ORDER BY e.date, e.key")
            args = [universe, pd.Timestamp(start), pd.Timestamp(end)]
        else:
            sql = f"SELECT {sel} FROM equity_daily e WHERE e.date BETWEEN ? AND ? ORDER BY e.date, e.key"
            args = [pd.Timestamp(start), pd.Timestamp(end)]
        return self.con.execute(sql, args).df()

    def asof(self, table: str, at_ts, keys: list[str] | None = None, key_col: str = "series") -> pd.DataFrame:
        """Latest row per `key_col` with available_at <= at_ts. `at_ts` must be tz-aware."""
        at_ts = pd.Timestamp(at_ts)
        if at_ts.tzinfo is None:
            raise ValueError("at_ts must be timezone-aware (Asia/Kolkata); naive availability timestamps are a bug")
        where, args = "available_at <= ?", [at_ts.to_pydatetime()]
        if keys:
            where += f" AND {key_col} IN ({', '.join('?' * len(keys))})"
            args += list(keys)
        sql = (f"SELECT * FROM {table} WHERE {where} "
               f"QUALIFY row_number() OVER (PARTITION BY {key_col} ORDER BY available_at DESC, valid_date DESC) = 1 ORDER BY {key_col}")
        return self.con.execute(sql, args).df()

    def global_for_india_date(self, d, keys: list[str] | None = None) -> pd.DataFrame:
        """Global/macro facts usable for a forecast made at the close of India session d."""
        return self.asof("global_long", cutoff(d), keys)

    def macro_asof(self, d, keys: list[str] | None = None) -> pd.DataFrame:
        """Alias kept for the contract in 02 §6: monthly series are in the same table with lagged available_at."""
        return self.asof("global_long", cutoff(d), keys)

    def universe(self, d, name: str = "MC500") -> pd.DataFrame:
        d = pd.Timestamp(d)
        return self.con.execute("SELECT key, rank, avg_traded_value_6m, rebalance_date FROM universe "
                                "WHERE universe = ? AND effective_from <= ? AND effective_to >= ? ORDER BY rank", [name, d, d]).df()

    def sector_tree(self, d) -> pd.DataFrame:
        """Sector rows valid on d. Before the first snapshot the snapshot itself is returned with
        hindsight_class = True (decision D-08) rather than nothing, and the flag tells the caller."""
        d = pd.Timestamp(d)
        df = self.con.execute("SELECT * FROM sector_map").df()
        df["is_hindsight_for_date"] = df["valid_from"] > d
        return df

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "PITStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def check(n: int = 2000, seed: int = 0) -> dict:
    """`python -m marketcast pit check`: property test on the live store.

    Draws random cutoffs and asserts that no as-of row is newer than its cutoff and that the value agrees
    with an independent pandas computation.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    with PITStore() as st:
        if "global_long" not in st.tables():
            raise FileNotFoundError("global_long view missing; run `python -m marketcast data global` first")
        long = st.con.execute("SELECT * FROM global_long").df()
        lo, hi = long["available_at"].min(), long["available_at"].max()
        series = long["series"].unique()
        bad = 0
        for _ in range(n):
            ts = lo + (hi - lo) * float(rng.uniform())
            s = str(rng.choice(series))
            got = st.asof("global_long", ts, [s])
            ref = long[(long["series"] == s) & (long["available_at"] <= ts)]
            if len(got) and (got["available_at"] > ts).any():
                bad += 1
            elif len(ref) != 0 and (len(got) != 1 or got["available_at"].iloc[0] != ref["available_at"].max()):
                bad += 1
            elif len(ref) == 0 and len(got):
                bad += 1
    return {"queries": n, "violations": bad}
