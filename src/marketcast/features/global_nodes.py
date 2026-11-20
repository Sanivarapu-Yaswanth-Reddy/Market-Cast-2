"""Global-node features (03 §8): 16 nodes x up to 8 channels (+1 availability channel), one row per session.

INPUT   as-of LEVELS from data/global_align.py (already PIT: session t only sees values with
        available_at <= t 15:30 IST) and the market series from data/targets/market.parquet.
OUTPUT  array [T, 16, C_G] float32 with NaN for unavailable channels, plus the (node, channel) name table.

Because differences are taken on the as-of level series, a US holiday or a not-yet-published month simply
shows as a zero change; nothing is interpolated and nothing from the future can enter.

"yoy" for monthly India macro = ln(level_t / level_{t-252 sessions}) on the as-of series: both ends are what
was public on their own date, so the feature never uses a revised or late-published number... except that
FRED serves latest-revised values (gap G9), which is documented in configs/pit_lags.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketcast.data.wide import robust_z

N_CH = 8                      # data channels per node; channel N_CH is "node has any data today"
NODES = ["MARKET", "VIX_IN", "USDINR", "OIL", "US_RATES", "USD", "US_EQ", "ASIA_EQ", "GLOBAL_VOL", "CREDIT",
         "INDIA_RATES", "INDIA_MACRO", "FLOWS", "METALS", "AGRI", "POLICY_UNC"]


def _dln(s: pd.Series, n: int) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(s.where(s > 0))
    return x - x.shift(n)


def _rz(s: pd.Series, n: int) -> pd.Series:
    return pd.Series(robust_z(s.to_numpy(dtype=np.float32)[:, None], n)[:, 0], index=s.index)


def _slog(s: pd.Series, scale: float) -> pd.Series:
    """Signed log for flow amounts that change sign: sign(x) * ln(1 + |x| / scale)."""
    return np.sign(s) * np.log1p(s.abs() / scale)


def build(levels: pd.DataFrame, market: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Return (array [T, 16, N_CH + 1], channel names per node). Rows follow `market.index` (sessions)."""
    idx = market.index
    L = levels.reindex(idx)

    def g(name: str) -> pd.Series:
        return L[name] if name in L.columns else pd.Series(np.nan, index=idx)

    m = market
    sig = m["m_sigma"]
    ell = m["mkt_ell"].where(m["mkt_ell"].notna(), m["m_ell"])
    last_change = g("RBI_REPO").diff().replace(0.0, np.nan)
    chg_row = pd.Series(np.where(last_change.notna(), np.arange(len(idx)), np.nan), index=idx).ffill()
    feats: dict[str, dict[str, pd.Series]] = {
        "MARKET": {"z": m["m_z"], "v": m["m_v"], **{f"mom{n}": ell.rolling(n, min_periods=n // 2).sum() / (sig * np.sqrt(n)) for n in (5, 20, 60)},
                   "breadth": m["mkt_breadth"], "dispersion": m["mkt_dispersion"], "ln_sigma": np.log(sig)},
        "VIX_IN": {"ln": np.log(m["vix"].where(m["vix"] > 0)), "dln1": _dln(m["vix"], 1), "z252": _rz(np.log(m["vix"].where(m["vix"] > 0)), 252)},
        "USDINR": {**{f"dln{n}": _dln(g("DEXINUS"), n) for n in (1, 5, 20)},
                   "ln_rv20": np.log(_dln(g("DEXINUS"), 1).rolling(20, min_periods=10).std().replace(0.0, np.nan))},
        "OIL": {**{f"brent_dln{n}": _dln(g("DCOILBRENTEU"), n) for n in (1, 5, 20)}, **{f"wti_dln{n}": _dln(g("DCOILWTICO"), n) for n in (1, 5, 20)}},
        "US_RATES": {"dgs2": g("DGS2"), "dgs10": g("DGS10"), "dff": g("DFF"), "t10y2y": g("T10Y2Y"),
                     **{f"d_dgs10_{n}": g("DGS10").diff(n) for n in (1, 5, 20)}, "d_dgs2_5": g("DGS2").diff(5)},
        "USD": {**{f"twex_dln{n}": _dln(g("DTWEXBGS"), n) for n in (1, 5, 20)}, "cny_dln1": _dln(g("DEXCHUS"), 1), "cny_dln5": _dln(g("DEXCHUS"), 5),
                "jpy_dln1": _dln(g("DEXJPUS"), 1), "jpy_dln5": _dln(g("DEXJPUS"), 5), "eur_dln5": _dln(g("DEXUSEU"), 5)},
        "US_EQ": {**{f"nasdaq_dln{n}": _dln(g("NASDAQCOM"), n) for n in (1, 5, 20)}, **{f"sp500_dln{n}": _dln(g("SP500"), n) for n in (1, 5, 20)},
                  "djia_dln1": _dln(g("DJIA"), 1), "djia_dln5": _dln(g("DJIA"), 5)},
        "ASIA_EQ": {f"nikkei_dln{n}": _dln(g("NIKKEI225"), n) for n in (1, 5, 20)},
        "GLOBAL_VOL": {"ln_vix": np.log(g("VIXCLS").where(g("VIXCLS") > 0)), "dln1": _dln(g("VIXCLS"), 1), "dln5": _dln(g("VIXCLS"), 5)},
        "CREDIT": {"hy": g("BAMLH0A0HYM2"), "ig": g("BAMLC0A0CM"), "d_hy5": g("BAMLH0A0HYM2").diff(5), "d_hy20": g("BAMLH0A0HYM2").diff(20),
                   "d_ig5": g("BAMLC0A0CM").diff(5), "d_ig20": g("BAMLC0A0CM").diff(20)},
        "INDIA_RATES": {"repo": g("RBI_REPO"), "ln_sessions_since_change": np.log1p(pd.Series(np.arange(len(idx)), index=idx) - chg_row),
                        "last_change_dir": np.sign(last_change).ffill(), "call_rate": g("IRSTCI01INM156N"), "in10y": g("INDIRLTLT01STM")},
        "INDIA_MACRO": {"cpi_yoy": _dln(g("INDCPIALLMINMEI"), 252), "iip_yoy": _dln(g("INDPROINDMISMEI"), 252), "m3_yoy": _dln(g("MABMM301INM189S"), 252),
                        "exports_yoy": _dln(g("XTEXVA01INM667S"), 252), "imports_yoy": _dln(g("XTIMVA01INM667S"), 252),
                        "fx_res_dln63": _dln(g("TRESEGINM052N"), 63), "ln_epu_in": np.log(g("INDEPUINDXM").where(g("INDEPUINDXM") > 0))},
        "FLOWS": {"fpi_last_month": _slog(g("FPI_EQUITY_NET"), 1000.0),
                  "fpi_3m": _slog(g("FPI_EQUITY_NET") + g("FPI_EQUITY_NET").shift(21) + g("FPI_EQUITY_NET").shift(42), 3000.0)},
        "METALS": {f"{n}_dln21": _dln(g(s), 21) for n, s in [("copper", "PCOPPUSDM"), ("alu", "PALUMUSDM"), ("iron", "PIORECRUSDM"), ("coal", "PCOALAUUSDM")]},
        "AGRI": {f"{n}_dln21": _dln(g(s), 21) for n, s in [("wheat", "PWHEAMTUSDM"), ("rice", "PRICENPQUSDM"), ("sugar", "PSUGAISAUSDM"),
                                                           ("cotton", "PCOTTINDUSDM"), ("rubber", "PRUBBUSDM")]},
        "POLICY_UNC": {"ln_epu_us": np.log(g("USEPUINDXD").where(g("USEPUINDXD") > 0)),
                       "z252": _rz(np.log(g("USEPUINDXD").where(g("USEPUINDXD") > 0)), 252)},
    }
    out = np.full((len(idx), len(NODES), N_CH + 1), np.nan, dtype=np.float32)
    names: list[list[str]] = []
    for j, node in enumerate(NODES):
        ch = feats[node]
        if len(ch) > N_CH:
            raise ValueError(f"global node {node} has {len(ch)} channels > {N_CH}")
        for c, (_, s) in enumerate(ch.items()):
            a = np.array(pd.Series(s, index=idx), dtype=np.float32)      # explicit copy: pandas 3 hands out read-only views
            a[~np.isfinite(a)] = np.nan
            out[:, j, c] = a
        out[:, j, N_CH] = np.isfinite(out[:, j, :len(ch)]).any(axis=1).astype(np.float32)
        names.append(list(ch) + [f"pad{k}" for k in range(len(ch), N_CH)] + ["has_data"])
    return out, names
