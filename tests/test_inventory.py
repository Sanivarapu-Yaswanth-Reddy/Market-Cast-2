from pathlib import Path

import pandas as pd

from marketcast.data.inventory import inventory, longest_missing_run


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_longest_missing_run():
    cal = pd.bdate_range("2024-01-01", periods=10)
    have = cal.delete([3, 4, 5, 8])
    assert longest_missing_run(cal, have) == (3, str(cal[3].date()))


def test_inventory_counts_and_gap(tmp_path):
    raw = tmp_path / "raw"
    days = pd.bdate_range("2012-07-02", periods=20)
    for d in days:
        _touch(raw / "nse/cm_bhavcopy" / str(d.year) / f"cm{d:%Y%m%d}bhav.csv.zip")
        _touch(raw / "nse/indices_close_all" / str(d.year) / f"ind_close_all_{d:%Y%m%d}.csv")
    for d in days.delete([5, 6, 7, 8]):          # a 4-session gap in delivery
        _touch(raw / "nse/delivery_mto" / str(d.year) / f"MTO_{d:%Y%m%d}.DAT")
    inv = inventory(raw, manifest=tmp_path / "none.csv")
    s = inv["summary"].set_index("folder")
    assert s.loc["cm_bhavcopy", "files"] == 20
    assert s.loc["delivery_mto", "longest_gap_sessions"] == 4
    checks = {n: ok for n, ok, _ in inv["checks"]}
    assert checks["delivery_mto: no gap > 3 sessions vs cm"] is False
    assert checks["indices_close_all: no gap > 3 sessions vs cm"] is True
