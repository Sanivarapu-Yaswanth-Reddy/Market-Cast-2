#!/usr/bin/env python3
"""MarketCast plan tracker.

Reads dev_plan/plan.yaml, checks which tasks are done (marker files in dev_plan/state/)
and which outputs exist, and prints what is ready to work on next.

Usage (from the project root or anywhere):
    python dev_plan/plan_status.py                 # full table + next ready tasks
    python dev_plan/plan_status.py --next          # only the ready tasks
    python dev_plan/plan_status.py --done P1.4 --note "PIT store with as-of API, 12 tests"
    python dev_plan/plan_status.py --undo P1.4
    python dev_plan/plan_status.py --json          # machine-readable status

Gates (G1..G6) can only be marked done with --done and should be marked only after the
owner signs off the gate report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required (it ships with Anaconda): pip install pyyaml")

PLAN_DIR = Path(__file__).resolve().parent
ROOT = PLAN_DIR.parent
STATE = PLAN_DIR / "state"


def load_plan() -> dict:
    with open(PLAN_DIR / "plan.yaml", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    ids = [t["id"] for t in plan["tasks"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        sys.exit(f"duplicate task ids in plan.yaml: {sorted(dupes)}")
    known = set(ids)
    for t in plan["tasks"]:
        for d in t.get("deps", []):
            if d not in known:
                sys.exit(f"task {t['id']} depends on unknown task {d}")
    return plan


def marker(task_id: str) -> Path:
    return STATE / f"{task_id}.done"


def task_status(plan: dict) -> list[dict]:
    done = {t["id"] for t in plan["tasks"] if marker(t["id"]).exists()}
    rows = []
    for t in plan["tasks"]:
        outs = t.get("outputs", []) or []
        present = [o for o in outs if (ROOT / o).exists()]
        if t["id"] in done:
            state = "done"
        elif all(d in done for d in t.get("deps", [])):
            state = "ready"
        else:
            state = "blocked"
        note = ""
        if marker(t["id"]).exists():
            try:
                note = json.loads(marker(t["id"]).read_text(encoding="utf-8")).get("note", "")
            except Exception:
                note = ""
        rows.append({
            "id": t["id"], "phase": t["phase"], "name": t["name"], "state": state,
            "gate": bool(t.get("gate")), "deps": t.get("deps", []),
            "waiting_on": [d for d in t.get("deps", []) if d not in done],
            "outputs_present": f"{len(present)}/{len(outs)}", "missing_outputs":
            [o for o in outs if o not in present], "note": note,
        })
    return rows


def print_table(plan: dict, rows: list[dict], only_next: bool) -> None:
    sym = {"done": "[x]", "ready": "[>]", "blocked": "[ ]"}
    if not only_next:
        for pid, ph in plan["phases"].items():
            prow = [r for r in rows if r["phase"] == pid]
            ndone = sum(r["state"] == "done" for r in prow)
            weeks = f"  weeks {ph['weeks']}" if ph.get("weeks") else ""
            print(f"\n{pid}  {ph['name']}{weeks}   ({ndone}/{len(prow)} done)")
            for r in prow:
                extra = ""
                if r["state"] == "blocked":
                    extra = f"  waits on {', '.join(r['waiting_on'])}"
                elif r["state"] == "done" and r["note"]:
                    extra = f"  - {r['note']}"
                print(f"  {sym[r['state']]} {r['id']:<5} {r['name']:<60} outputs {r['outputs_present']:>5}{extra}")
    ready = [r for r in rows if r["state"] == "ready"]
    print("\nReady now:" if ready else "\nNothing ready: check blockers above.")
    for r in ready:
        hint = " (gate: write the report and get owner sign-off)" if r["gate"] else ""
        print(f"  {r['id']:<5} {r['name']}{hint}")
        if r["missing_outputs"]:
            print(f"        expected outputs: {', '.join(r['missing_outputs'])}")
    total = len(rows)
    ndone = sum(r["state"] == "done" for r in rows)
    print(f"\nOverall: {ndone}/{total} tasks done. Specs: dev_plan/06_BUILD_PLAN.md")


def mark_done(plan: dict, task_id: str, note: str, force: bool) -> None:
    rows = {r["id"]: r for r in task_status(plan)}
    if task_id not in rows:
        sys.exit(f"unknown task {task_id}")
    r = rows[task_id]
    if r["waiting_on"] and not force:
        sys.exit(f"{task_id} still waits on {r['waiting_on']}; use --force to override (and log why)")
    if r["missing_outputs"] and not force:
        sys.exit(f"{task_id} is missing outputs {r['missing_outputs']}; create them or use --force")
    STATE.mkdir(exist_ok=True)
    marker(task_id).write_text(json.dumps({
        "id": task_id, "done_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": note, "forced": force}, indent=2), encoding="utf-8")
    print(f"marked {task_id} done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--next", action="store_true", help="show only ready tasks")
    ap.add_argument("--done", metavar="ID", help="mark a task done")
    ap.add_argument("--note", default="", help="one-line note stored with --done")
    ap.add_argument("--force", action="store_true", help="allow --done with unmet deps or missing outputs")
    ap.add_argument("--undo", metavar="ID", help="remove a done marker")
    ap.add_argument("--json", action="store_true", help="print status as JSON")
    a = ap.parse_args()
    plan = load_plan()
    if a.done:
        mark_done(plan, a.done, a.note, a.force)
        return
    if a.undo:
        m = marker(a.undo)
        if m.exists():
            m.unlink()
            print(f"removed done marker for {a.undo}")
        else:
            print(f"{a.undo} was not marked done")
        return
    rows = task_status(plan)
    if a.json:
        print(json.dumps(rows, indent=2))
        return
    print_table(plan, rows, a.next)


if __name__ == "__main__":
    main()
