"""Number validator for the explanation layer (01 §7, 06 P6.4; 00 rule 10).

A language model may only REPHRASE the risk JSON. This validator extracts every number from a draft text and
accepts the text only if each one is a number that appears in the JSON, under the allowed display forms:

    JSON 0.058   -> "0.058", "5.8%", "6%" (percent, rounded to 0 or 1 decimals), and the same without sign
    JSON 1000    -> "1000", "1,000"
    JSON 20      -> "20"
    plus the derived forms "N of 100" / "N in 100" for probabilities (0.21 -> "21 of 100")

Anything else (a number the model made up, or rounded differently) rejects the text, and the caller falls back
to `template_sentence`, which is built from the JSON by code. Ordinals and list markers ("3 holdings" when the
JSON has three, "top 3") must be passed explicitly as `extra_allowed`.
"""
from __future__ import annotations

import re

# Numbers that are part of the contract's own definitions (the 90% range, "of 100", the 5% / 10% drawdown levels).
CONTRACT_CONSTANTS = {"90", "90%", "100", "5%", "10%", "5", "10"}
_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?\s*%?")


def _flatten(obj) -> list[float]:
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _flatten(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _flatten(v)]
    return []


def allowed_forms(values: list[float]) -> set[str]:
    """Every string a JSON number may legitimately appear as (sign-free; signs are checked separately)."""
    out: set[str] = set()
    for v in values:
        a = abs(v)
        for nd in range(0, 7):
            out.add(f"{a:.{nd}f}")
        if a == int(a):
            out.add(str(int(a)))
            out.add(f"{int(a):,}")
        if a <= 1.0:                                   # fractions may be shown as percentages or "N of 100"
            for nd in (0, 1, 2):
                out.add(f"{100 * a:.{nd}f}%")
            out.add(f"{round(100 * a)}")
    return out


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).replace(" ", "").replace("−", "-").lstrip("+-") for m in _NUM.finditer(text)]


def validate(text: str, report: dict, extra_allowed: set[str] | None = None) -> tuple[bool, list[str]]:
    """(ok, offending numbers). ok is True only if every number in the text is traceable to the JSON."""
    ok_forms = allowed_forms(_flatten(report)) | CONTRACT_CONSTANTS | (extra_allowed or set())
    bad = [n for n in extract_numbers(text) if n not in ok_forms and n.replace(",", "") not in ok_forms]
    return (not bad), bad


def template_sentence(report: dict) -> str:
    """Fallback wording built by code from the JSON (always passes `validate`). Portfolio-level only."""
    r, d = report["outcome_range"], report["prob_drawdown"]
    lo, hi = abs(100 * r["p05"]), abs(100 * r["p95"])
    lo_s = f"a loss of {lo:.1f}%" if r["p05"] < 0 else f"a gain of {lo:.1f}%"
    hi_s = f"a gain of {hi:.1f}%" if r["p95"] >= 0 else f"a loss of {hi:.1f}%"
    return (f"Across {report['n_paths']} simulated scenarios of the next {report['horizon_days']} trading days, the portfolio ends between "
            f"{lo_s} and {hi_s} in 90 of 100 cases. The chance of being down 5% or more at some point is {100 * d['5pct']:.1f}%, "
            f"and of being down 10% or more is {100 * d['10pct']:.1f}%. These are ranges from a model, not promises.")
