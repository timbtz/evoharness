"""Post-hoc analysis over run ledgers: per-run results and axis main effects.

The ledger is the single source of truth (PyVRP splits this into Statistics/Result
objects; here one JSONL file plays both roles and this module is the read side).

Usage: uv run python experiments/report.py runs/screening/binpacking [--csv out.csv]
       (writes markdown to stdout; preserve historical reports and choose a new output path)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from evoharness.paths import STATE_ROOT, REPO_ROOT, resolve_artifact

_ROOT = STATE_ROOT

from evoharness.engine.config import SWITCHES  # noqa: E402

AXES = list(SWITCHES)
# A raw JSON ``-Infinity`` parses as nonfinite in Python, but tools that cannot
# preserve it may render the value as ``-sys.float_info.max``. Treat either
# representation as a failed outcome rather than an astronomically bad score.
_FAILURE_SENTINEL = -1e300


def load_run(run_dir: Path) -> dict | None:
    """One row per finished run: config switches + run_end summary + trajectory stats."""
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        return None
    row = {"run": run_dir.name}
    best, accepted, gens = float("-inf"), 0, 0
    for line in path.read_text().splitlines():
        ev = json.loads(line)
        if ev["type"] == "run_start":
            config = ev["config"]
            row.update(config["switches"], seed=config["seed"],
                       objective=config["objective"], _task=config.get("task"))
        elif ev["type"] == "candidate":
            gens = max(gens, ev.get("meta", {}).get("gen", 0))
            accepted += bool(ev.get("accepted"))
            s = ev.get("scores", {}).get("train")
            if s is not None and s > best:
                best = s
        elif ev["type"] == "run_end":
            row.update(train=ev["train"], private=ev["private"],
                       gap=ev["generalization_gap"], usd=ev["usd"],
                       calls=ev["calls"], seconds=ev["seconds"],
                       stop=ev["stop_reason"])
    if "private" not in row:
        return None  # still running or crashed
    row.update(gens=gens, accepted=accepted)
    return row


def _task_name(row: dict) -> str | None:
    """Accept ``task`` in hand-written rows; ledger rows retain it privately.

    ``_task`` keeps the long-standing per-run CSV schema unchanged while still
    stopping an accidental aggregation across incomparable task objectives.
    """
    task = row.get("_task", row.get("task"))
    return str(task) if task is not None else None


def _assert_one_task(rows: list[dict]) -> None:
    tasks = {_task_name(row) for row in rows}
    if len(tasks) > 1:
        labels = ", ".join(sorted(task if task is not None else "<unknown>" for task in tasks))
        raise ValueError(f"cannot pool main effects across tasks: {labels}")


def _private_stats(rows: list[dict], axis: str, option: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get(axis) == option]
    finite: list[float] = []
    missing = nonfinite = sentinel = 0
    for row in selected:
        value = row.get("private")
        if value is None:
            missing += 1
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            nonfinite += 1
            continue
        if not math.isfinite(value):
            nonfinite += 1
        elif value <= _FAILURE_SENTINEL:
            sentinel += 1
        else:
            finite.append(value)
    return {
        "option": option,
        "mean": statistics.mean(finite) if finite else None,
        "observed": len(selected),
        "finite": len(finite),
        "missing": missing,
        "nonfinite": nonfinite,
        "sentinel": sentinel,
    }


def main_effects(rows: list[dict]) -> list[dict]:
    """Compare every observed option to the first present declared baseline.

    Means use finite private scores only. Counts retain the denominator and
    omitted values so a partial/crashed screening cannot look like a complete
    factorial result. Binary, complete historical screenings keep their former
    A-vs-B means and deltas.
    """
    _assert_one_task(rows)
    out = []
    for axis in AXES:
        stats = [_private_stats(rows, axis, option) for option in SWITCHES[axis]]
        present = [item for item in stats if item["observed"]]
        if len(present) < 2:
            continue
        baseline = present[0]  # SWITCHES declaration order, not row order.
        comparisons = []
        for item in present[1:]:
            delta = None
            if baseline["mean"] is not None and item["mean"] is not None:
                delta = item["mean"] - baseline["mean"]
            comparisons.append({**item, "delta": delta})
        out.append({"axis": axis, "baseline": baseline, "comparisons": comparisons})
    return out


def _number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _sort_private(row: dict) -> float:
    try:
        value = float(row.get("private"))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) and value > _FAILURE_SENTINEL else float("-inf")


def _sample_note(stats: dict) -> str:
    note = (f"n={stats['finite']}/{stats['observed']} finite"
            f"; missing={stats['missing']}; nonfinite={stats['nonfinite']}")
    return note + (f"; failure_sentinel={stats['sentinel']}" if stats["sentinel"] else "")


def _private_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return _number(number) if number > _FAILURE_SENTINEL else "—"


def markdown(rows: list[dict]) -> str:
    _assert_one_task(rows)
    rows = sorted(rows, key=_sort_private, reverse=True)
    lines = [f"# Screening report — {len(rows)} runs", "",
             "## Runs (best finite private score first)", "",
             "| run | " + " | ".join(a[:4] for a in AXES)
             + " | train | private | gap | usd | calls |",
             "|" + "---|" * (len(AXES) + 6)]
    for r in rows:
        lines.append(
            f"| {r.get('run', '—')} | " + " | ".join(str(r.get(a, "?")) for a in AXES)
            + f" | {_number(r.get('train'))} | {_private_number(r.get('private'))}"
            f" | {_number(r.get('gap'))} | {_number(r.get('usd'), 2)} | {r.get('calls', '—')} |")
    lines += ["", "## Main effects (finite mean private score; delta = option − baseline)", ""]
    lines += ["Descriptive marginal means only: failures and missing scores are excluded from means "
              "and retained in the denominators below. Exclusions can break factorial balance; "
              "these differences do not isolate interactions or unequal budgets.", ""]
    effects = main_effects(rows)
    if not effects:
        lines.append("No axis has two observed options.")
    for effect in effects:
        base = effect["baseline"]
        for other in effect["comparisons"]:
            delta = "—" if other["delta"] is None else f"{other['delta']:+.4f}"
            lines.append(
                f"- **{effect['axis']}**: {base['option']} {_number(base['mean'])} vs "
                f"{other['option']} {_number(other['mean'])} → delta {delta} "
                f"({_sample_note(base)}; {_sample_note(other)})")
    return "\n".join(lines) + "\n"


def _csv_rows(rows: list[dict]) -> list[dict]:
    """Do not expose report-only task metadata in the historical CSV schema."""
    return [{key: value for key, value in row.items() if key != "_task"} for row in rows]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="run dirs or parents of run dirs")
    p.add_argument("--csv", help="also write per-run rows to this CSV file")
    args = p.parse_args()
    run_dirs = []
    for d in map(Path, args.dirs):
        run_dirs += [d] if (d / "ledger.jsonl").exists() else sorted(
            path for path in d.iterdir() if path.is_dir())
    rows = [row for row in (load_run(d) for d in run_dirs) if row]
    _assert_one_task(rows)  # Reject incomparable objectives before writing a CSV.
    if args.csv and rows:
        csv_rows = _csv_rows(rows)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({key for row in csv_rows for key in row}))
            w.writeheader()
            w.writerows(csv_rows)
    print(markdown(rows) if rows else "no finished runs found", end="")
