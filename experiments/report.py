"""Post-hoc analysis over run ledgers: per-run results, per-switch main effects.

The ledger is the single source of truth (PyVRP splits this into Statistics/Result
objects; here one JSONL file plays both roles and this module is the read side).

Usage: uv run python experiments/report.py runs/screening/binpacking [--csv out.csv]
       (writes markdown to stdout; redirect to experiments/RESULTS.md)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import SWITCHES  # noqa: E402

AXES = list(SWITCHES)


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
            row.update(ev["config"]["switches"], seed=ev["config"]["seed"],
                       objective=ev["config"]["objective"])
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


def main_effects(rows: list[dict]) -> list[dict]:
    """Per axis: mean private score with option A vs option B, and the delta."""
    out = []
    for axis in AXES:
        a_opt, b_opt = SWITCHES[axis]
        a = [r["private"] for r in rows if r.get(axis) == a_opt]
        b = [r["private"] for r in rows if r.get(axis) == b_opt]
        if not a or not b:
            continue
        ma, mb = statistics.mean(a), statistics.mean(b)
        out.append({"axis": axis, a_opt: round(ma, 4), b_opt: round(mb, 4),
                    "delta": round(mb - ma, 4), "n": f"{len(a)}/{len(b)}"})
    return out


def markdown(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: -r["private"])
    lines = [f"# Screening report — {len(rows)} runs", "",
             "## Runs (best private first)", "",
             "| run | " + " | ".join(a[:4] for a in AXES)
             + " | train | private | gap | usd | calls |",
             "|" + "---|" * (len(AXES) + 6)]
    for r in rows:
        lines.append(
            f"| {r['run']} | " + " | ".join(str(r.get(a, "?")) for a in AXES)
            + f" | {r['train']:.4f} | {r['private']:.4f} | {r['gap']:.4f}"
            f" | {r['usd']:.2f} | {r['calls']} |")
    lines += ["", "## Main effects (mean private score per option; delta = B − A)", ""]
    for e in main_effects(rows):
        axis = e.pop("axis")
        n = e.pop("n")
        delta = e.pop("delta")
        (a_opt, ma), (b_opt, mb) = e.items()
        lines.append(f"- **{axis}**: {a_opt} {ma} vs {b_opt} {mb} → delta {delta:+} (n={n})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="run dirs or parents of run dirs")
    p.add_argument("--csv", help="also write per-run rows to this CSV file")
    args = p.parse_args()
    run_dirs = []
    for d in map(Path, args.dirs):
        run_dirs += [d] if (d / "ledger.jsonl").exists() else sorted(
            p for p in d.iterdir() if p.is_dir())
    rows = [r for r in (load_run(d) for d in run_dirs) if r]
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    print(markdown(rows) if rows else "no finished runs found", end="")
