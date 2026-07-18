"""Enumerate all 2^5 switch configs × seeds at tiny budget; ledgers land in runs/screening/.

Usage: uv run python experiments/screening.py --task binpacking --seeds 1 2 3 \
       --max-usd 0.3 --max-calls 20 [--workers 2]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import SWITCHES, Config  # noqa: E402
from core.loop import run  # noqa: E402

AXES = list(SWITCHES)


def combos():
    for opts in itertools.product(*(SWITCHES[a] for a in AXES)):
        yield dict(zip(AXES, opts))


def combo_id(sw: dict) -> str:
    return "".join(str(SWITCHES[a].index(sw[a])) for a in AXES)  # e.g. "01011"


def one(task: str, sw: dict, seed: int, budget: dict) -> dict:
    cfg = Config(task=task, seed=seed, switches=dict(sw), budget=budget)
    run_dir = _ROOT / "runs" / "screening" / task / f"{combo_id(sw)}-s{seed}"
    if (run_dir / "ledger.jsonl").exists():
        print(f"skip {run_dir.name} (exists)")
        return {}
    summary = run(cfg, run_dir=run_dir)
    print(f"{run_dir.name}: private={summary['private']:.4f} usd={summary['usd']} "
          f"({summary['stop_reason']})")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="binpacking")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--max-usd", type=float, default=0.3)
    p.add_argument("--max-calls", type=int, default=20)
    p.add_argument("--max-seconds", type=int, default=900)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    budget = {"max_usd": a.max_usd, "max_calls": a.max_calls, "max_seconds": a.max_seconds}
    jobs = [(sw, s) for sw in combos() for s in a.seeds]
    print(f"{len(jobs)} runs, worst-case ${len(jobs) * a.max_usd:.2f}")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(lambda j: one(a.task, j[0], j[1], budget), jobs))
