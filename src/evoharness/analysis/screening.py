"""Screen a legacy 2^5 or current valid switch space at a small budget.

The default is the original first-two-options 32-cell grid, preserving old
combo IDs and historical comparisons. ``--space current`` enumerates every
Config-valid current combination (168 cells), including newer axis values.

Usage: uv run python experiments/screening.py --task binpacking --seeds 1 2 3 \\
       --max-usd 0.3 --max-calls 20 [--workers 2]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evoharness.paths import STATE_ROOT, REPO_ROOT, resolve_artifact

_ROOT = STATE_ROOT

from evoharness.engine.config import SWITCHES, Config  # noqa: E402
from evoharness.engine.loop import run  # noqa: E402

AXES = list(SWITCHES)
LEGACY_OPTIONS = {axis: SWITCHES[axis][:2] for axis in AXES}
DEFAULT_OUTPUT_ROOT = _ROOT / "runs" / "screening"


def combos(space: str = "original", task: str = "binpacking"):
    """Yield switch maps in declaration order, validating each before use."""
    if space not in {"original", "current"}:
        raise ValueError(f"unknown screening space: {space}")
    Config(task=task)  # Reject a bad task even when this helper is used directly.
    option_sets = LEGACY_OPTIONS if space == "original" else SWITCHES
    for opts in itertools.product(*(option_sets[axis] for axis in AXES)):
        switches = dict(zip(AXES, opts))
        try:
            Config(task=task, switches=dict(switches))
        except ValueError:
            # The only current cross-axis restriction is web -> memory. Keeping
            # validation here makes new config constraints safe by construction.
            continue
        yield switches


def combo_id(sw: dict) -> str:
    """Stable full-option indices; the legacy grid remains the historical 0/1 ID."""
    return "".join(str(SWITCHES[axis].index(sw[axis])) for axis in AXES)


def one(task: str, sw: dict, seed: int, budget: dict,
        output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict:
    cfg = Config(task=task, seed=seed, switches=dict(sw), budget=budget)
    run_dir = output_root / task / f"{combo_id(sw)}-s{seed}"
    if (run_dir / "ledger.jsonl").exists():
        print(f"skip {run_dir.name} (exists)")
        return {}
    summary = run(cfg, run_dir=run_dir)
    print(f"{run_dir.name}: private={summary['private']:.4f} usd={summary['usd']} "
          f"({summary['stop_reason']})")
    return summary


def build_jobs(space: str, task: str, seeds: list[int]) -> list[tuple[dict, int]]:
    return [(switches, seed) for switches in combos(space, task) for seed in seeds]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="binpacking")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--max-usd", type=float, default=0.3)
    p.add_argument("--max-calls", type=int, default=20)
    p.add_argument("--max-seconds", type=int, default=900)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--space", choices=("original", "current"), default="original",
                   help="original=first two options per axis (32); current=all valid configs")
    p.add_argument("--dry-run", action="store_true",
                   help="print an offline execution plan; do not call evaluators or models")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="directory containing <task>/<combo>-s<seed> run directories")
    a = p.parse_args(argv)
    # Fail before launching any thread when the task itself is invalid.
    Config(task=a.task)
    jobs = build_jobs(a.space, a.task, a.seeds)
    print(f"{len(jobs)} runs ({len({combo_id(sw) for sw, _ in jobs})} configurations), "
          f"sum of per-run USD caps ${len(jobs) * a.max_usd:.2f}; output={a.output_root}")
    if a.dry_run:
        for switches, seed in jobs:
            print(f"plan {a.task}/{combo_id(switches)}-s{seed} {switches}")
        return 0
    budget = {"max_usd": a.max_usd, "max_calls": a.max_calls,
              "max_seconds": a.max_seconds}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(lambda job: one(a.task, job[0], job[1], budget, a.output_root), jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
