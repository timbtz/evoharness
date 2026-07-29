"""Direct-call runner for the quality-diversity / fork optimizer arm.

Launches ONE run of the new design (or any config) as its OWN python process —
NOT via the stellar API server — so the running stellar campaign (server PID +
dag driver) is never touched, restarted, or re-imported. Same pattern as
experiments/screening.py: build a Config, call core.loop.run directly.

Examples (separate process, cheap tasks only):
    .venv/bin/python experiments/meta_runner.py --task binpacking \
        --gate quality_diversity --search bandit_fork --seed 1 --usd 0.3
    .venv/bin/python experiments/meta_runner.py --task binpacking \
        --gate holdout --search staged --seed 1 --usd 0.3   # control arm

The waiter (experiments/waiter.py) calls this in a 2x2 factorial once the stellar
campaign goes idle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from core import loop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one optimizer config directly (separate process).")
    ap.add_argument("--task", required=True, choices=["binpacking", "circles", "tsp", "cvrp"])
    ap.add_argument("--gate", default="holdout")
    ap.add_argument("--search", default="staged")
    ap.add_argument("--feedback", default="score_only")
    ap.add_argument("--knowledge", default="off")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--usd", type=float, default=0.30)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--run-id", default=None,
                    help="run dir name under runs/ (default qd-<task>-<gate>-<search>-s<seed>)")
    args = ap.parse_args()

    cfg = Config(
        task=args.task,
        seed=args.seed,
        switches={"feedback": args.feedback, "gate": args.gate, "search": args.search,
                  "knowledge": args.knowledge, "roles": "single_strong"},
        budget={"max_usd": args.usd, "max_calls": args.calls, "max_seconds": args.seconds},
    )
    run_id = args.run_id or f"qd-{args.task}-{args.gate}-{args.search}-s{args.seed}"
    run_dir = loop._ROOT / "runs" / run_id
    summary = loop.run(cfg, run_dir=run_dir)
    out = {k: summary.get(k) for k in
           ("best_id", "train", "val", "private", "usd", "calls", "seconds", "stop_reason")}
    print("RUN_SUMMARY " + json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
