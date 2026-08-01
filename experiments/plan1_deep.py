"""Deep stellar_p2 run in THIS process (never the :8777 server), for the
post-Plan-1 campaign: warm evals + soft-fail ON, deep eval budget via
STELLAR_TRAIN_OVERRIDES (set by the caller), memory feedback + wiki.

Usage: .venv/bin/python experiments/plan1_deep.py --seed 5 \
           --resume file:runs/<run>/best.py --usd 8 [--calls 400]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--usd", type=float, default=8.0)
    ap.add_argument("--calls", type=int, default=400)
    ap.add_argument("--seconds", type=int, default=12 * 3600)
    a = ap.parse_args()

    from core.config import Config
    from core import loop

    cfg = Config(
        task="stellar_p2", seed=a.seed,
        switches={"feedback": "memory", "gate": "holdout", "search": "staged",
                  "knowledge": "wiki_fs", "roles": "single_strong"},
        budget={"max_usd": a.usd, "max_calls": a.calls, "max_seconds": a.seconds},
        resume_from=a.resume,
        stall_stop=25,
    )
    run_dir = _ROOT / "runs" / f"plan1-deep-s{a.seed}-{int(time.time()) % 100000000}"
    summary = loop.run(cfg, run_dir=run_dir)
    print("RUN_SUMMARY " + json.dumps(
        {k: summary.get(k) for k in ("best_id", "train", "val", "private",
                                     "usd", "calls", "seconds", "stop_reason")},
        default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
