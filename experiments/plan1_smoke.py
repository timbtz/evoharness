"""Plan 1 live validation: two tiny stellar_p2 runs in THIS process (never the
:8777 server), identical config, features ON vs kill-switched OFF. Reports
evals/candidate, warm/soft counts, dead-eval rate, $ spent.

Usage: .venv/bin/python experiments/plan1_smoke.py --arm on|off [--usd 0.45]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["on", "off"])
    ap.add_argument("--usd", type=float, default=0.45)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--seconds", type=int, default=4800)
    a = ap.parse_args()

    flag = "1" if a.arm == "on" else "0"
    os.environ["STELLAR_HOT_RESTART"] = flag
    os.environ["STELLAR_SOFT_FAIL"] = flag

    from core.config import Config
    from core import loop

    cfg = Config(
        task="stellar_p2", seed=1,
        switches={"feedback": "score_only", "gate": "holdout",
                  "search": "greedy", "knowledge": "off",
                  "roles": "single_strong"},
        budget={"max_usd": a.usd, "max_calls": a.calls, "max_seconds": a.seconds},
        resume_from="file:experiments/branch_seeds/pinnae.py",
    )
    run_dir = _ROOT / "runs" / f"plan1-smoke-{a.arm}"
    summary = loop.run(cfg, run_dir=run_dir)
    print("RUN_SUMMARY " + json.dumps(
        {k: summary.get(k) for k in ("best_id", "train", "val", "private",
                                     "usd", "calls", "seconds", "stop_reason")},
        default=str), flush=True)

    rows = [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines()]
    cands = [r for r in rows if r.get("type") == "candidate"]
    stats = []
    for r in cands:
        m = (r.get("meta") or {}).get("metrics") or {}
        stats.append({"id": r.get("id"), "train": r["scores"].get("train"),
                      "evals": m.get("evals_used"), "warm": m.get("evals_warm"),
                      "soft": m.get("evals_soft"), "secs": m.get("seconds_used")})
    print("CANDIDATES " + json.dumps(stats, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
