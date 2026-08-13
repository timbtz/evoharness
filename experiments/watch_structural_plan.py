"""Low-cost watchdog: audits artifacts and records progress without launching spend."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from structural_plan_audit import audit

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--state", type=Path,
                    default=ROOT / "runs/structural-supervisor/watch-state.json")
    args = ap.parse_args()
    stop = False
    def halt(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)
    previous = None
    while not stop:
        result = audit()
        snapshot = {"ts": time.time(), "counts": result["counts"],
                    "large_run_ready": result["large_run_ready"]}
        if snapshot["counts"] != previous:
            args.state.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.state.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
            tmp.replace(args.state)
            with args.state.with_name("watch.log").open("a") as handle:
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            previous = snapshot["counts"]
        if args.once:
            break
        time.sleep(max(10, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
