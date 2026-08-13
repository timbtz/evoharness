"""Watch the large campaign and record stage/artifact progress."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args(); last = None
    while True:
        state_path = args.run_dir / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        accepted = args.run_dir / "operators/accepted.json"
        physics = args.run_dir / "physics-v2/state.json"
        snapshot = {"ts": time.time(), "stages": state.get("stages", {}),
            "accepted": len(json.loads(accepted.read_text()).get("accepted", [])) if accepted.exists() else None,
            "physics_jobs": len(json.loads(physics.read_text()).get("jobs", {})) if physics.exists() else None,
            "complete": state.get("complete", False),
            "halt": (None if (args.run_dir / "physics-v2/state.json").exists() else state.get("halt"))}
        compact = json.dumps(snapshot, sort_keys=True)
        if compact != last:
            print(compact, flush=True); last = compact
        if snapshot["complete"] or snapshot["halt"]:
            return 0 if snapshot["complete"] else 2
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
