"""Resumable large structural campaign: z.ai design -> physics -> continuation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.structural_census import atomic_json
from experiments.structural_plan_audit import audit


def run_stage(name: str, cmd: list[str], state: dict, state_path: Path,
              log_dir: Path) -> bool:
    if state["stages"].get(name, {}).get("exit") == 0:
        return True
    started = time.time(); log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{name}.log").open("a") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state["stages"][name] = {"exit": proc.returncode, "started": started,
                              "finished": time.time(), "command": cmd}
    atomic_json(state_path, state)
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=ROOT / "runs/structural-big-1")
    ap.add_argument("--proposals", type=int, default=48)
    ap.add_argument("--usd", type=float, default=25.0)
    args = ap.parse_args()
    gate = audit()
    if not gate["large_run_ready"]:
        print("large campaign gate closed: " + json.dumps(gate["counts"]))
        return 3
    run_dir = args.run_dir.resolve(); state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "started": time.time(), "stages": {}}
    atomic_json(state_path, state)
    operators = run_dir / "operators"
    physics = run_dir / "physics-v2"
    continuation = run_dir / "continuation"
    commands = {
        "operators": [sys.executable, str(ROOT / "experiments/structural_operator_campaign.py"),
            "--proposals", str(args.proposals), "--usd", str(args.usd), "--seconds", "21600",
            "--balanced", "--families", "nae,nae_axis,ellipse", "--run-dir", str(operators)],
        "physics": [sys.executable, str(ROOT / "experiments/structural_census.py"),
            "--operators", str(operators / "accepted.json"), "--seed", "7201",
            "--retries", "1", "--promote", "6", "--lf", "--families",
            "nae,nae_axis,ellipse", "--run-dir", str(physics)],
        "continuation": [sys.executable, str(ROOT / "experiments/structural_continuation.py"),
            str(physics / "report.json"), "--run-dir", str(continuation),
            "--max-bands", "3", "--seed", "8201"]}
    for name in ("operators", "physics", "continuation"):
        if not run_stage(name, commands[name], state, state_path, run_dir / "logs"):
            state["halt"] = f"stage {name} failed; inspect log and resume"
            atomic_json(state_path, state); return 2
    state.pop("halt", None)
    state["complete"] = True; state["finished"] = time.time()
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
