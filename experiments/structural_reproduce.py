"""Verify a frozen census specification and reproduce it in a clean run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(spec: dict) -> list[str]:
    errors = []
    for rel, expected in spec["source_sha256"].items():
        path = ROOT / rel
        if not path.exists() or sha(path) != expected:
            errors.append(f"source fingerprint differs: {rel}")
    if spec.get("operators"):
        path = ROOT / spec["operators"]
        if not path.exists() or sha(path) != spec["operators_sha256"]:
            errors.append("operator artifact fingerprint differs")
    return errors


def command(spec: dict, run_dir: Path) -> list[str]:
    cmd = [sys.executable, str(ROOT / "experiments/structural_census.py"),
           "--seed", str(spec["seed"]), "--per-arm", str(spec["per_arm"]),
           "--nae-mode", str(spec.get("nae_mode", 2)),
           "--families", str(spec.get("families", "nae,ellipse")),
           "--profile", str(spec.get("profile", "broad")),
           "--retries", str(spec["retries"]), "--promote", str(spec["promote"]),
           "--run-dir", str(run_dir)]
    if spec.get("max_jobs") is not None:
        cmd += ["--max-jobs", str(spec["max_jobs"])]
    if spec.get("lf"):
        cmd.append("--lf")
    if spec.get("operators"):
        cmd += ["--operators", str(ROOT / spec["operators"])]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_run", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    spec = json.loads((args.source_run / "experiment.json").read_text())
    errors = verify(spec)
    if errors:
        print(json.dumps({"verified": False, "errors": errors}))
        return 2
    print(json.dumps({"verified": True, "source_run": str(args.source_run)}))
    if args.verify_only:
        return 0
    if args.run_dir is None:
        ap.error("--run-dir is required unless --verify-only is used")
    return subprocess.run(command(spec, args.run_dir), cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
