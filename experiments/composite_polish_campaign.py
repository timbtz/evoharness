"""Queued empirical test: corrected champion polish, then partial multi-start polish.

The campaign is deliberately sequential on the 4-core / 7.5 GiB host.  It waits
for the older gradrun campaign and evaluator containers to leave, materializes
immutable input boundaries, and launches each JAX/VMEC polish in its own Docker
container and output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "composite_polish_campaign"
ARCHIVE = ROOT / "memory" / "stellar_p2" / "archive.jsonl"
IMAGE = "evoharness-stellar-eval"
CHAMPION_SHA = "ed8a22b423b8"  # gradrun/g2 best.py, private 0.6408556701


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUT / "campaign.log").open("a") as f:
        f.write(line + "\n")


def norm(boundary: dict) -> tuple[np.ndarray, np.ndarray]:
    rc = np.asarray(boundary["r_cos"], float)
    zs = np.asarray(boundary["z_sin"], float)
    scale = rc[0, rc.shape[1] // 2]
    scale = scale if scale > 0.1 else 1.0
    return rc / scale, zs / scale


def pad(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape)
    offset = (shape[1] - array.shape[1]) // 2
    out[:array.shape[0], offset:offset + array.shape[1]] = array
    return out


def distance(a: dict, b: dict) -> float:
    if a.get("n_field_periods") != b.get("n_field_periods"):
        return 1.0
    ar, az = norm(a)
    br, bz = norm(b)
    shape = (max(ar.shape[0], br.shape[0]), max(ar.shape[1], br.shape[1]))
    return float(max(np.abs(pad(ar, shape) - pad(br, shape)).max(),
                     np.abs(pad(az, shape) - pad(bz, shape)).max()))


def honest(row: dict) -> float:
    return float(row["p2"]) - 0.92 * max(0.0, float(row["feasibility"]) - 0.002)


def materialize_inputs(count: int = 3) -> list[dict]:
    rows = [json.loads(line) for line in ARCHIVE.read_text().splitlines()
            if line.strip()]
    eligible = [r for r in rows if r.get("boundary") and r.get("p2")
                and r.get("feasibility") is not None
                and float(r["feasibility"]) <= 0.01]
    champion_rows = [r for r in eligible if r.get("code_sha") == CHAMPION_SHA]
    if not champion_rows:
        raise RuntimeError(f"no archive boundary for champion code {CHAMPION_SHA}")
    # The optimizer's returned train winner is the highest raw-P2 boundary
    # archived under its exact code hash; that program was privately 0.64085567.
    champion = max(champion_rows, key=lambda r: float(r["p2"]))

    chosen: list[dict] = []
    for row in sorted(eligible, key=honest, reverse=True):
        b = row["boundary"]
        if distance(b, champion["boundary"]) < 0.01:
            continue
        if all(distance(b, old["boundary"]) >= 0.01 for old in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} pairwise-distinct feasible starts")

    inputs = OUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest = {"champion": {k: champion.get(k) for k in
                              ("key", "code_sha", "p2", "feasibility")},
                "champion_private_score": 0.6408556701040611,
                "starts": []}
    jobs = [("champion", champion, 17)]
    for index, row in enumerate(chosen, 1):
        name = f"start{index}-{row['key']}"
        jobs.append((name, row, 5))
        manifest["starts"].append({
            "name": name, "key": row["key"], "code_sha": row.get("code_sha"),
            "p2": row["p2"], "feasibility": row["feasibility"],
            "honest": honest(row),
            "distance_from_champion": distance(row["boundary"], champion["boundary"]),
        })
    for name, row, steps in jobs:
        payload = {"boundary": row["boundary"], "archive_key": row["key"],
                   "code_sha": row.get("code_sha"), "archive_p2": row["p2"],
                   "archive_feasibility": row["feasibility"], "steps": steps}
        (inputs / f"{name}.json").write_text(json.dumps(payload, indent=2))
    manifest["input_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(inputs.glob("*.json"))
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return [{"name": name, "steps": steps,
             "input": str((inputs / f"{name}.json").relative_to(ROOT))}
            for name, _row, steps in jobs]


def busy() -> list[str]:
    reasons = []
    p = subprocess.run(["pgrep", "-af", "experiments/gradrun.py --exec-run"],
                       capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        reasons.append("gradrun")
    p = subprocess.run(["docker", "ps", "--filter", "ancestor=" + IMAGE,
                        "--format", "{{.Names}}"], capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        reasons.append("stellar containers")
    return reasons


def run_job(job: dict) -> int:
    run_out = OUT / job["name"]
    run_out.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--rm", "--cpus", "2", "--memory", "5g",
        "--memory-swap", "5g", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-v", f"{ROOT}:/work", "-w", "/work", IMAGE,
        "python", "-u", "experiments/qi_polish.py",
        "--steps", str(job["steps"]), "--boundary-file", job["input"],
        "--out", str(run_out.relative_to(ROOT)),
    ]
    # A watchdog restart must preserve completed multi-hour steps. Only the
    # first launch is fresh; qi_polish resumes from state.json thereafter.
    if not (run_out / "state.json").exists():
        command.insert(command.index("--steps"), "--fresh")
    log(f"starting {job['name']} ({job['steps']} steps)")
    with (run_out / "driver.log").open("a") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    log(f"finished {job['name']} rc={result.returncode}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.status:
        print((OUT / "campaign.log").read_text() if (OUT / "campaign.log").exists()
              else "not started")
        return 0
    jobs = materialize_inputs()
    while busy():
        if args.no_wait:
            log("host busy: " + ", ".join(busy()))
            return 2
        log("waiting for host: " + ", ".join(busy()))
        time.sleep(60)
    for job in jobs:
        state = OUT / job["name"] / "state.json"
        failed = OUT / job["name"] / "FAILED"
        if failed.exists():
            log(f"skipping failed {job['name']}: {failed.read_text().strip()}")
            continue
        if state.exists():
            data = json.loads(state.read_text())
            if len(data.get("trajectory", [])) >= job["steps"] + 1:
                log(f"skipping completed {job['name']}")
                continue
        rc = run_job(job)
        if rc:
            failed.write_text(f"rc={rc}; preserved state for analysis\n")
            log(f"marked {job['name']} failed; continuing to next basin")
    log("campaign complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
