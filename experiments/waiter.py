"""Waiter: run the quality-diversity A/B once the stellar campaign goes idle.

WHY: "two stellar runs on 4 vCPUs is the documented way to make every eval time
out", and the box swap is full + coolify tenants crash-loop. So we do NOT overlap
the new-design runs with the stellar campaign. This waiter arms immediately, polls
until the stellar campaign is no longer active, then runs the A/B. It never touches
the stellar server (PID 2735324) or the dag driver (PID 2786924): it only READS
their liveness and the stellar ledger mtimes.

IDLE = (no `dag_campaign.py` process alive) AND (no stellar_p2-* ledger has been
appended in the last GRACE_S seconds). So the A/B fires when the campaign
finishes, is halted by the user, or halts itself on z.ai 429/1113 — all safe.

The A/B is a 2x2 factorial (gate in {holdout, quality_diversity} x search in
{staged, bandit_fork}) on binpacking + tsp, seed 1: 8 runs x ~$0.30. It reuses the
screening methodology so experiments/report.py can read the main effects directly.

Idempotent: a sentinel (runs/qd-ab/.fired) prevents a re-launch from double-running.
Run it detached:
    setsid nohup .venv/bin/python experiments/waiter.py >> runs/qd-ab/waiter.log 2>&1 &
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AB_DIR = ROOT / "runs" / "qd-ab"
SENTINEL = AB_DIR / ".fired"
GRACE_S = 300   # stellar ledger quiet for this long => idle
POLL_S = 60

# 2x2 factorial x 2 cheap tasks, seed 1. ~8 runs x ~$0.30 ~ $2.40.
TASKS = ("binpacking", "tsp")
GATES = ("holdout", "quality_diversity")
SEARCHES = ("staged", "bandit_fork")
SEED = 1
USD = 0.30
CALLS = 40


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def dag_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "dag_campaign.py"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def latest_stellar_write() -> float:
    """Newest mtime of any stellar_p2-* ledger.jsonl (0 if none)."""
    latest = 0.0
    for d in (ROOT / "runs").glob("stellar_p2-*"):
        lj = d / "ledger.jsonl"
        if lj.exists():
            latest = max(latest, lj.stat().st_mtime)
    return latest


def stellar_idle() -> bool:
    if dag_alive():
        return False
    last = latest_stellar_write()
    if last == 0.0:
        return True  # no stellar runs exist at all
    return (time.time() - last) > GRACE_S


def run_ab() -> None:
    AB_DIR.mkdir(parents=True, exist_ok=True)
    py = str(ROOT / ".venv" / "bin" / "python")
    runner = str(ROOT / "experiments" / "meta_runner.py")
    results = []
    total_usd = 0.0
    for task in TASKS:
        for gate in GATES:
            for search in SEARCHES:
                run_id = f"qdab-{task}-{gate}-{search}-s{SEED}"
                log(f"launch {run_id}")
                cmd = [py, runner, "--task", task, "--gate", gate, "--search", search,
                       "--seed", str(SEED), "--usd", str(USD), "--calls", str(CALLS),
                       "--run-id", run_id]
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                summary = ""
                for ln in p.stdout.splitlines():
                    if ln.startswith("RUN_SUMMARY "):
                        summary = ln[len("RUN_SUMMARY "):]
                usd = None
                if summary:
                    try:
                        usd = float(__import__("json").loads(summary).get("usd") or 0.0)
                    except Exception:
                        pass
                if usd:
                    total_usd += usd
                results.append((run_id, p.returncode, summary))
                log(f"done {run_id} rc={p.returncode} usd={usd} "
                    f"{'OK' if p.returncode == 0 else 'ERR: ' + p.stderr[-400:]}")
    (AB_DIR / "RESULT.md").write_text(
        "# QD A/B (2x2 factorial, gate x search, binpacking+tsp, seed 1)\n\n"
        + "\n".join(f"- {rid}: rc={rc} {summ}" for rid, rc, summ in results)
        + f"\n\nTotal spend ≈ ${total_usd:.2f}. Feed runs/qdab-*/ledger.jsonl to "
          "experiments/report.py for the gate x search main-effects table.\n")


def main() -> int:
    AB_DIR.mkdir(parents=True, exist_ok=True)
    if SENTINEL.exists():
        log("sentinel present (.fired) — A/B already ran; exiting without re-running")
        return 0
    log("waiter armed. Polling for stellar campaign idle "
        f"(no dag_campaign.py + no stellar ledger write for {GRACE_S}s)...")
    while not stellar_idle():
        time.sleep(POLL_S)
    log("stellar campaign idle — launching QD A/B (8 runs, ~$2.40)")
    try:
        run_ab()
        SENTINEL.touch()
        log("A/B complete. See runs/qd-ab/RESULT.md")
    except Exception as e:
        log(f"A/B FAILED: {e!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
