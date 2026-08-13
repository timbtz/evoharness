"""Persistent health watcher for the sequential composite-polish campaign."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "composite_polish_campaign"
LOG = RUN / "watchdog.log"
IMAGE = "evoharness-stellar-eval"
POLL = 60
STALE_SECONDS = 25 * 60   # measured steps are 11-12 min; allow >2x
IDLE_POLLS = 5            # require sustained low CPU, never one noisy sample


def note(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def output(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


def campaign_pid() -> str:
    lines = output(["pgrep", "-af", "experiments/composite_polish_campaign.py"])
    for line in lines.splitlines():
        if "watch_composite_polish" not in line:
            return line.split(maxsplit=1)[0]
    return ""


def containers() -> list[tuple[str, str]]:
    lines = output(["docker", "ps", "--filter", f"ancestor={IMAGE}",
                    "--format", "{{.Names}} {{.Status}}"])
    return [tuple(line.split(maxsplit=1)) for line in lines.splitlines() if line]


def latest_progress() -> tuple[float, str]:
    files = list(RUN.glob("*/state.json")) + list(RUN.glob("*/driver.log"))
    if not files:
        return RUN.stat().st_mtime, "no job files"
    newest = max(files, key=lambda p: p.stat().st_mtime)
    detail = newest.parent.name
    if newest.name == "state.json":
        try:
            state = json.loads(newest.read_text())
            detail += f" step={len(state.get('trajectory', [])) - 1} solves={state.get('solves')}"
        except Exception:
            detail += " state-unreadable"
    return newest.stat().st_mtime, detail


def cpu_percent(name: str) -> float:
    raw = output(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", name])
    try:
        return float(raw.rstrip("%"))
    except ValueError:
        return 0.0


def relaunch() -> None:
    if campaign_pid():
        return
    log = (RUN / "supervisor.log").open("a")
    subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u",
                      str(ROOT / "experiments/composite_polish_campaign.py")],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    note("relaunched composite supervisor in resume mode")


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    idle = 0
    note(f"watchdog started poll={POLL}s stale={STALE_SECONDS}s idle_polls={IDLE_POLLS}")
    while True:
        if "campaign complete" in ((RUN / "campaign.log").read_text()
                                    if (RUN / "campaign.log").exists() else ""):
            note("campaign complete; watchdog exiting")
            return 0
        pid = campaign_pid()
        active = containers()
        mtime, detail = latest_progress()
        age = time.time() - mtime
        cpus = [(name, cpu_percent(name)) for name, _status in active]
        busy = any(cpu >= 1.0 for _name, cpu in cpus)
        idle = 0 if busy or not active else idle + 1
        note(f"heartbeat supervisor={pid or 'none'} {detail} age={age/60:.1f}m "
             f"containers={cpus} idle={idle}/{IDLE_POLLS}")
        if active and age >= STALE_SECONDS and idle >= IDLE_POLLS:
            note("confirmed stall: stopping stale evaluator; completed state is preserved")
            for name, _status in active:
                subprocess.run(["docker", "stop", "-t", "10", name])
            time.sleep(15)
            relaunch()
            idle = 0
        elif not pid and not active:
            relaunch()
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
