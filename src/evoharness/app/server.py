"""FastAPI façade: assemble a config in the frontend, launch runs, stream ledger events (SSE).

Run: uv run uvicorn evoharness.app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

from evoharness.paths import STATE_ROOT, PACKAGE_ROOT

_ROOT = STATE_ROOT

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402

from evoharness.engine.config import OBJECTIVES, SWITCHES, TASKS, Config  # noqa: E402
from evoharness.engine.loop import resume_code, run
from evoharness.engine.ledger import Ledger
from evoharness.execution.policy import POLICIES  # noqa: E402

app = FastAPI(title="EvoHarness")
RUNS = _ROOT / "runs"


@app.get("/")
def index():
    return FileResponse(PACKAGE_ROOT / "app" / "web" / "index.html")


@app.get("/api/meta")
def meta():
    return {
        "execution": POLICIES, "switches": SWITCHES, "objectives": OBJECTIVES, "tasks": TASKS,
        "defaults": Config().to_dict(),
    }


@app.post("/api/runs")
def start_run(body: dict):
    try:
        cfg = Config.from_dict(body)
        if cfg.resume_from:
            resume_code(cfg.resume_from, cfg.task)  # fail fast: 422, not a dead run dir
        if cfg.merge_from:
            resume_code(cfg.merge_from, cfg.task)
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e))
    run_id = f"{cfg.task}-s{cfg.seed}-{int(time.time() * 1000) % 10**8}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    threading.Thread(target=_run_in_background, args=(cfg, run_dir), daemon=True).start()
    return {"run_id": run_id}


def _run_in_background(cfg: Config, run_dir: Path) -> None:
    """Persist startup failures so clients receive a terminal event instead of hanging."""
    try:
        run(cfg, run_dir=run_dir)
    except Exception as exc:
        ledger = Ledger(run_dir)
        if not (run_dir / "ledger.jsonl").exists():
            ledger.append({"type": "run_start", "config": cfg.to_dict(), "run_dir": str(run_dir)})
        ledger.append({"type": "run_end", "error": str(exc),
                       "stop_reason": "execution_error", "private": None})


def _summary(run_dir: Path) -> dict:
    out = {"run_id": run_dir.name, "status": "running"}
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        return out
    for line in ledger.read_text().splitlines():
        ev = json.loads(line)
        if ev["type"] == "run_start":
            out["config"] = ev["config"]
        elif ev["type"] == "run_end":
            best = ev.get("private")
            if isinstance(best, float) and not math.isfinite(best):
                best = None  # ledger JSON allows -Infinity; strict API JSON does not
            out.update(status="failed" if ev.get("error") else "done",
                       error=ev.get("error"), best=best, usd=ev.get("usd"),
                       stop_reason=ev.get("stop_reason"))
    return out


@app.get("/api/runs")
def list_runs():
    dirs = sorted((d for d in (RUNS.iterdir() if RUNS.exists() else []) if d.is_dir()), key=lambda d: d.name, reverse=True)
    return [_summary(d) for d in dirs[:50]]


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    d = RUNS / Path(run_id).name
    if not d.is_dir():
        raise HTTPException(404, "no such run")
    (d / "STOP").touch()
    return {"ok": True}


@app.get("/api/runs/{run_id}/events")
def events(run_id: str):
    path = RUNS / Path(run_id).name / "ledger.jsonl"

    def tail():
        pos, idle = 0, 0.0
        while idle < 600:
            if path.exists():
                with path.open() as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    idle = 0.0
                    done = False
                    for line in chunk.splitlines():
                        if line.strip():
                            yield f"data: {line}\n\n"
                            done = done or '"run_end"' in line
                    if done:
                        return
            time.sleep(0.5)
            idle += 0.5
            yield ": keepalive\n\n" if idle and int(idle * 2) % 30 == 0 else ""

    return StreamingResponse(tail(), media_type="text/event-stream")
