"""Shared task harness: run (candidate code + eval snippet) in the sandbox,
read one JSON object from the last stdout line (ReEvo's subprocess pattern, compressed)."""
from __future__ import annotations

import json

from core.candidate import EvalResult
from core.sandbox import run_python


def run_harness(script: str, timeout: float = 30.0, runner=run_python) -> EvalResult:
    """Script contract: print exactly one JSON line last:
    {"score": float, "metrics": {...}} or {"error": "..."}.
    `runner` is any sandbox entry point with run_python's signature (e.g. run_c)."""
    r = runner(script, timeout=timeout)
    if r.timed_out:
        return EvalResult(float("-inf"), error=f"timeout after {timeout}s", seconds=r.seconds)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, ValueError):
        err = (r.stderr or r.stdout or "no output").strip()[-800:]
        return EvalResult(float("-inf"), error=err or "no parseable output", seconds=r.seconds)
    if payload.get("error"):
        return EvalResult(float("-inf"), metrics=payload.get("metrics", {}),
                          error=str(payload["error"])[:800], seconds=r.seconds)
    return EvalResult(float(payload["score"]), metrics=payload.get("metrics", {}), seconds=r.seconds)
