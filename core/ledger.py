"""Append-only JSONL run ledger + hard budget guard (usd / calls / wall-clock)."""
from __future__ import annotations

import json
import time
from pathlib import Path


class BudgetExceeded(Exception):
    pass


class BudgetGuard:
    def __init__(self, max_usd: float, max_calls: int, max_seconds: float):
        self.max_usd, self.max_calls, self.max_seconds = max_usd, max_calls, max_seconds
        self.usd = 0.0
        self.calls = 0
        self.t0 = time.monotonic()

    def charge(self, usd: float | None) -> None:
        self.calls += 1
        if usd is not None:
            self.usd += usd

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def exceeded(self) -> str | None:
        if self.usd >= self.max_usd:
            return f"max_usd ({self.usd:.4f} >= {self.max_usd})"
        if self.calls >= self.max_calls:
            return f"max_calls ({self.calls} >= {self.max_calls})"
        if self.elapsed() >= self.max_seconds:
            return f"max_seconds ({self.elapsed():.0f} >= {self.max_seconds})"
        return None

    def check(self) -> None:
        reason = self.exceeded()
        if reason:
            raise BudgetExceeded(reason)


class Ledger:
    """One JSONL file per run; every event gets a timestamp. Flat data, no DB."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.jsonl"

    def append(self, event: dict) -> None:
        event = {"ts": round(time.time(), 3), **event}
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]
