"""Shared fixtures: repo root on sys.path + a deterministic MockLLM (golden tests 5–6)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.ledger import BudgetGuard, Ledger  # noqa: E402


class MockLLM:
    """Drop-in for core.llm.LLM: fixed $0.02/call, deterministic code sequence.

    Variant i shifts best-fit's preferred gap, so behaviours (and scores) differ
    across generations — enough for gates and parent selection to have real work.
    """

    COST = 0.02

    def __init__(self, ledger: Ledger, guard: BudgetGuard, prices: dict | None = None):
        self.ledger, self.guard = ledger, guard
        self.i = 0

    def chat(self, model, messages, temperature, role, max_tokens=4096,
             tools=None, tool_handler=None) -> str:
        self.guard.check()
        self.i += 1
        self.guard.charge(self.COST)
        self.ledger.append({"type": "llm_call", "role": role, "model": model,
                            "usd": self.COST, "mock": True})
        return (
            f"Variant {self.i}: prefer gaps near {self.i % 5}.\n"
            "```python\nimport numpy as np\n"
            f"def priority(item, bins):\n    return -np.abs(bins - item - {self.i % 5})\n```"
        )
