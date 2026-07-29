"""Offline integration test for the quality_diversity + bandit_fork axes.

Runs the REAL core.loop.run end-to-end against a MockTask + MockLLM in this process:
no z.ai calls, no sandbox, no stellar overlap, no box impact. Validates that the
two new axis objects wire into the loop's gate/search/adopt/prompt_sections
contract and that fork-on-stall actually fires. Not a golden test (mock physics);
keep the underscore prefix.

Run:  .venv/bin/python experiments/_qd_smoke.py
"""
from __future__ import annotations

import py_compile
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.candidate import EvalResult  # noqa: E402
from core import loop  # noqa: E402
from core.config import Config, SWITCHES  # noqa: E402
from core.ledger import Ledger  # noqa: E402
from axes.bandit_search import BanditFork  # noqa: E402


class MockLLM:
    """Returns a different variant each call; charges the guard like LLM._meter."""
    def __init__(self, ledger, guard, prices=None):
        self.guard = guard
        self.n = 0

    def chat(self, model, messages, temperature, role, max_tokens=16384,
             tools=None, tool_handler=None, rounds=3):
        self.guard.charge(None)
        self.n += 1
        return (f"Idea: variant {self.n}\nPrediction: cycles the score\n"
                f"reasoning: probe\n```python\nv = {self.n}\n```")


class MockTask:
    name = "binpacking"
    description = "mock"
    noise: dict = {}
    extra_splits: tuple = ()

    def __init__(self, wiki_dir: Path):
        self.wiki_dir = wiki_dir

    def seed_code(self) -> str:
        return "v = 0"

    def evaluate(self, code: str, split: str) -> EvalResult:
        m = re.search(r"v\s*=\s*(-?\d+)", code)
        n = int(m.group(1)) if m else 0
        return EvalResult(round(-0.05 - 0.001 * (n % 3), 5), seconds=0.0)  # non-monotonic

    def render(self, code, result):
        return {}


def main() -> int:
    failures = []

    for rel in ("axes/qd_gate.py", "axes/bandit_search.py", "experiments/meta_runner.py",
                "experiments/waiter.py", "core/config.py", "core/loop.py"):
        try:
            py_compile.compile(str(ROOT / rel), doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(f"compile {rel}: {e}")

    assert "quality_diversity" in loop._AXES["gate"]
    assert "bandit_fork" in loop._AXES["search"]
    assert "quality_diversity" in SWITCHES["gate"]
    assert "bandit_fork" in SWITCHES["search"]
    print("OK  switches registered (gate.quality_diversity, search.bandit_fork)")

    BanditFork.STALL = 3  # exercise fork-on-stall within a short run
    keep = Path(tempfile.mkdtemp(prefix="qd_smoke_"))
    orig_load = loop.load_task
    loop.load_task = lambda name: MockTask(keep / "wiki")
    try:
        cfg = Config(
            task="binpacking", seed=1,
            switches={"feedback": "score_only", "gate": "quality_diversity",
                      "search": "bandit_fork", "knowledge": "off", "roles": "single_strong"},
            budget={"max_usd": 0.5, "max_calls": 40, "max_seconds": 60},
            generations=8,
        )
        summary = loop.run(cfg, run_dir=keep / "run", llm_factory=MockLLM)
    finally:
        loop.load_task = orig_load
        shutil.rmtree(keep, ignore_errors=True)

    print(f"OK  loop.run completed: best_id={summary.get('best_id')} "
          f"stop={summary.get('stop_reason')} private={summary.get('private')}")
    if not summary.get("best_id"):
        failures.append("run_end has no best_id")
    if not summary.get("stop_reason"):
        failures.append("no stop_reason")

    # ledger was inside `keep`; re-run once more in a dir we keep long enough to inspect
    BanditFork.STALL = 3
    inspect = Path(tempfile.mkdtemp(prefix="qd_inspect_"))
    loop.load_task = lambda name: MockTask(inspect / "wiki")
    try:
        loop.run(Config(task="binpacking", seed=1,
                        switches={"feedback": "score_only", "gate": "quality_diversity",
                                  "search": "bandit_fork", "knowledge": "off",
                                  "roles": "single_strong"},
                        budget={"max_usd": 0.5, "max_calls": 40, "max_seconds": 60},
                        generations=8),
                 run_dir=inspect / "run", llm_factory=MockLLM)
        evs = Ledger(inspect / "run").read()
    finally:
        loop.load_task = orig_load
        shutil.rmtree(inspect, ignore_errors=True)
    cands = [e for e in evs if e.get("type") == "candidate"]
    accepted = [c for c in cands if c.get("accepted")]
    roots = [c for c in cands if (c.get("meta") or {}).get("branch_root")]
    print(f"OK  candidates={len(cands)} accepted={len(accepted)} fork_roots={len(roots)}")
    if not accepted:
        failures.append("QD gate accepted nothing")
    if not roots:
        failures.append("fork-on-stall never fired (no branch_root)")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: qd_gate + bandit_fork integrate and behave as intended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
