"""The six golden tests (Plan §10). All LLM use is mocked; only task evals cost time."""
from __future__ import annotations

import json

import pytest

from conftest import MockLLM
from core.config import Config
from core.loop import run
from core.sandbox import run_python
from tasks.common import run_harness

TINY = {"max_usd": 1.0, "max_calls": 4, "max_seconds": 300}


# 1. Best-fit seed reproduces published ≈4.0–4.2% excess on Weibull-5k (public split).
def test_binpacking_seed_matches_published():
    from tasks.binpacking.task import TASK
    r = TASK.evaluate(TASK.seed_code(), "public")
    assert r.error is None
    assert 3.5 <= r.metrics["excess_pct"] <= 4.5, r.metrics


# 2. Circle evaluator: trivial grid seed accepted, overlapping configuration rejected.
def test_circles_validator():
    from tasks.circles.task import TASK
    ok = TASK.evaluate(TASK.seed_code(), "train")
    assert ok.error is None and ok.score > 1.0, (ok.score, ok.error)
    bad = ("import numpy as np\n"
           "def pack(n):\n"
           "    a = np.full((n, 3), 0.5)\n"
           "    a[:, 2] = 0.3\n"
           "    return a\n")
    r = TASK.evaluate(bad, "train")
    assert r.error is not None and "overlap" in r.error.lower(), r.error


# 3. TSP NN+2-opt baseline vs known optima: sane gaps, byte-stable across runs.
def test_tsp_baseline_stable():
    from tasks.tsp.task import TASK
    a = TASK.evaluate(TASK.seed_code(), "private")
    b = TASK.evaluate(TASK.seed_code(), "private")
    assert a.error is None, a.error
    assert (a.score, a.metrics) == (b.score, b.metrics)
    assert -15.0 <= a.score < 0.0, a.score  # polished NN typically 5–9% above optimum


# 7. CVRP: seed valid with sane gaps, stable across repeats; invalid solution rejected.
def test_cvrp_seed_and_validator():
    from tasks.cvrp.task import TASK
    a = TASK.evaluate(TASK.seed_code(), "train")
    assert a.error is None, a.error
    assert -10.0 <= a.score <= 1.0, (a.score, a.metrics)  # savings + C LS: ~1-6% gap
    b = TASK.evaluate(TASK.seed_code(), "train")
    assert b.error is None and abs(a.score - b.score) < 0.5, (a.score, b.score)
    bad = ("def solve(coords, dist, demand, capacity, deadline, compile_c):\n"
           "    return [[1, 1]] + [[c] for c in range(2, len(demand))]\n")
    r = TASK.evaluate(bad, "train")
    assert r.score == float("-inf") and "once" in (r.error or ""), r.error


# 4. Sandbox: infinite loop killed at timeout and scored -inf; network access fails.
def test_sandbox_timeout_and_network():
    r = run_python("while True: pass", timeout=2)
    assert r.timed_out and r.seconds < 10
    res = run_harness("while True: pass", timeout=2)
    assert res.score == float("-inf") and "timeout" in res.error

    net = ("import json, urllib.request\n"
           "urllib.request.urlopen('http://example.com', timeout=5)\n"
           "print(json.dumps({'score': 1.0}))\n")
    res = run_harness(net, timeout=15)
    assert res.score == float("-inf")


# 5. Budget guard: run stops on max_usd, overshooting by at most one call.
def test_budget_guard_stops_run(tmp_path):
    cfg = Config(task="binpacking", seed=7,
                 budget={"max_usd": 0.05, "max_calls": 100, "max_seconds": 300})
    summary = run(cfg, run_dir=tmp_path / "r", llm_factory=MockLLM)
    assert summary["stop_reason"].startswith("budget: max_usd")
    assert summary["usd"] <= 0.05 + MockLLM.COST + 1e-9


# 6. Reproducibility: same (config, seed) + same LLM outputs → identical decisions.
def test_reproducible_decisions(tmp_path):
    def decisions(run_dir):
        cfg = Config(task="binpacking", seed=42, budget=dict(TINY),
                     switches={"feedback": "score_only", "gate": "public_only",
                               "search": "islands", "knowledge": "off",
                               "roles": "single_strong"})
        run(cfg, run_dir=run_dir, llm_factory=MockLLM)
        out = []
        for line in (run_dir / "ledger.jsonl").read_text().splitlines():
            ev = json.loads(line)
            if ev["type"] == "candidate":
                out.append((ev["id"], ev.get("parent_id"), ev["accepted"],
                            ev["scores"].get("train")))
        return out

    a = decisions(tmp_path / "a")
    b = decisions(tmp_path / "b")
    assert a == b and len(a) >= 3, (a, b)


# 7. Compile repair: a non-compiling candidate is fixed in-conversation (writer sees
#    the compiler's stderr) and the repaired candidate competes as the generation.
def test_compile_repair_loop(tmp_path, monkeypatch):
    import core.loop as loop
    from core.candidate import EvalResult

    class _Task:
        name, description, wiki_dir = "stub", "stub task", tmp_path

        def seed_code(self):
            return "SEED"

        def evaluate(self, code, split):
            if "BROKEN" in code:
                return EvalResult(float("-inf"),
                                  error="RuntimeError: C compile failed:\nk.c:1:1: error: boom")
            return EvalResult(1.0 if code == "SEED" else 2.0)

        def render(self, code, result):
            return {}

    prompts = []

    class _LLM(MockLLM):
        def chat(self, model, messages, temperature, role, max_tokens=4096,
                 tools=None, tool_handler=None):
            self.guard.check()
            self.guard.charge(self.COST)
            prompts.append(messages[-1]["content"])
            return ("```python\nBROKEN = 1\n```" if len(prompts) == 1
                    else "```python\nFIXED = 1\n```")

    monkeypatch.setattr(loop, "load_task", lambda name: _Task())
    cfg = Config(budget={"max_usd": 1.0, "max_calls": 2, "max_seconds": 60})
    summary = loop.run(cfg, run_dir=tmp_path / "r", llm_factory=_LLM)

    assert "failed to compile" in prompts[1] and "k.c:1:1" in prompts[1]
    cands = [json.loads(l) for l in (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()
             if json.loads(l)["type"] == "candidate"]
    by_id = {c["id"]: c for c in cands}
    assert by_id["c0001"]["accepted"] is False and "compile failed" in by_id["c0001"]["meta"]["error"]
    assert by_id["c0001r1"]["accepted"] is True and by_id["c0001r1"]["meta"]["repair"] == 1
    assert summary["best_id"] == "c0001r1"


# 8. Resume: a new run seeded from a prior run's best accepted candidate.
def test_resume_from_prior_run(tmp_path, monkeypatch):
    import core.loop as loop
    monkeypatch.setattr(loop, "_ROOT", tmp_path)
    first = tmp_path / "runs" / "first"
    loop.run(Config(task="binpacking", budget=TINY), run_dir=first, llm_factory=MockLLM)
    best = (first / "best.py").read_text()

    with pytest.raises(ValueError, match="no ledger"):
        loop.resume_code("nope", "binpacking")
    with pytest.raises(ValueError, match="is task"):
        loop.resume_code("first", "tsp")

    second = tmp_path / "runs" / "second"
    loop.run(Config(task="binpacking", budget=TINY, resume_from="first"),
             run_dir=second, llm_factory=MockLLM)
    ev0 = json.loads((second / "ledger.jsonl").read_text().splitlines()[1])
    assert ev0["id"] == "c0000" and ev0["meta"]["resume_from"] == "first"
    assert ev0["code"] == best
