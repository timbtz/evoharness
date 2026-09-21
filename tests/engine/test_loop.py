from __future__ import annotations

import json

import pytest

from conftest import MockLLM

from evoharness.engine.config import Config

from evoharness.engine.loop import run




TINY = {"max_usd": 1.0, "max_calls": 4, "max_seconds": 300}



def test_budget_guard_stops_run(tmp_path):
    cfg = Config(task="binpacking", seed=7,
                 budget={"max_usd": 0.05, "max_calls": 100, "max_seconds": 300})
    summary = run(cfg, run_dir=tmp_path / "r", llm_factory=MockLLM)
    assert summary["stop_reason"].startswith("budget: max_usd")
    assert summary["usd"] <= 0.05 + MockLLM.COST + 1e-9



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



def test_compile_repair_loop(tmp_path, monkeypatch):
    import evoharness.engine.loop as loop
    from evoharness.engine.candidate import EvalResult

    class _Task:
        name, description, knowledge_path = "stub", "stub task", tmp_path

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
            return ("Idea: break it\n```python\nBROKEN = 1\n```" if len(prompts) == 1
                    else "Idea: fix it\n```python\nFIXED = 1\n```")

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



def test_resume_from_prior_run(tmp_path, monkeypatch):
    import evoharness.engine.loop as loop
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



def test_reasoning_traced(tmp_path):
    class _LLM(MockLLM):
        def chat(self, model, messages, temperature, role, max_tokens=4096,
                 tools=None, tool_handler=None):
            self.guard.check()
            self.guard.charge(self.COST)
            self.i += 1
            return ("Idea: prefer smaller gaps.\nBecause ties waste bins.\n"
                    "```python\nimport numpy as np\n"
                    f"def priority(item, bins):\n    return -np.abs(bins - item - {self.i % 3})\n```")

    cfg = Config(task="binpacking", budget=dict(TINY))
    run(cfg, run_dir=tmp_path / "r", llm_factory=_LLM)
    cands = [json.loads(l) for l in (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()
             if json.loads(l)["type"] == "candidate"]
    gen1 = next(c for c in cands if c["meta"]["gen"] == 1)
    assert gen1["meta"]["idea"] == "prefer smaller gaps."
    assert "Because ties waste bins." in gen1["meta"]["reasoning"]
    assert "```" not in gen1["meta"]["reasoning"]



def test_memory_wiki_review(tmp_path, monkeypatch):
    import evoharness.strategies.memory
    from evoharness.strategies.memory import ReviewedMemory
    monkeypatch.setattr(ReviewedMemory, "REVIEW_EVERY", 3)
    monkeypatch.setattr(evoharness.strategies.memory, "_ROOT", tmp_path)  # task-shared wiki under tmp

    class _LLM(MockLLM):
        def chat(self, model, messages, temperature, role, max_tokens=4096,
                 tools=None, tool_handler=None):
            self.guard.check()
            self.guard.charge(self.COST)
            if role == "reviewer":
                return ("=== FILE: successful-patterns/gap-shift.md ===\n"
                        "Shifting the preferred gap helps.\n"
                        "=== FILE: ../evil.md ===\nnope\n"
                        "=== FILE: index.md ===\n# Memory index\n"
                        "- successful-patterns/gap-shift.md — gap shifting\n")
            self.i += 1
            return ("Idea: shift preferred gap.\n"
                    "```python\nimport numpy as np\n"
                    f"def priority(item, bins):\n    return -np.abs(bins - item - {self.i % 5})\n```")

    cfg = Config(task="binpacking", budget={"max_usd": 1.0, "max_calls": 6, "max_seconds": 300},
                 switches={"feedback": "memory", "gate": "public_only", "search": "greedy",
                           "knowledge": "off", "roles": "single_strong"})
    run(cfg, run_dir=tmp_path / "r", llm_factory=_LLM)
    mem = tmp_path / "memory" / "binpacking"  # task-scoped, shared across runs
    assert (mem / "SCHEMA.md").exists() and (mem / "index.md").exists()
    assert (mem / "successful-patterns" / "gap-shift.md").exists()
    assert not (tmp_path / "r" / "evil.md").exists() and not (mem / "evil.md").exists()
    assert "gap-shift.md" in (mem / "index.md").read_text()
    events = [json.loads(l) for l in (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    reviews = [e for e in events if e["type"] == "memory_review"]
    assert reviews and reviews[0]["files"] == ["successful-patterns/gap-shift.md", "index.md"]



def test_web_researcher(tmp_path, monkeypatch):
    import evoharness.strategies.memory
    import evoharness.strategies.research
    from evoharness.strategies.research import Researcher
    from evoharness.strategies.roles import SingleStrong
    from evoharness.engine.candidate import Candidate, Pool
    from evoharness.engine.ledger import BudgetGuard, Ledger

    with pytest.raises(ValueError, match="requires feedback=memory"):
        Config(switches={"feedback": "score_only", "knowledge": "web"})

    monkeypatch.setattr(evoharness.strategies.memory, "_ROOT", tmp_path)
    monkeypatch.setattr(Researcher, "EVERY", 3)
    monkeypatch.setattr(evoharness.strategies.research, "web_search",
                        lambda q, count=8: [{"title": "Ant colony trails",
                                             "url": "https://x.test/ants", "snippet": "S"}])
    monkeypatch.setattr(evoharness.strategies.research, "fetch_url", lambda u, cap=8000: "PAGE TEXT")

    ledger = Ledger(tmp_path / "r")

    class _LLM(MockLLM):
        def chat(self, model, messages, temperature, role, max_tokens=4096,
                 tools=None, tool_handler=None, rounds=3):
            self.guard.check()
            self.guard.charge(self.COST)
            assert role == "researcher"
            if tools:  # per-question session: exercise both tools through the handler
                assert "https://x.test/ants" in tool_handler("web_search", {"query": "q"})
                assert tool_handler("fetch_url", {"url": "https://x.test/ants"}) == "PAGE TEXT"
                return "notes: pheromone-style edge scoring (https://x.test/ants)"
            if "QUESTION" in messages[-1]["content"]:
                return "QUESTION: how do ant colonies balance route loads?"
            return ("=== FILE: new-ideas/web-ant-routing.md ===\n# Ant routing\n"
                    "Pheromone-decay edge scores could guide ruin selection.\n"
                    "## Sources\n- https://x.test/ants\n"
                    "=== FILE: new-ideas/../evil.md ===\nnope\n"
                    "=== FILE: successful-patterns/web-fake.md ===\nnope\n")

    class _Task:
        name, description, knowledge_path = "binpacking", "stub task", tmp_path

    mem = tmp_path / "memory" / "binpacking"
    (mem / "new-ideas").mkdir(parents=True)
    (mem / "index.md").write_text("# Memory index\n")

    r = Researcher()
    r.bind(_Task(), _LLM(ledger, BudgetGuard(max_usd=1.0, max_calls=20, max_seconds=60)),
           SingleStrong({"strong": "m", "cheap": "m"}), {"reflect": 0.4}, ledger)
    pool = Pool()
    for i in range(3):  # EVERY=3 rejected candidates -> one research session
        r.observe(pool, Candidate(code="x", id=f"c{i}", meta={"gen": i + 1}), accepted=False)

    assert (mem / "new-ideas" / "web-ant-routing.md").exists()
    assert not (mem / "evil.md").exists() and not list((mem / "new-ideas").glob("evil*"))
    assert not (mem / "successful-patterns" / "web-fake.md").exists()
    assert "web-ant-routing.md" in (mem / "index.md").read_text()
    events = [json.loads(l) for l in (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    research = [e for e in events if e["type"] == "research"]
    assert research and research[0]["files"] == ["new-ideas/web-ant-routing.md"]
    assert "https://x.test/ants" in research[0]["sources"]
    assert research[0]["questions"] == ["how do ant colonies balance route loads?"]
    assert r.sessions == 1 and not [e for e in events if e["type"] == "research_error"]



def test_gate_val_tie_rejects_train_regression():
    from evoharness.strategies.gate import HoldoutGate
    from evoharness.engine.candidate import Candidate, Pool

    class _T:
        noise = {"train": 0.1}

        def evaluate(self, code, split):
            raise AssertionError("no evals expected")

    gate = HoldoutGate(_T(), lambda c, pool, split="train": c.score(split))
    pool = Pool()
    parent = Candidate(code="a", id="p", scores={"train": -1.0, "val": -2.0})
    pool.add(parent)
    reg = Candidate(code="b", id="c1", parent_id="p",
                    scores={"train": -1.5, "val": -2.0}, meta={"novelty": 0.5})
    assert gate.accept(reg, pool) is False          # tie on val, train regressed
    ok = Candidate(code="c", id="c2", parent_id="p",
                   scores={"train": -1.05, "val": -2.0}, meta={"novelty": 0.5})
    assert gate.accept(ok, pool) is True            # train within noise band



def test_memory_index_guard(tmp_path, monkeypatch):
    import evoharness.strategies.memory as am
    monkeypatch.setattr(am, "_ROOT", tmp_path)

    class _T:
        name = "guardtask"

    mem = am.ReviewedMemory(_T(), None, None, {})
    mem.bind(tmp_path / "runs" / "r1", None)
    (mem.dir / "new-ideas" / "kept.md").write_text("# kept\n")
    (mem.dir / "new-ideas" / "dropped.md").write_text("# dropped\n")
    (mem.dir / "index.md").write_text("# Memory index\n- new-ideas/kept.md — x\n")
    restored = mem._index_guard()
    assert restored == ["new-ideas/dropped.md"]
    idx = (mem.dir / "index.md").read_text()
    assert "new-ideas/dropped.md" in idx and "index guard" in idx
    assert mem._index_guard() == []  # idempotent
