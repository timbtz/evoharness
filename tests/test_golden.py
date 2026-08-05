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


# 9. Tracing: candidate records carry the writer's Idea line + reasoning preamble.
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


# 10. Memory wiki: scaffold created, reviewer fires on a new best, FILE blocks written,
#     and the review is recorded in the ledger.
def test_memory_wiki_review(tmp_path, monkeypatch):
    import axes.memory
    from axes.memory import MemoryWiki
    monkeypatch.setattr(MemoryWiki, "REVIEW_EVERY", 3)
    monkeypatch.setattr(axes.memory, "_ROOT", tmp_path)  # task-shared wiki under tmp

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


# 12. Web researcher (knowledge=web): fires on the candidate cadence, runs the
#     question -> tool-session -> synthesis map-reduce offline (web mocked), writes only
#     namespaced new-ideas/web-*.md pages, appends them to the index, ledgers provenance.
def test_web_researcher(tmp_path, monkeypatch):
    import axes.memory
    import axes.research
    from axes.research import Researcher
    from axes.roles import SingleStrong
    from core.candidate import Candidate, Pool
    from core.ledger import BudgetGuard, Ledger

    with pytest.raises(ValueError, match="requires feedback=memory"):
        Config(switches={"feedback": "score_only", "knowledge": "web"})

    monkeypatch.setattr(axes.memory, "_ROOT", tmp_path)
    monkeypatch.setattr(Researcher, "EVERY", 3)
    monkeypatch.setattr(axes.research, "web_search",
                        lambda q, count=8: [{"title": "Ant colony trails",
                                             "url": "https://x.test/ants", "snippet": "S"}])
    monkeypatch.setattr(axes.research, "fetch_url", lambda u, cap=8000: "PAGE TEXT")

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
        name, description, wiki_dir = "binpacking", "stub task", tmp_path

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


# ---- stellar_p2 golden tests (Briefing MS0) --------------------------------

def _stellar_ready():
    from tasks.stellar_p2 import task as st
    return st.docker_image_ready(st._IMAGE, st._DOCKERFILE, str(st._DIR))


TRIVIAL_STELLAR = '''
def solve(fm, rng):
    b = fm.seed_nae(aspect_ratio=9.0, max_elongation=4.0, rotational_transform=0.9,
                    mirror_ratio=0.2, n_field_periods=3, max_poloidal_mode=1,
                    max_toroidal_mode=1)
    fm.eval(b)
    return b
'''


# 13a. Deterministic sim: identical double-eval, bit-identical score + metrics.
def test_stellar_deterministic(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 3)
    a = st.TASK.evaluate(TRIVIAL_STELLAR, "train")
    b = st.TASK.evaluate(TRIVIAL_STELLAR, "train")
    assert a.error is None, a.error
    ka = {k: v for k, v in a.metrics.items() if k != "seconds_used"}
    kb = {k: v for k, v in b.metrics.items() if k != "seconds_used"}
    assert (a.score, ka) == (b.score, kb)
    assert a.metrics["evals_used"] == 1 and a.score < 0  # NAE seed is infeasible


# 13b. Crash boundaries: fm.eval survives them in-candidate; a garbage RETURNED
# boundary scores -inf with the simulator's error string.
def test_stellar_crash_paths(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 4)
    inside = '''
def solve(fm, rng):
    bad = {"r_cos": [[0.0, 1.0, 0.0]], "z_sin": [[0.0, 0.0, 0.0]], "n_field_periods": 3}
    assert fm.eval(bad) is None and fm.last_error, "crash must yield None + reason"
    b = fm.seed_nae(aspect_ratio=9.0, max_elongation=4.0, rotational_transform=0.9,
                    mirror_ratio=0.2, n_field_periods=3, max_poloidal_mode=1,
                    max_toroidal_mode=1)
    fm.eval(b)
    return b
'''
    r = st.TASK.evaluate(inside, "train")
    assert r.error is None and r.metrics["evals_used"] == 2, (r.error, r.metrics)
    returned_garbage = '''
def solve(fm, rng):
    return {"r_cos": [[0.0, 1.0, 0.0]], "z_sin": [[0.0, 0.0, 0.0]], "n_field_periods": 3}
'''
    r = st.TASK.evaluate(returned_garbage, "train")
    assert r.score == float("-inf") and r.error, r.error


# 13c. Budget metering: fm.eval stops returning at max_evals.
def test_stellar_budget_metering(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 4)
    greedy = '''
def solve(fm, rng):
    b = fm.seed_nae(aspect_ratio=9.0, max_elongation=4.0, rotational_transform=0.9,
                    mirror_ratio=0.2, n_field_periods=3, max_poloidal_mode=1,
                    max_toroidal_mode=1)
    n = 0
    while fm.eval(b) is not None:
        n += 1
    assert fm.remaining() <= 0 and "exhausted" in fm.last_error
    return b
'''
    r = st.TASK.evaluate(greedy, "train")
    assert r.error is None and r.metrics["evals_used"] == 4, (r.error, r.metrics)


# 13d. CPU kill: a candidate that ignores the deadline is killed at the host
# timeout and scored -inf, like any sandbox timeout.
def test_stellar_cpu_kill(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "cpu_budget", 5.0)
    monkeypatch.setattr(st, "_SLACK", 40.0)
    r = st.TASK.evaluate("def solve(fm, rng):\n    \n    while True: pass\n", "train")
    assert r.score == float("-inf") and "timeout" in (r.error or ""), r.error


# 13e. Seed optimizer improves feasibility beyond the raw NAE portfolio even on
# a reduced budget (raw seeds sit around shaped -0.9..-1.1).
def test_stellar_seed_improves(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 25)
    r = st.TASK.evaluate(st.TASK.seed_code(), "train")
    assert r.error is None, r.error
    assert r.score > -0.9, (r.score, r.metrics)
    assert r.metrics["evals_used"] == 25


# 13f. Official high-fidelity evaluate reproduces the spike-era numbers for the
# known NAE boundary (feasibility ~1.11, score exactly 0). ~90 s: opt-in.
def test_stellar_official_highfid(monkeypatch):
    import os
    if not os.environ.get("STELLAR_SLOW"):
        pytest.skip("set STELLAR_SLOW=1 to run the ~90s official evaluate test")
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 3)
    st.TASK.evaluate(TRIVIAL_STELLAR, "train")
    r = st.TASK.evaluate(TRIVIAL_STELLAR, "private")
    assert r.error is None and r.metrics.get("official") is True
    assert r.score == 0.0 and 1.05 < r.metrics["feasibility"] < 1.17, r.metrics


# 14. Archive: near-frontier boundaries dedupe by rounded-coefficient key.
def test_stellar_archive_dedupe(tmp_path, monkeypatch):
    from tasks.stellar_p2 import task as st
    monkeypatch.setattr(st, "_ARCHIVE", tmp_path / "archive.jsonl")
    t = st._StellarP2Task.__new__(st._StellarP2Task)
    t._bcache, t._arch_keys, t._arch_best = {}, None, float("-inf")
    b = {"r_cos": [[0.0, 1.0, -0.04]], "z_sin": [[0.0, 0.0, -0.15]],
         "n_field_periods": 3}
    b_dup = json.loads(json.dumps(b))
    b_dup["r_cos"][0][2] += 1e-6  # inside the 1e-4 rounding -> same key
    b_new = json.loads(json.dumps(b))
    b_new["r_cos"][0][2] = 0.09
    t._archive([{"shaped": -0.5, "boundary": b}], "aaa", "very_low_fidelity")
    t._archive([{"shaped": -0.4, "boundary": b_dup}], "bbb", "very_low_fidelity")
    t._archive([{"shaped": -0.3, "boundary": b_new}], "ccc", "very_low_fidelity")
    t._archive([{"shaped": -0.6, "boundary": b_new}], "ddd", "very_low_fidelity")
    lines = [json.loads(l) for l in (tmp_path / "archive.jsonl").read_text().splitlines()]
    assert [e["code_sha"] for e in lines] == ["aaa", "ccc"]  # dup + regression dropped


# 15a. Soft-fail penalty (Plan 1): graded, monotone in the force residual, and
# STRICTLY dominated by every converged shaped score (reward-hack guard — a
# candidate that only "improves" residuals of unconverged solves never outranks
# one converged eval).
def test_stellar_soft_fail_dominance():
    from tasks.stellar_p2 import task as st
    ftol = 1e-9
    resids = [1e-9, 1e-7, 1e-4, 1e-1, 1e2, 1e6]
    pens = [st._soft_penalty(r, 0.0, 0.0, ftol) for r in resids]
    assert all(a > b for a, b in zip(pens, pens[1:])), pens  # closer => higher
    assert all(-1001.0 <= p <= -1000.0 for p in pens), pens
    worst_converged = -999.0  # shaped = -feasibility; observed |feas| is O(1)
    assert max(pens) < worst_converged
    # the graded channel is train-only: the clean-room verify template and the
    # final authoritative re-score never see the patch
    assert "_soft_penalty" not in st._VERIFY and "restart_from" not in st._VERIFY


# 15b. Kill-switches restore the pre-Plan-1 solve path: with both env flags off
# the in-container patch never replaces run_vmec, and the cfg the host ships
# reflects the env at call time (no reimport needed).
def test_stellar_hot_restart_killswitch(monkeypatch):
    import ast

    from core.sandbox import SandboxResult
    from tasks.stellar_p2 import task as st
    seen = {}

    def fake_runner(script, timeout=30.0, mem_mb=0, cpus=1):
        seen["script"] = script
        return SandboxResult("", "", 1, 0.1, False)

    def shipped_cfg():
        raw = seen["script"].split("json.loads(")[1] \
                            .split(")  # JSON string literal")[0]
        return json.loads(ast.literal_eval(raw))

    monkeypatch.setattr(st, "_runner", fake_runner)
    monkeypatch.setenv("STELLAR_HOT_RESTART", "0")
    monkeypatch.setenv("STELLAR_SOFT_FAIL", "0")
    st.TASK._train("def solve(fm, rng):\n    return None\n")
    cfg = shipped_cfg()
    assert cfg["hot_restart"] is False and cfg["soft_fail"] is False
    monkeypatch.setenv("STELLAR_HOT_RESTART", "1")
    monkeypatch.setenv("STELLAR_SOFT_FAIL", "1")
    st.TASK._train("def solve(fm, rng):\n    return None\n")
    cfg = shipped_cfg()
    assert cfg["hot_restart"] is True and cfg["soft_fail"] is True
    # gate in the template: patch installs only when a flag is on
    assert "if _HR_ON or _SF_ON:\n    _vu.run_vmec = _run_vmec" in st._TEMPLATE


# 15. Gate tie-band (§6.3): a val-tie with a train regression is rejected.
def test_gate_val_tie_rejects_train_regression():
    from axes.gate import HoldoutGate
    from core.candidate import Candidate, Pool

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


# 16. Memory index guard (§6.1): pages dropped from index.md get re-appended.
def test_memory_index_guard(tmp_path, monkeypatch):
    import axes.memory as am
    monkeypatch.setattr(am, "_ROOT", tmp_path)

    class _T:
        name = "guardtask"

    mem = am.MemoryWiki(_T(), None, None, {})
    mem.bind(tmp_path / "runs" / "r1", None)
    (mem.dir / "new-ideas" / "kept.md").write_text("# kept\n")
    (mem.dir / "new-ideas" / "dropped.md").write_text("# dropped\n")
    (mem.dir / "index.md").write_text("# Memory index\n- new-ideas/kept.md — x\n")
    restored = mem._index_guard()
    assert restored == ["new-ideas/dropped.md"]
    idx = (mem.dir / "index.md").read_text()
    assert "new-ideas/dropped.md" in idx and "index guard" in idx
    assert mem._index_guard() == []  # idempotent


# 17. Plan-2 margin tools (2026-08-05): the exact aspect the train template
# hands candidates must BE the simulator's aspect (parity against the pinned
# oracle corpus, no docker needed), its gradient must match finite differences,
# and the Newton walk must land the normalized violation ON its target.
def test_stellar_aspect_parity_and_walk():
    import numpy as np

    from tasks.stellar_p2 import task as st
    cache = st._ROOT / "runs" / "diffscore" / "oracle_cache" / "index.jsonl"
    if not cache.exists():
        pytest.skip("oracle corpus not built (experiments.diffscore.difftest)")
    rows = [r for r in (json.loads(ln) for ln in cache.read_text().splitlines())
            if "boundary" in r]
    worst = 0.0
    for r in rows:
        b = r["boundary"]
        a, _, _ = st._aspect_full(b["r_cos"], b["z_sin"], b["n_field_periods"])
        worst = max(worst, abs(a - r["aspect_ratio"]))
    assert worst < 1e-11, worst          # measured 1.0e-13 over 196 boundaries

    b = rows[0]["boundary"]
    rc, zs = np.array(b["r_cos"]), np.array(b["z_sin"])
    nfp = b["n_field_periods"]
    a, g_rc, g_zs = st._aspect_full(rc, zs, nfp)
    rng = np.random.default_rng(0)
    d_rc, d_zs = rng.standard_normal(rc.shape), rng.standard_normal(zs.shape)
    scale = np.sqrt((d_rc**2).sum() + (d_zs**2).sum())
    d_rc, d_zs, h = d_rc / scale, d_zs / scale, 1e-6
    fd = (st._aspect_full(rc + h * d_rc, zs + h * d_zs, nfp)[0]
          - st._aspect_full(rc - h * d_rc, zs - h * d_zs, nfp)[0]) / (2 * h)
    assert abs(fd - float((g_rc * d_rc).sum() + (g_zs * d_zs).sum())) \
        / abs(fd) < 1e-6

    for target in (0.002, 0.0):
        wc, ws = st._aspect_walk(rc, zs, nfp, target, 3e-3)
        v = (st._aspect_full(wc, ws, nfp)[0] - 10.0) / 10.0
        assert abs(v - target) < 1e-8, (target, v)
        step = max(np.abs(wc - rc).max(), np.abs(ws - zs).max())
        assert step <= 3e-3 + 1e-12      # trust region holds
    # symmetry-pinned coefficients are never moved by a step
    wc, ws = st._aspect_walk(rc, zs, nfp, 0.005, 3e-3)
    ntor = (rc.shape[1] - 1) // 2
    assert np.array_equal(wc[0, :ntor], rc[0, :ntor])
    assert np.array_equal(ws[0, :ntor + 1], zs[0, :ntor + 1])


# 17b. The tools are a config-gated A/B arm: with STELLAR_MARGIN_GRAD=0 the
# container flag is off (fm raises) and the description never mentions them.
def test_stellar_margin_grad_killswitch(monkeypatch):
    import ast

    from core.sandbox import SandboxResult
    from tasks.stellar_p2 import task as st
    seen = {}

    def fake_runner(script, timeout=30.0, mem_mb=0, cpus=1):
        seen["script"] = script
        return SandboxResult("", "", 1, 0.1, False)

    monkeypatch.setattr(st, "_runner", fake_runner)
    for env, want in (("0", False), ("1", True)):
        monkeypatch.setenv("STELLAR_MARGIN_GRAD", env)
        st.TASK._train("def solve(fm, rng):\n    return None\n")
        raw = seen["script"].split("json.loads(")[1] \
                            .split(")  # JSON string literal")[0]
        assert json.loads(ast.literal_eval(raw))["margin_grad"] is want
    monkeypatch.setenv("STELLAR_MARGIN_GRAD", "0")
    doc = st._description()
    assert "fm.margin_step" not in doc and st._GRAD_DOC[0] not in doc
    monkeypatch.setenv("STELLAR_MARGIN_GRAD", "1")
    assert "fm.margin_step" in st.TASK.description
    # analysis-only guarantee: the clean-room verify template stays untouched
    assert "_aspect_walk" not in st._VERIFY and "margin_grad" not in st._VERIFY
