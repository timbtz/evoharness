from __future__ import annotations

import json

import pytest







def _stellar_ready():
    from evoharness.tasks.stellar_p2 import task as st
    return st.docker_image_ready(st._IMAGE, st._DOCKERFILE, str(st._DIR))



TRIVIAL_STELLAR = '''
def solve(fm, rng):
    b = fm.seed_nae(aspect_ratio=9.0, max_elongation=4.0, rotational_transform=0.9,
                    mirror_ratio=0.2, n_field_periods=3, max_poloidal_mode=1,
                    max_toroidal_mode=1)
    fm.eval(b)
    return b
'''



def test_stellar_deterministic(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from evoharness.tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 3)
    a = st.TASK.evaluate(TRIVIAL_STELLAR, "train")
    b = st.TASK.evaluate(TRIVIAL_STELLAR, "train")
    assert a.error is None, a.error
    ka = {k: v for k, v in a.metrics.items() if k != "seconds_used"}
    kb = {k: v for k, v in b.metrics.items() if k != "seconds_used"}
    assert (a.score, ka) == (b.score, kb)
    assert a.metrics["evals_used"] == 1 and a.score < 0  # NAE seed is infeasible



def test_stellar_crash_paths(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from evoharness.tasks.stellar_p2 import task as st
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



def test_stellar_budget_metering(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from evoharness.tasks.stellar_p2 import task as st
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



def test_stellar_cpu_kill(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from evoharness.tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "cpu_budget", 5.0)
    monkeypatch.setattr(st, "_SLACK", 40.0)
    r = st.TASK.evaluate("def solve(fm, rng):\n    \n    while True: pass\n", "train")
    assert r.score == float("-inf") and "timeout" in (r.error or ""), r.error



def test_stellar_seed_improves(monkeypatch):
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from evoharness.tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 25)
    r = st.TASK.evaluate(st.TASK.seed_code(), "train")
    assert r.error is None, r.error
    assert r.score > -0.9, (r.score, r.metrics)
    assert r.metrics["evals_used"] == 25



def test_stellar_official_highfid(monkeypatch):
    import os
    if not os.environ.get("STELLAR_SLOW"):
        pytest.skip("set STELLAR_SLOW=1 to run the ~90s official evaluate test")
    if not _stellar_ready():
        pytest.skip("stellar docker image unavailable")
    from evoharness.tasks.stellar_p2 import task as st
    monkeypatch.setitem(st._TRAIN, "max_evals", 3)
    st.TASK.evaluate(TRIVIAL_STELLAR, "train")
    r = st.TASK.evaluate(TRIVIAL_STELLAR, "private")
    assert r.error is None and r.metrics.get("official") is True
    assert r.score == 0.0 and 1.05 < r.metrics["feasibility"] < 1.17, r.metrics



def test_stellar_archive_dedupe(tmp_path, monkeypatch):
    from evoharness.tasks.stellar_p2 import task as st
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



def test_stellar_soft_fail_dominance():
    from evoharness.tasks.stellar_p2 import task as st
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



def test_stellar_hot_restart_killswitch(monkeypatch):
    import ast

    from evoharness.execution.sandbox import SandboxResult
    from evoharness.tasks.stellar_p2 import task as st
    seen = {}

    def fake_runner(script, timeout=30.0, mem_mb=0, cpus=1, extra_files=None):
        seen["script"] = script
        seen["extra_files"] = extra_files
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



def test_stellar_aspect_parity_and_walk():
    import numpy as np

    from evoharness.tasks.stellar_p2 import task as st
    from evoharness.paths import RUNS_DIR
    cache = RUNS_DIR / "diffscore" / "oracle_cache" / "index.jsonl"
    if not cache.exists():
        pytest.skip("oracle corpus not built (evoharness.tasks.stellar_p2.physics.metrics.difftest)")
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



def test_stellar_margin_grad_killswitch(monkeypatch):
    import ast

    from evoharness.execution.sandbox import SandboxResult
    from evoharness.tasks.stellar_p2 import task as st
    seen = {}

    def fake_runner(script, timeout=30.0, mem_mb=0, cpus=1, extra_files=None):
        seen["script"] = script
        seen["extra_files"] = extra_files
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



def test_stellar_solver_grad_killswitch(monkeypatch):
    """STELLAR_FULL_GRAD gates the metric_grad/grad_step tools, their docs AND
    the diffscore files shipped into the sandbox — all three together, so a run
    without the flag is byte-identical to one from before they existed."""
    import ast

    from evoharness.execution.sandbox import SandboxResult
    from evoharness.tasks.stellar_p2 import task as st
    seen = {}

    def fake_runner(script, timeout=30.0, mem_mb=0, cpus=1, extra_files=None):
        seen["script"], seen["extra_files"] = script, extra_files
        return SandboxResult("", "", 1, 0.1, False)

    monkeypatch.setattr(st, "_runner", fake_runner)
    for env, want in (("0", False), ("1", True)):
        monkeypatch.setenv("STELLAR_FULL_GRAD", env)
        st.TASK._train("def solve(fm, rng):\n    return None\n")
        raw = seen["script"].split("json.loads(")[1] \
                            .split(")  # JSON string literal")[0]
        assert json.loads(ast.literal_eval(raw))["full_grad"] is want
        assert (seen["extra_files"] is not None) is want
        doc = st._description()
        assert ("fm.metric_grad" in doc) is want
        if not want:                      # stripped section leaves no marker behind
            assert st._SOLVEGRAD_DOC[0] not in doc and st._SOLVEGRAD_DOC[1] not in doc

    ship = seen["extra_files"]
    assert "evoharness/tasks/stellar_p2/physics/metrics/boundary_grad.py" in ship
    assert "evoharness/__init__.py" in ship             # namespace pkg synthesized
    for rel, src in ship.items():
        compile(src, rel, "exec")                        # every shipped file parses
    # the clean-room verify path never gets the gradient machinery
    assert "metric_grad" not in st._VERIFY and "boundary_grad" not in st._VERIFY



def test_stellar_grad_step_composite_objective():
    """The step ascends honest score (including normalized qi), not raw L."""
    import numpy as np

    from evoharness.tasks.stellar_p2 import task as st
    ns = {"np": np}
    exec(st._GRAD_SRC, ns)
    exec(st._SOLVEGRAD_SRC, ns)

    rng = np.random.default_rng(0)
    shape = (4, 7)
    def blk(a): return {"r_cos": a[0].tolist(), "z_sin": a[1].tolist()}
    gL = rng.normal(size=(2, *shape))
    ga = rng.normal(size=(2, *shape))
    gq = rng.normal(size=(2, *shape))
    grad = {"grad_L": blk(gL), "grad_aspect": blk(ga), "grad_qi": blk(gq),
            "base": {"violations": [0.003, -0.05, -0.0005, -0.01, -0.1]}}

    cap = 3e-5
    step, active = ns["_grad_step_vec"](grad, cap, 0.002)
    assert active == ["aspect"]
    assert abs(np.abs(step).max() - cap) < 1e-18        # cap is exact, not a bound
    flat = ns["_gflat"]
    expected = flat(grad["grad_L"]) / 20 - 0.92 * flat(grad["grad_aspect"])
    expected = expected / np.abs(expected).max() * cap
    assert np.allclose(step, expected)

    grad["base"]["violations"] = [0.001, -0.05, 0.004, -0.01, -0.1]
    step2, active2 = ns["_grad_step_vec"](grad, cap, 0.002)
    assert active2 == ["qi"]
    expected2 = flat(grad["grad_L"]) / 20 - 0.92 * flat(grad["grad_qi"]) / 4
    expected2 = expected2 / np.abs(expected2).max() * cap
    assert np.allclose(step2, expected2)
