from __future__ import annotations









def test_circles_validator():
    from evoharness.tasks.circles.task import TASK
    ok = TASK.evaluate(TASK.seed_code(), "train")
    assert ok.error is None and ok.score > 1.0, (ok.score, ok.error)
    bad = ("import numpy as np\n"
           "def pack(n):\n"
           "    a = np.full((n, 3), 0.5)\n"
           "    a[:, 2] = 0.3\n"
           "    return a\n")
    r = TASK.evaluate(bad, "train")
    assert r.error is not None and "overlap" in r.error.lower(), r.error
