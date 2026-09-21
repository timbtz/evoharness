from __future__ import annotations


import pytest







def test_cvrp_seed_and_validator():
    from evoharness.paths import task_data
    if not list(task_data("cvrp").glob("*.vrp")):
        pytest.skip("run evoharness fetch cvrp for the downloaded-data integration test")
    from evoharness.tasks.cvrp.task import TASK
    a = TASK.evaluate(TASK.seed_code(), "train")
    assert a.error is None, a.error
    assert -10.0 <= a.score <= 1.0, (a.score, a.metrics)  # savings + C LS: ~1-6% gap
    b = TASK.evaluate(TASK.seed_code(), "train")
    assert b.error is None and abs(a.score - b.score) < 0.5, (a.score, b.score)
    bad = ("def solve(coords, dist, demand, capacity, deadline, compile_c):\n"
           "    return [[1, 1]] + [[c] for c in range(2, len(demand))]\n")
    r = TASK.evaluate(bad, "train")
    assert r.score == float("-inf") and "once" in (r.error or ""), r.error
