from __future__ import annotations


import pytest







def test_tsp_baseline_stable():
    from evoharness.paths import task_data
    if not list(task_data("tsp").glob("*.tsp")):
        pytest.skip("run evoharness fetch tsp for the downloaded-data integration test")
    from evoharness.tasks.tsp.task import TASK
    a = TASK.evaluate(TASK.seed_code(), "private")
    b = TASK.evaluate(TASK.seed_code(), "private")
    assert a.error is None, a.error
    assert (a.score, a.metrics) == (b.score, b.metrics)
    assert -15.0 <= a.score < 0.0, a.score  # polished NN typically 5–9% above optimum
