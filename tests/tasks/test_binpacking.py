from __future__ import annotations









def test_binpacking_seed_matches_published():
    from evoharness.tasks.binpacking.task import TASK
    r = TASK.evaluate(TASK.seed_code(), "public")
    assert r.error is None
    assert 3.5 <= r.metrics["excess_pct"] <= 4.5, r.metrics
