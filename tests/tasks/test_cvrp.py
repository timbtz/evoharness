from __future__ import annotations

import numpy as np
import pytest

from evoharness.engine.candidate import EvalResult
from evoharness.tasks.cvrp import task


@pytest.fixture
def tiny_instance():
    return {
        "name": "tiny", "n": 4, "capacity": 3,
        "coords": [[0, 0], [1, 1], [3, 4], [0, 2]],
        "demand": [0, 1, 2, 1], "bks": 15,
    }


def test_fetch_covers_every_active_split():
    from evoharness.tasks.cvrp.fetch import GALGOS_IDS, PYVRP_NAMES

    downloadable = set(GALGOS_IDS) | set(PYVRP_NAMES)
    required = {name for names in task._SPLITS.values() for name in names}
    assert required <= downloadable, f"missing downloads: {required - downloadable}"


def test_missing_instance_gives_current_fetch_command(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "_INST_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="uv run evoharness fetch cvrp"):
        task._load_vrp("X-n186-k15")


@pytest.mark.parametrize("allowed, expected", [
    ({0, 1, 2, 3}, "3"), ({8, 9}, "8"), ({1}, "1"), (set(), None),
])
def test_default_cpuset_respects_process_affinity(monkeypatch, allowed, expected):
    monkeypatch.setattr(task.os, "sched_getaffinity", lambda _: allowed, raising=False)
    assert task._default_cpuset() == expected


def test_default_cpuset_without_affinity_api(monkeypatch):
    monkeypatch.delattr(task.os, "sched_getaffinity", raising=False)
    assert task._default_cpuset() is None


def test_required_cvrp_runner_never_uses_fallback(monkeypatch):
    monkeypatch.setattr(task, "current_policy", lambda: "require_container")
    monkeypatch.setattr(task, "docker_image_ready", lambda *args: False)

    def forbidden_fallback(*args, **kwargs):
        pytest.fail("required CVRP isolation must not switch to host or generic image")

    monkeypatch.setattr(task, "run_python", forbidden_fallback)
    result = task.run_harness("unused", runner=task._runner)
    assert result.score == float("-inf")
    assert "fallback disabled" in result.error


def test_required_cvrp_runner_stays_closed_after_image_pruning(monkeypatch):
    from evoharness.execution import sandbox

    readiness = iter([True, False, False])
    monkeypatch.setattr(task, "current_policy", lambda: "require_container")
    monkeypatch.setattr(task, "docker_image_ready", lambda *args: next(readiness))
    monkeypatch.setattr(sandbox, "_DOCKER_READY", {task._IMAGE: True})
    failure = sandbox.SandboxResult("", "Unable to find image locally", 125, 0.2, False)
    monkeypatch.setattr(task, "run_python_docker", lambda *args, **kwargs: failure)

    def forbidden_fallback(*args, **kwargs):
        pytest.fail("image pruning must not enable host or generic-image fallback")

    monkeypatch.setattr(task, "run_python", forbidden_fallback)
    # First call detects pruning and cannot rebuild; later calls remain closed.
    assert task._runner("unused") is failure
    assert task._IMAGE not in sandbox._DOCKER_READY
    later = task._runner("unused")
    assert later.returncode != 0 and "fallback disabled" in later.stderr


@pytest.mark.parametrize("routes, message", [
    (None, "list of routes"),
    ([[1, 1], [2, 3]], "exactly once"),
    ([[1], [2]], "exactly once"),
    ([[0, 1], [2, 3]], "customer indices"),
    ([[True], [2, 3]], "customer indices"),
    ([[1.0], [2, 3]], "customer indices"),
    ([[1, 2, 3]], "exceeds capacity"),
])
def test_validator_rejects_invalid_routes(tiny_instance, routes, message):
    error, cost = task._validate_and_cost(tiny_instance, routes)
    assert message in error
    assert cost == 0


def test_validator_recosts_only_visited_edges(monkeypatch, tiny_instance):
    def dense_matrix_forbidden(_):
        raise AssertionError("route validation must not build a dense matrix")

    monkeypatch.setattr(task, "_dmat", dense_matrix_forbidden)
    # nint(0->1)=1, nint(1->2)=4, nint(2->0)=5; singleton 3 costs 4.
    assert task._validate_and_cost(tiny_instance, [[1, 2], [], [3]]) == (None, 14)


def test_edge_recosting_matches_dense_reference():
    rng = np.random.default_rng(7)
    coords = rng.uniform(-50, 50, size=(51, 2))
    inst = {"n": 51, "coords": coords.tolist(), "demand": [0] + [1] * 50,
            "capacity": 10}
    permutation = rng.permutation(np.arange(1, 51)).tolist()
    routes = [permutation[i:i + 10] for i in range(0, 50, 10)]
    dense = task._dmat(inst)
    expected = sum(int(dense[a, b]) for route in routes
                   for a, b in zip([0] + route, route + [0]))
    assert task._validate_and_cost(inst, routes) == (None, expected)


def test_host_score_ignores_candidate_score_and_cost(monkeypatch, tiny_instance):
    monkeypatch.setattr(task, "_inst", lambda _: tiny_instance)
    payload = EvalResult(9999.0, metrics={"solutions": [
        {"routes": [[1, 2], [3]], "cost": 0, "seconds": 0.1},
    ]}, seconds=0.2)
    result = task._score(["tiny"], payload)
    assert result.error is None
    assert result.metrics["costs"] == [14]
    assert result.score == pytest.approx(100 / 15)
    assert result.seconds == 0.2


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
