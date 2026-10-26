"""Frozen-program comparisons must retain failures and use trusted costs."""
import json

import pytest

from evoharness.analysis import cvrp_benchmark as bench
from evoharness.engine.candidate import EvalResult


def result(cost=785, **kwargs):
    return EvalResult(-100 * (cost - 784) / 784,
                      metrics={"costs": [cost], "seconds": [1.2]}, **kwargs)


def test_exact_costs_override_rounded_display_metrics():
    evaluated = result()
    evaluated.metrics["mean_gap_pct"] = 0.0
    evaluated.metrics["gaps_pct"] = [0.0]
    row = bench.evaluation_record(evaluated, ["A-n32-k5"])
    assert row["status"] == "ok"
    assert row["mean_gap_pct"] == pytest.approx(100 / 784)
    assert row["instances"][0]["bks"] == 784
    assert row["instances"][0]["cost"] == 785


@pytest.mark.parametrize("evaluated", [
    EvalResult(float("-inf"), error="timeout"),
    EvalResult(float("nan")),
    EvalResult(0.0, metrics={"costs": [True]}),
    EvalResult(0.0, metrics={"costs": [785]}),
    EvalResult(0.0),
])
def test_invalid_results_never_become_zero_gap(evaluated):
    row = bench.evaluation_record(evaluated, ["A-n32-k5"])
    assert row["status"] == "error"
    assert row["mean_gap_pct"] is None
    assert row["score"] is None
    json.dumps(row, allow_nan=False)


def test_untrusted_timing_cannot_break_json_export():
    evaluated = result()
    evaluated.metrics["seconds"] = [float("nan")]
    row = bench.evaluation_record(evaluated, ["A-n32-k5"])
    assert row["status"] == "ok"
    assert row["instances"][0]["reported_solver_seconds"] is None
    json.dumps(row, allow_nan=False)


def test_sequential_rotation_checkpointing_and_failed_repeat(tmp_path, monkeypatch):
    monkeypatch.setitem(bench.cvrp._SPLITS, "public", ["A-n32-k5"])
    candidates = [{"name": name, "code": name, "sha256": name} for name in ["seed", "evolved"]]
    seen = []
    output = tmp_path / "benchmark.json"

    def evaluate(code, split):
        # The previous results must already be checkpointed before the next call.
        saved = json.loads(output.read_text())
        assert len(saved["records"]) == len(seen)
        assert all("code" not in c for c in saved["candidates"])
        seen.append((code, split))
        if len(seen) == 3:
            raise RuntimeError("worker crashed")
        return result()

    report = bench.run_benchmark(candidates, ["public"], 2, output, {}, evaluate)
    assert seen == [("seed", "public"), ("evolved", "public"),
                    ("evolved", "public"), ("seed", "public")]
    assert report["model_calls"] == 0
    assert report["status"] == "completed_with_errors"
    seed, evolved = report["summary"]
    assert seed["complete"] is True
    assert seed["median_gap_pct"] == pytest.approx(100 / 784)
    assert evolved["failed_repeats"] == 1
    assert evolved["median_gap_pct"] is None
    assert report["records"][2]["error"] == "RuntimeError: worker crashed"
    assert json.loads(output.read_text()) == report
    assert len(report["config_sha256"]) == 64
    with pytest.raises(FileExistsError):
        bench.run_benchmark(candidates, ["public"], 2, output, {}, evaluate)


def test_candidate_is_read_without_import_and_hashes_change(tmp_path):
    source = tmp_path / "solver.py"
    source.write_text("raise RuntimeError('must never execute on host')\n")
    first = bench.load_candidates([f"evolved={source}"])
    source.write_text("another version\n")
    second = bench.load_candidates([f"evolved={source}"])
    assert first[0]["sha256"] == second[0]["sha256"]
    assert first[1]["sha256"] != second[1]["sha256"]
    with pytest.raises(ValueError, match="unique"):
        bench.load_candidates([f"seed={source}"])


def test_dry_run_does_not_touch_container_or_output(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("dry run must not preflight Docker or call evaluators")

    monkeypatch.setattr(bench, "environment_metadata", unexpected)
    monkeypatch.setattr(bench, "run_benchmark", unexpected)
    output = tmp_path / "uncreated.json"
    assert bench.main(["--dry-run", "--output", str(output)]) == 0
    assert not output.exists()


@pytest.mark.parametrize("fallback", [False, True])
def test_cli_scopes_execution_policy_through_all_evaluations(tmp_path, monkeypatch, fallback):
    from evoharness.execution import policy

    monkeypatch.setattr(bench, "environment_metadata", lambda *args: {})
    monkeypatch.setattr("evoharness.execution.sandbox.docker_image_ready", lambda *a: True)

    def evaluate(*args):
        assert policy.current_policy() == ("task_default" if fallback else "require_container")
        return {"status": "complete"}

    monkeypatch.setattr(bench, "run_benchmark", evaluate)
    argv = ["--output", str(tmp_path / "report.json")]
    if fallback:
        argv.append("--allow-fallback")
    assert bench.main(argv) == 0
    assert policy.current_policy() == "task_default"


def test_public_cli_preserves_benchmark_argument_order(monkeypatch):
    from evoharness import cli

    seen = []
    monkeypatch.setattr(cli, "_run_module", lambda modules, argv: seen.append((modules, argv)) or 0)
    args = ["--splits", "val", "private", "--candidate", "evolved=with spaces.py",
            "--repeats", "3", "--output", "report.json"]
    assert cli.main(["cvrp", "benchmark", *args]) == 0
    assert seen == [(("evoharness.analysis.cvrp_benchmark",), args)]
