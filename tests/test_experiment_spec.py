from __future__ import annotations

import json

from conftest import MockLLM
from core.config import Config
from core.loop import run


def test_run_start_is_first_and_contains_versioned_effective_spec(tmp_path):
    cfg = Config(task="binpacking", seed=23,
                 budget={"max_usd": 0.01, "max_calls": 0, "max_seconds": 60})
    run(cfg, run_dir=tmp_path / "run", llm_factory=MockLLM)

    events = [json.loads(line) for line in
              (tmp_path / "run" / "ledger.jsonl").read_text().splitlines()]
    start = events[0]
    spec = start["experiment_spec"]
    assert start["type"] == "run_start"
    assert spec["schema_version"] == 1
    assert spec["config"] == cfg.to_dict()
    assert spec["config"]["budget"] == cfg.budget
    assert spec["config"]["seed"] == 23
    assert spec["task"]["name"] == "binpacking"
    assert set(spec["code"]) == {"commit", "dirty", "dirty_fingerprint"}
    assert set(spec["memory"]) == {"path", "snapshot_hash"}


def test_stellar_spec_records_effective_environment(monkeypatch):
    monkeypatch.setenv("STELLAR_HOT_RESTART_TOL", "0.0025")
    monkeypatch.setenv("STELLAR_FULL_GRAD", "1")
    from tasks.stellar_p2.task import TASK

    spec = TASK.experiment_spec()
    assert spec["environment"]["STELLAR_HOT_RESTART_TOL"] == "0.0025"
    assert spec["environment"]["STELLAR_FULL_GRAD"] == "1"
    assert spec["fidelities"]["private"] == "official"
    assert spec["evaluation_budget"]["max_evals"] > 0
    assert spec["seed_bank"]["mode"] in {"available", "disabled"}
    assert len(spec["evaluator"]["dockerfile_sha256"]) == 64
