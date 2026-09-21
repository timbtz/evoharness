"""Regression checks for installed resources and explicit execution isolation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from evoharness.engine.config import Config, TASKS
from evoharness.engine.loop import load_task, run
from evoharness.execution import policy, sandbox
from evoharness.strategies.knowledge import TaskKnowledge


def test_all_task_resources_load_without_legacy_source_directories():
    for name in TASKS:
        task = load_task(name)
        assert task.name == name
        assert task.seed_code().strip()
        assert task.description.strip()
        if name != "stellar_p2":
            knowledge = TaskKnowledge(task.knowledge_path)
            assert knowledge.index and knowledge.pages


def test_curated_knowledge_injection_and_bounded_lookup():
    task = load_task("binpacking")
    notes = TaskKnowledge(task.knowledge_path, "tool")
    assert notes.prompt_sections(task, "scoring")
    page = next(iter(notes.pages))
    assert notes.call_tool("read_knowledge", {"path": page}) == notes.pages[page]
    assert "no such page" in notes.call_tool("read_knowledge", {"path": "../.env"})
    assert "read limit" in notes.call_tool("read_knowledge", {"path": page})
    injected = TaskKnowledge(task.knowledge_path).prompt_sections(task, "scoring")
    assert len(injected) == 3  # Index and two ranked pages.


@pytest.mark.parametrize("task", ["binpacking", "stellar_p2"])
def test_missing_required_container_stops_before_model_or_ledger(tmp_path, monkeypatch, task):
    monkeypatch.setattr(sandbox, "docker_image_ready", lambda *args, **kwargs: False)
    def unexpected_llm(*args, **kwargs):
        pytest.fail("missing isolation must fail before a model client is constructed")
    with pytest.raises(RuntimeError, match="Container execution required"):
        run(Config(task=task, execution="require_container"), tmp_path / "run", unexpected_llm)
    assert not (tmp_path / "run" / "ledger.jsonl").exists()


def test_execution_policy_is_scoped_per_thread_and_resets(monkeypatch):
    monkeypatch.setattr(sandbox, "docker_image_ready", lambda *args, **kwargs: True)
    barrier = Barrier(2)
    def worker(value):
        with policy.execution_policy(value, "binpacking"):
            barrier.wait(timeout=5)
            result = policy.current_policy()
        return result, policy.current_policy()
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(worker, policy.POLICIES))
    assert result == [(value, "task_default") for value in policy.POLICIES]


def test_legacy_knowledge_config_maps_to_current_component():
    cfg = Config.from_dict({"switches": {"knowledge": "wiki_fs"}, "wiki_mode": "tool"})
    assert cfg.switches["knowledge"] == "task_notes"
    assert cfg.knowledge_mode == "tool"
    assert "wiki_mode" not in cfg.to_dict()
    with pytest.raises(ValueError, match="unknown switches"):
        Config(switches={"typo": "off"})


def test_seed_only_run_needs_no_provider_credentials(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from evoharness.engine import llm, loop
    from evoharness.engine.candidate import EvalResult
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    task = SimpleNamespace(name="binpacking", description="offline fixture",
        knowledge_path=tmp_path / "knowledge.json", seed_code=lambda: "seed",
        evaluate=lambda code, split: EvalResult(1.0), render=lambda code, result: {})
    monkeypatch.setattr(loop, "load_task", lambda name: task)
    result = run(Config(budget={"max_calls": 0, "max_usd": 0, "max_seconds": 30}),
                 tmp_path / "offline")
    assert result["calls"] == 0
    assert result["private"] == 1.0


def test_api_startup_failure_produces_terminal_status(tmp_path, monkeypatch):
    from evoharness.app import server
    def unavailable(*args, **kwargs):
        raise RuntimeError("required evaluator is absent")
    monkeypatch.setattr(server, "run", unavailable)
    server._run_in_background(Config(execution="require_container"), tmp_path)
    summary = server._summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["stop_reason"] == "execution_error"
    assert summary["error"] == "required evaluator is absent"
