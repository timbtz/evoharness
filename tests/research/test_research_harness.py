from __future__ import annotations

import json
import hashlib

import pytest

from evoharness.research.store import (AgentTask, CliAgentRunner, ResearchStore, artifact_claims,
                                   coordinator_prompt, parse_coordinator_plan,
                                   parse_backend_minimums, parse_coordinator_tasks,
                                   synthesis_prompt,
                                   validate_backend_minimums, validate_portfolio)


def task(hypothesis, role="explorer", backend="claude"):
    return AgentTask(hypothesis, hypothesis, role, backend,
                     "Produce one falsifiable analysis with artifact paths.")


def test_portfolio_forces_breadth_and_review():
    with pytest.raises(ValueError, match="hypothesis"):
        validate_portfolio([task("one"), task("one"), task("one")])
    with pytest.raises(ValueError, match="review"):
        validate_portfolio([task("a"), task("b"), task("c"), task("d")])
    validate_portfolio([task("a"), task("b"), task("c"),
                        task("review", "referee", "codex")])


def test_store_is_persistent_and_results_are_hashed(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    created = store.add_tasks([task("a"), task("b")])
    result = store.record_result(created[0].task_id, "measured conclusion",
                                 returncode=0, seconds=1.2, command=["fake"])
    assert result.read_text().startswith("measured")
    resumed = ResearchStore(tmp_path / "run", "objective")
    assert resumed.state["tasks"][created[0].task_id]["status"] == "completed"
    canonical_hash = hashlib.sha256(result.read_bytes()).hexdigest()
    returned = json.loads((result.parent / "return-handoff.json").read_text())
    assert resumed.state["tasks"][created[0].task_id]["result_sha256"] == canonical_hash
    assert returned["result_sha256"] == canonical_hash
    assert "measured conclusion" in resumed.evidence_digest()


def test_parallel_store_instances_merge_status_without_lost_updates(tmp_path):
    first = ResearchStore(tmp_path / "run", "objective")
    created = first.add_tasks([task("a"), task("b")])
    stale = ResearchStore(tmp_path / "run", "objective")
    first.set_status(created[0].task_id, "completed", marker="first")
    stale.set_status(created[1].task_id, "failed", marker="second")
    final = ResearchStore(tmp_path / "run", "objective")
    assert final.state["tasks"][created[0].task_id]["marker"] == "first"
    assert final.state["tasks"][created[1].task_id]["marker"] == "second"
    stale.set_backend_health("codex", "unavailable", "sandbox")
    assert "codex" not in CliAgentRunner(tmp_path, stale).usable()


def test_local_handoff_has_stable_thread_and_idempotent_transfer(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    first, second = store.add_tasks([task("same"), task("same")])
    envelope = store.prepare_handoff(first, "selected evidence")
    retry = store.prepare_handoff(first, "selected evidence")
    other = store.prepare_handoff(second, "different selected evidence")
    assert envelope == retry
    assert envelope["thread_id"] == other["thread_id"]
    assert envelope["idempotency_key"] != other["idempotency_key"]
    assert envelope["resources"][1]["trust"] == "untrusted-context"
    assert store.claim_task(first.task_id)
    assert not store.claim_task(first.task_id)
    with pytest.raises(RuntimeError, match="cannot change"):
        store.prepare_handoff(first, "mutated retry evidence")


def test_coordinator_schema_and_prompt():
    payload = {"rationale": "breadth first", "tasks": [
        {"hypothesis_id": "axis", "title": "Axis method", "role": "explorer",
         "backend": "claude", "instructions": "derive and test", "read_only": True},
        {"hypothesis_id": "ellipse", "title": "Ellipse repair", "role": "experimentalist",
         "backend": "codex", "instructions": "implement diagnostic", "read_only": False},
    ]}
    rationale, tasks = parse_coordinator_tasks(json.dumps(payload))
    assert rationale == "breadth first" and {t.backend for t in tasks} == {"claude", "codex"}
    assert "not a parameter tuner" in coordinator_prompt("goal", "evidence", {"claude"}, 2)
    assert "SHADOW READ-ONLY" in coordinator_prompt("goal", "evidence", {"claude"}, 2)
    assert "UNIQUE_SENTINEL" not in coordinator_prompt(
        "goal", "UNIQUE_SENTINEL", {"claude"}, 2)
    assert "['claude', 'zai']" in coordinator_prompt(
        "goal", "evidence", {"claude", "zai"}, 2,
        required_backends={"claude", "zai"})
    assert "{'zai': 3}" in coordinator_prompt(
        "goal", "evidence", {"claude", "zai"}, 4,
        backend_minimums={"zai": 3})
    assert "second turn" in synthesis_prompt("goal", "evidence")
    assert "UNIQUE_SENTINEL" not in synthesis_prompt("goal", "UNIQUE_SENTINEL")
    assert "highest discriminatory value per cost" in synthesis_prompt("goal", "evidence")


def test_backend_minimums_are_parsed_and_enforced():
    minimums = parse_backend_minimums("zai=3, claude=1")
    assert minimums == {"zai": 3, "claude": 1}
    workers = [task("a", backend="zai"), task("b", backend="zai"),
               task("c", backend="zai"), task("d", backend="claude")]
    validate_backend_minimums(workers, minimums)
    with pytest.raises(ValueError, match="minimums"):
        validate_backend_minimums(workers[:2], minimums)
    with pytest.raises(ValueError, match="positive"):
        parse_backend_minimums("zai=0")


def test_coordinator_can_preregister_discriminating_experiments():
    payload = {
        "rationale": "distinguish mechanisms",
        "hypotheses": [
            {"id": "axis", "title": "Axis", "mechanism": "axis controls QI",
             "prior_weight": 1.0, "lineage": []},
            {"id": "shape", "title": "Shape", "mechanism": "boundary repair controls QI",
             "prior_weight": 1.0, "lineage": []},
        ],
        "experiments": [
            {"id": "tight-aspect", "condition": "aspect <= 10", "metric": "objective_L",
             "estimated_cost": 2.0, "fidelity": "vlf"},
        ],
        "predictions": [
            {"hypothesis_id": "axis", "experiment_id": "tight-aspect", "mean": 10.0,
             "scale": 1.0, "rationale": "axis survives"},
            {"hypothesis_id": "shape", "experiment_id": "tight-aspect", "mean": 5.0,
             "scale": 1.0, "rationale": "repair collapses"},
        ],
        "tasks": [
            {"hypothesis_id": "axis", "title": "Axis", "role": "explorer",
             "backend": "claude", "instructions": "derive", "read_only": True,
             "depends_on": []},
            {"hypothesis_id": "shape", "title": "Shape", "role": "experimentalist",
             "backend": "claude", "instructions": "test", "read_only": True,
             "depends_on": []},
        ],
    }
    rationale, tasks, design = parse_coordinator_plan(json.dumps(payload))
    assert rationale == "distinguish mechanisms" and len(tasks) == 2
    assert design["predictions"][1]["mean"] == 5.0


def test_belief_state_selects_information_per_cost_and_reopens_bad_pool(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    store.register_hypothesis("a", "A", "mechanism A")
    store.register_hypothesis("b", "B", "mechanism B")
    store.register_experiment("strong", "strong discriminator", "objective_L",
                              estimated_cost=2.0)
    store.register_experiment("weak", "weak discriminator", "objective_L",
                              estimated_cost=1.0)
    store.register_experiment("weak", "weak discriminator", "objective_L",
                              estimated_cost=1.0)
    with pytest.raises(ValueError, match="idempotency conflict"):
        store.register_experiment("weak", "changed condition", "objective_L",
                                  estimated_cost=1.0)
    for hypothesis, strong, weak in (("a", 0.0, 0.0), ("b", 10.0, 2.0)):
        store.record_prediction(hypothesis, "strong", strong, scale=1.0)
        store.record_prediction(hypothesis, "weak", weak, scale=1.0)
    selected = store.select_discriminating_experiment()
    assert selected["experiment_id"] == "strong"
    assert selected["score"] == pytest.approx(12.5)
    store.record_observation("strong", 20.0, actual_cost=2.2,
                             evaluator_hash="evaluator-sha", artifact="runs/eval.json")
    assert store.pool_signal()["action"] == "expand"
    resumed = ResearchStore(tmp_path / "run", "objective")
    assert len(resumed.state["observations"]) == 1
    assert "best hypothesis fails held-out" in resumed.evidence_digest()
    store.set_hypothesis_status("a", "retired", "falsified by observation")
    store.revise_hypothesis("b", "mechanism B narrowed", "direction was wrong")
    assert store.state["hypotheses"]["a"]["status"] == "retired"
    assert store.state["hypotheses"]["a"]["posterior_weight"] == 0.0
    assert store.state["hypotheses"]["b"]["posterior_weight"] == 1.0
    assert store.state["hypotheses"]["b"]["revisions"][0]["mechanism"] == "mechanism B"


def test_cli_commands_keep_read_only_workers_sandboxed(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    runner = CliAgentRunner(tmp_path, store)
    claude = task("a")
    codex = task("b", backend="codex")
    assert "Read,Glob,Grep" in runner.command(claude, tmp_path, tmp_path / "out")
    command = runner.command(codex, tmp_path, tmp_path / "out")
    assert command[command.index("-s") + 1] == "read-only"
    assert 'approval_policy="never"' in command


def test_claude_role_model_routing(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    runner = CliAgentRunner(
        tmp_path, store,
        limits=__import__("evoharness.research.store", fromlist=["AgentLimits"]).AgentLimits(
            claude_coordinator_model="opus", claude_worker_model="fable"))
    coordinator = task("coordinator-r1", role="analyst")
    worker = task("hypothesis", role="explorer")
    assert runner.command(coordinator, tmp_path, tmp_path / "out")[-2:] == ["--model", "opus"]
    assert runner.command(worker, tmp_path, tmp_path / "out")[-2:] == ["--model", "fable"]


def test_zai_can_coordinate_from_context_without_repository(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    runner = CliAgentRunner(tmp_path, store)
    coordinator = AgentTask("coord", "coord", "analyst", "zai", "allocate",
                            requires_repository=False)
    assert runner.command(coordinator, tmp_path, tmp_path / "out") == [
        "zai-api", "glm-5.2"]
    created = store.add_tasks([coordinator])[0]
    store.record_result(created.task_id, "See src/evoharness/research/store.py", returncode=0,
                        seconds=1.0, command=["zai-api", "glm-5.2"])
    assert store.state["tasks"][created.task_id]["evidence_quality"] == \
        "context-citation-report"


def test_parallel_dispatch_interleaves_backends(monkeypatch, tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    runner = CliAgentRunner(tmp_path, store)
    seen = []
    monkeypatch.setattr(runner, "run", lambda task, context: seen.append(task.backend) or tmp_path)
    jobs = [task("c1"), task("c2"), task("c3"),
            AgentTask("z", "z", "analyst", "zai", "context synthesis",
                      requires_repository=False)]
    runner.run_parallel(jobs, "evidence", workers=1)
    assert seen == ["claude", "zai", "claude", "claude"]


def test_zai_is_a_persistent_evidence_scout_not_a_shell_backend(tmp_path):
    store = ResearchStore(tmp_path / "run", "objective")
    runner = CliAgentRunner(tmp_path, store)
    zai = task("scout", role="analyst", backend="zai")
    assert runner.command(zai, tmp_path, tmp_path / "out")[0] == "zai-api"
    with pytest.raises(ValueError, match="no repository tools"):
        validate_portfolio([zai])
    validate_portfolio([AgentTask("scout", "scout", "analyst", "zai",
                                  "Synthesize only supplied evidence.",
                                  requires_repository=False)])


def test_missing_artifact_claims_are_not_silently_promoted():
    verified, missing = artifact_claims(
        "See src/evoharness/research/store.py and runs/this-does-not-exist/result.json")
    assert "src/evoharness/research/store.py" in verified
    assert "runs/this-does-not-exist/result.json" in missing
    verified, missing = artifact_claims(
        "See src/evoharness/research/store.py:artifact_claims and src/evoharness/research/")
    assert verified == ["src/evoharness/research/store.py"] and missing == []
    verified, missing = artifact_claims("runs/structural-census{1,2}-*/report.json")
    assert verified == [] and missing == []
