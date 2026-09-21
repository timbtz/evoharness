"""Persistent Claude/Codex research portfolio above the trusted physics tools.

Default mode is shadow research: workers can inspect the repository but cannot
edit it or enqueue physics. Use --allow-worktrees to let implementation workers
edit isolated git worktrees. Physics remains a separate host-controlled stage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evoharness.paths import REPO_ROOT, RUNS_DIR

from evoharness.research.store import (AgentLimits, AgentTask, CliAgentRunner,
                                   ResearchStore, coordinator_prompt, synthesis_prompt,
                                   parse_backend_minimums, parse_coordinator_plan,
                                   validate_backend_minimums)


DEFAULT_OBJECTIVE = ("Discover a genuinely independent Stellar P2 basin with a reproducible "
                     "official score near 0.70; learn from failures and avoid repeatedly "
                     "polishing the inherited champion.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=RUNS_DIR / "stellar-research-harness-1")
    ap.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--coordinator", choices=("claude", "codex", "zai"), default="claude")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--claude-usd", type=float, default=5.0)
    ap.add_argument("--zai-usd", type=float, default=2.0)
    ap.add_argument("--zai-model", default="glm-5.2")
    ap.add_argument("--model")
    ap.add_argument("--claude-coordinator-model", default=None,
                    help="Claude model/alias for coordinator and synthesis, e.g. opus")
    ap.add_argument("--claude-worker-model", default=None,
                    help="Claude model/alias for repository workers, e.g. fable")
    ap.add_argument("--disable-backends", default="",
                    help="comma-separated provider backends excluded for this host/run")
    ap.add_argument("--require-backends", default="",
                    help="comma-separated backends that must each receive a worker task")
    ap.add_argument("--backend-minimums", default="",
                    help="minimum worker allocation, for example zai=4,claude=1")
    ap.add_argument("--portfolio-json", type=Path,
                    help="pre-registered host portfolio; bypasses the model coordinator")
    ap.add_argument("--evidence-chars", type=int, default=50_000,
                    help="maximum persistent evidence characters supplied to each agent")
    ap.add_argument("--allow-worktrees", action="store_true",
                    help="allow coordinator-selected implementation tasks to edit isolated worktrees")
    ap.add_argument("--plan-only", action="store_true",
                    help="ask the coordinator for a portfolio but do not dispatch workers")
    ap.add_argument("--skip-synthesis", action="store_true",
                    help="do not give the strong coordinator a post-worker adjudication turn")
    ap.add_argument("--walltime-hours", type=float, default=None,
                    help="persistent campaign walltime deadline")
    ap.add_argument("--total-budget-usd", type=float, default=None,
                    help="shared conservative provider reservation cap")
    args = ap.parse_args()
    if args.rounds < 1 or args.tasks < 2 or args.workers < 1 or args.evidence_chars < 2_000:
        ap.error("rounds>=1, tasks>=2, and workers>=1 are required")
    if args.portfolio_json and args.rounds != 1:
        ap.error("a fixed --portfolio-json can only be dispatched for one round")
    store = ResearchStore(args.run_dir, args.objective)
    runtime_path = args.run_dir / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    if args.walltime_hours is not None and "deadline_epoch" not in runtime:
        runtime = {"started_epoch": time.time(),
                   "deadline_epoch": time.time() + args.walltime_hours * 3600,
                   "walltime_hours": args.walltime_hours}
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    # A deadline is opt-in for this invocation; omitting --walltime-hours
    # means the campaign is budget-bound rather than walltime-bound.
    deadline = runtime.get("deadline_epoch") if args.walltime_hours is not None else None
    limits = AgentLimits(args.timeout, args.claude_usd, args.model,
                         args.zai_usd, args.zai_model,
                         args.claude_coordinator_model,
                         args.claude_worker_model, args.total_budget_usd, deadline)
    runner = CliAgentRunner(REPO_ROOT, store, limits)
    disabled = {x.strip() for x in args.disable_backends.split(",") if x.strip()}
    available = runner.usable() - disabled
    required = {x.strip() for x in args.require_backends.split(",") if x.strip()}
    try:
        minimums = parse_backend_minimums(args.backend_minimums)
    except ValueError as exc:
        ap.error(str(exc))
    for backend in required:
        minimums[backend] = max(1, minimums.get(backend, 0))
    if not required <= available:
        raise RuntimeError(f"required backends unavailable: {sorted(required - available)}")
    if not set(minimums) <= available:
        raise RuntimeError(f"minimum backends unavailable: {sorted(set(minimums) - available)}")
    if sum(minimums.values()) > args.tasks:
        ap.error("--tasks must be at least the sum of backend minimums")
    if not args.portfolio_json and args.coordinator not in available:
        raise RuntimeError(f"coordinator CLI unavailable: {args.coordinator}")

    for _ in range(args.rounds):
        if deadline is not None and time.time() >= float(deadline):
            store.event("campaign_stopped", reason="walltime_deadline")
            break
        store.state["round"] = int(store.state.get("round", 0)) + 1
        store.save()
        evidence = store.evidence_digest(args.evidence_chars)
        if args.portfolio_json:
            rationale, proposed, design = parse_coordinator_plan(
                args.portfolio_json.read_text())
            coordinator_task_id = None
        else:
            brief = coordinator_prompt(args.objective, evidence, available, args.tasks,
                                       allow_worktrees=args.allow_worktrees,
                                       required_backends=required,
                                       backend_minimums=minimums)
            coordinator = store.add_tasks([AgentTask(
                f"coordinator-r{store.state['round']}",
                f"Research portfolio round {store.state['round']}", "analyst",
                args.coordinator, brief, read_only=True,
                requires_repository=args.coordinator != "zai")])[0]
            result_path = runner.run(coordinator, evidence)
            rationale, proposed, design = parse_coordinator_plan(result_path.read_text())
            coordinator_task_id = coordinator.task_id
        if len(proposed) != args.tasks:
            raise RuntimeError(
                f"portfolio returned {len(proposed)} tasks; expected {args.tasks}")
        validate_backend_minimums(proposed, minimums)
        for row in design["hypotheses"]:
            store.register_hypothesis(row["id"], row["title"], row["mechanism"],
                                      prior_weight=float(row["prior_weight"]),
                                      lineage=row["lineage"])
        for row in design["experiments"]:
            store.register_experiment(row["id"], row["condition"], row["metric"],
                                      estimated_cost=float(row["estimated_cost"]),
                                      fidelity=row["fidelity"], strict=False)
        for row in design["predictions"]:
            store.record_prediction(row["hypothesis_id"], row["experiment_id"],
                                    float(row["mean"]), scale=float(row["scale"]),
                                    rationale=row["rationale"])
        if not args.allow_worktrees:
            proposed = [AgentTask(task.hypothesis_id, task.title, task.role,
                                  task.backend, task.instructions, read_only=True,
                                  depends_on=task.depends_on,
                                  requires_repository=task.requires_repository)
                        for task in proposed]
        workers = store.add_tasks(proposed)
        store.event("coordinator_decision", round=store.state["round"],
                    coordinator_task=coordinator_task_id, rationale=rationale,
                    source=(str(args.portfolio_json) if args.portfolio_json else "model"),
                    worker_tasks=[task.task_id for task in workers],
                    hypothesis_ids=[row["id"] for row in design["hypotheses"]],
                    experiment_ids=[row["id"] for row in design["experiments"]])
        print(json.dumps({"round": store.state["round"], "rationale": rationale,
                          "tasks": [task.task_id for task in workers],
                          "plan_only": args.plan_only}), flush=True)
        if args.plan_only:
            continue
        runner.run_parallel(workers, store.evidence_digest(args.evidence_chars),
                            min(args.workers, len(workers)))
        if deadline is not None and time.time() >= float(deadline):
            store.event("campaign_stopped", reason="walltime_deadline_after_workers")
            break
        if not args.skip_synthesis:
            synthesis_evidence = store.evidence_digest()
            synthesis = store.add_tasks([AgentTask(
                f"coordinator-synthesis-r{store.state['round']}",
                f"Scientific adjudication round {store.state['round']}", "referee",
                args.coordinator, synthesis_prompt(args.objective, synthesis_evidence),
                read_only=True, requires_repository=args.coordinator != "zai")])[0]
            synthesis_result = runner.run(synthesis, synthesis_evidence)
            store.event("synthesis_completed", round=store.state["round"],
                        synthesis_task=synthesis.task_id,
                        result_path=str(synthesis_result.relative_to(store.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
