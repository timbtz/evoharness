"""Closed-loop independent structural discovery: propose -> screen -> learn.

Unlike structural_operator_campaign.py, no later proposal is generated until
the previous proposal has received a fresh bank-disabled physics verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from evoharness.paths import PACKAGE_ROOT, RUNS_DIR

from evoharness.engine.campaign_lock import CampaignLock
from evoharness.engine.ledger import BudgetExceeded, BudgetGuard, Ledger
from evoharness.engine.llm import LLM
from evoharness.tasks.stellar_p2.discovery.boundaries import (BasinRecord, ParetoArchive, Provenance, census_promotable,
                                       construction_acquisition, describe, enrich_metrics, stable_hash,
                                       resolution_signature, successive_halving)
from evoharness.tasks.stellar_p2.discovery.evidence import (choose_family, failure_class,
                                      family_statistics, feedback_evidence)
from evoharness.tasks.stellar_p2.discovery.designer import CONSTRUCTION_FAMILIES, OperatorDesigner
from evoharness.tasks.stellar_p2.discovery.census import (atomic_json, candidate_code,
                                            evaluate_spec, file_sha256)


def _entry_record(entry: dict, fingerprint: str, seed: int) -> BasinRecord:
    record = BasinRecord(
        entry["operator_id"], entry["boundary"], entry["metrics"],
        Provenance(entry["operator_id"], seed, independent=True,
                   public_bank_enabled=False), fingerprint,
        "very_low_fidelity", solves=entry.get("solves", 2))
    describe(record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=RUNS_DIR / "structural-independent-loop-1")
    ap.add_argument("--rounds", type=int, default=48)
    ap.add_argument("--usd", type=float, default=15.0)
    ap.add_argument("--seconds", type=int, default=36000)
    ap.add_argument("--seed", type=int, default=7601)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--critic-model", default="glm-5.2")
    ap.add_argument("--families", default=",".join(sorted(CONSTRUCTION_FAMILIES)))
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--screen-niter", type=int, default=20000)
    ap.add_argument("--promote", type=int, default=6)
    ap.add_argument("--repair-promote", type=int, default=2,
                    help="diverse Pareto stepping stones receiving cheap continuation probes")
    ap.add_argument("--official-confirmations", type=int, default=3)
    args = ap.parse_args()
    if args.rounds < 1 or args.retries not in (0, 1, 2) or args.repair_promote < 0:
        ap.error("rounds must be positive, retries must be 0..2, and repair-promote nonnegative")
    families = {x.strip() for x in args.families.split(",") if x.strip()}
    if not families or not families <= CONSTRUCTION_FAMILIES:
        ap.error("families must be a non-empty subset of " +
                 ",".join(sorted(CONSTRUCTION_FAMILIES)))

    # This is the security boundary: task configuration is frozen at import.
    os.environ["STELLAR_NO_BANK"] = "1"
    os.environ.setdefault("STELLAR_TRAIN_OVERRIDES", json.dumps({
        "max_evals": 2, "cpu_budget": 400.0, "collect_top": 1,
        "workers": 1, "eval_timeout": 150.0}))
    base_niter = os.environ.setdefault("STELLAR_SOFT_FAIL_NITER", "5000")
    from evoharness.tasks.stellar_p2.task import TASK, verify_boundary

    run_dir = args.run_dir.resolve()
    state_path = run_dir / "state.json"
    frozen = {"schema_version": 1, "arm": "independent", "seed": args.seed,
              "rounds": args.rounds, "families": sorted(families),
              "usd": args.usd, "seconds": args.seconds,
              "model": args.model, "critic_model": args.critic_model,
              "retries": args.retries, "screen_niter": args.screen_niter,
              "promote": args.promote,
              "repair_promote": args.repair_promote,
              "official_confirmations": args.official_confirmations,
              "source_sha256": {p: file_sha256(PACKAGE_ROOT / p) for p in (
                  "tasks/stellar_p2/workflows/structural.py",
                  "tasks/stellar_p2/discovery/census.py",
                  "tasks/stellar_p2/discovery/evidence.py",
                  "tasks/stellar_p2/discovery/designer.py",
                  "tasks/stellar_p2/discovery/boundaries.py",
                  "tasks/stellar_p2/task.py")}}
    experiment_path = run_dir / "experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != frozen:
        raise RuntimeError("frozen experiment specification differs; use a new run directory")
    if not experiment_path.exists():
        atomic_json(experiment_path, frozen)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "arm": "independent", "seed": args.seed,
        "started_at": time.time(), "rounds": []}
    if state.get("seed") != args.seed or state.get("arm") != "independent":
        raise RuntimeError("refusing incompatible resume state")

    # Fingerprint the executable contract, not only the container identity:
    # fidelity presets, scoring knobs, bank state, and environment all affect
    # whether two campaign measurements are comparable.
    fingerprint = stable_hash(TASK.experiment_spec())
    if state.get("evaluator_fingerprint", fingerprint) != fingerprint:
        raise RuntimeError("stale evaluator fingerprint; use a new run directory")
    state["evaluator_fingerprint"] = fingerprint

    ledger = Ledger(run_dir)
    guard = BudgetGuard(args.usd, args.rounds * 3, args.seconds)
    # A resumed process inherits call/USD spend rather than silently receiving
    # a fresh allowance. The seconds cap remains per active process so downtime
    # between resumptions is not charged.
    old_events = ledger.read()
    guard.calls = sum(row.get("type") in {"llm_call", "llm_error"}
                      for row in old_events)
    guard.usd = sum(float(row.get("usd", 0.0) or 0.0) for row in old_events
                    if row.get("type") == "llm_call")
    designer = OperatorDesigner(LLM(ledger, guard), args.model,
                                args.critic_model, ledger)
    for row in state["rounds"]:
        op = row.get("operator") or {}
        if op.get("family") and op.get("parameters"):
            designer.seen.add(stable_hash({"family": op["family"],
                                           "parameters": op["parameters"]}))

    official_left = args.official_confirmations - sum(
        (r.get("official") or {}).get("score") is not None for r in state["rounds"])

    while len(state["rounds"]) < args.rounds:
        index = len(state["rounds"]) + 1
        family = choose_family(state["rounds"], families)
        designer.allowed_families = {family}
        evidence = feedback_evidence(state["rounds"])
        try:
            proposal, verdict = designer.propose(evidence)
        except BudgetExceeded as exc:
            state["halt"] = f"llm budget exhausted: {exc}"
            break
        except Exception as exc:
            entry = {"round": index, "status": "proposal_failed",
                     "operator": {"family": family, "version": "none",
                                  "parameters": {}, "expected_l_gain_fraction": 0.0},
                     "error": str(exc)[:500]}
            state["rounds"].append(entry)
            atomic_json(state_path, state)
            continue
        if proposal is None:
            entry = {"round": index, "status": "rejected",
                     "operator": {"family": family, "version": "none",
                                  "parameters": {}, "expected_l_gain_fraction": 0.0},
                     "error": "; ".join(verdict["reasons"])[:500]}
            state["rounds"].append(entry)
            atomic_json(state_path, state)
            print(json.dumps({"round": index, "family": family,
                              "status": "rejected"}), flush=True)
            continue

        spec = proposal.operator
        acquire, acquisition = construction_acquisition(state["rounds"], spec)
        if not acquire:
            entry = {"round": index, "status": "surrogate_rejected",
                     "operator": asdict(spec), "operator_id": spec.id,
                     "acquisition": acquisition,
                     "error": acquisition["reason"],
                     "failure_class": "surrogate-kill"}
            state["rounds"].append(entry)
            atomic_json(state_path, state)
            print(json.dumps({"round": index, "family": family,
                              "status": "surrogate_rejected"}), flush=True)
            continue
        # Serialize physics only while the solver runs. LLM thinking and review
        # must not hold the scarce host-wide evaluator lock.
        with CampaignLock(RUNS_DIR / ".stellar-physics.lock"):
            result, errors, escalated = evaluate_spec(
                TASK, spec, None, args.retries, args.screen_niter, base_niter)
        entry = {"round": index, "operator": asdict(spec),
                 "operator_id": spec.id, "status": "failed", "errors": errors,
                 "error": errors[-1] if errors else None, "escalated": escalated,
                 "acquisition": acquisition}
        if result is not None and not result.error:
            code = candidate_code(spec)
            key = hashlib.sha1(code.encode()).hexdigest()[:12]
            entry.update({"status": "evaluated", "boundary": TASK._bcache[key],
                          "metrics": enrich_metrics(result.metrics), "solves": 2})
            record = _entry_record(entry, fingerprint, args.seed)
            entry["descriptor"] = asdict(record.descriptor)
            entry["provenance"] = asdict(record.provenance)
            entry["promotable"] = census_promotable(record)
            if entry["promotable"]:
                with CampaignLock(RUNS_DIR / ".stellar-physics.lock"):
                    lf = verify_boundary(entry["boundary"], fidelity="low_fidelity")
                entry["lf"] = {"score": lf.score, "error": lf.error,
                               "metrics": enrich_metrics(lf.metrics)
                               if not lf.error else lf.metrics,
                               "seconds": lf.seconds}
                lf_metrics = entry["lf"].get("metrics") or {}
                in_corridor = (not lf.error and
                    lf_metrics.get("feasibility", float("inf")) <= .1 and
                    (record.feasibility <= .01 or
                     (resolution_signature(record) == (0, 0) and
                      (lf_metrics.get("selection_score", float("-inf")) >= .66 or
                       lf_metrics.get("objective_L", float("-inf")) >= 13.2))))
                if in_corridor and official_left > 0:
                    official_left -= 1
                    with CampaignLock(RUNS_DIR / ".stellar-physics.lock"):
                        official = verify_boundary(entry["boundary"], official=True)
                    entry["official"] = {"score": official.score,
                                         "error": official.error,
                                         "metrics": official.metrics,
                                         "seconds": official.seconds}
        entry["failure_class"] = failure_class(entry)
        state["rounds"].append(entry)
        atomic_json(state_path, state)
        print(json.dumps({"round": index, "family": family,
                          "status": entry["status"],
                          "failure_class": entry["failure_class"],
                          "promotable": entry.get("promotable", False)}), flush=True)

    records = [_entry_record(r, fingerprint, args.seed) for r in state["rounds"]
               if r.get("status") == "evaluated"]
    leaders = successive_halving(filter(census_promotable, records), args.promote)
    pareto = ParetoArchive()
    for record in records:
        pareto.add(record)
    leader_ids = {r.job_id for r in leaders}
    stepping_stones = sorted((r for r in pareto.records if r.job_id not in leader_ids),
                             key=lambda r: (r.feasibility, -r.objective_l))[:args.repair_promote]
    track = {r.job_id: "promotion" for r in leaders} | {
             r.job_id: "repair-probe" for r in stepping_stones}
    report_leaders = []
    for row in state["rounds"]:
        if row.get("operator_id") not in track:
            continue
        report_leaders.append({k: row[k] for k in (
            "operator_id", "boundary", "metrics", "descriptor", "provenance",
            "lf", "official") if k in row} | {"job_id": row["operator_id"],
                                                "vlf_metrics": row["metrics"],
                                                "track": track[row["operator_id"]]})
    report = {"schema_version": 1, "report_kind": "structural_independent_loop",
              "arm": "independent", "bank_disabled": True,
              "evaluator_fingerprint": fingerprint,
              "rounds": len(state["rounds"]), "evaluated": len(records),
              "failed": sum(r.get("status") == "failed" for r in state["rounds"]),
              "rejected": sum(r.get("status") == "rejected" for r in state["rounds"]),
              "distinct_cells": len({r.descriptor.cell for r in records}),
              "promotable": sum(census_promotable(r) for r in records),
              "pareto_stepping_stones": len(stepping_stones),
              "family_statistics": family_statistics(state["rounds"]),
              "leaders": report_leaders, "llm_calls": guard.calls,
              "usd": guard.usd, "halt": state.get("halt")}
    atomic_json(run_dir / "report.json", report)
    state["complete"] = not state.get("halt")
    atomic_json(state_path, state)
    print("INDEPENDENT_REPORT " + json.dumps(
        {k: v for k, v in report.items() if k != "leaders"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
