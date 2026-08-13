"""Closed-loop inherited structural campaign: propose -> screen -> feed back.

The big campaign proposes every operator BEFORE any physics runs, so a writer
never learns what its own mechanism measured: 96 proposals across two live
campaigns produced 0 promotable designs and repeated the same aggressive
parameter regions. This loop screens each accepted operator immediately and
puts the measured outcome -- L, feasibility, the active violation, prediction
versus result -- into the evidence for the next round.

Inherited arm by construction: it transforms a supplied source basin, marks
provenance non-independent, and is reported separately from independent
discovery. A candidate that beats the source at very low fidelity is confirmed
at low AND official fidelity before it is called an improvement -- the band
continuation that gains +0.0136 vlf loses 0.0007-0.0017 official, so the vlf
screen alone would have recorded a resolution artifact as the campaign's win.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.campaign_lock import CampaignLock
from core.ledger import BudgetExceeded, BudgetGuard, Ledger
from core.llm import LLM
from core.structural_discovery import (BasinRecord, Provenance, describe,
                                       enrich_metrics, stable_hash)
from core.structural_llm import TRANSFORM_FAMILIES, OperatorDesigner
from experiments.structural_census import atomic_json, candidate_code

BASELINE_TEMPLATE = """## Measured source basin (the incumbent you must beat)
objective_L={L:.4f} feasibility={feasibility:.6f} honest_score={honest}
active_violation={active} log10_qi={qi} aspect_ratio={aspect:.4f}
official_score={official}
A candidate is an improvement only if honest_score rises AND feasibility does
not. feasibility > 0 means infeasible; the source is nearly at the wall. The
official score is the verdict; the vlf honest score is only the screen."""

ROUND_TEMPLATE = """- round {round}: {family}/{version} {parameters}
  predicted {predicted:+.0%} L; measured {outcome}"""


def outcome_text(entry: dict) -> str:
    if entry.get("status") != "evaluated":
        return f"FAILED ({str(entry.get('error'))[:160]})"
    m = entry["metrics"]
    text = ("L={L:.4f} feasibility={f:.6f} honest={h} active={a}".format(
        L=m.get("objective_L"), f=m.get("feasibility"),
        h=("%.6f" % m["honest_score"]) if m.get("honest_score") is not None else "none",
        a=m.get("active_violation")))
    # A vlf gain means nothing until the fidelity it is not optimizing agrees:
    # band continuation at amplitude 1e-4 gains +0.0136 vlf and LOSES 0.0007 to
    # 0.0017 official (runs/transform-frontier/official.json, 2026-08-13).
    official = entry.get("official") or {}
    if official.get("score") is not None:
        text += " | OFFICIAL {s:.6f} ({d:+.6f} vs source)".format(
            s=official["score"], d=official.get("delta", float("nan")))
    elif official.get("error"):
        text += f" | OFFICIAL failed ({str(official['error'])[:80]})"
    return text


def evidence_text(baseline: dict, rounds: list[dict], plan_chars: int = 6000) -> str:
    plan = (ROOT / "plans/stellar-070-structural-discovery-plan.md")
    blocks = [plan.read_text()[:plan_chars] if plan.exists() else ""]
    metrics = baseline["metrics"]
    blocks.append(BASELINE_TEMPLATE.format(
        L=metrics.get("objective_L", float("nan")),
        feasibility=metrics.get("feasibility", float("nan")),
        honest=("%.6f" % metrics["honest_score"]
                if metrics.get("honest_score") is not None else "none"),
        active=metrics.get("active_violation"), qi=metrics.get("log10_qi"),
        aspect=metrics.get("aspect_ratio", float("nan")),
        official=baseline.get("official_score", "not yet measured")))
    if rounds:
        lines = [ROUND_TEMPLATE.format(
            round=r["round"], family=r["operator"]["family"],
            version=r["operator"]["version"],
            parameters=json.dumps(r["operator"]["parameters"], sort_keys=True),
            predicted=r["operator"]["expected_l_gain_fraction"],
            outcome=outcome_text(r)) for r in rounds]
        blocks.append("## Measured outcomes of your previous operators\n"
                      "Do not repeat a parameter region that already failed.\n"
                      + "\n".join(lines))
    return "\n\n".join(b for b in blocks if b)


def screen(task, spec, source: dict) -> dict:
    """One fresh VLF screen of a host-transformed boundary."""
    import hashlib
    code = candidate_code(spec, source)
    result = task.evaluate(code, "train")
    entry = {"operator": asdict(spec), "operator_id": spec.id,
             "status": "failed", "error": result.error}
    if not result.error:
        key = hashlib.sha1(code.encode()).hexdigest()[:12]
        entry.update({"status": "evaluated", "boundary": task._bcache[key],
                      "metrics": enrich_metrics(result.metrics)})
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs/structural-inherited-1")
    ap.add_argument("--source", type=Path,
                    default=ROOT / "runs/composite_polish_campaign/champion/best_boundary.json")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--usd", type=float, default=15.0)
    ap.add_argument("--seconds", type=int, default=36000)
    ap.add_argument("--seed", type=int, default=7501)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--critic-model", default="glm-5.2")
    ap.add_argument("--screen-niter", type=int, default=20000)
    ap.add_argument("--official-confirmations", type=int, default=8,
                    help="how many vlf leaders may be re-scored at official "
                         "fidelity (~120 s each); a vlf gain is not a result "
                         "until one of these agrees")
    args = ap.parse_args()

    payload = json.loads(args.source.read_text())
    source = payload.get("boundary", payload)
    source_hash = stable_hash(source)

    os.environ["STELLAR_NO_BANK"] = "1"
    os.environ.setdefault("STELLAR_TRAIN_OVERRIDES", json.dumps({
        "max_evals": 2, "cpu_budget": 400.0, "collect_top": 1, "workers": 1,
        "eval_timeout": 150.0}))
    os.environ.setdefault("STELLAR_SOFT_FAIL_NITER", str(args.screen_niter))
    from tasks.stellar_p2.task import TASK, verify_boundary

    run_dir = args.run_dir.resolve()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "arm": "inherited", "seed": args.seed,
        "source_boundary_hash": source_hash, "started_at": time.time(), "rounds": []}
    if state.get("source_boundary_hash") != source_hash:
        raise RuntimeError("state was created against a different source basin")

    ledger = Ledger(run_dir)
    guard = BudgetGuard(args.usd, args.rounds * 3, args.seconds)
    designer = OperatorDesigner(LLM(ledger, guard), args.model, args.critic_model,
                                ledger, allowed_families=set(TRANSFORM_FAMILIES))
    for row in state["rounds"]:                      # resume without re-proposing
        designer.seen.add(stable_hash({"family": row["operator"]["family"],
                                       "parameters": row["operator"]["parameters"]}))

    lock = CampaignLock(ROOT / "runs/.stellar-physics.lock")
    lock.__enter__()
    atexit.register(lock.__exit__, None, None, None)

    if "baseline" not in state:
        from core.structural_discovery import OperatorSpec
        identity = OperatorSpec("structural_dilation", "baseline-identity",
                                {"radial_scale": 1., "vertical_scale": 1., "mode_decay": 0.},
                                "unmodified source basin, measured under the current evaluator",
                                .05, "n/a: this is the control")
        state["baseline"] = screen(TASK, identity, source)
        atomic_json(state_path, state)
        print("BASELINE " + outcome_text(state["baseline"]), flush=True)
    if state["baseline"]["status"] != "evaluated":
        raise RuntimeError(f"source basin does not screen: {state['baseline']['error']}")
    base_metrics = state["baseline"]["metrics"]
    base_honest = base_metrics.get("honest_score")

    # The source's OFFICIAL score is the number every candidate is measured
    # against; without it a vlf delta has nothing to transfer to.
    if "source_official" not in state:
        verdict = verify_boundary(source, official=True)
        state["source_official"] = {"score": verdict.score, "error": verdict.error}
        state["baseline"]["official_score"] = verdict.score
        atomic_json(state_path, state)
        print("SOURCE_OFFICIAL " + json.dumps(state["source_official"]), flush=True)
    source_official = state["source_official"].get("score")
    if not isinstance(source_official, float):
        raise RuntimeError(f"source basin does not score officially: "
                           f"{state['source_official'].get('error')}")
    official_budget = {"left": args.official_confirmations}

    while len(state["rounds"]) < args.rounds:
        index = len(state["rounds"]) + 1
        try:
            proposal, verdict = designer.propose(
                evidence_text(state["baseline"], state["rounds"]))
        except BudgetExceeded as exc:
            state["halt"] = f"llm budget exhausted: {exc}"
            break
        except Exception as exc:                     # a bad round must not end the run
            ledger.append({"type": "structural_round_error", "round": index,
                           "error": str(exc)[:500]})
            state["rounds"].append({"round": index, "status": "proposal_failed",
                                    "operator": {"family": "none", "version": "none",
                                                 "parameters": {},
                                                 "expected_l_gain_fraction": 0.},
                                    "error": str(exc)[:500]})
            atomic_json(state_path, state)
            continue
        if proposal is None:
            state["rounds"].append({"round": index, "status": "rejected",
                                    "operator": {"family": "none", "version": "none",
                                                 "parameters": {},
                                                 "expected_l_gain_fraction": 0.},
                                    "error": "; ".join(verdict["reasons"])[:400]})
            atomic_json(state_path, state)
            print(json.dumps({"round": index, "status": "rejected",
                              "reasons": verdict["reasons"]}), flush=True)
            continue
        entry = screen(TASK, proposal.operator, source)
        entry["round"] = index
        entry["predicted_l_gain_fraction"] = proposal.operator.expected_l_gain_fraction
        if entry["status"] == "evaluated":
            honest = entry["metrics"].get("honest_score")
            entry["beats_source"] = bool(
                honest is not None and base_honest is not None and honest > base_honest
                and entry["metrics"]["feasibility"] <= base_metrics["feasibility"])
            record = BasinRecord(
                proposal.operator.id, entry["boundary"], entry["metrics"],
                Provenance(proposal.operator.id, args.seed, (source_hash,),
                           independent=False, public_bank_enabled=False),
                stable_hash(TASK.experiment_spec()["evaluator"]), "very_low_fidelity",
                solves=2)
            entry["descriptor"] = asdict(describe(record))
            if entry["beats_source"]:
                lf = verify_boundary(entry["boundary"], fidelity="low_fidelity")
                entry["lf"] = {"score": lf.score, "error": lf.error,
                               "metrics": (enrich_metrics(lf.metrics) if not lf.error
                                           else lf.metrics)}
                # Official confirmation is the only verdict that counts. It is
                # ~120 s, so it is budgeted, but a vlf gain that is never
                # officially checked is exactly how the band artifact would have
                # been recorded as a win.
                if official_budget["left"] > 0:
                    official_budget["left"] -= 1
                    verdict = verify_boundary(entry["boundary"], official=True)
                    entry["official"] = {
                        "score": verdict.score, "error": verdict.error,
                        "feasibility": (verdict.metrics or {}).get("feasibility"),
                        "delta": ((verdict.score - source_official)
                                  if isinstance(verdict.score, float)
                                  and source_official is not None else None)}
                    entry["beats_source_official"] = bool(
                        entry["official"].get("delta") is not None
                        and entry["official"]["delta"] > 0)
        state["rounds"].append(entry)
        atomic_json(state_path, state)
        print(json.dumps({"round": index, "family": proposal.operator.family,
                          "status": entry["status"],
                          "beats_source": entry.get("beats_source", False),
                          "outcome": outcome_text(entry)}), flush=True)

    evaluated = [r for r in state["rounds"] if r.get("status") == "evaluated"]
    winners = sorted((r for r in evaluated if r.get("beats_source")),
                     key=lambda r: -r["metrics"]["honest_score"])
    official_winners = [r for r in evaluated if r.get("beats_source_official")]
    confirmed = [r for r in evaluated if (r.get("official") or {}).get("score") is not None]
    report = {"schema_version": 1, "report_kind": "structural_inherited_loop",
              "arm": "inherited", "source_boundary_hash": source_hash,
              "source_metrics": {k: base_metrics.get(k) for k in
                                 ("objective_L", "feasibility", "honest_score",
                                  "active_violation")},
              "rounds": len(state["rounds"]), "evaluated": len(evaluated),
              "rejected": sum(r.get("status") == "rejected" for r in state["rounds"]),
              "failed_screens": sum(r.get("status") == "failed" for r in state["rounds"]),
              "beat_source_vlf": len(winners),
              "officially_confirmed": len(confirmed),
              "beat_source_official": len(official_winners),
              "source_official_score": source_official,
              "best_official_delta": max(
                  ((r["official"]["delta"]) for r in confirmed
                   if r["official"].get("delta") is not None), default=None),
              "llm_calls": guard.calls, "usd": guard.usd,
              "best": (official_winners[0] if official_winners
                       else winners[0] if winners else None),
              "best_L": max((r["metrics"]["objective_L"] for r in evaluated), default=None),
              "families": {f: sum(r["operator"]["family"] == f for r in evaluated)
                           for f in sorted(TRANSFORM_FAMILIES)},
              "halt": state.get("halt")}
    atomic_json(run_dir / "report.json", report)
    state["complete"] = True
    atomic_json(state_path, state)
    print("INHERITED_REPORT " + json.dumps(
        {k: v for k, v in report.items() if k != "best"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
