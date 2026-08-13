"""Is the census's dominant failure physics, or the soft-fail iteration cap?

The census screens each start with ONE very-low-fidelity solve and drops the
start whenever that solve comes back `soft_fail`.  41 of 49 recorded force-
residual failures missed ftol by less than 1e-4, and 13 by less than 1e-6 --
the signature of a solve that was cut off, not one that diverged.  The screen
runs under STELLAR_SOFT_FAIL_NITER=5000 while vlf's own niter is 20000.

This re-evaluates the recorded near-misses with the cap lifted and the eval
timeout raised, and reports convergence and metrics per start.  It changes no
physics: same boundary, same evaluator, same fidelity, more iterations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def collect(limit: int) -> list[dict]:
    """Recorded soft-fail starts, closest to ftol first."""
    import glob
    import re
    rows, seen = [], set()
    states = (glob.glob(str(ROOT / "runs/structural-*/**/state.json"), recursive=True)
              + glob.glob(str(ROOT / "runs/structural-*/state.json")))
    for path in states:
        try:
            state = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        jobs = state.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if job.get("status") != "failed" or job_id in seen:
                continue
            for err in job.get("errors") or []:
                m = re.search(r"fsqr=(\S+) fsqz=(\S+) fsql=(\S+)\)", str(err))
                if not m:
                    continue
                worst = max(float(g) for g in m.groups())
                seen.add(job_id)
                rows.append({"job_id": job_id, "worst_residual": worst,
                             "operator": job["operator"], "run": path})
                break
    rows.sort(key=lambda r: r["worst_residual"])
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--niter", type=int, default=20000)
    ap.add_argument("--eval-timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/screen-niter-probe/report.json")
    args = ap.parse_args()

    targets = collect(args.limit)
    if not targets:
        print("no recorded soft-fail starts found")
        return 1

    os.environ["STELLAR_NO_BANK"] = "1"
    os.environ["STELLAR_SOFT_FAIL_NITER"] = str(args.niter)
    os.environ["STELLAR_TRAIN_OVERRIDES"] = json.dumps({
        "max_evals": 2, "cpu_budget": 4.0 * args.eval_timeout, "collect_top": 1,
        "workers": 1, "eval_timeout": args.eval_timeout})
    from core.campaign_lock import CampaignLock
    from core.structural_discovery import OperatorSpec
    from experiments.structural_census import candidate_code
    from tasks.stellar_p2.task import TASK

    results = []
    with CampaignLock(ROOT / "runs/.stellar-physics.lock"):
        for row in targets:
            spec = OperatorSpec(**row["operator"])
            started = time.time()
            result = TASK.evaluate(candidate_code(spec), "train")
            entry = {"job_id": row["job_id"], "family": spec.family,
                     "version": spec.version, "parameters": spec.parameters,
                     "previous_worst_residual": row["worst_residual"],
                     "seconds": round(time.time() - started, 1),
                     "error": result.error,
                     "converged_now": bool(result and not result.error)}
            if entry["converged_now"]:
                m = result.metrics or {}
                entry["metrics"] = {k: m.get(k) for k in (
                    "objective_L", "feasibility", "honest_score", "p2_score",
                    "active_violation", "log10_qi", "aspect_ratio")}
            results.append(entry)
            print(json.dumps(entry), flush=True)

    recovered = sum(r["converged_now"] for r in results)
    report = {"schema_version": 1, "report_kind": "screen_niter_probe",
              "niter": args.niter, "eval_timeout": args.eval_timeout,
              "tested": len(results), "recovered": recovered, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"PROBE recovered {recovered}/{len(results)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
