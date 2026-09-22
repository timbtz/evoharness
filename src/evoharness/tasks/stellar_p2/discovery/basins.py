"""Multi-report tournament with separate independent/inherited leaderboards."""
from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path

from evoharness.paths import RUNS_DIR

from evoharness.engine.campaign_lock import CampaignLock
from evoharness.tasks.stellar_p2.discovery.boundaries import enrich_metrics, stable_hash
from evoharness.tasks.stellar_p2.discovery.tournament import distance
from evoharness.tasks.stellar_p2.discovery.census import atomic_json


def collect_candidates(reports: list[dict], minimum_distance: float = .003) -> list[dict]:
    candidates = []
    for report in reports:
        report_arm = report.get("arm")
        leaders = list(report.get("leaders", []))
        if report.get("best"):
            leaders.append(report["best"])
        for leader in leaders:
            boundary = leader.get("boundary")
            if not boundary:
                continue
            provenance = leader.get("provenance") or {}
            independent = provenance.get("independent", report_arm == "independent")
            row = {"job_id": leader.get("job_id") or leader.get("operator_id") or
                             stable_hash(boundary)[:16],
                   "boundary": boundary,
                   "arm": "independent" if independent else "inherited",
                   "source_report_hash": stable_hash(report)}
            metrics = leader.get("vlf_metrics") or leader.get("metrics") or {}
            row["prior_metrics"] = metrics
            candidates.append(row)
    # Dedupe only within an arm: an inherited near-copy must remain visible as
    # an inherited control, never erase an independent result or vice versa.
    chosen = []
    for row in sorted(candidates, key=lambda r: (
            r["prior_metrics"].get("selection_score", float("-inf")),
            r["prior_metrics"].get("objective_L", float("-inf"))), reverse=True):
        if all(old["arm"] != row["arm"] or
               distance(old["boundary"], row["boundary"]) >= minimum_distance
               for old in chosen):
            chosen.append(row)
    return chosen


def _order(jobs: dict, arm: str, fidelity: str) -> list[str]:
    valid = [(key, row) for key, row in jobs.items() if row["arm"] == arm and
             not (row.get(fidelity) or {}).get("error")]
    return [key for key, _ in sorted(valid, key=lambda item:
        (item[1][fidelity]["metrics"].get("selection_score", float("-inf")),
         item[1][fidelity]["metrics"].get("objective_L", float("-inf"))), reverse=True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--min-distance", type=float, default=.003)
    ap.add_argument("--official", action="store_true",
                    help="officially verify only LF candidates in the 0.66/L13.2 corridor")
    args = ap.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    candidates = collect_candidates(reports, args.min_distance)
    from evoharness.tasks.stellar_p2.task import verify_boundary
    lock = CampaignLock(RUNS_DIR / ".stellar-physics.lock")
    lock.__enter__(); atexit.register(lock.__exit__, None, None, None)
    state_path = args.run_dir / "state.json"
    source_hash = stable_hash(reports)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "source_hash": source_hash, "jobs": {}}
    if state["source_hash"] != source_hash:
        raise RuntimeError("source reports changed; use a new tournament directory")
    for candidate in candidates:
        job_id = f"{candidate['arm']}:{candidate['job_id']}"
        entry = state["jobs"].get(job_id, candidate)
        for fidelity in ("very_low_fidelity", "low_fidelity"):
            if fidelity in entry:
                continue
            result = verify_boundary(candidate["boundary"], fidelity=fidelity)
            entry[fidelity] = {"score": result.score, "error": result.error,
                "metrics": enrich_metrics(result.metrics) if not result.error else result.metrics,
                "seconds": result.seconds}
            state["jobs"][job_id] = entry; atomic_json(state_path, state)
        lf = entry.get("low_fidelity") or {}
        lm = lf.get("metrics") or {}
        corridor = (not lf.get("error") and
                    (lm.get("selection_score", float("-inf")) >= .66 or
                     lm.get("objective_L", float("-inf")) >= 13.2))
        if args.official and corridor and "official" not in entry:
            result = verify_boundary(candidate["boundary"], official=True)
            entry["official"] = {"score": result.score, "error": result.error,
                                 "metrics": result.metrics, "seconds": result.seconds}
            state["jobs"][job_id] = entry; atomic_json(state_path, state)
    leaderboards = {arm: {fid: _order(state["jobs"], arm, fid) for fid in
                         ("very_low_fidelity", "low_fidelity")}
                    for arm in ("independent", "inherited")}
    out = {"schema_version": 1, "report_kind": "structural_basin_tournament",
           "starts": len(candidates), "minimum_distance": args.min_distance,
           "official_bypass_allowed": False, "leaderboards": leaderboards,
           "jobs": state["jobs"]}
    atomic_json(args.run_dir / "report.json", out)
    print("BASIN_TOURNAMENT " + json.dumps({k: v for k, v in out.items() if k != "jobs"}))
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
