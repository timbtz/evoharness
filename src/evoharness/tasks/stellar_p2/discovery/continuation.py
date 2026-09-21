"""Resumable one-band-at-a-time continuation for promoted independent basins."""
from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path

from evoharness.paths import RUNS_DIR

from evoharness.engine.campaign_lock import CampaignLock
from evoharness.tasks.stellar_p2.discovery.operators import ModeBand, activate_band, geometry_screen
from evoharness.tasks.stellar_p2.discovery.boundaries import enrich_metrics, stable_hash
from evoharness.tasks.stellar_p2.discovery.census import atomic_json


def accept_band(base: dict, child: dict, min_l_gain: float = .2,
                repair_fraction: float = .2) -> tuple[bool, str]:
    """Keep material L movement or substantial feasibility repair, never noise."""
    if child.get("objective_L", float("-inf")) - base.get("objective_L", float("-inf")) >= min_l_gain:
        return True, "material L gain"
    bf, cf = base.get("feasibility", float("inf")), child.get("feasibility", float("inf"))
    if bf > 0 and cf <= bf * (1.0 - repair_fraction):
        return True, "material feasibility repair"
    return False, "no material L gain or repair"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_report", type=Path)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--amplitudes", default="0.002,0.005,0.01")
    ap.add_argument("--max-bands", type=int, default=3)
    ap.add_argument("--seed", type=int, default=8101)
    args = ap.parse_args()
    os.environ["STELLAR_NO_BANK"] = "1"
    from evoharness.tasks.stellar_p2.task import verify_boundary
    lock = CampaignLock(RUNS_DIR / ".stellar-physics.lock"); lock.__enter__()
    atexit.register(lock.__exit__, None, None, None)
    report = json.loads(args.parent_report.read_text())
    leaders = report.get("leaders", [])
    state_path = args.run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "parent_report_sha": stable_hash(report), "jobs": {}}
    if state["parent_report_sha"] != stable_hash(report):
        raise RuntimeError("parent report changed; refusing stale resume")
    amplitudes = [float(x) for x in args.amplitudes.split(",")]
    for leader_i, leader in enumerate(leaders):
        parent, base = leader["boundary"], enrich_metrics(leader["vlf_metrics"])
        provenance = leader.get("provenance") or {}
        if not provenance.get("independent", report.get("arm") == "independent"):
            raise RuntimeError("independent continuation received an inherited leader")
        for band_i, amp in enumerate(amplitudes[:args.max_bands]):
            # Accepted bands become the parent of the next probe. This is actual
            # one-band-at-a-time continuation, rather than independent siblings
            # that all restart from the original low-order surface.
            rc = parent["r_cos"]
            old_m, old_n = len(rc) - 1, (len(rc[0]) - 1) // 2
            band = ModeBand(old_m + 1, old_n + 1, amp, args.seed + leader_i * 100 + band_i)
            parent_hash = stable_hash(parent)
            job_id = f"{leader['job_id']}:{parent_hash[:8]}:{stable_hash(band.__dict__)[:10]}"
            if job_id in state["jobs"]:
                previous = state["jobs"][job_id]
                if previous.get("status") == "accepted":
                    parent = previous["boundary"]
                    base = enrich_metrics(previous["metrics"])
                continue
            child = activate_band(parent, band)
            ok, reason = geometry_screen(child)
            entry = {"parent": leader["job_id"], "parent_boundary_hash": parent_hash,
                     "band": band.__dict__, "status": "failed",
                     "provenance": {"independent": True,
                                    "public_bank_enabled": False,
                                    "source_boundary_hashes": [parent_hash]}}
            if not ok:
                entry["error"] = reason
            else:
                result = verify_boundary(child, fidelity="very_low_fidelity")
                if result.error:
                    entry["error"] = result.error
                else:
                    metrics = enrich_metrics(result.metrics)
                    accepted, why = accept_band(base, metrics)
                    entry.update({"status": "accepted" if accepted else "rejected",
                                  "decision": why, "metrics": metrics, "boundary": child,
                                  "seconds": result.seconds})
                    if accepted:
                        parent, base = child, metrics
            state["jobs"][job_id] = entry
            atomic_json(state_path, state)
            print(json.dumps({"job": job_id, "status": entry["status"],
                              "decision": entry.get("decision"),
                              "error": entry.get("error")}), flush=True)
    accepted = [v for v in state["jobs"].values() if v["status"] == "accepted"]
    out = {"schema_version": 1, "report_kind": "structural_continuation",
           "arm": "independent", "bank_disabled": True,
           "parents": len(leaders), "probes": len(state["jobs"]),
           "accepted": len(accepted), "leaders": accepted}
    atomic_json(args.run_dir / "report.json", out)
    print("CONTINUATION_REPORT " + json.dumps({k: v for k, v in out.items() if k != "leaders"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
