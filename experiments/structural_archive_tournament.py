"""Fresh VLF/LF tournament for distinct inherited archive controls."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.campaign_lock import CampaignLock
from core.structural_discovery import enrich_metrics, stable_hash
from experiments.structural_census import atomic_json


def distance(a: dict, b: dict) -> float:
    import numpy as np
    if a.get("n_field_periods") != b.get("n_field_periods"):
        return float("inf")
    def norm(x):
        r, z = np.asarray(x["r_cos"], float), np.asarray(x["z_sin"], float)
        s = r[0, r.shape[1] // 2]
        return r / s, z / s
    ar, az = norm(a); br, bz = norm(b)
    shape = (max(ar.shape[0], br.shape[0]), max(ar.shape[1], br.shape[1]))
    def pad(x):
        out = np.zeros(shape); off = (shape[1] - x.shape[1]) // 2
        out[:x.shape[0], off:off + x.shape[1]] = x; return out
    return max(float(abs(pad(ar) - pad(br)).max()), float(abs(pad(az) - pad(bz)).max()))


def select(rows: list[dict], count: int, minimum_distance: float) -> list[dict]:
    eligible = sorted((r for r in rows if r.get("boundary") and r.get("p2")),
                      key=lambda r: r.get("p2", 0), reverse=True)
    chosen = []
    for row in eligible:
        if all(distance(row["boundary"], old["boundary"]) >= minimum_distance
               for old in chosen):
            chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--min-distance", type=float, default=.003)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    from tasks.stellar_p2.task import verify_boundary
    lock = CampaignLock(ROOT / "runs/.stellar-physics.lock"); lock.__enter__()
    atexit.register(lock.__exit__, None, None, None)
    archive = ROOT / "memory/stellar_p2/archive.jsonl"
    rows = [json.loads(x) for x in archive.read_text().splitlines() if x.strip()]
    starts = select(rows, args.count, args.min_distance)
    state_path = args.run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "archive_sha": stable_hash(rows), "jobs": {}}
    if state["archive_sha"] != stable_hash(rows):
        raise RuntimeError("archive changed; refusing stale tournament resume")
    for row in starts:
        job = row["key"]
        entry = state["jobs"].get(job, {"provenance": "inherited", "boundary": row["boundary"]})
        for fidelity in ("very_low_fidelity", "low_fidelity"):
            if fidelity in entry:
                continue
            result = verify_boundary(row["boundary"], fidelity=fidelity)
            entry[fidelity] = {"score": result.score,
                "metrics": enrich_metrics(result.metrics) if not result.error else result.metrics,
                "error": result.error, "seconds": result.seconds}
            state["jobs"][job] = entry; atomic_json(state_path, state)
        print(json.dumps({"job": job, "vlf": entry["very_low_fidelity"]["score"],
                          "lf": entry["low_fidelity"]["score"]}), flush=True)
    valid = [(k, v) for k, v in state["jobs"].items()
             if not v["very_low_fidelity"]["error"] and not v["low_fidelity"]["error"]]
    def order(fid):
        return [k for k, _ in sorted(valid, key=lambda kv: kv[1][fid]["metrics"].get(
            "selection_score", float("-inf")), reverse=True)]
    vo, lo = order("very_low_fidelity"), order("low_fidelity")
    pos = {x: i for i, x in enumerate(vo)}
    inversions = sum(pos[a] > pos[b] for i, a in enumerate(lo) for b in lo[i + 1:])
    report = {"schema_version": 1, "report_kind": "archive_tournament",
              "provenance": "inherited-control", "starts": len(starts),
              "vlf_order": vo, "lf_order": lo, "rank_inversions": inversions,
              "official_bypass_allowed": False, "jobs": state["jobs"]}
    atomic_json(args.run_dir / "report.json", report)
    print("TOURNAMENT_REPORT " + json.dumps({k: v for k, v in report.items() if k != "jobs"}))
    return 0 if len(valid) >= 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
