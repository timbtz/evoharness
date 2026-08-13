"""Does the frontier's very-low-fidelity gain survive official scoring?

The band-continuation points gain up to +0.0136 honest at vlf with feasibility
unchanged -- nine times what 17 gradient steps and 715 solves earned. The vlf
level does not transfer (the measured vlf->official gap on this task is
0.01-0.02), so only the DELTA matters, and only official evaluation settles it.

Re-scores the source basin and the top feasible frontier points at
default_high_fidelity, ~128 s each, and reports the official deltas.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", type=Path,
                    default=ROOT / "runs/transform-frontier/state.json")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/transform-frontier/official.json")
    args = ap.parse_args()

    from tasks.stellar_p2.task import verify_boundary

    state = json.loads(args.frontier.read_text())
    points = state["points"]
    baseline = points["baseline"]
    feasible = []
    for key, entry in points.items():
        if key == "baseline" or entry.get("status") != "evaluated":
            continue
        m = entry["metrics"]
        if m.get("honest_score") is None:
            continue                      # infeasible: nothing to transfer
        feasible.append((m["honest_score"], key, entry))
    feasible.sort(reverse=True, key=lambda row: row[0])

    cases = [("source", baseline["metrics"]["honest_score"], baseline["boundary"])]
    cases += [(key, honest, entry["boundary"])
              for honest, key, entry in feasible[:args.top]]

    out = {"schema_version": 1, "report_kind": "frontier_official", "cases": []}
    for name, vlf_honest, boundary in cases:
        started = time.time()
        result = verify_boundary(boundary, official=True)
        metrics = result.metrics or {}
        row = {"case": name, "vlf_honest": vlf_honest,
               "official_score": result.score,
               "official_feasibility": metrics.get("feasibility"),
               "official_L": metrics.get("objective_L"),
               "error": result.error, "wall_s": round(time.time() - started, 1)}
        out["cases"].append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(json.dumps(row), flush=True)

    source = out["cases"][0]
    for row in out["cases"][1:]:
        if isinstance(row["official_score"], float) and \
                isinstance(source["official_score"], float):
            row["delta_official"] = row["official_score"] - source["official_score"]
            row["delta_vlf"] = row["vlf_honest"] - source["vlf_honest"]
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("OFFICIAL " + json.dumps(
        [{k: r.get(k) for k in ("case", "official_score", "delta_official", "delta_vlf")}
         for r in out["cases"]], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
