"""How much qi damage does a structural move of a given size actually cost?

The inherited campaign keeps producing the same trade: dilation raises L and
drives feasibility from 0.0078 to 0.12-0.30 with qi becoming the active
violation. Whether that is fixable decides the whole architecture -- the
gradient repair loop moves feasibility by ~4e-4 over 17 steps and 715 solves,
so a 0.12 violation is roughly 300 steps away, i.e. out of reach.

This sweeps one transform parameter at a time and reports (L, feasibility, qi)
per magnitude, so the operator bounds and the writer's contract can be set from
the measured frontier instead of from the plan's assumed >=5% L gain.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SWEEPS = {
    "dilation": [{"family": "structural_dilation",
                  "parameters": {"radial_scale": s, "vertical_scale": s, "mode_decay": 0.}}
                 for s in (1.0003, 1.001, 1.003, 1.01, 1.03)],
    "dilation_anisotropic": [{"family": "structural_dilation",
                              "parameters": {"radial_scale": s, "vertical_scale": 1.,
                                             "mode_decay": 0.}}
                             for s in (1.001, 1.003, 1.01)],
    "band_fine": [{"family": "mode_continuation",
                   "parameters": {"max_poloidal_mode": 8, "max_toroidal_mode": 8,
                                  "band_amplitude": a, "phase_seed": 5}}
                  for a in (1e-5, 3e-5, 5e-5, 1.5e-4, 2e-4)],
    "band_seed": [{"family": "mode_continuation",
                   "parameters": {"max_poloidal_mode": 8, "max_toroidal_mode": 8,
                                  "band_amplitude": 1e-4, "phase_seed": s}}
                  for s in (1, 2, 3)],
    "band_shape": [{"family": "mode_continuation",
                    "parameters": {"max_poloidal_mode": m, "max_toroidal_mode": n,
                                   "band_amplitude": 1e-4, "phase_seed": 5}}
                   for m, n in ((8, 7), (7, 8), (8, 8))],
    "band": [{"family": "mode_continuation",
              "parameters": {"max_poloidal_mode": 8, "max_toroidal_mode": 8,
                             "band_amplitude": a, "phase_seed": 5}}
             for a in (1e-4, 3e-4, 1e-3, 3e-3)],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=ROOT / "runs/composite_polish_campaign/champion/best_boundary.json")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs/transform-frontier")
    ap.add_argument("--sweeps", default="dilation,band")
    args = ap.parse_args()

    payload = json.loads(args.source.read_text())
    source = payload.get("boundary", payload)

    os.environ["STELLAR_NO_BANK"] = "1"
    os.environ.setdefault("STELLAR_TRAIN_OVERRIDES", json.dumps({
        "max_evals": 2, "cpu_budget": 400.0, "collect_top": 1, "workers": 1,
        "eval_timeout": 150.0}))
    os.environ.setdefault("STELLAR_SOFT_FAIL_NITER", "20000")
    from core.campaign_lock import CampaignLock
    from core.structural_discovery import OperatorSpec
    from experiments.structural_census import atomic_json
    from experiments.structural_inherited_loop import outcome_text, screen
    from tasks.stellar_p2.task import TASK

    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"points": {}}

    lock = CampaignLock(ROOT / "runs/.stellar-physics.lock")
    lock.__enter__()
    atexit.register(lock.__exit__, None, None, None)

    baseline_key = "baseline"
    if baseline_key not in state["points"]:
        identity = OperatorSpec("structural_dilation", "frontier-baseline",
                                {"radial_scale": 1., "vertical_scale": 1., "mode_decay": 0.},
                                "unmodified source", .05, "n/a")
        state["points"][baseline_key] = screen(TASK, identity, source)
        atomic_json(state_path, state)
    print("BASELINE " + outcome_text(state["points"][baseline_key]), flush=True)

    for name in (s.strip() for s in args.sweeps.split(",") if s.strip()):
        for point in SWEEPS[name]:
            key = f"{name}:{json.dumps(point['parameters'], sort_keys=True)}"
            if key in state["points"]:
                continue
            spec = OperatorSpec(point["family"], f"frontier-{name}", point["parameters"],
                                "measured frontier sweep", .05,
                                "n/a: this is a calibration point")
            started = time.time()
            entry = screen(TASK, spec, source)
            entry["seconds"] = round(time.time() - started, 1)
            state["points"][key] = entry
            atomic_json(state_path, state)
            print(json.dumps({"sweep": name, "parameters": point["parameters"],
                              "outcome": outcome_text(entry)}), flush=True)

    base = state["points"][baseline_key]["metrics"]
    rows = []
    for key, entry in state["points"].items():
        if key == baseline_key or entry.get("status") != "evaluated":
            continue
        m = entry["metrics"]
        rows.append({"point": key, "objective_L": m["objective_L"],
                     "feasibility": m["feasibility"], "log10_qi": m.get("log10_qi"),
                     "active_violation": m.get("active_violation"),
                     "d_L": m["objective_L"] - base["objective_L"],
                     "d_feasibility": m["feasibility"] - base["feasibility"]})
    rows.sort(key=lambda r: r["d_feasibility"])
    report = {"schema_version": 1, "report_kind": "transform_frontier",
              "baseline": {k: base.get(k) for k in
                           ("objective_L", "feasibility", "log10_qi", "active_violation")},
              "points": rows}
    atomic_json(args.run_dir / "report.json", report)
    print("FRONTIER_REPORT " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
