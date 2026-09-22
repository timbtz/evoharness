"""Fit a conservative construction surrogate from fresh census evidence.

The surrogate only schedules future screens; it never promotes a boundary or
substitutes for VLF/LF/official evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from evoharness.tasks.stellar_p2.discovery.boundaries import ConstructionSurrogate

def fit(report: dict, state: dict | None = None) -> dict:
    rows = list(report.get("leaders", []))
    if state:
        rows.extend(e for e in state.get("jobs", {}).values() if e.get("status") == "evaluated")
    observations = []
    for row in rows:
        params = row.get("operator", {}).get("parameters", {})
        family = row.get("operator", {}).get("family")
        metrics = row.get("vlf_metrics", row.get("metrics", {}))
        if family and params and metrics.get("objective_L") is not None and \
                metrics.get("feasibility") is not None:
            observations.append({"family": family, "parameters": params,
                                 "objective_L": metrics["objective_L"],
                                 "feasibility": metrics["feasibility"]})
    predictions = []
    for index, target in enumerate(observations):
        peers = [x for i, x in enumerate(observations) if i != index and
                 x["family"] == target["family"] and
                 set(x["parameters"]) == set(target["parameters"])]
        if not peers:
            continue
        keys = tuple(sorted(target["parameters"]))
        categorical = tuple(k for k in keys if k in {
            "n_field_periods", "max_poloidal_mode", "max_toroidal_mode",
            "repair_mode", "phase_seed"})
        surrogate = ConstructionSurrogate(keys, categorical)
        for peer in peers:
            surrogate.add(peer["parameters"], peer["objective_L"], peer["feasibility"])
        pred = surrogate.predict(target["parameters"])
        predictions.append({"family": target["family"],
            "predicted_L": pred.objective_l, "actual_L": target["objective_L"],
            "predicted_feasibility": pred.feasibility,
            "actual_feasibility": target["feasibility"],
            "uncertainty": pred.uncertainty, "neighbors": pred.neighbors})
    return {"report_kind":"structural_surrogate", "schema_version":1,
            "observations":len(observations), "predictions":predictions}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("report",type=Path); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); raw=json.loads(a.report.read_text()); state_path=a.report.with_name("state.json"); state=json.loads(state_path.read_text()) if state_path.exists() else None; out=fit(raw,state); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print("STRUCTURAL_SURROGATE "+json.dumps(out,sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
