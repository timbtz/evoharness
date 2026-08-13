"""Fit a conservative construction surrogate from fresh census evidence.

The surrogate only schedules future screens; it never promotes a boundary or
substitutes for VLF/LF/official evaluation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.structural_discovery import ConstructionSurrogate

def fit(report: dict, state: dict | None = None) -> dict:
    rows = list(report.get("leaders", []))
    if state:
        rows.extend(e for e in state.get("jobs", {}).values() if e.get("status") == "evaluated")
    keys = ("aspect_ratio", "max_elongation", "rotational_transform",
            "mirror_ratio", "n_field_periods", "max_poloidal_mode", "max_toroidal_mode")
    surrogate = ConstructionSurrogate(keys)
    for row in rows:
        params = row.get("operator", {}).get("parameters", {})
        metrics = row.get("vlf_metrics", row.get("metrics", {}))
        if all(k in params for k in keys) and metrics.get("objective_L") is not None:
            surrogate.add(params, metrics["objective_L"], metrics.get("feasibility", float("inf")))
    return {"report_kind":"structural_surrogate", "schema_version":1,
            "observations":surrogate.observations, "predictions":[]}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("report",type=Path); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); raw=json.loads(a.report.read_text()); state_path=a.report.with_name("state.json"); state=json.loads(state_path.read_text()) if state_path.exists() else None; out=fit(raw,state); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print("STRUCTURAL_SURROGATE "+json.dumps(out,sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
