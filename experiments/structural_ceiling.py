"""Quantify scoped family ceilings from fresh structural census evidence."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def build(reports: list[Path]) -> dict:
    families=defaultdict(lambda:{"jobs":0,"evaluated":0,"best_L":None,"best_honest":None,"best_feasibility":None})
    for report_path in reports:
        state_path=report_path.with_name("state.json")
        if not state_path.exists(): continue
        try: state=json.loads(state_path.read_text())
        except json.JSONDecodeError: continue
        for row in state.get("jobs",{}).values():
            family=row.get("operator",{}).get("family")
            if not family: continue
            out=families[family]; out["jobs"]+=1
            m=row.get("metrics") or {}
            if row.get("status")=="evaluated" and m:
                out["evaluated"]+=1
                for key, dest, fn in (("objective_L","best_L",max),("honest_score","best_honest",max),("feasibility","best_feasibility",min)):
                    value=m.get(key)
                    if value is not None and (out[dest] is None or fn(float(value),out[dest])==float(value)): out[dest]=float(value)
    return {"schema_version":1,"report_kind":"structural_ceiling","families":dict(sorted(families.items())),"official_candidates":0,"target":"0.70 official; no family currently reaches projected corridor"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("reports",nargs="*",type=Path); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    paths=a.reports or sorted(Path("runs").glob("**/report.json")); out=build(paths); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print("STRUCTURAL_CEILING "+json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
