"""Does the polish loop's gain survive the fidelity it is not optimizing?

The FD polish loop descends `very_low_fidelity`; the leaderboard scores
`default_high_fidelity`, and the measured vlf->official gap on this task is
0.01-0.02 — roughly five times the +0.00256 the loop has earned so far. So the
loop's whole premise rests on the DELTA transferring even though the level does
not. Two official evals answer it: re-score the boundary the loop started from
and the best boundary it has accepted, and compare the differences.

Cheap on purpose (~2 x 128 s). It is the decision the loop's remaining hours hang
on: a delta that transfers funds more gradient work, one that vanishes stops it.

  STELLAR_MEM_MB=3072 .venv/bin/python experiments/transfer_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "runs" / "qi_polish" / "transfer.json"


def main() -> int:
    from experiments.fd_qi_probe import load_cases
    from tasks.stellar_p2.task import verify_boundary

    st = json.loads((_ROOT / "runs" / "qi_polish" / "state.json").read_text())
    traj = st["trajectory"]
    acc = [r for r in traj if r.get("accepted")]
    cases = [("step0_champion", load_cases(["champion"])[0]["boundary"],
              traj[0]["honest"]),
             ("polished", st["boundary"], acc[-1]["honest"] if acc else None)]

    out = {"vlf_steps_accepted": len(acc), "solves": st.get("solves"),
           "trust": st.get("trust"), "cases": []}
    for name, b, vlf_honest in cases:
        t0 = time.time()
        r = verify_boundary(b, official=True)
        m = r.metrics or {}
        rec = {"case": name, "vlf_honest": vlf_honest, "official_score": r.score,
               "official_feasibility": m.get("feasibility"),
               "error": r.error, "wall_s": round(time.time() - t0, 1)}
        out["cases"].append(rec)
        print(json.dumps(rec), flush=True)

    a, b_ = out["cases"][0], out["cases"][1]
    if a["vlf_honest"] is not None and b_["vlf_honest"] is not None:
        out["delta_vlf"] = b_["vlf_honest"] - a["vlf_honest"]
    if isinstance(a["official_score"], float) and isinstance(b_["official_score"], float):
        out["delta_official"] = b_["official_score"] - a["official_score"]
        out["transferred"] = (out.get("delta_official", 0.0) > 0.0)
    OUT.write_text(json.dumps(out, indent=2))
    print("TRANSFER " + json.dumps({k: v for k, v in out.items() if k != "cases"}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
