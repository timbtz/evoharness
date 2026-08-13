"""End-to-end smoke for the solver-gradient tools inside the eval sandbox.

Exercises the parts golden tests cannot: that the diffscore package really lands in
the read-only mount, that the child process finds it on PYTHONPATH, that jax runs
there without disturbing the fork pool, and that a projected step actually moves the
boundary. k=2 keeps it to ~6 solves (a real gradient is k=20).

Diagnostics come back through a deliberate RuntimeError — the template surfaces a
candidate's exception text as EvalResult.error, which is the only channel out of the
sandbox that survives the JSON contract (the probe_bank.py trick).

  .venv/bin/python experiments/_gradsmoke.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# must precede the task import: task.py reads these at import time
os.environ["STELLAR_FULL_GRAD"] = "1"
os.environ["STELLAR_MARGIN_GRAD"] = "1"
os.environ["STELLAR_TRAIN_OVERRIDES"] = '{"max_evals":20,"cpu_budget":900.0,"eval_timeout":180.0}'
os.environ["STELLAR_MEM_MB"] = "6144"

CODE = '''
def solve(fm, rng):
    import numpy as np
    b = fm.seed_bank(0)
    g = fm.metric_grad(b, k=2)
    if g is None:
        raise RuntimeError("metric_grad returned None: %s" % fm.last_error)
    b2 = fm.grad_step(b, g, cap=3e-5)
    if b2 is None:
        raise RuntimeError("grad_step returned None: %s" % fm.last_error)
    moved = float(np.abs(np.asarray(b2["r_cos"], float)
                         - np.asarray(b["r_cos"], float)).max())
    def nz(name):
        v = g.get(name)
        return None if v is None else float(np.abs(np.asarray(v["r_cos"], float)).max())
    raise RuntimeError("SMOKE_OK " + json.dumps({
        "solves": g["solves"], "used": fm.used, "trust": g["trust"],
        "base_honest": g["base"]["honest_score"],
        "base_feas": g["base"]["feasibility"],
        "coeffs": g.get("coeffs"),
        "max_grad_L": nz("grad_L"), "max_grad_qi": nz("grad_qi"),
        "max_grad_aspect": nz("grad_aspect"),
        "max_grad_elong": nz("grad_elongation"),
        "elong_error": g.get("elongation_error"),
        "step_maxcoeff": moved}))
'''


def main() -> int:
    from tasks.stellar_p2 import task as st
    print("running smoke in the eval image (~6 solves, a few minutes)...", flush=True)
    res = st.TASK._train(CODE)
    err = res.error or ""
    ok = "SMOKE_OK" in err
    print(("PASS " if ok else "FAIL ") + err[:1200])
    if ok:
        payload = json.loads(err.split("SMOKE_OK ", 1)[1].split("\n")[0].split(" [line")[0])
        assert payload["solves"] == 5, payload   # 1 base + 2*k solves actually run
        assert payload["used"] == 5, payload     # and exactly that many charged
        assert payload["step_maxcoeff"] > 0, payload
        print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
