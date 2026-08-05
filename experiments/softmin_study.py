"""How much smoothing does the P2 objective need before its gradient is usable?

Measured 2026-08-05: min L-gradB's argmin SWITCHES under 1e-4 max-coefficient
boundary steps, so at the step sizes where VMEC re-converges the objective is
only piecewise smooth. Consequence measured on the same solver: a finite
difference of the hard min and its own a.e. gradient disagree IN SIGN
(+773 vs -2069 on a feasible boundary at h=3e-4). A gradient method cannot work
on a function whose gradient and secant point opposite ways.

`lgradb_jax.min_normalized_l_grad_b(..., smoothing=tau)` softens the min to
-tau*logsumexp(-x/tau). This sweep finds the tau window where BOTH hold:

  usable gradient : the analytic d(softmin)/dp agrees with the secant of
                    softmin over the same step (they cannot agree while a
                    branch switch sits inside the step, so agreement IS the
                    signal that tau spans the switch)
  honest value    : softmin is below min by less than the score resolution we
                    care about. P2 = L/20, so a bias of 0.02 in L is 0.001 in
                    score — already a quarter of the honest gap we are chasing.

Costs NO new solves: it reuses the stencil equilibria already on disk under
runs/polish/dirs* (base/plus/minus x pinned/sidecar), so it can run while the
LLM campaign has the CPUs.

  docker run ... evoharness-stellar-eval python experiments/softmin_study.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

ARR = ["rmnc", "zmns", "gmnc", "bmnc", "bsupumnc", "bsupvmnc"]


def study(dirs: Path, side: str, taus: list[float]) -> list[dict]:
    import jax
    import jax.numpy as jnp
    from experiments.diffscore import lgradb_jax as lg

    index = json.loads((dirs / "index.json").read_text())
    rows = []
    for row in index:
        key, h = row["key"], row["h"]
        try:
            z = {t: dict(np.load(dirs / f"{key}.{t}.{side}.npz", allow_pickle=True))
                 for t in ("base", "plus", "minus")}
        except Exception as e:
            print(f"skip {key}/{side}: {type(e).__name__}", flush=True)
            continue
        b = z["base"]
        fixed = dict(xm=b["xm"], xn=b["xn"], xm_nyq=b["xm_nyq"],
                     xn_nyq=b["xn_nyq"], ns=int(b["ns"]),
                     aminor_p=float(b["Aminor_p"]), mpol=int(b["mpol"]),
                     ntor=int(b["ntor"]), nfp=int(b["nfp"]),
                     lasym=bool(b["lasym"]))

        def f(arrs, tau):
            d = dict(zip(ARR, arrs))
            return lg.min_normalized_l_grad_b(
                d["rmnc"], d["zmns"], d["gmnc"], d["bmnc"], d["bsupumnc"],
                d["bsupvmnc"], fixed["xm"], fixed["xn"], fixed["xm_nyq"],
                fixed["xn_nyq"], fixed["ns"], fixed["aminor_p"], fixed["mpol"],
                fixed["ntor"], fixed["nfp"], fixed["lasym"], smoothing=tau)

        hard = float(f([jnp.asarray(b[k]) for k in ARR], 0.0))
        for tau in taus:
            base_args = [jnp.asarray(b[k]) for k in ARR]
            val, grads = jax.value_and_grad(
                lambda *a, t=tau: f(list(a), t),
                argnums=tuple(range(len(ARR))))(*base_args)
            g_an = 0.0
            for k, g in zip(ARR, grads):
                g_an += float(np.sum(np.asarray(g)
                                     * (z["plus"][k] - z["minus"][k]) / (2 * h)))
            g_fd = (float(f([jnp.asarray(z["plus"][k]) for k in ARR], tau))
                    - float(f([jnp.asarray(z["minus"][k]) for k in ARR], tau))) / (2 * h)
            rel = abs(g_an - g_fd) / max(abs(g_an), abs(g_fd), 1e-30)
            rows.append({"key": key, "side": side, "h": h, "tau": tau,
                         "hard_min": hard, "soft_value": float(val),
                         "value_bias": hard - float(val),
                         "score_bias": (hard - float(val)) / 20.0,
                         "grad_analytic": g_an, "grad_secant": g_fd,
                         "rel_disagreement": rel,
                         "sign_agrees": bool(np.sign(g_an) == np.sign(g_fd))})
            print(f"{key[:18]:18s} {side:7s} tau={tau:<8g} "
                  f"value_bias={rows[-1]['value_bias']:.4g} "
                  f"(score {rows[-1]['score_bias']:.2g})  "
                  f"grad an={g_an:+.5g} secant={g_fd:+.5g} "
                  f"rel={rel:.3g} sign={rows[-1]['sign_agrees']}", flush=True)
            jax.clear_caches()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default=os.environ.get(
        "VD_DIRS", str(_ROOT / "runs" / "polish" / "dirs")))
    ap.add_argument("--sides", default="pinned")
    ap.add_argument("--taus", default="0,1e-3,1e-2,3e-2,1e-1,3e-1,1")
    a = ap.parse_args()

    dirs = Path(a.dirs)
    taus = [float(t) for t in a.taus.split(",")]
    rows = []
    for side in a.sides.split(","):
        rows += study(dirs, side, taus)
    out = dirs / "softmin_study.json"
    out.write_text(json.dumps(rows, indent=2))

    # the window: gradient trustworthy AND value bias small
    print("\n=== tau window (sign agreement + rel<0.2 + score bias<1e-3) ===",
          flush=True)
    for tau in taus:
        sub = [r for r in rows if r["tau"] == tau]
        if not sub:
            continue
        signs = sum(r["sign_agrees"] for r in sub)
        rel = float(np.median([r["rel_disagreement"] for r in sub]))
        bias = float(np.max([r["score_bias"] for r in sub]))
        ok = signs == len(sub) and rel < 0.2 and bias < 1e-3
        print(f"tau={tau:<8g} sign {signs}/{len(sub)}  rel_median={rel:.3g}  "
              f"max_score_bias={bias:.3g}   {'USABLE' if ok else ''}", flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
