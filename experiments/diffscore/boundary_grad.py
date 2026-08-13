"""d(P2 metrics)/d(boundary) as a one-shot service — the gradient candidate code gets.

Everything downstream of the solver was already differentiable (Plan 2): qi w.r.t.
the Boozer arrays, L-gradB w.r.t. the wout arrays, aspect and elongation w.r.t. the
boundary itself. The missing link is d(solver output)/d(boundary), and until an
adjoint exists it is finite differences: ~2 solves per coefficient. That is what
`qi_polish.gradients` does inside the standalone polish loop, and this module is the
same computation packaged so the optimizer's evolved code can call it.

Why a separate process rather than a function in the task template: the template's
parent process forks the eval pool, and an initialized XLA runtime in the parent
deadlocks the pool respawn after an eval timeout (documented in task.py's _GRAD_SRC).
jax therefore may not be imported there. This runs as a child with PYTHONPATH=/work,
so the parent stays jax-free; the only things crossing the boundary are one JSON
request in and one JSON line out.

What comes back, and what each part costs:

  grad_L        d(min normalized L-gradB)/d(coeff)   FD, 2 solves per coefficient
  grad_qi       d(log10 qi)/d(coeff)                 FD, same solves (shared)
  grad_aspect   d(normalized aspect violation)/d(coeff)   exact, analytic, free
  grad_elong    d(max elongation)/d(coeff)           analytic, free, AXIS FROZEN

grad_elong holds the magnetic axis fixed at the base equilibrium's, so it misses
d(axis)/d(boundary) — measured to move the metric by ~4e-4 between fidelities. It is
a direction, not a prediction. grad_L and grad_qi are honest to the trust region
below and nowhere else.

TRUST REGION: 3e-5 in max-coefficient distance. Measured, not guessed — the
predicted-vs-actual ratio is 0.99 at 3e-5 and -0.05 at 1e-4, i.e. one order up the
step direction inverts (runs/fd_probe/champion/small_caps.json). The objective's
argmin switches under 1e-4 steps, so this is a property of the landscape, not of the
estimator, and no amount of smoothing substitutes for a small step.

  echo '{"boundary": {...}, "k": 20}' | PYTHONPATH=/work python boundary_grad.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

TRUST = 3e-5
DEFAULT_K, DEFAULT_H = 20, 3e-5


def _split(vec: np.ndarray, rc: np.ndarray, zs: np.ndarray) -> dict:
    """Flat (r_cos|z_sin) gradient vector back into the boundary's two blocks."""
    return {"r_cos": vec[:rc.size].reshape(rc.shape).tolist(),
            "z_sin": vec[rc.size:].reshape(zs.shape).tolist()}


def compute(boundary: dict, k: int = DEFAULT_K, h: float = DEFAULT_H,
            fidelity: str = "very_low_fidelity",
            want: tuple = ("L", "qi", "aspect", "elongation")) -> dict:
    """One re-linearization at `boundary`. Returns a JSON-safe dict; the solve
    count is exact so the caller can charge it to an eval budget."""
    from experiments.diffscore.difftest import run_oracle
    from experiments import fd_qi_probe as fp
    from experiments.qi_polish import gradients, honest_of

    def log(msg):                      # stdout carries the JSON result only
        print(msg, file=sys.stderr, flush=True)

    rc = np.asarray(boundary["r_cos"], float)
    zs = np.asarray(boundary["z_sin"], float)

    m0, arrays = run_oracle(boundary, fidelity)
    out = {"ok": True, "solves": 1, "trust": TRUST, "h": h,
           "base": {"honest_score": honest_of(m0),
                    "p2_score": m0["minimum_normalized_magnetic_gradient_scale_length"] / 20.0,
                    "L": m0["minimum_normalized_magnetic_gradient_scale_length"],
                    "feasibility": m0["feasibility"],
                    "violations": list(m0["violations"]),
                    "qi": m0["qi"], "aspect": m0["aspect_ratio"]}}

    if "L" in want or "qi" in want:
        coeffs = fp.pick_coeffs(boundary, k)
        gL, gq, ns = gradients(boundary, arrays, coeffs, h, fidelity, log=log)
        out["solves"] += ns
        out["coeffs"] = [[t, i, j] for t, i, j, _v in coeffs]
        if "L" in want:
            out["grad_L"] = _split(gL, rc, zs)
        if "qi" in want:
            out["grad_qi"] = _split(gq, rc, zs)

    if "aspect" in want:
        a, viol, g_rc, g_zs = fp.aspect_grad(boundary)
        out["aspect"] = a
        out["aspect_violation"] = viol
        out["grad_aspect"] = {"r_cos": np.asarray(g_rc).tolist(),
                              "z_sin": np.asarray(g_zs).tolist()}

    if "elongation" in want:
        try:
            import jax
            import jax.numpy as jnp
            from experiments.diffscore import margins_jax as mj
            nfp = int(boundary["n_field_periods"])
            ax_c = jnp.asarray(arrays["raxis_cc"])
            ax_s = jnp.asarray(arrays["zaxis_cs"])
            mp, nt = int(arrays["mpol"]), int(arrays["ntor"])

            def f_e(r, z):
                # max_elongation returns (max, per_phi_elongations) — grad needs
                # the scalar, and value_and_grad rejects the tuple outright
                return mj.max_elongation_boundary(r, z, nfp, ax_c, ax_s, mp, nt)[0]

            e, (ge_rc, ge_zs) = jax.value_and_grad(f_e, argnums=(0, 1))(
                jnp.asarray(rc), jnp.asarray(zs))
            mr, mz = fp.free_mask(rc, zs)
            out["elongation_boundary"] = float(e)
            out["grad_elongation"] = {"r_cos": (np.asarray(ge_rc) * mr).tolist(),
                                      "z_sin": (np.asarray(ge_zs) * mz).tolist()}
        except Exception as e:                       # never lose the FD result
            out["elongation_error"] = f"{type(e).__name__}: {e}"[:200]
            log(f"elongation gradient unavailable: {out['elongation_error']}")

    return out


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        res = compute(req["boundary"], int(req.get("k", DEFAULT_K)),
                      float(req.get("h", DEFAULT_H)),
                      req.get("fidelity", "very_low_fidelity"),
                      tuple(req.get("want", ("L", "qi", "aspect", "elongation"))))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
