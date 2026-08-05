"""Margin-gradient walk demo (Plan 2, Phase 4). Host-side polish utility —
proposal machinery only; every step is verified at the pinned evaluator and
the pinned evaluator remains the only accept/rank authority.

Runs INSIDE the eval image:
  docker run --rm --cpus=1.5 --memory=3g -v <repo>:/work -w /work \
      evoharness-stellar-eval python -m experiments.diffscore.margin_walk \
      --source both --target-margin 0.002

Per step: exact boundary-space gradients of the two boundary-controlled
margins (aspect exactly; elongation via the input-parameterization port, axis
frozen) form a min-norm projected step targeting the requested normalized
margin for every controlled constraint; equilibrium-carried margins (iota,
mirror, qi) are measured, not modeled. VMEC hard failures shrink the step.
"""

import argparse
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT = _ROOT / "runs" / "diffscore"
_BOUNDS = np.array([10.0, 0.25, -4.0, 0.2, 5.0])
_STEP_CAP = 3e-3   # max-coeff trust region per verified step


def _best_archive_boundary():
    best = None
    for line in (_ROOT / "memory" / "stellar_p2" /
                 "archive.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["p2"] > 0 and (best is None or r["p2"] > best["p2"]):
            best = r
    return best


def _oracle(boundary_dict):
    from experiments.diffscore.difftest import run_oracle
    return run_oracle(boundary_dict)


def _masks(rc, zs):
    ntor = (rc.shape[1] - 1) // 2
    mask_rc = np.ones_like(rc)
    mask_zs = np.ones_like(zs)
    mask_rc[0, :ntor] = 0.0       # m=0, n<0 fixed by stellarator symmetry
    mask_zs[0, :ntor + 1] = 0.0   # m=0, n<=0
    return mask_rc, mask_zs


def _gradients(rc, zs, nfp, arrays):
    import jax
    from experiments.diffscore import margins_jax
    mask_rc, mask_zs = _masks(rc, zs)
    ga_rc, ga_zs = margins_jax.aspect_grad(rc, zs, nfp)
    ga = np.concatenate([(np.asarray(ga_rc) * mask_rc).ravel(),
                         (np.asarray(ga_zs) * mask_zs).ravel()])
    mpol_w, ntor_w = int(arrays["mpol"]), int(arrays["ntor"])

    def elong(rc_, zs_):
        v, _ = margins_jax.max_elongation_boundary(
            rc_, zs_, nfp, arrays["raxis_cc"], arrays["zaxis_cs"],
            mpol_w, ntor_w)
        return v
    ge_rc, ge_zs = jax.grad(elong, argnums=(0, 1))(rc, zs)
    ge = np.concatenate([(np.asarray(ge_rc) * mask_rc).ravel(),
                         (np.asarray(ge_zs) * mask_zs).ravel()])
    return ga / _BOUNDS[0], ge / _BOUNDS[4]   # violation units


def walk(boundary_dict, target_margin, max_steps=6, label="", spend=False):
    rc = np.array(boundary_dict["r_cos"], dtype=float)
    zs = np.array(boundary_dict["z_sin"], dtype=float)
    nfp = boundary_dict["n_field_periods"]

    m0, arrays = _oracle(boundary_dict)
    trail = [{"step": 0, "step_norm": 0.0, **m0}]
    print(f"[{label}] start feas={m0['feasibility']:.5f} v={np.round(m0['violations'], 5).tolist()} "
          f"LgradB={m0['minimum_normalized_magnetic_gradient_scale_length']:.4f}",
          flush=True)

    for step in range(1, max_steps + 1):
        v = np.array(trail[-1]["violations"])
        # controlled set: aspect (0) and elongation (4). Default: drive each
        # that is above the target down to it. Spend mode: additionally push
        # the aspect margin UP toward the target (buying score with unused
        # tolerance); elongation is never pushed up.
        idx, want = [], []
        for i in (0, 4):
            if v[i] > target_margin + 1e-4 or (
                    spend and i == 0 and v[i] < target_margin - 1e-4):
                idx.append(i)
                want.append(target_margin - v[i])
        if not idx:
            print(f"[{label}] controlled margins at target; stopping")
            break
        gv, ge = _gradients(rc, zs, nfp, arrays)
        G = np.stack([{0: gv, 4: ge}[i] for i in idx])
        want = np.array(want)
        x = G.T @ np.linalg.solve(G @ G.T, want)      # min-norm step
        cap = np.max(np.abs(x))
        if cap > _STEP_CAP:
            x *= _STEP_CAP / cap
        n = rc.size
        for attempt in range(5):
            rc_t = rc + x[:n].reshape(rc.shape)
            zs_t = zs + x[n:].reshape(zs.shape)
            bd_t = dict(boundary_dict)
            bd_t["r_cos"], bd_t["z_sin"] = rc_t.tolist(), zs_t.tolist()
            try:
                m_t, arrays_t = _oracle(bd_t)
                break
            except Exception as e:
                print(f"[{label}] step {step} eval failed ({type(e).__name__});"
                      f" halving", flush=True)
                x *= 0.5
        else:
            print(f"[{label}] step {step}: VMEC kept failing; stopping")
            break
        # crossing refinement: if an uncontrolled margin (e.g. qi) rose enough
        # to make the step a net feasibility loss, evaluate the balanced point
        # on the segment predicted by the two endpoint margin vectors
        if m_t["feasibility"] > trail[-1]["feasibility"] and not spend:
            v0, v1 = np.array(trail[-1]["violations"]), np.array(
                m_t["violations"])
            best_s, best_f = 0.0, np.max(v0)
            for s in np.linspace(0.05, 0.95, 19):
                f = np.max(v0 + s * (v1 - v0))
                if f < best_f:
                    best_s, best_f = s, f
            if best_s > 0.0:
                rc_s = rc + best_s * x[:n].reshape(rc.shape)
                zs_s = zs + best_s * x[n:].reshape(zs.shape)
                bd_s = dict(boundary_dict)
                bd_s["r_cos"], bd_s["z_sin"] = rc_s.tolist(), zs_s.tolist()
                try:
                    m_s, arrays_s = _oracle(bd_s)
                    print(f"[{label}] crossing refine s={best_s:.2f}: feas "
                          f"{m_s['feasibility']:.5f} (predicted {best_f:.5f})",
                          flush=True)
                    if m_s["feasibility"] < m_t["feasibility"]:
                        rc_t, zs_t, bd_t = rc_s, zs_s, bd_s
                        m_t, arrays_t = m_s, arrays_s
                        x = best_s * x
                except Exception:
                    pass
        rc, zs, boundary_dict, arrays = rc_t, zs_t, bd_t, arrays_t
        trail.append({"step": step, "step_norm": float(np.max(np.abs(x))),
                      "targets": {int(i): float(w)
                                  for i, w in zip(idx, want)}, **m_t})
        print(f"[{label}] step {step}: feas {trail[-2]['feasibility']:.5f} -> "
              f"{m_t['feasibility']:.5f} v={np.round(m_t['violations'], 5).tolist()} "
              f"LgradB {m_t['minimum_normalized_magnetic_gradient_scale_length']:.4f}",
              flush=True)
    return {"label": label, "target_margin": target_margin,
            "boundary_final": boundary_dict, "trail": trail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="archive-best",
                    choices=["archive-best", "bank-0", "both"])
    ap.add_argument("--target-margin", type=float, default=0.002)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--spend", action="store_true",
                    help="push the aspect margin UP toward the target")
    args = ap.parse_args()

    jobs = []
    if args.source in ("archive-best", "both"):
        best = _best_archive_boundary()
        print(f"archive best: p2={best['p2']:.5f} feas={best['feasibility']:.5f}")
        jobs.append((best["boundary"], f"archive-best[{best['key']}]"))
    if args.source in ("bank-0", "both"):
        bank = json.loads((_ROOT / "tasks" / "stellar_p2" /
                           "seed_bank.json").read_text())["seeds"]
        jobs.append((bank[0]["boundary"], "bank-0[davidkh]"))

    results = [walk(bd, args.target_margin, args.max_steps, label,
                    spend=args.spend)
               for bd, label in jobs]
    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / ("margin_walk_spend.json" if args.spend
                  else "margin_walk.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
